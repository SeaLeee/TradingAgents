"""
TradingAgents Web Application
FastAPI server for running trading analysis via web interface
"""

import os
import asyncio
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Optional, List
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request, Response, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn

# Load environment variables
load_dotenv()

# Import TradingAgents
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

app = FastAPI(
    title="TradingAgents",
    description="Multi-Agents LLM Financial Trading Framework",
    version="1.0.0",
    docs_url=None,  # Disable docs for security
    redoc_url=None
)

# ============== Authentication ==============
# Get credentials from environment variables
AUTH_USERNAME = os.environ.get("AUTH_USERNAME", "admin")
AUTH_PASSWORD = os.environ.get("AUTH_PASSWORD", "trading123")  # Change this!
SESSION_SECRET = os.environ.get("SESSION_SECRET", secrets.token_hex(32))

# Simple session store (in production, use Redis or database)
sessions = {}

def hash_password(password: str) -> str:
    """Hash password with salt"""
    return hashlib.sha256(f"{password}{SESSION_SECRET}".encode()).hexdigest()

def verify_password(password: str) -> bool:
    """Verify password"""
    return password == AUTH_PASSWORD

def create_session() -> str:
    """Create a new session token"""
    token = secrets.token_urlsafe(32)
    sessions[token] = {
        "created": datetime.now(),
        "expires": datetime.now() + timedelta(hours=24)
    }
    return token

def verify_session(token: str) -> bool:
    """Verify session token"""
    if not token or token not in sessions:
        return False
    session = sessions[token]
    if datetime.now() > session["expires"]:
        del sessions[token]
        return False
    return True

def get_session_token(request: Request) -> Optional[str]:
    """Extract session token from cookie"""
    return request.cookies.get("session_token")

def require_auth(request: Request):
    """Dependency to require authentication"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(
            status_code=401,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"}
        )
    return True

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store analysis results and status
analysis_tasks = {}


class AnalysisRequest(BaseModel):
    ticker: str = "SPY"
    date: str = datetime.now().strftime("%Y-%m-%d")
    llm_provider: str = "google"
    quick_think_llm: str = "gemini-2.0-flash"
    deep_think_llm: str = "gemini-2.0-flash"
    analysts: List[str] = ["market", "news", "fundamentals", "social"]
    max_debate_rounds: int = 1


class AnalysisStatus(BaseModel):
    task_id: str
    status: str  # "pending", "running", "completed", "failed"
    progress: Optional[str] = None
    result: Optional[dict] = None
    error: Optional[str] = None


def run_analysis_sync(task_id: str, request: AnalysisRequest):
    """Run analysis synchronously (called in background)"""
    try:
        analysis_tasks[task_id]["status"] = "running"
        analysis_tasks[task_id]["progress"] = "Initializing agents..."
        
        # Create config
        config = DEFAULT_CONFIG.copy()
        config["llm_provider"] = request.llm_provider.lower()
        config["quick_think_llm"] = request.quick_think_llm
        config["deep_think_llm"] = request.deep_think_llm
        config["max_debate_rounds"] = request.max_debate_rounds
        config["max_risk_discuss_rounds"] = request.max_debate_rounds
        
        # Initialize graph
        analysis_tasks[task_id]["progress"] = "Setting up trading agents..."
        graph = TradingAgentsGraph(request.analysts, config=config, debug=False)
        
        # Run analysis
        analysis_tasks[task_id]["progress"] = f"Analyzing {request.ticker}..."
        state, decision = graph.propagate(request.ticker, request.date)
        
        # Extract results
        result = {
            "ticker": request.ticker,
            "date": request.date,
            "decision": decision,
            "reports": {
                "market": state.get("market_report", ""),
                "sentiment": state.get("sentiment_report", ""),
                "news": state.get("news_report", ""),
                "fundamentals": state.get("fundamentals_report", ""),
            },
            "investment_plan": state.get("investment_plan", ""),
            "final_decision": state.get("final_trade_decision", ""),
        }
        
        analysis_tasks[task_id]["status"] = "completed"
        analysis_tasks[task_id]["result"] = result
        analysis_tasks[task_id]["progress"] = "Analysis complete!"
        
    except Exception as e:
        analysis_tasks[task_id]["status"] = "failed"
        analysis_tasks[task_id]["error"] = str(e)
        analysis_tasks[task_id]["progress"] = f"Error: {str(e)}"


# ============== Login Page ==============
def get_login_html():
    """Return login page HTML"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login - TradingAgents</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .gradient-bg {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        }
    </style>
</head>
<body class="gradient-bg text-white min-h-screen flex items-center justify-center">
    <div class="w-full max-w-md p-8">
        <div class="bg-gray-800/50 backdrop-blur-lg rounded-2xl p-8 border border-gray-700">
            <div class="text-center mb-8">
                <h1 class="text-3xl font-bold mb-2">🤖 TradingAgents</h1>
                <p class="text-gray-400">Please login to continue</p>
            </div>
            
            <form action="/login" method="POST" class="space-y-6">
                <div>
                    <label class="block text-sm font-medium text-gray-300 mb-2">Username</label>
                    <input type="text" name="username" required
                        class="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-3 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        placeholder="Enter username">
                </div>
                
                <div>
                    <label class="block text-sm font-medium text-gray-300 mb-2">Password</label>
                    <input type="password" name="password" required
                        class="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-3 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        placeholder="Enter password">
                </div>
                
                <button type="submit"
                    class="w-full bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white font-semibold py-3 px-6 rounded-lg transition-all duration-200">
                    🔐 Login
                </button>
            </form>
            
            <p class="text-center text-gray-500 text-sm mt-6">
                Protected by session-based authentication
            </p>
        </div>
    </div>
</body>
</html>
"""


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Show login page"""
    # If already logged in, redirect to home
    token = get_session_token(request)
    if verify_session(token):
        return RedirectResponse(url="/", status_code=302)
    return HTMLResponse(content=get_login_html())


@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    """Process login form"""
    if username == AUTH_USERNAME and verify_password(password):
        token = create_session()
        response = RedirectResponse(url="/", status_code=302)
        response.set_cookie(
            key="session_token",
            value=token,
            httponly=True,
            secure=True,  # Use HTTPS in production
            samesite="lax",
            max_age=86400  # 24 hours
        )
        return response
    else:
        return HTMLResponse(
            content=get_login_html().replace(
                "Please login to continue",
                '<span class="text-red-400">Invalid username or password</span>'
            ),
            status_code=401
        )


@app.get("/logout")
async def logout(request: Request):
    """Logout and clear session"""
    token = get_session_token(request)
    if token and token in sessions:
        del sessions[token]
    response = RedirectResponse(url="/login", status_code=302)
    response.delete_cookie("session_token")
    return response


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main HTML page (protected)"""
    # Check authentication
    token = get_session_token(request)
    if not verify_session(token):
        return RedirectResponse(url="/login", status_code=302)
    
    html_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content=get_default_html())


@app.get("/api/health")
async def health_check():
    """Health check endpoint (public)"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/api/analyze")
async def start_analysis(request: Request, analysis_request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Start a new analysis task (protected)"""
    # Check authentication
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    import uuid
    task_id = str(uuid.uuid4())[:8]
    
    analysis_tasks[task_id] = {
        "status": "pending",
        "progress": "Queued for analysis...",
        "result": None,
        "error": None,
        "request": analysis_request.dict()
    }
    
    # Run analysis in background
    background_tasks.add_task(run_analysis_sync, task_id, analysis_request)
    
    return {"task_id": task_id, "message": "Analysis started"}


@app.get("/api/status/{task_id}")
async def get_status(request: Request, task_id: str):
    """Get the status of an analysis task (protected)"""
    # Check authentication
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    if task_id not in analysis_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return analysis_tasks[task_id]


@app.get("/api/config")
async def get_config(request: Request):
    """Get available configuration options (protected)"""
    # Check authentication
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")
    return {
        "llm_providers": ["google", "openai", "anthropic", "openrouter", "ollama"],
        "models": {
            "google": {
                "quick": ["gemini-2.0-flash-lite", "gemini-2.0-flash", "gemini-2.5-flash-preview-05-20"],
                "deep": ["gemini-2.0-flash-lite", "gemini-2.0-flash", "gemini-2.5-flash-preview-05-20", "gemini-2.5-pro-preview-06-05"]
            },
            "openai": {
                "quick": ["gpt-4o-mini", "gpt-4.1-nano", "gpt-4.1-mini", "gpt-4o"],
                "deep": ["gpt-4o", "o4-mini", "o3-mini", "o1"]
            },
            "anthropic": {
                "quick": ["claude-3-5-haiku-latest", "claude-3-5-sonnet-latest"],
                "deep": ["claude-3-5-sonnet-latest", "claude-sonnet-4-0", "claude-opus-4-0"]
            },
            "openrouter": {
                "quick": ["meta-llama/llama-4-scout:free", "google/gemini-2.0-flash-exp:free"],
                "deep": ["deepseek/deepseek-chat-v3-0324:free"]
            },
            "ollama": {
                "quick": ["llama3.1", "llama3.2"],
                "deep": ["llama3.1", "qwen3"]
            }
        },
        "analysts": ["market", "social", "news", "fundamentals"]
    }


def get_default_html():
    """Return default HTML if template not found"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TradingAgents</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-900 text-white min-h-screen">
    <div class="container mx-auto px-4 py-8">
        <h1 class="text-4xl font-bold text-center mb-8">🤖 TradingAgents</h1>
        <p class="text-center text-gray-400">Loading interface...</p>
        <p class="text-center mt-4"><a href="/docs" class="text-blue-400 hover:underline">API Documentation</a></p>
    </div>
</body>
</html>
"""


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
