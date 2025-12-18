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
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn
import io

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

# Store search history (in production, use database)
search_history = []
MAX_HISTORY = 50  # Maximum history records to keep


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


class HistoryRecord(BaseModel):
    task_id: str
    ticker: str
    date: str
    llm_provider: str
    status: str
    created_at: str
    decision_summary: Optional[str] = None


def save_to_history(task_id: str, request, status: str, decision_summary: Optional[str] = None):
    """Save analysis record to history
    
    Args:
        request: Can be AnalysisRequest object or dict
    """
    global search_history
    # Handle both AnalysisRequest and dict
    if isinstance(request, dict):
        ticker = request.get("ticker", "UNKNOWN")
        date = request.get("date", "")
        llm_provider = request.get("llm_provider", "")
    else:
        ticker = request.ticker
        date = request.date
        llm_provider = request.llm_provider
    
    record = {
        "task_id": task_id,
        "ticker": ticker,
        "date": date,
        "llm_provider": llm_provider,
        "status": status,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "decision_summary": decision_summary
    }
    search_history.insert(0, record)  # Add to beginning
    # Keep only MAX_HISTORY records
    if len(search_history) > MAX_HISTORY:
        search_history = search_history[:MAX_HISTORY]


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
        
        # Set backend_url based on LLM provider
        if config["llm_provider"] == "openrouter":
            config["backend_url"] = "https://openrouter.ai/api/v1"
            # OpenRouter uses OPENROUTER_API_KEY, need to set OPENAI_API_KEY for ChatOpenAI
            openrouter_key = os.getenv("OPENROUTER_API_KEY")
            if openrouter_key:
                os.environ["OPENAI_API_KEY"] = openrouter_key
        elif config["llm_provider"] == "deepseek":
            config["backend_url"] = "https://api.deepseek.com"
            # DeepSeek uses DEEPSEEK_API_KEY, need to set OPENAI_API_KEY for ChatOpenAI
            deepseek_key = os.getenv("DEEPSEEK_API_KEY")
            if deepseek_key:
                os.environ["OPENAI_API_KEY"] = deepseek_key
        elif config["llm_provider"] == "openai":
            config["backend_url"] = "https://api.openai.com/v1"
        elif config["llm_provider"] == "ollama":
            config["backend_url"] = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        
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
        
        # Save to history
        decision_summary = decision[:100] + "..." if len(decision) > 100 else decision
        save_to_history(task_id, analysis_tasks[task_id]["request"], "completed", decision_summary)
        
    except Exception as e:
        analysis_tasks[task_id]["status"] = "failed"
        analysis_tasks[task_id]["error"] = str(e)
        analysis_tasks[task_id]["progress"] = f"Error: {str(e)}"
        # Save failed record to history
        save_to_history(task_id, analysis_tasks[task_id]["request"], "failed", str(e)[:100])


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
        "llm_providers": ["google", "openai", "anthropic", "deepseek", "openrouter", "ollama"],
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
            "deepseek": {
                "quick": ["deepseek-chat"],
                "deep": ["deepseek-chat", "deepseek-reasoner"]
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


@app.get("/api/export/pdf/{task_id}")
async def export_pdf(request: Request, task_id: str):
    """Export analysis result as PDF (protected)"""
    # Check authentication
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    if task_id not in analysis_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = analysis_tasks[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Analysis not completed yet")
    
    result = task["result"]
    
    # Generate PDF using reportlab
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
        
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='Title2', parent=styles['Heading1'], fontSize=24, alignment=TA_CENTER, spaceAfter=30))
        styles.add(ParagraphStyle(name='Section', parent=styles['Heading2'], fontSize=14, spaceAfter=12, textColor=colors.darkblue))
        styles.add(ParagraphStyle(name='Body2', parent=styles['Normal'], fontSize=10, spaceAfter=8, leading=14))
        
        story = []
        
        # Title
        story.append(Paragraph(f"📊 Trading Analysis Report", styles['Title2']))
        story.append(Paragraph(f"<b>Ticker:</b> {result['ticker']} | <b>Date:</b> {result['date']}", styles['Body2']))
        story.append(Spacer(1, 20))
        
        # Decision Summary
        story.append(Paragraph("🎯 Final Decision", styles['Section']))
        decision_text = result.get('final_decision', result.get('decision', 'N/A'))
        if isinstance(decision_text, dict):
            decision_text = str(decision_text)
        story.append(Paragraph(decision_text[:2000] if decision_text else "N/A", styles['Body2']))
        story.append(Spacer(1, 15))
        
        # Reports
        reports = result.get('reports', {})
        
        if reports.get('market'):
            story.append(Paragraph("📈 Market Analysis", styles['Section']))
            story.append(Paragraph(str(reports['market'])[:3000], styles['Body2']))
            story.append(Spacer(1, 10))
        
        if reports.get('news'):
            story.append(Paragraph("📰 News Analysis", styles['Section']))
            story.append(Paragraph(str(reports['news'])[:3000], styles['Body2']))
            story.append(Spacer(1, 10))
        
        if reports.get('fundamentals'):
            story.append(Paragraph("📋 Fundamentals Analysis", styles['Section']))
            story.append(Paragraph(str(reports['fundamentals'])[:3000], styles['Body2']))
            story.append(Spacer(1, 10))
        
        if reports.get('sentiment'):
            story.append(Paragraph("💬 Sentiment Analysis", styles['Section']))
            story.append(Paragraph(str(reports['sentiment'])[:3000], styles['Body2']))
            story.append(Spacer(1, 10))
        
        # Investment Plan
        if result.get('investment_plan'):
            story.append(Paragraph("📝 Investment Plan", styles['Section']))
            story.append(Paragraph(str(result['investment_plan'])[:3000], styles['Body2']))
        
        # Footer
        story.append(Spacer(1, 30))
        story.append(Paragraph(f"<i>Generated by TradingAgents on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>", styles['Body2']))
        
        doc.build(story)
        buffer.seek(0)
        
        filename = f"TradingAgents_{result['ticker']}_{result['date']}.pdf"
        
        return StreamingResponse(
            buffer,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except ImportError:
        raise HTTPException(status_code=500, detail="PDF generation requires reportlab. Install with: pip install reportlab")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")


@app.get("/api/history")
async def get_history(request: Request, limit: int = 20):
    """Get analysis history (protected)"""
    # Check authentication
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    return {"history": search_history[:limit], "total": len(search_history)}


@app.get("/api/history/{task_id}")
async def get_history_detail(request: Request, task_id: str):
    """Get detail of a historical analysis (protected)"""
    # Check authentication
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    # First check if task is in current memory
    if task_id in analysis_tasks:
        return analysis_tasks[task_id]
    
    # Check in history
    for record in search_history:
        if record["task_id"] == task_id:
            return {"history_record": record, "message": "Full result not in memory"}
    
    raise HTTPException(status_code=404, detail="History record not found")


@app.delete("/api/history")
async def clear_history(request: Request):
    """Clear all history (protected)"""
    global search_history
    # Check authentication
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    search_history = []
    return {"message": "History cleared", "success": True}


class TranslateRequest(BaseModel):
    text: str
    target_lang: str = "zh"  # Default to Chinese


@app.post("/api/translate")
async def translate_text(request: Request, translate_request: TranslateRequest):
    """Translate text using LLM (protected)"""
    # Check authentication
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    text = translate_request.text
    target_lang = translate_request.target_lang
    
    if not text or len(text.strip()) == 0:
        raise HTTPException(status_code=400, detail="No text provided")
    
    lang_names = {
        "zh": "Chinese (Simplified)",
        "zh-TW": "Chinese (Traditional)",
        "ja": "Japanese",
        "ko": "Korean",
        "es": "Spanish",
        "fr": "French",
        "de": "German"
    }
    target_lang_name = lang_names.get(target_lang, "Chinese")
    
    try:
        # Try to use available LLM for translation
        # Priority: DeepSeek > OpenRouter > OpenAI > Google (since DeepSeek is most cost-effective)
        
        # Try DeepSeek first
        deepseek_key = os.getenv("DEEPSEEK_API_KEY")
        if deepseek_key:
            from openai import OpenAI
            client = OpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com")
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": f"You are a professional financial translator. Translate the following text to {target_lang_name}. Keep the structure, formatting and professional terminology. Only output the translation, no explanations."},
                    {"role": "user", "content": text}
                ]
            )
            return {"translated": response.choices[0].message.content, "source_lang": "en", "target_lang": target_lang}
        
        # Try OpenRouter
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if openrouter_key:
            from openai import OpenAI
            client = OpenAI(api_key=openrouter_key, base_url="https://openrouter.ai/api/v1")
            response = client.chat.completions.create(
                model="google/gemini-2.0-flash-exp:free",
                messages=[
                    {"role": "system", "content": f"You are a professional financial translator. Translate the following text to {target_lang_name}. Keep the structure, formatting and professional terminology. Only output the translation, no explanations."},
                    {"role": "user", "content": text}
                ]
            )
            return {"translated": response.choices[0].message.content, "source_lang": "en", "target_lang": target_lang}
        
        # Try OpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key and openai_key.startswith("sk-"):
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"You are a professional financial translator. Translate the following text to {target_lang_name}. Keep the structure, formatting and professional terminology. Only output the translation, no explanations."},
                    {"role": "user", "content": text}
                ]
            )
            return {"translated": response.choices[0].message.content, "source_lang": "en", "target_lang": target_lang}
        
        # Try Google last (often has quota issues)
        google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if google_key:
            try:
                from google import genai
                client = genai.Client(api_key=google_key)
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=f"Translate the following financial analysis text to {target_lang_name}. Keep the structure and formatting. Only output the translation, no explanations:\n\n{text}"
                )
                return {"translated": response.text, "source_lang": "en", "target_lang": target_lang}
            except Exception as google_error:
                # Google failed, continue to raise no API error
                pass
        
        raise HTTPException(status_code=500, detail="No valid LLM API key available for translation. Please set DEEPSEEK_API_KEY, OPENROUTER_API_KEY, or OPENAI_API_KEY.")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")


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
