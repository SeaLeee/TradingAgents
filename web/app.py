"""
TradingAgents Web Application
FastAPI server for running trading analysis via web interface
Supports GitHub OAuth authentication with database storage
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

# Import database and auth modules - handle both package and direct imports
try:
    from .database import (get_db, init_db, get_user_by_id, can_user_analyze, save_analysis,
                           create_journal, get_user_journals, get_user_journal_by_id, 
                           update_journal, delete_journal, get_user_journal_count,
                           get_or_create_password_user,
                           create_portfolio, get_user_portfolios, get_user_portfolio_by_id,
                           update_portfolio, delete_portfolio, recalculate_portfolio_balance,
                           create_paper_trade, get_user_trades, get_user_trade_by_id,
                           close_trade, update_trade, cancel_trade,
                           get_profitable_trades, get_trades_by_strategy,
                           get_trades_by_ticker, create_performance_snapshot,
                           get_performance_history, get_portfolio_stats,
                           update_strategy)
    from .report_generator import generate_backtest_report
    from .portfolio_analyzer import analyze_portfolio
    from .signal_generator import generate_signal
    from .strategy_validator import validate_strategy_payload
    from . import auth as github_auth
except ImportError:
    from database import (get_db, init_db, get_user_by_id, can_user_analyze, save_analysis,
                          create_journal, get_user_journals, get_user_journal_by_id,
                          update_journal, delete_journal, get_user_journal_count,
                          get_or_create_password_user,
                          create_portfolio, get_user_portfolios, get_user_portfolio_by_id,
                          update_portfolio, delete_portfolio, recalculate_portfolio_balance,
                          create_paper_trade, get_user_trades, get_user_trade_by_id,
                          close_trade, update_trade, cancel_trade,
                          get_profitable_trades, get_trades_by_strategy,
                          get_trades_by_ticker, create_performance_snapshot,
                          get_performance_history, get_portfolio_stats,
                          update_strategy)
    from report_generator import generate_backtest_report
    from portfolio_analyzer import analyze_portfolio
    from signal_generator import generate_signal
    from strategy_validator import validate_strategy_payload
    import auth as github_auth

# Daily analysis limit for GitHub OAuth users
DAILY_ANALYSIS_LIMIT = int(os.getenv("DAILY_ANALYSIS_LIMIT", "1"))

app = FastAPI(
    title="TradingAgents",
    description="Multi-Agents LLM Financial Trading Framework",
    version="1.0.0",
    docs_url=None,  # Disable docs for security
    redoc_url=None
)

# Store backtest tasks
app.backtest_tasks = {}

# ============== Authentication Configuration ==============
# Get credentials from environment variables (fallback for password login)
AUTH_USERNAME = os.environ.get("AUTH_USERNAME", "admin")
AUTH_PASSWORD = os.environ.get("AUTH_PASSWORD", "trading123")  # Change this!
SESSION_SECRET = os.environ.get("SESSION_SECRET", secrets.token_hex(32))

# GitHub OAuth configuration
GITHUB_CLIENT_ID = os.environ.get("GITHUB_CLIENT_ID", "")
GITHUB_CLIENT_SECRET = os.environ.get("GITHUB_CLIENT_SECRET", "")

# ============== TEST MODE ==============
TEST_MODE = os.getenv("TEST_MODE", "false").lower() == "true"

# In-memory session store for password login (kept for backward compatibility)
# For GitHub OAuth, sessions are stored in database
sessions = {}

# Store OAuth state tokens
oauth_states = {}


def hash_password(password: str) -> str:
    """Hash password with salt"""
    return hashlib.sha256(f"{password}{SESSION_SECRET}".encode()).hexdigest()


def verify_password(password: str) -> bool:
    """Verify password"""
    return password == AUTH_PASSWORD


def create_session(user_data: dict = None) -> str:
    """Create a new session token"""
    token = secrets.token_urlsafe(32)
    sessions[token] = {
        "created": datetime.now(),
        "expires": datetime.now() + timedelta(hours=24),
        "user": user_data  # Store user info for display
    }
    return token


def verify_session(token: str) -> bool:
    """Verify session token - always returns True in TEST_MODE"""
    # In TEST_MODE, skip authentication
    if TEST_MODE:
        return True
    
    # Check in-memory sessions (password login)
    if token and token in sessions:
        session = sessions[token]
        if datetime.now() > session["expires"]:
            del sessions[token]
            return False
        return True
    
    # Check database sessions (GitHub OAuth)
    if token:
        with get_db() as db:
            user = github_auth.validate_session(db, token)
            if user:
                return True
    
    return False


def get_current_user(request: Request) -> Optional[dict]:
    """Get current user info from session"""
    token = get_session_token(request)
    if not token:
        return None
    
    # TEST MODE: Return mock user
    if TEST_MODE:
        return {
            "id": 0,
            "username": "test_user",
            "name": "Test User",
            "avatar_url": None,
            "is_admin": True
        }
    
    # Check in-memory sessions
    if token in sessions:
        return sessions[token].get("user")
    
    # Check database sessions (GitHub OAuth)
    with get_db() as db:
        user = github_auth.validate_session(db, token)
        if user:
            return user.to_dict()
    
    return None


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

# Store search history with full results (in production, use database)
search_history = []
MAX_HISTORY = 50  # Maximum history records to keep

# Create data directory for persistent storage
HISTORY_DIR = os.path.join(os.path.dirname(__file__), "data", "history")
os.makedirs(HISTORY_DIR, exist_ok=True)


def load_history_from_disk():
    """Load history from disk on startup"""
    global search_history
    history_file = os.path.join(HISTORY_DIR, "history.json")
    if os.path.exists(history_file):
        try:
            import json
            with open(history_file, "r", encoding="utf-8") as f:
                search_history = json.load(f)
        except Exception as e:
            print(f"Failed to load history: {e}")
            search_history = []


def save_history_to_disk():
    """Save history to disk"""
    history_file = os.path.join(HISTORY_DIR, "history.json")
    try:
        import json
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(search_history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Failed to save history: {e}")


# Load history on startup
load_history_from_disk()


class AnalysisRequest(BaseModel):
    ticker: str = "SPY"
    date: str = datetime.now().strftime("%Y-%m-%d")
    llm_provider: str = "deepseek"
    quick_think_llm: str = "deepseek-chat"
    deep_think_llm: str = "deepseek-chat"
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


def save_to_history(task_id: str, request, status: str, decision_summary: Optional[str] = None, full_result: Optional[dict] = None):
    """Save analysis record to history with full result
    
    Args:
        request: Can be AnalysisRequest object or dict
        full_result: The complete analysis result to store
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
        "decision_summary": decision_summary,
        "full_result": full_result  # Store complete result
    }
    search_history.insert(0, record)  # Add to beginning
    # Keep only MAX_HISTORY records
    if len(search_history) > MAX_HISTORY:
        search_history = search_history[:MAX_HISTORY]
    
    # Persist to disk
    save_history_to_disk()


def translate_text_sync(text: str, target_lang: str = "zh") -> str:
    """Translate text using available LLM - synchronous version"""
    if not text or len(text.strip()) == 0:
        return ""
    
    # TEST MODE: Return mock translation
    if TEST_MODE:
        return f"[中文翻译] {text[:200]}..." if len(text) > 200 else f"[中文翻译] {text}"
    
    lang_names = {
        "zh": "Chinese (Simplified)",
        "zh-TW": "Chinese (Traditional)", 
        "ja": "Japanese",
        "ko": "Korean"
    }
    target_lang_name = lang_names.get(target_lang, "Chinese (Simplified)")
    
    try:
        # Try DeepSeek first (most reliable)
        deepseek_key = os.getenv("DEEPSEEK_API_KEY")
        if deepseek_key:
            from openai import OpenAI
            client = OpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com")
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": f"You are a professional financial translator. Translate the following text to {target_lang_name}. Keep the structure, formatting and professional terminology. Only output the translation, no explanations or additional text."},
                    {"role": "user", "content": text}
                ],
                temperature=0.3
            )
            return response.choices[0].message.content
        
        # Try OpenRouter
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if openrouter_key:
            from openai import OpenAI
            client = OpenAI(api_key=openrouter_key, base_url="https://openrouter.ai/api/v1")
            response = client.chat.completions.create(
                model="google/gemini-2.0-flash-exp:free",
                messages=[
                    {"role": "system", "content": f"You are a professional financial translator. Translate the following text to {target_lang_name}. Keep the structure, formatting and professional terminology. Only output the translation."},
                    {"role": "user", "content": text}
                ]
            )
            return response.choices[0].message.content
            
        return ""  # No translation available
    except Exception as e:
        print(f"Translation error: {e}")
        return ""  # Return empty on error


def translate_result(result: dict) -> dict:
    """Translate all text fields in a result dict to Chinese"""
    translated = {}
    
    # Translate decision
    if result.get("decision"):
        translated["decision"] = translate_text_sync(result["decision"])
    
    # Translate reports
    if result.get("reports"):
        translated["reports"] = {}
        for key, value in result["reports"].items():
            if value:
                translated["reports"][key] = translate_text_sync(value)
    
    # Translate investment plan
    if result.get("investment_plan"):
        translated["investment_plan"] = translate_text_sync(result["investment_plan"])
    
    # Translate final decision
    if result.get("final_decision"):
        translated["final_decision"] = translate_text_sync(result["final_decision"])
    
    return translated


def get_mock_analysis_result(ticker: str, date: str) -> dict:
    """Generate mock analysis result for testing"""
    import time
    time.sleep(2)  # Simulate processing time
    
    return {
        "ticker": ticker,
        "date": date,
        "decision": f"""## Trading Recommendation for {ticker}

Based on our comprehensive multi-agent analysis, we recommend a **MODERATE BUY** position for {ticker}.

### Key Factors:
1. **Technical Analysis**: The stock shows bullish momentum with RSI at 58
2. **Fundamental Strength**: P/E ratio is reasonable at 22.5x
3. **Market Sentiment**: Positive social media sentiment (65% bullish)
4. **News Impact**: Recent product announcements are favorable

### Risk Assessment:
- Volatility: Medium
- Market Risk: Low to Medium
- Sector Risk: Low

### Suggested Action:
Consider accumulating shares at current levels with a 6-month horizon.""",
        "reports": {
            "market": f"""## Market Analysis Report for {ticker}

### Price Action
- Current Price: $185.50
- 52-Week High: $199.62
- 52-Week Low: $124.17
- Average Volume: 52.3M shares

### Technical Indicators
- RSI (14): 58.2 (Neutral-Bullish)
- MACD: Bullish crossover detected
- Moving Averages: Price above 50-day and 200-day MA
- Bollinger Bands: Price in upper half

### Support & Resistance
- Key Support: $175.00
- Key Resistance: $195.00
- Trend: Upward channel formation

### Volume Analysis
Recent volume surge indicates institutional accumulation.""",
            
            "news": f"""## News Analysis Report for {ticker}

### Recent Headlines
1. **{ticker} Announces New AI Product Line** (2 days ago)
   - Sentiment: Very Positive
   - Market Impact: High
   
2. **Quarterly Earnings Beat Expectations** (1 week ago)
   - EPS: $2.15 vs $1.98 expected
   - Revenue: $35.2B vs $33.8B expected

3. **Strategic Partnership Announced** (2 weeks ago)
   - Partnership with major cloud provider
   - Expected to boost enterprise sales

### News Sentiment Score: 78/100 (Bullish)

### Key Takeaways
- Strong product pipeline
- Positive earnings momentum
- Expanding market presence""",
            
            "fundamentals": f"""## Fundamentals Analysis for {ticker}

### Valuation Metrics
- P/E Ratio: 22.5x (Industry avg: 25.3x)
- P/S Ratio: 8.2x
- P/B Ratio: 12.1x
- EV/EBITDA: 18.3x

### Financial Health
- Revenue Growth (YoY): 25.3%
- Net Profit Margin: 28.5%
- ROE: 45.2%
- Debt/Equity: 0.42

### Cash Flow
- Operating Cash Flow: $18.5B
- Free Cash Flow: $12.3B
- Cash Position: $28.7B

### Growth Prospects
- 5-Year Revenue CAGR: 22%
- Analyst Consensus: Outperform
- Average Price Target: $210

### Dividend
- Dividend Yield: 0.52%
- Payout Ratio: 12%""",
            
            "sentiment": f"""## Social Sentiment Analysis for {ticker}

### Overall Sentiment Score: 72/100

### Social Media Analysis
- Twitter/X Mentions: 15,420 (last 24h)
- Reddit Discussions: 2,340 threads
- StockTwits Activity: Very High

### Sentiment Breakdown
- Bullish: 65%
- Neutral: 22%
- Bearish: 13%

### Key Topics Trending
1. AI integration and growth potential
2. Strong earnings performance
3. New product announcements
4. Institutional buying activity

### Influencer Sentiment
- Tech analysts: Mostly positive
- Financial media: Cautiously optimistic
- Retail investors: Very bullish

### Social Volume Trend
Volume up 45% compared to 7-day average, indicating heightened interest."""
        },
        "investment_plan": f"""## Investment Plan for {ticker}

### Position Sizing
- Recommended allocation: 3-5% of portfolio
- Entry strategy: Dollar-cost averaging over 2-3 weeks

### Entry Points
- Primary entry: Current price ($185.50)
- Secondary entry: $175-178 (on pullback)

### Exit Strategy
- Target 1: $200 (take 30% profit)
- Target 2: $215 (take 40% profit)
- Final target: $230+ (hold remainder)

### Stop Loss
- Initial stop: $165 (10.8% downside)
- Trailing stop: 15% from peak

### Timeline
- Investment horizon: 6-12 months
- Review frequency: Monthly""",
        "final_decision": f"""## Final Trading Decision

**Action: MODERATE BUY**

| Metric | Value |
|--------|-------|
| Ticker | {ticker} |
| Direction | Long |
| Confidence | 72% |
| Risk Level | Medium |
| Time Horizon | 6-12 months |

**Summary**: {ticker} presents a favorable risk-reward opportunity based on strong fundamentals, positive sentiment, and technical setup. Recommend gradual accumulation with defined risk parameters."""
    }


def run_analysis_sync(task_id: str, request: AnalysisRequest):
    """Run analysis synchronously (called in background)"""
    try:
        analysis_tasks[task_id]["status"] = "running"
        analysis_tasks[task_id]["progress"] = "Initializing agents..."
        
        # Get user_id for database tracking
        user_id = analysis_tasks[task_id].get("user_id")
        
        # TEST MODE: Use mock data
        if TEST_MODE:
            analysis_tasks[task_id]["progress"] = f"[TEST MODE] Generating mock analysis for {request.ticker}..."
            result = get_mock_analysis_result(request.ticker, request.date)
            
            # Auto-translate to Chinese (mock)
            analysis_tasks[task_id]["progress"] = "[TEST MODE] Generating mock translation..."
            translated = translate_result(result)
            result["translated"] = translated
            
            analysis_tasks[task_id]["status"] = "completed"
            analysis_tasks[task_id]["result"] = result
            analysis_tasks[task_id]["progress"] = "Analysis complete!"
            
            decision_summary = result["decision"][:100] + "..."
            save_to_history(task_id, analysis_tasks[task_id]["request"], "completed", decision_summary, full_result=result)
            return
        
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
        
        # Extract results (English)
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
        
        # Auto-translate to Chinese
        analysis_tasks[task_id]["progress"] = "Translating to Chinese..."
        try:
            translated = translate_result(result)
            result["translated"] = translated  # Add Chinese translation
        except Exception as translate_error:
            print(f"Translation failed: {translate_error}")
            result["translated"] = None  # Translation failed
        
        analysis_tasks[task_id]["status"] = "completed"
        analysis_tasks[task_id]["result"] = result
        analysis_tasks[task_id]["progress"] = "Analysis complete!"
        
        # Save to history with full result (JSON file)
        decision_summary = decision[:100] + "..." if len(decision) > 100 else decision
        save_to_history(task_id, analysis_tasks[task_id]["request"], "completed", decision_summary, full_result=result)
        
        # Save to database for quota tracking (GitHub OAuth users)
        if user_id and user_id > 0:
            try:
                import json
                with get_db() as db:
                    save_analysis(
                        db,
                        user_id=user_id,
                        ticker=request.ticker,
                        analysis_date=request.date,
                        llm_provider=request.llm_provider,
                        decision=decision,
                        decision_cn=translated.get("decision") if translated else None,
                        reports=json.dumps(result.get("reports", {})),
                        reports_cn=json.dumps(translated.get("reports", {})) if translated else None
                    )
            except Exception as db_error:
                print(f"Failed to save analysis to database: {db_error}")
        
    except Exception as e:
        analysis_tasks[task_id]["status"] = "failed"
        analysis_tasks[task_id]["error"] = str(e)
        analysis_tasks[task_id]["progress"] = f"Error: {str(e)}"
        # Save failed record to history
        save_to_history(task_id, analysis_tasks[task_id]["request"], "failed", str(e)[:100], full_result=None)


# ============== Login Page ==============
def get_login_html(error_msg: str = None, github_enabled: bool = False):
    """Return login page HTML with GitHub OAuth option"""
    error_html = ""
    if error_msg:
        error_html = f'<span class="text-red-400">{error_msg}</span>'
    else:
        error_html = "请登录以继续 / Please login to continue"
    
    github_button = ""
    divider = ""
    if github_enabled:
        github_button = """
            <a href="/auth/github"
                class="w-full flex items-center justify-center gap-3 bg-gray-900 hover:bg-gray-700 text-white font-semibold py-4 px-6 rounded-lg transition-all duration-200 border-2 border-gray-600 hover:border-gray-500 mb-4">
                <svg class="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
                    <path fill-rule="evenodd" d="M10 0C4.477 0 0 4.484 0 10.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0110 4.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.203 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.942.359.31.678.921.678 1.856 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0020 10.017C20 4.484 15.522 0 10 0z" clip-rule="evenodd"/>
                </svg>
                <span>使用 GitHub 登录 / Login with GitHub</span>
            </a>
        """
        divider = """
            <div class="relative my-6">
                <div class="absolute inset-0 flex items-center">
                    <div class="w-full border-t border-gray-600"></div>
                </div>
                <div class="relative flex justify-center text-sm">
                    <span class="px-4 bg-gray-800 text-gray-400">或使用密码 / Or use password</span>
                </div>
            </div>
        """
    
    return f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>登录 - TradingAgents</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .gradient-bg {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        }}
    </style>
</head>
<body class="gradient-bg text-white min-h-screen flex items-center justify-center">
    <div class="w-full max-w-md p-8">
        <div class="bg-gray-800/50 backdrop-blur-lg rounded-2xl p-8 border border-gray-700">
            <div class="text-center mb-8">
                <h1 class="text-3xl font-bold mb-2">🤖 TradingAgents</h1>
                <p class="text-gray-400">{error_html}</p>
            </div>
            
            {github_button}
            
            {divider}
            
            <form action="/login" method="POST" class="space-y-6">
                <div>
                    <label class="block text-sm font-medium text-gray-300 mb-2">用户名 / Username</label>
                    <input type="text" name="username" required
                        class="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-3 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        placeholder="输入用户名 / Enter username">
                </div>
                
                <div>
                    <label class="block text-sm font-medium text-gray-300 mb-2">密码 / Password</label>
                    <input type="password" name="password" required
                        class="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-3 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        placeholder="输入密码 / Enter password">
                </div>
                
                <button type="submit"
                    class="w-full bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white font-semibold py-3 px-6 rounded-lg transition-all duration-200">
                    🔐 登录 / Login
                </button>
            </form>
            
            <p class="text-center text-gray-500 text-sm mt-6">
                安全会话认证 / Protected by session-based authentication
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
    
    # Check if GitHub OAuth is configured
    github_enabled = github_auth.is_github_configured()
    return HTMLResponse(content=get_login_html(github_enabled=github_enabled))


@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    """Process login form (password authentication)"""
    if username == AUTH_USERNAME and verify_password(password):
        # Create or get user in database for password login
        with get_db() as db:
            user = get_or_create_password_user(db, username)
            user_data = {
                "id": user.id,  # Real database ID for journal/trading features
                "username": user.username,
                "name": user.name or username,
                "avatar_url": user.avatar_url,
                "is_admin": True,
                "login_type": "password"
            }
        
        token = create_session(user_data)
        response = RedirectResponse(url="/", status_code=302)
        # Check if running in production (Railway sets PORT env var)
        is_production = os.environ.get("RAILWAY_ENVIRONMENT") or os.environ.get("PORT")
        response.set_cookie(
            key="session_token",
            value=token,
            httponly=True,
            secure=bool(is_production),  # Only use secure cookie in production (HTTPS)
            samesite="lax",
            max_age=86400  # 24 hours
        )
        return response
    else:
        github_enabled = github_auth.is_github_configured()
        return HTMLResponse(
            content=get_login_html(
                error_msg="用户名或密码错误 / Invalid username or password",
                github_enabled=github_enabled
            ),
            status_code=401
        )


# ============== GitHub OAuth Routes ==============
@app.get("/auth/github")
async def github_login(request: Request):
    """Redirect to GitHub OAuth authorization page"""
    if not github_auth.is_github_configured():
        raise HTTPException(status_code=500, detail="GitHub OAuth is not configured")
    
    # Generate state for CSRF protection
    state = github_auth.generate_oauth_state()
    oauth_states[state] = {
        "created": datetime.now(),
        "expires": datetime.now() + timedelta(minutes=10)
    }
    
    # Clean up old states
    now = datetime.now()
    expired_states = [s for s, v in oauth_states.items() if now > v["expires"]]
    for s in expired_states:
        del oauth_states[s]
    
    auth_url = github_auth.get_github_authorize_url(state=state)
    return RedirectResponse(url=auth_url, status_code=302)


@app.get("/auth/github/callback")
async def github_callback(request: Request, code: str = None, state: str = None, error: str = None):
    """Handle GitHub OAuth callback"""
    # Check for errors from GitHub
    if error:
        return HTMLResponse(
            content=get_login_html(
                error_msg=f"GitHub 登录失败: {error}",
                github_enabled=True
            ),
            status_code=400
        )
    
    if not code:
        return HTMLResponse(
            content=get_login_html(
                error_msg="缺少授权码 / Missing authorization code",
                github_enabled=True
            ),
            status_code=400
        )
    
    # Verify state to prevent CSRF
    if state:
        if state not in oauth_states:
            return HTMLResponse(
                content=get_login_html(
                    error_msg="无效的状态参数 / Invalid state parameter",
                    github_enabled=True
                ),
                status_code=400
            )
        del oauth_states[state]
    
    try:
        # Exchange code for access token
        access_token = await github_auth.exchange_code_for_token(code)
        if not access_token:
            return HTMLResponse(
                content=get_login_html(
                    error_msg="获取访问令牌失败 / Failed to get access token",
                    github_enabled=True
                ),
                status_code=400
            )
        
        # Get GitHub user info
        github_user = await github_auth.get_github_user(access_token)
        if not github_user:
            return HTMLResponse(
                content=get_login_html(
                    error_msg="获取用户信息失败 / Failed to get user info",
                    github_enabled=True
                ),
                status_code=400
            )
        
        # Create or update user in database
        with get_db() as db:
            user = github_auth.create_user_from_github(db, github_user)
            
            # Create session in database
            client_ip = request.client.host if request.client else None
            user_agent = request.headers.get("user-agent", "")[:500]
            session, token = github_auth.create_user_session(db, user, client_ip, user_agent)
        
        # Redirect to home with session cookie
        response = RedirectResponse(url="/", status_code=302)
        is_production = os.environ.get("RAILWAY_ENVIRONMENT") or os.environ.get("PORT")
        response.set_cookie(
            key="session_token",
            value=token,
            httponly=True,
            secure=bool(is_production),
            samesite="lax",
            max_age=86400 * 7  # 7 days for GitHub OAuth sessions
        )
        return response
        
    except Exception as e:
        print(f"GitHub OAuth error: {e}")
        return HTMLResponse(
            content=get_login_html(
                error_msg=f"登录过程中出错 / Login error: {str(e)[:50]}",
                github_enabled=True
            ),
            status_code=500
        )


@app.get("/logout")
async def logout(request: Request):
    """Logout and clear session"""
    token = get_session_token(request)
    
    # Remove from in-memory sessions
    if token and token in sessions:
        del sessions[token]
    
    # Remove from database sessions
    if token:
        with get_db() as db:
            github_auth.logout_user(db, token)
    
    response = RedirectResponse(url="/login", status_code=302)
    response.delete_cookie("session_token")
    return response


# ============== User Info API ==============
@app.get("/api/user")
async def get_user_info(request: Request):
    """Get current user information including daily analysis quota"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    user = get_current_user(request)
    
    # Default response for password login or no user data
    result = {
        "id": 0,
        "username": "user",
        "name": "User",
        "avatar_url": None,
        "login_type": "password",
        "daily_limit": -1,  # -1 means unlimited
        "used_today": 0,
        "remaining_today": -1
    }
    
    if user:
        result.update(user)
        
        # Check quota for GitHub OAuth users
        login_type = user.get("login_type")
        user_id = user.get("id", 0)
        
        if login_type != "password" and user_id > 0:
            # GitHub OAuth user - check daily limit
            with get_db() as db:
                can_analyze, used, remaining = can_user_analyze(db, user_id, DAILY_ANALYSIS_LIMIT)
                result["daily_limit"] = DAILY_ANALYSIS_LIMIT
                result["used_today"] = used
                result["remaining_today"] = remaining
        else:
            # Password login - unlimited
            result["daily_limit"] = -1
            result["used_today"] = 0
            result["remaining_today"] = -1
    
    return result


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Redirect to dashboard"""
    token = get_session_token(request)
    if not verify_session(token):
        return RedirectResponse(url="/login", status_code=302)
    return RedirectResponse(url="/dashboard", status_code=302)


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Serve the dashboard page (protected)"""
    token = get_session_token(request)
    if not verify_session(token):
        return RedirectResponse(url="/login", status_code=302)
    
    html_path = os.path.join(os.path.dirname(__file__), "templates", "dashboard.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return RedirectResponse(url="/analyze", status_code=302)


@app.get("/analyze", response_class=HTMLResponse)
async def analyze_page(request: Request):
    """Serve the analysis page (protected)"""
    token = get_session_token(request)
    if not verify_session(token):
        return RedirectResponse(url="/login", status_code=302)
    
    html_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content=get_default_html())


@app.get("/api/health")
async def health_check():
    """Health check endpoint (public)"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# ============== Stock Data APIs ==============
@app.get("/api/stock/{ticker}")
async def get_stock_info(request: Request, ticker: str):
    """Get stock information by ticker (protected)"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker.upper())
        info = stock.info
        
        return {
            "symbol": info.get("symbol", ticker.upper()),
            "name": info.get("longName") or info.get("shortName", "N/A"),
            "price": info.get("regularMarketPrice") or info.get("currentPrice"),
            "change": info.get("regularMarketChange", 0),
            "changePercent": info.get("regularMarketChangePercent", 0),
            "open": info.get("regularMarketOpen"),
            "high": info.get("regularMarketDayHigh"),
            "low": info.get("regularMarketDayLow"),
            "volume": info.get("regularMarketVolume"),
            "marketCap": info.get("marketCap"),
            "peRatio": info.get("trailingPE"),
            "eps": info.get("trailingEps"),
            "pbRatio": info.get("priceToBook"),
            "dividendYield": info.get("dividendYield"),
            "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
            "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
            "fiftyDayAverage": info.get("fiftyDayAverage"),
            "twoHundredDayAverage": info.get("twoHundredDayAverage"),
            "beta": info.get("beta"),
            "averageVolume": info.get("averageVolume"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "description": info.get("longBusinessSummary", "")[:500] if info.get("longBusinessSummary") else ""
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch stock data: {str(e)}")


@app.get("/api/market/overview")
async def get_market_overview(request: Request):
    """Get market overview data (protected)"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        import yfinance as yf
        
        tickers = ["SPY", "QQQ", "DIA", "IWM"]
        result = {}
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                result[ticker] = {
                    "price": info.get("regularMarketPrice") or info.get("previousClose", 0),
                    "change": info.get("regularMarketChange", 0),
                    "changePercent": info.get("regularMarketChangePercent", 0)
                }
            except:
                result[ticker] = {"price": 0, "change": 0, "changePercent": 0}
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch market data: {str(e)}")


@app.post("/api/analyze")
async def start_analysis(request: Request, analysis_request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Start a new analysis task (protected)"""
    # Check authentication
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    # Get current user info
    current_user = get_current_user(request)
    user_id = None
    login_type = None
    
    if current_user:
        user_id = current_user.get("id")
        login_type = current_user.get("login_type")
    
    # Check daily limit for GitHub OAuth users (not password login or TEST_MODE)
    if not TEST_MODE and login_type != "password" and user_id and user_id > 0:
        with get_db() as db:
            can_analyze, used, remaining = can_user_analyze(db, user_id, DAILY_ANALYSIS_LIMIT)
            if not can_analyze:
                raise HTTPException(
                    status_code=429,
                    detail={
                        "message": "今日分析次数已用完 / Daily analysis limit reached",
                        "used": used,
                        "limit": DAILY_ANALYSIS_LIMIT,
                        "remaining": 0
                    }
                )
    
    import uuid
    task_id = str(uuid.uuid4())[:8]
    
    analysis_tasks[task_id] = {
        "status": "pending",
        "progress": "Queued for analysis...",
        "result": None,
        "error": None,
        "request": analysis_request.dict(),
        "user_id": user_id  # Track which user started this analysis
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
        "llm_providers": ["deepseek", "google", "openai", "anthropic", "openrouter", "ollama"],
        "default_provider": "deepseek",
        "models": {
            "deepseek": {
                "quick": ["deepseek-chat"],
                "deep": ["deepseek-chat", "deepseek-reasoner"]
            },
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
    if task_id in analysis_tasks and analysis_tasks[task_id].get("result"):
        return {"result": analysis_tasks[task_id]["result"], "source": "memory"}
    
    # Check in history for full result
    for record in search_history:
        if record["task_id"] == task_id:
            if record.get("full_result"):
                return {"result": record["full_result"], "source": "history"}
            else:
                return {"history_record": record, "message": "Full result not available"}
    
    raise HTTPException(status_code=404, detail="History record not found")


@app.get("/api/history/{task_id}/download")
async def download_history_report(request: Request, task_id: str, format: str = "pdf"):
    """Download historical analysis report (protected)"""
    # Check authentication
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    # Find the record
    result = None
    for record in search_history:
        if record["task_id"] == task_id and record.get("full_result"):
            result = record["full_result"]
            break
    
    if not result:
        # Check in memory
        if task_id in analysis_tasks and analysis_tasks[task_id].get("result"):
            result = analysis_tasks[task_id]["result"]
    
    if not result:
        raise HTTPException(status_code=404, detail="Report not found")
    
    if format == "pdf":
        # Generate PDF
        return await generate_pdf_from_result(result)
    elif format == "md":
        # Generate Markdown
        return generate_markdown_from_result(result)
    elif format == "txt":
        # Generate plain text
        return generate_text_from_result(result)
    else:
        raise HTTPException(status_code=400, detail="Unsupported format. Use: pdf, md, txt")


def generate_markdown_from_result(result: dict):
    """Generate a well-formatted Markdown report"""
    ticker = result.get("ticker", "Unknown")
    date = result.get("date", "")
    
    md_content = f"""# 📊 Trading Analysis Report

**Stock:** {ticker}  
**Date:** {date}  
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## 🎯 Trading Decision

{result.get('decision', 'N/A')}

---

## 📈 Market Analysis

{result.get('reports', {}).get('market', 'N/A')}

---

## 📰 News Analysis

{result.get('reports', {}).get('news', 'N/A')}

---

## 📋 Fundamentals Analysis

{result.get('reports', {}).get('fundamentals', 'N/A')}

---

## 💬 Sentiment Analysis

{result.get('reports', {}).get('sentiment', 'N/A')}

---

## 📝 Investment Plan

{result.get('investment_plan', 'N/A')}

---

## 🏁 Final Decision

{result.get('final_decision', 'N/A')}

---

*Generated by TradingAgents - Multi-Agent LLM Financial Trading Framework*
"""
    
    filename = f"TradingAgents_{ticker}_{date}.md"
    return Response(
        content=md_content,
        media_type="text/markdown",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


def generate_text_from_result(result: dict):
    """Generate plain text report"""
    ticker = result.get("ticker", "Unknown")
    date = result.get("date", "")
    
    text_content = f"""════════════════════════════════════════════════════════════════
                    TRADING ANALYSIS REPORT
════════════════════════════════════════════════════════════════

Stock: {ticker}
Date: {date}
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

════════════════════════════════════════════════════════════════
                      TRADING DECISION
════════════════════════════════════════════════════════════════

{result.get('decision', 'N/A')}

════════════════════════════════════════════════════════════════
                      MARKET ANALYSIS
════════════════════════════════════════════════════════════════

{result.get('reports', {}).get('market', 'N/A')}

════════════════════════════════════════════════════════════════
                       NEWS ANALYSIS
════════════════════════════════════════════════════════════════

{result.get('reports', {}).get('news', 'N/A')}

════════════════════════════════════════════════════════════════
                   FUNDAMENTALS ANALYSIS
════════════════════════════════════════════════════════════════

{result.get('reports', {}).get('fundamentals', 'N/A')}

════════════════════════════════════════════════════════════════
                    SENTIMENT ANALYSIS
════════════════════════════════════════════════════════════════

{result.get('reports', {}).get('sentiment', 'N/A')}

════════════════════════════════════════════════════════════════
                      INVESTMENT PLAN
════════════════════════════════════════════════════════════════

{result.get('investment_plan', 'N/A')}

════════════════════════════════════════════════════════════════
                      FINAL DECISION
════════════════════════════════════════════════════════════════

{result.get('final_decision', 'N/A')}

════════════════════════════════════════════════════════════════
Generated by TradingAgents - Multi-Agent LLM Financial Trading Framework
════════════════════════════════════════════════════════════════
"""
    
    filename = f"TradingAgents_{ticker}_{date}.txt"
    return Response(
        content=text_content,
        media_type="text/plain",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


async def generate_pdf_from_result(result: dict):
    """Generate PDF from result"""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50)
        
        styles = getSampleStyleSheet()
        
        # Custom styles for better formatting
        styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=styles['Heading1'],
            fontSize=22,
            alignment=TA_CENTER,
            spaceAfter=20,
            textColor=colors.HexColor('#1a365d')
        ))
        styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=styles['Heading2'],
            fontSize=14,
            spaceBefore=15,
            spaceAfter=10,
            textColor=colors.HexColor('#2c5282'),
            borderWidth=1,
            borderColor=colors.HexColor('#e2e8f0'),
            borderPadding=5
        ))
        styles.add(ParagraphStyle(
            name='BodyText',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=8,
            leading=14,
            textColor=colors.HexColor('#2d3748')
        ))
        styles.add(ParagraphStyle(
            name='MetaInfo',
            parent=styles['Normal'],
            fontSize=11,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#718096'),
            spaceAfter=20
        ))
        
        story = []
        
        ticker = result.get('ticker', 'Unknown')
        date = result.get('date', '')
        
        # Title
        story.append(Paragraph("📊 Trading Analysis Report", styles['ReportTitle']))
        story.append(Paragraph(f"<b>Stock:</b> {ticker} | <b>Date:</b> {date}", styles['MetaInfo']))
        story.append(Spacer(1, 10))
        
        # Decision Summary
        story.append(Paragraph("🎯 Trading Decision", styles['SectionHeader']))
        decision_text = result.get('decision', 'N/A')
        if decision_text:
            # Clean and format text for PDF
            clean_text = str(decision_text).replace('\n', '<br/>')[:4000]
            story.append(Paragraph(clean_text, styles['BodyText']))
        story.append(Spacer(1, 10))
        
        # Reports
        reports = result.get('reports', {})
        
        section_configs = [
            ('market', '📈 Market Analysis'),
            ('news', '📰 News Analysis'),
            ('fundamentals', '📋 Fundamentals Analysis'),
            ('sentiment', '💬 Sentiment Analysis')
        ]
        
        for key, title in section_configs:
            content = reports.get(key, '')
            if content:
                story.append(Paragraph(title, styles['SectionHeader']))
                clean_text = str(content).replace('\n', '<br/>')[:4000]
                story.append(Paragraph(clean_text, styles['BodyText']))
                story.append(Spacer(1, 8))
        
        # Investment Plan
        if result.get('investment_plan'):
            story.append(Paragraph("📝 Investment Plan", styles['SectionHeader']))
            clean_text = str(result['investment_plan']).replace('\n', '<br/>')[:4000]
            story.append(Paragraph(clean_text, styles['BodyText']))
        
        # Final Decision
        if result.get('final_decision'):
            story.append(Paragraph("🏁 Final Decision", styles['SectionHeader']))
            clean_text = str(result['final_decision']).replace('\n', '<br/>')[:4000]
            story.append(Paragraph(clean_text, styles['BodyText']))
        
        # Footer
        story.append(Spacer(1, 30))
        story.append(Paragraph(
            f"<i>Generated by TradingAgents on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>",
            styles['MetaInfo']
        ))
        
        doc.build(story)
        buffer.seek(0)
        
        filename = f"TradingAgents_{ticker}_{date}.pdf"
        
        return StreamingResponse(
            buffer,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except ImportError:
        raise HTTPException(status_code=500, detail="PDF generation requires reportlab")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")


@app.delete("/api/history")
async def clear_history(request: Request):
    """Clear all history (protected)"""
    global search_history
    # Check authentication
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    search_history = []
    save_history_to_disk()  # Persist the clear
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


# ============== Trading Journal API ==============

class JournalCreate(BaseModel):
    title: str
    content: str
    journal_date: str
    ticker: Optional[str] = None
    trade_type: Optional[str] = None  # buy, sell, hold, watch
    trade_price: Optional[str] = None
    tags: Optional[List[str]] = None
    mood: Optional[str] = None  # bullish, bearish, neutral, cautious


class JournalUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    journal_date: Optional[str] = None
    ticker: Optional[str] = None
    trade_type: Optional[str] = None
    trade_price: Optional[str] = None
    tags: Optional[List[str]] = None
    mood: Optional[str] = None


@app.get("/api/journals")
async def get_journals(request: Request, limit: int = 50, offset: int = 0):
    """Get current user's trading journals"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    current_user = get_current_user(request)
    if not current_user or current_user.get("id", 0) <= 0:
        # Password login users don't have database ID
        return {"journals": [], "total": 0, "message": "请先登录 / Please login first"}
    
    user_id = current_user["id"]
    
    with get_db() as db:
        journals = get_user_journals(db, user_id, limit=limit, offset=offset)
        total = get_user_journal_count(db, user_id)
        
        return {
            "journals": [j.to_dict() for j in journals],
            "total": total,
            "limit": limit,
            "offset": offset
        }


@app.post("/api/journals")
async def create_journal_entry(request: Request, journal_data: JournalCreate):
    """Create a new trading journal entry"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    current_user = get_current_user(request)
    if not current_user or current_user.get("id", 0) <= 0:
        raise HTTPException(status_code=403, detail="请先登录 / Please login first")
    
    user_id = current_user["id"]
    
    # Convert tags list to comma-separated string
    tags_str = ",".join(journal_data.tags) if journal_data.tags else None
    
    with get_db() as db:
        journal = create_journal(
            db,
            user_id=user_id,
            title=journal_data.title,
            content=journal_data.content,
            journal_date=journal_data.journal_date,
            ticker=journal_data.ticker.upper() if journal_data.ticker else None,
            trade_type=journal_data.trade_type,
            trade_price=journal_data.trade_price,
            tags=tags_str,
            mood=journal_data.mood
        )
        
        return {"message": "Journal created", "journal": journal.to_dict()}


@app.get("/api/journals/{journal_id}")
async def get_journal_detail(request: Request, journal_id: int):
    """Get a specific journal entry"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    current_user = get_current_user(request)
    if not current_user or current_user.get("id", 0) <= 0:
        raise HTTPException(status_code=403, detail="请先登录 / Please login first")
    
    user_id = current_user["id"]
    
    with get_db() as db:
        journal = get_user_journal_by_id(db, user_id, journal_id)
        if not journal:
            raise HTTPException(status_code=404, detail="Journal not found")
        
        return {"journal": journal.to_dict()}


@app.put("/api/journals/{journal_id}")
async def update_journal_entry(request: Request, journal_id: int, journal_data: JournalUpdate):
    """Update a journal entry"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    current_user = get_current_user(request)
    if not current_user or current_user.get("id", 0) <= 0:
        raise HTTPException(status_code=403, detail="请先登录 / Please login first")
    
    user_id = current_user["id"]
    
    with get_db() as db:
        journal = get_user_journal_by_id(db, user_id, journal_id)
        if not journal:
            raise HTTPException(status_code=404, detail="Journal not found")
        
        # Build update dict from provided fields
        update_data = {}
        if journal_data.title is not None:
            update_data["title"] = journal_data.title
        if journal_data.content is not None:
            update_data["content"] = journal_data.content
        if journal_data.journal_date is not None:
            update_data["journal_date"] = journal_data.journal_date
        if journal_data.ticker is not None:
            update_data["ticker"] = journal_data.ticker.upper() if journal_data.ticker else None
        if journal_data.trade_type is not None:
            update_data["trade_type"] = journal_data.trade_type
        if journal_data.trade_price is not None:
            update_data["trade_price"] = journal_data.trade_price
        if journal_data.tags is not None:
            update_data["tags"] = ",".join(journal_data.tags) if journal_data.tags else None
        if journal_data.mood is not None:
            update_data["mood"] = journal_data.mood
        
        if update_data:
            journal = update_journal(db, journal, **update_data)
        
        return {"message": "Journal updated", "journal": journal.to_dict()}


@app.delete("/api/journals/{journal_id}")
async def delete_journal_entry(request: Request, journal_id: int):
    """Delete a journal entry"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    current_user = get_current_user(request)
    if not current_user or current_user.get("id", 0) <= 0:
        raise HTTPException(status_code=403, detail="请先登录 / Please login first")
    
    user_id = current_user["id"]
    
    with get_db() as db:
        journal = get_user_journal_by_id(db, user_id, journal_id)
        if not journal:
            raise HTTPException(status_code=404, detail="Journal not found")
        
        delete_journal(db, journal)
        
        return {"message": "Journal deleted", "id": journal_id}


@app.get("/journal", response_class=HTMLResponse)
async def journal_page(request: Request):
    """Serve the trading journal page (protected)"""
    token = get_session_token(request)
    if not verify_session(token):
        return RedirectResponse(url="/login", status_code=302)
    
    html_path = os.path.join(os.path.dirname(__file__), "templates", "journal.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    
    # Return a basic journal page if template not found
    return HTMLResponse(content=get_journal_html())


def get_journal_html():
    """Return basic journal HTML if template not found"""
    return """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>交易日记 - TradingAgents</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-900 text-white min-h-screen">
    <div class="container mx-auto px-4 py-8">
        <h1 class="text-4xl font-bold text-center mb-8">📔 交易日记</h1>
        <p class="text-center text-gray-400">请使用完整的模板页面</p>
        <p class="text-center mt-4"><a href="/dashboard" class="text-blue-400 hover:underline">返回首页</a></p>
    </div>
</body>
</html>
"""


# ==================== Simulated Trading (Paper Trading) API ====================

class PortfolioCreate(BaseModel):
    name: str
    initial_balance: float = 100000.0
    description: Optional[str] = None


class PortfolioUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None


class TradeCreate(BaseModel):
    portfolio_id: int
    ticker: str
    asset_type: str  # stock, call, put, spread
    strategy: str  # momentum, value, swing, scalp, dividend, covered_call, wheel, other
    trade_direction: str  # long, short
    entry_price: float
    quantity: float
    entry_date: Optional[str] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    notes: Optional[str] = None
    market_conditions: Optional[str] = None
    news_events: Optional[str] = None
    tags: Optional[str] = None


class TradeClose(BaseModel):
    exit_price: float
    exit_date: Optional[str] = None
    notes: Optional[str] = None


class TradeUpdate(BaseModel):
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    notes: Optional[str] = None
    market_conditions: Optional[str] = None
    news_events: Optional[str] = None
    tags: Optional[str] = None


# Portfolio API endpoints

@app.get("/api/portfolios")
async def list_portfolios(request: Request, active_only: bool = True):
    """List user's portfolios"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    current_user = get_current_user(request)
    if not current_user or current_user.get("id", 0) <= 0:
        raise HTTPException(status_code=403, detail="请先登录 / Please login first")
    
    user_id = current_user["id"]
    
    with get_db() as db:
        portfolios = get_user_portfolios(db, user_id, active_only=active_only)
        return {"portfolios": [p.to_dict() for p in portfolios]}


@app.post("/api/portfolios")
async def create_portfolio_endpoint(request: Request, portfolio_data: PortfolioCreate):
    """Create a new portfolio"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    current_user = get_current_user(request)
    if not current_user or current_user.get("id", 0) <= 0:
        raise HTTPException(status_code=403, detail="请先登录 / Please login first")
    
    user_id = current_user["id"]
    
    with get_db() as db:
        portfolio = create_portfolio(
            db,
            user_id=user_id,
            name=portfolio_data.name,
            initial_balance=portfolio_data.initial_balance,
            description=portfolio_data.description
        )
        return {"message": "Portfolio created", "portfolio": portfolio.to_dict()}


@app.get("/api/portfolios/{portfolio_id}")
async def get_portfolio_endpoint(request: Request, portfolio_id: int):
    """Get portfolio details with stats"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    current_user = get_current_user(request)
    if not current_user or current_user.get("id", 0) <= 0:
        raise HTTPException(status_code=403, detail="请先登录 / Please login first")
    
    user_id = current_user["id"]
    
    with get_db() as db:
        portfolio = get_user_portfolio_by_id(db, user_id, portfolio_id)
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")
        
        stats = get_portfolio_stats(db, portfolio_id)
        return stats


@app.put("/api/portfolios/{portfolio_id}")
async def update_portfolio_endpoint(request: Request, portfolio_id: int, update_data: PortfolioUpdate):
    """Update portfolio"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    current_user = get_current_user(request)
    if not current_user or current_user.get("id", 0) <= 0:
        raise HTTPException(status_code=403, detail="请先登录 / Please login first")
    
    user_id = current_user["id"]
    
    with get_db() as db:
        portfolio = get_user_portfolio_by_id(db, user_id, portfolio_id)
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")
        
        update_fields = {k: v for k, v in update_data.dict().items() if v is not None}
        if update_fields:
            portfolio = update_portfolio(db, portfolio, **update_fields)
        
        return {"message": "Portfolio updated", "portfolio": portfolio.to_dict()}


@app.delete("/api/portfolios/{portfolio_id}")
async def delete_portfolio_endpoint(request: Request, portfolio_id: int):
    """Delete portfolio (soft delete)"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    current_user = get_current_user(request)
    if not current_user or current_user.get("id", 0) <= 0:
        raise HTTPException(status_code=403, detail="请先登录 / Please login first")
    
    user_id = current_user["id"]
    
    with get_db() as db:
        portfolio = get_user_portfolio_by_id(db, user_id, portfolio_id)
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")
        
        delete_portfolio(db, portfolio)
        return {"message": "Portfolio deleted", "id": portfolio_id}


# Trade API endpoints

@app.get("/api/trades")
async def list_trades(
    request: Request,
    portfolio_id: Optional[int] = None,
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """List user's trades"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    current_user = get_current_user(request)
    if not current_user or current_user.get("id", 0) <= 0:
        raise HTTPException(status_code=403, detail="请先登录 / Please login first")
    
    user_id = current_user["id"]
    
    with get_db() as db:
        trades = get_user_trades(db, user_id, portfolio_id=portfolio_id, status=status, limit=limit, offset=offset)
        return {"trades": [t.to_dict() for t in trades]}


@app.post("/api/trades")
async def create_trade_endpoint(request: Request, trade_data: TradeCreate):
    """Create a new paper trade"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    current_user = get_current_user(request)
    if not current_user or current_user.get("id", 0) <= 0:
        raise HTTPException(status_code=403, detail="请先登录 / Please login first")
    
    user_id = current_user["id"]
    
    # Validate asset_type
    valid_asset_types = ['stock', 'call', 'put', 'spread']
    if trade_data.asset_type not in valid_asset_types:
        raise HTTPException(status_code=400, detail=f"Invalid asset_type. Must be one of: {valid_asset_types}")
    
    # Validate strategy
    valid_strategies = ['momentum', 'value', 'swing', 'scalp', 'dividend', 'covered_call', 'wheel', 'other']
    if trade_data.strategy not in valid_strategies:
        raise HTTPException(status_code=400, detail=f"Invalid strategy. Must be one of: {valid_strategies}")
    
    # Validate trade_direction
    valid_directions = ['long', 'short']
    if trade_data.trade_direction not in valid_directions:
        raise HTTPException(status_code=400, detail=f"Invalid trade_direction. Must be one of: {valid_directions}")
    
    with get_db() as db:
        # Verify portfolio belongs to user
        portfolio = get_user_portfolio_by_id(db, user_id, trade_data.portfolio_id)
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")
        
        trade = create_paper_trade(
            db,
            user_id=user_id,
            portfolio_id=trade_data.portfolio_id,
            ticker=trade_data.ticker,
            asset_type=trade_data.asset_type,
            strategy=trade_data.strategy,
            trade_direction=trade_data.trade_direction,
            entry_price=trade_data.entry_price,
            quantity=trade_data.quantity,
            entry_date=trade_data.entry_date,
            stop_loss=trade_data.stop_loss,
            take_profit=trade_data.take_profit,
            notes=trade_data.notes,
            market_conditions=trade_data.market_conditions,
            news_events=trade_data.news_events,
            tags=trade_data.tags
        )
        return {"message": "Trade created", "trade": trade.to_dict()}


@app.get("/api/trades/{trade_id}")
async def get_trade_endpoint(request: Request, trade_id: int):
    """Get trade details"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    current_user = get_current_user(request)
    if not current_user or current_user.get("id", 0) <= 0:
        raise HTTPException(status_code=403, detail="请先登录 / Please login first")
    
    user_id = current_user["id"]
    
    with get_db() as db:
        trade = get_user_trade_by_id(db, user_id, trade_id)
        if not trade:
            raise HTTPException(status_code=404, detail="Trade not found")
        
        return {"trade": trade.to_dict()}


@app.put("/api/trades/{trade_id}")
async def update_trade_endpoint(request: Request, trade_id: int, update_data: TradeUpdate):
    """Update trade (for open trades)"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    current_user = get_current_user(request)
    if not current_user or current_user.get("id", 0) <= 0:
        raise HTTPException(status_code=403, detail="请先登录 / Please login first")
    
    user_id = current_user["id"]
    
    with get_db() as db:
        trade = get_user_trade_by_id(db, user_id, trade_id)
        if not trade:
            raise HTTPException(status_code=404, detail="Trade not found")
        
        update_fields = {k: v for k, v in update_data.dict().items() if v is not None}
        if update_fields:
            trade = update_trade(db, trade, **update_fields)
        
        return {"message": "Trade updated", "trade": trade.to_dict()}


@app.post("/api/trades/{trade_id}/close")
async def close_trade_endpoint(request: Request, trade_id: int, close_data: TradeClose):
    """Close a trade"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    current_user = get_current_user(request)
    if not current_user or current_user.get("id", 0) <= 0:
        raise HTTPException(status_code=403, detail="请先登录 / Please login first")
    
    user_id = current_user["id"]
    
    with get_db() as db:
        trade = get_user_trade_by_id(db, user_id, trade_id)
        if not trade:
            raise HTTPException(status_code=404, detail="Trade not found")
        
        if trade.status != 'open':
            raise HTTPException(status_code=400, detail="Trade is not open")
        
        trade = close_trade(
            db,
            trade,
            exit_price=close_data.exit_price,
            exit_date=close_data.exit_date,
            notes=close_data.notes
        )
        return {"message": "Trade closed", "trade": trade.to_dict()}


@app.post("/api/trades/{trade_id}/cancel")
async def cancel_trade_endpoint(request: Request, trade_id: int):
    """Cancel a trade"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    current_user = get_current_user(request)
    if not current_user or current_user.get("id", 0) <= 0:
        raise HTTPException(status_code=403, detail="请先登录 / Please login first")
    
    user_id = current_user["id"]
    
    with get_db() as db:
        trade = get_user_trade_by_id(db, user_id, trade_id)
        if not trade:
            raise HTTPException(status_code=404, detail="Trade not found")
        
        if trade.status != 'open':
            raise HTTPException(status_code=400, detail="Trade is not open")
        
        trade = cancel_trade(db, trade)
        return {"message": "Trade cancelled", "trade": trade.to_dict()}


# Performance API endpoints

@app.get("/api/performance")
async def get_performance(
    request: Request,
    portfolio_id: Optional[int] = None,
    period_type: Optional[str] = None,
    limit: int = 30
):
    """Get performance history"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    current_user = get_current_user(request)
    if not current_user or current_user.get("id", 0) <= 0:
        raise HTTPException(status_code=403, detail="请先登录 / Please login first")
    
    user_id = current_user["id"]
    
    with get_db() as db:
        snapshots = get_performance_history(db, user_id, portfolio_id=portfolio_id, period_type=period_type, limit=limit)
        return {"performance": [s.to_dict() for s in snapshots]}


@app.post("/api/performance/snapshot")
async def create_snapshot(
    request: Request,
    portfolio_id: int,
    period_type: str = "daily"
):
    """Create a performance snapshot"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    current_user = get_current_user(request)
    if not current_user or current_user.get("id", 0) <= 0:
        raise HTTPException(status_code=403, detail="请先登录 / Please login first")
    
    user_id = current_user["id"]
    
    valid_periods = ['daily', 'weekly', 'monthly']
    if period_type not in valid_periods:
        raise HTTPException(status_code=400, detail=f"Invalid period_type. Must be one of: {valid_periods}")
    
    with get_db() as db:
        portfolio = get_user_portfolio_by_id(db, user_id, portfolio_id)
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")
        
        snapshot = create_performance_snapshot(db, user_id, portfolio_id, period_type)
        return {"message": "Snapshot created", "snapshot": snapshot.to_dict()}


@app.get("/api/trades/profitable")
async def get_profitable_trades_endpoint(
    request: Request,
    portfolio_id: Optional[int] = None,
    limit: int = 50
):
    """Get profitable trades (highlighted)"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    current_user = get_current_user(request)
    if not current_user or current_user.get("id", 0) <= 0:
        raise HTTPException(status_code=403, detail="请先登录 / Please login first")
    
    user_id = current_user["id"]
    
    with get_db() as db:
        trades = get_profitable_trades(db, user_id, portfolio_id=portfolio_id, limit=limit)
        return {"trades": [t.to_dict() for t in trades]}


@app.get("/api/trades/by-strategy/{strategy}")
async def get_trades_by_strategy_endpoint(
    request: Request,
    strategy: str,
    portfolio_id: Optional[int] = None,
    limit: int = 100
):
    """Get trades by strategy"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    current_user = get_current_user(request)
    if not current_user or current_user.get("id", 0) <= 0:
        raise HTTPException(status_code=403, detail="请先登录 / Please login first")
    
    user_id = current_user["id"]
    
    with get_db() as db:
        trades = get_trades_by_strategy(db, user_id, strategy, portfolio_id=portfolio_id, limit=limit)
        return {"trades": [t.to_dict() for t in trades], "strategy": strategy}


@app.get("/api/trades/by-ticker/{ticker}")
async def get_trades_by_ticker_endpoint(
    request: Request,
    ticker: str,
    limit: int = 50
):
    """Get trades by ticker"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    current_user = get_current_user(request)
    if not current_user or current_user.get("id", 0) <= 0:
        raise HTTPException(status_code=403, detail="请先登录 / Please login first")
    
    user_id = current_user["id"]
    
    with get_db() as db:
        trades = get_trades_by_ticker(db, user_id, ticker, limit=limit)
        return {"trades": [t.to_dict() for t in trades], "ticker": ticker.upper()}


# Simulator page route

@app.get("/simulator", response_class=HTMLResponse)
async def simulator_page(request: Request):
    """Serve the paper trading simulator page (protected)"""
    token = get_session_token(request)
    if not verify_session(token):
        return RedirectResponse(url="/login", status_code=302)
    
    html_path = os.path.join(os.path.dirname(__file__), "templates", "simulator.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    
    # Return a basic simulator page if template not found
    return HTMLResponse(content=get_simulator_html())


def get_simulator_html():
    """Return basic simulator HTML if template not found"""
    return """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>模拟交易 - TradingAgents</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-900 text-white min-h-screen">
    <div class="container mx-auto px-4 py-8">
        <h1 class="text-4xl font-bold text-center mb-8">📈 模拟交易</h1>
        <p class="text-center text-gray-400">请使用完整的模板页面</p>
        <p class="text-center mt-4"><a href="/dashboard" class="text-blue-400 hover:underline">返回首页</a></p>
    </div>
</body>
</html>
"""


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


# ==================== Strategy & Backtest API ====================

try:
    from .backtest_engine import run_strategy_backtest
    from .database import (
        get_strategies, get_strategy_by_id, create_strategy, update_strategy,
        get_user_backtests, get_backtest_by_id, get_user_backtest_by_id,
        save_backtest_result
    )
except ImportError:
    from backtest_engine import run_strategy_backtest
    from database import (
        get_strategies, get_strategy_by_id, create_strategy, update_strategy,
        get_user_backtests, get_backtest_by_id, get_user_backtest_by_id,
        save_backtest_result
    )


class BacktestRequest(BaseModel):
    strategy_id: int
    ticker: str
    start_date: str
    end_date: str
    initial_capital: float = 100000.0
    params: Optional[dict] = None


class StrategyCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None
    strategy_type: str
    category: Optional[str] = None
    default_params: Optional[dict] = None
    strategy_code: Optional[str] = None
    is_public: bool = False


class StrategyUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    strategy_type: Optional[str] = None
    category: Optional[str] = None
    default_params: Optional[dict] = None
    strategy_code: Optional[str] = None
    is_public: Optional[bool] = None
    is_active: Optional[bool] = None


class SignalRequest(BaseModel):
    ticker: str
    strategy_type: str
    params: Optional[dict] = None


class DataFetchRequest(BaseModel):
    ticker: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    interval: str = "1d"


@app.get("/api/strategies")
async def list_strategies(request: Request, category: Optional[str] = None, strategy_type: Optional[str] = None):
    """List all available strategies"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    current_user = get_current_user(request)
    user_id = current_user.get("id") if current_user else None
    
    with get_db() as db:
        strategies = get_strategies(db, user_id=user_id, public_only=False, active_only=True)
        
        # Filter by category and strategy_type if provided
        if category:
            strategies = [s for s in strategies if s.category == category]
        if strategy_type:
            strategies = [s for s in strategies if s.strategy_type == strategy_type]
        
        return {"strategies": [s.to_dict() for s in strategies]}


@app.get("/api/strategies/library")
async def list_strategy_library(request: Request):
    """List strategies with grouped metadata for library view"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")

    current_user = get_current_user(request)
    user_id = current_user.get("id") if current_user else None

    with get_db() as db:
        strategies = get_strategies(db, user_id=user_id, public_only=False, active_only=True)
        payload = [s.to_dict() for s in strategies]

    categories = sorted({s.get("category") for s in payload if s.get("category")})
    types = sorted({s.get("strategy_type") for s in payload if s.get("strategy_type")})
    return {
        "strategies": payload,
        "categories": categories,
        "strategy_types": types,
    }


@app.post("/api/strategies/create")
async def create_strategy_endpoint(request: Request, data: StrategyCreateRequest):
    """Create a custom strategy"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")

    current_user = get_current_user(request)
    if not current_user or current_user.get("id", 0) <= 0:
        raise HTTPException(status_code=403, detail="请先登录 / Please login first")

    payload = data.dict()
    is_valid, message = validate_strategy_payload(payload)
    if not is_valid:
        raise HTTPException(status_code=400, detail=message)

    with get_db() as db:
        strategy = create_strategy(
            db,
            name=data.name,
            description=data.description or "",
            strategy_type=data.strategy_type,
            category=data.category,
            default_params=data.default_params,
            strategy_code=data.strategy_code,
            user_id=current_user["id"],
            is_public=data.is_public
        )
        return {"strategy": strategy.to_dict()}


@app.put("/api/strategies/{strategy_id}/update")
async def update_strategy_endpoint(request: Request, strategy_id: int, data: StrategyUpdateRequest):
    """Update a custom strategy"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")

    current_user = get_current_user(request)
    if not current_user or current_user.get("id", 0) <= 0:
        raise HTTPException(status_code=403, detail="请先登录 / Please login first")

    with get_db() as db:
        strategy = get_strategy_by_id(db, strategy_id)
        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")
        if strategy.user_id != current_user["id"]:
            raise HTTPException(status_code=403, detail="Only owner can update this strategy")

        strategy = update_strategy(
            db,
            strategy,
            name=data.name,
            description=data.description,
            strategy_type=data.strategy_type,
            category=data.category,
            default_params=data.default_params,
            strategy_code=data.strategy_code,
            is_public=data.is_public,
            is_active=data.is_active
        )
        return {"strategy": strategy.to_dict()}


@app.post("/api/strategies/validate")
async def validate_strategy_endpoint(request: Request, data: StrategyCreateRequest):
    """Validate strategy payload"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")

    payload = data.dict()
    is_valid, message = validate_strategy_payload(payload)
    return {"is_valid": is_valid, "message": message}


@app.get("/api/strategies/{strategy_id}")
async def get_strategy_detail(request: Request, strategy_id: int):
    """Get strategy details"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    with get_db() as db:
        strategy = get_strategy_by_id(db, strategy_id)
        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        return {"strategy": strategy.to_dict()}


@app.post("/api/backtest/run")
async def run_backtest_endpoint(request: Request, backtest_request: BacktestRequest, background_tasks: BackgroundTasks):
    """Run a backtest"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    current_user = get_current_user(request)
    if not current_user or current_user.get("id", 0) <= 0:
        raise HTTPException(status_code=403, detail="请先登录 / Please login first")
    
    user_id = current_user["id"]
    
    with get_db() as db:
        strategy = get_strategy_by_id(db, backtest_request.strategy_id)
        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        # Run backtest in background
        import uuid
        task_id = str(uuid.uuid4())[:8]
        
        # Store task info
        app.backtest_tasks[task_id] = {
            "status": "running",
            "user_id": user_id,
            "strategy_id": backtest_request.strategy_id,
            "request": backtest_request.dict()
        }
        
        # Run backtest
        background_tasks.add_task(
            execute_backtest,
            task_id,
            user_id,
            backtest_request.strategy_id,
            strategy.strategy_type,
            backtest_request.ticker,
            backtest_request.start_date,
            backtest_request.end_date,
            backtest_request.initial_capital,
            backtest_request.params or strategy.default_params
        )
        
        return {"task_id": task_id, "message": "Backtest started"}


def execute_backtest(
    task_id: str,
    user_id: int,
    strategy_id: int,
    strategy_type: str,
    ticker: str,
    start_date: str,
    end_date: str,
    initial_capital: float,
    params: dict
):
    """Execute backtest and save results"""
    try:
        # Import params if it's a string
        if isinstance(params, str):
            import json
            params = json.loads(params)
        
        # Run backtest
        results = run_strategy_backtest(
            strategy_type=strategy_type,
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            params=params
        )
        
        if 'error' in results:
            app.backtest_tasks[task_id]["status"] = "failed"
            app.backtest_tasks[task_id]["error"] = results['error']
            return
        
        # Save to database
        with get_db() as db:
            save_backtest_result(
                db,
                user_id=user_id,
                strategy_id=strategy_id,
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                params=params,
                results=results
            )
        
        app.backtest_tasks[task_id]["status"] = "completed"
        app.backtest_tasks[task_id]["results"] = results
        
    except Exception as e:
        app.backtest_tasks[task_id]["status"] = "failed"
        app.backtest_tasks[task_id]["error"] = str(e)


@app.get("/api/backtest/status/{task_id}")
async def get_backtest_status(request: Request, task_id: str):
    """Get backtest task status"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    if task_id not in app.backtest_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return app.backtest_tasks[task_id]


@app.get("/api/backtests")
async def list_backtests(request: Request, strategy_id: Optional[int] = None, limit: int = 50):
    """List user's backtest results"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    current_user = get_current_user(request)
    if not current_user or current_user.get("id", 0) <= 0:
        raise HTTPException(status_code=403, detail="请先登录 / Please login first")
    
    user_id = current_user["id"]
    
    with get_db() as db:
        backtests = get_user_backtests(db, user_id, strategy_id=strategy_id, limit=limit)
        return {"backtests": [b.to_dict() for b in backtests]}


@app.get("/api/backtests/{backtest_id}")
async def get_backtest_detail(request: Request, backtest_id: int):
    """Get backtest result details"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    current_user = get_current_user(request)
    if not current_user or current_user.get("id", 0) <= 0:
        raise HTTPException(status_code=403, detail="请先登录 / Please login first")
    
    user_id = current_user["id"]
    
    with get_db() as db:
        backtest = get_user_backtest_by_id(db, user_id, backtest_id)
        if not backtest:
            raise HTTPException(status_code=404, detail="Backtest not found")
        
        return {"backtest": backtest.to_dict()}


@app.get("/api/reports/backtest/{backtest_id}")
async def get_backtest_report(request: Request, backtest_id: int):
    """Generate a detailed backtest report"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")

    current_user = get_current_user(request)
    if not current_user or current_user.get("id", 0) <= 0:
        raise HTTPException(status_code=403, detail="请先登录 / Please login first")

    user_id = current_user["id"]
    with get_db() as db:
        backtest = get_user_backtest_by_id(db, user_id, backtest_id)
        if not backtest:
            raise HTTPException(status_code=404, detail="Backtest not found")
        report = generate_backtest_report(backtest.to_dict())
        return {"report": report}


@app.get("/api/portfolios/{portfolio_id}/analysis")
async def get_portfolio_analysis(request: Request, portfolio_id: int):
    """Get portfolio analysis and risk metrics"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")

    current_user = get_current_user(request)
    if not current_user or current_user.get("id", 0) <= 0:
        raise HTTPException(status_code=403, detail="请先登录 / Please login first")

    user_id = current_user["id"]
    with get_db() as db:
        portfolio = get_user_portfolio_by_id(db, user_id, portfolio_id)
        if not portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found")
        stats = get_portfolio_stats(db, portfolio_id)
        analysis = analyze_portfolio(stats)
        return {"analysis": analysis}


@app.post("/api/signals/generate")
async def generate_signal_endpoint(request: Request, data: SignalRequest):
    """Generate a trading signal for a ticker"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")

    result = generate_signal(data.ticker, data.strategy_type, data.params or {})
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return {"signal": result}


@app.get("/api/signals")
async def list_signals(request: Request, tickers: Optional[str] = None, strategy_type: str = "momentum"):
    """Generate signals for a list of tickers"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")

    tickers_list = [t.strip() for t in (tickers or "").split(",") if t.strip()]
    if not tickers_list:
        tickers_list = ["AAPL", "MSFT", "TSLA"]

    results = []
    for ticker in tickers_list:
        signal = generate_signal(ticker, strategy_type, {})
        if "error" not in signal:
            results.append(signal)

    return {"signals": results}


@app.get("/api/data/sources")
async def list_data_sources(request: Request):
    """List available data sources"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")

    return {
        "sources": [
            {"id": "yfinance", "name": "Yahoo Finance", "coverage": "US Equities"},
            {"id": "alpha_vantage", "name": "Alpha Vantage", "coverage": "Stocks/FX/Crypto"},
        ]
    }


@app.post("/api/data/fetch")
async def fetch_data_endpoint(request: Request, data: DataFetchRequest):
    """Fetch historical data for a ticker"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")

    import yfinance as yf
    stock = yf.Ticker(data.ticker)
    df = stock.history(start=data.start_date, end=data.end_date, interval=data.interval)
    if df.empty:
        raise HTTPException(status_code=404, detail="No data available for the ticker")

    records = []
    for idx, row in df.iterrows():
        records.append({
            "date": idx.strftime("%Y-%m-%d"),
            "open": float(row.get("Open", 0)),
            "high": float(row.get("High", 0)),
            "low": float(row.get("Low", 0)),
            "close": float(row.get("Close", 0)),
            "volume": float(row.get("Volume", 0)) if "Volume" in row else 0,
        })

    return {"ticker": data.ticker.upper(), "records": records}


@app.get("/strategies", response_class=HTMLResponse)
async def strategies_page(request: Request):
    """Serve the strategies page (protected)"""
    token = get_session_token(request)
    if not verify_session(token):
        return RedirectResponse(url="/login", status_code=302)
    
    html_path = os.path.join(os.path.dirname(__file__), "templates", "strategies.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    
    # Return a basic strategies page if template not found
    return HTMLResponse(content=get_strategies_html())


@app.get("/strategy-library", response_class=HTMLResponse)
async def strategy_library_page(request: Request):
    """Serve the strategy library page (protected)"""
    token = get_session_token(request)
    if not verify_session(token):
        return RedirectResponse(url="/login", status_code=302)

    html_path = os.path.join(os.path.dirname(__file__), "templates", "strategy_library.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content=get_strategies_html())


@app.get("/strategy-editor", response_class=HTMLResponse)
async def strategy_editor_page(request: Request):
    """Serve the strategy editor page (protected)"""
    token = get_session_token(request)
    if not verify_session(token):
        return RedirectResponse(url="/login", status_code=302)

    html_path = os.path.join(os.path.dirname(__file__), "templates", "strategy_editor.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content=get_strategies_html())


@app.get("/reports", response_class=HTMLResponse)
async def reports_page(request: Request):
    """Serve the reports page (protected)"""
    token = get_session_token(request)
    if not verify_session(token):
        return RedirectResponse(url="/login", status_code=302)

    html_path = os.path.join(os.path.dirname(__file__), "templates", "reports.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content=get_strategies_html())


@app.get("/signals", response_class=HTMLResponse)
async def signals_page(request: Request):
    """Serve the signals page (protected)"""
    token = get_session_token(request)
    if not verify_session(token):
        return RedirectResponse(url="/login", status_code=302)

    html_path = os.path.join(os.path.dirname(__file__), "templates", "signals.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content=get_strategies_html())


@app.get("/data-hub", response_class=HTMLResponse)
async def data_hub_page(request: Request):
    """Serve the data hub page (protected)"""
    token = get_session_token(request)
    if not verify_session(token):
        return RedirectResponse(url="/login", status_code=302)

    html_path = os.path.join(os.path.dirname(__file__), "templates", "data_hub.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content=get_strategies_html())


def get_strategies_html():
    """Return basic strategies HTML if template not found"""
    return """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>量化策略 - TradingAgents</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-900 text-white min-h-screen">
    <div class="container mx-auto px-4 py-8">
        <h1 class="text-4xl font-bold text-center mb-8">📊 量化策略</h1>
        <p class="text-center text-gray-400">请使用完整的模板页面</p>
        <p class="text-center mt-4"><a href="/dashboard" class="text-blue-400 hover:underline">返回首页</a></p>
    </div>
</body>
</html>
"""


# ==================== 批量回测 API ====================

from pydantic import BaseModel as PydanticBaseModel


class BatchBacktestRequest(PydanticBaseModel):
    ticker: str
    start_date: str
    end_date: str
    initial_capital: float = 100000
    trading_frequency: str = "daily"  # daily, monthly


@app.post("/api/backtest/batch")
async def start_batch_backtest_endpoint(request: Request, data: BatchBacktestRequest):
    """启动批量回测任务"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")

    current_user = get_current_user(request)
    if not current_user or current_user.get("id", 0) <= 0:
        raise HTTPException(status_code=403, detail="请先登录")

    user_id = current_user["id"]

    from .batch_backtest_service import start_batch_backtest, execute_batch_backtest

    # 创建任务
    job = start_batch_backtest(
        user_id=user_id,
        ticker=data.ticker,
        start_date=data.start_date,
        end_date=data.end_date,
        initial_capital=data.initial_capital,
        trading_frequency=data.trading_frequency
    )

    # 在后台线程执行
    import threading
    thread = threading.Thread(target=execute_batch_backtest, args=(job.id,))
    thread.start()

    return {"job_id": job.id, "message": "批量回测已启动"}


@app.get("/api/backtest/batch/{job_id}")
async def get_batch_backtest_status(request: Request, job_id: int):
    """查询批量回测任务状态和结果"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")

    from .batch_backtest_service import get_batch_backtest_summary
    from .database import get_batch_backtest_job

    with get_db() as db:
        job = get_batch_backtest_job(db, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="任务不存在")

        summary = get_batch_backtest_summary(job)
        return summary


@app.get("/api/backtest/batch/history")
async def get_batch_backtest_history(request: Request):
    """获取批量回测历史"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")

    current_user = get_current_user(request)
    if not current_user or current_user.get("id", 0) <= 0:
        raise HTTPException(status_code=403, detail="请先登录")

    user_id = current_user["id"]

    from .database import get_user_batch_backtest_jobs

    with get_db() as db:
        jobs = get_user_batch_backtest_jobs(db, user_id)
        return {"jobs": [j.to_dict() for j in jobs]}


# ==================== AI模拟交易 API ====================

class SimulationStartRequest(PydanticBaseModel):
    ticker: str
    strategy_id: int
    duration_days: int = 14
    initial_capital: float = 100000
    check_interval: str = "daily"


@app.post("/api/simulation/start")
async def start_simulation_endpoint(request: Request, data: SimulationStartRequest):
    """启动AI模拟交易"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")

    current_user = get_current_user(request)
    if not current_user or current_user.get("id", 0) <= 0:
        raise HTTPException(status_code=403, detail="请先登录")

    user_id = current_user["id"]

    from .ai_simulation_service import start_simulation

    try:
        session = start_simulation(
            user_id=user_id,
            ticker=data.ticker,
            strategy_id=data.strategy_id,
            duration_days=data.duration_days,
            initial_capital=data.initial_capital,
            check_interval=data.check_interval
        )
        return {"session_id": session.id, "message": "模拟交易已启动"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/simulation/{session_id}")
async def get_simulation_status(request: Request, session_id: int):
    """查询模拟交易会话状态"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")

    from .ai_simulation_service import calculate_session_statistics
    from .database import get_ai_simulation_session, get_strategy_by_id

    with get_db() as db:
        session = get_ai_simulation_session(db, session_id)
        if not session:
            raise HTTPException(status_code=404, detail="会话不存在")

        strategy = get_strategy_by_id(db, session.strategy_id)

        stats = calculate_session_statistics(session)
        result = session.to_dict()
        result["strategy_name"] = strategy.name if strategy else None
        result["statistics"] = stats

        return result


@app.post("/api/simulation/{session_id}/check")
async def trigger_simulation_check(request: Request, session_id: int):
    """手动触发信号检查"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")

    from .ai_simulation_service import check_signal_and_trade

    result = check_signal_and_trade(session_id)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.post("/api/simulation/{session_id}/stop")
async def stop_simulation_endpoint(request: Request, session_id: int):
    """停止模拟交易"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")

    from .ai_simulation_service import stop_simulation

    result = stop_simulation(session_id)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.post("/api/simulation/{session_id}/pause")
async def pause_simulation_endpoint(request: Request, session_id: int):
    """暂停模拟交易"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")

    from .ai_simulation_service import pause_simulation

    result = pause_simulation(session_id)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.post("/api/simulation/{session_id}/resume")
async def resume_simulation_endpoint(request: Request, session_id: int):
    """恢复模拟交易"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")

    from .ai_simulation_service import resume_simulation

    result = resume_simulation(session_id)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.post("/api/simulation/{session_id}/qualify")
async def qualify_simulation_endpoint(request: Request, session_id: int):
    """手动将策略加入股票适配库"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")

    from .ai_simulation_service import manual_qualify_for_library

    result = manual_qualify_for_library(session_id)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.get("/api/simulations")
async def list_simulations(request: Request, status: Optional[str] = None):
    """获取所有模拟交易会话"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")

    current_user = get_current_user(request)
    if not current_user or current_user.get("id", 0) <= 0:
        raise HTTPException(status_code=403, detail="请先登录")

    user_id = current_user["id"]

    from .database import get_user_ai_simulation_sessions

    with get_db() as db:
        sessions = get_user_ai_simulation_sessions(db, user_id, status)
        return {"sessions": [s.to_dict() for s in sessions]}


# ==================== 股票策略适配库 API ====================

@app.get("/api/stock-library")
async def get_stock_library_endpoint(request: Request):
    """获取股票策略适配库"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")

    current_user = get_current_user(request)
    if not current_user or current_user.get("id", 0) <= 0:
        raise HTTPException(status_code=403, detail="请先登录")

    user_id = current_user["id"]

    from .stock_library_service import get_stock_library

    stocks = get_stock_library(user_id)
    return {"stocks": stocks}


@app.get("/api/stock-library/{ticker}")
async def get_stock_detail_endpoint(request: Request, ticker: str):
    """获取单只股票详情"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")

    current_user = get_current_user(request)
    if not current_user or current_user.get("id", 0) <= 0:
        raise HTTPException(status_code=403, detail="请先登录")

    user_id = current_user["id"]

    from .stock_library_service import get_stock_detail

    detail = get_stock_detail(user_id, ticker)
    return detail


@app.post("/api/stock-library/{ticker}/analyze")
async def analyze_stock_endpoint(request: Request, ticker: str):
    """分析股票性格"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")

    current_user = get_current_user(request)
    user_id = current_user["id"] if current_user else None

    from .stock_library_service import save_stock_personality

    result = save_stock_personality(ticker, user_id)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.get("/api/stock-library/{ticker}/personality")
async def get_stock_personality_endpoint(request: Request, ticker: str):
    """获取股票性格"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")

    from .database import get_stock_personality

    with get_db() as db:
        personality = get_stock_personality(db, ticker)
        if not personality:
            raise HTTPException(status_code=404, detail="未找到该股票的性格分析")
        return personality.to_dict()


@app.get("/api/stock-library/{ticker}/best-strategies")
async def get_best_strategies_endpoint(request: Request, ticker: str, limit: int = 5):
    """获取最适合该股票的策略"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")

    current_user = get_current_user(request)
    user_id = current_user["id"] if current_user else None

    from .stock_library_service import get_best_strategies_for_stock

    strategies = get_best_strategies_for_stock(ticker, user_id, limit)
    return {"strategies": strategies}


class AddStrategyMatchRequest(PydanticBaseModel):
    strategy_id: int
    backtest_id: Optional[int] = None
    simulation_id: Optional[int] = None
    win_rate: Optional[float] = None
    total_return: Optional[float] = None


@app.post("/api/stock-library/{ticker}/add-strategy")
async def add_strategy_match_endpoint(request: Request, ticker: str, data: AddStrategyMatchRequest):
    """添加策略匹配"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")

    current_user = get_current_user(request)
    if not current_user or current_user.get("id", 0) <= 0:
        raise HTTPException(status_code=403, detail="请先登录")

    user_id = current_user["id"]

    from .stock_library_service import add_strategy_match

    metrics = {}
    if data.win_rate is not None:
        metrics["win_rate"] = data.win_rate
    if data.total_return is not None:
        metrics["total_return"] = data.total_return

    result = add_strategy_match(
        user_id=user_id,
        ticker=ticker,
        strategy_id=data.strategy_id,
        backtest_id=data.backtest_id,
        simulation_id=data.simulation_id,
        metrics=metrics if metrics else None
    )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.delete("/api/stock-library/{ticker}/strategy/{strategy_id}")
async def remove_strategy_match_endpoint(request: Request, ticker: str, strategy_id: int):
    """移除策略匹配"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")

    current_user = get_current_user(request)
    if not current_user or current_user.get("id", 0) <= 0:
        raise HTTPException(status_code=403, detail="请先登录")

    user_id = current_user["id"]

    from .stock_library_service import remove_strategy_match

    result = remove_strategy_match(user_id, ticker, strategy_id)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


# ==================== 数据库备份 API ====================

class BackupCreateRequest(PydanticBaseModel):
    backup_type: str = "full"  # full, strategies_only, user_data, trades_only
    format: str = "json"  # json, sqlite
    destination: str = "local"  # local, github, aliyun_drive
    custom_path: Optional[str] = None
    github_token: Optional[str] = None
    github_repo: Optional[str] = None
    github_path: Optional[str] = None
    aliyun_refresh_token: Optional[str] = None
    aliyun_folder: Optional[str] = None


@app.post("/api/backup/create")
async def create_backup_endpoint(request: Request, data: BackupCreateRequest):
    """创建数据库备份"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")

    current_user = get_current_user(request)
    user_id = current_user["id"] if current_user else None

    from .backup_service import create_backup

    github_config = None
    if data.destination == "github" and data.github_token:
        github_config = {
            "token": data.github_token,
            "repo": data.github_repo,
            "path": data.github_path or "backups/"
        }

    aliyun_config = None
    if data.destination == "aliyun_drive" and data.aliyun_refresh_token:
        aliyun_config = {
            "refresh_token": data.aliyun_refresh_token,
            "folder": data.aliyun_folder or "/TradingAgents/backups/"
        }

    result = create_backup(
        backup_type=data.backup_type,
        format=data.format,
        destination=data.destination,
        user_id=user_id,
        custom_path=data.custom_path,
        github_config=github_config,
        aliyun_config=aliyun_config
    )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.get("/api/backup/{backup_id}")
async def get_backup_status(request: Request, backup_id: int):
    """查询备份状态"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")

    from .database import get_backup_records

    with get_db() as db:
        record = db.query(BackupRecord).filter(BackupRecord.id == backup_id).first()
        if not record:
            raise HTTPException(status_code=404, detail="备份记录不存在")
        return record.to_dict()


@app.get("/api/backup/history")
async def get_backup_history_endpoint(request: Request):
    """获取备份历史"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")

    current_user = get_current_user(request)
    user_id = current_user["id"] if current_user else None

    from .backup_service import get_backup_history

    history = get_backup_history(user_id)
    return {"backups": history}


class RestoreRequest(PydanticBaseModel):
    source: str = "local"  # local, upload
    file_path: Optional[str] = None
    restore_type: str = "full"  # full, strategies_only, merge


@app.post("/api/backup/restore")
async def restore_backup_endpoint(request: Request, data: RestoreRequest):
    """恢复数据库"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")

    current_user = get_current_user(request)
    if not current_user or not current_user.get("is_admin"):
        raise HTTPException(status_code=403, detail="需要管理员权限")

    from .backup_service import restore_from_json, restore_from_sqlite
    import json as json_module

    if data.source == "local" and data.file_path:
        if data.file_path.endswith('.json'):
            with open(data.file_path, 'r', encoding='utf-8') as f:
                json_data = json_module.load(f)
            result = restore_from_json(json_data, data.restore_type)
        elif data.file_path.endswith('.db'):
            result = restore_from_sqlite(data.file_path)
        else:
            raise HTTPException(status_code=400, detail="不支持的文件格式")

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    else:
        raise HTTPException(status_code=400, detail="请提供有效的文件路径")


@app.get("/api/backup/download/{backup_id}")
async def download_backup_endpoint(request: Request, backup_id: int):
    """下载备份文件"""
    token = get_session_token(request)
    if not verify_session(token):
        raise HTTPException(status_code=401, detail="Not authenticated")

    from .backup_service import download_backup
    from fastapi.responses import FileResponse

    result = download_backup(backup_id)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return FileResponse(
        path=result["file_path"],
        filename=result["filename"],
        media_type="application/octet-stream"
    )


# ==================== 股票适配库页面 ====================

@app.get("/stock-library", response_class=HTMLResponse)
async def stock_library_page(request: Request):
    """股票策略适配库页面"""
    token = get_session_token(request)
    if not verify_session(token):
        return RedirectResponse(url="/login", status_code=302)

    html_path = os.path.join(os.path.dirname(__file__), "templates", "stock_library.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content=get_stock_library_html())


def get_stock_library_html():
    """股票适配库基本页面"""
    return """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>股票策略适配库 - TradingAgents</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-900 text-white min-h-screen">
    <div class="container mx-auto px-4 py-8">
        <h1 class="text-4xl font-bold text-center mb-8">📚 股票策略适配库</h1>
        <p class="text-center text-gray-400">页面模板正在加载中...</p>
        <p class="text-center mt-4"><a href="/dashboard" class="text-blue-400 hover:underline">返回首页</a></p>
    </div>
</body>
</html>
"""


# 导入备份记录模型
from .database import BackupRecord


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
