"""
Database module for TradingAgents
Supports both SQLite (local) and PostgreSQL (production)
"""

import os
from datetime import datetime
from typing import Optional
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from contextlib import contextmanager

# Determine database URL
# Railway provides DATABASE_URL for PostgreSQL
# For local development, use SQLite
DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL:
    # Railway PostgreSQL URL starts with postgres://, but SQLAlchemy needs postgresql://
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
else:
    # Local SQLite database
    DB_PATH = os.path.join(os.path.dirname(__file__), "data", "tradingagents.db")
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    DATABASE_URL = f"sqlite:///{DB_PATH}"
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


# ============== Models ==============

class User(Base):
    """User model - stores user information from GitHub OAuth"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    github_id = Column(Integer, unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    email = Column(String(255), nullable=True)
    avatar_url = Column(String(500), nullable=True)
    name = Column(String(200), nullable=True)  # Display name
    bio = Column(Text, nullable=True)
    
    # Status
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")
    analyses = relationship("AnalysisHistory", back_populates="user", cascade="all, delete-orphan")
    journals = relationship("TradingJournal", back_populates="user", cascade="all, delete-orphan")
    portfolios = relationship("SimPortfolio", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User {self.username}>"
    
    def to_dict(self):
        return {
            "id": self.id,
            "github_id": self.github_id,
            "username": self.username,
            "email": self.email,
            "avatar_url": self.avatar_url,
            "name": self.name,
            "is_admin": self.is_admin,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Session(Base):
    """Session model - stores user login sessions"""
    __tablename__ = "sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    token = Column(String(100), unique=True, index=True, nullable=False)
    
    # Session info
    ip_address = Column(String(50), nullable=True)
    user_agent = Column(String(500), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="sessions")
    
    def __repr__(self):
        return f"<Session {self.token[:8]}...>"
    
    def is_expired(self) -> bool:
        return datetime.utcnow() > self.expires_at


class AnalysisHistory(Base):
    """Analysis history - stores user's stock analysis records"""
    __tablename__ = "analysis_history"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Analysis info
    ticker = Column(String(20), nullable=False, index=True)
    analysis_date = Column(String(20), nullable=False)
    llm_provider = Column(String(50), nullable=True)
    
    # Results (stored as JSON string)
    decision = Column(Text, nullable=True)
    decision_cn = Column(Text, nullable=True)  # Chinese translation
    reports = Column(Text, nullable=True)  # JSON string
    reports_cn = Column(Text, nullable=True)  # Chinese translation
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="analyses")
    
    def __repr__(self):
        return f"<AnalysisHistory {self.ticker} by User {self.user_id}>"


class TradingJournal(Base):
    """Trading Journal - stores user's daily trading notes and reflections"""
    __tablename__ = "trading_journals"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Journal content
    title = Column(String(200), nullable=False)
    content = Column(Text, nullable=False)  # Markdown content
    
    # Trading info (optional)
    ticker = Column(String(20), nullable=True, index=True)  # Related stock
    trade_type = Column(String(20), nullable=True)  # buy, sell, hold, watch
    trade_price = Column(String(50), nullable=True)  # Entry/exit price
    
    # Tags and categories
    tags = Column(String(500), nullable=True)  # Comma-separated tags
    mood = Column(String(20), nullable=True)  # bullish, bearish, neutral, cautious
    
    # Journal date (user can backdate entries)
    journal_date = Column(String(20), nullable=False, index=True)
    
    # Status
    is_public = Column(Boolean, default=False)  # Allow sharing in future
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="journals")
    
    def __repr__(self):
        return f"<TradingJournal {self.title} by User {self.user_id}>"
    
    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "title": self.title,
            "content": self.content,
            "ticker": self.ticker,
            "trade_type": self.trade_type,
            "trade_price": self.trade_price,
            "tags": self.tags.split(",") if self.tags else [],
            "mood": self.mood,
            "journal_date": self.journal_date,
            "is_public": self.is_public,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# ============== Simulation Trading Models ==============

class SimPortfolio(Base):
    """Simulation Portfolio - user's virtual trading portfolio"""
    __tablename__ = "sim_portfolios"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    name = Column(String(100), nullable=False)  # Portfolio name
    description = Column(Text, nullable=True)
    
    # Initial capital
    initial_capital = Column(String(50), default="100000")  # Starting amount
    current_cash = Column(String(50), default="100000")  # Available cash
    
    # Status
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="portfolios")
    trades = relationship("SimTrade", back_populates="portfolio", cascade="all, delete-orphan")
    positions = relationship("SimPosition", back_populates="portfolio", cascade="all, delete-orphan")
    daily_snapshots = relationship("SimDailySnapshot", back_populates="portfolio", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<SimPortfolio {self.name} by User {self.user_id}>"
    
    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "description": self.description,
            "initial_capital": float(self.initial_capital),
            "current_cash": float(self.current_cash),
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class SimTrade(Base):
    """Simulation Trade - individual trade records"""
    __tablename__ = "sim_trades"
    
    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("sim_portfolios.id"), nullable=False)
    
    # Trade info
    ticker = Column(String(20), nullable=False, index=True)
    asset_type = Column(String(20), nullable=False)  # stock, option_call, option_put
    action = Column(String(10), nullable=False)  # buy, sell, short, cover
    
    # Quantity and price
    quantity = Column(Integer, nullable=False)
    price = Column(String(50), nullable=False)  # Entry/exit price
    total_value = Column(String(50), nullable=False)  # quantity * price
    
    # Option specific fields
    strike_price = Column(String(50), nullable=True)
    expiry_date = Column(String(20), nullable=True)
    option_premium = Column(String(50), nullable=True)
    
    # Strategy
    strategy = Column(String(100), nullable=True)  # swing, day_trade, covered_call, iron_condor, etc.
    strategy_notes = Column(Text, nullable=True)
    
    # Trade date and status
    trade_date = Column(String(20), nullable=False, index=True)
    trade_time = Column(String(20), nullable=True)
    status = Column(String(20), default="open")  # open, closed, expired
    
    # P&L (filled when trade is closed)
    close_price = Column(String(50), nullable=True)
    close_date = Column(String(20), nullable=True)
    realized_pnl = Column(String(50), nullable=True)  # Profit/Loss
    pnl_percent = Column(String(20), nullable=True)
    
    # Market context at trade time
    market_price_at_trade = Column(String(50), nullable=True)
    market_event = Column(Text, nullable=True)  # News/events at the time
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    portfolio = relationship("SimPortfolio", back_populates="trades")
    
    def __repr__(self):
        return f"<SimTrade {self.action} {self.quantity} {self.ticker}>"
    
    def to_dict(self):
        return {
            "id": self.id,
            "portfolio_id": self.portfolio_id,
            "ticker": self.ticker,
            "asset_type": self.asset_type,
            "action": self.action,
            "quantity": self.quantity,
            "price": float(self.price),
            "total_value": float(self.total_value),
            "strike_price": float(self.strike_price) if self.strike_price else None,
            "expiry_date": self.expiry_date,
            "option_premium": float(self.option_premium) if self.option_premium else None,
            "strategy": self.strategy,
            "strategy_notes": self.strategy_notes,
            "trade_date": self.trade_date,
            "trade_time": self.trade_time,
            "status": self.status,
            "close_price": float(self.close_price) if self.close_price else None,
            "close_date": self.close_date,
            "realized_pnl": float(self.realized_pnl) if self.realized_pnl else None,
            "pnl_percent": float(self.pnl_percent) if self.pnl_percent else None,
            "market_price_at_trade": float(self.market_price_at_trade) if self.market_price_at_trade else None,
            "market_event": self.market_event,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class SimPosition(Base):
    """Simulation Position - current holdings"""
    __tablename__ = "sim_positions"
    
    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("sim_portfolios.id"), nullable=False)
    
    # Position info
    ticker = Column(String(20), nullable=False, index=True)
    asset_type = Column(String(20), nullable=False)  # stock, option_call, option_put
    
    # Quantity and cost basis
    quantity = Column(Integer, nullable=False)
    avg_cost = Column(String(50), nullable=False)  # Average cost per share
    total_cost = Column(String(50), nullable=False)  # Total cost basis
    
    # Option specific
    strike_price = Column(String(50), nullable=True)
    expiry_date = Column(String(20), nullable=True)
    
    # Current value (updated periodically)
    current_price = Column(String(50), nullable=True)
    current_value = Column(String(50), nullable=True)
    unrealized_pnl = Column(String(50), nullable=True)
    unrealized_pnl_percent = Column(String(20), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    portfolio = relationship("SimPortfolio", back_populates="positions")
    
    def __repr__(self):
        return f"<SimPosition {self.quantity} {self.ticker}>"
    
    def to_dict(self):
        return {
            "id": self.id,
            "portfolio_id": self.portfolio_id,
            "ticker": self.ticker,
            "asset_type": self.asset_type,
            "quantity": self.quantity,
            "avg_cost": float(self.avg_cost),
            "total_cost": float(self.total_cost),
            "strike_price": float(self.strike_price) if self.strike_price else None,
            "expiry_date": self.expiry_date,
            "current_price": float(self.current_price) if self.current_price else None,
            "current_value": float(self.current_value) if self.current_value else None,
            "unrealized_pnl": float(self.unrealized_pnl) if self.unrealized_pnl else None,
            "unrealized_pnl_percent": float(self.unrealized_pnl_percent) if self.unrealized_pnl_percent else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class SimDailySnapshot(Base):
    """Daily portfolio snapshot for tracking performance over time"""
    __tablename__ = "sim_daily_snapshots"
    
    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("sim_portfolios.id"), nullable=False)
    
    # Snapshot date
    snapshot_date = Column(String(20), nullable=False, index=True)
    
    # Portfolio value
    total_value = Column(String(50), nullable=False)  # Cash + positions
    cash_value = Column(String(50), nullable=False)
    positions_value = Column(String(50), nullable=False)
    
    # Daily P&L
    daily_pnl = Column(String(50), nullable=True)
    daily_pnl_percent = Column(String(20), nullable=True)
    
    # Cumulative P&L from start
    total_pnl = Column(String(50), nullable=True)
    total_pnl_percent = Column(String(20), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    portfolio = relationship("SimPortfolio", back_populates="daily_snapshots")
    
    def __repr__(self):
        return f"<SimDailySnapshot {self.snapshot_date} Portfolio {self.portfolio_id}>"
    
    def to_dict(self):
        return {
            "id": self.id,
            "portfolio_id": self.portfolio_id,
            "snapshot_date": self.snapshot_date,
            "total_value": float(self.total_value),
            "cash_value": float(self.cash_value),
            "positions_value": float(self.positions_value),
            "daily_pnl": float(self.daily_pnl) if self.daily_pnl else 0,
            "daily_pnl_percent": float(self.daily_pnl_percent) if self.daily_pnl_percent else 0,
            "total_pnl": float(self.total_pnl) if self.total_pnl else 0,
            "total_pnl_percent": float(self.total_pnl_percent) if self.total_pnl_percent else 0,
        }


# ============== Database Functions ==============

def init_db():
    """Initialize database - create all tables"""
    Base.metadata.create_all(bind=engine)
    print(f"Database initialized: {DATABASE_URL}")


def drop_db():
    """Drop all tables - USE WITH CAUTION"""
    Base.metadata.drop_all(bind=engine)
    print("All tables dropped")


@contextmanager
def get_db():
    """Get database session context manager"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db_session():
    """Get database session (for FastAPI dependency injection)"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ============== User Operations ==============

def get_user_by_github_id(db, github_id: int) -> Optional[User]:
    """Get user by GitHub ID"""
    return db.query(User).filter(User.github_id == github_id).first()


def get_user_by_username(db, username: str) -> Optional[User]:
    """Get user by username"""
    return db.query(User).filter(User.username == username).first()


def get_user_by_id(db, user_id: int) -> Optional[User]:
    """Get user by ID"""
    return db.query(User).filter(User.id == user_id).first()


def get_or_create_password_user(db, username: str) -> User:
    """Get or create a user for password-based login
    Uses negative github_id to distinguish from GitHub users
    """
    user = get_user_by_username(db, username)
    if user:
        # Update last login
        user.last_login = datetime.utcnow()
        db.commit()
        db.refresh(user)
        return user
    
    # Create new user with negative github_id (to distinguish from real GitHub users)
    # Use hash of username as negative ID
    import hashlib
    hash_val = int(hashlib.md5(username.encode()).hexdigest()[:8], 16)
    negative_github_id = -abs(hash_val)  # Ensure negative
    
    user = User(
        github_id=negative_github_id,
        username=username,
        email=None,
        avatar_url=None,
        name=username
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def create_user(db, github_id: int, username: str, email: str = None, 
                avatar_url: str = None, name: str = None) -> User:
    """Create a new user"""
    user = User(
        github_id=github_id,
        username=username,
        email=email,
        avatar_url=avatar_url,
        name=name
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def update_user(db, user: User, **kwargs) -> User:
    """Update user information"""
    for key, value in kwargs.items():
        if hasattr(user, key):
            setattr(user, key, value)
    user.last_login = datetime.utcnow()
    db.commit()
    db.refresh(user)
    return user


# ============== Session Operations ==============

def create_session(db, user_id: int, token: str, expires_at: datetime,
                   ip_address: str = None, user_agent: str = None) -> Session:
    """Create a new session"""
    session = Session(
        user_id=user_id,
        token=token,
        expires_at=expires_at,
        ip_address=ip_address,
        user_agent=user_agent
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def get_session_by_token(db, token: str) -> Optional[Session]:
    """Get session by token"""
    return db.query(Session).filter(Session.token == token).first()


def delete_session(db, token: str):
    """Delete a session"""
    session = get_session_by_token(db, token)
    if session:
        db.delete(session)
        db.commit()


def cleanup_expired_sessions(db):
    """Remove all expired sessions"""
    db.query(Session).filter(Session.expires_at < datetime.utcnow()).delete()
    db.commit()


# ============== Analysis History Operations ==============

def save_analysis(db, user_id: int, ticker: str, analysis_date: str,
                  llm_provider: str = None, decision: str = None,
                  decision_cn: str = None, reports: str = None,
                  reports_cn: str = None) -> AnalysisHistory:
    """Save analysis to history"""
    analysis = AnalysisHistory(
        user_id=user_id,
        ticker=ticker,
        analysis_date=analysis_date,
        llm_provider=llm_provider,
        decision=decision,
        decision_cn=decision_cn,
        reports=reports,
        reports_cn=reports_cn
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    return analysis


def get_user_analyses(db, user_id: int, limit: int = 50):
    """Get user's analysis history"""
    return db.query(AnalysisHistory)\
        .filter(AnalysisHistory.user_id == user_id)\
        .order_by(AnalysisHistory.created_at.desc())\
        .limit(limit)\
        .all()


def get_analysis_by_id(db, analysis_id: int) -> Optional[AnalysisHistory]:
    """Get analysis by ID"""
    return db.query(AnalysisHistory).filter(AnalysisHistory.id == analysis_id).first()


def get_user_daily_analysis_count(db, user_id: int) -> int:
    """
    Get the number of analyses a user has performed today
    
    Args:
        db: Database session
        user_id: User ID
    
    Returns:
        Number of analyses today
    """
    from sqlalchemy import func
    today = datetime.utcnow().date()
    
    count = db.query(func.count(AnalysisHistory.id))\
        .filter(AnalysisHistory.user_id == user_id)\
        .filter(func.date(AnalysisHistory.created_at) == today)\
        .scalar()
    
    return count or 0


def can_user_analyze(db, user_id: int, daily_limit: int = 1) -> tuple:
    """
    Check if user can perform analysis based on daily limit
    
    Args:
        db: Database session
        user_id: User ID
        daily_limit: Maximum analyses per day (default: 1)
    
    Returns:
        Tuple of (can_analyze: bool, used_count: int, remaining: int)
    """
    used = get_user_daily_analysis_count(db, user_id)
    remaining = max(0, daily_limit - used)
    can_analyze = remaining > 0
    
    return can_analyze, used, remaining


# ============== Trading Journal Operations ==============

def create_journal(db, user_id: int, title: str, content: str, journal_date: str,
                   ticker: str = None, trade_type: str = None, trade_price: str = None,
                   tags: str = None, mood: str = None, is_public: bool = False) -> TradingJournal:
    """Create a new trading journal entry"""
    journal = TradingJournal(
        user_id=user_id,
        title=title,
        content=content,
        journal_date=journal_date,
        ticker=ticker,
        trade_type=trade_type,
        trade_price=trade_price,
        tags=tags,
        mood=mood,
        is_public=is_public
    )
    db.add(journal)
    db.commit()
    db.refresh(journal)
    return journal


def get_user_journals(db, user_id: int, limit: int = 50, offset: int = 0):
    """Get user's journal entries, ordered by journal_date desc"""
    return db.query(TradingJournal)\
        .filter(TradingJournal.user_id == user_id)\
        .order_by(TradingJournal.journal_date.desc(), TradingJournal.created_at.desc())\
        .offset(offset)\
        .limit(limit)\
        .all()


def get_journal_by_id(db, journal_id: int) -> Optional[TradingJournal]:
    """Get journal by ID"""
    return db.query(TradingJournal).filter(TradingJournal.id == journal_id).first()


def get_user_journal_by_id(db, user_id: int, journal_id: int) -> Optional[TradingJournal]:
    """Get journal by ID, ensuring it belongs to the user"""
    return db.query(TradingJournal)\
        .filter(TradingJournal.id == journal_id)\
        .filter(TradingJournal.user_id == user_id)\
        .first()


def update_journal(db, journal: TradingJournal, **kwargs) -> TradingJournal:
    """Update journal entry"""
    for key, value in kwargs.items():
        if hasattr(journal, key) and key not in ['id', 'user_id', 'created_at']:
            setattr(journal, key, value)
    journal.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(journal)
    return journal


def delete_journal(db, journal: TradingJournal):
    """Delete journal entry"""
    db.delete(journal)
    db.commit()


def get_user_journals_by_date(db, user_id: int, journal_date: str):
    """Get user's journals for a specific date"""
    return db.query(TradingJournal)\
        .filter(TradingJournal.user_id == user_id)\
        .filter(TradingJournal.journal_date == journal_date)\
        .order_by(TradingJournal.created_at.desc())\
        .all()


def get_user_journals_by_ticker(db, user_id: int, ticker: str, limit: int = 20):
    """Get user's journals for a specific ticker"""
    return db.query(TradingJournal)\
        .filter(TradingJournal.user_id == user_id)\
        .filter(TradingJournal.ticker == ticker.upper())\
        .order_by(TradingJournal.journal_date.desc())\
        .limit(limit)\
        .all()


def get_user_journal_count(db, user_id: int) -> int:
    """Get total count of user's journals"""
    from sqlalchemy import func
    return db.query(func.count(TradingJournal.id))\
        .filter(TradingJournal.user_id == user_id)\
        .scalar() or 0


# ==================== Simulated Portfolio CRUD ====================

def create_portfolio(
    db,
    user_id: int,
    name: str,
    initial_balance: float = 100000.0,
    description: str = None
) -> SimPortfolio:
    """Create a new simulated portfolio"""
    portfolio = SimPortfolio(
        user_id=user_id,
        name=name,
        description=description,
        initial_capital=str(initial_balance),
        current_cash=str(initial_balance),
        is_active=True
    )
    db.add(portfolio)
    db.commit()
    db.refresh(portfolio)
    return portfolio


def get_user_portfolios(db, user_id: int, active_only: bool = True):
    """Get user's portfolios"""
    query = db.query(SimPortfolio).filter(SimPortfolio.user_id == user_id)
    if active_only:
        query = query.filter(SimPortfolio.is_active == True)
    return query.order_by(SimPortfolio.created_at.desc()).all()


def get_portfolio_by_id(db, portfolio_id: int) -> Optional[SimPortfolio]:
    """Get portfolio by ID"""
    return db.query(SimPortfolio).filter(SimPortfolio.id == portfolio_id).first()


def get_user_portfolio_by_id(db, user_id: int, portfolio_id: int) -> Optional[SimPortfolio]:
    """Get portfolio by ID, ensuring it belongs to the user"""
    return db.query(SimPortfolio)\
        .filter(SimPortfolio.id == portfolio_id)\
        .filter(SimPortfolio.user_id == user_id)\
        .first()


def update_portfolio(db, portfolio: SimPortfolio, **kwargs) -> SimPortfolio:
    """Update portfolio"""
    for key, value in kwargs.items():
        if hasattr(portfolio, key) and key not in ['id', 'user_id', 'created_at']:
            setattr(portfolio, key, value)
    portfolio.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(portfolio)
    return portfolio


def delete_portfolio(db, portfolio: SimPortfolio):
    """Delete portfolio (soft delete by setting is_active=False)"""
    portfolio.is_active = False
    portfolio.updated_at = datetime.utcnow()
    db.commit()


def recalculate_portfolio_balance(db, portfolio: SimPortfolio):
    """Recalculate portfolio balance based on closed trades"""
    # Get all closed trades for this portfolio
    closed_trades = db.query(SimTrade)\
        .filter(SimTrade.portfolio_id == portfolio.id)\
        .filter(SimTrade.status == 'closed')\
        .all()
    
    total_realized_pnl = sum(float(t.realized_pnl or 0) for t in closed_trades)
    
    # Update current cash = initial + realized P&L
    initial = float(portfolio.initial_capital)
    portfolio.current_cash = str(initial + total_realized_pnl)
    portfolio.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(portfolio)
    return portfolio


# ==================== Paper Trade CRUD ====================

def create_paper_trade(
    db,
    user_id: int,
    portfolio_id: int,
    ticker: str,
    asset_type: str,
    strategy: str,
    trade_direction: str,
    entry_price: float,
    quantity: float,
    entry_date: str = None,
    stop_loss: float = None,
    take_profit: float = None,
    notes: str = None,
    market_conditions: str = None,
    news_events: str = None,
    price_at_entry: float = None,
    chart_data: str = None,
    tags: str = None
) -> SimTrade:
    """Create a new paper trade"""
    # Map trade_direction to action
    action = 'buy' if trade_direction == 'long' else 'short'
    total_value = entry_price * quantity
    
    trade = SimTrade(
        portfolio_id=portfolio_id,
        ticker=ticker.upper(),
        asset_type=asset_type,
        action=action,
        quantity=int(quantity),
        price=str(entry_price),
        total_value=str(total_value),
        strategy=strategy,
        strategy_notes=notes,
        trade_date=entry_date or datetime.utcnow().strftime('%Y-%m-%d'),
        status='open',
        market_price_at_trade=str(price_at_entry or entry_price),
        market_event=news_events
    )
    db.add(trade)
    db.commit()
    db.refresh(trade)
    return trade


def get_user_trades(db, user_id: int, portfolio_id: int = None, status: str = None, limit: int = 100, offset: int = 0):
    """Get user's paper trades"""
    # Get all portfolios for this user first
    user_portfolios = db.query(SimPortfolio.id).filter(SimPortfolio.user_id == user_id).all()
    portfolio_ids = [p.id for p in user_portfolios]
    
    if not portfolio_ids:
        return []
    
    query = db.query(SimTrade).filter(SimTrade.portfolio_id.in_(portfolio_ids))
    
    if portfolio_id:
        query = query.filter(SimTrade.portfolio_id == portfolio_id)
    if status:
        query = query.filter(SimTrade.status == status)
    
    return query.order_by(SimTrade.created_at.desc())\
        .offset(offset)\
        .limit(limit)\
        .all()


def get_trade_by_id(db, trade_id: int) -> Optional[SimTrade]:
    """Get trade by ID"""
    return db.query(SimTrade).filter(SimTrade.id == trade_id).first()


def get_user_trade_by_id(db, user_id: int, trade_id: int) -> Optional[SimTrade]:
    """Get trade by ID, ensuring it belongs to the user"""
    trade = db.query(SimTrade).filter(SimTrade.id == trade_id).first()
    if not trade:
        return None
    
    # Verify portfolio belongs to user
    portfolio = get_user_portfolio_by_id(db, user_id, trade.portfolio_id)
    if not portfolio:
        return None
    
    return trade


def close_trade(
    db,
    trade: SimTrade,
    exit_price: float,
    exit_date: str = None,
    notes: str = None
) -> SimTrade:
    """Close a paper trade and calculate P&L"""
    trade.close_price = str(exit_price)
    trade.close_date = exit_date or datetime.utcnow().strftime('%Y-%m-%d')
    trade.status = 'closed'
    
    entry = float(trade.price)
    qty = trade.quantity
    
    # Calculate P&L based on action (buy=long, short=short)
    if trade.action in ['buy', 'cover']:
        pnl = (exit_price - entry) * qty
    else:  # short, sell
        pnl = (entry - exit_price) * qty
    
    trade.realized_pnl = str(pnl)
    
    # Calculate percentage
    pnl_pct = (pnl / (entry * qty)) * 100 if entry * qty > 0 else 0
    trade.pnl_percent = str(round(pnl_pct, 2))
    
    if notes:
        trade.strategy_notes = (trade.strategy_notes or '') + '\n\n--- Exit Notes ---\n' + notes
    
    trade.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(trade)
    
    # Recalculate portfolio balance
    portfolio = get_portfolio_by_id(db, trade.portfolio_id)
    if portfolio:
        recalculate_portfolio_balance(db, portfolio)
    
    return trade


def update_trade(db, trade: SimTrade, **kwargs) -> SimTrade:
    """Update trade"""
    for key, value in kwargs.items():
        if hasattr(trade, key) and key not in ['id', 'portfolio_id', 'created_at']:
            setattr(trade, key, value)
    trade.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(trade)
    return trade


def cancel_trade(db, trade: SimTrade) -> SimTrade:
    """Cancel a trade"""
    trade.status = 'expired'  # Use 'expired' as cancelled equivalent
    trade.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(trade)
    return trade


def get_open_positions(db, user_id: int, portfolio_id: int = None):
    """Get all open positions"""
    return get_user_trades(db, user_id, portfolio_id=portfolio_id, status='open', limit=500)


def get_closed_trades(db, user_id: int, portfolio_id: int = None, limit: int = 100):
    """Get closed trades"""
    return get_user_trades(db, user_id, portfolio_id=portfolio_id, status='closed', limit=limit)


def get_profitable_trades(db, user_id: int, portfolio_id: int = None, limit: int = 50):
    """Get profitable trades"""
    user_portfolios = db.query(SimPortfolio.id).filter(SimPortfolio.user_id == user_id).all()
    portfolio_ids = [p.id for p in user_portfolios]
    
    if not portfolio_ids:
        return []
    
    query = db.query(SimTrade)\
        .filter(SimTrade.portfolio_id.in_(portfolio_ids))\
        .filter(SimTrade.status == 'closed')
    
    if portfolio_id:
        query = query.filter(SimTrade.portfolio_id == portfolio_id)
    
    # Get all closed trades and filter profitable ones
    trades = query.all()
    profitable = [t for t in trades if t.realized_pnl and float(t.realized_pnl) > 0]
    profitable.sort(key=lambda t: float(t.realized_pnl or 0), reverse=True)
    
    return profitable[:limit]


def get_trades_by_strategy(db, user_id: int, strategy: str, portfolio_id: int = None, limit: int = 100):
    """Get trades by strategy"""
    user_portfolios = db.query(SimPortfolio.id).filter(SimPortfolio.user_id == user_id).all()
    portfolio_ids = [p.id for p in user_portfolios]
    
    if not portfolio_ids:
        return []
    
    query = db.query(SimTrade)\
        .filter(SimTrade.portfolio_id.in_(portfolio_ids))\
        .filter(SimTrade.strategy == strategy)
    
    if portfolio_id:
        query = query.filter(SimTrade.portfolio_id == portfolio_id)
    
    return query.order_by(SimTrade.created_at.desc()).limit(limit).all()


def get_trades_by_ticker(db, user_id: int, ticker: str, limit: int = 50):
    """Get trades by ticker"""
    user_portfolios = db.query(SimPortfolio.id).filter(SimPortfolio.user_id == user_id).all()
    portfolio_ids = [p.id for p in user_portfolios]
    
    if not portfolio_ids:
        return []
    
    return db.query(SimTrade)\
        .filter(SimTrade.portfolio_id.in_(portfolio_ids))\
        .filter(SimTrade.ticker == ticker.upper())\
        .order_by(SimTrade.created_at.desc())\
        .limit(limit)\
        .all()


# ==================== Performance Snapshot CRUD ====================

def create_performance_snapshot(
    db,
    user_id: int,
    portfolio_id: int,
    period_type: str,
    snapshot_date: str = None
) -> SimDailySnapshot:
    """Create a performance snapshot for a portfolio"""
    portfolio = get_portfolio_by_id(db, portfolio_id)
    if not portfolio:
        return None
    
    # Get positions value
    positions = db.query(SimPosition)\
        .filter(SimPosition.portfolio_id == portfolio_id)\
        .all()
    
    positions_value = sum(float(p.current_value or 0) for p in positions)
    cash_value = float(portfolio.current_cash)
    total_value = cash_value + positions_value
    
    # Calculate P&L
    initial = float(portfolio.initial_capital)
    total_pnl = total_value - initial
    total_pnl_pct = (total_pnl / initial * 100) if initial > 0 else 0
    
    snapshot = SimDailySnapshot(
        portfolio_id=portfolio_id,
        snapshot_date=snapshot_date or datetime.utcnow().strftime('%Y-%m-%d'),
        total_value=str(total_value),
        cash_value=str(cash_value),
        positions_value=str(positions_value),
        total_pnl=str(total_pnl),
        total_pnl_percent=str(round(total_pnl_pct, 2))
    )
    db.add(snapshot)
    db.commit()
    db.refresh(snapshot)
    return snapshot


def get_performance_history(db, user_id: int, portfolio_id: int = None, period_type: str = None, limit: int = 30):
    """Get performance history snapshots"""
    user_portfolios = db.query(SimPortfolio.id).filter(SimPortfolio.user_id == user_id).all()
    portfolio_ids = [p.id for p in user_portfolios]
    
    if not portfolio_ids:
        return []
    
    query = db.query(SimDailySnapshot).filter(SimDailySnapshot.portfolio_id.in_(portfolio_ids))
    
    if portfolio_id:
        query = query.filter(SimDailySnapshot.portfolio_id == portfolio_id)
    
    return query.order_by(SimDailySnapshot.snapshot_date.desc()).limit(limit).all()


def get_portfolio_stats(db, portfolio_id: int) -> dict:
    """Get comprehensive stats for a portfolio"""
    portfolio = get_portfolio_by_id(db, portfolio_id)
    if not portfolio:
        return {}
    
    all_trades = db.query(SimTrade)\
        .filter(SimTrade.portfolio_id == portfolio_id)\
        .all()
    
    closed_trades = [t for t in all_trades if t.status == 'closed']
    open_trades = [t for t in all_trades if t.status == 'open']
    
    total_trades = len(closed_trades)
    
    # Calculate profitable trades
    profitable_trades = [t for t in closed_trades if t.realized_pnl and float(t.realized_pnl) > 0]
    winning_trades = len(profitable_trades)
    losing_trades = total_trades - winning_trades
    
    pnls = [float(t.realized_pnl or 0) for t in closed_trades]
    total_pnl = sum(pnls)
    
    # Group by strategy
    strategy_stats = {}
    for trade in closed_trades:
        strat = trade.strategy or 'unknown'
        if strat not in strategy_stats:
            strategy_stats[strat] = {'trades': 0, 'wins': 0, 'pnl': 0}
        strategy_stats[strat]['trades'] += 1
        strategy_stats[strat]['pnl'] += float(trade.realized_pnl or 0)
        if trade.realized_pnl and float(trade.realized_pnl) > 0:
            strategy_stats[strat]['wins'] += 1
    
    # Calculate win rate per strategy
    for strat, stats in strategy_stats.items():
        stats['win_rate'] = (stats['wins'] / stats['trades'] * 100) if stats['trades'] > 0 else 0
    
    initial = float(portfolio.initial_capital)
    current = float(portfolio.current_cash)
    
    # Extended to_dict for trades
    def trade_to_dict(t):
        return {
            "id": t.id,
            "ticker": t.ticker,
            "asset_type": t.asset_type,
            "trade_direction": "long" if t.action in ['buy', 'cover'] else "short",
            "strategy": t.strategy,
            "entry_price": float(t.price),
            "exit_price": float(t.close_price) if t.close_price else None,
            "quantity": t.quantity,
            "entry_date": t.trade_date,
            "exit_date": t.close_date,
            "status": t.status,
            "pnl": float(t.realized_pnl) if t.realized_pnl else None,
            "pnl_percentage": float(t.pnl_percent) if t.pnl_percent else None,
            "is_profitable": float(t.realized_pnl or 0) > 0 if t.realized_pnl else None,
            "notes": t.strategy_notes,
            "market_conditions": None,
            "news_events": t.market_event,
        }
    
    return {
        'portfolio': {
            "id": portfolio.id,
            "user_id": portfolio.user_id,
            "name": portfolio.name,
            "description": portfolio.description,
            "initial_balance": initial,
            "current_balance": current,
            "realized_pnl": total_pnl,
            "unrealized_pnl": 0,  # TODO: Calculate from positions
            "is_active": portfolio.is_active,
        },
        'total_trades': total_trades,
        'open_positions': len(open_trades),
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
        'total_pnl': total_pnl,
        'pnl_percentage': (total_pnl / initial * 100) if initial > 0 else 0,
        'best_trade': max(pnls) if pnls else 0,
        'worst_trade': min(pnls) if pnls else 0,
        'average_trade': (total_pnl / total_trades) if total_trades > 0 else 0,
        'strategy_stats': strategy_stats,
        'open_trades': [trade_to_dict(t) for t in open_trades]
    }


# ==================== Strategy & Backtest Models ====================

class Strategy(Base):
    """Trading Strategy - predefined or user-created strategies"""
    __tablename__ = "strategies"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # None for system strategies
    
    # Strategy info
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    strategy_type = Column(String(50), nullable=False)  # momentum, mean_reversion, trend_following, arbitrage, etc.
    category = Column(String(50), nullable=True)  # technical, fundamental, quantitative
    
    # Strategy parameters (JSON string)
    default_params = Column(Text, nullable=True)  # JSON string of default parameters
    
    # Strategy code/logic (optional, for custom strategies)
    strategy_code = Column(Text, nullable=True)
    
    # Performance metrics (from best backtest)
    best_sharpe_ratio = Column(String(20), nullable=True)
    best_total_return = Column(String(20), nullable=True)
    best_max_drawdown = Column(String(20), nullable=True)
    best_win_rate = Column(String(20), nullable=True)
    
    # Status
    is_public = Column(Boolean, default=True)  # System strategies are public
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", foreign_keys=[user_id])
    backtests = relationship("BacktestResult", back_populates="strategy", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Strategy {self.name}>"
    
    def to_dict(self):
        import json
        return {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "description": self.description,
            "strategy_type": self.strategy_type,
            "category": self.category,
            "default_params": json.loads(self.default_params) if self.default_params else {},
            "strategy_code": self.strategy_code,
            "best_sharpe_ratio": float(self.best_sharpe_ratio) if self.best_sharpe_ratio else None,
            "best_total_return": float(self.best_total_return) if self.best_total_return else None,
            "best_max_drawdown": float(self.best_max_drawdown) if self.best_max_drawdown else None,
            "best_win_rate": float(self.best_win_rate) if self.best_win_rate else None,
            "is_public": self.is_public,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class BacktestResult(Base):
    """Backtest Result - stores backtest execution results"""
    __tablename__ = "backtest_results"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=False)
    
    # Backtest configuration
    ticker = Column(String(20), nullable=False, index=True)
    start_date = Column(String(20), nullable=False)
    end_date = Column(String(20), nullable=False)
    initial_capital = Column(String(50), default="100000")
    
    # Strategy parameters used (JSON string)
    params = Column(Text, nullable=True)
    
    # Results
    final_value = Column(String(50), nullable=True)
    total_return = Column(String(20), nullable=True)  # Percentage
    annualized_return = Column(String(20), nullable=True)
    sharpe_ratio = Column(String(20), nullable=True)
    max_drawdown = Column(String(20), nullable=True)
    win_rate = Column(String(20), nullable=True)
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    avg_win = Column(String(20), nullable=True)
    avg_loss = Column(String(20), nullable=True)
    profit_factor = Column(String(20), nullable=True)
    
    # Detailed results (JSON string)
    equity_curve = Column(Text, nullable=True)  # JSON array of daily values
    trade_history = Column(Text, nullable=True)  # JSON array of trades
    daily_returns = Column(Text, nullable=True)  # JSON array of daily returns
    
    # Status
    status = Column(String(20), default="completed")  # pending, running, completed, failed
    error_message = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    
    # Relationships
    user = relationship("User", foreign_keys=[user_id])
    strategy = relationship("Strategy", back_populates="backtests")
    
    def __repr__(self):
        return f"<BacktestResult {self.ticker} Strategy {self.strategy_id}>"
    
    def to_dict(self):
        import json
        return {
            "id": self.id,
            "user_id": self.user_id,
            "strategy_id": self.strategy_id,
            "ticker": self.ticker,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "initial_capital": float(self.initial_capital),
            "params": json.loads(self.params) if self.params else {},
            "final_value": float(self.final_value) if self.final_value else None,
            "total_return": float(self.total_return) if self.total_return else None,
            "annualized_return": float(self.annualized_return) if self.annualized_return else None,
            "sharpe_ratio": float(self.sharpe_ratio) if self.sharpe_ratio else None,
            "max_drawdown": float(self.max_drawdown) if self.max_drawdown else None,
            "win_rate": float(self.win_rate) if self.win_rate else None,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "avg_win": float(self.avg_win) if self.avg_win else None,
            "avg_loss": float(self.avg_loss) if self.avg_loss else None,
            "profit_factor": float(self.profit_factor) if self.profit_factor else None,
            "equity_curve": json.loads(self.equity_curve) if self.equity_curve else [],
            "trade_history": json.loads(self.trade_history) if self.trade_history else [],
            "daily_returns": json.loads(self.daily_returns) if self.daily_returns else [],
            "status": self.status,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


# ==================== Strategy & Backtest Functions ====================

def create_strategy(
    db,
    name: str,
    description: str,
    strategy_type: str,
    category: str = None,
    default_params: dict = None,
    strategy_code: str = None,
    user_id: int = None,
    is_public: bool = True
) -> Strategy:
    """Create a new strategy"""
    import json
    strategy = Strategy(
        user_id=user_id,
        name=name,
        description=description,
        strategy_type=strategy_type,
        category=category,
        default_params=json.dumps(default_params) if default_params else None,
        strategy_code=strategy_code,
        is_public=is_public,
        is_active=True
    )
    db.add(strategy)
    db.commit()
    db.refresh(strategy)
    return strategy


def update_strategy(
    db,
    strategy: Strategy,
    name: Optional[str] = None,
    description: Optional[str] = None,
    strategy_type: Optional[str] = None,
    category: Optional[str] = None,
    default_params: Optional[dict] = None,
    strategy_code: Optional[str] = None,
    is_public: Optional[bool] = None,
    is_active: Optional[bool] = None
) -> Strategy:
    """Update an existing strategy."""
    import json
    if name is not None:
        strategy.name = name
    if description is not None:
        strategy.description = description
    if strategy_type is not None:
        strategy.strategy_type = strategy_type
    if category is not None:
        strategy.category = category
    if default_params is not None:
        strategy.default_params = json.dumps(default_params)
    if strategy_code is not None:
        strategy.strategy_code = strategy_code
    if is_public is not None:
        strategy.is_public = is_public
    if is_active is not None:
        strategy.is_active = is_active

    strategy.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(strategy)
    return strategy


def get_strategies(db, user_id: int = None, public_only: bool = True, active_only: bool = True):
    """Get strategies"""
    query = db.query(Strategy)
    
    if public_only:
        query = query.filter(Strategy.is_public == True)
    elif user_id:
        # Get public strategies OR user's own strategies
        query = query.filter(
            (Strategy.is_public == True) | (Strategy.user_id == user_id)
        )
    
    if active_only:
        query = query.filter(Strategy.is_active == True)
    
    return query.order_by(Strategy.created_at.desc()).all()


def get_strategy_by_id(db, strategy_id: int) -> Optional[Strategy]:
    """Get strategy by ID"""
    return db.query(Strategy).filter(Strategy.id == strategy_id).first()


def get_user_backtests(db, user_id: int, strategy_id: int = None, limit: int = 50):
    """Get user's backtest results"""
    query = db.query(BacktestResult).filter(BacktestResult.user_id == user_id)
    
    if strategy_id:
        query = query.filter(BacktestResult.strategy_id == strategy_id)
    
    return query.order_by(BacktestResult.created_at.desc()).limit(limit).all()


def get_backtest_by_id(db, backtest_id: int) -> Optional[BacktestResult]:
    """Get backtest by ID"""
    return db.query(BacktestResult).filter(BacktestResult.id == backtest_id).first()


def get_user_backtest_by_id(db, user_id: int, backtest_id: int) -> Optional[BacktestResult]:
    """Get backtest by ID, ensuring it belongs to the user"""
    return db.query(BacktestResult)\
        .filter(BacktestResult.id == backtest_id)\
        .filter(BacktestResult.user_id == user_id)\
        .first()


def save_backtest_result(
    db,
    user_id: int,
    strategy_id: int,
    ticker: str,
    start_date: str,
    end_date: str,
    initial_capital: float,
    params: dict,
    results: dict
) -> BacktestResult:
    """Save backtest result to database"""
    import json
    from datetime import datetime
    
    backtest = BacktestResult(
        user_id=user_id,
        strategy_id=strategy_id,
        ticker=ticker.upper(),
        start_date=start_date,
        end_date=end_date,
        initial_capital=str(initial_capital),
        params=json.dumps(params) if params else None,
        final_value=str(results.get('final_value', initial_capital)),
        total_return=str(results.get('total_return', 0)),
        annualized_return=str(results.get('annualized_return', 0)),
        sharpe_ratio=str(results.get('sharpe_ratio', 0)),
        max_drawdown=str(results.get('max_drawdown', 0)),
        win_rate=str(results.get('win_rate', 0)),
        total_trades=results.get('total_trades', 0),
        winning_trades=results.get('winning_trades', 0),
        losing_trades=results.get('losing_trades', 0),
        avg_win=str(results.get('avg_win', 0)),
        avg_loss=str(results.get('avg_loss', 0)),
        profit_factor=str(results.get('profit_factor', 0)),
        equity_curve=json.dumps(results.get('equity_curve', [])),
        trade_history=json.dumps(results.get('trade_history', [])),
        daily_returns=json.dumps(results.get('daily_returns', [])),
        status='completed',
        completed_at=datetime.utcnow()
    )
    db.add(backtest)
    db.commit()
    db.refresh(backtest)
    
    # Update strategy's best metrics if this is better
    strategy = get_strategy_by_id(db, strategy_id)
    if strategy:
        update_strategy_best_metrics(db, strategy, results)
    
    return backtest


def update_strategy_best_metrics(db, strategy: Strategy, results: dict):
    """Update strategy's best metrics if current results are better"""
    should_update = False
    
    sharpe = results.get('sharpe_ratio', 0)
    if sharpe and (not strategy.best_sharpe_ratio or sharpe > float(strategy.best_sharpe_ratio)):
        strategy.best_sharpe_ratio = str(sharpe)
        should_update = True
    
    total_return = results.get('total_return', 0)
    if total_return and (not strategy.best_total_return or total_return > float(strategy.best_total_return)):
        strategy.best_total_return = str(total_return)
        should_update = True
    
    if should_update:
        strategy.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(strategy)


def init_default_strategies(db):
    """Initialize default strategies in database"""
    strategies = [
        {
            "name": "双均线策略 (Moving Average Crossover)",
            "description": "当短期均线上穿长期均线时买入，下穿时卖出。经典的趋势跟踪策略。",
            "strategy_type": "trend_following",
            "category": "technical",
            "default_params": {"short_window": 20, "long_window": 50}
        },
        {
            "name": "RSI 超买超卖策略",
            "description": "当RSI低于30时买入（超卖），高于70时卖出（超买）。适合震荡市场。",
            "strategy_type": "rsi",
            "category": "technical",
            "default_params": {"rsi_period": 14, "oversold": 30, "overbought": 70}
        },
        {
            "name": "MACD 金叉死叉策略",
            "description": "MACD线上穿信号线时买入，下穿时卖出。捕捉趋势变化。",
            "strategy_type": "macd",
            "category": "technical",
            "default_params": {"fast_period": 12, "slow_period": 26, "signal_period": 9}
        },
        {
            "name": "布林带突破策略",
            "description": "价格突破上轨时买入，跌破下轨时卖出。利用波动率进行交易。",
            "strategy_type": "breakout",
            "category": "technical",
            "default_params": {"period": 20, "std_dev": 2}
        },
        {
            "name": "动量策略 (Momentum)",
            "description": "基于价格动量，买入近期表现强势的股票，卖出弱势股票。",
            "strategy_type": "momentum",
            "category": "technical",
            "default_params": {"lookback_period": 20, "momentum_threshold": 0.05}
        },
        {
            "name": "均值回归策略",
            "description": "当价格偏离移动平均线一定幅度时，预期回归均值。",
            "strategy_type": "mean_reversion",
            "category": "technical",
            "default_params": {"ma_period": 20, "deviation_threshold": 0.02}
        },
        {
            "name": "价值投资策略",
            "description": "使用长期均值回归方法识别低估值买入机会，适合价值投资理念。",
            "strategy_type": "value",
            "category": "fundamental",
            "default_params": {"ma_period": 50, "deviation_threshold": 0.03}
        },
        {
            "name": "成长股策略",
            "description": "追踪高动量成长股，捕捉强势上涨趋势。",
            "strategy_type": "growth",
            "category": "fundamental",
            "default_params": {"lookback_period": 20, "momentum_threshold": 0.05}
        },
        {
            "name": "低波动防御策略",
            "description": "偏好波动率较低的标的，使用更温和的均值回归参数。",
            "strategy_type": "custom",
            "category": "risk",
            "default_params": {"base_strategy": "value", "ma_period": 60, "deviation_threshold": 0.015}
        },
        {
            "name": "趋势强化 (长短均线)",
            "description": "延长均线周期以过滤噪音，捕捉更稳定的趋势。",
            "strategy_type": "trend_following",
            "category": "technical",
            "default_params": {"short_window": 30, "long_window": 120}
        },
        {
            "name": "波动率突破加强版",
            "description": "使用更严格的布林带参数，减少震荡期误触发。",
            "strategy_type": "breakout",
            "category": "technical",
            "default_params": {"period": 30, "std_dev": 2.5}
        },
        {
            "name": "动量轮动策略",
            "description": "关注中期动量强度，适合趋势中后段的轮动机会。",
            "strategy_type": "custom",
            "category": "quantitative",
            "default_params": {"base_strategy": "momentum", "lookback_period": 60, "momentum_threshold": 0.08}
        },
        {
            "name": "反转捕捉策略",
            "description": "在动量快速衰减时转向均值回归，偏向反转交易。",
            "strategy_type": "custom",
            "category": "quantitative",
            "default_params": {"base_strategy": "value", "ma_period": 15, "deviation_threshold": 0.025}
        }
    ]
    
    for strat_data in strategies:
        existing = db.query(Strategy).filter(Strategy.name == strat_data["name"]).first()
        if not existing:
            create_strategy(
                db,
                name=strat_data["name"],
                description=strat_data["description"],
                strategy_type=strat_data["strategy_type"],
                category=strat_data["category"],
                default_params=strat_data["default_params"],
                user_id=None,  # System strategy
                is_public=True
            )


# ==================== Batch Backtest & AI Simulation Models ====================

class BatchBacktestJob(Base):
    """批量回测任务 - 对某股票执行所有策略的回测"""
    __tablename__ = "batch_backtest_jobs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # 配置
    ticker = Column(String(20), nullable=False, index=True)
    start_date = Column(String(20), nullable=False)
    end_date = Column(String(20), nullable=False)
    initial_capital = Column(String(50), default="100000")
    trading_frequency = Column(String(20), default="daily")  # daily, monthly

    # 状态
    status = Column(String(20), default="pending")  # pending, running, completed, failed
    progress = Column(Integer, default=0)  # 0-100
    total_strategies = Column(Integer, default=0)
    completed_strategies = Column(Integer, default=0)

    # 最佳策略结果
    best_strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=True)
    best_strategy_name = Column(String(200), nullable=True)
    best_win_rate = Column(String(20), nullable=True)
    best_total_return = Column(String(20), nullable=True)
    best_sharpe_ratio = Column(String(20), nullable=True)

    # 全部结果（JSON数组）
    all_results = Column(Text, nullable=True)

    # 错误信息
    error_message = Column(Text, nullable=True)

    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    user = relationship("User", foreign_keys=[user_id])
    best_strategy = relationship("Strategy", foreign_keys=[best_strategy_id])

    def __repr__(self):
        return f"<BatchBacktestJob {self.ticker} by User {self.user_id}>"

    def to_dict(self):
        import json
        return {
            "id": self.id,
            "user_id": self.user_id,
            "ticker": self.ticker,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "initial_capital": float(self.initial_capital) if self.initial_capital else 100000,
            "trading_frequency": self.trading_frequency,
            "status": self.status,
            "progress": self.progress,
            "total_strategies": self.total_strategies,
            "completed_strategies": self.completed_strategies,
            "best_strategy_id": self.best_strategy_id,
            "best_strategy_name": self.best_strategy_name,
            "best_win_rate": float(self.best_win_rate) if self.best_win_rate else None,
            "best_total_return": float(self.best_total_return) if self.best_total_return else None,
            "best_sharpe_ratio": float(self.best_sharpe_ratio) if self.best_sharpe_ratio else None,
            "all_results": json.loads(self.all_results) if self.all_results else [],
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class AISimulationSession(Base):
    """AI模拟交易会话 - 对回测好的策略进行模拟交易"""
    __tablename__ = "ai_simulation_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # 关联
    ticker = Column(String(20), nullable=False, index=True)
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=False)
    portfolio_id = Column(Integer, ForeignKey("sim_portfolios.id"), nullable=True)

    # 配置
    duration_days = Column(Integer, default=14)  # 模拟时长（默认2周）
    start_date = Column(String(20), nullable=False)
    end_date = Column(String(20), nullable=True)
    initial_capital = Column(String(50), default="100000")
    check_interval = Column(String(20), default="daily")  # daily, hourly

    # 状态
    status = Column(String(20), default="active")  # active, paused, completed, stopped
    current_day = Column(Integer, default=0)

    # 持仓
    current_position = Column(String(20), default="cash")  # cash, long, short
    shares = Column(Integer, default=0)
    entry_price = Column(String(50), nullable=True)
    entry_date = Column(String(20), nullable=True)

    # 统计
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    total_pnl = Column(String(50), default="0")
    current_value = Column(String(50), nullable=True)

    # 交易历史（JSON数组）
    trade_history = Column(Text, nullable=True)
    daily_checks = Column(Text, nullable=True)  # 每日信号检查记录

    # 是否合格加入适配库
    qualified_for_library = Column(Boolean, default=False)
    qualification_date = Column(DateTime, nullable=True)

    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_check_at = Column(DateTime, nullable=True)

    # Relationships
    user = relationship("User", foreign_keys=[user_id])
    strategy = relationship("Strategy", foreign_keys=[strategy_id])
    portfolio = relationship("SimPortfolio", foreign_keys=[portfolio_id])

    def __repr__(self):
        return f"<AISimulationSession {self.ticker} Strategy {self.strategy_id}>"

    def to_dict(self):
        import json
        return {
            "id": self.id,
            "user_id": self.user_id,
            "ticker": self.ticker,
            "strategy_id": self.strategy_id,
            "portfolio_id": self.portfolio_id,
            "duration_days": self.duration_days,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "initial_capital": float(self.initial_capital) if self.initial_capital else 100000,
            "check_interval": self.check_interval,
            "status": self.status,
            "current_day": self.current_day,
            "current_position": self.current_position,
            "shares": self.shares,
            "entry_price": float(self.entry_price) if self.entry_price else None,
            "entry_date": self.entry_date,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "total_pnl": float(self.total_pnl) if self.total_pnl else 0,
            "current_value": float(self.current_value) if self.current_value else None,
            "win_rate": (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0,
            "trade_history": json.loads(self.trade_history) if self.trade_history else [],
            "daily_checks": json.loads(self.daily_checks) if self.daily_checks else [],
            "qualified_for_library": self.qualified_for_library,
            "qualification_date": self.qualification_date.isoformat() if self.qualification_date else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_check_at": self.last_check_at.isoformat() if self.last_check_at else None,
        }


class StockStrategyMatch(Base):
    """股票策略适配 - 记录每只股票适合的策略"""
    __tablename__ = "stock_strategy_matches"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # 股票信息
    ticker = Column(String(20), nullable=False, index=True)
    stock_name = Column(String(200), nullable=True)
    sector = Column(String(100), nullable=True)
    industry = Column(String(100), nullable=True)

    # 策略信息
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=False)
    strategy_name = Column(String(200), nullable=False)
    strategy_type = Column(String(50), nullable=False)

    # 验证数据
    backtest_id = Column(Integer, ForeignKey("backtest_results.id"), nullable=True)
    simulation_id = Column(Integer, ForeignKey("ai_simulation_sessions.id"), nullable=True)

    # 性能指标
    backtest_win_rate = Column(String(20), nullable=True)
    backtest_return = Column(String(20), nullable=True)
    simulation_win_rate = Column(String(20), nullable=True)
    simulation_return = Column(String(20), nullable=True)

    # 置信度和评分
    confidence_score = Column(String(20), nullable=True)  # 0-100
    match_grade = Column(String(5), nullable=True)  # A, B, C, D

    # 状态
    is_active = Column(Boolean, default=True)
    is_recommended = Column(Boolean, default=False)  # 推荐使用

    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_verified_at = Column(DateTime, nullable=True)

    # Relationships
    user = relationship("User", foreign_keys=[user_id])
    strategy = relationship("Strategy", foreign_keys=[strategy_id])
    backtest = relationship("BacktestResult", foreign_keys=[backtest_id])
    simulation = relationship("AISimulationSession", foreign_keys=[simulation_id])

    def __repr__(self):
        return f"<StockStrategyMatch {self.ticker} - {self.strategy_name}>"

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "ticker": self.ticker,
            "stock_name": self.stock_name,
            "sector": self.sector,
            "industry": self.industry,
            "strategy_id": self.strategy_id,
            "strategy_name": self.strategy_name,
            "strategy_type": self.strategy_type,
            "backtest_id": self.backtest_id,
            "simulation_id": self.simulation_id,
            "backtest_win_rate": float(self.backtest_win_rate) if self.backtest_win_rate else None,
            "backtest_return": float(self.backtest_return) if self.backtest_return else None,
            "simulation_win_rate": float(self.simulation_win_rate) if self.simulation_win_rate else None,
            "simulation_return": float(self.simulation_return) if self.simulation_return else None,
            "confidence_score": float(self.confidence_score) if self.confidence_score else None,
            "match_grade": self.match_grade,
            "is_active": self.is_active,
            "is_recommended": self.is_recommended,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_verified_at": self.last_verified_at.isoformat() if self.last_verified_at else None,
        }


class StockPersonality(Base):
    """股票性格分析 - 记录股票的交易特征"""
    __tablename__ = "stock_personalities"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # None表示系统分析

    # 股票信息
    ticker = Column(String(20), nullable=False, index=True)
    stock_name = Column(String(200), nullable=True)

    # 波动性特征
    volatility_level = Column(String(20), nullable=True)  # low, medium, high, extreme
    avg_daily_range = Column(String(20), nullable=True)  # 平均日波动幅度
    beta = Column(String(20), nullable=True)

    # 趋势特征
    trend_tendency = Column(String(20), nullable=True)  # trending, mean_reverting, random
    trend_strength = Column(String(20), nullable=True)  # weak, moderate, strong

    # 动量特征
    momentum_profile = Column(String(20), nullable=True)  # positive, negative, neutral
    momentum_persistence = Column(String(20), nullable=True)  # low, medium, high

    # 推荐策略类型（JSON数组）
    recommended_strategies = Column(Text, nullable=True)

    # 不推荐策略类型（JSON数组）
    avoided_strategies = Column(Text, nullable=True)

    # 分析备注
    analysis_notes = Column(Text, nullable=True)

    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_analyzed_at = Column(DateTime, nullable=True)

    # Relationships
    user = relationship("User", foreign_keys=[user_id])

    def __repr__(self):
        return f"<StockPersonality {self.ticker}>"

    def to_dict(self):
        import json
        return {
            "id": self.id,
            "user_id": self.user_id,
            "ticker": self.ticker,
            "stock_name": self.stock_name,
            "volatility_level": self.volatility_level,
            "avg_daily_range": float(self.avg_daily_range) if self.avg_daily_range else None,
            "beta": float(self.beta) if self.beta else None,
            "trend_tendency": self.trend_tendency,
            "trend_strength": self.trend_strength,
            "momentum_profile": self.momentum_profile,
            "momentum_persistence": self.momentum_persistence,
            "recommended_strategies": json.loads(self.recommended_strategies) if self.recommended_strategies else [],
            "avoided_strategies": json.loads(self.avoided_strategies) if self.avoided_strategies else [],
            "analysis_notes": self.analysis_notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_analyzed_at": self.last_analyzed_at.isoformat() if self.last_analyzed_at else None,
        }


class BackupRecord(Base):
    """备份记录"""
    __tablename__ = "backup_records"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    # 备份信息
    backup_type = Column(String(50), nullable=False)  # full, incremental, strategies_only, etc.
    format = Column(String(20), nullable=False)  # json, sqlite

    # 目标位置
    destination = Column(String(50), nullable=False)  # local, github, aliyun_drive
    file_path = Column(String(500), nullable=True)  # 本地路径或远程路径
    github_repo = Column(String(200), nullable=True)
    github_path = Column(String(200), nullable=True)
    aliyun_folder = Column(String(200), nullable=True)

    # 备份内容摘要
    tables_included = Column(Text, nullable=True)  # JSON数组
    record_counts = Column(Text, nullable=True)  # JSON: {"users": 10, "strategies": 13, ...}
    file_size = Column(String(50), nullable=True)

    # 状态
    status = Column(String(20), default="pending")  # pending, running, completed, failed
    error_message = Column(Text, nullable=True)

    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    user = relationship("User", foreign_keys=[user_id])

    def __repr__(self):
        return f"<BackupRecord {self.backup_type} {self.destination}>"

    def to_dict(self):
        import json
        return {
            "id": self.id,
            "user_id": self.user_id,
            "backup_type": self.backup_type,
            "format": self.format,
            "destination": self.destination,
            "file_path": self.file_path,
            "github_repo": self.github_repo,
            "github_path": self.github_path,
            "aliyun_folder": self.aliyun_folder,
            "tables_included": json.loads(self.tables_included) if self.tables_included else [],
            "record_counts": json.loads(self.record_counts) if self.record_counts else {},
            "file_size": self.file_size,
            "status": self.status,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


# ==================== New Model CRUD Functions ====================

def create_batch_backtest_job(
    db,
    user_id: int,
    ticker: str,
    start_date: str,
    end_date: str,
    initial_capital: float = 100000,
    trading_frequency: str = "daily"
) -> BatchBacktestJob:
    """创建批量回测任务"""
    job = BatchBacktestJob(
        user_id=user_id,
        ticker=ticker.upper(),
        start_date=start_date,
        end_date=end_date,
        initial_capital=str(initial_capital),
        trading_frequency=trading_frequency,
        status="pending"
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def get_batch_backtest_job(db, job_id: int) -> Optional[BatchBacktestJob]:
    """获取批量回测任务"""
    return db.query(BatchBacktestJob).filter(BatchBacktestJob.id == job_id).first()


def get_user_batch_backtest_jobs(db, user_id: int, limit: int = 50):
    """获取用户的批量回测任务列表"""
    return db.query(BatchBacktestJob)\
        .filter(BatchBacktestJob.user_id == user_id)\
        .order_by(BatchBacktestJob.created_at.desc())\
        .limit(limit)\
        .all()


def update_batch_backtest_job(db, job: BatchBacktestJob, **kwargs) -> BatchBacktestJob:
    """更新批量回测任务"""
    for key, value in kwargs.items():
        if hasattr(job, key):
            setattr(job, key, value)
    db.commit()
    db.refresh(job)
    return job


def create_ai_simulation_session(
    db,
    user_id: int,
    ticker: str,
    strategy_id: int,
    duration_days: int = 14,
    initial_capital: float = 100000,
    check_interval: str = "daily"
) -> AISimulationSession:
    """创建AI模拟交易会话"""
    session = AISimulationSession(
        user_id=user_id,
        ticker=ticker.upper(),
        strategy_id=strategy_id,
        duration_days=duration_days,
        start_date=datetime.utcnow().strftime('%Y-%m-%d'),
        initial_capital=str(initial_capital),
        check_interval=check_interval,
        current_value=str(initial_capital),
        status="active"
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def get_ai_simulation_session(db, session_id: int) -> Optional[AISimulationSession]:
    """获取AI模拟交易会话"""
    return db.query(AISimulationSession).filter(AISimulationSession.id == session_id).first()


def get_user_ai_simulation_sessions(db, user_id: int, status: str = None, limit: int = 50):
    """获取用户的AI模拟交易会话列表"""
    query = db.query(AISimulationSession).filter(AISimulationSession.user_id == user_id)
    if status:
        query = query.filter(AISimulationSession.status == status)
    return query.order_by(AISimulationSession.created_at.desc()).limit(limit).all()


def get_active_ai_simulation_sessions(db):
    """获取所有活跃的AI模拟交易会话"""
    return db.query(AISimulationSession).filter(AISimulationSession.status == "active").all()


def update_ai_simulation_session(db, session: AISimulationSession, **kwargs) -> AISimulationSession:
    """更新AI模拟交易会话"""
    for key, value in kwargs.items():
        if hasattr(session, key):
            setattr(session, key, value)
    session.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(session)
    return session


def create_stock_strategy_match(
    db,
    user_id: int,
    ticker: str,
    strategy_id: int,
    strategy_name: str,
    strategy_type: str,
    backtest_id: int = None,
    simulation_id: int = None,
    backtest_win_rate: float = None,
    backtest_return: float = None,
    simulation_win_rate: float = None,
    simulation_return: float = None,
    confidence_score: float = None,
    match_grade: str = None
) -> StockStrategyMatch:
    """创建股票-策略匹配记录"""
    match = StockStrategyMatch(
        user_id=user_id,
        ticker=ticker.upper(),
        strategy_id=strategy_id,
        strategy_name=strategy_name,
        strategy_type=strategy_type,
        backtest_id=backtest_id,
        simulation_id=simulation_id,
        backtest_win_rate=str(backtest_win_rate) if backtest_win_rate else None,
        backtest_return=str(backtest_return) if backtest_return else None,
        simulation_win_rate=str(simulation_win_rate) if simulation_win_rate else None,
        simulation_return=str(simulation_return) if simulation_return else None,
        confidence_score=str(confidence_score) if confidence_score else None,
        match_grade=match_grade
    )
    db.add(match)
    db.commit()
    db.refresh(match)
    return match


def get_stock_strategy_matches(db, user_id: int, ticker: str = None, limit: int = 100):
    """获取用户的股票-策略匹配列表"""
    query = db.query(StockStrategyMatch)\
        .filter(StockStrategyMatch.user_id == user_id)\
        .filter(StockStrategyMatch.is_active == True)
    if ticker:
        query = query.filter(StockStrategyMatch.ticker == ticker.upper())
    return query.order_by(StockStrategyMatch.confidence_score.desc()).limit(limit).all()


def get_stock_personality(db, ticker: str, user_id: int = None) -> Optional[StockPersonality]:
    """获取股票性格"""
    query = db.query(StockPersonality).filter(StockPersonality.ticker == ticker.upper())
    if user_id:
        query = query.filter(StockPersonality.user_id == user_id)
    return query.first()


def create_or_update_stock_personality(
    db,
    ticker: str,
    user_id: int = None,
    **kwargs
) -> StockPersonality:
    """创建或更新股票性格"""
    import json
    personality = get_stock_personality(db, ticker, user_id)

    if personality:
        for key, value in kwargs.items():
            if hasattr(personality, key):
                if isinstance(value, (list, dict)):
                    setattr(personality, key, json.dumps(value))
                else:
                    setattr(personality, key, value)
        personality.updated_at = datetime.utcnow()
        personality.last_analyzed_at = datetime.utcnow()
    else:
        personality = StockPersonality(
            ticker=ticker.upper(),
            user_id=user_id,
            last_analyzed_at=datetime.utcnow()
        )
        for key, value in kwargs.items():
            if hasattr(personality, key):
                if isinstance(value, (list, dict)):
                    setattr(personality, key, json.dumps(value))
                else:
                    setattr(personality, key, value)
        db.add(personality)

    db.commit()
    db.refresh(personality)
    return personality


def create_backup_record(
    db,
    backup_type: str,
    format: str,
    destination: str,
    user_id: int = None,
    file_path: str = None,
    github_repo: str = None,
    github_path: str = None,
    aliyun_folder: str = None
) -> BackupRecord:
    """创建备份记录"""
    record = BackupRecord(
        user_id=user_id,
        backup_type=backup_type,
        format=format,
        destination=destination,
        file_path=file_path,
        github_repo=github_repo,
        github_path=github_path,
        aliyun_folder=aliyun_folder,
        status="pending"
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_backup_records(db, user_id: int = None, limit: int = 50):
    """获取备份记录列表"""
    query = db.query(BackupRecord)
    if user_id:
        query = query.filter(BackupRecord.user_id == user_id)
    return query.order_by(BackupRecord.created_at.desc()).limit(limit).all()


def update_backup_record(db, record: BackupRecord, **kwargs) -> BackupRecord:
    """更新备份记录"""
    for key, value in kwargs.items():
        if hasattr(record, key):
            setattr(record, key, value)
    db.commit()
    db.refresh(record)
    return record


# Initialize database on module load
init_db()

# Initialize default strategies on first run
try:
    with get_db() as db:
        init_default_strategies(db)
except Exception as e:
    print(f"Note: Could not initialize default strategies: {e}")
