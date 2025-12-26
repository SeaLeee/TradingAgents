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


# Initialize database on module load
init_db()
