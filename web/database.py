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


# Initialize database on module load
init_db()
