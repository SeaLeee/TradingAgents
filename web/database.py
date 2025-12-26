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


# Initialize database on module load
init_db()
