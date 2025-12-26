"""
GitHub OAuth Authentication Module for TradingAgents
"""

import os
import secrets
import httpx
from datetime import datetime, timedelta
from typing import Optional, Tuple
from dataclasses import dataclass

# GitHub OAuth Configuration
GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID", "")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET", "")
GITHUB_REDIRECT_URI = os.getenv("GITHUB_REDIRECT_URI", "http://localhost:8000/auth/github/callback")

# OAuth URLs
GITHUB_AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_USER_API_URL = "https://api.github.com/user"
GITHUB_USER_EMAILS_URL = "https://api.github.com/user/emails"

# Session configuration
SESSION_COOKIE_NAME = "session_token"
SESSION_EXPIRE_DAYS = 7


@dataclass
class GitHubUser:
    """GitHub user data"""
    id: int
    login: str
    name: Optional[str]
    email: Optional[str]
    avatar_url: Optional[str]
    bio: Optional[str]


def get_github_authorize_url(state: str = None) -> str:
    """
    Generate GitHub OAuth authorization URL
    
    Args:
        state: Optional state parameter for CSRF protection
    
    Returns:
        GitHub authorization URL
    """
    if not GITHUB_CLIENT_ID:
        raise ValueError("GITHUB_CLIENT_ID is not configured")
    
    params = {
        "client_id": GITHUB_CLIENT_ID,
        "redirect_uri": GITHUB_REDIRECT_URI,
        "scope": "read:user user:email",  # Request user info and email
    }
    
    if state:
        params["state"] = state
    
    query_string = "&".join(f"{k}={v}" for k, v in params.items())
    return f"{GITHUB_AUTHORIZE_URL}?{query_string}"


async def exchange_code_for_token(code: str) -> Optional[str]:
    """
    Exchange authorization code for access token
    
    Args:
        code: Authorization code from GitHub callback
    
    Returns:
        Access token or None if failed
    """
    if not GITHUB_CLIENT_ID or not GITHUB_CLIENT_SECRET:
        raise ValueError("GitHub OAuth credentials not configured")
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            GITHUB_TOKEN_URL,
            data={
                "client_id": GITHUB_CLIENT_ID,
                "client_secret": GITHUB_CLIENT_SECRET,
                "code": code,
                "redirect_uri": GITHUB_REDIRECT_URI,
            },
            headers={
                "Accept": "application/json",
            },
            timeout=10.0,
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get("access_token")
        
        return None


async def get_github_user(access_token: str) -> Optional[GitHubUser]:
    """
    Get GitHub user information using access token
    
    Args:
        access_token: GitHub OAuth access token
    
    Returns:
        GitHubUser object or None if failed
    """
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
    }
    
    async with httpx.AsyncClient() as client:
        # Get user info
        response = await client.get(GITHUB_USER_API_URL, headers=headers, timeout=10.0)
        
        if response.status_code != 200:
            return None
        
        user_data = response.json()
        
        # Try to get primary email if not public
        email = user_data.get("email")
        if not email:
            email_response = await client.get(GITHUB_USER_EMAILS_URL, headers=headers, timeout=10.0)
            if email_response.status_code == 200:
                emails = email_response.json()
                # Find primary email
                for e in emails:
                    if e.get("primary"):
                        email = e.get("email")
                        break
                # Fallback to first email
                if not email and emails:
                    email = emails[0].get("email")
        
        return GitHubUser(
            id=user_data.get("id"),
            login=user_data.get("login"),
            name=user_data.get("name"),
            email=email,
            avatar_url=user_data.get("avatar_url"),
            bio=user_data.get("bio"),
        )


def generate_session_token() -> str:
    """Generate a secure session token"""
    return secrets.token_urlsafe(32)


def generate_oauth_state() -> str:
    """Generate a secure state parameter for OAuth"""
    return secrets.token_urlsafe(16)


def get_session_expiry() -> datetime:
    """Get session expiry datetime"""
    return datetime.utcnow() + timedelta(days=SESSION_EXPIRE_DAYS)


def is_github_configured() -> bool:
    """Check if GitHub OAuth is properly configured"""
    return bool(GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET)


# ============== Helper Functions ==============

def create_user_from_github(db, github_user: GitHubUser):
    """
    Create or update user from GitHub data
    
    Args:
        db: Database session
        github_user: GitHubUser object
    
    Returns:
        User object
    """
    try:
        from .database import get_user_by_github_id, create_user, update_user
    except ImportError:
        from database import get_user_by_github_id, create_user, update_user
    
    # Check if user exists
    user = get_user_by_github_id(db, github_user.id)
    
    if user:
        # Update existing user
        user = update_user(
            db, user,
            username=github_user.login,
            email=github_user.email,
            avatar_url=github_user.avatar_url,
            name=github_user.name,
        )
    else:
        # Create new user
        user = create_user(
            db,
            github_id=github_user.id,
            username=github_user.login,
            email=github_user.email,
            avatar_url=github_user.avatar_url,
            name=github_user.name,
        )
    
    return user


def create_user_session(db, user, ip_address: str = None, user_agent: str = None):
    """
    Create a new session for user
    
    Args:
        db: Database session
        user: User object
        ip_address: Client IP address
        user_agent: Client user agent
    
    Returns:
        Tuple of (Session object, session token)
    """
    try:
        from .database import create_session
    except ImportError:
        from database import create_session
    
    token = generate_session_token()
    expires_at = get_session_expiry()
    
    session = create_session(
        db,
        user_id=user.id,
        token=token,
        expires_at=expires_at,
        ip_address=ip_address,
        user_agent=user_agent,
    )
    
    return session, token


def validate_session(db, token: str):
    """
    Validate session token and return user
    
    Args:
        db: Database session
        token: Session token
    
    Returns:
        User object if valid, None if invalid/expired
    """
    try:
        from .database import get_session_by_token, delete_session
    except ImportError:
        from database import get_session_by_token, delete_session
    
    if not token:
        return None
    
    session = get_session_by_token(db, token)
    
    if not session:
        return None
    
    if session.is_expired():
        delete_session(db, token)
        return None
    
    return session.user


def logout_user(db, token: str):
    """
    Logout user by deleting session
    
    Args:
        db: Database session
        token: Session token
    """
    try:
        from .database import delete_session
    except ImportError:
        from database import delete_session
    delete_session(db, token)
