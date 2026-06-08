"""
Authentication and Security Manager for Cognitive Mesh APIs

Provides JWT-based authentication, role-based access control, rate limiting,
and security auditing for distributed cognitive operations.
"""

import asyncio
import hashlib
import hmac
import jwt
import time
import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import secrets


class UserRole(Enum):
    """User roles for access control"""
    ADMIN = "admin"
    DEVELOPER = "developer"
    ANALYST = "analyst"
    VIEWER = "viewer"
    SERVICE = "service"  # For inter-service communication


class Permission(Enum):
    """System permissions"""
    READ_STATE = "read_state"
    WRITE_STATE = "write_state"
    DELETE_STATE = "delete_state"
    EXECUTE_QUERY = "execute_query"
    EXECUTE_TASK = "execute_task"
    MANAGE_NODES = "manage_nodes"
    VIEW_METRICS = "view_metrics"
    MANAGE_USERS = "manage_users"
    ADMIN_ACCESS = "admin_access"


@dataclass
class User:
    """User account information"""
    user_id: str
    username: str
    email: str
    role: UserRole
    permissions: Set[Permission]
    created_at: datetime
    last_active: Optional[datetime] = None
    is_active: bool = True
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class APIKey:
    """API key for service authentication"""
    key_id: str
    key_hash: str
    user_id: str
    permissions: Set[Permission]
    created_at: datetime
    expires_at: Optional[datetime] = None
    is_active: bool = True
    last_used: Optional[datetime] = None
    usage_count: int = 0


@dataclass
class RateLimitRule:
    """Rate limiting rule"""
    resource: str  # API endpoint or resource type
    max_requests: int
    window_seconds: int
    per_user: bool = True  # If True, applies per user; if False, global


class SecurityAuditEvent:
    """Security audit event"""
    def __init__(self, event_type: str, user_id: str, resource: str, 
                 action: str, result: str, details: Dict[str, Any] = None):
        self.event_id = f"audit_{int(time.time() * 1000)}_{secrets.token_hex(4)}"
        self.event_type = event_type  # auth, access, rate_limit, etc.
        self.user_id = user_id
        self.resource = resource
        self.action = action
        self.result = result  # success, failure, blocked
        self.details = details or {}
        self.timestamp = datetime.now()
        self.ip_address = self.details.get("ip_address", "unknown")


class AuthenticationManager:
    """
    Comprehensive authentication and security manager
    
    Features:
    - JWT token authentication
    - API key management
    - Role-based access control (RBAC)
    - Rate limiting with sliding windows
    - Security audit logging
    - Session management
    """
    
    def __init__(self, jwt_secret: str = None, token_expire_hours: int = 24):
        self.jwt_secret = jwt_secret or secrets.token_urlsafe(32)
        self.token_expire_hours = token_expire_hours
        
        # User and key management
        self.users: Dict[str, User] = {}
        self.api_keys: Dict[str, APIKey] = {}
        self.active_sessions: Dict[str, Dict[str, Any]] = {}  # token -> session_info
        
        # Rate limiting
        self.rate_limit_rules: List[RateLimitRule] = []
        self.rate_limit_counters: Dict[str, deque] = defaultdict(deque)  # key -> timestamps
        
        # Security audit
        self.audit_log: List[SecurityAuditEvent] = []
        self.max_audit_entries = 10000
        
        # Role permissions mapping
        self.role_permissions = {
            UserRole.ADMIN: {
                Permission.READ_STATE, Permission.WRITE_STATE, Permission.DELETE_STATE,
                Permission.EXECUTE_QUERY, Permission.EXECUTE_TASK, Permission.MANAGE_NODES,
                Permission.VIEW_METRICS, Permission.MANAGE_USERS, Permission.ADMIN_ACCESS
            },
            UserRole.DEVELOPER: {
                Permission.READ_STATE, Permission.WRITE_STATE, Permission.DELETE_STATE,
                Permission.EXECUTE_QUERY, Permission.EXECUTE_TASK, Permission.VIEW_METRICS
            },
            UserRole.ANALYST: {
                Permission.READ_STATE, Permission.EXECUTE_QUERY, Permission.VIEW_METRICS
            },
            UserRole.VIEWER: {
                Permission.READ_STATE, Permission.VIEW_METRICS
            },
            UserRole.SERVICE: {
                Permission.READ_STATE, Permission.WRITE_STATE, Permission.EXECUTE_QUERY,
                Permission.EXECUTE_TASK
            }
        }
        
        self.logger = logging.getLogger(__name__)
        self._setup_default_rate_limits()
        self._create_default_users()
    
    def _setup_default_rate_limits(self):
        """Setup default rate limiting rules"""
        self.rate_limit_rules = [
            RateLimitRule("query", 100, 60, True),  # 100 queries per minute per user
            RateLimitRule("state_write", 50, 60, True),  # 50 state writes per minute per user
            RateLimitRule("task_execute", 20, 60, True),  # 20 task executions per minute per user
            RateLimitRule("auth", 10, 60, False),  # 10 auth attempts per minute globally
            RateLimitRule("global", 1000, 60, False)  # 1000 requests per minute globally
        ]
    
    def _create_default_users(self):
        """Create default system users"""
        # Admin user
        admin_user = User(
            user_id="admin_001",
            username="admin",
            email="admin@cognitivemesh.local",
            role=UserRole.ADMIN,
            permissions=self.role_permissions[UserRole.ADMIN],
            created_at=datetime.now()
        )
        self.users[admin_user.user_id] = admin_user
        
        # Service user for inter-service communication
        service_user = User(
            user_id="service_001",
            username="cognitive_service",
            email="service@cognitivemesh.local",
            role=UserRole.SERVICE,
            permissions=self.role_permissions[UserRole.SERVICE],
            created_at=datetime.now()
        )
        self.users[service_user.user_id] = service_user
    
    def create_user(self, username: str, email: str, role: UserRole, 
                   custom_permissions: Set[Permission] = None) -> User:
        """Create a new user"""
        user_id = f"user_{int(time.time())}_{secrets.token_hex(4)}"
        
        # Get permissions from role or use custom
        permissions = custom_permissions or self.role_permissions.get(role, set())
        
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            role=role,
            permissions=permissions,
            created_at=datetime.now()
        )
        
        self.users[user_id] = user
        
        # Audit log
        self._log_audit_event("user_management", user_id, "user", "create", "success")
        
        self.logger.info(f"Created user: {username} ({role.value})")
        return user
    
    def generate_jwt_token(self, user_id: str, additional_claims: Dict[str, Any] = None) -> str:
        """Generate JWT token for user"""
        if user_id not in self.users:
            raise ValueError("User not found")
        
        user = self.users[user_id]
        now = datetime.now()
        
        # Token payload
        payload = {
            "user_id": user_id,
            "username": user.username,
            "role": user.role.value,
            "permissions": [p.value for p in user.permissions],
            "iat": now,
            "exp": now + timedelta(hours=self.token_expire_hours),
            "iss": "cognitive_mesh_api"
        }
        
        # Add additional claims
        if additional_claims:
            payload.update(additional_claims)
        
        # Generate token
        token = jwt.encode(payload, self.jwt_secret, algorithm="HS256")
        
        # Store active session - handle None additional_claims
        additional_claims = additional_claims or {}
        self.active_sessions[token] = {
            "user_id": user_id,
            "created_at": now,
            "last_activity": now,
            "ip_address": additional_claims.get("ip_address", "unknown")
        }
        
        # Update user last active
        user.last_active = now
        
        self._log_audit_event("authentication", user_id, "jwt_token", "generate", "success")
        
        return token
    
    def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            
            # Check if token is in active sessions
            if token not in self.active_sessions:
                self._log_audit_event("authentication", payload.get("user_id", "unknown"), 
                                    "jwt_token", "verify", "failure", {"reason": "session_not_found"})
                return None
            
            # Update session activity
            self.active_sessions[token]["last_activity"] = datetime.now()
            
            # Update user last active
            user_id = payload.get("user_id")
            if user_id in self.users:
                self.users[user_id].last_active = datetime.now()
            
            self._log_audit_event("authentication", user_id, "jwt_token", "verify", "success")
            
            return payload
            
        except jwt.ExpiredSignatureError:
            self._log_audit_event("authentication", "unknown", "jwt_token", "verify", 
                                "failure", {"reason": "token_expired"})
            return None
        except jwt.InvalidTokenError:
            self._log_audit_event("authentication", "unknown", "jwt_token", "verify", 
                                "failure", {"reason": "invalid_token"})
            return None
    
    def revoke_jwt_token(self, token: str):
        """Revoke a JWT token"""
        if token in self.active_sessions:
            user_id = self.active_sessions[token]["user_id"]
            del self.active_sessions[token]
            
            self._log_audit_event("authentication", user_id, "jwt_token", "revoke", "success")
            self.logger.info(f"Revoked JWT token for user {user_id}")
    
    def create_api_key(self, user_id: str, permissions: Set[Permission] = None, 
                      expires_in_days: int = None) -> tuple[str, APIKey]:
        """
        Create API key for user
        Returns (raw_key, api_key_object)
        """
        if user_id not in self.users:
            raise ValueError("User not found")
        
        # Generate key
        raw_key = f"cmk_{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        
        # Use user permissions if not specified
        user = self.users[user_id]
        key_permissions = permissions or user.permissions
        
        # Calculate expiry
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now() + timedelta(days=expires_in_days)
        
        api_key = APIKey(
            key_id=f"key_{int(time.time())}_{secrets.token_hex(4)}",
            key_hash=key_hash,
            user_id=user_id,
            permissions=key_permissions,
            created_at=datetime.now(),
            expires_at=expires_at
        )
        
        self.api_keys[api_key.key_id] = api_key
        
        self._log_audit_event("key_management", user_id, "api_key", "create", "success", 
                            {"key_id": api_key.key_id})
        
        return raw_key, api_key
    
    def verify_api_key(self, raw_key: str) -> Optional[APIKey]:
        """Verify API key"""
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        
        for api_key in self.api_keys.values():
            if api_key.key_hash == key_hash and api_key.is_active:
                # Check expiry
                if api_key.expires_at and datetime.now() > api_key.expires_at:
                    self._log_audit_event("authentication", api_key.user_id, "api_key", 
                                        "verify", "failure", {"reason": "expired"})
                    return None
                
                # Update usage
                api_key.last_used = datetime.now()
                api_key.usage_count += 1
                
                self._log_audit_event("authentication", api_key.user_id, "api_key", 
                                    "verify", "success", {"key_id": api_key.key_id})
                
                return api_key
        
        self._log_audit_event("authentication", "unknown", "api_key", "verify", 
                            "failure", {"reason": "invalid_key"})
        return None
    
    def check_permission(self, user_id: str, permission: Permission) -> bool:
        """Check if user has specific permission"""
        if user_id not in self.users:
            return False
        
        user = self.users[user_id]
        has_permission = permission in user.permissions
        
        self._log_audit_event("access_control", user_id, permission.value, 
                            "check_permission", "success" if has_permission else "denied")
        
        return has_permission
    
    def check_rate_limit(self, resource: str, user_id: str = None, 
                        ip_address: str = None) -> tuple[bool, Dict[str, Any]]:
        """
        Check if request is within rate limits
        Returns (is_allowed, rate_limit_info)
        """
        now = time.time()
        
        # Find applicable rate limit rules
        applicable_rules = [rule for rule in self.rate_limit_rules 
                          if rule.resource == resource or rule.resource == "global"]
        
        if not applicable_rules:
            return True, {"blocked": False}
        
        for rule in applicable_rules:
            # Determine key for rate limiting
            if rule.per_user and user_id:
                rate_key = f"{rule.resource}:{user_id}"
            elif not rule.per_user:
                rate_key = f"{rule.resource}:global"
            elif ip_address:
                rate_key = f"{rule.resource}:{ip_address}"
            else:
                continue  # Skip rule if no applicable key
            
            # Clean old entries
            timestamps = self.rate_limit_counters[rate_key]
            while timestamps and timestamps[0] < (now - rule.window_seconds):
                timestamps.popleft()
            
            # Check limit
            if len(timestamps) >= rule.max_requests:
                self._log_audit_event("rate_limit", user_id or "unknown", resource, 
                                    "check", "blocked", {
                                        "rule": rule.resource,
                                        "current_count": len(timestamps),
                                        "max_requests": rule.max_requests,
                                        "window_seconds": rule.window_seconds
                                    })
                
                return False, {
                    "blocked": True,
                    "rule": rule.resource,
                    "current_count": len(timestamps),
                    "max_requests": rule.max_requests,
                    "window_seconds": rule.window_seconds,
                    "retry_after": int(timestamps[0] + rule.window_seconds - now) if timestamps else rule.window_seconds
                }
            
            # Add current request
            timestamps.append(now)
        
        return True, {
            "blocked": False,
            "remaining_requests": max(rule.max_requests - len(self.rate_limit_counters.get(f"{rule.resource}:global", [])) for rule in applicable_rules) if applicable_rules else float('inf')
        }
    
    def _log_audit_event(self, event_type: str, user_id: str, resource: str, 
                        action: str, result: str, details: Dict[str, Any] = None):
        """Log security audit event"""
        event = SecurityAuditEvent(event_type, user_id, resource, action, result, details)
        self.audit_log.append(event)
        
        # Keep audit log size manageable
        if len(self.audit_log) > self.max_audit_entries:
            self.audit_log = self.audit_log[-self.max_audit_entries//2:]
    
    def get_audit_events(self, event_type: str = None, user_id: str = None, 
                        limit: int = 100) -> List[SecurityAuditEvent]:
        """Get audit events with optional filtering"""
        events = self.audit_log
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if user_id:
            events = [e for e in events if e.user_id == user_id]
        
        return events[-limit:]
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get security and authentication metrics"""
        now = datetime.now()
        
        # Count active sessions
        active_sessions = sum(1 for session in self.active_sessions.values() 
                            if (now - session["last_activity"]).seconds < 3600)  # Active in last hour
        
        # Count recent auth events
        recent_auth = len([e for e in self.audit_log 
                         if e.event_type == "authentication" and 
                            (now - e.timestamp).seconds < 3600])
        
        # Count failed auth attempts
        failed_auth = len([e for e in self.audit_log 
                         if e.event_type == "authentication" and 
                            e.result == "failure" and 
                            (now - e.timestamp).seconds < 3600])
        
        # Count rate limit blocks
        rate_limit_blocks = len([e for e in self.audit_log 
                               if e.event_type == "rate_limit" and 
                                  e.result == "blocked" and 
                                  (now - e.timestamp).seconds < 3600])
        
        return {
            "total_users": len(self.users),
            "active_users": len([u for u in self.users.values() 
                               if u.last_active and (now - u.last_active).seconds < 3600]),
            "total_api_keys": len(self.api_keys),
            "active_api_keys": len([k for k in self.api_keys.values() 
                                  if k.is_active and (k.expires_at is None or k.expires_at > now)]),
            "active_sessions": active_sessions,
            "total_sessions": len(self.active_sessions),
            "recent_auth_attempts": recent_auth,
            "failed_auth_attempts": failed_auth,
            "rate_limit_blocks": rate_limit_blocks,
            "total_audit_events": len(self.audit_log),
            "rate_limit_rules": len(self.rate_limit_rules)
        }
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions and keys"""
        now = datetime.now()
        
        # Remove expired JWT sessions
        expired_tokens = []
        for token, session in self.active_sessions.items():
            if (now - session["last_activity"]).seconds > 3600 * self.token_expire_hours:
                expired_tokens.append(token)
        
        for token in expired_tokens:
            del self.active_sessions[token]
        
        # Deactivate expired API keys
        for api_key in self.api_keys.values():
            if api_key.expires_at and now > api_key.expires_at:
                api_key.is_active = False
        
        if expired_tokens:
            self.logger.info(f"Cleaned up {len(expired_tokens)} expired sessions")
    
    async def start_maintenance_task(self):
        """Start background maintenance task"""
        while True:
            try:
                self.cleanup_expired_sessions()
                await asyncio.sleep(3600)  # Run every hour
            except Exception as e:
                self.logger.error(f"Error in auth maintenance task: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error