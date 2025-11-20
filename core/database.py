# core/database.py
import os
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING, DESCENDING
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

@dataclass
class Message:
    id: str
    role: str
    content: str
    timestamp: datetime
    session_id: str
    tool_calls: Optional[List[Dict]] = None
    thinking_content: Optional[str] = None
    metadata: Optional[Dict] = None
    tool_call_id: Optional[str] = None
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert message to OpenAI format"""
        message_data = {
            "role": self.role,
            "content": self.content
        }
        
        if self.tool_calls:
            message_data["tool_calls"] = self.tool_calls
        
        if self.tool_call_id:
            message_data["tool_call_id"] = self.tool_call_id
            
        return message_data

@dataclass
class Session:
    id: str
    user_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    model: str
    provider: str
    settings: Dict[str, Any]
    message_count: int = 0

@dataclass
class UserSettings:
    user_id: str
    theme: str = "default"
    default_model: str = "gpt-4"
    default_provider: str = "openai"
    tool_calling_enabled: bool = True
    multimodal_enabled: bool = True
    data_sharing: bool = False
    custom_mcp_servers: List[Dict] = None
    
    def __post_init__(self):
        if self.custom_mcp_servers is None:
            self.custom_mcp_servers = []

class DatabaseManager:
    """MongoDB-like database manager for persistent storage"""
    
    def __init__(self, connection_string: str = None, db_name: str = "agent_ui"):
        self.client = None
        self.db = None
        self.connection_string = connection_string or os.getenv("MONGODB_URL", "mongodb://localhost:27017")
        self.db_name = db_name or os.getenv("DB_NAME", "agent_ui")
        
    async def connect(self):
        """Connect to database"""
        try:
            self.client = AsyncIOMotorClient(self.connection_string)
            self.db = self.client[self.db_name]
            
            # Create indexes
            await self._create_indexes()
            logger.info(f"Connected to database: {self.db_name}")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from database"""
        if self.client:
            self.client.close()
            logger.info("Database disconnected")
    
    async def _create_indexes(self):
        """Create database indexes for performance"""

        # Messages index
        await self.db.messages.create_index([("session_id", ASCENDING)])
        await self.db.messages.create_index([("user_id", ASCENDING)])
        await self.db.messages.create_index([("timestamp", DESCENDING)])
        
        # Sessions index
        await self.db.sessions.create_index([("user_id", ASCENDING)])
        await self.db.sessions.create_index([("updated_at", DESCENDING)])
        
        # Users index
        await self.db.users.create_index([("user_id", ASCENDING)], unique=True)


    # Message operations
    async def save_message(self, message: Message) -> str:
        """Save message to database"""
        message_dict = asdict(message)
        message_dict["timestamp"] = message.timestamp.isoformat()
        result = await self.db.messages.insert_one(message_dict)
        return str(result.inserted_id)
    
    async def get_messages(self, session_id: str, limit: int = 50) -> List[Message]:
        """Get messages for a session"""

        cursor = self.db.messages.find({"session_id": session_id}).sort("timestamp", ASCENDING).limit(limit)
        messages = []
        async for doc in cursor:
            doc["timestamp"] = datetime.fromisoformat(doc["timestamp"])
            messages.append(Message(**doc))
        return messages
    
    async def delete_session_messages(self, session_id: str):
        """Delete all messages for a session"""
        await self.db.messages.delete_many({"session_id": session_id})

    # Session operations
    async def save_session(self, session: Session) -> str:
        """Save session to database"""
        session_dict = asdict(session)
        session_dict["created_at"] = session.created_at.isoformat()
        session_dict["updated_at"] = session.updated_at.isoformat()
        result = await self.db.sessions.insert_one(session_dict)
        return str(result.inserted_id)
    
    async def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID"""
        doc = await self.db.sessions.find_one({"id": session_id})
        if doc:
            doc["created_at"] = datetime.fromisoformat(doc["created_at"])
            doc["updated_at"] = datetime.fromisoformat(doc["updated_at"])
            return Session(**doc)
        return None
    
    async def get_user_sessions(self, user_id: str, limit: int = 20) -> List[Session]:
        """Get all sessions for a user"""

        cursor = self.db.sessions.find({"user_id": user_id}).sort("updated_at", DESCENDING).limit(limit)
        sessions = []
        async for doc in cursor:
            doc["created_at"] = datetime.fromisoformat(doc["created_at"])
            doc["updated_at"] = datetime.fromisoformat(doc["updated_at"])
            sessions.append(Session(**doc))
        return sessions
    
    async def delete_session(self, session_id: str):
        """Delete session and all its messages"""
        await self.db.sessions.delete_one({"id": session_id})
        await self.delete_session_messages(session_id)

    # User operations
    async def save_user_settings(self, settings: UserSettings):
        """Save user settings"""
        settings_dict = asdict(settings)
        await self.db.users.update_one(
            {"user_id": settings.user_id},
            {"$set": settings_dict},
            upsert=True
        )
    
    async def get_user_settings(self, user_id: str) -> Optional[UserSettings]:
        """Get user settings"""
        doc = await self.db.users.find_one({"user_id": user_id})
        if doc:
            return UserSettings(**doc)
        return None
    
    async def cleanup_old_messages(self, session_id: str, keep_last: int):
        """Clean up old messages while keeping recent ones"""

        cursor = self.db.messages.find({"session_id": session_id}).sort("timestamp", ASCENDING)
        messages = []
        async for doc in cursor:
            messages.append(doc)
        
        if len(messages) > keep_last:
            # Delete older messages
            messages_to_delete = messages[:-keep_last]
            ids_to_delete = [msg["id"] for msg in messages_to_delete]
            await self.db.messages.delete_many({"id": {"$in": ids_to_delete}})
    
    async def update_session_title(self, session_id: str, title: str):
        """Update session title"""
        await self.db.sessions.update_one(
            {"id": session_id},
            {"$set": {"title": title, "updated_at": datetime.now(timezone.utc).isoformat()}}
        )
    
    async def ping(self):
        """Ping the database"""
        try:
            await self.db.admin.command('ping')
            return True
        except:
            return False