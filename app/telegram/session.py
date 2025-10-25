from enum import Enum
from typing import Dict
from datetime import datetime
import uuid

class UserState(Enum):
    NEW = "NEW"
    PENDING_OPTIN = "PENDING_OPTIN"
    ACTIVE = "ACTIVE"
    EXITED = "EXITED"

class UserSession:
    def __init__(self):
        self.state = UserState.NEW
        self.session_id = str(uuid.uuid4())
        self.mode = "breve"                
        self.provider = "llama"          
        self.chat_history = []             
        self.last_sources = []             
        self.last_activity = datetime.now()

# Almacenamiento en memoria
user_sessions: Dict[int, UserSession] = {}
