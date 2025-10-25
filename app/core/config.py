from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    PROJECT_NAME: str = "Backend Chatbot"
    API_V1_STR: str = "/api/v1"
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000", "http://localhost:5173"]
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    
    # ChromaDB Configuration
    USE_CHROMA_CLOUD: bool = True  # True para usar cloud, False para local
    CHROMA_DB_PATH: str = "./chroma_db"  # Ruta local (solo si USE_CHROMA_CLOUD=False)
    CHROMA_CLOUD_API_KEY: str = ""
    CHROMA_CLOUD_TENANT: str = ""
    CHROMA_CLOUD_DATABASE: str = "inteligentes"
    
    # Gemini Configuration
    GOOGLE_API_KEY: str = ""  # API Key de Google Gemini
    
    # Ollama Configuration
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.1:latest"

    # Telegram Bot Configuration
    TELEGRAM_BOT_TOKEN: str
    TELEGRAM_POLLING: bool = True
    TELEGRAM_DEFAULT_MODE: str = "breve"
    TELEGRAM_REQUEST_TIMEOUT: int = 10
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
