from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    PROJECT_NAME: str = "Backend Chatbot"
    API_V1_STR: str = "/api/v1"
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    
    # ChromaDB Configuration
    CHROMA_DB_PATH: str = "./chroma_db"  # Ruta para la persistencia de ChromaDB
    
    # Gemini Configuration
    GOOGLE_API_KEY: str = ""  # API Key de Google Gemini
    
    # Ollama Configuration
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.1:latest"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
