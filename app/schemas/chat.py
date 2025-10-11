from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    message: str = Field(..., description="Mensaje del usuario")
    user_id: Optional[str] = Field(None, description="ID del usuario (opcional)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Hola, ¿cómo estás?",
                "user_id": "user123"
            }
        }


class ChatResponse(BaseModel):
    response: str = Field(..., description="Respuesta del chatbot")
    success: bool = Field(True, description="Estado de la respuesta")
    
    class Config:
        json_schema_extra = {
            "example": {
                "response": "¡Hola! Estoy bien, gracias por preguntar.",
                "success": True
            }
        }
