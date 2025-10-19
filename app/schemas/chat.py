from pydantic import BaseModel, Field
from typing import Optional, List, Dict


class ChatRequest(BaseModel):
    message: str = Field(..., description="Mensaje del usuario")
    user_id: Optional[str] = Field(None, description="ID del usuario (opcional)")
    use_rag: Optional[bool] = Field(False, description="Usar RAG (búsqueda en documentos)")
    n_results: Optional[int] = Field(3, description="Número de documentos a recuperar si use_rag=True")
    use_rerank: Optional[bool] = Field(True, description="Aplicar reranking para mejorar calidad (recomendado)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "¿Qué dice el documento sobre inteligencia artificial?",
                "user_id": "user123",
                "use_rag": True,
                "n_results": 3,
                "use_rerank": True
            }
        }


class ChatWithHistoryRequest(BaseModel):
    message: str = Field(..., description="Mensaje del usuario")
    chat_history: List[Dict[str, str]] = Field(
        default=[],
        description="Historial de conversación [{'role': 'user'/'assistant', 'content': '...'}]"
    )
    use_rag: Optional[bool] = Field(False, description="Usar RAG (búsqueda en documentos)")
    n_results: Optional[int] = Field(3, description="Número de documentos a recuperar si use_rag=True")
    use_rerank: Optional[bool] = Field(True, description="Aplicar reranking para mejorar calidad")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "¿Puedes explicarlo mejor?",
                "chat_history": [
                    {"role": "user", "content": "¿Qué es Python?"},
                    {"role": "assistant", "content": "Python es un lenguaje de programación..."}
                ],
                "use_rag": True,
                "use_rerank": True
            }
        }


class ChatResponse(BaseModel):
    response: str = Field(..., description="Respuesta del chatbot")
    success: bool = Field(True, description="Estado de la respuesta")
    sources: Optional[List[str]] = Field(None, description="Documentos usados como contexto (si use_rag=True)")
    metadatas: Optional[List[dict]] = Field(None, description="Metadatos de los documentos fuente")
    found_documents: Optional[bool] = Field(None, description="Si se encontraron documentos relevantes")
    reranked: Optional[bool] = Field(None, description="Si se aplicó reranking a los documentos")
    
    class Config:
        json_schema_extra = {
            "example": {
                "response": "Según los documentos, la inteligencia artificial...",
                "success": True,
                "found_documents": True,
                "reranked": True,
                "sources": ["...texto del documento 1...", "...texto del documento 2..."]
            }
        }

