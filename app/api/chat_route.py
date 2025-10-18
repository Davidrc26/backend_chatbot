from fastapi import APIRouter, HTTPException, Query
from typing import List
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.embedding_service import embedding_service

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/", response_model=ChatResponse)
async def chat_endpoint_with_embeddings(
    chat_request: ChatRequest,
    provider: str = Query(default="llama", description="Provider de embeddings: 'llama' o 'gemini'")
):
    """
    Endpoint para manejar consultas de chat utilizando embeddings
    
    Args:
        chat_request: Objeto ChatRequest con la consulta del usuario
        provider: "llama" o "gemini" - elige qué modelo usar para embeddings
        
    Returns:
        ChatResponse con la respuesta generada
    """
    # Validar provider
    if provider not in ["llama", "gemini"]:
        raise HTTPException(status_code=400, detail="Provider debe ser 'llama' o 'gemini'")
    
    # Generar embedding para la consulta
    query_embedding = embedding_service.generate_query_embedding(
        query=chat_request.query,
        provider=provider
    )
    
    # Aquí se podría agregar lógica adicional para buscar en la base de datos
    # y generar una respuesta basada en los documentos encontrados.
    
    # Respuesta simulada por ahora
    response_text = f"Respuesta generada para la consulta: {chat_request.query}"
    
    return ChatResponse(answer=response_text)


router.post("/", response_model=ChatResponse)
async def chat_endpoint_without_embeddings(
    chat_request: ChatRequest
):
    """
    Endpoint para manejar consultas de chat sin utilizar embeddings
    
    Args:
        chat_request: Objeto ChatRequest con la consulta del usuario
        
    Returns:
        ChatResponse con la respuesta generada
    """
    # Respuesta simulada por ahora
    response_text = f"Respuesta generada para la consulta: {chat_request.query}"
    
    return ChatResponse(answer=response_text)