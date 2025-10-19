from fastapi import APIRouter, HTTPException, Query
from app.schemas.chat import ChatRequest, ChatResponse, ChatWithHistoryRequest
from app.services.chat_service import chat_service

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/simple", response_model=ChatResponse)
async def chat_simple(
    chat_request: ChatRequest,
    provider: str = Query(default="llama", description="Provider LLM: 'llama' o 'gemini'")
):
    """
    Chat simple sin contexto de documentos
    
    Args:
        chat_request: Mensaje del usuario
        provider: "llama" o "gemini"
        
    Returns:
        Respuesta de la LLM
    """
    if provider not in ["llama", "gemini"]:
        raise HTTPException(status_code=400, detail="Provider debe ser 'llama' o 'gemini'")
    
    try:
        response = chat_service.get_simple_response(
            message=chat_request.message,
            provider=provider
        )
        
        return ChatResponse(
            response=response,
            success=True
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el chat: {str(e)}")


@router.post("/rag", response_model=ChatResponse)
async def chat_with_rag(
    chat_request: ChatRequest,
    provider: str = Query(default="llama", description="Provider: 'llama' o 'gemini'")
):
    """
    Chat con RAG (Retrieval Augmented Generation) - busca en documentos
    
    Args:
        chat_request: Mensaje del usuario y configuración (incluye use_rerank)
        provider: "llama" o "gemini" (debe coincidir con embeddings de documentos)
        
    Returns:
        Respuesta con contexto de documentos
    """
    if provider not in ["llama", "gemini"]:
        raise HTTPException(status_code=400, detail="Provider debe ser 'llama' o 'gemini'")
    
    try:
        result = chat_service.get_rag_response(
            message=chat_request.message,
            provider=provider,
            n_results=chat_request.n_results,
            use_rerank=chat_request.use_rerank
        )
        
        return ChatResponse(
            response=result["response"],
            success=True,
            sources=result.get("sources"),
            metadatas=result.get("metadatas"),
            found_documents=result.get("found_documents"),
            reranked=result.get("reranked")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el chat RAG: {str(e)}")


@router.post("/conversation", response_model=ChatResponse)
async def chat_with_history(
    chat_request: ChatWithHistoryRequest,
    provider: str = Query(default="llama", description="Provider: 'llama' o 'gemini'")
):
    """
    Chat con historial de conversación (y opcionalmente RAG con reranking)
    
    Args:
        chat_request: Mensaje, historial y configuración (incluye use_rerank)
        provider: "llama" o "gemini"
        
    Returns:
        Respuesta considerando el historial
    """
    if provider not in ["llama", "gemini"]:
        raise HTTPException(status_code=400, detail="Provider debe ser 'llama' o 'gemini'")
    
    try:
        result = chat_service.get_response_with_history(
            message=chat_request.message,
            chat_history=chat_request.chat_history,
            provider=provider,
            use_rag=chat_request.use_rag,
            n_results=chat_request.n_results,
            use_rerank=chat_request.use_rerank
        )
        
        return ChatResponse(
            response=result["response"],
            success=True,
            sources=result.get("sources"),
            metadatas=result.get("metadatas"),
            found_documents=result.get("found_documents"),
            reranked=result.get("reranked")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el chat con historial: {str(e)}")