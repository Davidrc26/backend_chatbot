from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Form
from typing import List, Optional
import uuid
from datetime import datetime
from app.schemas.document import (
    DocumentUpload,
    DocumentResponse,
    DocumentQuery,
    DocumentSearchResponse
)
from app.services.chroma_service import chroma_service
from app.services.pdf_service import pdf_service
from app.services.embedding_service import embedding_service

router = APIRouter(prefix="/documents", tags=["documents"])



@router.post("/upload-file", response_model=DocumentResponse)
async def upload_file(
    file: UploadFile = File(..., description="Archivo PDF a procesar"),
    provider: str = Form(default="llama", description="Provider de embeddings: 'llama' o 'gemini'"),
    author: Optional[str] = Form(None, description="Autor del documento"),
    category: Optional[str] = Form(None, description="Categoría del documento"),
    tags: Optional[str] = Form(None, description="Tags separados por comas (ej: 'python,fastapi,tutorial')"),
    year: Optional[str] = Form(None, description="Año del documento"),
):
    """
    Endpoint para subir un archivo PDF a ChromaDB con embeddings y metadatos personalizados
    
    Args:
        file: Archivo PDF a procesar
        provider: "llama" o "gemini" - elige qué modelo usar para embeddings
        author: Autor del documento (opcional)
        category: Categoría del documento (opcional)
        tags: Tags separados por comas (opcional)
        year: Año del documento (opcional)
        
    Returns:
        DocumentResponse con el ID y mensaje de confirmación
    """
    # Validar provider
    if provider not in ["llama", "gemini"]:
        raise HTTPException(status_code=400, detail="Provider debe ser 'llama' o 'gemini'")
    
    # Validar que sea un PDF
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos PDF")
    
    try:
        # Leer el contenido del archivo
        content = await file.read()
        
        # Extraer texto del PDF
        text = pdf_service.extract_text(content)
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="El PDF no contiene texto extraíble")
        
        # Dividir en chunks
        chunks = pdf_service.split_text_into_chunks(text)
        
        # Obtener la colección según el provider
        collection = chroma_service.get_collection(provider=provider)
        
        # Preparar metadatos personalizados base
        custom_metadata = {
            "filename": file.filename,
            "upload_date": datetime.now().isoformat(),
            "provider": provider,
        }
        
        # Agregar metadatos opcionales solo si se proporcionaron
        if author:
            custom_metadata["author"] = author
        if category:
            custom_metadata["category"] = category
        if tags:
            # Convertir string de tags a lista
            custom_metadata["tags"] = [tag.strip() for tag in tags.split(",")]
        if year:
            custom_metadata["year"] = year

        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            # Generar ID único para cada chunk
            chunk_id = f"{uuid.uuid4()}"
            ids.append(chunk_id)
            
            # Generar embedding con el provider seleccionado
            embedding = embedding_service.generate_embedding(chunk, provider=provider)
            embeddings.append(embedding)
            
            documents.append(chunk)
            
            # Combinar metadata base con metadata personalizada
            chunk_metadata = {
                **custom_metadata,
                "chunk_index": i,
                "total_chunks": len(chunks)
            }
            metadatas.append(chunk_metadata)
        
        # Guardar en ChromaDB
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        # Construir mensaje de respuesta
        metadata_info = []
        if author:
            metadata_info.append(f"Autor: {author}")
        if category:
            metadata_info.append(f"Categoría: {category}")
        if year:
            metadata_info.append(f"Año: {year}")
        metadata_summary = " | ".join(metadata_info) if metadata_info else "Sin metadatos adicionales"
        
        return DocumentResponse(
            id=ids[0],  # Retornamos el ID del primer chunk
            message=f"PDF procesado exitosamente con {provider}. {len(chunks)} chunks creados. {metadata_summary}"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando el archivo: {str(e)}")


@router.post("/search", response_model=DocumentSearchResponse)
async def search_documents(
    query: DocumentQuery,
    provider: str = Query(default="llama", description="Provider de embeddings: 'llama' o 'gemini'")
):
    """
    Endpoint para buscar documentos similares usando embeddings
    
    Args:
        query: Consulta con texto de búsqueda y número de resultados
        provider: "llama" o "gemini" - debe coincidir con el usado al subir documentos
        
    Returns:
        DocumentSearchResponse con los documentos encontrados
    """
    # Validar provider
    if provider not in ["llama", "gemini"]:
        raise HTTPException(status_code=400, detail="Provider debe ser 'llama' o 'gemini'")
    
    try:
        # Obtener la colección según el provider
        collection = chroma_service.get_collection(provider=provider)
        
        # Generar embedding de la query
        query_embedding = embedding_service.generate_query_embedding(
            query.query, 
            provider=provider
        )
        
        # Buscar en ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=query.n_results
        )
        
        return DocumentSearchResponse(
            documents=results['documents'][0] if results['documents'] else [],
            distances=results['distances'][0] if results['distances'] else [],
            metadatas=results['metadatas'][0] if results['metadatas'] else []
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la búsqueda: {str(e)}")


@router.delete("/delete/{document_id}")
async def delete_document(
    document_id: str,
    provider: str = Query(default="llama", description="Provider de embeddings: 'llama' o 'gemini'")
):
    """
    Endpoint para eliminar un documento de ChromaDB
    
    Args:
        document_id: ID del documento a eliminar
        provider: "llama" o "gemini" - colección donde buscar el documento
        
    Returns:
        Mensaje de confirmación
    """
    # Validar provider
    if provider not in ["llama", "gemini"]:
        raise HTTPException(status_code=400, detail="Provider debe ser 'llama' o 'gemini'")
    
    try:
        collection = chroma_service.get_collection(provider=provider)
        collection.delete(ids=[document_id])
        
        return {"message": f"Documento {document_id} eliminado exitosamente de la colección {provider}"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error eliminando documento: {str(e)}")


@router.get("/list")
async def list_documents(
    provider: str = Query(default="llama", description="Provider de embeddings: 'llama' o 'gemini'")
):
    """
    Endpoint para listar todos los documentos en ChromaDB
    
    Args:
        provider: "llama" o "gemini" - colección a listar
    
    Returns:
        Lista de documentos con sus metadatos
    """
    # Validar provider
    if provider not in ["llama", "gemini"]:
        raise HTTPException(status_code=400, detail="Provider debe ser 'llama' o 'gemini'")
    
    try:
        collection = chroma_service.get_collection(provider=provider)
        results = collection.get()
        
        return {
            "provider": provider,
            "count": len(results['ids']),
            "documents": results['ids'],
            "metadatas": results['metadatas']
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listando documentos: {str(e)}")


@router.get("/count")
async def count_documents(
    provider: str = Query(default="llama", description="Provider de embeddings: 'llama' o 'gemini'")
):
    """
    Endpoint para obtener el número total de documentos
    
    Args:
        provider: "llama" o "gemini" - colección a contar
    
    Returns:
        Cantidad de documentos en la colección
    """
    # Validar provider
    if provider not in ["llama", "gemini"]:
        raise HTTPException(status_code=400, detail="Provider debe ser 'llama' o 'gemini'")
    
    try:
        collection = chroma_service.get_collection(provider=provider)
        count = collection.count()
        
        return {
            "provider": provider,
            "count": count
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error contando documentos: {str(e)}")
