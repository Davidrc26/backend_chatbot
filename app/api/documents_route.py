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
from app.services.llamaIndex import llamaindex_service

router = APIRouter(prefix="/documents", tags=["documents"])



@router.post("/upload-file-own", response_model=DocumentResponse)
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
            custom_metadata["tags"] = tags
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


@router.post("/upload-file-langchain", response_model=DocumentResponse)
async def upload_file_langchain(
    file: UploadFile = File(..., description="Archivo PDF a procesar con LlamaIndex"),
    provider: str = Form(default="llama", description="Provider de embeddings: 'llama' o 'gemini'"),
    buffer_size: int = Form(
        default=1, 
        description="Ventana de comparación semántica (1-3 recomendado)"
    ),
    breakpoint_percentile_threshold: int = Form(
        default=95,
        description="Umbral percentil para chunking semántico (50-95: menor=chunks más pequeños)"
    ),
    author: Optional[str] = Form(None, description="Autor del documento"),
    category: Optional[str] = Form(None, description="Categoría del documento"),
    tags: Optional[str] = Form(None, description="Tags separados por comas (ej: 'python,fastapi,tutorial')"),
    year: Optional[str] = Form(None, description="Año del documento"),
):
    """
    Endpoint para subir un archivo PDF a ChromaDB usando LlamaIndex con Semantic Chunking
    
    Este endpoint utiliza:
    - SimpleDirectoryReader: Extracción simple y robusta de PDFs
    - SemanticSplitterNodeParser: Chunking inteligente basado en similitud semántica (versión estable de LlamaIndex)
    
    SEMANTIC CHUNKING con LlamaIndex:
    Divide el documento basándose en similitud semántica entre oraciones usando embeddings.
    La implementación de LlamaIndex es más estable y madura que la de LangChain.
    
    Args:
        file: Archivo PDF a procesar
        provider: "llama" o "gemini" - elige qué modelo usar para embeddings
        buffer_size: Ventana de comparación semántica
            - 1: Compara oración por oración (más preciso, más chunks)
            - 2-3: Compara múltiples oraciones (más contexto, menos chunks)
        breakpoint_percentile_threshold: Control del tamaño de chunks
            - 90-95: Chunks más grandes y conservadores (mejor para documentos técnicos)
            - 70-85: Tamaño medio (uso general)
            - 50-70: Chunks más pequeños y precisos (mejor para búsquedas específicas)
        author: Autor del documento (opcional)
        category: Categoría del documento (opcional)
        tags: Tags separados por comas (opcional)
        year: Año del documento (opcional)
        
    Returns:
        DocumentResponse con el ID de la colección y detalles del procesamiento
        
    Ejemplo de uso:
        - Para documentos técnicos/legales: breakpoint_percentile_threshold=95, buffer_size=1
        - Para uso general: breakpoint_percentile_threshold=85, buffer_size=2
        - Para búsquedas precisas: breakpoint_percentile_threshold=70, buffer_size=1
    """
    # Validar provider
    if provider not in ["llama", "gemini"]:
        raise HTTPException(
            status_code=400, 
            detail="Provider debe ser 'llama' o 'gemini'"
        )
    
    # Validar buffer_size
    if buffer_size < 1 or buffer_size > 5:
        raise HTTPException(
            status_code=400,
            detail="buffer_size debe estar entre 1 y 5"
        )
    
    # Validar breakpoint_percentile_threshold
    if breakpoint_percentile_threshold < 50 or breakpoint_percentile_threshold > 99:
        raise HTTPException(
            status_code=400,
            detail="breakpoint_percentile_threshold debe estar entre 50 y 99"
        )
    
    # Validar que sea un PDF
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=400, 
            detail="Solo se aceptan archivos PDF"
        )
    
    try:
        # Leer el contenido del archivo
        content = await file.read()
        
        # Preparar metadata personalizada
        custom_metadata = {}
        if author:
            custom_metadata["author"] = author
        if category:
            custom_metadata["category"] = category
        if tags:
            custom_metadata["tags"] = tags
        if year:
            custom_metadata["year"] = year
        
        # Procesar PDF con LlamaIndex Service (Semantic Chunking)
        result = llamaindex_service.process_pdf_and_store(
            file_content=content,
            filename=file.filename,
            provider=provider,
            buffer_size=buffer_size,
            breakpoint_percentile_threshold=breakpoint_percentile_threshold,
            metadata=custom_metadata if custom_metadata else None
        )
        
        # Construir mensaje de respuesta con detalles
        details = []
        details.append(f"📄 {result['original_documents']} documentos extraídos")
        details.append(f"🧩 {result['chunks_created']} chunks semánticos creados")
        details.append(f"🤖 Provider: {provider.upper()}")
        details.append(f"📊 Estrategia: Semantic (LlamaIndex)")
        details.append(f"🎯 Umbral: {breakpoint_percentile_threshold}%")
        details.append(f"🔄 Buffer: {buffer_size}")
        
        if author:
            details.append(f"✍️ Autor: {author}")
        if category:
            details.append(f"📁 Categoría: {category}")
        if year:
            details.append(f"📅 Año: {year}")
        if tags:
            details.append(f"🏷️ Tags: {tags}")
        
        message = " | ".join(details)
        
        return DocumentResponse(
            id=result['collection_name'],
            message=message
        )
        
    except ValueError as e:
        # Errores de validación o configuración
        raise HTTPException(
            status_code=400, 
            detail=f"Error de configuración: {str(e)}"
        )
    except Exception as e:
        # Otros errores (procesamiento, embeddings, etc.)
        raise HTTPException(
            status_code=500, 
            detail=f"Error procesando el archivo: {str(e)}"
        )
    