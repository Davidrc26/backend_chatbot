from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
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
async def upload_file(file: UploadFile = File(...)):
    """
    Endpoint para subir un archivo PDF a ChromaDB con embeddings de Gemini
    
    Args:
        file: Archivo PDF a procesar
        
    Returns:
        DocumentResponse con el ID y mensaje de confirmación
    """
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
        
        # Generar embeddings y guardar en ChromaDB
        collection = chroma_service.get_collection()
        
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            # Generar ID único para cada chunk
            chunk_id = f"{uuid.uuid4()}"
            ids.append(chunk_id)
            
            # Generar embedding
            embedding = embedding_service.generate_embedding(chunk)
            embeddings.append(embedding)
            
            documents.append(chunk)
            
            # Metadata
            metadatas.append({
                "filename": file.filename,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "upload_date": datetime.now().isoformat()
            })
        
        # Guardar en ChromaDB
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        return DocumentResponse(
            id=ids[0],  # Retornamos el ID del primer chunk
            message=f"PDF procesado exitosamente. {len(chunks)} chunks creados."
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando el archivo: {str(e)}")


@router.post("/search", response_model=DocumentSearchResponse)
async def search_documents(query: DocumentQuery):
    """
    Endpoint para buscar documentos similares usando embeddings
    
    Args:
        query: Consulta con texto de búsqueda y número de resultados
        
    Returns:
        DocumentSearchResponse con los documentos encontrados
    """
    # TODO: Implementar la lógica de búsqueda
    pass


@router.delete("/delete/{document_id}")
async def delete_document(document_id: str):
    """
    Endpoint para eliminar un documento de ChromaDB
    
    Args:
        document_id: ID del documento a eliminar
        
    Returns:
        Mensaje de confirmación
    """
    # TODO: Implementar la lógica de eliminación
    pass


@router.get("/list")
async def list_documents():
    """
    Endpoint para listar todos los documentos en ChromaDB
    
    Returns:
        Lista de documentos con sus metadatos
    """
    # TODO: Implementar la lógica de listado
    pass


@router.get("/count")
async def count_documents():
    """
    Endpoint para obtener el número total de documentos
    
    Returns:
        Cantidad de documentos en la colección
    """
    # TODO: Implementar la lógica de conteo
    pass
