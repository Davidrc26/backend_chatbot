from pydantic import BaseModel, Field
from typing import Optional, List


class DocumentUpload(BaseModel):
    """Schema para subir un documento"""
    content: str = Field(..., description="Contenido del documento")
    metadata: Optional[dict] = Field(default={}, description="Metadatos adicionales del documento")
    

class DocumentResponse(BaseModel):
    """Schema para la respuesta de un documento subido"""
    id: str = Field(..., description="ID del documento en ChromaDB")
    message: str = Field(..., description="Mensaje de confirmación")
    

class DocumentQuery(BaseModel):
    """Schema para consultar documentos"""
    query: str = Field(..., description="Texto de búsqueda")
    n_results: Optional[int] = Field(default=5, description="Número de resultados a retornar")


class DocumentSearchResponse(BaseModel):
    """Schema para la respuesta de búsqueda"""
    documents: List[str] = Field(..., description="Lista de documentos encontrados")
    distances: List[float] = Field(..., description="Distancias de similitud")
    metadatas: List[dict] = Field(..., description="Metadatos de los documentos")
