import google.generativeai as genai
from app.core.config import settings
from typing import List


class EmbeddingService:
    """
    Servicio para generar embeddings usando Google Gemini
    """
    
    def __init__(self):
        # Configurar Gemini
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self.model = "models/text-embedding-004"  # Modelo de embeddings de Google
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Genera un embedding para un texto usando Gemini
        
        Args:
            text: Texto a vectorizar
            
        Returns:
            Lista de floats representando el vector de embedding
        """
        result = genai.embed_content(
            model=self.model,
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """
        Genera un embedding para una consulta (query)
        
        Args:
            query: Texto de búsqueda
            
        Returns:
            Lista de floats representando el vector de embedding
        """
        result = genai.embed_content(
            model=self.model,
            content=query,
            task_type="retrieval_query"
        )
        return result['embedding']


# Instancia singleton del servicio
embedding_service = EmbeddingService()
