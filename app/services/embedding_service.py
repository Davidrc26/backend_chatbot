import google.generativeai as genai
import ollama
from app.core.config import settings
from typing import List


class EmbeddingService:
    """
    Servicio para generar embeddings usando Google Gemini u Ollama
    """
    
    def __init__(self):
        self.provider = settings.EMBEDDING_PROVIDER
        
        # Configurar según el provider
        if self.provider == "gemini":
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            self.model = "models/text-embedding-004"
        elif self.provider == "ollama":
            self.model = settings.OLLAMA_MODEL
            self.ollama_client = ollama.Client(host=settings.OLLAMA_BASE_URL)
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Genera un embedding para un texto
        
        Args:
            text: Texto a vectorizar
            
        Returns:
            Lista de floats representando el vector de embedding
        """
        if self.provider == "gemini":
            return self._generate_gemini_embedding(text, task_type="retrieval_document")
        elif self.provider == "ollama":
            return self._generate_ollama_embedding(text)
        else:
            raise ValueError(f"Provider no soportado: {self.provider}")
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """
        Genera un embedding para una consulta (query)
        
        Args:
            query: Texto de búsqueda
            
        Returns:
            Lista de floats representando el vector de embedding
        """
        if self.provider == "gemini":
            return self._generate_gemini_embedding(query, task_type="retrieval_query")
        elif self.provider == "ollama":
            return self._generate_ollama_embedding(query)
        else:
            raise ValueError(f"Provider no soportado: {self.provider}")
    
    def _generate_gemini_embedding(self, text: str, task_type: str) -> List[float]:
        """Genera embedding usando Gemini"""
        result = genai.embed_content(
            model=self.model,
            content=text,
            task_type=task_type
        )
        return result['embedding']
    
    def _generate_ollama_embedding(self, text: str) -> List[float]:
        """Genera embedding usando Ollama"""
        response = self.ollama_client.embeddings(
            model=self.model,
            prompt=text
        )
        return response['embedding']


# Instancia singleton del servicio
embedding_service = EmbeddingService()
