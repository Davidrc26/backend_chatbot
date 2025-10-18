import chromadb
from chromadb.config import Settings as ChromaSettings
from app.core.config import settings


class ChromaDBService:
    """
    Servicio para manejar la conexión y operaciones con ChromaDB (local o cloud)
    """
    
    def __init__(self):
        # Decidir si usar ChromaDB Cloud o Local
        if settings.USE_CHROMA_CLOUD:
            # Usar ChromaDB Cloud
            self.client = chromadb.CloudClient(
                tenant=settings.CHROMA_CLOUD_TENANT,
                database=settings.CHROMA_CLOUD_DATABASE,
                api_key=settings.CHROMA_CLOUD_API_KEY
            )
            print("✅ Conectado a ChromaDB Cloud")
        else:
            # Usar ChromaDB Local con persistencia
            self.client = chromadb.PersistentClient(
                path=settings.CHROMA_DB_PATH,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            print("✅ Conectado a ChromaDB Local")
        
        # Diccionario para cachear colecciones
        self.collections = {}
    
    def get_collection(self, provider: str = "llama"):
        """
        Retorna la colección de ChromaDB según el provider
        
        Args:
            provider: "llama" o "gemini"
            
        Returns:
            Colección de ChromaDB
        """
        if provider not in ["llama", "gemini"]:
            raise ValueError(f"Provider inválido: {provider}. Usar 'llama' o 'gemini'")
        
        # Si ya está en caché, retornarla
        if provider in self.collections:
            return self.collections[provider]
        
        # Crear o obtener la colección
        collection_name = f"documents_{provider}"
        collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine", "provider": provider}
        )
        
        # Cachear la colección
        self.collections[provider] = collection
        
        return collection
    
    def get_client(self):
        """Retorna el cliente de ChromaDB"""
        return self.client


# Instancia singleton del servicio
chroma_service = ChromaDBService()
