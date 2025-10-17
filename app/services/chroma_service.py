import chromadb
from chromadb.config import Settings as ChromaSettings
from app.core.config import settings


class ChromaDBService:
    """
    Servicio para manejar la conexión y operaciones con ChromaDB
    """
    
    def __init__(self):
        # Crear cliente de ChromaDB con persistencia
        self.client = chromadb.PersistentClient(
            path=settings.CHROMA_DB_PATH,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Obtener o crear la colección
        self.collection = self.client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}  # Usando similitud coseno
        )
    
    def get_collection(self):
        """Retorna la colección de ChromaDB"""
        return self.collection
    
    def get_client(self):
        """Retorna el cliente de ChromaDB"""
        return self.client


# Instancia singleton del servicio
chroma_service = ChromaDBService()
