from typing import List, Dict, Optional
import tempfile
import os
from datetime import datetime
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from app.core.config import settings
from app.services.chroma_service import chroma_service


class LangChainService:
    """
    Servicio para procesar documentos PDF y vectorizarlos usando LangChain
    
    Características:
    - Extracción de PDFs con UnstructuredPDFLoader (más robusto que PyPDF)
    - Chunking inteligente con RecursiveCharacterTextSplitter optimizado
    - Soporte para múltiples providers de embeddings (Gemini 2.5 Flash / Llama 3.1)
    - Almacenamiento en ChromaDB (local o cloud)
    
    Nota: Si necesitas SemanticChunker, actualiza a langchain-text-splitters>=0.3.4
    """
    
    def __init__(self):
        self.chroma_path = settings.CHROMA_DB_PATH
        self.use_cloud = settings.USE_CHROMA_CLOUD
    
    def _get_embeddings(self, provider: str = "llama"):
        """
        Obtiene el modelo de embeddings según el provider
        
        Args:
            provider: "llama" para Ollama, "gemini" para Google Gemini 2.5 Flash
            
        Returns:
            Modelo de embeddings de LangChain
        """
        if provider.lower() == "gemini":
            return GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",  # Modelo de embeddings para Gemini 2.5
                google_api_key=settings.GOOGLE_API_KEY
            )
        else:  # llama/ollama
            return OllamaEmbeddings(
                model="llama3.1:8b",
                base_url=settings.OLLAMA_BASE_URL
            )
    
    def _get_smart_splitter(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Crea un RecursiveCharacterTextSplitter optimizado para RAG
        
        ESTRATEGIA INTELIGENTE:
        - Usa separadores jerárquicos para mantener contexto
        - Prioriza división por párrafos y oraciones completas
        - Mantiene overlap para no perder información entre chunks
        
        Args:
            chunk_size: Tamaño objetivo de cada chunk (caracteres)
            chunk_overlap: Overlap entre chunks para mantener contexto
            
        Returns:
            RecursiveCharacterTextSplitter configurado
        """
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            # Separadores jerárquicos (de más general a más específico)
            separators=[
                "\n\n\n",  # Secciones grandes
                "\n\n",    # Párrafos
                "\n",      # Líneas
                ". ",      # Oraciones (con espacio)
                "? ",      # Preguntas
                "! ",      # Exclamaciones
                "; ",      # Punto y coma
                ", ",      # Comas
                " ",       # Palabras
                ""         # Caracteres
            ],
            is_separator_regex=False,
            keep_separator=True  # Mantener separadores para mejor contexto
        )
    
    def extract_pdf_with_unstructured(
        self,
        file_content: bytes,
        filename: str,
        mode: str = "elements"
    ) -> List[Document]:
        """
        Extrae contenido de PDF usando UnstructuredPDFLoader
        
        Args:
            file_content: Contenido del PDF en bytes
            filename: Nombre del archivo
            mode: "single" (todo el doc), "elements" (por elementos), "paged" (por página)
            
        Returns:
            Lista de documentos de LangChain
        """
        # Crear archivo temporal para UnstructuredPDFLoader
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file_content)
            tmp_path = tmp_file.name
        
        try:
            # Cargar PDF con UnstructuredPDFLoader
            loader = UnstructuredPDFLoader(
                file_path=tmp_path,
                mode=mode
            )
            documents = loader.load()
            
            # Agregar metadata base
            for doc in documents:
                doc.metadata.update({
                    "filename": filename,
                    "extraction_method": "unstructured",
                    "extraction_mode": mode,
                    "upload_date": datetime.now().isoformat()
                })
            
            return documents
        
        finally:
            # Limpiar archivo temporal
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def process_pdf_and_store(
        self,
        file_content: bytes,
        filename: str,
        provider: str = "llama",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        extraction_mode: str = "elements",
        metadata: Optional[Dict] = None
    ) -> Dict[str, any]:
        """
        Procesa un PDF completo: extracción -> chunking inteligente -> embeddings -> almacenamiento
        
        Args:
            file_content: Contenido del PDF en bytes
            filename: Nombre del archivo
            provider: "llama" o "gemini"
            chunk_size: Tamaño objetivo de chunks (1000-1500 recomendado para RAG)
            chunk_overlap: Overlap entre chunks (150-250 recomendado)
            extraction_mode: "elements", "single" o "paged"
            metadata: Metadata adicional (autor, categoría, tags, etc.)
                
        Returns:
            Dict con información del procesamiento
        """
        # 1. Extraer contenido del PDF con UnstructuredPDFLoader
        documents = self.extract_pdf_with_unstructured(
            file_content=file_content,
            filename=filename,
            mode=extraction_mode
        )
        
        # 2. Crear splitter inteligente
        text_splitter = self._get_smart_splitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # 3. Dividir documentos en chunks inteligentes
        splits = text_splitter.split_documents(documents)
        
        # 4. Agregar metadata personalizada
        for i, doc in enumerate(splits):
            doc.metadata.update({
                "chunk_index": i,
                "total_chunks": len(splits),
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "chunking_strategy": "recursive_optimized",
                "source": "langchain",
                "provider": provider
            })
            
            # Agregar metadata adicional si se proporcionó
            if metadata:
                doc.metadata.update(metadata)
        
        # 5. Obtener embeddings
        embeddings = self._get_embeddings(provider)
        
        # 6. Crear/actualizar vectorstore en ChromaDB usando chroma_service
        collection_name = f"documents_langchain_{provider}"
        
        # Obtener cliente de ChromaDB del servicio existente
        client = chroma_service.get_client()
        
        # Crear vectorstore con LangChain usando el cliente compartido
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            collection_name=collection_name,
            client=client
        )
        
        return {
            "success": True,
            "chunks_created": len(splits),
            "original_documents": len(documents),
            "collection_name": collection_name,
            "extraction_mode": extraction_mode,
            "chunking_strategy": "recursive_optimized",
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "provider": provider,
            "message": f"PDF procesado con LangChain (chunking inteligente). {len(documents)} documentos extraídos, {len(splits)} chunks creados."
        }


# Singleton
langchain_service = LangChainService()