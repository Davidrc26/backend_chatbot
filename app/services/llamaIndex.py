from typing import List, Dict, Optional
import tempfile
import os
from datetime import datetime
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.llms.ollama import Ollama

from llama_index.vector_stores.chroma import ChromaVectorStore
from app.core.config import settings
from app.services.chroma_service import chroma_service


class LlamaIndexService:
    """
    Servicio para procesar documentos PDF y vectorizarlos usando LlamaIndex
    
    Características:
    - Extracción de PDFs con SimpleDirectoryReader (robusto y simple)
    - Chunking semántico con SemanticSplitterNodeParser (estable y maduro)
    - Soporte para múltiples providers de embeddings (Gemini / Ollama)
    - Almacenamiento en ChromaDB (local o cloud)
    """
    
    def __init__(self):
        self.chroma_path = settings.CHROMA_DB_PATH
        self.use_cloud = settings.USE_CHROMA_CLOUD
    
    def _get_embeddings(self, provider: str = "llama"):
        """
        Obtiene el modelo de embeddings según el provider
        
        Args:
            provider: "llama" para Ollama, "gemini" para Google Gemini
            
        Returns:
            Modelo de embeddings de LlamaIndex
        """
        if provider.lower() == "gemini":
            return GeminiEmbedding(
                model_name="models/text-embedding-004",
                api_key=settings.GOOGLE_API_KEY
            )
        else:  # llama/ollama
            return OllamaEmbedding(
                model_name=settings.OLLAMA_MODEL,
                base_url=settings.OLLAMA_BASE_URL
            )
    
    def _get_llm(self, provider: str = "llama"):
        """
        Obtiene el modelo LLM según el provider
        
        Args:
            provider: "llama" para Ollama, "gemini" para Google Gemini
            
        Returns:
            Modelo LLM de LlamaIndex
        """
        if provider.lower() == "gemini":
            return Gemini(
                model="models/gemini-2.5-flash-lite",
                api_key=settings.GOOGLE_API_KEY
            )
        else:  # llama/ollama
            return Ollama(
                model=settings.OLLAMA_MODEL,
                base_url=settings.OLLAMA_BASE_URL
            )
    
    def _get_semantic_splitter(
        self,
        provider: str = "llama",
        buffer_size: int = 1,
        breakpoint_percentile_threshold: int = 95
    ):
        """
        Crea un SemanticSplitterNodeParser configurado
        
        CHUNKING SEMÁNTICO (LlamaIndex):
        Divide el texto basándose en la similitud semántica entre oraciones
        usando embeddings. Es más estable y maduro que la versión de LangChain.
        
        Args:
            provider: "llama" o "gemini"
            buffer_size: Número de oraciones a comparar simultáneamente (1-3)
            breakpoint_percentile_threshold: Umbral percentil para dividir
                - 90-95: Chunks grandes (conservador)
                - 70-85: Tamaño medio
                - 50-70: Chunks pequeños (agresivo)
            
        Returns:
            SemanticSplitterNodeParser configurado
        """
        embeddings = self._get_embeddings(provider)
        
        return SemanticSplitterNodeParser(
            buffer_size=buffer_size,
            breakpoint_percentile_threshold=breakpoint_percentile_threshold,
            embed_model=embeddings
        )
    
    def extract_pdf(
        self,
        file_content: bytes,
        filename: str
    ) -> List:
        """
        Extrae contenido de PDF usando SimpleDirectoryReader
        
        Args:
            file_content: Contenido del PDF en bytes
            filename: Nombre del archivo
            
        Returns:
            Lista de documentos de LlamaIndex
        """
        # Crear archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf', mode='wb') as tmp_file:
            tmp_file.write(file_content)
            tmp_path = tmp_file.name
            tmp_dir = os.path.dirname(tmp_path)
        
        try:
            # Cargar PDF con SimpleDirectoryReader
            # Es más simple pero igual de robusto que UnstructuredPDFLoader
            reader = SimpleDirectoryReader(
                input_files=[tmp_path]
            )
            documents = reader.load_data()
            
            # Agregar metadata base
            for doc in documents:
                doc.metadata.update({
                    "filename": filename,
                    "extraction_method": "llamaindex_simple",
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
        buffer_size: int = 1,
        breakpoint_percentile_threshold: int = 95,
        metadata: Optional[Dict] = None
    ) -> Dict[str, any]:
        """
        Procesa un PDF completo: extracción -> chunking semántico -> embeddings -> almacenamiento
        
        Args:
            file_content: Contenido del PDF en bytes
            filename: Nombre del archivo
            provider: "llama" o "gemini"
            buffer_size: Ventana de comparación (1-3 recomendado)
            breakpoint_percentile_threshold: Umbral percentil (50-95)
            metadata: Metadata adicional (autor, categoría, tags, etc.)
                
        Returns:
            Dict con información del procesamiento
        """
        # 1. Extraer contenido del PDF
        documents = self.extract_pdf(
            file_content=file_content,
            filename=filename
        )
        
        # 2. Crear semantic splitter
        splitter = self._get_semantic_splitter(
            provider=provider,
            buffer_size=buffer_size,
            breakpoint_percentile_threshold=breakpoint_percentile_threshold
        )
        
        # 3. Dividir documentos en nodos semánticos
        nodes = splitter.get_nodes_from_documents(documents)
        
        # 4. Agregar metadata personalizada
        for i, node in enumerate(nodes):
            node.metadata.update({
                "chunk_index": i,
                "total_chunks": len(nodes),
                "chunking_strategy": "semantic",
                "buffer_size": buffer_size,
                "breakpoint_percentile_threshold": breakpoint_percentile_threshold,
                "source": "llamaindex",
                "provider": provider
            })
            
            # Agregar metadata adicional si se proporcionó
            if metadata:
                node.metadata.update(metadata)
        
        # 5. Obtener embeddings
        embed_model = self._get_embeddings(provider)
        
        # 6. Crear vectorstore en ChromaDB usando chroma_service
        collection_name = f"documents_llamaindex_{provider}_semantic"
        
        # Obtener o crear colección en ChromaDB
        client = chroma_service.get_client()
        chroma_collection = client.get_or_create_collection(collection_name)
        
        # Crear vector store de LlamaIndex
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        # Crear storage context
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Crear índice con los nodos
        index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            embed_model=embed_model,
            show_progress=True
        )
        
        return {
            "success": True,
            "chunks_created": len(nodes),
            "original_documents": len(documents),
            "collection_name": collection_name,
            "chunking_strategy": "semantic",
            "buffer_size": buffer_size,
            "breakpoint_percentile_threshold": breakpoint_percentile_threshold,
            "provider": provider,
            "index_id": index.index_id,
            "message": f"PDF procesado con LlamaIndex (semantic chunking). {len(documents)} documentos extraídos, {len(nodes)} nodos creados."
        }
    
    def query_index(
        self,
        collection_name: str,
        query: str,
        provider: str = "llama",
        top_k: int = 3
    ) -> Dict[str, any]:
        """
        Consulta un índice existente
        
        Args:
            collection_name: Nombre de la colección en ChromaDB
            query: Consulta del usuario
            provider: Provider de embeddings usado
            top_k: Número de resultados a retornar
            
        Returns:
            Dict con la respuesta y los nodos fuente
        """
        # Obtener embeddings y LLM
        embed_model = self._get_embeddings(provider)
        llm = self._get_llm(provider)
        
        # Obtener colección de ChromaDB
        client = chroma_service.get_client()
        
        try:
            chroma_collection = client.get_collection(collection_name)
        except Exception as e:
            return {
                "success": False,
                "error": f"Colección no encontrada: {collection_name}",
                "message": str(e)
            }
        
        # Crear vector store
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        # Cargar índice desde storage
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model
        )
        
        # Crear query engine con configuración para responder SOLO basándose en los documentos
        from llama_index.core.prompts import PromptTemplate
        
        # Prompt personalizado que obliga a usar solo el contexto
        qa_prompt_template = PromptTemplate(
            """Eres un asistente que SOLO puede responder usando la información proporcionada en el contexto.

Reglas estrictas:
1. SOLO usa información que esté explícitamente en el contexto proporcionado
2. Si la información no está en el contexto, responde: "No tengo información sobre eso en los documentos disponibles"
3. NO inventes, supongas o uses conocimiento externo
4. Cita de dónde sacas la información cuando sea relevante

Contexto:
{context_str}

Pregunta: {query_str}

Respuesta basada ÚNICAMENTE en el contexto:"""
        )
        
        query_engine = index.as_query_engine(
            similarity_top_k=top_k,
            text_qa_template=qa_prompt_template,
            response_mode="compact",  # Usa el contexto de forma compacta
            llm=llm  # Usar el LLM especificado
        )
        
        # Ejecutar query
        response = query_engine.query(query)
        
        # Extraer información de los nodos fuente
        source_nodes = []
        for source_node in response.source_nodes:
            source_nodes.append({
                "text": source_node.node.get_content(),
                "score": source_node.score,
                "metadata": source_node.node.metadata
            })
        
        return {
            "success": True,
            "response": str(response),
            "source_nodes": source_nodes,
            "query": query,
            "collection_name": collection_name
        }


# Singleton
llamaindex_service = LlamaIndexService()