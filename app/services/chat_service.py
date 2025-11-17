
from typing import List, Dict
from app.services.llm_service import llm_service
from app.services.embedding_service import embedding_service
from app.services.chroma_service import chroma_service
from app.services.rerank_service import rerank_service
from app.services.llamaIndex import llamaindex_service


class ChatService:
    """
    Servicio de chat que integra LLM con RAG (Retrieval Augmented Generation)
    """
    
    def __init__(self):
        self.llm = llm_service
        self.embedding = embedding_service
        self.chroma = chroma_service
        self.rerank = rerank_service
    
    def get_simple_response(self, message: str, provider: str = "llama") -> str:
        """
        Obtiene una respuesta simple sin contexto de documentos
        
        Args:
            message: Mensaje del usuario
            provider: "llama" o "gemini"
            
        Returns:
            Respuesta de la LLM
        """
        return self.llm.get_response(message, provider=provider)
    
    def get_rag_response(
        self,
        message: str,
        provider: str = "llama",
        n_results: int = 3,
        use_rerank: bool = True,
        rerank_top_k: int = None
    ) -> Dict[str, any]:
        """
        Obtiene una respuesta usando RAG (búsqueda en documentos + LLM)
        
        Args:
            message: Pregunta del usuario
            provider: "llama" o "gemini" (debe coincidir con los embeddings)
            n_results: Número de documentos a recuperar (antes de rerank)
            use_rerank: Si se debe aplicar reranking
            rerank_top_k: Número de documentos después de rerank (default: n_results)
            
        Returns:
            Dict con la respuesta y los documentos usados como contexto
        """
        # Por defecto, rerank_top_k es igual a n_results
        if rerank_top_k is None:
            rerank_top_k = n_results
        
        # Si usamos reranking, recuperar más documentos inicialmente
        initial_results = n_results * 3 if use_rerank else n_results
        
        # 1. Generar embedding de la pregunta
        query_embedding = self.embedding.generate_query_embedding(message, provider=provider)
        
        # 2. Buscar documentos relevantes en ChromaDB
        collection = self.chroma.get_collection(provider=provider)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=initial_results
        )
        
        # 3. Extraer documentos encontrados
        documents = results['documents'][0] if results['documents'] else []
        distances = results['distances'][0] if results['distances'] else []
        metadatas = results['metadatas'][0] if results['metadatas'] else []
        
        if not documents:
            # Si no hay documentos, respuesta simple
            response = self.llm.get_response(
                f"{message}\n\n(Nota: No se encontraron documentos relevantes en la base de datos)",
                provider=provider
            )
            return {
                "response": response,
                "sources": [],
                "found_documents": False,
                "reranked": False
            }
        
        # 4. Aplicar reranking si está habilitado
        if use_rerank and len(documents) > rerank_top_k:
            documents, scores, metadatas = self.rerank.rerank_documents(
                query=message,
                documents=documents,
                distances=distances,
                metadatas=metadatas,
                top_k=rerank_top_k
            )
            reranked = True
        else:
            # Sin reranking, solo tomar los primeros n_results
            documents = documents[:n_results]
            metadatas = metadatas[:n_results]
            reranked = False
        
        # 5. Generar respuesta con contexto
        response = self.llm.get_response_with_context(
            message=message,
            context_documents=documents,
            provider=provider
        )
        
        return {
            "response": response,
            "sources": documents,
            "metadatas": metadatas,
            "found_documents": True,
            "reranked": reranked
        }
    
    def get_response_with_history(
        self,
        message: str,
        chat_history: List[Dict[str, str]],
        provider: str = "llama",
        use_rag: bool = False,
        n_results: int = 3,
        use_rerank: bool = True
    ) -> Dict[str, any]:
        """
        Obtiene una respuesta considerando el historial de conversación
        
        Args:
            message: Mensaje actual del usuario
            chat_history: Historial de mensajes [{"role": "user"/"assistant", "content": "..."}]
            provider: "llama" o "gemini"
            use_rag: Si se debe buscar en documentos
            n_results: Número de documentos a recuperar (si use_rag=True)
            use_rerank: Si se debe aplicar reranking (si use_rag=True)
            
        Returns:
            Dict con la respuesta y metadatos
        """
        if use_rag:
            # Recuperar más documentos si vamos a hacer reranking
            initial_results = n_results * 3 if use_rerank else n_results
            
            # RAG con historial
            query_embedding = self.embedding.generate_query_embedding(message, provider=provider)
            collection = self.chroma.get_collection(provider=provider)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=initial_results
            )
            
            documents = results['documents'][0] if results['documents'] else []
            distances = results['distances'][0] if results['distances'] else []
            metadatas = results['metadatas'][0] if results['metadatas'] else []
            
            if documents:
                # Aplicar reranking si está habilitado
                if use_rerank and len(documents) > n_results:
                    documents, scores, metadatas = self.rerank.rerank_documents(
                        query=message,
                        documents=documents,
                        distances=distances,
                        metadatas=metadatas,
                        top_k=n_results
                    )
                else:
                    documents = documents[:n_results]
                    metadatas = metadatas[:n_results]
                
                # Construir contexto
                context = "\n\n".join([f"Documento {i+1}:\n{doc}" for i, doc in enumerate(documents)])
                
                # Agregar contexto al mensaje
                enhanced_message = f"""Contexto de documentos:
{context}

Pregunta del usuario: {message}"""
                
                response = self.llm.get_response_with_history(
                    message=enhanced_message,
                    chat_history=chat_history,
                    provider=provider
                )
                
                return {
                    "response": response,
                    "sources": documents,
                    "metadatas": metadatas,
                    "found_documents": True,
                    "reranked": use_rerank
                }
        
        # Sin RAG o sin documentos encontrados
        response = self.llm.get_response_with_history(
            message=message,
            chat_history=chat_history,
            provider=provider
        )
        
        return {
            "response": response,
            "sources": [],
            "found_documents": False,
            "reranked": False
        }
        
    def get_rag_response_with_llamaindex(
        self,
        message: str,
        provider: str = "gemini",
        n_results: int = 3,
        use_rerank: bool = False
    ) -> Dict[str, any]:
        """
        Obtiene una respuesta usando RAG con LlamaIndex
        Usa la colección documents_llamaindex_gemini_semantic
        
        Args:
            message: Pregunta del usuario
            provider: "llama" o "gemini" (para el LLM de respuesta)
            n_results: Número de documentos a recuperar
            use_rerank: Si se debe aplicar reranking a los resultados
            
        Returns:
            Dict con la respuesta y los documentos usados como contexto
        """
        # Colección fija de LlamaIndex con embeddings de Gemini
        collection_name = "documents_llamaindex_gemini_semantic"
        
        try:
            # 1. Consultar usando LlamaIndex (ya genera la respuesta internamente)
            result = llamaindex_service.query_index(
                collection_name=collection_name,
                query=message,
                provider="gemini",  # Siempre usar gemini para embeddings en esta colección
                top_k=n_results * 3 if use_rerank else n_results
            )
            
            if not result.get("success"):
                # Si falla la consulta, respuesta simple
                response = self.llm.get_response(
                    f"{message}\n\n(Nota: No se pudo consultar la base de datos de documentos)",
                    provider=provider
                )
                print("primer error")
                return {
                    "response": response,
                    "sources": [],
                    "metadatas": [],
                    "found_documents": False,
                    "reranked": False
                }
            
            # 2. Extraer nodos fuente
            source_nodes = result.get("source_nodes", [])
            
            if not source_nodes:
                # Si no hay documentos, respuesta simple
                response = self.llm.get_response(
                    f"{message}\n\n(Nota: No se encontraron documentos relevantes en la base de datos)",
                    provider=provider
                )
                print("no se encontraron documentos")
                return {
                    "response": response,
                    "sources": [],
                    "metadatas": [],
                    "found_documents": False,
                    "reranked": False
                }
            
            # 3. Extraer documentos y metadatos de los source_nodes
            documents = [node["text"] for node in source_nodes]
            metadatas = [node["metadata"] for node in source_nodes]
            scores = [node["score"] for node in source_nodes]
            
            # 4. Aplicar reranking si está habilitado
            if use_rerank and len(documents) > n_results:
                # Convertir scores a distancias (menor es mejor)
                # LlamaIndex usa similarity scores (mayor es mejor)
                distances = [1.0 - score for score in scores]
                
                documents, rerank_scores, metadatas = self.rerank.rerank_documents(
                    query=message,
                    documents=documents,
                    distances=distances,
                    metadatas=metadatas,
                    top_k=n_results
                )
                reranked = True
                print("aplicando reranking")
                # Regenerar respuesta con documentos rerankeados
                response = self.llm.get_response_with_context(
                    message=message,
                    context_documents=documents,
                    provider=provider
                )
            else:
                # Sin reranking, usar la respuesta de LlamaIndex directamente
                # Solo limitar a top n_results
                documents = documents[:n_results]
                metadatas = metadatas[:n_results]
                reranked = False
                
                # Usar la respuesta generada por LlamaIndex
                response = result.get("response", "")
                print("sin reranking")
            return {
                "response": response,
                "sources": documents,
                "metadatas": metadatas,
                "found_documents": True,
                "reranked": reranked
            }
            
        except Exception as e:
            # En caso de error, respuesta simple
            response = self.llm.get_response(
                f"{message}\n\n(Nota: Error al consultar documentos: {str(e)})",
                provider=provider
            )
            
            print(f"Error al consultar documentos: {str(e)} yair")
            return {
                "response": response,
                "sources": [],
                "metadatas": [],
                "found_documents": False,
                "reranked": False,
                "error": str(e)
            }


# Instancia singleton del servicio
chat_service = ChatService()


