from typing import List, Dict, Tuple
import re


class RerankService:
    """
    Servicio para reordenar documentos recuperados por relevancia
    
    Implementa varias técnicas de reranking sin dependencias externas:
    1. Score por distancia (de ChromaDB)
    2. Score por longitud óptima
    3. Score por densidad de keywords
    4. Score combinado normalizado
    """
    
    @staticmethod
    def rerank_documents(
        query: str,
        documents: List[str],
        distances: List[float],
        metadatas: List[Dict] = None,
        top_k: int = 3
    ) -> Tuple[List[str], List[float], List[Dict]]:
        """
        Reordena documentos por relevancia combinando múltiples señales
        
        Args:
            query: Pregunta del usuario
            documents: Lista de documentos recuperados
            distances: Distancias de similitud de ChromaDB
            metadatas: Metadatos de los documentos
            top_k: Número de documentos a retornar
            
        Returns:
            Tupla de (documentos reordenados, scores, metadatas reordenadas)
        """
        if not documents:
            return [], [], []
        
        # Calcular scores para cada documento
        scored_docs = []
        
        for i, (doc, distance) in enumerate(zip(documents, distances)):
            # 1. Score de similitud semántica (invertir distancia)
            similarity_score = RerankService._distance_to_score(distance)
            
            # 2. Score de longitud (documentos muy cortos o muy largos penalizan)
            length_score = RerankService._length_score(doc)
            
            # 3. Score de relevancia por keywords
            keyword_score = RerankService._keyword_overlap_score(query, doc)
            
            # 4. Score por metadata (si está disponible)
            metadata_score = RerankService._metadata_score(
                metadatas[i] if metadatas and i < len(metadatas) else None
            )
            
            # Combinar scores con pesos
            combined_score = (
                similarity_score * 0.4 +  # 40% similitud semántica
                keyword_score * 0.3 +      # 30% keywords
                length_score * 0.2 +       # 20% longitud
                metadata_score * 0.1       # 10% metadata
            )
            
            scored_docs.append({
                'document': doc,
                'score': combined_score,
                'distance': distance,
                'metadata': metadatas[i] if metadatas and i < len(metadatas) else {},
                'similarity_score': similarity_score,
                'keyword_score': keyword_score,
                'length_score': length_score,
                'metadata_score': metadata_score
            })
        
        # Ordenar por score descendente
        scored_docs.sort(key=lambda x: x['score'], reverse=True)
        
        # Retornar top_k documentos
        top_docs = scored_docs[:top_k]
        
        reranked_documents = [doc['document'] for doc in top_docs]
        reranked_scores = [doc['score'] for doc in top_docs]
        reranked_metadatas = [doc['metadata'] for doc in top_docs]
        
        return reranked_documents, reranked_scores, reranked_metadatas
    
    @staticmethod
    def _distance_to_score(distance: float) -> float:
        """
        Convierte distancia de ChromaDB a score de similitud (0-1)
        Menor distancia = mayor score
        """
        # Normalizar distancia coseno (típicamente 0-2) a score 0-1
        return max(0, 1 - (distance / 2))
    
    @staticmethod
    def _length_score(document: str, optimal_min: int = 200, optimal_max: int = 1500) -> float:
        """
        Score basado en longitud óptima del documento
        
        Penaliza documentos muy cortos (poco contexto) o muy largos (ruido)
        """
        length = len(document)
        
        if length < optimal_min:
            # Muy corto: penalizar proporcionalmente
            return length / optimal_min
        elif length > optimal_max:
            # Muy largo: penalizar levemente
            return max(0.7, optimal_max / length)
        else:
            # Longitud óptima
            return 1.0
    
    @staticmethod
    def _keyword_overlap_score(query: str, document: str) -> float:
        """
        Score basado en overlap de keywords entre query y documento
        """
        # Normalizar y tokenizar
        query_tokens = set(RerankService._tokenize(query.lower()))
        doc_tokens = set(RerankService._tokenize(document.lower()))
        
        # Remover stopwords comunes
        stopwords = {'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'ser', 'se', 'no',
                    'por', 'con', 'para', 'una', 'su', 'es', 'al', 'lo', 'del', 'las',
                    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        
        query_tokens = query_tokens - stopwords
        doc_tokens = doc_tokens - stopwords
        
        if not query_tokens:
            return 0.5
        
        # Calcular intersección
        overlap = len(query_tokens.intersection(doc_tokens))
        
        # Score normalizado (Jaccard similarity adaptado)
        score = overlap / len(query_tokens)
        
        return min(1.0, score)
    
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Tokeniza texto en palabras
        """
        # Extraer palabras (letras y números)
        tokens = re.findall(r'\b\w+\b', text)
        return [t for t in tokens if len(t) > 2]  # Filtrar tokens muy cortos
    
    @staticmethod
    def _metadata_score(metadata: Dict) -> float:
        """
        Score basado en metadata del documento
        
        Puede dar boost a documentos con ciertas características:
        - Documentos más recientes
        - Documentos de fuentes confiables
        - Documentos con mejor estructura
        """
        if not metadata:
            return 0.5
        
        score = 0.5
        
        # Boost por chunk_index (primeros chunks suelen ser más relevantes)
        if 'chunk_index' in metadata:
            chunk_idx = metadata['chunk_index']
            # Dar más peso a los primeros 3 chunks
            if chunk_idx < 3:
                score += 0.3
            elif chunk_idx < 6:
                score += 0.2
            else:
                score += 0.1
        
        # Boost por total_chunks (documentos con pocos chunks = más concisos)
        if 'total_chunks' in metadata:
            total = metadata['total_chunks']
            if total <= 5:
                score += 0.2
            elif total <= 10:
                score += 0.1
        
        return min(1.0, score)
    
    @staticmethod
    def get_rerank_explanation(
        query: str,
        documents: List[str],
        distances: List[float],
        metadatas: List[Dict] = None
    ) -> List[Dict]:
        """
        Genera explicación detallada del reranking para debugging
        
        Returns:
            Lista de dicts con scores desglosados por documento
        """
        explanations = []
        
        for i, (doc, distance) in enumerate(zip(documents, distances)):
            similarity_score = RerankService._distance_to_score(distance)
            length_score = RerankService._length_score(doc)
            keyword_score = RerankService._keyword_overlap_score(query, doc)
            metadata_score = RerankService._metadata_score(
                metadatas[i] if metadatas and i < len(metadatas) else None
            )
            
            combined_score = (
                similarity_score * 0.4 +
                keyword_score * 0.3 +
                length_score * 0.2 +
                metadata_score * 0.1
            )
            
            explanations.append({
                'document_preview': doc[:100] + '...' if len(doc) > 100 else doc,
                'document_length': len(doc),
                'combined_score': round(combined_score, 3),
                'similarity_score': round(similarity_score, 3),
                'keyword_score': round(keyword_score, 3),
                'length_score': round(length_score, 3),
                'metadata_score': round(metadata_score, 3),
                'original_distance': round(distance, 3),
                'metadata': metadatas[i] if metadatas and i < len(metadatas) else {}
            })
        
        return explanations


# Instancia singleton del servicio
rerank_service = RerankService()
