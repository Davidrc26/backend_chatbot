import PyPDF2
from io import BytesIO
from typing import List
import re


class PDFService:
    """
    Servicio para procesar archivos PDF con chunking semántico
    """
    
    @staticmethod
    def extract_text(file_content: bytes) -> str:
        """
        Extrae el texto de un archivo PDF
        
        Args:
            file_content: Contenido del archivo PDF en bytes
            
        Returns:
            Texto extraído del PDF
        """
        pdf_file = BytesIO(file_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        
        return text.strip()
    
    @staticmethod
    def split_text_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Divide el texto en chunks semánticos respetando la estructura del documento
        
        Estrategia:
        1. Divide primero por párrafos (doble salto de línea)
        2. Agrupa párrafos hasta alcanzar el tamaño deseado
        3. Si un párrafo es muy grande, divide por oraciones
        4. Mantiene overlap inteligente (última oración del chunk anterior)
        
        Args:
            text: Texto completo a dividir
            chunk_size: Tamaño objetivo de cada chunk en caracteres
            overlap: Caracteres aproximados de solapamiento
            
        Returns:
            Lista de chunks de texto con coherencia semántica
        """
        # Normalizar saltos de línea
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Dividir por párrafos
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        previous_sentence = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Si el párrafo solo cabe en el chunk actual
            if len(current_chunk) + len(paragraph) + 2 <= chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            
            # Si el párrafo hace que se exceda, guardar chunk actual
            elif current_chunk:
                chunks.append(current_chunk.strip())
                
                # Overlap inteligente: agregar última oración del chunk anterior
                if previous_sentence and len(previous_sentence) <= overlap:
                    current_chunk = previous_sentence + "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                
                # Guardar última oración para próximo overlap
                previous_sentence = PDFService._get_last_sentence(current_chunk)
            
            # Si el párrafo es muy grande, dividirlo por oraciones
            else:
                if len(paragraph) > chunk_size:
                    sentence_chunks = PDFService._split_large_paragraph(
                        paragraph, 
                        chunk_size, 
                        overlap
                    )
                    
                    # Agregar chunks de oraciones
                    for i, sent_chunk in enumerate(sentence_chunks):
                        if i == 0 and previous_sentence and len(previous_sentence) <= overlap:
                            sent_chunk = previous_sentence + " " + sent_chunk
                        
                        chunks.append(sent_chunk.strip())
                        previous_sentence = PDFService._get_last_sentence(sent_chunk)
                    
                    current_chunk = ""
                else:
                    current_chunk = paragraph
                    previous_sentence = PDFService._get_last_sentence(paragraph)
        
        # Agregar el último chunk si existe
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return [c for c in chunks if c and len(c) > 50]  # Filtrar chunks muy pequeños
    
    @staticmethod
    def _split_large_paragraph(paragraph: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Divide un párrafo grande por oraciones
        
        Args:
            paragraph: Párrafo a dividir
            chunk_size: Tamaño objetivo del chunk
            overlap: Solapamiento entre chunks
            
        Returns:
            Lista de chunks basados en oraciones
        """
        # Dividir por oraciones (puntos seguidos de espacio y mayúscula)
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', paragraph)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Si agregar la oración no excede el tamaño
            if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
            else:
                # Guardar chunk actual y empezar uno nuevo
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    
                    # Overlap: agregar última parte del chunk anterior
                    if len(sentence) <= chunk_size:
                        overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                        current_chunk = overlap_text + " " + sentence
                    else:
                        current_chunk = sentence
                else:
                    current_chunk = sentence
        
        # Agregar último chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    @staticmethod
    def _get_last_sentence(text: str) -> str:
        """
        Obtiene la última oración completa de un texto
        
        Args:
            text: Texto del cual extraer la última oración
            
        Returns:
            Última oración del texto
        """
        # Buscar última oración (termina en punto, exclamación o interrogación)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if sentences:
            return sentences[-1].strip()
        return ""


# Instancia del servicio
pdf_service = PDFService()
