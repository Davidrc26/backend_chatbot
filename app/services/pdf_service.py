import pypdf
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
        Extrae el texto de un archivo PDF y limpia información basura
        
        Args:
            file_content: Contenido del archivo PDF en bytes
            
        Returns:
            Texto extraído y limpiado del PDF
        """
        pdf_file = BytesIO(file_content)
        pdf_reader = pypdf.PdfReader(pdf_file)
        
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        
        cleaned_text = PDFService._clean_extracted_text(text)
        
        return cleaned_text.strip()
    
    @staticmethod
    def _clean_extracted_text(text: str) -> str:
        """
        Limpia el texto extraído del PDF removiendo basura común
        
        Args:
            text: Texto extraído del PDF
            
        Returns:
            Texto limpio
        """
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
    
        text = re.sub(r'(.)\1{10,}', '', text)
        
    
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            special_chars = len(re.findall(r'[^a-zA-Z0-9\s\.,;:¿?¡!\-áéíóúÁÉÍÓÚñÑüÜ]', line))
            total_chars = len(line)
            
           
            if total_chars > 0 and (special_chars / total_chars) > 0.4:
                continue
        
            if len(line) < 15 and re.match(r'^[\d\s\-\|\.]+$', line):
                continue
            
            footer_patterns = [
                r'^\s*p[aá]gina\s+\d+\s*$',
                r'^\s*page\s+\d+\s*$',
                r'^\s*\d+\s*/\s*\d+\s*$',
                r'^\s*\d+\s+de\s+\d+\s*$',
                r'^\s*\[\s*\d+\s*\]\s*$',
                r'^\s*-\s*\d+\s*-\s*$'
            ]
            
            is_footer = any(re.match(pattern, line.lower()) for pattern in footer_patterns)
            if is_footer:
                continue
            
            if re.match(r'^(https?://|www\.|[\w\.-]+@[\w\.-]+).*$', line) and len(line) < 100:
                continue
            
            cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines)
        
        text = re.sub(r' {2,}', ' ', text)
        
        text = re.sub(r'\n{4,}', '\n\n\n', text)
        
        text = '\n'.join(line.strip() for line in text.split('\n'))
        
        text = PDFService._remove_repetitive_headers(text)
        
        return text
    
    @staticmethod
    def _remove_repetitive_headers(text: str) -> str:
        """
        Elimina encabezados y pies de página repetitivos que aparecen en múltiples páginas
        
        Args:
            text: Texto a limpiar
            
        Returns:
            Texto sin líneas repetitivas
        """
        lines = text.split('\n')
        line_frequency = {}
        for line in lines:
            stripped = line.strip()
            if stripped and len(stripped) < 100:
                line_frequency[stripped] = line_frequency.get(stripped, 0) + 1
        

        repetitive_lines = {line for line, count in line_frequency.items() if count > 3}
        
    
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped and stripped not in repetitive_lines:
                cleaned_lines.append(line)
            elif not stripped:
                cleaned_lines.append('')  
        
        return '\n'.join(cleaned_lines)
    
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
