import PyPDF2
from io import BytesIO
from typing import List


class PDFService:
    """
    Servicio para procesar archivos PDF
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
        Divide el texto en chunks manejables para embeddings
        
        Args:
            text: Texto completo a dividir
            chunk_size: Tamaño de cada chunk en caracteres
            overlap: Solapamiento entre chunks
            
        Returns:
            Lista de chunks de texto
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Intentar cortar en un punto natural (espacio, punto, salto de línea)
            if end < len(text):
                last_space = chunk.rfind(' ')
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                
                cut_point = max(last_space, last_period, last_newline)
                if cut_point > chunk_size * 0.5:  # Solo si está en la segunda mitad
                    chunk = chunk[:cut_point]
                    end = start + cut_point
            
            chunks.append(chunk.strip())
            start = end - overlap
        
        return [c for c in chunks if c]  # Filtrar chunks vacíos


# Instancia del servicio
pdf_service = PDFService()
