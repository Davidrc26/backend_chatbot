import google.generativeai as genai
import ollama
from app.core.config import settings
from typing import List, Dict
import time


class LLMService:
    """
    Servicio para generar respuestas de chat usando Gemini u Ollama (Llama)
    """
    
    def __init__(self):
        # Configurar Gemini
        if settings.GOOGLE_API_KEY:
            genai.configure(api_key=settings.GOOGLE_API_KEY)
        
        # Configurar Ollama
        self.ollama_client = ollama.Client(host=settings.OLLAMA_BASE_URL)
    
    def get_response(self, message: str, provider: str = "llama") -> str:
        """
        Genera una respuesta de chat simple sin contexto
        
        Args:
            message: Mensaje del usuario
            provider: "llama" o "gemini"
            
        Returns:
            Respuesta generada por la LLM
        """
        if provider == "gemini":
            return self._get_gemini_response(message)
        elif provider == "llama":
            return self._get_llama_response(message)
        else:
            raise ValueError(f"Provider no soportado: {provider}")
    
    def get_response_with_context(
        self, 
        message: str, 
        context_documents: List[str],
        provider: str = "llama"
    ) -> str:
        """
        Genera una respuesta usando documentos de contexto (RAG)
        
        Args:
            message: Pregunta del usuario
            context_documents: Lista de documentos relevantes del contexto
            provider: "llama" o "gemini"
            
        Returns:
            Respuesta generada con el contexto
        """
        # Construir el contexto
        context = "\n\n".join([f"Documento {i+1}:\n{doc}" for i, doc in enumerate(context_documents)])
        
        # Prompt con contexto
        prompt = f"""Eres un asistente experto. Responde SOLO basándote en el contexto.
Si no hay info suficiente, di "No tengo información suficiente".

Contexto:
{context}

Pregunta: {message}

Instrucciones:
- Cita los documentos que uses
- Sé específico y preciso
- No inventes información

Respuesta:"""
        
        if provider == "gemini":
            return self._get_gemini_response(prompt)
        elif provider == "llama":
            return self._get_llama_response(prompt)
        else:
            raise ValueError(f"Provider no soportado: {provider}")
    
    def get_response_with_history(
        self,
        message: str,
        chat_history: List[Dict[str, str]],
        provider: str = "llama"
    ) -> str:
        """
        Genera una respuesta considerando el historial de conversación
        
        Args:
            message: Mensaje actual del usuario
            chat_history: Lista de mensajes previos [{"role": "user"/"assistant", "content": "..."}]
            provider: "llama" o "gemini"
            
        Returns:
            Respuesta generada considerando el historial
        """
        if provider == "gemini":
            return self._get_gemini_response_with_history(message, chat_history)
        elif provider == "llama":
            return self._get_llama_response_with_history(message, chat_history)
        else:
            raise ValueError(f"Provider no soportado: {provider}")
    
    def _get_gemini_response(self, prompt: str, max_retries: int = 3) -> str:
        """
        Genera respuesta usando Gemini con reintentos automáticos
        
        Args:
            prompt: El prompt a enviar
            max_retries: Número máximo de reintentos en caso de error
            
        Returns:
            Respuesta de Gemini
        """
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                return response.text
            except Exception as e:
                if attempt < max_retries - 1:
                    # Esperar con backoff exponencial: 2s, 4s, 8s
                    wait_time = 2 ** (attempt + 1)
                    print(f"⚠️  Error en Gemini (intento {attempt + 1}/{max_retries}): {str(e)}")
                    print(f"   Reintentando en {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    # Último intento falló
                    print(f"❌ Error en Gemini después de {max_retries} intentos: {str(e)}")
                    raise Exception(f"Gemini API error después de {max_retries} intentos: {str(e)}")
    
    def _get_llama_response(self, prompt: str) -> str:
        """Genera respuesta usando Llama (Ollama)"""
        response = self.ollama_client.chat(
            model=settings.OLLAMA_MODEL,
            messages=[
                {'role': 'user', 'content': prompt}
            ]
        )
        return response['message']['content']
    
    def _get_gemini_response_with_history(
        self,
        message: str,
        chat_history: List[Dict[str, str]],
        max_retries: int = 3
    ) -> str:
        """
        Genera respuesta con historial usando Gemini con reintentos automáticos
        
        Args:
            message: Mensaje actual
            chat_history: Historial de conversación
            max_retries: Número máximo de reintentos
            
        Returns:
            Respuesta de Gemini
        """
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        
        for attempt in range(max_retries):
            try:
                # Convertir historial al formato de Gemini
                chat = model.start_chat(history=[
                    {
                        'role': 'user' if msg['role'] == 'user' else 'model',
                        'parts': [msg['content']]
                    }
                    for msg in chat_history
                ])
                
                response = chat.send_message(message)
                return response.text
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** (attempt + 1)
                    print(f"⚠️  Error en Gemini con historial (intento {attempt + 1}/{max_retries}): {str(e)}")
                    print(f"   Reintentando en {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ Error en Gemini después de {max_retries} intentos: {str(e)}")
                    raise Exception(f"Gemini API error después de {max_retries} intentos: {str(e)}")
    
    def _get_llama_response_with_history(
        self,
        message: str,
        chat_history: List[Dict[str, str]]
    ) -> str:
        """Genera respuesta con historial usando Llama"""
        # Agregar el mensaje actual al historial
        messages = chat_history + [{'role': 'user', 'content': message}]
        
        response = self.ollama_client.chat(
            model=settings.OLLAMA_MODEL,
            messages=messages
        )
        return response['message']['content']


# Instancia singleton del servicio
llm_service = LLMService()
