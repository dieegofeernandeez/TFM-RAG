import chainlit as cl
import requests
import os
import json
import logging
from typing import List, Dict

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_BACKEND_URL = os.getenv("API_BACKEND_URL", "http://localhost:8000")


# Configuración de timeouts
HEALTH_CHECK_TIMEOUT = 10
QUERY_TIMEOUT = 60

def validate_api_response(response: requests.Response) -> Dict:
    """Validar y parsear respuesta del API"""
    try:
        # Verificar content-type
        content_type = response.headers.get('content-type', '')
        if 'application/json' not in content_type:
            logger.warning(f"Respuesta no es JSON. Content-Type: {content_type}")
            return {"error": "Respuesta inválida del servicio"}
        
        # Intentar parsear JSON
        data = response.json()
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Error parseando JSON: {e}")
        return {"error": "Respuesta JSON inválida del servicio"}
    except Exception as e:
        logger.error(f"Error validando respuesta: {e}")
        return {"error": "Error procesando respuesta del servicio"}

def check_api_health() -> bool:
    """Verificar que el API esté disponible (Groq Llama 3.1)"""
    try:
        logger.info("Verificando estado del API Groq Llama 3.1...")
        response = requests.get(
            f"{API_BACKEND_URL}/health", 
            timeout=HEALTH_CHECK_TIMEOUT
        )
        if response.status_code == 200:
            data = validate_api_response(response)
            if "error" not in data:
                logger.info("API Groq Llama 3.1 disponible")
                return True
            else:
                logger.warning(f"API respondió con error: {data.get('error')}")
                return False
        else:
            logger.warning(f"API respondió con código: {response.status_code}")
            return False
            
    except requests.exceptions.Timeout:
        logger.error(f"Timeout verificando API (>{HEALTH_CHECK_TIMEOUT}s)")
        return False
    except requests.exceptions.ConnectionError:
        logger.error("No se puede conectar al API")
        return False
    except Exception as e:
        logger.error(f"Error inesperado verificando API: {e}")
        return False

class ChatSession:
    def __init__(self):
        self.messages: List[Dict[str, str]] = []
    
    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
    
    def get_response(self, user_message: str) -> str:
        # Agregar mensaje del usuario
        self.add_message("user", user_message)
        
        try:
            # Preparar payload
            payload = {
                "messages": self.messages,
                "model": os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
                "temperature": 0.7,
                "max_tokens": 1000
            }
            
            logger.info(f"Sending request to {API_BACKEND_URL}/chat")
            
            # Hacer llamada al backend
            response = requests.post(
                f"{API_BACKEND_URL}/chat",
                json=payload,
                timeout=QUERY_TIMEOUT,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = validate_api_response(response)
                
                if "error" in data:
                    return f"Error del servicio: {data['error']}"
                
                ai_response = data.get("response", "No se pudo obtener respuesta")
                usage = data.get("usage", {})
                
                # Agregar respuesta del asistente
                self.add_message("assistant", ai_response)
                
                logger.info(f"Response received. Tokens used: {usage.get('total_tokens', 'N/A')}")
                return ai_response
            else:
                data = validate_api_response(response)
                error_msg = data.get("error", f"Error HTTP {response.status_code}")
                logger.error(f"API error: {error_msg}")
                return f"Lo siento, hubo un error al procesar tu mensaje: {error_msg}"
                
        except requests.exceptions.Timeout:
            error_msg = f"Timeout al conectar con el backend (>{QUERY_TIMEOUT}s)"
            logger.error(error_msg)
            return f"Lo siento, la conexión tardó demasiado: {error_msg}"
        except requests.exceptions.ConnectionError:
            error_msg = "No se puede conectar al API"
            logger.error(error_msg)
            return f"Lo siento, no puedo conectar con el servicio: {error_msg}"
        except Exception as e:
            error_msg = f"Error inesperado: {str(e)}"
            logger.error(error_msg)
            return f"Lo siento, ocurrió un error inesperado: {error_msg}"

@cl.on_chat_start
async def start():
    """Inicializar el chat"""
    try:
        # Inicializar sesión de chat
        session = ChatSession()
        cl.user_session.set("chat_session", session)
        
        # Verificar que el API esté disponible
        service_available = check_api_health()
        
        if service_available:
            await cl.Message(
                content="¡Hola! Soy tu asistente de IA. ¿En qué puedo ayudarte hoy?",
                author="Asistente"
            ).send()
        else:
            await cl.Message(
                content="⚠️ **Sistema no disponible**\n\n"
                       "El servicio de IA no está respondiendo. "
                       "Por favor, verifica que todos los servicios estén ejecutándose correctamente.",
                author="Asistente"
            ).send()
        
        logger.info("Nueva sesión de chat iniciada")
    except Exception as e:
        logger.error(f"Error en inicialización: {e}")
        await cl.Message(
            content="💥 **Error de inicialización**\n\n"
                   "Ocurrió un error al inicializar el sistema. "
                   "Por favor, reinicia la aplicación.",
            author="Asistente"
        ).send()

@cl.on_message
async def main(message: cl.Message):
    """Procesar mensaje del usuario"""
    user_message = message.content.strip()
    
    # Validar que el mensaje no esté vacío
    if not user_message:
        await cl.Message(
            content="⚠️ Por favor, envía una pregunta válida.",
            author="Asistente"
        ).send()
        return
    
    # Obtener sesión de chat
    session: ChatSession = cl.user_session.get("chat_session")
    
    if not session:
        await cl.Message(
            content="Error: Sesión no encontrada. Por favor, recarga la página.",
            author="Sistema"
        ).send()
        return
    
    try:
        # Mostrar indicador de carga
        loading_msg = cl.Message(content="🤔 Procesando tu consulta...", author="Asistente")
        await loading_msg.send()
        
        # Verificar disponibilidad del servicio antes de la consulta
        service_available = check_api_health()
        if not service_available:
            await loading_msg.remove()
            await cl.Message(
                content="🔌 **Servicio no disponible**\n\n"
                       "No puedo conectar con el servicio de IA en este momento. "
                       "Por favor, verifica que todos los servicios estén ejecutándose.",
                author="Asistente"
            ).send()
            return
        
        # Obtener respuesta del backend
        logger.info(f"Procesando mensaje: {user_message[:50]}...")
        response = session.get_response(user_message)
        
        # Eliminar mensaje de carga
        await loading_msg.remove()
        
        # Validar que tengamos una respuesta válida
        if not response or response.strip() == "":
            await cl.Message(
                content="⚠️ El servicio no pudo generar una respuesta para tu consulta. "
                       "Intenta reformular tu pregunta.",
                author="Asistente"
            ).send()
            return
        
        await cl.Message(content=response, author="Asistente").send()
        logger.info("Respuesta enviada exitosamente")
        
    except Exception as e:
        logger.error(f"Error procesando mensaje: {e}")
        await cl.Message(
            content="💥 **Error inesperado**\n\n"
                   "Ocurrió un error inesperado procesando tu consulta. "
                   "Por favor, intenta de nuevo.",
            author="Asistente"
        ).send()