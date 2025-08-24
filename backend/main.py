from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import os
from typing import List, Optional
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Llama Chat API", version="1.0.0")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3-70b-8192")
client = Groq(api_key=GROQ_API_KEY)

# Modelos Pydantic
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    model: Optional[str] = GROQ_MODEL
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000

class ChatResponse(BaseModel):
    response: str
    usage: dict

@app.get("/")
async def root():
    return {"message": "Groq Llama 3.1 Chat API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "groq-llama-3.1"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        groq_messages = [
            {"role": m.role, "content": m.content} for m in request.messages
        ]
        response = client.chat.completions.create(
            model=request.model,
            messages=groq_messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        return ChatResponse(
            response=response.choices[0].message.content,
            usage=response.usage.dict() if hasattr(response, "usage") else {}
        )
    except Exception as e:
        logger.error(f"Groq API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Endpoint para streaming de respuestas (opcional)"""
    try:
        Llama_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in request.messages
        ]
        
        response = client.chat.completions.create(
            model=request.model,
            messages=Llama_messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=True
        )
        
        def generate():
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    yield f"data: {chunk.choices[0].delta.content}\n\n"
            yield "data: [DONE]\n\n"
        
        from fastapi.responses import StreamingResponse
        return StreamingResponse(generate(), media_type="text/plain")
        
    except Exception as e:
        logger.error(f"Error in streaming chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in streaming: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)