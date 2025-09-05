from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import os
from typing import List, Optional
import logging

# --- nuevos imports para RAG
from transformers import AutoTokenizer, AutoModel
import torch
from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType

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
# Modelo primario configurable y lista de fallback si no existe / acceso denegado
PRIMARY_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
# Lista de modelos vigentes (ajusta según disponibilidad en tu cuenta Groq)
FALLBACK_MODELS = [
    PRIMARY_GROQ_MODEL,
    "llama-3.1-70b-versatile",
]
client = Groq(api_key=GROQ_API_KEY)

"""Modelo de embeddings: carga diferida para reducir RAM inicial.
Se carga la primera vez que se necesita (retrieve o ingest).
"""
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
_tokenizer = None
_model = None

def get_embedding_model():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        logger.info(f"Cargando modelo de embeddings (lazy): {EMBED_MODEL_NAME}")
        _tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
        _model = AutoModel.from_pretrained(EMBED_MODEL_NAME)
    return _tokenizer, _model

# Función de mean pooling igual que en embedding_generator.py

def _mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return (sum_embeddings / sum_mask)

# Conectar a Milvus (host y puerto desde variables de entorno para flexibilidad)
MILVUS_HOST = os.getenv("MILVUS_HOST", "milvus")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
try:
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    logger.info(f"Conectado a Milvus en {MILVUS_HOST}:{MILVUS_PORT}")
except Exception as e:
    logger.warning(f"No se pudo conectar a Milvus en inicio: {e}")

def ensure_text_embeddings_collection(dim: int = 384) -> Collection:
    """Ensure the 'text_embeddings' collection exists in Milvus. If not, create it with a basic schema and an IVF_FLAT index.
    Returns the Collection instance (loaded) or raises an exception on failure.
    """
    try:
        # reconectar si hace falta
        try:
            connections.connect("default")
        except Exception:
            pass

        if not utility.has_collection("text_embeddings"):
            logger.info("Colección 'text_embeddings' no existe. Creando...")
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=500)
            ]
            schema = CollectionSchema(fields, description="Colección de embeddings de texto")
            col = Collection("text_embeddings", schema=schema)

            # Crear índice vectorial
            try:
                index_params = {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 1024}}
                col.create_index(field_name="embedding", index_params=index_params)
                logger.info("Índice creado para 'text_embeddings'")
            except Exception as e:
                logger.warning(f"No se pudo crear índice (puede ser tolerable): {e}")

            col.load()
            logger.info("Colección 'text_embeddings' creada y cargada.")
            return col
        else:
            col = Collection("text_embeddings")
            # Intentar cargar si no está cargada
            try:
                col.load()
            except Exception:
                pass
            return col

    except Exception as e:
        logger.error(f"Fallo asegurando colección text_embeddings: {e}")
        raise

# Modelos Pydantic
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    model: Optional[str] = PRIMARY_GROQ_MODEL
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000

class ChatResponse(BaseModel):
    response: str
    usage: dict

# Nuevos modelos para RAG
class RagRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5
    use_llm: Optional[bool] = True
    model: Optional[str] = PRIMARY_GROQ_MODEL

class RagRetrieveResponse(BaseModel):
    contexts: List[str]
    distances: List[float]

class IngestRequest(BaseModel):
    texts: List[str]

@app.get("/")
async def root():
    return {"message": "Groq Llama 3.1 Chat API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "groq-llama-3.1"}

def _call_groq_chat(model: str, messages: List[dict], temperature: float, max_tokens: int):
    """Intenta llamar al modelo especificado y aplica fallback si el modelo no existe o está deprecado.
    Devuelve (resp, modelo_utilizado) o lanza excepción final.
    """
    tried_errors = []
    candidate_models = [model] + [m for m in FALLBACK_MODELS if m != model]
    for m in candidate_models:
        try:
            logger.info(f"Llamando Groq model={m}")
            resp = client.chat.completions.create(
                model=m,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            if m != model:
                logger.warning(f"Se usó modelo fallback '{m}' (solicitado '{model}').")
            return resp, m
        except Exception as ex:
            err_txt = str(ex)
            tried_errors.append((m, err_txt))
            lower_err = err_txt.lower()
            # Patrones de errores tolerables para intentar siguiente modelo
            if any(pat in lower_err for pat in ["model_not_found", "does not exist", "decommissioned", "model_decommissioned", "invalid_request_error"]):
                logger.warning(f"Modelo {m} no disponible/deprecado ({err_txt}). Probando siguiente...")
                continue
            # Errores no recuperables (auth, rate limit, etc.) se relanzan
            logger.error(f"Error no recuperable con modelo {m}: {err_txt}")
            raise
    # Si todos fallan con errores de 'modelo no disponible'
    summary = "; ".join([f"{m}:{e}" for m, e in tried_errors])
    raise HTTPException(status_code=400, detail=f"Ningún modelo disponible. Errores: {summary}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        groq_messages = [
            {"role": m.role, "content": m.content} for m in request.messages
        ]
        raw_resp, used_model = _call_groq_chat(
            model=request.model or PRIMARY_GROQ_MODEL,
            messages=groq_messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        return ChatResponse(
            response=raw_resp.choices[0].message.content,
            usage=raw_resp.usage.dict() if hasattr(raw_resp, "usage") else {"model": used_model}
        )
    except Exception as e:
        logger.error(f"Groq API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint que solo recupera los contextos desde Milvus
@app.post("/rag/retrieve", response_model=RagRetrieveResponse)
async def rag_retrieve(request: RagRequest):
    try:
        question = request.question
        top_k = request.top_k or 5

        # Asegurar conexión a Milvus
        try:
            connections.connect("default")
        except Exception:
            pass

        # Asegurar que la colección existe (se crea si falta)
        col = ensure_text_embeddings_collection(dim=384)

        # Calcular embedding de la pregunta (carga lazy)
        tokenizer, model = get_embedding_model()
        inputs = tokenizer(question, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            q_emb = _mean_pooling(outputs, inputs['attention_mask']).squeeze().cpu().numpy().tolist()

        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = col.search(
            data=[q_emb],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["text"]
        )

        contexts = []
        distances = []
        for r in results[0]:
            contexts.append(r.entity.get("text"))
            distances.append(r.distance)
        logger.info(f"RAG retrieve: devueltos {len(contexts)}/{top_k} contextos")
        return RagRetrieveResponse(contexts=contexts, distances=distances)

    except Exception as e:
        logger.error(f"RAG retrieve error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint RAG completo: recuperar + llamar a Groq para generar respuesta usando los contextos
@app.post("/rag", response_model=ChatResponse)
async def rag(request: RagRequest):
    try:
        # Recuperar contextos
        retrieve_req = RagRequest(question=request.question, top_k=request.top_k)
        retrieve_res = await rag_retrieve(retrieve_req)

        # Construir prompt con contextos
        context_text = "\n\n--- Contextos relevantes (extraídos del corpus) ---\n\n"
        for i, c in enumerate(retrieve_res.contexts):
            context_text += f"[{i+1}] {c}\n\n"

        system_message = {"role": "system", "content": "Utiliza los siguientes fragmentos de contexto para responder de forma concisa y precisa a la pregunta del usuario. Si la respuesta no está en los contextos, responde indicando que no se encontró información relevante."}
        # Mensajes: primero contexto como system + luego user question
        messages = [system_message, {"role": "user", "content": context_text + "\nPregunta: " + request.question}]

        # Llamar a Groq para generar respuesta
        raw_resp, used_model = _call_groq_chat(
            model=request.model or PRIMARY_GROQ_MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=800
        )
        return ChatResponse(
            response=raw_resp.choices[0].message.content,
            usage=raw_resp.usage.dict() if hasattr(raw_resp, "usage") else {"model": used_model}
        )

    except Exception as e:
        logger.error(f"RAG error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/ingest")
async def rag_ingest(req: IngestRequest):
    """Ingesta una lista de textos en la colección 'text_embeddings'.
    Trunca cada texto a 500 caracteres (schema VARCHAR(500)).
    """
    try:
        if not req.texts:
            raise HTTPException(status_code=400, detail="Lista 'texts' vacía")
        try:
            connections.connect("default")
        except Exception:
            pass
        col = ensure_text_embeddings_collection(dim=384)
        tokenizer, model = get_embedding_model()
        embeddings = []
        stored_texts = []
        for t in req.texts:
            if not t or not t.strip():
                continue
            inputs = tokenizer(t, return_tensors='pt', truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
                emb = _mean_pooling(outputs, inputs['attention_mask']).squeeze().cpu().numpy().tolist()
            embeddings.append(emb)
            stored_texts.append(t[:500])
        if not embeddings:
            raise HTTPException(status_code=400, detail="No se generaron embeddings válidos")
        # Insertar (id auto por schema)
        col.insert([embeddings, stored_texts])
        col.flush()
        return {"inserted": len(embeddings)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingest error: {e}")
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

# Endpoint de diagnóstico para revisar estado del RAG
@app.get("/rag/status")
async def rag_status():
    try:
        info = {}
        try:
            connections.connect("default")
        except Exception:
            pass
        exists = utility.has_collection("text_embeddings")
        info["collection_exists"] = exists
        if exists:
            col = Collection("text_embeddings")
            try:
                count = col.num_entities
            except Exception:
                count = None
            info["num_entities"] = count
            try:
                indexes = col.indexes
                info["indexes"] = [idx.to_dict() if hasattr(idx, 'to_dict') else str(idx) for idx in indexes]
            except Exception as e:
                info["indexes_error"] = str(e)
        info["primary_model"] = PRIMARY_GROQ_MODEL
        info["fallback_models"] = FALLBACK_MODELS
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))