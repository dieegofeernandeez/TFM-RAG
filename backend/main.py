from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import os
from typing import List, Optional, Set
import logging
from contextlib import suppress
import unicodedata
import re

# --- nuevos imports para RAG
from transformers import AutoTokenizer, AutoModel
import torch
from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType
import math

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
PRIMARY_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
_fallback_raw = os.getenv("GROQ_FALLBACK_MODELS", "")
FALLBACK_MODELS = [m.strip() for m in _fallback_raw.split(',') if m.strip()] or ["llama-3.1-70b-versatile"]
GROQ_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.2"))
client = Groq(api_key=GROQ_API_KEY)
MODEL_CANDIDATES = [PRIMARY_GROQ_MODEL] + [m for m in FALLBACK_MODELS if m != PRIMARY_GROQ_MODEL]

"""Modelo de embeddings: carga diferida para reducir RAM inicial.
Se carga la primera vez que se necesita (retrieve o ingest).
"""
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_NORMALIZE = os.getenv("EMBED_NORMALIZE", "false").lower() == "true"
METRIC_TYPE = os.getenv("METRIC_TYPE", "L2")  # Debe coincidir con el índice creado
EMBED_MAX_LENGTH = int(os.getenv("EMBED_MAX_LENGTH", "512"))
IS_BGE_M3 = "bge-m3" in EMBED_MODEL_NAME.lower()

# Re-ranking configurable
RERANK_ENABLE = os.getenv("RERANK_ENABLE", "true").lower() == "true"
RERANK_FETCH_MULT = int(os.getenv("RERANK_FETCH_MULT", "5"))  # multiplicador de candidatos iniciales
RERANK_TITLE_BOOST = float(os.getenv("RERANK_TITLE_BOOST", "0.35"))
RERANK_SECTION_BOOST = float(os.getenv("RERANK_SECTION_BOOST", "0.15"))
RERANK_TEXT_BOOST = float(os.getenv("RERANK_TEXT_BOOST", "0.25"))
RERANK_POSITION_DECAY = float(os.getenv("RERANK_POSITION_DECAY", "0.02"))  # penalización leve por posición tardía en texto (si incluyéramos position)
RERANK_MMR = os.getenv("RERANK_MMR", "true").lower() == "true"
RERANK_MMR_LAMBDA = float(os.getenv("RERANK_MMR_LAMBDA", "0.7"))
NPROBE_SEARCH = int(os.getenv("NPROBE_SEARCH", "48"))
RAG_MAX_TOKENS = int(os.getenv("RAG_MAX_TOKENS", "1400"))
EXPAND_NEIGHBORS = os.getenv("EXPAND_NEIGHBORS", "false").lower() == "true"
NEIGHBOR_RADIUS = int(os.getenv("NEIGHBOR_RADIUS", "1"))
_tokenizer = None
_model = None
_fe_bge_model = None  # FlagEmbedding para BGE-M3

def get_embedding_model():
    """Carga perezosa del modelo de embeddings.
    - Para BGE-M3 usa FlagEmbedding (mismo que el ingestor)
    - Para otros, usa Transformers + mean pooling
    """
    global _tokenizer, _model, _fe_bge_model
    if IS_BGE_M3:
        if _fe_bge_model is None:
            try:
                from FlagEmbedding import BGEM3FlagModel  # type: ignore
            except ImportError as e:
                logger.error("Falta FlagEmbedding para BGE-M3: pip install FlagEmbedding")
                raise
            logger.info("Cargando modelo de embeddings (lazy, FlagEmbedding BGEM3): %s", EMBED_MODEL_NAME)
            use_fp16 = torch.cuda.is_available()
            _fe_bge_model = BGEM3FlagModel(EMBED_MODEL_NAME, use_fp16=use_fp16)
        return None, _fe_bge_model
    else:
        if _tokenizer is None or _model is None:
            logger.info(f"Cargando modelo de embeddings (lazy, Transformers): {EMBED_MODEL_NAME}")
            _tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
            _model = AutoModel.from_pretrained(EMBED_MODEL_NAME)
            _model.eval()
            if torch.cuda.is_available():
                _model.to("cuda")
        return _tokenizer, _model

# Función de mean pooling igual que en embedding_generator.py

def _mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return (sum_embeddings / sum_mask)

def encode_query(text: str) -> List[float]:
    """Codifica la consulta a vector denso list[float] usando el modelo configurado."""
    if IS_BGE_M3:
        _, fe_model = get_embedding_model()
        out = fe_model.encode([text], batch_size=1, max_length=min(EMBED_MAX_LENGTH, 384))
        vec = out["dense_vecs"][0]
        import numpy as np
        v = np.asarray(vec, dtype=np.float32)
        if EMBED_NORMALIZE:
            n = float(np.linalg.norm(v))
            if n > 0.0:
                v = v / n
        return v.astype(np.float32).tolist()
    else:
        tokenizer, model = get_embedding_model()
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=EMBED_MAX_LENGTH)
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            q_vec = _mean_pooling(outputs, inputs['attention_mask'])
            if EMBED_NORMALIZE:
                q_vec = torch.nn.functional.normalize(q_vec, p=2, dim=1)
            v = q_vec.squeeze().cpu().numpy().astype('float32').tolist()
        return v

# Conectar a Milvus (host y puerto desde variables de entorno para flexibilidad)
MILVUS_HOST = os.getenv("MILVUS_HOST", "milvus")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
try:
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    logger.info(f"Conectado a Milvus en {MILVUS_HOST}:{MILVUS_PORT}")
except Exception as e:
    logger.warning(f"No se pudo conectar a Milvus en inicio: {e}")

# --- Utilidades de texto para el reranker ---
def strip_accents(s: str) -> str:
    try:
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    except Exception:
        return s

STOPWORDS_ES: Set[str] = {
    # Lista breve de stopwords comunes en español (se puede ampliar)
    'el','la','los','las','un','una','unos','unas','y','o','u','de','del','al','a','en','por','para','con','sin','es','son','ser','se','que','qué','como','cómo','cuando','cuándo','donde','dónde','cual','cuál','cuales','cuáles','cuanto','cuánto','cuanta','cuánta','cuantos','cuántos','muy','mas','más','pero','si','sí','no','ya','lo','su','sus','mi','mis','tu','tus','su','sus','nuestro','nuestra','nuestros','nuestras','este','esta','estos','estas','ese','esa','esos','esas','aqui','aquí','alli','allí','ahi','ahí','debe','deben','debes','debo','hay','haber','he','ha','han','hace','hacen','hacer'
}

def tokenize_es(text: str) -> List[str]:
    t = strip_accents(text.lower())
    # separar por no-letras/números
    parts = re.split(r"[^a-z0-9áéíóúñü]+", t)
    return [p for p in parts if len(p) > 2 and p not in STOPWORDS_ES]

def jaccard_set(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    union = len(a | b)
    if union == 0:
        return 0.0
    return inter / union

def ensure_text_embeddings_collection(dim: int = 384) -> Collection:
    """Asegura la colección con esquema extendido (text + metadatos), índice y carga en memoria.

    - Evita llamar load() antes de crear el índice (para no disparar "index doesn't exist").
    - Si no hay índice, lo crea y espera brevemente a que termine la construcción.
    - Intenta cargar la colección; si falla, registra y devuelve igualmente el handler (el caller podrá decidir).
    """
    try:
        with suppress(Exception):
            connections.connect("default")
        name = "text_embeddings"
        if not utility.has_collection(name):
            logger.info("Colección 'text_embeddings' no existe. Creando (extendida)...")
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000),
                FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=400),
                FieldSchema(name="position", dtype=DataType.INT64),
                FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=200),
                FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=80),
                FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=120),
                FieldSchema(name="source_type", dtype=DataType.VARCHAR, max_length=32),
                FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=400),
            ]
            schema = CollectionSchema(fields, description="Embeddings con metadatos unificados")
            col = Collection(name, schema=schema)
            # Crear índice y cargar
            try:
                index_params = {"index_type": "IVF_FLAT", "metric_type": METRIC_TYPE, "params": {"nlist": 1024}}
                col.create_index(field_name="embedding", index_params=index_params)
            except Exception as e:
                logger.warning(f"No se pudo crear índice inicial: {e}")
            with suppress(Exception):
                col.load()
            return col

        # Colección existente
        col = Collection(name)
        existing = {f.name for f in col.schema.fields}
        expected = {"text", "title", "category", "section", "source_type", "url", "doc_id", "position"}
        if not expected.issubset(existing):
            logger.warning("Colección existente sin todos los campos extendidos (%s). Reingesta recomendada.", expected - existing)

        # Asegurar índice en 'embedding'
        need_index = False
        try:
            idx_list = getattr(col, "indexes", [])
            need_index = not idx_list
        except Exception:
            need_index = True
        if need_index:
            try:
                index_params = {"index_type": "IVF_FLAT", "metric_type": METRIC_TYPE, "params": {"nlist": 1024}}
                col.create_index(field_name="embedding", index_params=index_params)
            except Exception as e:
                logger.warning(f"No se pudo crear índice en colección existente: {e}")

        # Esperar un poco a que termine el index build (rápido si está vacía)
        try:
            from time import sleep
            for _ in range(10):  # ~1s total
                try:
                    prog = utility.index_building_progress(name)
                    total = int(prog.get("total_rows", 0))
                    indexed = int(prog.get("indexed_rows", 0))
                    if total == 0 or indexed >= total:
                        break
                except Exception:
                    break
                sleep(0.1)
        except Exception:
            pass

        # Cargar la colección
        with suppress(Exception):
            col.load()
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
    # overrides opcionales por petición (si no se envían, se usan los ENV)
    expand_neighbors: Optional[bool] = None
    neighbor_radius: Optional[int] = None

class RetrievedContext(BaseModel):
    text: str
    title: Optional[str] = None
    category: Optional[str] = None
    section: Optional[str] = None
    source_type: Optional[str] = None
    distance: float
    score: float
    url: Optional[str] = None

class RagRetrieveResponse(BaseModel):
    contexts: List[RetrievedContext]

class IngestRequest(BaseModel):
    texts: List[str]

@app.get("/")
async def root():
    return {"message": "Groq Llama 3.1 Chat API is running"}

@app.get("/health")
async def health_check():
    """Health extendido: muestra modelos y estado de Milvus/colección."""
    info = {"status": "healthy", "service": "api-backend"}
    info["llm_primary"] = PRIMARY_GROQ_MODEL
    info["llm_fallbacks"] = FALLBACK_MODELS
    info["embed_model"] = EMBED_MODEL_NAME
    info["embed_normalize"] = EMBED_NORMALIZE
    info["metric_type"] = METRIC_TYPE
    info["nprobe_search"] = NPROBE_SEARCH
    info["rerank"] = {
        "enable": RERANK_ENABLE,
        "fetch_mult": RERANK_FETCH_MULT,
        "title_boost": RERANK_TITLE_BOOST,
        "section_boost": RERANK_SECTION_BOOST,
        "text_boost": RERANK_TEXT_BOOST,
        "mmr": RERANK_MMR,
        "mmr_lambda": RERANK_MMR_LAMBDA,
    }
    # Milvus / colección
    try:
        try:
            connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        except Exception:
            pass
        info["milvus_host"] = MILVUS_HOST
        info["milvus_port"] = MILVUS_PORT
        exists = utility.has_collection("text_embeddings")
        info["collection_exists"] = exists
        if exists:
            with suppress(Exception):
                col = Collection("text_embeddings")
                with suppress(Exception):
                    info["num_entities"] = col.num_entities
                with suppress(Exception):
                    # primera dimensión del campo embedding
                    emb_field = next(f for f in col.schema.fields if f.name == "embedding")
                    info["collection_dim"] = emb_field.params.get("dim")
    except Exception as e:
        info["milvus_error"] = str(e)
    return info

def call_groq_with_fallback(messages: List[dict], max_tokens: int, temperature: float | None = None, primary: str | None = None):
    """Itera modelos (primario + fallbacks) hasta obtener respuesta válida."""
    temp = temperature if temperature is not None else GROQ_TEMPERATURE
    ordered = [primary] if primary else []
    ordered += [m for m in MODEL_CANDIDATES if m and m not in ordered]
    errors = []
    for m in ordered:
        try:
            logger.info(f"Groq→ intentando modelo='{m}'")
            resp = client.chat.completions.create(
                model=m,
                messages=messages,
                temperature=temp,
                max_tokens=max_tokens
            )
            if m != (primary or PRIMARY_GROQ_MODEL):
                logger.warning(f"Usado fallback '{m}'")
            return resp, m
        except Exception as ex:
            msg = str(ex)
            lower = msg.lower()
            errors.append(f"{m}:{msg}")
            if any(p in lower for p in ["model_not_found", "does not exist", "decommissioned", "model_decommissioned", "invalid_request_error"]):
                logger.warning(f"Modelo {m} no disponible ({msg}). Siguiente...")
                continue
            # Errores no recuperables
            logger.error(f"Error no recuperable con {m}: {msg}")
            raise
    raise HTTPException(status_code=400, detail="; ".join(errors) or "No modelos válidos")

def _usage_to_dict(raw_resp, used_model: str) -> dict:
    """Best-effort conversion of Groq usage payload to plain dict without Pydantic v2 deprecation warnings."""
    try:
        usage = getattr(raw_resp, "usage", None)
        if usage is None:
            return {"model": used_model}
        # Prefer Pydantic v2 API
        if hasattr(usage, "model_dump"):
            return usage.model_dump()
        # Fallback to v1 API if present
        if hasattr(usage, "dict"):
            return usage.dict()
        # As a last resort, stringify
        return {"model": used_model, "raw": str(usage)}
    except Exception:
        return {"model": used_model}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        groq_messages = [
            {"role": m.role, "content": m.content} for m in request.messages
        ]
        raw_resp, used_model = call_groq_with_fallback(
            groq_messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            primary=request.model or PRIMARY_GROQ_MODEL
        )
        return ChatResponse(
            response=raw_resp.choices[0].message.content,
            usage=_usage_to_dict(raw_resp, used_model)
        )
    except Exception as e:
        logger.error(f"Groq API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint que solo recupera los contextos desde Milvus
@app.post("/rag/retrieve", response_model=RagRetrieveResponse)
async def rag_retrieve(request: RagRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Pregunta vacía")
    top_k = request.top_k or 5

    with suppress(Exception):
        connections.connect("default")
    col = ensure_text_embeddings_collection(dim=1024 if IS_BGE_M3 else 384)
    q_emb = encode_query(question)

    fetch_limit = top_k * RERANK_FETCH_MULT if RERANK_ENABLE else top_k
    search_params = {"metric_type": METRIC_TYPE, "params": {"nprobe": NPROBE_SEARCH}}
    # Determinar output_fields disponibles (compatibilidad con colecciones antiguas sin 'url')
    available_fields = {f.name for f in col.schema.fields}
    wanted_fields = ["text", "title", "category", "section", "source_type"]
    if "url" in available_fields:
        wanted_fields.append("url")
    if "doc_id" in available_fields:
        wanted_fields.append("doc_id")
    if "position" in available_fields:
        wanted_fields.append("position")

    # Si la colección está vacía, devolvemos sin contextos para evitar error Milvus
    try:
        if getattr(col, 'num_entities', 0) == 0:
            logger.info("Colección vacía: devolviendo sin contextos")
            return RagRetrieveResponse(contexts=[])
    except Exception:
        pass

    # Intentar cargar la colección antes de buscar
    with suppress(Exception):
        col.load()

    # Si sigue vacía o no cargada, devolver vacío
    try:
        if getattr(col, 'num_entities', 0) == 0:
            logger.info("Colección vacía o no cargada: devolviendo sin contextos")
            return RagRetrieveResponse(contexts=[])
    except Exception:
        return RagRetrieveResponse(contexts=[])

    # Ejecutar búsqueda, capturando errores de carga/índice
    try:
        results = col.search(
            data=[q_emb],
            anns_field="embedding",
            param=search_params,
            limit=fetch_limit,
            output_fields=wanted_fields
        )
    except Exception as e:
        logger.error(f"Milvus search error: {e}")
        return RagRetrieveResponse(contexts=[])

    raw = []
    for r in results[0]:
        ent = r.entity
        item = {
            "text": ent.get("text"),
            "title": ent.get("title") or "",
            "category": ent.get("category") or "",
            "section": ent.get("section") or "",
            "source_type": ent.get("source_type") or "",
            "distance": r.distance
        }
        if "url" in available_fields:
            item["url"] = ent.get("url") or ""
        if "doc_id" in available_fields:
            item["doc_id"] = ent.get("doc_id") or ""
        if "position" in available_fields:
            try:
                item["position"] = int(ent.get("position"))
            except Exception:
                item["position"] = None
        # tokenizar para posibles MMR/diversidad
        try:
            item["_tokens"] = set(tokenize_es(item["text"] or ""))
        except Exception:
            item["_tokens"] = set()
        raw.append(item)

    # Re-ranking ligero
    if RERANK_ENABLE and raw:
        q_terms = tokenize_es(question)
        def term_match_score(text: str, terms: List[str]) -> float:
            if not text:
                return 0.0
            ltoks = set(tokenize_es(text))
            if not ltoks or not terms:
                return 0.0
            hits = sum(1 for t in terms if t in ltoks)
            return hits / max(1, len(terms))
        reranked = []
        for item in raw:
            base_sim = item['distance'] if METRIC_TYPE.upper() == 'IP' else -item['distance']
            title_bonus = term_match_score(item['title'], q_terms) * RERANK_TITLE_BOOST
            section_bonus = term_match_score(item['section'], q_terms) * RERANK_SECTION_BOOST
            text_bonus = term_match_score(item['text'], q_terms) * RERANK_TEXT_BOOST
            length_penalty = 0.0
            l = len(item['text']) if item['text'] else 0
            if l > 950:
                length_penalty = 0.05
            score = base_sim + title_bonus + section_bonus + text_bonus - length_penalty
            item['score'] = score
            reranked.append(item)
        reranked.sort(key=lambda x: x['score'], reverse=True)
        # Diversidad MMR opcional basada en Jaccard de tokens
        if RERANK_MMR:
            selected_mm: List[dict] = []
            pool = reranked.copy()
            while pool and len(selected_mm) < top_k:
                if not selected_mm:
                    selected_mm.append(pool.pop(0))
                    continue
                best_idx = -1
                best_mmr = -1e9
                for i, cand in enumerate(pool[: min(len(pool), 200) ]):
                    sim = 0.0
                    for s in selected_mm:
                        sim = max(sim, jaccard_set(cand.get("_tokens", set()), s.get("_tokens", set())))
                    mmr_score = RERANK_MMR_LAMBDA * cand['score'] - (1.0 - RERANK_MMR_LAMBDA) * sim
                    if mmr_score > best_mmr:
                        best_mmr = mmr_score
                        best_idx = i
                if best_idx >= 0:
                    selected_mm.append(pool.pop(best_idx))
                else:
                    break
            selected = selected_mm
        else:
            selected = reranked[:top_k]
    else:
        for item in raw:
            item['score'] = item['distance'] if METRIC_TYPE.upper() == 'IP' else -item['distance']
        selected = raw[:top_k]

    # Determinar overrides por petición o usar ENV
    expand_neighbors = EXPAND_NEIGHBORS if request.expand_neighbors is None else bool(request.expand_neighbors)
    neighbor_radius = NEIGHBOR_RADIUS if request.neighbor_radius is None else int(request.neighbor_radius)

    # Expansión de vecinos contiguos por doc_id/position (si existe en schema y está activado)
    if expand_neighbors and selected and {"doc_id", "position"}.issubset(available_fields):
        try:
            with suppress(Exception):
                col.load()
            extra: List[dict] = []
            for it in selected:
                did = it.get("doc_id")
                pos = it.get("position")
                if not did or pos is None:
                    continue
                radius = max(1, neighbor_radius)
                # construir lista de posiciones vecinas
                neigh = [pos + d for d in range(-radius, radius + 1) if d != 0]
                if not neigh:
                    continue
                safe_did = did.replace("'", "\\'")
                expr = f"doc_id == '{safe_did}' and position in {neigh}"
                try:
                    neigh_rows = col.query(expr=expr, output_fields=wanted_fields)
                except Exception:
                    neigh_rows = []
                for nr in neigh_rows:
                    extra.append({
                        "text": nr.get("text"),
                        "title": nr.get("title") or "",
                        "category": nr.get("category") or "",
                        "section": nr.get("section") or "",
                        "source_type": nr.get("source_type") or "",
                        "distance": it.get("distance", 0.0),
                        "score": it.get("score", 0.0) * 0.98,  # leve degradación por ser vecino
                        "url": nr.get("url") or "",
                        "doc_id": nr.get("doc_id") or did,
                        "position": nr.get("position", None),
                        "_tokens": set(tokenize_es(nr.get("text") or ""))
                    })
            # fusionar y deduplicar por (doc_id, position, text)
            seen_keys = set()
            fused: List[dict] = []
            for it in (selected + extra):
                k = (it.get("doc_id"), it.get("position"), it.get("text"))
                if k in seen_keys:
                    continue
                seen_keys.add(k)
                fused.append(it)
            # ordenar por score y recortar a top_k (presupuesto)
            fused.sort(key=lambda x: x.get('score', 0.0), reverse=True)
            selected = fused[:top_k]
        except Exception:
            pass

    contexts: List[RetrievedContext] = [
        RetrievedContext(
            text=i['text'],
            title=i['title'],
            category=i['category'],
            section=i['section'],
            source_type=i['source_type'],
            distance=i['distance'],
            score=i['score'],
            url=i.get('url') or None
        ) for i in selected
    ]
    logger.info(f"RAG retrieve: candidatos={len(raw)} devueltos={len(contexts)} rerank={'on' if RERANK_ENABLE else 'off'}")
    return RagRetrieveResponse(contexts=contexts)

# Endpoint RAG completo: recuperar + llamar a Groq para generar respuesta usando los contextos
@app.post("/rag", response_model=ChatResponse)
async def rag(request: RagRequest):
    try:
        # Recuperar contextos
        retrieve_req = RagRequest(
            question=request.question,
            top_k=request.top_k,
            expand_neighbors=request.expand_neighbors,
            neighbor_radius=request.neighbor_radius,
        )
        retrieve_res = await rag_retrieve(retrieve_req)
        # Si no hay contextos, devolver respuesta amable sin LLM para evitar 500 y explicar el estado
        if not retrieve_res.contexts:
            msg = (
                "No he encontrado contextos relevantes en la base de conocimiento aún. "
                "Es posible que la colección esté vacía o la reingestión no haya finalizado. "
                "Intenta de nuevo en unos minutos."
            )
            return ChatResponse(response=msg, usage={"model": request.model or PRIMARY_GROQ_MODEL, "note": "no_contexts"})
        context_text = "\n\n--- Contextos relevantes (cita siempre las fuentes con su URL) ---\n\n"
        for i, c in enumerate(retrieve_res.contexts):
            source_url = f" | URL: {c.url}" if c.url else ""
            meta = f"{c.title or 'Sin título'} ({c.category or '-' } > {c.section or '-'}) [{c.source_type or '-'}]{source_url} dist={c.distance:.4f} score={c.score:.4f}"
            context_text += f"[{i+1}] {meta}\n{c.text}\n\n"

        system_message = {"role": "system", "content": (
            "Eres un asistente que responde de forma clara, completa y bien estructurada, y siempre cita las fuentes. "
            "Responde solo con información presente en los contextos. "
            "Estructura la respuesta con secciones y listas cuando ayude a la comprensión. "
            "Incluye una sección 'Fuentes' al final con las URLs de los fragmentos utilizados. "
            "Si no hay información suficiente, dilo claramente."
        )}
        messages = [system_message, {"role": "user", "content": context_text + "\nPregunta: " + request.question}]

        try:
            raw_resp, used_model = call_groq_with_fallback(
                messages,
                max_tokens=RAG_MAX_TOKENS,
                temperature=0.0,
                primary=request.model or PRIMARY_GROQ_MODEL
            )
        except Exception as ex:
            # Devolver respuesta controlada si falla el LLM (p. ej., sin red o clave)
            fallback = (
                "No he podido generar la respuesta con el LLM ahora mismo. "
                "Aun así, he recuperado fragmentos relevantes arriba que pueden ayudarte. "
                "Vuelve a intentarlo más tarde."
            )
            return ChatResponse(response=fallback, usage={"error": str(ex)})
        answer = raw_resp.choices[0].message.content
        # Adjuntar sección de fuentes con URLs (si existen)
        try:
            unique_sources = []
            seen = set()
            for c in retrieve_res.contexts:
                if c.url and c.url not in seen:
                    title = c.title or "Fuente"
                    unique_sources.append(f"- {title}: {c.url}")
                    seen.add(c.url)
            if unique_sources:
                answer = answer.rstrip() + "\n\nFuentes:\n" + "\n".join(unique_sources)
        except Exception:
            pass
        return ChatResponse(response=answer, usage=_usage_to_dict(raw_resp, used_model))
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
            model=request.model or PRIMARY_GROQ_MODEL,
            messages=Llama_messages,
            temperature=request.temperature if request.temperature is not None else GROQ_TEMPERATURE,
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