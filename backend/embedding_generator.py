import os
import logging
import json
from typing import List
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================
# Configuración
# ============================
MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2").strip()
MILVUS_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = os.getenv("EMBED_COLLECTION", "text_embeddings")
# Dentro del contenedor sólo se monta /app (backend) y /data (read-only con corpus preparado)
# Usar /data/prepared_corpus.jsonl como default para evitar rutas relativas inválidas
JSONL_PATH = os.getenv("EMBED_JSONL", "/data/prepared_corpus.jsonl")
FORCE_REINGEST = os.getenv("FORCE_REINGEST", "false").lower() == "true"
BATCH_SIZE = int(os.getenv("EMBED_BATCH", "64"))
MARKER_PATH = os.getenv("INGEST_MARKER_FILE", ".ingest_marker")
INDEX_TYPE = os.getenv("INDEX_TYPE", "IVF_FLAT")
METRIC_TYPE = os.getenv("METRIC_TYPE", "IP")  # Usar IP (inner product) para embeddings normalizados (similitud coseno)
NLIST = int(os.getenv("NLIST", "1024"))
RUN_EXAMPLE = os.getenv("RUN_EXAMPLE", "false").lower() == "true"
EMBED_NORMALIZE = os.getenv("EMBED_NORMALIZE", "true").lower() == "true"
EMBED_MAX_LENGTH = int(os.getenv("EMBED_MAX_LENGTH", "512"))  # menor => más rápido en CPU
EMBED_LIMIT = int(os.getenv("EMBED_LIMIT", "0"))  # 0 = sin límite

logger.info(f"Configuración: MODEL={MODEL_NAME} MILVUS={MILVUS_HOST}:{MILVUS_PORT} JSONL={JSONL_PATH} FORCE_REINGEST={FORCE_REINGEST}")


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu" and BATCH_SIZE > 16:
    logger.info("Reduciendo BATCH_SIZE por CPU (<=16 recomendado)")
    BATCH_SIZE = 16

IS_BGE_M3 = "bge-m3" in MODEL_NAME.lower()

tokenizer = None  # para modelos HF estándar
model = None

def load_model():
    global tokenizer, model
    if model is not None:
        return
    if IS_BGE_M3:
        try:
            from FlagEmbedding import BGEM3FlagModel  # type: ignore
        except ImportError:
            logger.error("Falta FlagEmbedding. Añade 'FlagEmbedding' a requirements.txt")
            raise
        logger.info("Cargando BGEM3FlagModel ...")
        model = BGEM3FlagModel(MODEL_NAME, use_fp16=(DEVICE == "cuda"))
        logger.info("Modelo BGEM3 cargado")
    else:
        from transformers import AutoTokenizer, AutoModel
        logger.info(f"Cargando modelo HF genérico: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME)
        model.eval()
        if DEVICE == "cuda":
            model.to(DEVICE)
        logger.info(f"Modelo HF listo en {DEVICE}")

load_model()

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    pooled = (token_embeddings * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)
    if EMBED_NORMALIZE:
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
    return pooled

def connect_milvus():
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

def drop_collection_if_needed():
    if FORCE_REINGEST and utility.has_collection(COLLECTION_NAME):
        logger.info(f"FORCE_REINGEST=true → eliminando colección existente '{COLLECTION_NAME}'")
        utility.drop_collection(COLLECTION_NAME)

def ensure_collection(dim: int) -> Collection:
    if not utility.has_collection(COLLECTION_NAME):
        logger.info(f"Creando colección '{COLLECTION_NAME}' (dim={dim})")
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
        schema = CollectionSchema(fields, description="Embeddings con metadatos unificados (JSONL)")
        col = Collection(name=COLLECTION_NAME, schema=schema)
    else:
        col = Collection(COLLECTION_NAME)
        logger.info(f"Colección '{COLLECTION_NAME}' existente.")
    return col

def create_index_if_needed(col: Collection):
    if col.indexes:
        logger.info("Índice ya existe. Saltando.")
        return
    index_params = {"index_type": INDEX_TYPE, "metric_type": METRIC_TYPE, "params": {"nlist": NLIST}}
    logger.info(f"Creando índice {INDEX_TYPE} metric={METRIC_TYPE} nlist={NLIST}")
    col.create_index("embedding", index_params)

def read_jsonl(path: str) -> List[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe JSONL: {path}")
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                logger.warning(f"Línea {i} inválida: {e}")
    logger.info(f"Leídos {len(rows)} chunks desde JSONL")
    return rows

def batch(iterable, size: int):
    for i in range(0, len(iterable), size):
        yield iterable[i:i+size]

def embed_texts(texts: List[str]) -> List[List[float]]:
    vectors: List[List[float]] = []
    if IS_BGE_M3:
        # BGEM3FlagModel soporta batch interno
        processed = 0
        total = len(texts)
        for blk in batch(texts, BATCH_SIZE):
            # Usar parámetros compatibles con versiones antiguas de FlagEmbedding
            out = model.encode(
                blk,
                batch_size=len(blk),
                max_length=EMBED_MAX_LENGTH,
            )
            dense = out["dense_vecs"]
            # Asegurar formato list[list[float]] (no ndarrays) y normalización opcional
            dense_list: List[List[float]] = []
            for v in dense:
                vv = np.asarray(v, dtype=np.float32)
                if EMBED_NORMALIZE:
                    n = float(np.linalg.norm(vv))
                    if n > 0.0:
                        vv = vv / n
                dense_list.append(vv.astype(np.float32).tolist())
            vectors.extend(dense_list)
            processed += len(blk)
            if processed % max(1, BATCH_SIZE * 10) == 0 or processed == total:
                logger.info(f"Progreso embeddings: {processed}/{total}")
    else:
        for blk in batch(texts, BATCH_SIZE):
            inputs = tokenizer(blk, return_tensors='pt', truncation=True, padding=True, max_length=512)
            if DEVICE == "cuda":
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                pooled = mean_pooling(outputs, inputs['attention_mask'])
            pooled_cpu = pooled.cpu().numpy().astype(np.float32)
            vectors.extend(pooled_cpu.tolist())
    return vectors

def already_ingested() -> bool:
    if FORCE_REINGEST:
        return False
    if not os.path.exists(MARKER_PATH):
        return False
    try:
        with open(MARKER_PATH, "r", encoding="utf-8") as f:
            content = f.read().strip()
        return content == MODEL_NAME
    except Exception:
        return False

def set_marker():
    try:
        with open(MARKER_PATH, "w", encoding="utf-8") as f:
            f.write(MODEL_NAME)
    except Exception as e:
        logger.warning(f"No se pudo escribir marker: {e}")

def ingest_jsonl(path: str):
    connect_milvus()
    drop_collection_if_needed()

    # Determinar dimensión
    if IS_BGE_M3:
        probe = model.encode(
            ["test"],
            batch_size=1,
            max_length=min(EMBED_MAX_LENGTH, 64),
        )["dense_vecs"][0]
        dim = len(probe)
    else:
        tmp_inputs = tokenizer("test", return_tensors='pt')
        with torch.no_grad():
            tmp_out = model(**tmp_inputs)
            dim = mean_pooling(tmp_out, tmp_inputs['attention_mask']).shape[1]

    col = ensure_collection(dim=dim)

    # Descubrir max_length reales del schema para truncar de forma segura
    def _get_max_len(field_name: str, default: int) -> int:
        try:
            f = next(ff for ff in col.schema.fields if ff.name == field_name)
            ml = getattr(f, "max_length", None)
            if isinstance(ml, int) and ml > 0:
                return ml
            # Algunos backends exponen en params
            ml2 = f.params.get("max_length") if hasattr(f, "params") else None
            if isinstance(ml2, int) and ml2 > 0:
                return ml2
        except Exception:
            pass
        return default

    TEXT_MAX = _get_max_len("text", 1000)
    TITLE_MAX = _get_max_len("title", 200)
    CATEGORY_MAX = _get_max_len("category", 80)
    SECTION_MAX = _get_max_len("section", 120)
    SOURCE_MAX = _get_max_len("source_type", 32)
    URL_MAX = _get_max_len("url", 400)
    DOCID_MAX = _get_max_len("doc_id", 400)

    # Recorte por bytes UTF-8: Milvus puede validar max_length por bytes, no por caracteres
    def _trim_to_max_bytes(s: object, max_bytes: int) -> str:
        if s is None:
            return ""
        t = str(s)
        if len(t.encode("utf-8")) <= max_bytes:
            return t
        lo, hi = 0, len(t)
        best = ""
        while lo <= hi:
            mid = (lo + hi) // 2
            cand = t[:mid]
            if len(cand.encode("utf-8")) <= max_bytes:
                best = cand
                lo = mid + 1
            else:
                hi = mid - 1
        return best

    existing = col.num_entities
    if already_ingested() and existing > 0:
        logger.info(f"Marcador indica ya ingerido con este modelo y colección tiene {existing} entidades. (FORCE_REINGEST=false) → skip.")
        return col, 0, existing
    if existing > 0 and not FORCE_REINGEST and not already_ingested():
        logger.info(f"Colección ya tiene {existing} entidades. (FORCE_REINGEST=false, sin marker) → no se ingesta.")
        return col, 0, existing

    rows = read_jsonl(path)
    if not rows:
        logger.warning("No hay filas para ingerir.")
        return col, 0, existing

    # Primer recorte por bytes (seguro para UTF-8)
    texts = [_trim_to_max_bytes(r.get("text", ""), TEXT_MAX) for r in rows]
    titles = [_trim_to_max_bytes(r.get("title", ""), TITLE_MAX) for r in rows]
    categories = [_trim_to_max_bytes(r.get("category", ""), CATEGORY_MAX) for r in rows]
    sections = [_trim_to_max_bytes(r.get("section", ""), SECTION_MAX) for r in rows]
    source_types = [_trim_to_max_bytes(r.get("source_type", ""), SOURCE_MAX) for r in rows]
    urls = [_trim_to_max_bytes(r.get("url", ""), URL_MAX) for r in rows]
    # doc_id preferentemente la URL; si no hay, usar combinación estable (source_type + title)
    def _mk_doc_id(r: dict) -> str:
        u = str(r.get("url", "") or "").strip()
        if u:
            return u
        st = str(r.get("source_type", "") or "").strip()
        ti = str(r.get("title", "") or "").strip()
        base = f"{st}|{ti}" if (st or ti) else str(r.get("id", ""))
        return base
    doc_ids = [_trim_to_max_bytes(_mk_doc_id(r), DOCID_MAX) for r in rows]
    positions = [int(r.get("position", 0) or 0) for r in rows]

    # Segundo pase de seguridad y métricas de recorte
    def _enforce_and_count_bytes(items: list[str], max_bytes: int, label: str) -> list[str]:
        trimmed = 0
        out: list[str] = []
        for s in items:
            ss = s if isinstance(s, str) else str(s)
            if len(ss.encode("utf-8")) > max_bytes:
                ss = _trim_to_max_bytes(ss, max_bytes)
                trimmed += 1
            out.append(ss)
        if trimmed:
            logger.info(f"{label}: recortados {trimmed} elementos a max {max_bytes} bytes")
        return out

    texts = _enforce_and_count_bytes(texts, TEXT_MAX, "text")
    titles = _enforce_and_count_bytes(titles, TITLE_MAX, "title")
    categories = _enforce_and_count_bytes(categories, CATEGORY_MAX, "category")
    sections = _enforce_and_count_bytes(sections, SECTION_MAX, "section")
    source_types = _enforce_and_count_bytes(source_types, SOURCE_MAX, "source_type")
    urls = _enforce_and_count_bytes(urls, URL_MAX, "url")
    doc_ids = _enforce_and_count_bytes(doc_ids, DOCID_MAX, "doc_id")

    if EMBED_LIMIT and EMBED_LIMIT > 0:
        logger.info(f"EMBED_LIMIT activo → procesando solo los primeros {EMBED_LIMIT} registros")
        texts = texts[:EMBED_LIMIT]
        titles = titles[:EMBED_LIMIT]
        categories = categories[:EMBED_LIMIT]
        sections = sections[:EMBED_LIMIT]
        source_types = source_types[:EMBED_LIMIT]
    urls = urls[:EMBED_LIMIT]
    doc_ids = doc_ids[:EMBED_LIMIT]
    positions = positions[:EMBED_LIMIT]

    logger.info("Generando embeddings...")
    embeddings = embed_texts(texts)
    logger.info("Embeddings listos. Insertando en Milvus...")

    # Construir lista de filas (row-based) para evitar ambigüedad de tipos
    rows_to_insert = []
    for i in range(len(texts)):
        rows_to_insert.append({
            "embedding": embeddings[i],  # list[float]
            "text": texts[i],
            "doc_id": doc_ids[i] if i < len(doc_ids) else "",
            "position": int(positions[i] if i < len(positions) else 0),
            "title": titles[i] if i < len(titles) else "",
            "category": categories[i] if i < len(categories) else "",
            "section": sections[i] if i < len(sections) else "",
            "source_type": source_types[i] if i < len(source_types) else "",
            "url": urls[i] if i < len(urls) else "",
        })
    insert_res = col.insert(rows_to_insert)
    logger.info(f"Insertadas {len(texts)} filas. Ejemplo IDs: {insert_res.primary_keys[:3]}")
    set_marker()
    create_index_if_needed(col)
    col.load()
    logger.info("Colección lista para búsquedas.")
    return col, len(texts), existing

def example_search(col: Collection, query: str, top_k: int = 5):
    inputs = tokenizer(query, return_tensors='pt', truncation=True, padding=True, max_length=512)
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
        q_emb = mean_pooling(out, inputs['attention_mask']).squeeze().cpu().numpy().tolist()
    params = {"metric_type": METRIC_TYPE, "params": {"nprobe": 10}}
    results = col.search([q_emb], "embedding", params, limit=top_k,
                         output_fields=["text", "title", "category", "section", "source_type"])
    for i, r in enumerate(results[0]):
        md = r.entity
        print(f"[{i+1}] dist={r.distance:.4f} {md.get('title')} ({md.get('category')} > {md.get('section')}) [{md.get('source_type')}]\n{md.get('text')[:220]}\n---")

if __name__ == "__main__":
    col, inserted, prev = ingest_jsonl(JSONL_PATH)
    if inserted:
        logger.info(f"Resumen ingestión: nuevos={inserted} previos_existían={prev}")
    if RUN_EXAMPLE and inserted:
        example_search(col, "cómo acceder a la plataforma de formación", top_k=5)
