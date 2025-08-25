import os
import logging
import pandas as pd
from typing import List, Tuple
import torch
from transformers import AutoTokenizer, AutoModel
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================
# Configuración por variables
# ============================
MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MILVUS_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")  # Dentro de Docker backend usa 'milvus'
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = os.getenv("EMBED_COLLECTION", "text_embeddings")
CSV_PATH = os.getenv("EMBED_CSV", os.path.join("..", "data", "wiki_scraped.csv"))
FORCE_REINGEST = os.getenv("FORCE_REINGEST", "false").lower() == "true"
RUN_EXAMPLE = os.getenv("RUN_EXAMPLE", "false").lower() == "true"
BATCH_SIZE = int(os.getenv("EMBED_BATCH", "64"))  # Batch lógico de generación (token limit ~512)
MAX_TEXT_BYTES = int(os.getenv("MAX_TEXT_BYTES", "500"))
INDEX_TYPE = os.getenv("INDEX_TYPE", "IVF_FLAT")
METRIC_TYPE = os.getenv("METRIC_TYPE", "L2")  # Si en el futuro normalizas, puedes usar IP para coseno
NLIST = int(os.getenv("NLIST", "1024"))

logger.info(f"Configuración: MODEL={MODEL_NAME} MILVUS={MILVUS_HOST}:{MILVUS_PORT} CSV={CSV_PATH}")

# ============================
# Carga de modelo de embeddings
# ============================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask

def connect_milvus():
    try:
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        logger.info("Conectado a Milvus correctamente")
    except Exception as e:
        logger.error(f"Error conectando a Milvus: {e}")
        raise

def ensure_collection(dim: int = 384) -> Collection:
    exists = utility.has_collection(COLLECTION_NAME)
    if not exists:
        logger.info(f"Colección '{COLLECTION_NAME}' no existe. Creando...")
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=MAX_TEXT_BYTES)
        ]
        schema = CollectionSchema(fields, description="Embeddings de texto")
        col = Collection(name=COLLECTION_NAME, schema=schema)
        logger.info("Colección creada.")
    else:
        col = Collection(COLLECTION_NAME)
        logger.info(f"Colección '{COLLECTION_NAME}' encontrada.")
    return col

def load_csv(path: str) -> pd.DataFrame:
    logger.info(f"Cargando CSV: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("El CSV está vacío")
    return df

def build_text(row, columns: List[str]) -> str:
    return ". ".join([str(row[c]) for c in columns if c in row and pd.notna(row[c])])

def generate_embeddings_texts(df: pd.DataFrame, text_columns: List[str]) -> Tuple[List[List[float]], List[str], int]:
    embeddings: List[List[float]] = []
    texts: List[str] = []
    truncated = 0
    for _, row in df.iterrows():
        combined = build_text(row, text_columns)
        # Tokenizado + embedding
        inputs = tokenizer(combined, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            pooled = mean_pooling(outputs, inputs['attention_mask']).squeeze().cpu().numpy().tolist()
        # Truncar texto a límite de bytes si hace falta
        b = combined.encode('utf-8')
        if len(b) > MAX_TEXT_BYTES:
            truncated += 1
            combined = b[:MAX_TEXT_BYTES].decode('utf-8', errors='ignore')
        embeddings.append(pooled)
        texts.append(combined)
    return embeddings, texts, truncated

def create_index_if_needed(col: Collection):
    try:
        if col.indexes:
            logger.info("Índice ya existente. Saltando creación.")
            return
    except Exception:
        pass
    index_params = {"index_type": INDEX_TYPE, "metric_type": METRIC_TYPE, "params": {"nlist": NLIST}}
    logger.info(f"Creando índice {INDEX_TYPE} metric={METRIC_TYPE} nlist={NLIST}...")
    col.create_index(field_name="embedding", index_params=index_params)
    logger.info("Índice creado.")

def ingest(csv_path: str, text_columns: List[str], force: bool = False):
    connect_milvus()
    col = ensure_collection(dim=384)
    # Idempotencia simple: si ya hay entidades y no se fuerza, salir.
    existing = 0
    try:
        existing = col.num_entities
    except Exception:
        pass
    if existing > 0 and not force:
        logger.info(f"Colección ya contiene {existing} entidades. Usa FORCE_REINGEST=true para reingestar.")
        return col, existing, 0, 0
    df = load_csv(csv_path)
    embeddings, texts, truncated = generate_embeddings_texts(df, text_columns)
    logger.info(f"Generados {len(embeddings)} embeddings. Truncados {truncated} textos.")
    data = [embeddings, texts]
    insert_res = col.insert(data)
    logger.info(f"Insertadas {len(embeddings)} filas. IDs: rango inicial -> {insert_res.primary_keys[:3]} ...")
    create_index_if_needed(col)
    col.load()
    logger.info("Colección cargada para búsqueda.")
    return col, len(embeddings), truncated, existing

def example_search(col: Collection, query: str, top_k: int = 5):
    inputs = tokenizer(query, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        q_emb = mean_pooling(outputs, inputs['attention_mask']).squeeze().cpu().numpy().tolist()
    params = {"metric_type": METRIC_TYPE, "params": {"nprobe": 10}}
    results = col.search(data=[q_emb], anns_field="embedding", param=params, limit=top_k, output_fields=["text"])
    for i, r in enumerate(results[0]):
        print(f"Resultado {i+1}: distancia={r.distance}\n{r.entity.get('text')[:180]}\n---")

if __name__ == "__main__":
    TEXT_COLUMNS = ["category", "section", "title", "paragraph"]
    col, inserted, truncated, prev = ingest(CSV_PATH, TEXT_COLUMNS, force=FORCE_REINGEST)
    if inserted:
        logger.info(f"Resumen ingestión: nuevos={inserted} previos={prev} truncados={truncated}")
    if RUN_EXAMPLE and inserted:
        logger.info("Ejecutando búsqueda de ejemplo...")
        example_search(col, "¿Cómo utilizar múltiples vistas en participantes?", top_k=5)
