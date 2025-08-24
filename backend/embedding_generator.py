import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import os
import logging
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from pymilvus import utility

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modelo preentrenado para embeddings
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# Conectar con Milvus
connections.connect("default", host="127.0.0.1", port="19530")

# Definir el esquema de la colección
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=500)
]
schema = CollectionSchema(fields, description="Colección para almacenar embeddings y textos")

# Crear o cargar la colección
collection_name = "text_embeddings"
if not utility.has_collection(collection_name):
    collection = Collection(name=collection_name, schema=schema)
    logger.info(f"Colección '{collection_name}' creada.")
else:
    collection = Collection(name=collection_name)
    logger.info(f"Colección '{collection_name}' cargada.")


def load_csv(file_path):
    """Cargar datos desde un archivo CSV."""
    try:
        logger.info(f"Cargando datos desde {file_path}...")
        return pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Error cargando el archivo CSV: {e}")
        raise


def _mean_pooling(model_output, attention_mask):
    """Mean pooling, taking into account the attention mask."""
    token_embeddings = model_output.last_hidden_state  # (batch_size, seq_len, dim)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return (sum_embeddings / sum_mask)


def generate_embeddings(dataframe, text_columns):
    """Generar embeddings a partir de un DataFrame."""
    embeddings = []
    combined_texts = []

    for _, row in dataframe.iterrows():
        # Combinar columnas de texto
        combined_text = ". ".join([str(row[col]) for col in text_columns if col in row])
        combined_texts.append(combined_text)

        # Tokenizar y generar embeddings
        inputs = tokenizer(combined_text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            pooled = _mean_pooling(outputs, inputs['attention_mask'])
            embedding = pooled.squeeze().cpu().numpy()
            embeddings.append(embedding.tolist())

    logger.info(f"Generados {len(embeddings)} embeddings.")
    return embeddings, combined_texts


def search_embeddings(query_text, top_k=5):
    """Buscar los embeddings más cercanos en Milvus."""
    # Generar embedding para el texto de consulta
    inputs = tokenizer(query_text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        query_embedding = _mean_pooling(outputs, inputs['attention_mask']).squeeze().cpu().numpy()

    # Realizar búsqueda en Milvus
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(
        data=[query_embedding.tolist()],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["text"]
    )

    # Procesar resultados
    matches = []
    for result in results[0]:
        matches.append({
            "text": result.entity.get("text"),
            "distance": result.distance
        })

    logger.info(f"Encontrados {len(matches)} resultados para la consulta.")
    return matches

# Insertar datos en Milvus
if __name__ == "__main__":
    # Ruta del archivo CSV
    csv_path = os.path.join("..", "data", "wiki_scraped.csv")

    # Cargar datos
    df = load_csv(csv_path)

    # Generar embeddings
    text_columns = ["category", "section", "title", "paragraph"]
    embeddings, combined_texts = generate_embeddings(df, text_columns)

    # Truncar textos que excedan el límite de VARCHAR (500 bytes en UTF-8)
    MAX_TEXT_BYTES = 500
    truncated_texts = []
    truncated_count = 0
    for t in combined_texts:
        t_bytes = t.encode('utf-8')
        if len(t_bytes) > MAX_TEXT_BYTES:
            truncated_count += 1
            # Truncar por bytes y decodificar ignorando bytes parciales al final
            truncated = t_bytes[:MAX_TEXT_BYTES].decode('utf-8', errors='ignore')
            truncated_texts.append(truncated)
        else:
            truncated_texts.append(t)

    if truncated_count:
        logger.warning(f"Se truncaron {truncated_count} textos a {MAX_TEXT_BYTES} bytes para cumplir el esquema de la colección.")

    # Insertar en Milvus
    # Nota: el orden de las columnas para insertar debe coincidir con el esquema (excluyendo la PK auto_id).
    data_to_insert = [
        embeddings,       # embeddings (FLOAT_VECTOR)
        truncated_texts   # textos (VARCHAR)
    ]
    insert_result = collection.insert(data_to_insert)
    logger.info(f"Insertados {len(embeddings)} registros en la colección '{collection_name}'.")

    # Crear índice para el campo vectorial si no existe (necesario antes de load/search)
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 1024}
    }
    try:
        logger.info("Creando índice en el campo 'embedding'...")
        collection.create_index(field_name="embedding", index_params=index_params)
        logger.info("Índice creado correctamente.")
    except Exception as e:
        logger.warning(f"No se pudo crear el índice (puede ya existir): {e}")

    # Confirmar inserción y cargar la colección
    try:
        collection.load()
        logger.info(f"Colección '{collection_name}' lista para consultas.")
    except Exception as e:
        logger.error(f"Error cargando la colección: {e}")
        raise

    # Ejemplo de búsqueda
    query = "¿Cómo utilizar múltiples vistas en participantes?"
    top_results = search_embeddings(query)
    for idx, match in enumerate(top_results):
        print(f"Resultado {idx + 1}:\nTexto: {match['text']}\nDistancia: {match['distance']}\n")
