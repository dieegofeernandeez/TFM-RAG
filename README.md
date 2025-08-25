# TFM-RAG (RAG con Milvus, FastAPI, Chainlit y Groq)

Proyecto de fin de máster: sistema RAG (Retrieval Augmented Generation) completo que:

1. Ingresa un corpus (CSV) en Milvus como vectores.
2. Recupera fragmentos relevantes vía búsqueda vectorial.
3. Construye un prompt con contexto.
4. Genera la respuesta final usando un LLM servido por Groq.

## 🗺️ Arquitectura

Servicios (Docker Compose):
- `api-backend` (FastAPI):
	- Endpoints `/chat`, `/rag/retrieve`, `/rag`, `/rag/status`.
	- Carga el modelo de embeddings sentence-transformers (por defecto `all-MiniLM-L6-v2`).
	- Conecta con Milvus para búsqueda vectorial.
	- Llama a Groq (modelos Llama / Mixtral) con fallback automático.
- `chainlit-app`: UI de chat que llama al backend (`/rag` por defecto).
- `milvus`: Milvus standalone (vector DB) + etcd + MinIO.
- `etcd`: metadatos de Milvus.
- `minio`: almacenamiento de segmentos.

Flujo RAG:
```
Usuario -> Chainlit -> /rag (FastAPI) -> Embedding pregunta -> Milvus (search) ->
Contextos -> Prompt compuesto -> Groq LLM -> Respuesta -> Chainlit
```

## 📂 Estructura principal
```
backend/
	main.py                # API FastAPI + endpoints RAG
	embedding_generator.py  # Script de ingestión parametrizable
	requirements.txt
frontend/
	app.py                 # Interfaz Chainlit
data/
	wiki_scraped.csv       # Dataset de ejemplo
docker-compose.yml
.env / .env.example
```

## ⚙️ Variables de entorno

Archivo `.env` (ejemplo):
```
GROQ_API_KEY=TU_CLAVE_GROQ
GROQ_MODEL=llama3-8b-8192  # (se hará fallback si no existe)
API_PORT=8000
CHAINLIT_PORT=8080
```

Variables extra usadas por `embedding_generator.py` (puedes exportarlas antes de ejecutar):
```
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
MILVUS_HOST=127.0.0.1        # Dentro del contenedor backend sería 'milvus'
MILVUS_PORT=19530
EMBED_COLLECTION=text_embeddings
EMBED_CSV=../data/wiki_scraped.csv
FORCE_REINGEST=false         # true para reinsertar aunque ya haya datos
RUN_EXAMPLE=false            # true para imprimir búsqueda de ejemplo tras ingestión
INDEX_TYPE=IVF_FLAT
METRIC_TYPE=L2
NLIST=1024
MAX_TEXT_BYTES=500
```

## 🚀 Puesta en marcha

1. Coloca tu clave de Groq en `.env` (o usa variables de entorno del sistema).
2. Levanta la infraestructura:
```powershell
docker compose up -d --build
```
3. Verifica salud del backend:
```powershell
curl http://localhost:8000/health
```
4. Abre Chainlit: http://localhost:8080

> Nota: Si preguntas antes de haber insertado embeddings, las respuestas RAG no tendrán contexto útil.

## 📥 Ingestión de datos (embeddings)

Ejecuta el script desde el host (requiere Python con dependencias) o dentro del contenedor backend.

### Opción A: Host local
```powershell
cd backend
pip install -r requirements.txt
python embedding_generator.py
```

### Opción B: Dentro del contenedor backend
```powershell
docker exec -it groq-api python embedding_generator.py
```

La ingestión:
- Crea la colección `text_embeddings` si no existe.
- Genera embeddings (dim=384) y trunca textos a 500 bytes UTF-8.
- Inserta datos y crea índice IVF_FLAT (L2).
- Evita reinserción si ya hay datos (a menos que actives FORCE_REINGEST=true).

### Ver estado del RAG
```powershell
curl http://localhost:8000/rag/status
```
Respuesta esperada (ejemplo):
```json
{
	"collection_exists": true,
	"num_entities": 2481,
	"indexes": [...],
	"primary_model": "llama3-8b-8192",
	"fallback_models": ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"]
}
```

## 💬 Uso del endpoint RAG

Petición directa al retrieve:
```powershell
Invoke-RestMethod -Uri http://localhost:8000/rag/retrieve -Method Post -ContentType 'application/json' -Body '{"question":"¿Qué es RAG?","top_k":5}'
```

Petición completa (con generación):
```powershell
Invoke-RestMethod -Uri http://localhost:8000/rag -Method Post -ContentType 'application/json' -Body '{"question":"¿Qué es RAG?","top_k":5}'
```

Chainlit llamará automáticamente a `/rag` (variable `USE_RAG=true`).

## 🧪 Comprobaciones rápidas

1. `docker ps` muestra todos los contenedores (groq-api, chainlit-app, milvus, milvus-etcd, milvus-minio).
2. `/rag/status` muestra `num_entities > 0`.
3. `/rag/retrieve` devuelve contextos (lista no vacía).
4. `/rag` devuelve una respuesta sin error 500; si hay error 404 de modelo se selecciona fallback.

## 🔍 Troubleshooting

| Problema | Causa probable | Solución |
|----------|----------------|----------|
| `/rag/retrieve` devuelve listas vacías | No se han insertado embeddings | Ejecuta `embedding_generator.py` |
| Error 404 model_not_found en Groq | Modelo no disponible en tu clave | Se activa fallback; revisa `GROQ_MODEL` |
| Timeout en Chainlit | Backend caído o bloqueo de red | Ver logs: `docker logs groq-api` |
| Campo `text` demasiado largo | Texto > 500 bytes UTF-8 | Ajustar `MAX_TEXT_BYTES` o chunkear |
| Distancias altas irrelevantes | Embeddings pobres o pocos datos | Aumentar corpus / cambiar modelo |
| Falta de contexto útil | Textos muy largos sin segmentar | Implementar chunking futuro |

Logs útiles:
```powershell
docker logs -f groq-api
docker logs -f chainlit-app
```

## 🔧 Mejoras futuras sugeridas
- Normalizar embeddings y usar métrica coseno (IP + normalización).
- Chunking de documentos y metadata adicional (fuente, título, url).
- Re-ranking (ej. cross-encoder) para mejorar precisión.
- Cache de respuestas frecuentes.
- Paginación y filtrado semántico por campos estructurados.

## 🛡️ Seguridad
- No publiques tu `GROQ_API_KEY` en repositorios públicos.
- Usa `.env` (añadido a `.gitignore`).
- Controla el tamaño del corpus para evitar consumo excesivo de memoria.

## 🧾 Licencia
Proyecto académico. Añade la licencia que corresponda si se distribuye.

---
Si necesitas automatizar la ingestión al levantar los contenedores o añadir nuevas fuentes (PDF, web scraping), se puede extender con tareas adicionales. Pídelo y lo integramos.