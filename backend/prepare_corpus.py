import csv
import json
import os
import re
import hashlib
from datetime import datetime, timezone
from typing import List, Dict, Iterable

import pandas as pd

# --- Configuración ---
# Resolver la ruta base del repositorio (un nivel arriba de backend)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))

# Permitir override completo por variables; si vienen rutas relativas, resolver contra DATA_DIR
def _resolve_path(env_name: str, default_filename: str) -> str:
    val = os.getenv(env_name)
    if not val:
        return os.path.join(DATA_DIR, default_filename)
    # Si es ruta absoluta la dejamos, si no, interpretamos relativa a DATA_DIR
    if not os.path.isabs(val):
        return os.path.join(DATA_DIR, val)
    return val

WIKI_CSV = _resolve_path("WIKI_CSV", "wiki_scraped.csv")
PDF_CSV = _resolve_path("PDF_CSV", "pdfs_scraped.csv")
OUTPUT_JSONL = _resolve_path("OUTPUT_JSONL", "prepared_corpus.jsonl")
# Tamaños de chunk orientativos (caracteres después de normalización y SIN prefijo metadatos)
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "900"))
CHUNK_MIN_CHARS = int(os.getenv("CHUNK_MIN_CHARS", "180"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
# Prefijo con metadatos en el texto final para embedding (True recomendado)
INCLUDE_METADATA_PREFIX = os.getenv("INCLUDE_METADATA_PREFIX", "true").lower() == "true"

# --- Utilidades ---
_whitespace_re = re.compile(r"\s+")
_non_word_space = re.compile(r"\u00A0")  # nbsp


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r", " ")
    text = _non_word_space.sub(" ", text)
    text = _whitespace_re.sub(" ", text)
    return text.strip()

def est_tokens(text: str) -> int:
    # Estimación rápida (~1 token cada 4 chars en español general)
    return max(1, len(text) // 4)

def sha1(text: str) -> str:
    return hashlib.sha1(text.encode('utf-8')).hexdigest()[:16]

def load_csv_safe(path: str, source_type: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[WARN] No existe {path}, se omite {source_type}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df['__source_type'] = source_type
    return df

# --- Chunking simple por caracteres con solape ---

def chunk_text(text: str, max_chars: int, overlap: int) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunk = text[start:end]
        if len(chunk) < CHUNK_MIN_CHARS and chunks:
            # Adjuntar residuo pequeño al último chunk
            chunks[-1] = chunks[-1] + " " + chunk
            break
        chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap  # retroceder para solape
    return chunks

# --- Preparación de filas ---

def iter_rows(df: pd.DataFrame) -> Iterable[Dict]:
    required_cols = {"paragraph", "category", "section", "title"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"[WARN] Faltan columnas requeridas: {missing}")
    for idx, row in df.iterrows():
        paragraph = normalize_text(str(row.get('paragraph', '') or ''))
        if not paragraph or len(paragraph) < 20:
            continue  # descartar vacíos y muy cortos
        category = normalize_text(str(row.get('category', '') or '')).replace('\n', ' ')
        section = normalize_text(str(row.get('section', '') or '')).replace('\n', ' ')
        title = normalize_text(str(row.get('title', '') or '')).replace('\n', ' ')
        url_raw = row.get('url', '')
        # Limpiar NaN / 'nan'
        try:
            import math
            if isinstance(url_raw, float) and math.isnan(url_raw):
                url = ''
            else:
                url = str(url_raw).strip()
        except Exception:
            url = str(url_raw).strip() if url_raw is not None else ''
        if url.lower() == 'nan':
            url = ''
        video_raw = row.get('video_urls', '')
        # Convertir a lista si viene como string tipo ['a','b'] o una sola URL
        video_urls: List[str] = []
        if isinstance(video_raw, str) and video_raw.strip():
            if video_raw.startswith('[') and video_raw.endswith(']'):
                # intentar parseo simple
                inner = video_raw.strip()[1:-1].strip()
                if inner:
                    parts = re.split(r"['\"]?\s*,\s*['\"]?", inner)
                    for p in parts:
                        p = p.strip().strip("'\"")
                        if p.startswith('http'):
                            video_urls.append(p)
            elif video_raw.startswith('http'):
                video_urls.append(video_raw.strip())
        source_type = row.get('__source_type', 'unknown')
        yield {
            'raw_text': paragraph,
            'category': category,
            'section': section,
            'title': title,
            'url': url,
            'video_urls': video_urls,
            'has_image': bool(str(row.get('image', '') or '').strip()),
            'is_code': str(row.get('is_code', '')).lower() in ('true', '1'),
            'source_type': source_type
        }


def build_chunks(row: Dict) -> List[Dict]:
    text = row['raw_text']
    base_chunks = chunk_text(text, CHUNK_MAX_CHARS, CHUNK_OVERLAP)
    results = []
    for pos, ck in enumerate(base_chunks):
        ck_norm = normalize_text(ck)
        if len(ck_norm) < CHUNK_MIN_CHARS and len(base_chunks) > 1:
            continue
        # Prefijo de metadatos opcional
        if INCLUDE_METADATA_PREFIX:
            prefix = f"[{row['title']}] ({row['category']} > {row['section']})\n"
            final_text = (prefix + ck_norm).strip()
        else:
            final_text = ck_norm
        h = sha1(final_text)
        results.append({
            'id': f"{row['source_type']}|{h}|{pos:04d}",
            'source_type': row['source_type'],
            'category': row['category'],
            'section': row['section'],
            'title': row['title'],
            'url': row['url'],
            'position': pos,
            'text': final_text,
            'tokens_est': est_tokens(final_text),
            'has_image': row['has_image'],
            'has_video': bool(row['video_urls']),
            'video_urls': row['video_urls'],
            'is_code': row['is_code'],
            'content_hash': h,
            'extracted_at': datetime.now(timezone.utc).isoformat()
        })
    return results


def main():
    print("[INFO] Cargando CSVs...")
    wiki_df = load_csv_safe(WIKI_CSV, 'wiki')
    pdf_df = load_csv_safe(PDF_CSV, 'lessons_pdf')
    combined = pd.concat([wiki_df, pdf_df], ignore_index=True)
    print(f"[INFO] Filas combinadas iniciales: {len(combined)}")

    print("[INFO] Normalizando y generando filas base...")
    logical_rows = list(iter_rows(combined))
    print(f"[INFO] Filas válidas tras limpieza mínima: {len(logical_rows)}")

    print("[INFO] Generando chunks...")
    all_chunks: List[Dict] = []
    for r in logical_rows:
        all_chunks.extend(build_chunks(r))
    print(f"[INFO] Chunks generados: {len(all_chunks)}")

    # Deduplicación por content_hash
    seen = set()
    deduped = []
    for c in all_chunks:
        if c['content_hash'] in seen:
            continue
        seen.add(c['content_hash'])
        deduped.append(c)
    print(f"[INFO] Tras deduplicación hashes: {len(deduped)}")

    # Guardar JSONL
    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
        for item in deduped:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"[INFO] Guardado: {OUTPUT_JSONL}")

    # Estadísticas rápidas
    avg_tokens = sum(c['tokens_est'] for c in deduped) / max(1, len(deduped))
    if deduped:
        print(f"[STATS] avg_tokens≈{avg_tokens:.1f} max_len_chars={max(len(c['text']) for c in deduped)}")
    else:
        print("[STATS] Corpus vacío: no se generaron chunks. Verifica rutas o variables WIKI_CSV / PDF_CSV / DATA_DIR.")

if __name__ == '__main__':
    main()
