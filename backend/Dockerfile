## Habilita sintaxis moderna para mounts de cache (requiere BuildKit)
# syntax=docker/dockerfile:1.6
FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

## Copiar solo requirements primero para maximizar cache
COPY requirements.txt ./requirements.txt

## Usar cache de pip (BuildKit) – si BuildKit no está activo, se ignora el mount
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY . .

# Exponer puerto
EXPOSE 8000

# Comando por defecto
CMD ["python", "main.py"]