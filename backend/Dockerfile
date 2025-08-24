FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .

#RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -r requirements.txt

# Copiar código fuente
COPY . .

# Exponer puerto
EXPOSE 8000

# Comando por defecto
CMD ["python", "main.py"]