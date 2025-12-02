# Dockerfile para Backend Chatbot con FastAPI + UV
# Optimizado para AWS ECS/Fargate

FROM python:3.11-slim

WORKDIR /app

# Variables de entorno
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    PORT=8000 \
    WORKERS=1

# Instalar dependencias del sistema + UV
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Instalar UV (ultrafast package installer)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copiar archivos de dependencias
COPY pyproject.toml .

# Instalar dependencias con UV (10-100x más rápido que pip)
RUN uv pip install --system -r pyproject.toml

# Copiar código de la aplicación
COPY . .

# Crear usuario no-root para seguridad
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/chroma_db && \
    chown -R appuser:appuser /app

USER appuser

# Exponer puerto
EXPOSE 8000

# Health check para ECS
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando con múltiples workers para producción
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers ${WORKERS}
