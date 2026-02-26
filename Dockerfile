# ============================================================================
# Dockerfile - API de Predicción de Pagos ML
# ============================================================================
# Imagen Docker para desplegar API FastAPI con modelo de Machine Learning
# 
# Construcción:
#   docker build -t ml-payment-prediction-api:latest .
#
# Ejecución:
#   docker run -d -p 8000:8000 --name ml-api ml-payment-prediction-api:latest
#
# Autor: Alexis Jacquet - Data Science Expert
# Proyecto: HENRY M5 - Avance 3
# ============================================================================

# Usar imagen base oficial de Python 3.11 (slim para reducir tamaño)
FROM python:3.11-slim

# Metadata de la imagen
LABEL maintainer="Alexis Jacquet <alexis.jacquet@example.com>"
LABEL description="API de predicción de pagos con Machine Learning - HENRY M5"
LABEL version="1.0.0"

# Establecer variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    APP_HOME=/app

# Crear usuario no-root para seguridad
RUN useradd -m -u 1000 appuser && \
    mkdir -p ${APP_HOME} && \
    chown -R appuser:appuser ${APP_HOME}

# Establecer directorio de trabajo
WORKDIR ${APP_HOME}

# Instalar dependencias del sistema necesarias
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copiar archivo de requerimientos primero (para aprovechar cache de Docker)
COPY --chown=appuser:appuser requirements.txt .

# Instalar dependencias de Python
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar código de la aplicación
COPY --chown=appuser:appuser api_main.py .
COPY --chown=appuser:appuser app_streamlit.py .
COPY --chown=appuser:appuser mlops_pipeline/ mlops_pipeline/
COPY --chown=appuser:appuser data/ data/
# Nota: models/ NO se copia en la imagen - se monta como volumen en docker-compose.yml
# Para producción con modelo embebido: copiar models/ DESPUÉS de entrenar con run_pipeline.py

# Crear directorios necesarios
RUN mkdir -p models results data && \
    chown -R appuser:appuser models results data

# Cambiar a usuario no-root
USER appuser

# Exponer puerto de la API
EXPOSE 8000

# Health check para monitoreo
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Comando de inicio de la aplicación
CMD ["uvicorn", "api_main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
