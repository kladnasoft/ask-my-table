# Use a slim Python base
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# OS deps:
#  - msodbcsql18 for SQL Server/Synapse via pyodbc
#  - unixodbc-dev for pyodbc compile/runtime
#  - curl for healthcheck
#  - build-essential only for any wheel builds if needed by dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gnupg2 apt-transport-https ca-certificates \
    build-essential unixodbc-dev \
 && curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/microsoft-prod.gpg arch=amd64] https://packages.microsoft.com/debian/12/prod bookworm main" > /etc/apt/sources.list.d/microsoft-prod.list \
 && apt-get update && apt-get install -y --no-install-recommends \
    msodbcsql18 \
 && apt-get purge -y --auto-remove \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better caching
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
# Your code lives under ./app in the repo root
COPY app ./app

# Include default metaschema into the image (compose will bind-mount over it for live editing)
COPY app/metaschema ./metaschema

# Runtime config
ENV HOST=0.0.0.0 \
    PORT=8080 \
    METASCHEMA_DIR=/app/metaschema

EXPOSE 8080

# Healthcheck hits FastAPI health endpoint
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD curl -fsS http://localhost:${PORT}/healthz || exit 1

# Start the app (main.py runs uvicorn when __main__)
CMD ["python", "-m", "app.main"]
