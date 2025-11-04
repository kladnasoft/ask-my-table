# Ask-My-Table – Dockerized

This folder contains a ready-to-run **Docker setup** for your FastAPI app, which connects to **SQL Server or Synapse** via `pyodbc` and the **Microsoft ODBC 18 driver**.

---

## Files

- `Dockerfile` – Python 3.11 + msodbcsql18 + uvicorn
- `docker-compose.yml` – simple orchestration with `.env` and bind-mount for `metaschema/`
- `.dockerignore` – excludes local secrets (e.g. `.env`) from the image
- `ask-my-table.py` – main FastAPI router (formerly `data_chat.py`)
- `prompts.py` – minimal prompt templates (replace with your own if needed)

---

## 1) Prepare your `.env` (⚠️ never commit this)

Create a `.env` file next to `docker-compose.yml` with **valid** environment variables:

```bash
# Provider
AI_PROVIDER=openai_assistant
OPENAI_API_KEY=sk-***REDACTED***

# Assistants
OPENAI_GENERIC_ID=asst_...
OPENAI_METABUILDER_ID=asst_...
OPENAI_SQLGENERATOR_ID=asst_...
ASSISTANT_MAX_WAIT_SEC=300
ASSISTANT_HTTP_TIMEOUT=300

# DB (if the DB runs on your host, use host.docker.internal)
DB_MODE=sqlserver
SYNAPSE_SERVER=192.168.168.130
SYNAPSE_DATABASE=Drinks
SYNAPSE_USERNAME=sa
SYNAPSE_PASSWORD=YourStrong!Password123
SYNAPSE_ODBC_DRIVER=ODBC Driver 18 for SQL Server

# Metaschema directory
METASCHEMA_DIR=/app/metaschema

# Optional HUD/debug
HUD_PAGE_INTERVAL=800
HUD_PAGE_SIZE=3
HUD_LABEL_MAX=16
FORCE_MULTILINE_PROGRESS=0
HTTPX_LOG_LEVEL=INFO
PRINT_HTTP_BODIES=1
MAX_PRINT_CHARS=8000

# Retries
SQL_MAX_RETRIES=5
AUX_MAX_RETRIES=5
```

> **Important:** Make sure you spell it exactly `METASCHEMA_DIR` — not `METASSCHEMA_DIR`.

---

## 2) Add your metaschema JSON files

Put your AI-metadata JSON definitions in `./metaschema/` (on your host).  
These are automatically mounted into the container at `/app/metaschema`.

---

## 3) Build & Run

Using **Docker Compose** (recommended):

```bash
docker compose up --build
# UI:     http://localhost:8080/ui
# Health: http://localhost:8080/healthz
```

Or manually with plain Docker:

```bash
docker build -t ask-my-table:latest .
docker run --rm -p 8080:8080 --env-file .env   --add-host host.docker.internal:host-gateway   -v "$PWD/metaschema:/app/metaschema:ro"   ask-my-table:latest
```

---

## 4) API Endpoints

| Endpoint | Method | Description |
|-----------|---------|-------------|
| `/ask-my-table/start` | POST | Start a new SQL assistant request |
| `/ask-my-table/progress/{req_id}` | GET | Check background progress |
| `/ask-my-table/result/{req_id}` | GET | Retrieve final query result |
| `/ui` | GET | Lightweight debug web UI |
| `/healthz` | GET | Health check endpoint |

---

## 5) Troubleshooting

- **ODBC driver not found:**  
  The container installs `msodbcsql18` and `unixodbc`. Logs will confirm:
  `DB driver selected: ODBC Driver 18 for SQL Server`.

- **Database connection errors:**  
  If the DB runs locally, use `SYNAPSE_SERVER=host.docker.internal` and keep `extra_hosts` in `docker-compose.yml`.

- **Module import issues:**  
  The refactored app dynamically loads `ask-my-table.py`, so filenames with hyphens work fine in Python.

- **METASCHEMA not loading:**  
  Confirm that `METASCHEMA_DIR` points to the correct JSON directory inside the container.

- **UI not responding / port closed:**  
  The app binds to `0.0.0.0:8080`. Check your firewall or port mappings.

---

### Example

After starting with Docker Compose, visit:

```
http://localhost:8080/ui
```

Then try sending a query or SQL request.  
The backend routes are now served under `/ask-my-table/...`.

---

**Maintainer:** KladnaSoft  
**Version:** Ask-My-Table (refactored successor to Data Chat)
