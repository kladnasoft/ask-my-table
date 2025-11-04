# 🧠 Ask My Table

**Ask-My-Table** is an AI-powered **data chat application** that connects directly to your **[SuperTable](https://pypi.org/project/supertable/)** data warehouse.  
It allows business users and analysts to **query data in natural language**, automatically generating and executing SQL through Azure OpenAI or other LLM providers — and rendering visual results, tables, and execution traces interactively in a modern web UI.

---

## 🚀 Features

- **Natural-language data queries** — ask questions like *“Show me monthly revenue by region this year”*
- **AI-generated SQL** (DuckDB / SuperTable dialect) with strong validation and retry logic  
- **Automatic explanations & chart suggestions** (bar, line, pie, table, …)
- **Full execution trace** for transparency across planning, SQL generation, validation, execution, and explanation phases
- **Multi-provider AI support**
  - ✅ Azure OpenAI (default)
  - ⚙️ OpenAI (Assistants v2)
  - 🧩 Anthropic Claude (optional)
- **SuperTable integration**
  - REST-based metadata discovery
  - Query execution via `/execute` API
  - Auto-context resolution (organization, super_name, user_hash)
- **Built-in Admin UI**
  - Workspace (chat and results)
  - Settings editor (`.env` values live-editable)
  - Hot Cache viewer (metaschema explorer)
- **Secure access** via `UI_ACCESS_TOKEN`
- **Health endpoint:** `/healthz`

---

## 🧩 Architecture Overview

| Layer | Module | Description |
|-------|---------|-------------|
| **FastAPI backend** | `main.py` | Server entrypoint, auth, routing |
| **LLM Integration** | `ai_chat_azure.py`, `ai_openai.py`, `ai_claude.py` | Azure / OpenAI / Claude connectors |
| **Pipeline Engine** | `ask_my_table.py` | Planning → SQL → Execution → Explanation |
| **Prompts & Templates** | `prompts.py` | Instruction and prompt builders |
| **SuperTable REST client** | `_connect_supertable.py` | Authenticated connector to SuperTable API |
| **Metadata management** | `hot_cache.py`, `gen_metasample.py`, `gen_metadata.py` | Sample generation and AI-assisted schema enrichment |
| **Configuration API** | `settings.py` | Live `.env` editing and reload |
| **Frontend** | `templates/ask.html`, `ui.py` | Modern single-page app (Chart.js + Highlight.js) |

---

## ⚙️ Environment Variables

Create a `.env` file at the project root (auto-loaded via `python-dotenv`):

```bash
# Server
HOST=127.0.0.1
PORT=8080
UI_ACCESS_TOKEN=your_secret_token_here

# SuperTable connection
SUPERTABLE_URL=http://0.0.0.0:8000
SUPERTABLE_ADMIN_TOKEN=your_admin_token
SUPERTABLE_ORGANIZATION=kladna-soft
SUPER_NAME=ProdBigChange
SUPER_USER_HASH=...

# Azure OpenAI (default)
AI_PROVIDER=azure_openai
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-5-nano-2
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_API_KEY=sk-...

# (Optional OpenAI / Anthropic)
OPENAI_API_KEY=sk-...
OPENAI_METABUILDER_ID=asst_...
ANTHROPIC_API_KEY=...
```

---

## 🖥️ Running Locally

### Prerequisites
- Python 3.10+
- FastAPI, Uvicorn, Requests, Python-Dotenv

Install dependencies:
```bash
pip install -r requirements.txt
```

### Start the server
```bash
python main.py
```

### Access the UI
Open your browser at:  
👉 [http://127.0.0.1:8080/ui?token=YOUR_TOKEN](http://127.0.0.1:8080/ui?token=YOUR_TOKEN)

---

## 🧠 Workflow Example

1. User enters a question in plain English.  
2. **PLAN phase:** AI selects relevant tables/columns from SuperTable metaschema.  
3. **SQL phase:** AI generates a compliant DuckDB SQL query.  
4. **EXEC phase:** Query runs via `/execute` API (SuperTable).  
5. **EXPLAIN phase:** AI summarizes results and suggests a chart.  
6. UI displays:
   - Generated SQL  
   - Data table  
   - Visualization  
   - Trace timeline  

---

## 🧩 Related Utilities

| Script | Purpose |
|--------|----------|
| `gen_metasample.py` | Create sampled metadata from SuperTable tables |
| `gen_metadata.py` | Enrich metasample → AI metadata (semantic model) |
| `hot_cache.py` | Load and cache AI-enriched metaschemas |
| `settings.py` | View & edit environment configuration via UI |

---

## 🧱 Folder Structure

```
app/
 ├── main.py
 ├── ask_my_table.py
 ├── ai_chat_azure.py
 ├── ai_openai.py
 ├── ai_claude.py
 ├── _connect_supertable.py
 ├── prompts.py
 ├── settings.py
 ├── hot_cache.py
 ├── gen_metasample.py
 ├── gen_metadata.py
 ├── templates/
 │    └── ask.html
 └── metaschema/
      └── *.json
```

---

## 🔒 Security

- Access protected via `UI_ACCESS_TOKEN`
- Tokens accepted via:
  - `Authorization: Bearer <token>`
  - `?token=<token>` query parameter
  - `ui_token` cookie (auto-set after login)
- `.env` secrets never exposed to the frontend

---

## 🧾 License

© 2025 Kladna Soft  
Licensed under the **Super Table Public Use License (STPUL) v1.0**

---

## 💡 Acknowledgements

- [SuperTable](https://pypi.org/project/supertable/) — metadata-driven data warehouse
- [FastAPI](https://fastapi.tiangolo.com/)
- [Azure OpenAI Service](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
- [Chart.js](https://www.chartjs.org/)
- [Highlight.js](https://highlightjs.org/)
