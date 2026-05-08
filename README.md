# Taiwan Stock Advisor Interface

This project has migrated from a Streamlit-only app to a React + FastAPI web app.

The old Streamlit app is still available as a fallback, but the primary path is now:

- FastAPI backend: `api_server.py`
- React frontend: `web/`
- Shared analysis code: `data_fetch.py`, `sentiment_analysis.py`, `config.py`
- Shared chat history: `data/user_histories/`

## Current App Paths

| App | Purpose | URL |
| --- | --- | --- |
| React + FastAPI | Primary web UI | `http://127.0.0.1:5173/` |
| FastAPI | Backend API used by React | `http://127.0.0.1:8000/` |
| Streamlit | Legacy fallback UI | `http://localhost:8501/` |

## Start The New React Web App

Run these in separate terminals from the project root.

Replace `/path/to/interface` with the location of the `interface` folder on the target machine, and use `interface_react` as the local conda environment name.

### 1. Start FastAPI

```bash
conda activate interface_react
python -m uvicorn api_server:app --host 127.0.0.1 --port 8000
```

Verify the API:

```bash
curl http://127.0.0.1:8000/api/health
```

Expected response:

```json
{"status":"ok","default_model":"openai/gpt-oss-20b"}
```

### 2. Start React

```bash
cd /path/to/interface/web
npm install
npm run dev -- --port 5173
```

Open:

```text
http://127.0.0.1:5173/
```

## Stop The New React Web App

If the servers are running in foreground terminals, stop each one with:

```text
Ctrl+C
```

If a port is stuck, find and stop the process.

FastAPI on port `8000`:

```bash
lsof -iTCP:8000 -sTCP:LISTEN
kill <PID>
```

React/Vite on port `5173`:

```bash
lsof -iTCP:5173 -sTCP:LISTEN
kill <PID>
```

Use `kill -9 <PID>` only if the normal `kill` does not stop the process.

## Run The Legacy Streamlit App

Streamlit remains available as a legacy fallback.

```bash
cd /path/to/interface
source /path/to/conda.sh
conda activate interface_react
streamlit run app_main.py
```

Open:

```text
http://localhost:8501/
```

Stop it with `Ctrl+C` in the Streamlit terminal.

## Migration Notes

The Streamlit app previously owned the whole user experience. In the new architecture, Streamlit is only a fallback UI. The React app owns the frontend, and FastAPI exposes the same analysis/session behavior over HTTP.

React now owns:

- Chat sessions and history navigation.
- Groq API key entry and model override.
- Investor risk settings for style and maximum acceptable loss.
- Streaming analysis responses.
- Dashboard metrics, price chart, news table, and quant/model details.

Keep backend logic in the existing Python modules whenever possible:

- `api_server.py` should expose sessions, profile updates, and streaming responses.
- `data_fetch.py` should continue to own stock/news/quant analysis workflows.
- `sentiment_analysis.py` should continue to own sentiment-specific behavior.
- `config.py` should continue to own Groq model/API configuration.

Keep frontend behavior in `web/src/`:

- `web/src/main.tsx` owns React UI state, session rendering, streaming updates, and controls.
- `web/src/api.ts` owns HTTP calls to FastAPI.
- `web/src/styles.css` owns the Codex-inspired app shell and responsive layout.
- `web/src/types.ts` mirrors the API payload shapes.

## Future Migration Checklist

Use this checklist if a remaining Streamlit-only feature needs to move into React:

1. Identify the existing Streamlit behavior in `app_main.py` or `ui_components.py`.
2. Move reusable analysis logic into backend modules if it is still embedded in Streamlit UI code.
3. Add or reuse a FastAPI endpoint in `api_server.py`.
4. Add a typed API helper in `web/src/api.ts`.
5. Add or update React UI in `web/src/main.tsx`.
6. Style the UI in `web/src/styles.css` using the neutral React design system.
7. Run the validation commands below.

## Validation Commands

Frontend build:

```bash
cd /path/to/interface/web
npm run build
```

Backend syntax check:

```bash
cd /path/to/interface
source /path/to/conda.sh
conda activate interface_react
python -m py_compile config.py api_server.py sentiment_analysis.py
```

FastAPI health check:

```bash
curl http://127.0.0.1:8000/api/health
```

## Troubleshooting

If React says it cannot connect to FastAPI, make sure `api_server.py` is running on `127.0.0.1:8000`.

If the app says the Groq API key is missing, open the model/settings panel in the React UI and enter the key there. The frontend sends it to FastAPI per request using the `X-Groq-API-Key` header.

If a port is already in use, either stop the old process with `lsof` and `kill`, or start the service on a different port. If you change the FastAPI port, set `VITE_API_BASE` for the React app.

Example:

```bash
cd /path/to/interface/web
VITE_API_BASE=http://127.0.0.1:8001 npm run dev -- --port 5173
```

## Production Build Preview

Build the React app:

```bash
cd /path/to/interface/web
npm run build
```

Preview the built app locally:

```bash
npm run preview
```

The preview server still needs FastAPI running separately for real API calls.
