# Taiwan Stock Advisor Web App

This project now runs as a single React + FastAPI application.

## App Structure

- FastAPI backend: `/Users/yanwanzhen/Downloads/react/api_server.py`
- React frontend: `/Users/yanwanzhen/Downloads/react/web`
- Shared analysis logic: `/Users/yanwanzhen/Downloads/react/data_fetch.py`, `/Users/yanwanzhen/Downloads/react/sentiment_analysis.py`, `/Users/yanwanzhen/Downloads/react/config.py`
- Session storage: `/Users/yanwanzhen/Downloads/react/data/user_histories`

## Local Run

Start the backend:

```bash
python -m uvicorn api_server:app --host 127.0.0.1 --port 8000
```

Start the frontend:

```bash
cd web
npm install
npm run dev -- --port 5173
```

Open [http://127.0.0.1:5173](http://127.0.0.1:5173).

## Validation

Frontend build:

```bash
cd web
npm run build
```

Backend syntax check:

```bash
python -m py_compile api_server.py data_fetch.py config.py sentiment_analysis.py
```

FastAPI health check:

```bash
curl http://127.0.0.1:8000/api/health
```

## Notes

- Every user must provide their own API key through the frontend sidebar.
- The frontend sends that key in `X-Groq-API-Key` for each request.
- Investor profile and chat history are managed by the React + FastAPI app only.
- This project is currently configured for local execution only.
