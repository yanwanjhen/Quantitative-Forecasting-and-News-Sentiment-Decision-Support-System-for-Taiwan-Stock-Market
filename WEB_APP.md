# React Web App Deployment

This project uses a Vite React frontend and a FastAPI backend.

## Local Run

Backend:

```bash
python -m uvicorn api_server:app --host 127.0.0.1 --port 8000 --reload
```

Frontend:

```bash
cd web
npm install
npm run dev
```

Open [http://127.0.0.1:5173](http://127.0.0.1:5173).

## Environment Variables

Frontend:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Production frontend should set `VITE_API_BASE_URL` to the Render backend URL, for example:

```bash
VITE_API_BASE_URL=https://taiwan-stock-advisor-api.onrender.com
```

Backend:

```bash
CORS_ALLOW_ORIGINS=https://your-frontend.vercel.app
PYTHON_VERSION=3.11.10
REQUIRE_USER_API_KEY=1
```

The frontend sends the user API key per request using `X-Groq-API-Key`. The backend does not store a shared Groq key.

## Render Backend

Use the included `render.yaml`, or configure manually:

- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn api_server:app --host 0.0.0.0 --port $PORT`
- Health check: `/api/health`
- Python version: `3.11.10`

After Vercel gives you the frontend URL, set `CORS_ALLOW_ORIGINS` on Render to that exact origin.

## Vercel Frontend

Use the included `vercel.json` from the repository root:

- Install command: `cd web && npm ci`
- Build command: `cd web && npm run build`
- Output directory: `web/dist`

Set `VITE_API_BASE_URL` to the Render backend URL before deploying production.

## News Fetching

Sentiment news collection uses Yahoo Finance RSS first, Google News RSS second, and GDELT DOC API as a fallback. Each source is cached briefly in memory to reduce repeated RSS/API calls from the same deployment host. If all news sources fail, the API keeps the stock/quant analysis running and returns `news_count_status: "fetch_failed"`.

To clear local chat history before sharing the project folder:

```bash
rm -f data/user_histories/*.json
```
