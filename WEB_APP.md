# React Web App

This repository uses React for the frontend and FastAPI for the backend.

## Run

Backend:

```bash
python -m uvicorn api_server:app --reload --host 127.0.0.1 --port 8000
```

Frontend:

```bash
cd web
npm install
npm run dev
```

Open [http://127.0.0.1:5173](http://127.0.0.1:5173).

## Notes

- The frontend sends the user API key per request using `X-Groq-API-Key`.
- API key entry lives in the sidebar API panel.
- Investor style and maximum acceptable loss live in the composer risk panel.
- Chat history is stored in `/Users/yanwanzhen/Downloads/react/data/user_histories`.

## Deploy (Vercel + Render)

### Render

1. Create a Render web service from this repo root.
2. Build command: `pip install -r requirements.txt`
3. Start command: `sh -c "uvicorn api_server:app --host 0.0.0.0 --port ${PORT}"`
4. Optional env vars:
   - `CORS_ALLOW_ORIGINS`
   - `CORS_ALLOW_ORIGIN_REGEX`

### Vercel

1. Import this repo in Vercel.
2. The project is configured by `/Users/yanwanzhen/Downloads/react/vercel.json` to build from `web/`.
3. Set `VITE_API_BASE` to your backend URL.
