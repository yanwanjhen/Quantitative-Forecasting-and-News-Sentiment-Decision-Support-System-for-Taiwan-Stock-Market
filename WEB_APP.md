# React Web App

This is the primary React web UI for the Taiwan stock advisor. The Streamlit app
remains available as a legacy fallback.

## Run

Start the Python API:

```bash
conda activate interface
python -m uvicorn api_server:app --reload --host 127.0.0.1 --port 8000
```

Start the React client:

```bash
cd web
npm install
npm run dev
```

Open `http://127.0.0.1:5173`.

## Notes

- The frontend sends the Groq API key per request using `X-Groq-API-Key`.
- The Groq API key and advanced model override live in the sidebar API Key panel.
- Investor style and maximum acceptable loss live in the composer risk panel.
- Chat history is stored in the existing `data/user_histories` directory.
- In production, set `REQUIRE_USER_API_KEY=1` on the backend to require every user to supply their own key.

## Deploy (Vercel + Render)

This repo can be deployed as:

- React (Vite) frontend on Vercel
- FastAPI backend on Render

### Render (FastAPI)

1. Create a Render "Web Service" from this repo (root directory).
2. Use the commands from `render.yaml`, or set:
   - Build command: `pip install -r requirements-interface-st-bottom.txt`
   - Start command: `sh -c "uvicorn api_server:app --host 0.0.0.0 --port ${PORT}"`
3. Environment variables (recommended):
   - `CORS_ALLOW_ORIGINS` (optional): comma-separated exact origins
   - `CORS_ALLOW_ORIGIN_REGEX` (optional): origin regex, default matches `https://*.vercel.app`

After deploy, copy the backend URL, e.g. `https://your-service.onrender.com`.

### Vercel (React)

1. Import this repo in Vercel.
2. The project is configured by the root `vercel.json` to build from `web/`.
3. Set Vercel environment variable:
   - `VITE_API_BASE` = Render backend URL (no trailing slash), e.g. `https://your-service.onrender.com`

Redeploy after setting the env var so Vite bakes it into the build.
