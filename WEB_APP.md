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
- The app is configured for local execution only. The React frontend calls `http://127.0.0.1:8000`.

To clear local chat history before sharing the project folder:
`rm -f data/user_histories/*.json`
