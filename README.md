# TimesFM overnight gap dashboard (faisalm1)

**Primary UI:** React (Vite) + **FastAPI** backend. Alpaca keys live only in **`.env`** on the Python process; the browser talks to `/api/*` and never sees secrets.

Optional Streamlit UI remains in `src/gap_dashboard/app.py` if you need it.

## Setup

```powershell
cd "C:\Users\User\Desktop\Cursor 2026\faisalm1\TimesFM"
python -m venv .venv
.\.venv\Scripts\pip install -e ".[dev]"
copy .env.example .env
# put APCA_API_KEY_ID and APCA_API_SECRET_KEY in .env (not in the React app)
cd web
npm install
```

## Run locally (two terminals)

**Terminal 1 — API (loads `.env`, TimesFM, Alpaca):**

```powershell
cd "C:\Users\User\Desktop\Cursor 2026\faisalm1\TimesFM"
.\.venv\Scripts\python -m uvicorn gap_dashboard.api:app --app-dir src --reload --host 127.0.0.1 --port 8000
```

**Terminal 2 — React (proxies `/api` → :8000):**

```powershell
cd "C:\Users\User\Desktop\Cursor 2026\faisalm1\TimesFM\web"
npm run dev
```

Open the URL Vite prints (usually `http://localhost:5173`). API docs: `http://127.0.0.1:8000/docs`.

## Optional: Streamlit

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\streamlit run src/gap_dashboard/app.py
```

## Model

First run may download `google/timesfm-1.0-200m-pytorch` from Hugging Face. Optional: `TIMESFM_CUDA=1` for GPU.

## GSD

Get Shit Done is under `.cursor/` (`npx get-shit-done-cc@latest --cursor --local`).

## Disclaimer

Not financial advice. Validate with your own backtests. Historical bars cache under `data/cache/`.
