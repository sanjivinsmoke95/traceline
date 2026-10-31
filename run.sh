#!/usr/bin/env bash
# Start the API (:8000) and the web app (:5173) together. Ctrl-C stops both.
set -e
cd "$(dirname "$0")"
if [ ! -x backend/.venv/bin/python ]; then
  echo "Backend is not set up yet: backend/.venv is missing. Follow 'First-time setup' in README.md (Python 3.11, PostgreSQL 16, .env, alembic upgrade head)." >&2
  exit 1
fi
if command -v pg_isready >/dev/null 2>&1 && ! pg_isready -q; then
  echo "PostgreSQL is not accepting connections. Start it, then run this again (the API needs it)." >&2
  exit 1
fi
(cd backend && ./.venv/bin/python -m uvicorn app.main:app --port 8000) &
API=$!
trap 'kill $API 2>/dev/null' EXIT
for i in $(seq 1 30); do curl -fs localhost:8000/api/v1/health >/dev/null 2>&1 && break; sleep 1; done
curl -fs localhost:8000/api/v1/health >/dev/null 2>&1 || { echo "The API did not come up on :8000 (see the error above)." >&2; exit 1; }
cd frontend && [ -d node_modules ] || npm install
echo "API is up. Open http://localhost:5173"
npm run dev -- --port 5173
