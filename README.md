# TraceLine

AI-assisted reconstruction of road-accident evidence for the CBIT Hacktoberfest '26 hackathon (problem: *AI-Powered Timeline Reconstruction for Road Accident Forensics*).
It lines up dashcam and CCTV video, witness statements and vehicle sensor logs on one timeline, shows where they agree or conflict, lists what evidence is missing, and produces a validated report.
**It supports an investigation; it never assigns fault.** All demo data is **synthetic** and labelled as such.

## Run it
Already set up? `./run.sh` starts the API and the web app together, then open http://localhost:5173. The site needs **both** running, plus PostgreSQL.

First-time setup:
```bash
cp .env.example .env                                                        # in the REPO ROOT (not backend/): set DATABASE_URL (PostgreSQL 16); FEATHERLESS_API_KEY is optional
cd backend
python3.11 -m venv .venv && ./.venv/bin/pip install -r requirements.txt     # once
./.venv/bin/pip install -r requirements-vision.txt                          # detection/tracking and the DETR accident cue (needed for real footage)
./.venv/bin/alembic upgrade head
./.venv/bin/python scripts/fetch_models.py                                  # optional: detector weights (only needed for real footage)
./.venv/bin/python scripts/build_scenarios.py                               # render the four synthetic scenarios
./.venv/bin/python scripts/seed_demo.py --reset                             # four demo cases + reconstructions + reports
./.venv/bin/python -m uvicorn app.main:app --port 8000                      # API (docs at /docs)

cd ../frontend && npm install && npm run dev                                # http://localhost:5173 (proxies /api to :8000)
```
Tests: `cd backend && ./.venv/bin/python -m pytest -q` (175 tests; need `TEST_DATABASE_URL`). Evaluation: `scripts/eval_scenarios.py` (results in `research/phase14/`).

## Where things are
`backend/` FastAPI + PostgreSQL pipeline and rules · `frontend/` React workspace (UI kit copied into `src/ui-kit`) · `docs/` architecture, schema, API, risks, roadmap · `research/` measured results · `data/` evidence store and weights (git-ignored).

## Honest status
See `docs/IMPLEMENTATION_ROADMAP.md` (status table) and `docs/RISKS_AND_LIMITATIONS.md`. In short: the pipeline, rules, report and UI work end to end on synthetic scenarios; accuracy on real footage and real witness language is **not yet measured**.
