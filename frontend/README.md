# Frontend architecture and service boundaries

React, Vite, TypeScript, Tailwind CSS, shadcn/ui, Recharts, Lucide, and React Router.

## Running

From the project root, run `npm.cmd --prefix frontend ci`, then `npm.cmd --prefix frontend run build`, then `./scripts/start.ps1`. Open http://127.0.0.1:8180. Build before starting the backend so the assets mount is registered. After edits, rebuild and reload the page.

For development, keep the backend on port 8180 and run `npm.cmd --prefix frontend run dev`. Open http://127.0.0.1:5173; API calls use the Vite proxy. Direct navigation to all eight routes works in both servers. Vite's preview command serves assets only; use the backend-served build to test live APIs.

## API service layer

All network communication is in `src/services/api.ts`. Credentials remain on the server. Preferences persist locally; uploaded data and current predictions are held in memory.

| Workflow | Live service | Preview mode |
| --- | --- | --- |
| Dataset/index metadata | GET `/api/workspace/datasets` | Explicit small news fixture |
| CSV/JSONL inspection | Local validation | Same validation |
| Dataset registration | Not connected | In-memory split registration |
| Index creation | Not connected; existing NumPy memory | Simulated state; no embeddings created |
| Detection / baseline | POST `/detect/compare` | Labeled deterministic simulator |
| Retrieval and prompt | POST `/api/workspace/retrieval` | Illustrative references and distances |
| Profile/cards/selection | `/selection/profile/{dataset}`, `/selection/cards`, `/selection/recommend` | Fixture geometry and candidate-order ranking; uploaded geometry unavailable |
| Saved evaluation | `/api/workspace/experiments` and `/{run}/predictions/{variant}` | Explicit fixture scores |
| Health | GET `/health` | Preview status |

Live model/embedding fields are read-only for backend compatibility. BGE and FAISS/Chroma options configure simulated index workflows only. No detector is trained by the frontend; recommendations are hypotheses about suitability.

## Data contracts

- Label 0 means normal, 1 anomaly. Training labels must be normal; test labels are optional.
- Empty text, invalid/boolean labels, exact duplicates, anomalous training rows, and exact split overlap block registration. Original records remain unchanged.
- Near-match review uses configurable lexical Jaccard on the first 100 records of each split, with up to 20 flags. It is not an exhaustive semantic check.
- Browser inspection supports 10 MB files. Batches support 1–100 rows and 1 MB files, with a `text` field of at most 8,000 characters. Use Datasets for other field mappings.
- Single and batch detection run both RAG and no-retrieval variants (two LLM calls per sample). The single-text view displays independent score, explanation, prompt, and error cards; batch exports include both scores and explanations. The session Evaluation link currently evaluates RAG scores. Prediction requests contain only text and configuration. Test labels are joined after inference. Reference category metadata is available for inspection but hidden from the live prediction prompt.
- Retrieval excludes all normalized exact self-matches. Test records never enter reference memory. Batch cancellation stops subsequent requests; an in-flight server call may still finish.
- AUROC and non-interpolated AP are computed from saved scores with ties grouped. Two classes are required in this interface. Failures and unlabeled rows reduce coverage. Threshold changes do not change ranking metrics.
- LLM scores are uncalibrated. Classical detector scores may use different scales; interpret threshold-based metrics in the detector's score units.
- Saved category-tagged CLI runs are marked legacy. Classical baseline results are distinguished from LLM comparisons. Preview values are never presented as research findings.

## Code and tests

`layouts/` owns navigation; `pages/` contains eight lazily loaded pages; `components/` and `charts/` provide shared UI. `hooks/use-workspace.tsx` isolates connection modes and session data. `utils/data.ts` handles parsing, validation, and metrics. `types/` defines contracts, and `services/demo.ts` contains explicitly illustrative fixtures.

Run `npm.cmd --prefix frontend run test`, `npm.cmd --prefix frontend run build`, and `npm.cmd --prefix frontend run lint`. Backend checks: `./.venv/Scripts/python.exe -m pytest -q`. Tests cover strict parsing, duplicates, overlaps, ties, failed-score coverage, label-free prediction payloads, prompt privacy, self-exclusion, and saved-run allowlists.

## Docker

From the repository root, `docker compose up -d --build` builds this frontend inside a Node build stage and packages the output with FastAPI. Open http://localhost:8180. No separate frontend process or prebuilt local `dist` is needed.
