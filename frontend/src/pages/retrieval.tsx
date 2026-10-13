import { useState } from "react";
import { Link } from "react-router-dom";
import { ArrowRight, Download, Search, ShieldCheck } from "lucide-react";
import { useWorkspace } from "../hooks/use-workspace";
import {
  Button,
  Empty,
  Field,
  Loading,
  Notice,
  PageHeader,
  Panel,
  References,
  Status,
} from "../components/common";
import { SimilarityChart } from "../charts/research-charts";
import { download } from "../lib/utils";
import type { Reference } from "../types";
export function Retrieval() {
  const w = useWorkspace(),
    [query, setQuery] = useState(w.latest?.query ?? ""),
    [result, setResult] = useState<{
      references: Reference[];
      prompt: string;
    } | null>(null),
    [busy, setBusy] = useState(false),
    [error, setError] = useState("");
  const refs = result?.references ?? w.latest?.response.retrieved ?? [],
    prompt = result?.prompt ?? w.latest?.response.rag?.prompt;
  async function search() {
    if (!w.dataset) return;
    setBusy(true);
    setError("");
    try {
      setResult(
        await w.api.retrieve(
          query,
          w.dataset.id,
          w.preferences.topK,
          w.dataset.local ? w.dataset.splits.train.preview : undefined,
        ),
      );
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <PageHeader
        eyebrow="RETRIEVAL TRANSPARENCY"
        title="Retrieval explorer"
        description="Follow the context from query to prompt. Inspect exactly what the model can see."
        actions={
          <Button
            variant="outline"
            disabled={!refs.length}
            onClick={() =>
              download("retrieval-evidence", {
                query,
                references: refs,
                prompt,
                mode: w.preferences.mode,
              })
            }
          >
            <Download />
            Export evidence
          </Button>
        }
      />
      <div className="flow-strip">
        {[
          "Query text",
          "Query embedding",
          "Cosine search",
          "Normal references",
          "Grounded prompt",
        ].map((s, i) => (
          <span key={s}>
            {i > 0 && <ArrowRight size={14} />}
            <strong>{s}</strong>
          </span>
        ))}
      </div>
      <Panel
        title="Search the reference memory"
        description="This runs retrieval only. No LLM inference call is made."
      >
        <div className="retrieval-query">
          <textarea
            aria-label="Retrieval query"
            placeholder="Enter a query, or run anomaly detection first…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            maxLength={8000}
          />
          <div>
            <Field label="Top-K">
              <select
                value={w.preferences.topK}
                onChange={(e) =>
                  w.setPreferences({
                    ...w.preferences,
                    topK: Number(e.target.value),
                  })
                }
              >
                {[1, 2, 3, 5, 8, 10].map((k) => (
                  <option key={k}>{k}</option>
                ))}
              </select>
            </Field>
            <Button
              disabled={busy || !query.trim() || !w.dataset?.index.ready}
              onClick={() => void search()}
            >
              <Search />
              Search references
            </Button>
          </div>
        </div>
        {error && <Notice tone="red">{error}</Notice>}
      </Panel>
      {busy ? (
        <Loading text="Searching the validated training memory…" />
      ) : (
        <div className="grid-main">
          <Panel
            title="Nearest normal references"
            description={`${refs.length} returned · train split only`}
            action={<Status tone="green">Self-match excluded</Status>}
          >
            {refs.length ? (
              <References
                references={refs}
                citations={w.latest?.response.rag?.citations}
              />
            ) : (
              <Empty
                title="Inspect your model’s context"
                description="Search for a text to see ranked reference documents, cosine distances, and provenance."
              />
            )}
          </Panel>
          <Panel
            title="Similarity evidence"
            description="Cosine similarity = 1 − cosine distance"
          >
            {refs.length ? (
              <SimilarityChart
                data={refs.map((r) => ({
                  name: `#${r.number}`,
                  similarity: 1 - r.distance,
                }))}
              />
            ) : (
              <Empty
                title="No similarity data"
                description="Real distances will appear after a search."
              />
            )}
            {refs.length > 0 &&
              Math.max(...refs.map((r) => 1 - r.distance)) <
                w.preferences.similarityThreshold && (
                <Notice tone="amber">
                  All returned references fall below your review threshold (
                  {w.preferences.similarityThreshold.toFixed(2)}). Inspect this
                  context carefully; this is a review indicator, not an anomaly
                  prediction.
                </Notice>
              )}
            <Notice>
              High semantic similarity does not establish whether a text is
              normal or anomalous.
            </Notice>
            <div className="metadata-list">
              <div>
                <span>Reference source</span>
                <strong>{w.dataset?.name ?? "—"}</strong>
              </div>
              <div>
                <span>Index version</span>
                <code>{w.dataset?.index.version ?? "—"}</code>
              </div>
              <div>
                <span>Ground-truth test labels</span>
                <Status tone="green">Excluded</Status>
              </div>
              <div>
                <span>Reference labels in prompt</span>
                <Status tone="green">Excluded</Status>
              </div>
            </div>
          </Panel>
        </div>
      )}
      <Panel
        title="Grounded prompt preview"
        description="The actual inference template and context returned by the service."
        action={<ShieldCheck size={18} />}
      >
        {prompt ? (
          <pre className="prompt-preview">{prompt}</pre>
        ) : (
          <Empty
            title="No prompt generated yet"
            description="Search references or complete a detection to inspect the prompt."
            action={
              <Button asChild variant="outline">
                <Link to="/detection">Open anomaly detection</Link>
              </Button>
            }
          />
        )}
      </Panel>
    </>
  );
}
