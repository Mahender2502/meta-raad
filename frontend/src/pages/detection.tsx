import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import {
  Activity,
  ArrowUpRight,
  Download,
  FileText,
  Layers3,
  Play,
  ShieldCheck,
  Square,
  Upload,
} from "lucide-react";
import { useWorkspace } from "../hooks/use-workspace";
import {
  Button,
  DataTable,
  Empty,
  Field,
  Loading,
  Notice,
  PageHeader,
  Panel,
  References,
  Status,
} from "../components/common";
import { DetectionComparison } from "../components/detection-comparison";
import { download, number } from "../lib/utils";
import { mapRows, parseFile, validateRows } from "../utils/data";
import type { Prediction, Row } from "../types";
const n24Examples = {
  Science:
    "Scientists are using a new telescope to study the atmosphere of a distant planet.",
  Food: "Mix the flour and butter, fold in the eggs, then simmer tomato sauce before adding fresh basil.",
  Sports:
    "The home team won the championship after scoring in the final minute.",
};
export function DetectionPage() {
  const w = useWorkspace(),
    [tab, setTab] = useState<"single" | "batch">("single"),
    [text, setText] = useState(""),
    [setting, setSetting] = useState("normal_anomaly"),
    [busy, setBusy] = useState(false),
    [error, setError] = useState(""),
    [rows, setRows] = useState<Row[]>([]),
    [batchFile, setBatchFile] = useState(""),
    [selected, setSelected] = useState<Prediction | null>(null);
  const abort = useRef<AbortController | null>(null),
    file = useRef<HTMLInputElement>(null);
  const d = w.dataset,
    latest = w.latest;
  const examples = d?.id === "bbc" ? {
    Business: "The company reported higher annual profits as demand for its products increased.",
    Entertainment: "The actor won the best performance award at the film festival for her latest movie.",
    Sport: "The football team won the league title after scoring twice in the final match.",
  } : n24Examples;
  useEffect(() => () => abort.current?.abort(), []);
  const blocked =
    !d ||
    !d.index.ready ||
    !d.validation.normal_train_only;
  async function single() {
    if (!d || !text.trim()) return;
    setBusy(true);
    setError("");
    const started = performance.now();
    abort.current = new AbortController();
    try {
      const response = await w.api.detect(
        text,
        d.id,
        w.preferences.topK,
        setting,
        true,
        d.local ? d.splits.train.preview : undefined,
        abort.current.signal,
      );
      if (abort.current.signal.aborted) return;
      w.setLatest({
        query: text,
        response,
        duration: (performance.now() - started) / 1000,
        timestamp: new Date().toISOString(),
        dataset: d.name,
        indexVersion: d.index.version,
        embedding: d.index.embedding,
        mode: w.preferences.mode,
      });
      if (response.rag?.error) setError(response.rag.error);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }
  async function load(f?: File) {
    if (!f) return;
    setError("");
    try {
      if (f.size > 1024 * 1024)
        throw Error("Batch upload supports up to 1 MB and 100 records.");
      const records = parseFile(await f.text(), f.name);
      if (!records.length || records.length > 100)
        throw Error("Use a file with 1–100 records.");
      const mapped = mapRows(records, "text", "label", "test");
      if (mapped.some((r) => !r.text.trim() || r.text.length > 8000))
        throw Error(
          "Every record needs a nonempty text field, up to 8,000 characters. Map other fields on the Datasets page first.",
        );
      setRows(mapped);
      setBatchFile(f.name);
    } catch (e) {
      setError(String(e));
    }
  }
  async function batch() {
    if (!d) return;
    setBusy(true);
    setError("");
    w.setBatch([]);
    const saved: Prediction[] = [];
    abort.current = new AbortController();
    for (const row of rows) {
      if (abort.current.signal.aborted) break;
      try {
        const response = await w.api.detect(
          row.text,
          d.id,
          w.preferences.topK,
          setting,
          true,
          d.local ? d.splits.train.preview : undefined,
          abort.current.signal,
        );
        if (abort.current.signal.aborted) break;
        const prediction = {
          id: row.id,
          text: row.text,
          score: response.rag?.score ?? null,
          error: response.rag?.error,
          reason: response.rag?.reason,
          baselineScore: response.baseline?.score ?? null,
          baselineError: response.baseline?.error,
          baselineReason: response.baseline?.reason,
        }; // Labels are joined only after a prediction has returned.
        saved.push({ ...prediction, label: row.label });
      } catch (e) {
        if (abort.current.signal.aborted) break;
        saved.push({
          id: row.id,
          text: row.text,
          label: row.label,
          score: null,
          error: String(e),
          baselineScore: null,
          baselineError: String(e),
        });
      }
      w.setBatch([...saved]);
    }
    setBusy(false);
  }
  const overlap =
    rows.length && d?.local
      ? validateRows(
          rows,
          d.splits.train.preview,
          w.preferences.nearDuplicateThreshold,
        ).overlap
      : 0;
  return (
    <>
      <PageHeader
        eyebrow="RA-ZAD · PRIMARY WORKSPACE"
        title="Anomaly detection"
        description="Give every prediction context. Compare a text against normal training references."
        actions={
          <Status tone={w.preferences.mode === "demo" ? "amber" : "blue"}>
            {w.preferences.mode === "demo"
              ? "Illustrative simulator"
              : "Live LLM inference"}
          </Status>
        }
      />
      <div className="tabs">
        <button
          onClick={() => setTab("single")}
          className={tab === "single" ? "active" : ""}
          disabled={busy}
        >
          <FileText size={16} />
          Single text
        </button>
        <button
          onClick={() => setTab("batch")}
          className={tab === "batch" ? "active" : ""}
          disabled={busy}
        >
          <Layers3 size={16} />
          Batch detection
        </button>
      </div>
      {blocked && (
        <Notice tone="amber">
          A validated normal-training index is
          required. <Link to="/vectors">Inspect your reference memory.</Link>
        </Notice>
      )}
      {!!d?.validation.exact_overlap && (
        <Notice tone="amber">
          The source benchmark has {d.validation.exact_overlap} exact train/test
          overlap(s). Retrieval excludes exact query matches. CLI held-out
          evaluation excludes overlapping test rows; batch uploads are checked separately.
        </Notice>
      )}
      {tab === "single" && latest && (
        <DetectionComparison
          result={latest}
          threshold={w.preferences.threshold}
        />
      )}
      <div className="detection-layout">
        <div>
          <Panel
            title={
              tab === "single" ? "Analyze a text sample" : "Run a test batch"
            }
            description={
              tab === "single"
                ? "The query is sent without ground-truth labels."
                : "Predictions are generated sequentially; labels are joined afterward."
            }
            action={<span className="section-number">01 / INPUT</span>}
          >
            {tab === "single" ? (
              <>
                <textarea
                  className="query-input"
                  aria-label="Text to analyze"
                  placeholder="Paste a news article, message, or text sample here…"
                  value={text}
                  maxLength={8000}
                  onChange={(e) => setText(e.target.value)}
                  disabled={busy}
                />
                <div className="input-bottom">
                  <div className="example-chips">
                    <span>Try an example</span>
                    {Object.entries(examples).map(([name, value]) => (
                      <button
                        key={name}
                        onClick={() => setText(value)}
                        disabled={busy}
                      >
                        {name}
                      </button>
                    ))}
                  </div>
                  <small>{text.length.toLocaleString()} / 8,000</small>
                </div>
              </>
            ) : (
              <>
                <input
                  ref={file}
                  type="file"
                  accept=".jsonl,.csv"
                  className="sr-only"
                  aria-label="Batch test file"
                  onChange={(e) => void load(e.target.files?.[0])}
                />
                <button
                  className="drop-zone compact"
                  onClick={() => file.current?.click()}
                  disabled={busy}
                >
                  <Upload size={23} />
                  <strong>
                    {batchFile || "Choose a test JSONL or CSV file"}
                  </strong>
                  <small>
                    {rows.length
                      ? `${rows.length} queries ready`
                      : "Up to 100 records · text required · label optional"}
                  </small>
                </button>
                {d?.local && d.splits.test.count > 0 && (
                  <Button
                    variant="ghost"
                    size="sm"
                    disabled={busy}
                    onClick={() => {
                      setRows(d.splits.test.preview.slice(0, 100));
                      setBatchFile("Selected test split");
                    }}
                  >
                    Use selected preview test split
                  </Button>
                )}
                {!!overlap && (
                  <Notice tone="red">
                    {overlap} exact overlaps with the reference split. Resolve
                    these before running this batch.
                  </Notice>
                )}
                <Notice>
                  Each row compares both methods using two LLM calls in Live
                  mode. Stop prevents further requests; an in-flight request may
                  still finish on the server.
                </Notice>
              </>
            )}
            <div className="form-grid detection-options">
              <Field label="Category knowledge">
                <select
                  value={setting}
                  onChange={(e) => setSetting(e.target.value)}
                  disabled={busy}
                >
                  <option value="normal_anomaly">
                    Normal + anomaly categories
                  </option>
                  <option value="normal_only">Normal categories only</option>
                </select>
              </Field>
              <Field label="Top-K references">
                <select
                  value={w.preferences.topK}
                  disabled={busy}
                  onChange={(e) =>
                    w.setPreferences({
                      ...w.preferences,
                      topK: Number(e.target.value),
                    })
                  }
                >
                  {[1, 2, 3, 5, 8, 10].map((k) => (
                    <option key={k} value={k}>
                      {k} nearest examples
                    </option>
                  ))}
                </select>
              </Field>
            </div>
            <Notice>
              Both methods use the same text, LLM, and category knowledge. Each
              comparison makes two LLM calls: one with retrieved context and one
              without.
            </Notice>
            <div className="run-bar">
              <span>
                <ShieldCheck size={14} /> Training-only retrieval · query
                self-match excluded
              </span>
              <Button
                disabled={
                  busy ||
                  blocked ||
                  (tab === "single" ? !text.trim() : !rows.length || !!overlap)
                }
                onClick={() => void (tab === "single" ? single() : batch())}
              >
                {busy ? <Activity className="spin" /> : <Play size={15} />}{" "}
                {busy
                  ? "Analyzing…"
                  : tab === "single"
                    ? "Compare with / without RAG"
                    : "Compare batch"}
              </Button>
              {busy && tab === "batch" && (
                <Button
                  variant="outline"
                  onClick={() => abort.current?.abort()}
                >
                  <Square size={14} />
                  Stop
                </Button>
              )}
            </div>
            {error && <Notice tone="red">{error}</Notice>}
          </Panel>
          {busy && tab === "single" && (
            <Loading text="Retrieving references and generating an explanation…" />
          )}
          {tab === "batch" && (
            <Panel
              title="Batch results"
              description={`${w.batch.length} / ${rows.length} processed · ${w.batch.filter((p) => p.error || p.baselineError).length} failed`}
              action={
                <div className="button-row">
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={!w.batch.length}
                    onClick={() => download("predictions", w.batch, "csv")}
                  >
                    <Download />
                    CSV
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={!w.batch.length}
                    onClick={() => download("predictions", w.batch, "jsonl")}
                  >
                    JSONL
                  </Button>
                </div>
              }
            >
              <progress max={Math.max(1, rows.length)} value={w.batch.length} />
              {w.batch.length ? (
                <>
                  <DataTable
                    rows={w.batch}
                    onRow={setSelected}
                    columns={[
                      { key: "id", label: "Sample ID" },
                      {
                        key: "text",
                        label: "Text",
                        render: (r) => (
                          <span className="table-text">{r.text}</span>
                        ),
                      },
                      {
                        key: "score",
                        label: "With RAG",
                        render: (r) => r.score?.toFixed(3) ?? "—",
                      },
                      {
                        key: "baselineScore",
                        label: "Without RAG",
                        render: (r) => r.baselineScore?.toFixed(3) ?? "�",
                      },
                      {
                        key: "comparison",
                        label: "Classifications",
                        sortValue: (r) =>
                          `${r.score == null ? -1 : Number(r.score >= w.preferences.threshold)}:${r.baselineScore == null ? -1 : Number(r.baselineScore >= w.preferences.threshold)}`,
                        render: (r) =>
                          `RAG: ${r.score == null ? "Unavailable" : r.score >= w.preferences.threshold ? "Anomaly" : "Normal"} / LLM: ${r.baselineScore == null ? "Unavailable" : r.baselineScore >= w.preferences.threshold ? "Anomaly" : "Normal"}`,
                      },
                      {
                        key: "error",
                        label: "Status",
                        render: (r) => (
                          <Status
                            tone={
                              r.error || r.baselineError
                                ? "red"
                                : r.baselineScore == null
                                  ? "amber"
                                  : "green"
                            }
                          >
                            {r.error || r.baselineError
                              ? "Partial / failed"
                              : r.baselineScore == null
                                ? "No baseline"
                                : "Both saved"}
                          </Status>
                        ),
                      },
                    ]}
                  />
                  {selected && (
                    <div className="prediction-detail">
                      <strong>
                        {selected.id} · score{" "}
                        {selected.score?.toFixed(3) ?? "unavailable"}
                      </strong>
                      <p>{selected.text}</p>
                      <div className="comparison-grid">
                        <div>
                          <strong>
                            Without RAG �{" "}
                            {selected.baselineScore?.toFixed(3) ??
                              "Unavailable"}
                          </strong>
                          <p>
                            {selected.baselineReason ??
                              selected.baselineError ??
                              "Not run"}
                          </p>
                        </div>
                        <div>
                          <strong>
                            With RAG �{" "}
                            {selected.score?.toFixed(3) ?? "Unavailable"}
                          </strong>
                          <p>
                            {selected.reason ??
                              selected.error ??
                              "No explanation returned"}
                          </p>
                        </div>
                      </div>
                      {selected.error && (
                        <Notice tone="red">{selected.error}</Notice>
                      )}
                    </div>
                  )}
                  <Button variant="ghost" asChild>
                    <Link to="/evaluation">
                      Evaluate RAG batch <ArrowUpRight size={14} />
                    </Link>
                  </Button>
                </>
              ) : (
                <Empty
                  title="No batch predictions yet"
                  description="Choose a test file and run the batch. Prediction requests contain text and configuration only."
                />
              )}
            </Panel>
          )}
        </div>
        <div>
          <Panel
            title="Inference configuration"
            action={<span className="section-number">CONFIGURATION</span>}
          >
            <div className="metadata-list">
              {[
                ["Dataset", d?.name ?? "Not selected"],
                ["Embedding", d?.index.embedding ?? "Not available"],
                [
                  "Language model",
                  w.catalog?.configuration.llm ?? "Not connected",
                ],
                ["Provider", w.catalog?.configuration.provider ?? "—"],
                [
                  "Reference memory",
                  `${number(d?.index.count)} normal vectors`,
                ],
                ["Threshold", w.preferences.threshold.toFixed(2)],
              ].map(([k, v]) => (
                <div key={k}>
                  <span>{k}</span>
                  <strong>{v}</strong>
                </div>
              ))}
            </div>
            <p className="fine-print">
              The backend controls model and embedding compatibility. Threshold
              and Top-K are configurable in Settings.
            </p>
            <Button asChild variant="outline" className="full-width">
              <Link to="/settings">Configuration settings</Link>
            </Button>
          </Panel>
          <Panel
            title="Retrieved evidence"
            description="Examples are evidence. Explanations are model interpretation."
            action={<span className="section-number">02 / CONTEXT</span>}
          >
            {latest?.response.retrieved.length ? (
              <>
                <References
                  references={latest.response.retrieved}
                  citations={latest.response.rag?.citations}
                />
                <Button asChild variant="ghost" className="full-width">
                  <Link to="/retrieval">
                    Open retrieval explorer <ArrowUpRight size={14} />
                  </Link>
                </Button>
              </>
            ) : (
              <Empty
                title="Your evidence will appear here"
                description="Run a query to inspect the nearest normal training examples and their cosine similarities."
              />
            )}
          </Panel>
        </div>
      </div>
    </>
  );
}
