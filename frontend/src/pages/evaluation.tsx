import { useEffect, useState } from "react";
import {
  BarChart3,
  Download,
  FileCheck2,
  FlaskConical,
  Target,
} from "lucide-react";
import { useWorkspace } from "../hooks/use-workspace";
import {
  Button,
  DataTable,
  Empty,
  Field,
  Loading,
  Metric,
  Notice,
  PageHeader,
  Panel,
  Status,
} from "../components/common";
import { ComparisonChart, CurveChart } from "../charts/research-charts";
import { download, number, title } from "../lib/utils";
import { evaluate } from "../utils/data";
import type { Prediction } from "../types";
export function Evaluation() {
  const w = useWorkspace();
  const [threshold, setThreshold] = useState(w.preferences.threshold);
  const [selectedRun, setRunId] = useState<string | null>(null),
    [selectedVariant, setVariant] = useState(""),
    [predictions, setPredictions] = useState<Prediction[]>([]),
    [busy, setBusy] = useState(false),
    [error, setError] = useState("");
  const experiments = w.experiments.filter((e) => e.dataset === w.dataset?.id);
  const runId =
    selectedRun ??
    (w.batch.length ? "session-batch" : (experiments[0]?.id ?? ""));
  const run = experiments.find((e) => e.id === runId);
  const variants =
    run?.results.map((r) => ({
      id: r.model ?? `k${r.k}`,
      label: r.model
        ? title(r.model)
        : r.k === 0
          ? "Zero-shot baseline"
          : `RA-ZAD � k=${r.k}`,
    })) ?? [];
  const variant = variants.some((v) => v.id === selectedVariant)
    ? selectedVariant
    : (variants[0]?.id ?? "");
  useEffect(() => {
    let active = true;
    setError("");
    setPredictions([]);
    setBusy(false);
    if (runId === "session-batch") {
      setPredictions(w.batch);
      return;
    }
    if (!runId || !variant) return;
    setBusy(true);
    w.api
      .predictions(runId, variant)
      .then((p) => active && setPredictions(p))
      .catch((e) => active && setError(String(e)))
      .finally(() => active && setBusy(false));
    return () => {
      active = false;
    };
  }, [runId, variant, w.api, w.batch]);
  const m = evaluate(predictions, threshold),
    valid = m.auroc !== null;
  return (
    <>
      <PageHeader
        eyebrow="MEASURED OUTCOMES"
        title="Research evaluation"
        description="Compare saved predictions against held-out labels. Every number has an experiment behind it."
        actions={
          <Button
            variant="outline"
            disabled={!predictions.length}
            onClick={() =>
              download("evaluation", {
                run,
                variant,
                threshold: threshold,
                mode: w.preferences.mode,
                metrics: m,
              })
            }
          >
            <Download />
            Export metrics
          </Button>
        }
      />
      <div className="filter-bar">
        <Field label="Experiment">
          <select value={runId} onChange={(e) => setRunId(e.target.value)}>
            <option value="">Select an experiment</option>
            {experiments.map((e) => (
              <option key={e.id} value={e.id}>
                {e.id}
              </option>
            ))}
            {w.batch.length > 0 && (
              <option value="session-batch">Current session batch</option>
            )}
          </select>
        </Field>
        <Field label="Prediction variant">
          <select
            value={variant}
            onChange={(e) => setVariant(e.target.value)}
            disabled={runId === "session-batch"}
          >
            {runId === "session-batch" ? (
              <option>RA-ZAD session predictions</option>
            ) : variants.length ? (
              variants.map((v) => (
                <option key={v.id} value={v.id}>
                  {v.label}
                </option>
              ))
            ) : (
              <option value="">No variants</option>
            )}
          </select>
        </Field>
        <Field label="Decision threshold">
          <input
            type="number"
            min={run?.command === "baselines" ? undefined : 0}
            max={run?.command === "baselines" ? undefined : 1}
            step={0.05}
            value={threshold}
            onChange={(e) => {
              const n = Number(e.target.value);
              if (Number.isFinite(n))
                setThreshold(
                  run?.command === "baselines"
                    ? n
                    : Math.max(0, Math.min(1, n)),
                );
            }}
          />
        </Field>
        <Status tone={w.preferences.mode === "demo" ? "amber" : "green"}>
          {w.preferences.mode === "demo"
            ? "Illustrative data"
            : "Recorded predictions"}
        </Status>
      </div>
      {error && <Notice tone="red">{error}</Notice>}
      {run?.command === "baselines" && (
        <Notice>
          Classical detectors return raw anomaly scores with different scales.
          Set the decision threshold in this detector's score units; the default
          is not a validated operating point. AUROC and AP are
          threshold-independent.
        </Notice>
      )}
      {run?.legacy_reference_categories && (
        <Notice tone="amber">
          Legacy experiment: these saved runs used category-tagged references.
          Current dashboard inference hides reference labels. Compare
          configurations explicitly.
        </Notice>
      )}
      {busy ? (
        <Loading text="Loading saved predictions…" />
      ) : !predictions.length ? (
        <Empty
          title="No predictions selected"
          description="Choose a saved run or generate a labeled batch in Anomaly Detection. Missing experiments and unimplemented baselines are never filled with mock numbers in Live mode."
        />
      ) : (
        <>
          <div className="metrics-grid">
            <Metric
              label="AUROC"
              value={m.auroc?.toFixed(3) ?? "Not available"}
              note="Threshold-independent ranking"
              icon={<BarChart3 size={18} />}
            />
            <Metric
              label="AUPRC / average precision"
              value={m.auprc?.toFixed(3) ?? "Not available"}
              note="Non-interpolated average precision"
              icon={<Target size={18} />}
            />
            <Metric
              label="F1-score"
              value={m.valid ? m.f1.toFixed(3) : "Not available"}
              note={`Precision ${m.precision.toFixed(3)} · recall ${m.recall.toFixed(3)}`}
              icon={<FlaskConical size={18} />}
            />
            <Metric
              label="Evaluation coverage"
              value={`${m.total ? ((m.valid / m.total) * 100).toFixed(1) : 0}%`}
              note={`${m.valid} valid labeled scores / ${m.total} predictions`}
              icon={<FileCheck2 size={18} />}
            />
          </div>
          {!valid && (
            <Notice tone="amber">
              AUROC and AP require both normal and anomalous labeled examples in
              this interface. This subset cannot establish ranking performance.
            </Notice>
          )}
          <div className="two-columns">
            <Panel
              title="ROC curve"
              description="True positive rate versus false positive rate"
            >
              {valid ? (
                <CurveChart data={m.curve} />
              ) : (
                <Empty
                  title="Two classes required"
                  description="Add both normal and anomalous labeled examples."
                />
              )}
            </Panel>
            <Panel
              title="Precision–recall curve"
              description="Precision at each observed score threshold"
            >
              {valid ? (
                <CurveChart data={m.curve} pr />
              ) : (
                <Empty
                  title="Two classes required"
                  description="No artificial curve is drawn for undefined metrics."
                />
              )}
            </Panel>
          </div>
          <div className="grid-main">
            <Panel
              title="Experiment comparison"
              description="Saved metric summaries; compare only compatible protocols."
            >
              {run ? (
                <ComparisonChart
                  data={run.results
                    .filter((r) => r.metrics)
                    .map((r) => ({
                      name:
                        r.model ??
                        (r.k === 0 ? "Zero-shot" : `RA-ZAD k=${r.k}`),
                      auroc: r.metrics?.auroc ?? null,
                      auprc: r.metrics?.auprc ?? null,
                    }))}
                />
              ) : (
                <Notice>
                  Run a matching baseline on the same sample IDs to obtain a
                  valid paired comparison.
                </Notice>
              )}
            </Panel>
            <Panel
              title="Confusion matrix"
              description={`Threshold ${threshold.toFixed(2)} · labeled valid predictions only`}
            >
              <div className="matrix">
                <div />
                <small>PRED. NORMAL</small>
                <small>PRED. ANOMALY</small>
                <small>ACTUAL NORMAL</small>
                <div className="correct">
                  <strong>{m.tn}</strong>
                  <span>True negative</span>
                </div>
                <div>
                  <strong>{m.fp}</strong>
                  <span>False positive</span>
                </div>
                <small>ACTUAL ANOMALY</small>
                <div>
                  <strong>{m.fn}</strong>
                  <span>False negative</span>
                </div>
                <div className="correct">
                  <strong>{m.tp}</strong>
                  <span>True positive</span>
                </div>
              </div>
            </Panel>
          </div>
          <Panel
            title="Per-sample predictions"
            description="Labels are used only here, after predictions have been generated."
            action={
              <Button
                size="sm"
                variant="outline"
                onClick={() =>
                  download("evaluated-predictions", predictions, "csv")
                }
              >
                <Download />
                Export CSV
              </Button>
            }
          >
            <DataTable
              rows={predictions}
              columns={[
                { key: "id", label: "Sample ID" },
                {
                  key: "score",
                  label: "Anomaly score",
                  render: (r) => r.score?.toFixed(4) ?? "—",
                },
                {
                  key: "label",
                  label: "Ground truth",
                  render: (r) =>
                    r.label === 1
                      ? "Anomaly"
                      : r.label === 0
                        ? "Normal"
                        : "Not labeled",
                },
                {
                  key: "predicted",
                  label: "Prediction",
                  sortValue: (r) =>
                    r.score == null ? -1 : Number(r.score >= threshold),
                  render: (r) =>
                    r.score == null
                      ? "Unavailable"
                      : r.score >= threshold
                        ? "Anomaly"
                        : "Normal",
                },
                {
                  key: "error",
                  label: "Status",
                  render: (r) => (
                    <Status
                      tone={r.error ? "red" : r.pending ? "amber" : "green"}
                    >
                      {r.error ? "Failed" : r.pending ? "Pending" : "Saved"}
                    </Status>
                  ),
                },
              ]}
            />
          </Panel>
          <Panel title="Experiment provenance">
            <div className="provenance-grid">
              {[
                ["Run", runId],
                ["Dataset", w.dataset?.name ?? "—"],
                ["Training rows", number(run?.n_train)],
                [
                  "Embedding",
                  run?.embedding ?? w.dataset?.index.embedding ?? "—",
                ],
                [
                  "LLM",
                  run?.command === "baselines"
                    ? "Not used"
                    : (run?.llm?.model ??
                      w.catalog?.configuration.llm ??
                      "Not recorded"),
                ],
                ["Seed", run?.seed ?? "—"],
                [
                  "Updated",
                  run?.updated_at
                    ? new Date(run.updated_at).toLocaleString()
                    : "Current session",
                ],
                ["Threshold", threshold],
              ].map(([k, v]) => (
                <div key={k}>
                  <span>{k}</span>
                  <strong>{v}</strong>
                </div>
              ))}
            </div>
          </Panel>
        </>
      )}
    </>
  );
}
