import { useState } from "react";
import { Link } from "react-router-dom";
import {
  Boxes,
  Database,
  Layers3,
  Play,
  RefreshCw,
  ShieldCheck,
} from "lucide-react";
import { useWorkspace } from "../hooks/use-workspace";
import {
  Button,
  Empty,
  Field,
  Metric,
  Notice,
  PageHeader,
  Panel,
  Status,
  Loading,
} from "../components/common";
import { number } from "../lib/utils";
export function Vectors() {
  const w = useWorkspace(),
    d = w.dataset,
    [busy, setBusy] = useState(false),
    [message, setMessage] = useState(""),
    [error, setError] = useState("");
  const preview = w.preferences.mode === "demo";
  async function build() {
    if (!d) return;
    setBusy(true);
    setError("");
    setMessage("");
    try {
      if (
        !d.validation.normal_train_only ||
        d.validation.exact_overlap > 0 ||
        d.splits.train.duplicates
      )
        throw Error(
          "Resolve the training validation report and exact split overlaps before building an index.",
        );
      await w.api.buildIndex();
      w.updateDataset({
        ...d,
        local: true,
        index: {
          ...d.index,
          ready: true,
          count: d.splits.train.count,
          embedding: w.preferences.embedding,
          provider: w.preferences.provider + " (simulated)",
          version: `preview-${Date.now()}`,
          created_at: new Date().toISOString(),
        },
        validation: { ...d.validation, vector_rows_aligned: true },
      });
      setMessage(
        "Preview index state created. No embeddings, ChromaDB collection, or FAISS index was built.",
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
        eyebrow="REFERENCE MEMORY"
        title="Vector database"
        description="Inspect your embedding space and the provenance of every normal reference."
        actions={
          <Button variant="outline" onClick={() => void w.refresh()}>
            <RefreshCw />
            Refresh status
          </Button>
        }
      />
      {!d ? (
        <Empty
          title="Choose a dataset first"
          description="Load a validated training split to inspect or configure the reference index."
          action={
            <Button asChild>
              <Link to="/datasets">Open datasets</Link>
            </Button>
          }
        />
      ) : (
        <>
          <div className="metrics-grid">
            <Metric
              label="Normal vectors"
              value={number(d.index.count)}
              note="Training references only"
              icon={<Database size={18} />}
            />
            <Metric
              label="Embedding dimensions"
              value={number(d.index.dimensions)}
              note="Reported by the vector source"
              icon={<Layers3 size={18} />}
            />
            <Metric
              label="Index status"
              value={
                <Status tone={d.index.ready ? "green" : "amber"}>
                  {d.index.ready ? "Ready" : "Not built"}
                </Status>
              }
              note={
                preview
                  ? "Simulated preview state"
                  : "Precomputed vectors are available"
              }
              icon={<Boxes size={18} />}
            />
            <Metric
              label="Reference split"
              value="Train only"
              note={
                d.validation.normal_train_only
                  ? "Normal-only validation passed"
                  : "Validation required"
              }
              icon={<ShieldCheck size={18} />}
            />
          </div>
          <div className="grid-main">
            <Panel
              title="Index configuration"
              description={
                preview
                  ? "Configure the simulated index workflow."
                  : "Active backend configuration; changing engines requires backend support."
              }
            >
              <div className="form-grid">
                <Field label="Dataset">
                  <input value={d.name} readOnly />
                </Field>
                <Field label="Training file / collection">
                  <input value={d.index.collection} readOnly />
                </Field>
                <Field label="Text column">
                  <input value="text" readOnly />
                </Field>
                <Field label="Label column">
                  <input value="label" readOnly />
                </Field>
                <Field label="Embedding model">
                  <select
                    disabled={!preview}
                    value={
                      preview ? w.preferences.embedding : d.index.embedding
                    }
                    onChange={(e) =>
                      w.setPreferences({
                        ...w.preferences,
                        embedding: e.target.value,
                      })
                    }
                  >
                    {!preview && <option>{d.index.embedding}</option>}
                    <option>BAAI/bge-base-en-v1.5</option>
                    <option>bert-base-uncased</option>
                  </select>
                </Field>
                <Field label="Vector provider">
                  <select
                    disabled={!preview}
                    value={preview ? w.preferences.provider : d.index.provider}
                    onChange={(e) =>
                      w.setPreferences({
                        ...w.preferences,
                        provider: e.target.value,
                      })
                    }
                  >
                    {!preview && <option>{d.index.provider}</option>}
                    <option>FAISS</option>
                    <option>ChromaDB</option>
                  </select>
                </Field>
                <Field label="Top-K references">
                  <input
                    type="number"
                    min={1}
                    max={10}
                    value={w.preferences.topK}
                    onChange={(e) =>
                      w.setPreferences({
                        ...w.preferences,
                        topK: Math.max(1, Math.min(10, Number(e.target.value))),
                      })
                    }
                  />
                </Field>
                <Field label="Similarity metric">
                  <select>
                    <option>Cosine similarity</option>
                  </select>
                </Field>
              </div>
              <Notice tone="green">
                Only normal training examples belong in this collection. Test
                records and evaluation labels are excluded.
              </Notice>
              <div className="button-row">
                <Button asChild variant="outline">
                  <Link to="/datasets">
                    <ShieldCheck />
                    Validate dataset
                  </Link>
                </Button>
                <Button
                  onClick={() => void build()}
                  disabled={!preview || busy || !d.validation.normal_train_only}
                >
                  <Play />
                  {busy
                    ? "Simulating index…"
                    : preview
                      ? "Build preview index"
                      : "Build index unavailable"}
                </Button>
                <Button asChild variant="ghost">
                  <Link to="/datasets">View indexed samples</Link>
                </Button>
              </div>
              {!preview && (
                <p className="fine-print">
                  The live backend loads its existing NumPy memory on demand.
                  ChromaDB/FAISS building is not implemented; Preview mode
                  demonstrates that workflow without claiming a real index.
                </p>
              )}
              {busy && (
                <Loading text="Simulating validation and index creation…" />
              )}
              {message && <Notice tone="amber">{message}</Notice>}
              {error && <Notice tone="red">{error}</Notice>}
            </Panel>
            <Panel
              title="Index provenance"
              description="Know what your retriever is searching"
            >
              <div className="metadata-list">
                {[
                  ["Collection", d.index.collection],
                  ["Version / fingerprint", d.index.version],
                  ["Dataset", d.name],
                  ["Permitted split", "Training"],
                  ["Embedding model", d.index.embedding],
                  [
                    "Created at",
                    d.index.created_at
                      ? new Date(d.index.created_at).toLocaleString()
                      : "Not recorded",
                  ],
                  [
                    "Vector file modified",
                    d.index.source_modified_at
                      ? new Date(d.index.source_modified_at).toLocaleString()
                      : "Not available",
                  ],
                  ["Exact query exclusion", "Enabled"],
                ].map(([k, v]) => (
                  <div key={k}>
                    <span>{k}</span>
                    <strong>{v}</strong>
                  </div>
                ))}
              </div>
              <Notice>
                File modification time is not an index creation timestamp. Live
                vectors are validated again when the memory loads.
              </Notice>
            </Panel>
          </div>
        </>
      )}
    </>
  );
}
