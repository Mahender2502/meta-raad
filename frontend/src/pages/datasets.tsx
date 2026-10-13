import { useState, useRef } from "react";
import {
  Download,
  FileJson,
  ShieldCheck,
  Upload,
  CheckCircle2,
} from "lucide-react";
import { useWorkspace } from "../hooks/use-workspace";
import {
  Button,
  DataTable,
  Empty,
  Field,
  Metric,
  Notice,
  PageHeader,
  Panel,
  Status,
} from "../components/common";
import { mapRows, parseFile, validateRows } from "../utils/data";
import { download, number } from "../lib/utils";
import type { Dataset, Row } from "../types";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "../components/ui/dialog";
export function Datasets() {
  const w = useWorkspace(),
    file = useRef<HTMLInputElement>(null),
    [raw, setRaw] = useState<Record<string, unknown>[]>([]),
    [name, setName] = useState(""),
    [textField, setTextField] = useState("text"),
    [labelField, setLabelField] = useState("label"),
    [split, setSplit] = useState<"train" | "test">("train"),
    [previewSplit, setPreviewSplit] = useState<"train" | "test">("train"),
    [error, setError] = useState(""),
    [message, setMessage] = useState(""),
    [selected, setSelected] = useState<Row | null>(null),
    [target, setTarget] = useState("new");
  const d = w.dataset,
    fields = raw.length ? Object.keys(raw[0]) : [];
  const mapped = mapRows(raw, textField, labelField, split);
  const existing = target === "selected" ? d : undefined;
  const other =
    existing?.splits[split === "train" ? "test" : "train"].preview ?? [];
  const validation = validateRows(
    mapped,
    other,
    w.preferences.nearDuplicateThreshold,
  );
  async function open(f?: File) {
    if (!f) return;
    setError("");
    setMessage("");
    if (f.size > 10 * 1024 * 1024) {
      setError(
        "Browser inspection supports files up to 10 MB. Use the Python data pipeline for larger files.",
      );
      return;
    }
    try {
      const r = parseFile(await f.text(), f.name);
      if (!r.length) throw Error("The file contains no records.");
      setRaw(r);
      setName(f.name);
      setTextField(
        Object.keys(r[0]).includes("text") ? "text" : Object.keys(r[0])[0],
      );
      setLabelField(Object.keys(r[0]).includes("label") ? "label" : "");
    } catch (e) {
      setError(String(e));
      setRaw([]);
    }
  }
  function add() {
    if (validation.blocked) {
      setError(
        "Resolve the validation blockers before registering this split. Original records have not been changed.",
      );
      return;
    }
    if (w.preferences.mode !== "demo") {
      setMessage(
        "Local validation complete. Dataset registration is not implemented by this backend. Enable Preview in Settings to exercise the full browser-only workflow.",
      );
      return;
    }
    const id = existing?.local ? existing.id : `upload_${Date.now()}`;
    const ds: Dataset = existing?.local
      ? structuredClone(existing)
      : {
          id,
          name: name.replace(/\.(jsonl|csv)$/i, ""),
          source: "Browser session upload",
          local: true,
          splits: {
            train: { count: 0, available: false, preview: [] },
            test: { count: 0, available: false, preview: [] },
          },
          validation: {
            normal_train_only: false,
            exact_overlap: 0,
            near_duplicates: "Not computed",
            vector_rows_aligned: false,
          },
          index: {
            ready: false,
            provider: w.preferences.provider + " (simulated)",
            embedding: w.preferences.embedding,
            dimensions: 768,
            count: 0,
            split: "train",
            collection: `${id}/normal_train`,
            version: "not-built",
            created_at: null,
          },
        };
    ds.splits[split] = {
      count: mapped.length,
      available: true,
      normal: validation.normal,
      anomaly: validation.anomaly,
      duplicates: validation.duplicates,
      empty_text: validation.empty,
      invalid_labels: validation.invalid,
      preview: mapped,
    };
    ds.validation.normal_train_only =
      ds.splits.train.available &&
      ds.splits.train.normal === ds.splits.train.count;
    ds.validation.exact_overlap = validation.overlap;
    ds.validation.near_duplicates = `${validation.near.length} lexical candidates in bounded review`;
    ds.index.ready = false;
    ds.index.count = 0;
    ds.validation.vector_rows_aligned = false;
    w.updateDataset(ds);
    w.selectDataset(ds.id);
    setTarget("selected");
    setMessage(
      `${split} split registered in preview session. No data was uploaded to a server or altered.`,
    );
  }
  return (
    <>
      <PageHeader
        eyebrow="DATA FOUNDATION"
        title="Datasets"
        description="Know your data before it reaches your model. Inspect, map, and validate every split."
        actions={
          <Button
            variant="outline"
            onClick={() =>
              download("dataset-validation", {
                dataset: d?.name,
                validation: d?.validation,
              })
            }
          >
            <Download />
            Export report
          </Button>
        }
      />
      <div className="metrics-grid three">
        <Metric
          label="Selected dataset"
          value={d?.name ?? "—"}
          note={d?.source}
          icon={<FileJson size={18} />}
        />
        <Metric
          label="Training / test records"
          value={`${number(d?.splits.train.count)} / ${number(d?.splits.test.count)}`}
          note="Each split has a distinct research role"
          icon={<ShieldCheck size={18} />}
        />
        <Metric
          label="Exact train/test overlap"
          value={number(d?.validation.exact_overlap)}
          note="Normalized text match · no automatic deletion"
          icon={<CheckCircle2 size={18} />}
        />
      </div>
      <div className="grid-main">
        <Panel
          title="Inspect a dataset"
          description="JSONL or CSV · up to 10 MB for browser validation"
        >
          <input
            ref={file}
            type="file"
            accept=".jsonl,.csv"
            className="sr-only"
            aria-label="Dataset file"
            onChange={(e) => void open(e.target.files?.[0])}
          />
          <button
            className="drop-zone"
            onClick={() => file.current?.click()}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              void open(e.dataTransfer.files[0]);
            }}
          >
            <span>
              <Upload size={23} />
            </span>
            <strong>Drop your dataset here, or browse files</strong>
            <small>Your original records are never modified.</small>
          </button>
          {raw.length > 0 && (
            <>
              <div className="file-loaded">
                <FileJson size={17} />
                <strong>{name}</strong>
                <span>{number(raw.length)} records</span>
              </div>
              <div className="form-grid">
                <Field label="Text field">
                  <select
                    value={textField}
                    onChange={(e) => setTextField(e.target.value)}
                  >
                    {fields.map((f) => (
                      <option key={f}>{f}</option>
                    ))}
                  </select>
                </Field>
                <Field label="Label field">
                  <select
                    value={labelField}
                    onChange={(e) => setLabelField(e.target.value)}
                  >
                    <option value="">No labels</option>
                    {fields.map((f) => (
                      <option key={f}>{f}</option>
                    ))}
                  </select>
                </Field>
                <Field label="Split role">
                  <select
                    value={split}
                    onChange={(e) =>
                      setSplit(e.target.value as "train" | "test")
                    }
                  >
                    <option value="train">Training / normal references</option>
                    <option value="test">Test / inference queries</option>
                  </select>
                </Field>
                <Field label="Register into">
                  <select
                    value={target}
                    onChange={(e) => setTarget(e.target.value)}
                  >
                    <option value="new">New preview dataset</option>
                    {d?.local && <option value="selected">{d.name}</option>}
                  </select>
                </Field>
              </div>
              <div className="validation-grid">
                {[
                  ["Empty text", validation.empty],
                  ["Invalid labels", validation.invalid],
                  ["Missing labels", validation.missing],
                  ["Exact duplicates", validation.duplicates],
                  ["Train/test overlap", validation.overlap],
                  ["Near-match flags", validation.near.length],
                ].map(([k, v]) => (
                  <div key={k}>
                    <span>{k}</span>
                    <strong className={Number(v) > 0 ? "amber-text" : ""}>
                      {v}
                    </strong>
                  </div>
                ))}
              </div>
              <Notice tone={validation.blocked ? "red" : "green"}>
                {validation.blocked
                  ? "Validation blocked. Training requires label 0 for every row. Empty texts, invalid labels, duplicates, and exact split overlap must be resolved."
                  : "Validation passed for this split. Test labels can be absent and are never used for prediction."}
              </Notice>
              <p className="fine-print">
                Near-match review uses lexical Jaccard similarity ≥{" "}
                {w.preferences.nearDuplicateThreshold}, on the first 100 rows
                per split. It flags candidates without removing them; it is not
                an exhaustive semantic check.
              </p>
              {validation.near.length > 0 && (
                <DataTable
                  rows={validation.near}
                  columns={[
                    { key: "train", label: "Training ID" },
                    { key: "test", label: "Test ID" },
                    {
                      key: "similarity",
                      label: "Similarity",
                      render: (r) => r.similarity.toFixed(3),
                    },
                  ]}
                />
              )}
              <Button onClick={add} disabled={validation.blocked}>
                <ShieldCheck />
                {w.preferences.mode === "demo"
                  ? "Register validated split"
                  : "Validate for backend integration"}
              </Button>
            </>
          )}
          {error && <Notice tone="red">{error}</Notice>}
          {message && <Notice tone="green">{message}</Notice>}
        </Panel>
        <Panel
          title="Split separation"
          description="A boundary that protects your experiment"
        >
          <div className="split-card">
            <Status tone="green">TRAINING</Status>
            <h3>Build the normal reference memory</h3>
            {d && (
              <p>
                <strong>
                  {number(d.splits.train.normal)} normal ·{" "}
                  {number(d.splits.train.anomaly)} anomalous
                </strong>
                <br />
                {number(d.splits.train.empty_text)} empty texts ·{" "}
                {number(d.splits.train.invalid_labels)} invalid labels ·{" "}
                {number(d.splits.train.duplicates)} duplicate records
              </p>
            )}
            <p>
              Only validated label-0 training records are permitted. Document
              IDs and source provenance stay attached.
            </p>
          </div>
          <div className="split-card">
            <Status tone="blue">TEST</Status>
            <h3>Query first. Evaluate afterward.</h3>
            {d && (
              <p>
                <strong>
                  {number(d.splits.test.normal)} normal ·{" "}
                  {number(d.splits.test.anomaly)} anomalous
                </strong>
                <br />
                {number(d.splits.test.empty_text)} empty texts ·{" "}
                {number(d.splits.test.invalid_labels)} invalid labels ·{" "}
                {number(d.splits.test.duplicates)} duplicate records
              </p>
            )}
            <p>
              Test records never enter the index. Ground-truth labels remain
              separate until scores are saved.
            </p>
          </div>
          <Notice tone="blue">
            Live dataset upload is not connected. File inspection is local;
            Preview mode supports session-only registration and simulated
            indexing.
          </Notice>
        </Panel>
      </div>
      <Panel
        title="Record preview"
        description={
          d
            ? `${d.name} · ${previewSplit} split · backend previews up to 50 rows`
            : "Select a dataset to inspect its records."
        }
        action={
          <div className="segmented">
            {(["train", "test"] as const).map((s) => (
              <button
                key={s}
                className={previewSplit === s ? "active" : ""}
                onClick={() => setPreviewSplit(s)}
              >
                {s === "train" ? "Training" : "Test"}
              </button>
            ))}
          </div>
        }
      >
        {d ? (
          <DataTable
            rows={d.splits[previewSplit].preview}
            onRow={setSelected}
            columns={[
              { key: "id", label: "Sample ID" },
              {
                key: "text",
                label: "Text",
                render: (r) => <span className="table-text">{r.text}</span>,
              },
              { key: "category", label: "Category" },
              {
                key: "label",
                label: "Ground truth",
                render: (r) => (
                  <Status
                    tone={
                      r.label === 1
                        ? "red"
                        : r.label === 0
                          ? "green"
                          : "neutral"
                    }
                  >
                    {r.label === 1
                      ? "Anomaly"
                      : r.label === 0
                        ? "Normal"
                        : "Unlabeled"}
                  </Status>
                ),
              },
            ]}
          />
        ) : (
          <Empty
            title="No dataset loaded"
            description="Connect the backend or enable Preview mode to explore datasets."
          />
        )}
      </Panel>
      <Dialog open={!!selected} onOpenChange={(v) => !v && setSelected(null)}>
        <DialogContent className="detail-dialog">
          <DialogHeader>
            <DialogTitle>{selected?.id}</DialogTitle>
            <DialogDescription>
              {selected?.split} split · dataset inspection only
            </DialogDescription>
          </DialogHeader>
          <p className="full-text">{selected?.text}</p>
          <Notice>
            Ground truth is shown for dataset inspection and is excluded from
            inference requests.
          </Notice>
        </DialogContent>
      </Dialog>
    </>
  );
}
