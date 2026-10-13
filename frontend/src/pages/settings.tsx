import { useState } from "react";
import { Check, Plug, Save, ShieldCheck } from "lucide-react";
import { useWorkspace } from "../hooks/use-workspace";
import {
  Button,
  Field,
  Notice,
  PageHeader,
  Panel,
  Status,
} from "../components/common";
import { createApi } from "../services/api";
export function Settings() {
  const w = useWorkspace(),
    [draft, setDraft] = useState(w.preferences),
    [message, setMessage] = useState(""),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false);
  function save() {
    setError("");
    try {
      if (draft.backendUrl) {
        const u = new URL(draft.backendUrl);
        if (
          !["http:", "https:"].includes(u.protocol) ||
          u.username ||
          u.password ||
          u.search ||
          u.hash
        )
          throw Error(
            "Use an HTTP(S) backend URL without credentials, query parameters, or fragments.",
          );
      }
      if (
        !Number.isFinite(draft.timeout) ||
        draft.timeout < 5 ||
        draft.timeout > 600
      )
        throw Error("Timeout must be between 5 and 600 seconds.");
      w.setPreferences(draft);
      setMessage("Preferences saved locally. No API credentials are stored.");
    } catch (e) {
      setError(String(e));
    }
  }
  async function test() {
    setBusy(true);
    setError("");
    try {
      const result = await createApi(draft).health();
      setMessage(`Connection successful: ${result.status}.`);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <PageHeader
        eyebrow="WORKSPACE CONFIGURATION"
        title="Settings"
        description="Configure your connection and research preferences. Credentials stay on the backend."
        actions={
          <Button onClick={save}>
            <Save />
            Save changes
          </Button>
        }
      />
      {message && <Notice tone="green">{message}</Notice>}
      {error && <Notice tone="red">{error}</Notice>}
      <div className="settings-layout">
        <Panel
          title="Connection & data mode"
          description="Live mode never falls back to illustrative results."
        >
          <div className="mode-options">
            <button
              className={draft.mode === "live" ? "selected" : ""}
              onClick={() => setDraft({ ...draft, mode: "live" })}
            >
              <Plug size={21} />
              <strong>Live backend</strong>
              <p>Real datasets, inference, and saved experiments.</p>
              {draft.mode === "live" && <Check size={16} />}
            </button>
            <button
              className={draft.mode === "demo" ? "selected" : ""}
              onClick={() => setDraft({ ...draft, mode: "demo" })}
            >
              <ShieldCheck size={21} />
              <strong>Illustrative preview</strong>
              <p>Simulated ML workflows with clearly marked fixtures.</p>
              {draft.mode === "demo" && <Check size={16} />}
            </button>
          </div>
          <Field
            label="Backend API URL"
            hint="Leave blank for the current server (or the Vite development proxy). Custom servers must allow this origin with CORS."
          >
            <input
              placeholder="Same origin (recommended)"
              value={draft.backendUrl}
              onChange={(e) =>
                setDraft({ ...draft, backendUrl: e.target.value })
              }
            />
          </Field>
          <div className="form-grid">
            <Field label="Request timeout (seconds)">
              <input
                type="number"
                min={5}
                max={600}
                value={draft.timeout}
                onChange={(e) =>
                  setDraft({ ...draft, timeout: Number(e.target.value) })
                }
              />
            </Field>
            <Field
              label="Active language model"
              hint="Configured on the Python backend, not in this browser."
            >
              <input
                value={w.catalog?.configuration.llm ?? "Not connected"}
                readOnly
              />
            </Field>
          </div>
          <Button variant="outline" onClick={() => void test()} disabled={busy}>
            <Plug />
            {busy ? "Checking…" : "Test connection"}
          </Button>
        </Panel>
        <Panel
          title="Research defaults"
          description="Changes apply to new requests and metric displays."
        >
          <div className="form-grid">
            <Field label="Top-K retrieval">
              <input
                type="number"
                min={1}
                max={10}
                value={draft.topK}
                onChange={(e) =>
                  setDraft({
                    ...draft,
                    topK: Math.max(1, Math.min(10, Number(e.target.value))),
                  })
                }
              />
            </Field>
            <Field label="Anomaly-score threshold">
              <input
                type="number"
                min={0}
                max={1}
                step={0.05}
                value={draft.threshold}
                onChange={(e) =>
                  setDraft({
                    ...draft,
                    threshold: Math.max(0, Math.min(1, Number(e.target.value))),
                  })
                }
              />
            </Field>
            <Field
              label="Similarity review threshold"
              hint="Used to flag low similarity in the UI; does not change predictions."
            >
              <input
                type="number"
                min={0}
                max={1}
                step={0.05}
                value={draft.similarityThreshold}
                onChange={(e) =>
                  setDraft({
                    ...draft,
                    similarityThreshold: Math.max(
                      0,
                      Math.min(1, Number(e.target.value)),
                    ),
                  })
                }
              />
            </Field>
            <Field label="Near-duplicate lexical threshold">
              <input
                type="number"
                min={0.5}
                max={1}
                step={0.05}
                value={draft.nearDuplicateThreshold}
                onChange={(e) =>
                  setDraft({
                    ...draft,
                    nearDuplicateThreshold: Math.max(
                      0.5,
                      Math.min(1, Number(e.target.value)),
                    ),
                  })
                }
              />
            </Field>
            <Field label="Preview embedding model">
              <select
                value={draft.embedding}
                onChange={(e) =>
                  setDraft({ ...draft, embedding: e.target.value })
                }
              >
                <option>BAAI/bge-base-en-v1.5</option>
                <option>bert-base-uncased</option>
              </select>
            </Field>
            <Field label="Preview vector provider">
              <select
                value={draft.provider}
                onChange={(e) =>
                  setDraft({ ...draft, provider: e.target.value })
                }
              >
                <option>FAISS</option>
                <option>ChromaDB</option>
              </select>
            </Field>
          </div>
          <Notice>
            {w.preferences.mode === "live"
              ? `The live service uses ${w.catalog?.configuration.embedding ?? "the configured embedding"} with ${w.dataset?.index.provider ?? "its configured search provider"}.`
              : "Preview mode simulates the selected embedding and vector provider."}{" "}
            Preview configuration does not reconfigure the Python backend.
          </Notice>
        </Panel>
      </div>
      <Panel
        title="Integration status"
        description="A clear contract between the interface and the research backend."
      >
        <div className="integration-list">
          {[
            [
              "Single / batch detection",
              "Connected · batch orchestrates sequential single calls",
            ],
            ["Training profiles & model selection", "Connected"],
            ["Retrieval & prompt inspection", "Connected"],
            ["Saved evaluation runs", "Connected · actual predictions"],
            [
              "Dataset upload & vector index builds",
              "Preview implementation only",
            ],
          ].map(([name, state]) => (
            <div key={name}>
              <strong>{name}</strong>
              <Status tone={state.startsWith("Preview") ? "amber" : "green"}>
                {w.preferences.mode === "demo"
                  ? "Preview implementation"
                  : !w.connected && !state.startsWith("Preview")
                    ? "Backend offline"
                    : state}
              </Status>
            </div>
          ))}
        </div>
        <Notice tone="green">
          No API keys appear in frontend environment variables, local storage,
          request forms, or source code.
        </Notice>
      </Panel>
    </>
  );
}
