import { useEffect, useState } from "react";
import {
  Check,
  Download,
  GitBranch,
  Layers3,
  Play,
  ShieldCheck,
} from "lucide-react";
import { useWorkspace } from "../hooks/use-workspace";
import {
  Button,
  Empty,
  Field,
  Loading,
  Metric,
  Notice,
  PageHeader,
  Panel,
  Status,
  Toggle,
} from "../components/common";
import { GeometryChart } from "../charts/research-charts";
import { download, number, title } from "../lib/utils";
import type { Card, Profile, Selection } from "../types";
export function Models() {
  const w = useWorkspace(),
    [profile, setProfile] = useState<Profile | null>(null),
    [cards, setCards] = useState<Record<string, Card>>({}),
    [candidates, setCandidates] = useState<string[]>([]),
    [condition, setCondition] = useState("full"),
    [result, setResult] = useState<Selection | null>(null),
    [busy, setBusy] = useState(false),
    [error, setError] = useState("");
  useEffect(() => {
    let active = true;
    setProfile(null);
    setResult(null);
    setError("");
    Promise.all([
      w.api.cards(),
      w.dataset
        ? w.api.profile(
            w.dataset.id,
            w.dataset.local ? w.dataset.splits.train.preview : undefined,
          )
        : Promise.resolve(null),
    ])
      .then(([c, p]) => {
        if (active) {
          setCards(c);
          setCandidates(Object.keys(c));
          setProfile(p);
        }
      })
      .catch((e) => active && setError(String(e)));
    return () => {
      active = false;
    };
  }, [w.api, w.dataset]);
  async function run() {
    if (!w.dataset) return;
    setBusy(true);
    setError("");
    try {
      setResult(
        await w.api.select(
          w.dataset.id,
          candidates,
          condition,
          w.dataset.local ? w.dataset.splits.train.preview : undefined,
        ),
      );
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }
  const f = profile?.features;
  return (
    <>
      <PageHeader
        eyebrow="FG-MOS · SECOND RESEARCH MODULE"
        title="Model selection"
        description="Match detector capabilities to measured data properties, then inspect the reasoning."
        actions={
          <Button
            disabled={!result}
            variant="outline"
            onClick={() =>
              download("model-recommendation", {
                ...result,
                mode: w.preferences.mode,
              })
            }
          >
            <Download />
            Export recommendation
          </Button>
        }
      />
      {error && <Notice tone="red">{error}</Notice>}
      <div className="metrics-grid">
        <Metric
          label="Intra-class compactness"
          value={number(f?.intra_class_compactness, 4)}
          note="Cosine distance · lower is tighter"
          icon={<Layers3 size={18} />}
        />
        <Metric
          label="Inter-class separability"
          value={number(f?.inter_class_separability, 4)}
          note="Between normal-category centroids"
          icon={<GitBranch size={18} />}
        />
        <Metric
          label="Training examples"
          value={number(f?.n_train)}
          note={`${number(f?.profile_sample_size)} samples used for profile`}
          icon={<ShieldCheck size={18} />}
        />
        <Metric
          label="Effective dimension"
          value={number(f?.effective_dimension, 2)}
          note={`${number(f?.embedding_dimension)} embedding dimensions`}
          icon={<Layers3 size={18} />}
        />
      </div>
      <div className="grid-main">
        <Panel
          title="Candidate detectors"
          description="Choose the algorithms to rank. Capability matchups are hypotheses, not guarantees."
          action={<Status tone="blue">{candidates.length} selected</Status>}
        >
          <div className="table-scroll">
            <table className="candidate-table">
              <thead>
                <tr>
                  <th>Candidate</th>
                  <th>Mechanism & limitations</th>
                  <th>Relevant properties</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(cards).map(([id, card]) => (
                  <tr key={id}>
                    <td>
                      <Toggle
                        on={candidates.includes(id)}
                        onChange={(on) =>
                          setCandidates((old) =>
                            on ? [...old, id] : old.filter((c) => c !== id),
                          )
                        }
                      >
                        <strong>{title(id)}</strong>
                      </Toggle>
                    </td>
                    <td>
                      <p>{card.mechanism}</p>
                      <small>{card.limitations}</small>
                      <details>
                        <summary>Capabilities and source</summary>
                        <p>{card.strengths}</p>
                        <a href={card.source} target="_blank" rel="noreferrer">
                          Research source ↗
                        </a>
                      </details>
                    </td>
                    <td>
                      <div className="property-tags">
                        {card.properties.map((p) => (
                          <span key={p}>{p.replaceAll("_", " ")}</span>
                        ))}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="selection-run">
            <Field label="Experimental condition">
              <select
                value={condition}
                onChange={(e) => setCondition(e.target.value)}
                disabled={busy}
              >
                <option value="full">Full · features + model cards</option>
                <option value="dataset_only">Dataset features only</option>
                <option value="models_only">Model cards only</option>
                <option value="baseline">Basic descriptions</option>
              </select>
            </Field>
            <Button
              onClick={() => void run()}
              disabled={
                !candidates.length ||
                !profile ||
                busy ||
                (!!w.dataset?.local && w.preferences.mode === "live")
              }
            >
              <Play />
              {busy ? "Analyzing…" : "Recommend model"}
            </Button>
          </div>
          <Notice>
            Profiles use permissible training data only. Held-out performance is
            not visible to the model-selection prompt.
          </Notice>
        </Panel>
        <div>
          <Panel
            title="Dataset geometry"
            description="Comparable cosine-distance measurements"
          >
            {f ? (
              <GeometryChart
                data={[
                  "intra_class_compactness",
                  "inter_class_separability",
                  "mean_knn_cosine_distance",
                ]
                  .filter((k) => f[k] != null)
                  .map((k) => ({
                    name:
                      k === "intra_class_compactness"
                        ? "Compactness"
                        : k === "inter_class_separability"
                          ? "Separation"
                          : "KNN distance",
                    value: f[k]!,
                  }))}
              />
            ) : (
              <Loading text="Loading the training profile…" />
            )}
            <p className="fine-print">
              These values describe normal training categories.
              Normal-versus-anomaly separation is not available.
            </p>
          </Panel>
          <Panel
            title="All measured features"
            description="Unavailable properties stay unavailable."
          >
            <div className="feature-list">
              {Object.entries(f ?? {}).map(([k, v]) => (
                <div key={k}>
                  <span>{k.replaceAll("_", " ")}</span>
                  <strong>{v == null ? "Not available" : number(v, 4)}</strong>
                </div>
              ))}
            </div>
          </Panel>
        </div>
      </div>
      <Panel
        title="Recommendation & evidence"
        description="The ranking is model-generated interpretation; numeric evidence is checked separately."
        action={result && <Status tone="blue">{result.model}</Status>}
      >
        {busy ? (
          <Loading text="Comparing dataset properties against candidate capabilities…" />
        ) : result ? (
          <div className="recommendation-grid">
            <div>
              <div className="recommended">
                <div className="recommend-icon">
                  <GitBranch size={25} />
                </div>
                <div>
                  <small>SELECTED DETECTOR</small>
                  <h2>{title(result.selected_model)}</h2>
                </div>
              </div>
              <p className="recommend-reason">{result.reason}</p>
              <div className="ranking-list">
                {result.ranking.map((id, i) => (
                  <div key={id}>
                    <span>{String(i + 1).padStart(2, "0")}</span>
                    <strong>{title(id)}</strong>
                    {i === 0 && <Status tone="blue">Recommended</Status>}
                  </div>
                ))}
              </div>
            </div>
            <div>
              <div className="grounding-summary">
                <div>
                  <small>NUMERIC GROUNDING</small>
                  <strong>
                    {(result.grounding.g_score * 100).toFixed(0)}
                    <span>%</span>
                  </strong>
                </div>
                <p>
                  {result.grounding.valid_citations} /{" "}
                  {result.grounding.total_citations} valid citations
                  <br />
                  {(result.grounding.feature_coverage * 100).toFixed(1)}%
                  feature coverage
                </p>
              </div>
              <div className="table-scroll">
                <table>
                  <thead>
                    <tr>
                      <th>Cited feature</th>
                      <th>Claimed</th>
                      <th>Measured</th>
                      <th>Check</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.grounding.checks.map((c, i) => (
                      <tr key={i}>
                        <td>{c.feature.replaceAll("_", " ")}</td>
                        <td>{number(c.value, 4)}</td>
                        <td>{number(c.expected, 4)}</td>
                        <td>
                          {c.valid ? (
                            <Check size={16} className="green-text" />
                          ) : (
                            "Mismatch"
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <Notice tone="amber">
                A correct numeric citation does not prove a good recommendation
                or a faithful explanation. Compare the selected detector against
                held-out results.
              </Notice>
              <details>
                <summary>Actual model-selection prompt</summary>
                <pre>{result.prompt}</pre>
              </details>
            </div>
          </div>
        ) : (
          <Empty
            title="A recommendation with a traceable rationale"
            description="Choose your candidates and run feature-grounded selection to inspect the ranking and evidence."
          />
        )}
      </Panel>
    </>
  );
}
