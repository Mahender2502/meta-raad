import { Link } from "react-router-dom";
import {
  Activity,
  ArrowRight,
  ArrowUpRight,
  Boxes,
  Check,
  ChevronRight,
  Database,
  Download,
  FileCheck2,
  FlaskConical,
  GitBranch,
  Layers3,
  Play,
  ShieldCheck,
  Sparkles,
  Upload,
} from "lucide-react";
import { useWorkspace } from "../hooks/use-workspace";
import {
  Button,
  Empty,
  Metric,
  PageHeader,
  Panel,
  Status,
  Loading,
} from "../components/common";
import { ComparisonChart } from "../charts/research-charts";
import { download, number } from "../lib/utils";
export function Overview() {
  const w = useWorkspace(),
    d = w.dataset;
  const exp = w.experiments.find(
    (e) =>
      e.dataset === d?.id && e.results.some((r) => r.metrics?.auroc != null),
  );
  const results = exp?.results.filter((r) => r.metrics) ?? [];
  const best = results.length
    ? Math.max(...results.map((r) => r.metrics?.auroc ?? 0))
    : null;
  const workflow = [
    { name: "Dataset", sub: "Source loaded", icon: Database, done: !!d },
    {
      name: "Validation",
      sub: "Normal training split",
      icon: FileCheck2,
      done: !!d?.validation.normal_train_only,
    },
    {
      name: "Vector index",
      sub: "Reference memory",
      icon: Boxes,
      done: !!d?.index.ready,
    },
    {
      name: "KNN retrieval",
      sub: "Local evidence",
      icon: Layers3,
      done: !!w.latest,
    },
    {
      name: "LLM inference",
      sub: "Grounded decision",
      icon: Sparkles,
      done: !!w.latest,
    },
    {
      name: "Evaluation",
      sub: "Held-out results",
      icon: FlaskConical,
      done: !!exp,
    },
  ];
  return (
    <>
      <PageHeader
        eyebrow="YOUR RESEARCH, AT A GLANCE"
        title="Research overview"
        description="From text to evidence. A transparent view of your anomaly detection workflow."
        actions={
          <>
            <Button
              variant="outline"
              onClick={() =>
                download("workspace-summary", {
                  dataset: d?.name,
                  experiments: w.experiments,
                  mode: w.preferences.mode,
                })
              }
            >
              <Download />
              Export summary
            </Button>
            <Button asChild>
              <Link to="/detection">
                <Play size={15} />
                Run detection
              </Link>
            </Button>
          </>
        }
      />
      <div className="overview-banner">
        <div>
          <span className="mini-label">
            <span />
            META-RAAD FRAMEWORK
          </span>
          <h2>
            Understand the anomaly.
            <br />
            <span>See the evidence behind it.</span>
          </h2>
          <p>
            Retrieve relevant training examples, inspect LLM decisions,
            <br className="desktop-only" /> and choose models using measured
            dataset properties.
          </p>
          <Link to="/detection">
            Open detection workspace <ArrowRight size={16} />
          </Link>
        </div>
        <div className="banner-diagram" aria-hidden="true">
          <span className="diagram-orbit" />
          <div className="diagram-node n1">
            <Database />
          </div>
          <div className="diagram-node n2">
            <Boxes />
          </div>
          <div className="diagram-center">
            <Sparkles size={28} />
            <small>Meta-RAAD</small>
          </div>
          <div className="diagram-node n3">
            <ShieldCheck />
          </div>
          <div className="diagram-node n4">
            <Activity />
          </div>
          <span className="diagram-caption">
            CONTEXT → REASONING → EVIDENCE
          </span>
        </div>
      </div>
      {w.loading ? (
        <Loading />
      ) : (
        <div className="metrics-grid">
          <Metric
            label="Training samples"
            value={number(d?.splits.train.count)}
            note={
              d
                ? `${d.name} · normal reference split`
                : "Select a dataset to begin"
            }
            icon={<Database size={19} />}
          />
          <Metric
            label="Normal references"
            value={number(d?.index.count)}
            note={
              d?.index.ready
                ? "Validated training vectors available"
                : "Reference index not ready"
            }
            icon={<Boxes size={19} />}
            tone="purple"
          />
          <Metric
            label="Test samples evaluated"
            value={number(exp?.n_test ?? results[0]?.metrics?.n_total)}
            note={
              exp ? "From the displayed saved run" : "No eligible results yet"
            }
            icon={<FileCheck2 size={19} />}
            tone="teal"
          />
          <Metric
            label="Best AUROC in run"
            value={best === null ? "—" : best.toFixed(3)}
            note={
              exp
                ? `${exp.command === "baselines" ? "Classical baselines" : "LLM experiment"} · not a general performance claim`
                : "Awaiting a two-class evaluation"
            }
            icon={<Activity size={19} />}
            tone="amber"
          />
        </div>
      )}
      <Panel
        title="Research pipeline"
        description="A traceable path from source data to measured outcomes."
        action={
          <Status tone={d?.validation.normal_train_only ? "green" : "amber"}>
            {d?.validation.normal_train_only
              ? "Training split validated"
              : "Validation pending"}
          </Status>
        }
      >
        <div className="pipeline">
          {workflow.map((step, i) => (
            <div className="pipeline-item" key={step.name}>
              <div className={`pipeline-icon ${step.done ? "done" : ""}`}>
                <step.icon size={20} />
                {step.done && (
                  <span>
                    <Check size={9} />
                  </span>
                )}
              </div>
              <strong>{step.name}</strong>
              <small>{step.sub}</small>
              <em className={step.done ? "complete" : ""}>
                {step.done ? "Completed" : "Pending"}
              </em>
              {i < 5 && <ChevronRight size={16} className="pipeline-chevron" />}
            </div>
          ))}
        </div>
      </Panel>
      <div className="grid-main">
        <Panel
          title="Evaluation snapshot"
          description={
            exp
              ? `${exp.id} · ${w.preferences.mode === "demo" ? "illustrative fixture" : "recorded experiment"}`
              : "Actual scores appear after a completed evaluation."
          }
          action={
            <Button variant="ghost" size="sm" asChild>
              <Link to="/evaluation">
                View evaluation <ArrowUpRight size={14} />
              </Link>
            </Button>
          }
        >
          {results.length ? (
            <ComparisonChart
              data={results.map((r) => ({
                name:
                  r.model ?? (r.k === 0 ? "Zero-shot" : `RA-ZAD · k=${r.k}`),
                auroc: r.metrics?.auroc ?? null,
                auprc: r.metrics?.auprc ?? null,
              }))}
            />
          ) : (
            <Empty
              title="No measured comparison yet"
              description="Run predictions with held-out labels to compare AUROC and AUPRC. We never fill this chart with assumed improvements."
            />
          )}
          <div className="panel-footnote">
            <ShieldCheck size={13} />
            {exp?.command === "baselines"
              ? "Classical detector scores. LLM improvement has not been established."
              : "Metrics reflect this experiment only; preview values are illustrative."}
          </div>
        </Panel>
        <Panel
          title="Workspace health"
          description="Configuration and research safeguards"
        >
          <div className="health-list">
            {[
              ["Dataset", d?.name ?? "Not loaded"],
              ["Reference split", "Train only"],
              ["Embedding model", d?.index.embedding ?? "Not available"],
              ["Search provider", d?.index.provider ?? "Not available"],
              [
                "Language model",
                w.catalog?.configuration.llm ?? "Not connected",
              ],
            ].map(([k, v]) => (
              <div key={k}>
                <span>{k}</span>
                <strong>{v}</strong>
              </div>
            ))}
          </div>
          <div className="integrity-check">
            <ShieldCheck size={19} />
            <div>
              <strong>Evaluation labels stay separate</strong>
              <p>Test labels are never sent with prediction requests.</p>
            </div>
          </div>
          <Button asChild variant="outline" className="full-width">
            <Link to="/vectors">
              Inspect reference index <ArrowRight size={14} />
            </Link>
          </Button>
        </Panel>
      </div>
      <div className="section-label">CONTINUE YOUR RESEARCH</div>
      <div className="quick-actions">
        {[
          {
            to: "/datasets",
            icon: Upload,
            title: "Bring your dataset",
            text: "Inspect JSONL or CSV and validate splits.",
          },
          {
            to: "/retrieval",
            icon: Layers3,
            title: "Explore the evidence",
            text: "Inspect neighbors and the grounded prompt.",
          },
          {
            to: "/models",
            icon: GitBranch,
            title: "Find a suitable model",
            text: "Compare capabilities with dataset geometry.",
          },
        ].map((a) => (
          <Link key={a.to} to={a.to}>
            <span className="quick-icon">
              <a.icon size={20} />
            </span>
            <div>
              <h3>{a.title}</h3>
              <p>{a.text}</p>
            </div>
            <ArrowUpRight size={17} />
          </Link>
        ))}
      </div>
    </>
  );
}
