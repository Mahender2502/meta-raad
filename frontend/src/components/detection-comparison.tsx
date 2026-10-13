import type { Detection, ResultDetail } from "../types";
import { Button, Notice, Panel, Score, Status } from "./common";
import { Download } from "lucide-react";
import { download } from "../lib/utils";
import { useEffect, useRef } from "react";

function Variant({
  title,
  description,
  result,
  threshold,
}: {
  title: string;
  description: string;
  result: Detection | null;
  threshold: number;
}) {
  const valid = result?.score != null && !result.error;
  return (
    <section className="comparison-variant" aria-label={title}>
      <h3>{title}</h3>
      <p className="muted">{description}</p>
      {valid ? (
        <>
          <Score score={result.score!} threshold={threshold} />
          <div className="explanation">
            <strong>LLM explanation</strong>
            <p>{result.reason || "No explanation returned."}</p>
          </div>
        </>
      ) : (
        <Notice tone="amber">
          {result?.error ||
            "This variant was not run or did not return a valid score. Run the comparison again."}
        </Notice>
      )}
      <p className="fine-print">Model: {result?.model ?? "Not returned"}</p>
      {result?.prompt && (
        <details>
          <summary>View actual prompt</summary>
          <pre>{result.prompt}</pre>
        </details>
      )}
    </section>
  );
}

export function DetectionComparison({
  result,
  threshold,
}: {
  result: ResultDetail;
  threshold: number;
}) {
  const comparison = useRef<HTMLDivElement>(null);
  useEffect(() => {
    comparison.current?.scrollIntoView({ block: "start", behavior: "auto" });
    comparison.current?.focus({ preventScroll: true });
  }, [result]);
  const { rag, baseline } = result.response;
  const paired =
    rag?.score != null &&
    !rag.error &&
    baseline?.score != null &&
    !baseline.error;
  const delta = paired ? rag.score! - baseline.score! : null;
  return (
    <div
      ref={comparison}
      tabIndex={-1}
      aria-label="Detection comparison results"
    >
      <Panel
        className="detection-comparison"
        title="With RAG vs without RAG"
        description={`${result.dataset} · ${new Date(result.timestamp).toLocaleTimeString()} · ${result.duration.toFixed(1)} seconds total`}
        action={
          <Button
            variant="outline"
            size="sm"
            onClick={() =>
              download("detection-comparison", { ...result, threshold })
            }
          >
            <Download />
            Export both results
          </Button>
        }
      >
        <div className="comparison-query">
          <strong>Compared text</strong>
          <p>{result.query}</p>
        </div>
        <div className="comparison-grid">
          <Variant
            title="Without RAG · normal LLM"
            description="Text + category definitions. No retrieved examples."
            result={baseline}
            threshold={threshold}
          />
          <Variant
            title="With RAG · RA-ZAD"
            description={`Same text and category definitions + ${result.response.retrieved.length} retrieved training references.`}
            result={rag}
            threshold={threshold}
          />
        </div>
        {paired ? (
          <div className="comparison-summary">
            <Status
              tone={
                rag.score! >= threshold === baseline.score! >= threshold
                  ? "blue"
                  : "amber"
              }
            >
              {rag.score! >= threshold === baseline.score! >= threshold
                ? "Same classification"
                : "Different classifications"}
            </Status>
            <span>
              Score difference (with − without RAG):{" "}
              <strong>
                {delta! > 0 ? "+" : ""}
                {delta!.toFixed(3)}
              </strong>
            </span>
            <span>Shared threshold: {threshold.toFixed(2)}</span>
          </div>
        ) : (
          <Notice tone="amber">
            Comparison incomplete. Any successful result remains visible; failed
            or missing scores are not replaced.
          </Notice>
        )}
        <p className="fine-print">
          Scores are uncalibrated model outputs, not probabilities. A higher or
          lower score does not establish which method is more accurate; that
          requires labeled evaluation.
        </p>
        <details>
          <summary>Shared configuration and provenance</summary>
          <div className="metadata-list">
            <div>
              <span>Category knowledge</span>
              <strong>{result.response.setting}</strong>
            </div>
            <div>
              <span>Embedding</span>
              <strong>{result.embedding}</strong>
            </div>
            <div>
              <span>Index version</span>
              <code>{result.indexVersion}</code>
            </div>
            <div>
              <span>Reference split</span>
              <strong>Normal training only</strong>
            </div>
          </div>
        </details>
      </Panel>
    </div>
  );
}
