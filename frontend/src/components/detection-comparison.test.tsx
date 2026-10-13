import { describe, it, expect } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { DetectionComparison } from "./detection-comparison";
import type { Detection, ResultDetail } from "../types";
const variant = (score: number, reason: string): Detection => ({
  score,
  reason,
  prompt: "An actual prompt",
  model: "test-model",
  error: null,
  verdict: null,
  citations: [],
});
const result: ResultDetail = {
  query: "The same query",
  dataset: "test",
  duration: 2,
  timestamp: "2026-10-10T00:00:00Z",
  indexVersion: "v1",
  embedding: "bert",
  mode: "live",
  response: {
    dataset: "test",
    setting: "normal_only",
    k: 3,
    retrieved: [],
    baseline: variant(0.3, "Baseline explanation"),
    rag: variant(0.7, "RAG explanation"),
  },
};
describe("paired detection results", () => {
  it("shows both explanations, prompts and the score difference without claiming accuracy", () => {
    const html = renderToStaticMarkup(
      <DetectionComparison result={result} threshold={0.5} />,
    );
    expect(html).toContain("Without RAG");
    expect(html).toContain("With RAG");
    expect(html).toContain("Baseline explanation");
    expect(html).toContain("RAG explanation");
    expect(html).toContain("Different classifications");
    expect(html).toContain("0.400");
    expect(html).toContain("does not establish which method is more accurate");
  });
  it("uses the same adjustable threshold for both variants", () => {
    const html = renderToStaticMarkup(
      <DetectionComparison result={result} threshold={0.8} />,
    );
    expect(html).toContain("Same classification");
    expect(html).not.toContain("Anomaly detected");
  });
  it("preserves the successful baseline if RAG fails", () => {
    const partial = {
      ...result,
      response: {
        ...result.response,
        rag: {
          ...variant(0.7, "unused"),
          score: null,
          error: "RAG service failed",
        },
      },
    };
    const html = renderToStaticMarkup(
      <DetectionComparison result={partial} threshold={0.5} />,
    );
    expect(html).toContain("Baseline explanation");
    expect(html).toContain("RAG service failed");
    expect(html).toContain("Comparison incomplete");
    expect(html).not.toContain("Score difference");
  });
  it("shows a missing baseline as unavailable, not zero", () => {
    const partial = {
      ...result,
      response: { ...result.response, baseline: null },
    };
    const html = renderToStaticMarkup(
      <DetectionComparison result={partial} threshold={0.5} />,
    );
    expect(html).toContain("RAG explanation");
    expect(html).toContain("not run or did not return");
    expect(html).toContain("Comparison incomplete");
  });
});
