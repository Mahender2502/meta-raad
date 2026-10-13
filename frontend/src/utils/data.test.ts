import { describe, it, expect, vi, afterEach } from "vitest";
import { parseFile, mapRows, validateRows, evaluate } from "./data";
import { createApi } from "../services/api";
import { demoCatalog, demoProfileFor } from "../services/demo";
import type { Preferences } from "../types";
const prefs: Preferences = {
  mode: "live",
  backendUrl: "",
  topK: 3,
  threshold: 0.5,
  similarityThreshold: 0.6,
  timeout: 10,
  embedding: "bert",
  provider: "numpy",
  nearDuplicateThreshold: 0.9,
};
afterEach(() => vi.unstubAllGlobals());
describe("data integrity", () => {
  it("reads quoted CSV and Unicode JSONL without modifying source", () => {
    const source = 'body,target\n"A comma, and text",0';
    const r = parseFile(source, "records.csv");
    expect(mapRows(r, "body", "target", "train")[0]).toMatchObject({
      text: "A comma, and text",
      label: 0,
    });
    expect(
      parseFile('\uFEFF{"text":"東京","label":1}', "r.jsonl")[0].text,
    ).toBe("東京");
    expect(source).toContain('"A comma, and text"');
  });
  it("rejects malformed lines and non-object JSON", () => {
    expect(() => parseFile("{}\nbroken", "r.jsonl")).toThrow("line 2");
    expect(() => parseFile("[]", "r.jsonl")).toThrow();
  });
  it("rejects boolean and invalid labels, requires normal train, allows unlabeled test", () => {
    const rows = mapRows(
      [
        { text: "a", label: false },
        { text: "b", label: "no" },
        { text: "c", label: 1 },
      ],
      "text",
      "label",
      "train",
    );
    expect(validateRows(rows)).toMatchObject({
      invalid: 2,
      anomaly: 1,
      blocked: true,
    });
    expect(
      validateRows(mapRows([{ text: "query" }], "text", "label", "test"))
        .blocked,
    ).toBe(false);
  });
  it("blocks normalized duplicates and exact train-test overlap", () => {
    const train = mapRows(
      [
        { text: "Normal TEXT", label: 0 },
        { text: " normal  text ", label: 0 },
      ],
      "text",
      "label",
      "train",
    );
    expect(validateRows(train).duplicates).toBe(1);
    expect(
      validateRows(
        mapRows([{ text: " normal text ", label: 1 }], "text", "label", "test"),
        train,
      ),
    ).toMatchObject({ overlap: 1, blocked: true });
  });
  it("flags near matches without deleting input", () => {
    const a = mapRows(
        [{ text: "one two three four five", label: 0 }],
        "text",
        "label",
        "train",
      ),
      b = mapRows(
        [{ text: "one two three four six", label: 1 }],
        "text",
        "label",
        "test",
      );
    expect(validateRows(a, b, 0.6).near).toHaveLength(1);
    expect(a).toHaveLength(1);
  });
});
describe("evaluation", () => {
  it("calculates threshold confusion, AUROC and AP from actual scores", () => {
    const p = [
      { id: "a", label: 0, score: 0.1 },
      { id: "b", label: 1, score: 0.45 },
      { id: "c", label: 0, score: 0.5 },
      { id: "d", label: 1, score: 0.8 },
    ];
    const m = evaluate(p, 0.5);
    expect(m.auroc).toBe(0.75);
    expect(m.auprc).toBeCloseTo(5 / 6);
    expect(m).toMatchObject({ tp: 1, fp: 1, tn: 1, fn: 1 });
    expect(evaluate(p, 0.9).auroc).toBe(m.auroc);
  });
  it("handles tied scores and excludes failed or unlabeled predictions", () => {
    const m = evaluate([
      { id: "a", label: 0, score: 0.5 },
      { id: "b", label: 1, score: 0.5 },
      { id: "c", label: 1, score: null },
      { id: "d", score: 0.2 },
    ]);
    expect(m).toMatchObject({ auroc: 0.5, auprc: 0.5, valid: 2, total: 4 });
    expect(evaluate([{ id: "a", label: 0, score: 0.2 }]).auroc).toBeNull();
  });
});
describe("service boundaries", () => {
  it("sends only text/configuration and hides reference labels", async () => {
    const fetcher = vi
      .fn()
      .mockResolvedValue({ ok: true, json: async () => ({}) });
    vi.stubGlobal("fetch", fetcher);
    await createApi(prefs).detect(
      "query",
      "n24_news",
      3,
      "normal_only",
      false,
      [{ id: "test", text: "private reference", label: 1, split: "test" }],
    );
    const body = JSON.parse(fetcher.mock.calls[0][1].body);
    expect(body).toEqual({
      text: "query",
      dataset: "n24_news",
      k: 3,
      setting: "normal_only",
      run_rag: true,
      run_baseline: false,
      include_reference_categories: false,
    });
    expect(JSON.stringify(body)).not.toContain("private reference");
  });
  it("does not silently substitute mock output on live failure", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(Error("offline")));
    await expect(createApi(prefs).catalog()).rejects.toThrow("offline");
  });
  it("explicit preview makes no network request and copies fixtures", async () => {
    const f = vi.fn();
    vi.stubGlobal("fetch", f);
    const catalog = await createApi({ ...prefs, mode: "demo" }).catalog();
    catalog.datasets[0].name = "changed";
    expect(demoCatalog.datasets[0].name).toBe("News sandbox");
    expect(f).not.toHaveBeenCalled();
    expect(
      demoProfileFor([{ id: "a", text: "one two", label: 0, split: "train" }])
        .features,
    ).toMatchObject({ n_train: 1, effective_dimension: null });
  });
});
