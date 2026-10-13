import type {
  Catalog,
  DetectResponse,
  Experiment,
  Preferences,
  Profile,
  Reference,
  Selection,
  Card,
  Prediction,
  Row,
} from "../types";
import {
  demoCatalog,
  demoCards,
  demoDetect,
  demoExperiments,
  demoProfileFor,
  demoRetrieve,
  demoSelection,
} from "./demo";

export function createApi(
  preferences: Pick<Preferences, "mode" | "backendUrl" | "timeout">,
) {
  const demo = preferences.mode === "demo";
  async function request<T>(
    path: string,
    body?: unknown,
    signal?: AbortSignal,
  ): Promise<T> {
    const controller = new AbortController();
    const abort = () => controller.abort();
    if (signal?.aborted) abort();
    signal?.addEventListener("abort", abort, { once: true });
    const timer = setTimeout(abort, preferences.timeout * 1000);
    try {
      const r = await fetch(preferences.backendUrl.replace(/\/$/, "") + path, {
        method: body === undefined ? "GET" : "POST",
        headers:
          body === undefined
            ? undefined
            : { "Content-Type": "application/json" },
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: controller.signal,
      });
      const data = await r.json();
      if (!r.ok)
        throw Error(
          typeof data.detail === "string"
            ? data.detail
            : JSON.stringify(data.detail ?? `HTTP ${r.status}`),
        );
      return data;
    } catch (e) {
      if (e instanceof Error && e.name === "AbortError")
        throw Error(
          "Request cancelled or timed out. A server-side inference may still complete.",
        );
      throw e;
    } finally {
      clearTimeout(timer);
      signal?.removeEventListener("abort", abort);
    }
  }
  return {
    catalog: () =>
      demo
        ? Promise.resolve(structuredClone(demoCatalog))
        : request<Catalog>("/api/workspace/datasets"),
    health: () =>
      demo
        ? Promise.resolve({ status: "preview" })
        : request<{ status: string }>("/health"),
    cards: () =>
      demo
        ? Promise.resolve(demoCards)
        : request<{ candidates: Record<string, Card> }>(
            "/selection/cards",
          ).then((d) => d.candidates),
    profile: (dataset: string, records?: Row[]) =>
      demo
        ? Promise.resolve(demoProfileFor(records))
        : request<Profile>(
            `/selection/profile/${encodeURIComponent(dataset)}?embedding=bert`,
          ),
    detect: (
      text: string,
      dataset: string,
      k: number,
      setting: string,
      baseline: boolean,
      records?: Row[],
      signal?: AbortSignal,
    ) =>
      demo
        ? new Promise<DetectResponse>((resolve) =>
            setTimeout(
              () => resolve(demoDetect(text, k, dataset, baseline, records)),
              450,
            ),
          )
        : request<DetectResponse>(
            "/detect/compare",
            {
              text,
              dataset,
              k,
              setting,
              run_rag: true,
              run_baseline: baseline,
              include_reference_categories: false,
            },
            signal,
          ),
    retrieve: (text: string, dataset: string, k: number, records?: Row[]) =>
      demo
        ? Promise.resolve({
            references: demoRetrieve(text, k, records, dataset),
            prompt: JSON.stringify(
              {
                query: text,
                references: demoRetrieve(text, k, records, dataset).map(
                  (r) => ({
                    id: r.doc_id,
                    text: r.text,
                  }),
                ),
              },
              null,
              2,
            ),
            query_excluded: true,
            reference_labels_in_prompt: false,
          })
        : request<{
            references: Reference[];
            prompt: string;
            query_excluded: boolean;
            reference_labels_in_prompt: boolean;
          }>("/api/workspace/retrieval", { text, dataset, k }),
    select: (
      dataset: string,
      candidates: string[],
      condition: string,
      records?: Row[],
    ) =>
      demo
        ? new Promise<Selection>((resolve) =>
            setTimeout(
              () => resolve(demoSelection(candidates, demoProfileFor(records))),
              650,
            ),
          )
        : request<Selection>("/selection/recommend", {
            dataset,
            candidates,
            condition,
            embedding: "bert",
          }),
    experiments: () =>
      demo
        ? Promise.resolve(demoExperiments)
        : request<{ experiments: Experiment[] }>(
            "/api/workspace/experiments",
          ).then((d) => d.experiments),
    predictions: (run: string, variant: string) =>
      demo
        ? Promise.resolve([
            { id: "test_0", label: 0, score: 0.1 },
            { id: "test_1", label: 1, score: variant === "k0" ? 0.45 : 0.9 },
            { id: "test_2", label: 0, score: 0.5 },
            { id: "test_3", label: 1, score: 0.8 },
          ] as Prediction[])
        : request<{ predictions: Prediction[] }>(
            `/api/workspace/experiments/${encodeURIComponent(run)}/predictions/${encodeURIComponent(variant)}`,
          ).then((d) => d.predictions),
    buildIndex: async () => {
      if (!demo)
        throw Error(
          "This backend uses an existing NumPy reference memory. ChromaDB/FAISS index-building is not connected.",
        );
      await new Promise((r) => setTimeout(r, 1000));
      return { simulated: true };
    },
  };
}
