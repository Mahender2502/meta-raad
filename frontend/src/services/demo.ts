import type {
  Card,
  Catalog,
  Dataset,
  DetectResponse,
  Experiment,
  Profile,
  Reference,
  Row,
  Selection,
} from "../types";
import { normalize } from "../utils/data";
const texts = [
  "Researchers mapped a distant galaxy using a new telescope.",
  "The home team scored twice in the final quarter to win the match.",
  "A new battery design could improve the range of electric vehicles.",
  "The central bank held interest rates steady after its latest meeting.",
  "Astronomers identified a planet orbiting a nearby star.",
  "The championship opens on Friday with eight teams competing.",
  "Engineers tested a lightweight solar panel for homes.",
  "Quarterly business earnings rose as sales improved.",
];
export const demoTrain: Row[] = texts.map((text, i) => ({
  id: `train_${i}`,
  text,
  label: 0,
  category: ["Science", "Sports", "Technology", "Business"][i % 4],
  split: "train",
}));
export const demoTest: Row[] = [
  {
    id: "test_0",
    text: "New satellite images reveal a storm on Jupiter.",
    label: 0,
    category: "Science",
    split: "test",
  },
  {
    id: "test_1",
    text: "Mix flour and butter, then bake the cake for thirty minutes.",
    label: 1,
    category: "Food",
    split: "test",
  },
  {
    id: "test_2",
    text: "The league announced its schedule for next season.",
    label: 0,
    category: "Sports",
    split: "test",
  },
  {
    id: "test_3",
    text: "Simmer onions and tomatoes to prepare the sauce.",
    label: 1,
    category: "Food",
    split: "test",
  },
];
export const demoDataset: Dataset = {
  id: "demo_news",
  name: "News sandbox",
  source: "Illustrative fixture · 12 records",
  splits: {
    train: {
      count: 8,
      available: true,
      normal: 8,
      anomaly: 0,
      duplicates: 0,
      empty_text: 0,
      invalid_labels: 0,
      categories: { Science: 2, Sports: 2, Technology: 2, Business: 2 },
      preview: demoTrain,
    },
    test: {
      count: 4,
      available: true,
      normal: 2,
      anomaly: 2,
      duplicates: 0,
      empty_text: 0,
      invalid_labels: 0,
      preview: demoTest,
    },
  },
  validation: {
    normal_train_only: true,
    exact_overlap: 0,
    near_duplicates: "Not computed",
    vector_rows_aligned: true,
  },
  index: {
    ready: true,
    provider: "FAISS (simulated)",
    embedding: "BAAI/bge-base-en-v1.5",
    dimensions: 768,
    count: 8,
    split: "train",
    collection: "demo_news/normal_train",
    version: "preview-fixture-v1",
    created_at: null,
  },
};
export const demoCatalog: Catalog = {
  datasets: [demoDataset],
  configuration: {
    llm: "Preview simulator",
    provider: "mock",
    embedding: "BAAI/bge-base-en-v1.5",
    version: "0.2.0",
  },
  capabilities: {
    upload: true,
    build_index: true,
    retrieval: true,
    detection: true,
    model_selection: true,
  },
};
export const demoCards: Record<string, Card> = Object.fromEntries(
  [
    [
      "iforest",
      "Isolation through random partitions",
      "Efficient without distance matrices",
      "Axis-aligned partition assumptions",
    ],
    [
      "lof",
      "Compares local neighborhood densities",
      "Local deviations across density regions",
      "Distance and neighborhood sensitivity",
    ],
    [
      "ecod",
      "Empirical marginal tail probabilities",
      "Fast non-iterative scoring",
      "Can miss dependence-only anomalies",
    ],
    [
      "autoencoder",
      "Neural reconstruction error",
      "Nonlinear compression",
      "May reconstruct anomalies well",
    ],
    [
      "vae",
      "Regularized probabilistic reconstruction",
      "Latent representations",
      "Prior and optimization sensitivity",
    ],
    [
      "deepsvdd",
      "Distance to a learned normal center",
      "One-class objective",
      "Multiple normal modes",
    ],
    [
      "lunar",
      "Learned neighborhood distance aggregation",
      "Adaptable local scoring",
      "Synthetic-negative assumptions",
    ],
    [
      "so_gaal",
      "Adversarial outlier discriminator",
      "Nonlinear decision boundaries",
      "Adversarial training instability",
    ],
  ].map(([id, mechanism, strengths, limitations]) => [
    id,
    {
      mechanism,
      strengths,
      limitations,
      properties: ["effective_dimension", "n_train"],
      source: "https://github.com/USC-FORTIS/AD-LLM",
    },
  ]),
);
export const demoProfile: Profile = {
  split: "train",
  seed: 42,
  sample_rows: [0, 1, 2, 3, 4, 5, 6, 7],
  features: {
    n_train: 8,
    embedding_dimension: 768,
    profile_sample_size: 8,
    normal_category_count: 4,
    mean_tokens: 12.1,
    intra_class_compactness: 0.18,
    inter_class_separability: 0.31,
    effective_dimension: 5.8,
    mean_knn_cosine_distance: 0.24,
    density_variation: 0.17,
    normal_anomaly_separability: null,
  },
  definitions: {
    intra_class_compactness:
      "Mean cosine distance to the normal-category centroid. Smaller is tighter.",
    inter_class_separability:
      "Distance between normal-category centroids. Does not compare normal vs anomaly.",
    effective_dimension:
      "Participation ratio of centered embedding singular values.",
  },
};
export function demoRetrieve(
  query: string,
  k: number,
  records = demoTrain,
  dataset = "demo_news",
): Reference[] {
  return records
    .filter((r) => normalize(r.text) !== normalize(query))
    .slice(0, k)
    .map((r, i) => ({
      number: i + 1,
      doc_id: r.id,
      text: r.text,
      category: r.category ?? "Unknown",
      distance: 0.16 + i * 0.08,
      split: "train",
      dataset,
    }));
}
export function demoDetect(
  text: string,
  k: number,
  dataset: string,
  baseline: boolean,
  records = demoTrain,
): DetectResponse {
  const value =
    ((Array.from(text).reduce((s, c) => s + c.charCodeAt(0), 0) % 80) + 10) /
    100;
  const refs = demoRetrieve(text, k, records, dataset);
  const result = {
    score: value,
    verdict: value >= 0.5 ? "anomaly" : "normal",
    reason:
      "Illustrative response only. This deterministic preview exercises the interface; it does not classify text or measure model performance.",
    model: "Preview simulator",
    error: null,
    prompt: JSON.stringify(
      {
        query: text,
        references: refs.map((r) => ({ id: r.doc_id, text: r.text })),
        notice: "Illustrative prompt. No labels included.",
      },
      null,
      2,
    ),
    citations: refs.length ? [1] : [],
  };
  return {
    dataset,
    setting: "normal_only",
    k,
    rag: result,
    baseline: baseline
      ? {
          ...result,
          score: 0.5,
          citations: [],
          prompt: JSON.stringify({
            query: text,
            notice: "Illustrative baseline prompt",
          }),
        }
      : null,
    retrieved: refs,
  };
}
export function demoProfileFor(records?: Row[]): Profile {
  if (!records) return demoProfile;
  return {
    ...demoProfile,
    sample_rows: [],
    features: {
      ...Object.fromEntries(
        Object.keys(demoProfile.features).map((k) => [k, null]),
      ),
      n_train: records.length,
      normal_category_count: new Set(
        records.map((r) => r.category).filter(Boolean),
      ).size,
      mean_tokens: records.length
        ? records.reduce((n, r) => n + r.text.trim().split(/\s+/).length, 0) /
          records.length
        : null,
    },
  };
}
export function demoSelection(
  candidates: string[],
  profile = demoProfile,
): Selection {
  return {
    selected_model: candidates[0],
    ranking: candidates,
    reason:
      "Preview ranking in candidate order, not an LLM recommendation. Connect the backend to obtain a measured training profile and a real model-selection response.",
    model: "Preview simulator",
    features: profile.features,
    prompt:
      "Illustrative model-selection request: training features and candidate capability cards only.",
    seconds: 0.6,
    grounding: {
      g_score: 1,
      valid_citations: 1,
      total_citations: 1,
      feature_coverage: 1 / 12,
      checks: [
        {
          feature: "n_train",
          value: profile.features.n_train ?? 0,
          expected: profile.features.n_train ?? 0,
          valid: true,
        },
      ],
    },
  };
}
export const demoExperiments: Experiment[] = [
  {
    id: "illustrative-comparison",
    dataset: "demo_news",
    embedding: "BAAI/bge-base-en-v1.5",
    command: "detect",
    n_train: 8,
    n_test: 4,
    seed: 42,
    updated_at: "2026-10-09T00:00:00Z",
    results: [
      {
        k: 0,
        metrics: {
          auroc: 0.75,
          auprc: 0.833333,
          n_total: 4,
          n_valid: 4,
          n_errors: 0,
          coverage: 1,
        },
      },
      {
        k: 3,
        metrics: {
          auroc: 1,
          auprc: 1,
          n_total: 4,
          n_valid: 4,
          n_errors: 0,
          coverage: 1,
        },
      },
    ],
  },
];
