export type Mode = "live" | "demo";
export type Row = {
  id: string;
  text: string;
  label?: number;
  category?: string;
  split: "train" | "test";
};
export type SplitInfo = {
  count: number;
  available: boolean;
  normal?: number;
  anomaly?: number;
  empty_text?: number;
  invalid_labels?: number;
  duplicates?: number;
  categories?: Record<string, number>;
  preview: Row[];
};
export type Dataset = {
  id: string;
  name: string;
  source: string;
  local?: boolean;
  splits: { train: SplitInfo; test: SplitInfo };
  validation: {
    normal_train_only: boolean;
    exact_overlap: number;
    near_duplicates: string;
    vector_rows_aligned: boolean;
  };
  index: IndexInfo;
};
export type IndexInfo = {
  ready: boolean;
  provider: string;
  embedding: string;
  dimensions: number | null;
  count: number;
  split: string;
  collection: string;
  version: string;
  created_at: string | null;
  source_modified_at?: string | null;
};
export type Configuration = {
  llm: string;
  provider: string;
  embedding: string;
  version: string;
};
export type Catalog = {
  datasets: Dataset[];
  configuration: Configuration;
  capabilities: Record<string, boolean>;
};
export type Reference = {
  number: number;
  doc_id: string;
  text: string;
  category: string;
  distance: number;
  split?: string;
  dataset?: string;
};
export type Detection = {
  score: number | null;
  verdict: string | null;
  reason: string | null;
  model: string | null;
  error: string | null;
  prompt: string;
  citations: number[];
};
export type DetectResponse = {
  dataset: string;
  setting: string;
  k: number;
  baseline: Detection | null;
  rag: Detection | null;
  retrieved: Reference[];
};
export type Prediction = {
  baselineScore?: number | null;
  baselineError?: string | null;
  baselineReason?: string | null;
  reason?: string | null;
  id: string;
  label?: number;
  score: number | null;
  error?: string | null;
  pending?: boolean;
  text?: string;
};
export type ResultDetail = {
  query: string;
  response: DetectResponse;
  duration: number;
  timestamp: string;
  dataset: string;
  indexVersion: string;
  embedding: string;
  mode: Mode;
};
export type Profile = {
  features: Record<string, number | null>;
  definitions: Record<string, string>;
  split: string;
  seed: number;
  sample_rows: number[];
};
export type Card = {
  mechanism: string;
  strengths: string;
  limitations: string;
  properties: string[];
  source: string;
};
export type Selection = {
  selected_model: string;
  ranking: string[];
  reason: string;
  model: string;
  features: Record<string, number | null>;
  prompt: string;
  seconds: number;
  grounding: {
    g_score: number;
    valid_citations: number;
    total_citations: number;
    feature_coverage: number;
    checks: {
      feature: string;
      value: number;
      expected: number;
      valid: boolean;
    }[];
  };
};
export type RunResult = {
  model?: string;
  k?: number;
  setting?: string;
  metrics?: {
    auroc: number | null;
    auprc: number | null;
    n_total: number;
    n_valid: number;
    n_errors: number;
    coverage: number;
  };
  error?: string;
  pending?: number;
};
export type Experiment = {
  id: string;
  dataset: string;
  embedding: string;
  command: string;
  n_train: number;
  n_test?: number;
  seed: number;
  results: RunResult[];
  updated_at: string;
  setting?: string;
  llm?: { model?: string; provider?: string };
  legacy_reference_categories?: boolean;
  split_audit?: Record<string, number>;
};
export type Preferences = {
  mode: Mode;
  backendUrl: string;
  topK: number;
  threshold: number;
  similarityThreshold: number;
  timeout: number;
  embedding: string;
  provider: string;
  nearDuplicateThreshold: number;
};
export type Validation = {
  empty: number;
  invalid: number;
  missing: number;
  duplicates: number;
  overlap: number;
  near: { train: string; test: string; similarity: number }[];
  normal: number;
  anomaly: number;
  blocked: boolean;
};
