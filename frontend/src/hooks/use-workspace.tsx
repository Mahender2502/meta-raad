import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  useCallback,
  type ReactNode,
} from "react";
import type {
  Catalog,
  Dataset,
  Experiment,
  Preferences,
  Prediction,
  ResultDetail,
} from "../types";
import { createApi } from "../services/api";
const defaults: Preferences = {
  mode: "live",
  backendUrl: "",
  topK: 3,
  threshold: 0.5,
  similarityThreshold: 0.6,
  timeout: 180,
  embedding: "BAAI/bge-base-en-v1.5",
  provider: "FAISS",
  nearDuplicateThreshold: 0.9,
};
function readPrefs() {
  try {
    const saved = JSON.parse(
      localStorage.getItem("meta-raad.preferences") ?? "{}",
    );
    return { ...defaults, ...saved } as Preferences;
  } catch {
    return defaults;
  }
}
function useWorkspaceValue() {
  const [preferences, setPreferencesState] = useState<Preferences>(readPrefs),
    [catalog, setCatalog] = useState<Catalog | null>(null),
    [uploaded, setUploaded] = useState<Dataset[]>([]),
    [datasetId, setDatasetId] = useState(""),
    [experiments, setExperiments] = useState<Experiment[]>([]),
    [error, setError] = useState(""),
    [loading, setLoading] = useState(true),
    [connected, setConnected] = useState(false),
    [latest, setLatest] = useState<ResultDetail | null>(null),
    [batch, setBatch] = useState<Prediction[]>([]);
  const refreshVersion = useRef(0);
  const { mode, backendUrl, timeout } = preferences;
  const api = useMemo(
    () => createApi({ mode, backendUrl, timeout }),
    [mode, backendUrl, timeout],
  );
  const setPreferences = (p: Preferences) => {
    if (
      p.mode !== preferences.mode ||
      p.backendUrl !== preferences.backendUrl
    ) {
      refreshVersion.current++;
      setUploaded([]);
      setLatest(null);
      setBatch([]);
      setDatasetId("");
      setCatalog(null);
      setExperiments([]);
      setConnected(false);
    }
    localStorage.setItem("meta-raad.preferences", JSON.stringify(p));
    setPreferencesState(p);
  };
  const refresh = useCallback(async () => {
    const version = ++refreshVersion.current;
    setLoading(true);
    setError("");
    try {
      const [c, e] = await Promise.all([api.catalog(), api.experiments()]);
      if (version !== refreshVersion.current) return;
      setCatalog(c);
      setExperiments(e);
      setConnected(true);
      setDatasetId((old) => old || c.datasets[0]?.id || "");
    } catch (e) {
      if (version !== refreshVersion.current) return;
      setError(e instanceof Error ? e.message : "Backend unavailable");
      setConnected(false);
      setCatalog(null);
      setExperiments([]);
    } finally {
      if (version === refreshVersion.current) setLoading(false);
    }
  }, [api]);
  useEffect(() => {
    void refresh();
    return () => {
      refreshVersion.current++;
    };
  }, [refresh]);
  const datasets = Array.from(
    new Map(
      [...(catalog?.datasets ?? []), ...uploaded].map((d) => [d.id, d]),
    ).values(),
  );
  const dataset = datasets.find((d) => d.id === datasetId);
  const selectDataset = (id: string) => {
    setDatasetId(id);
    setLatest(null);
    setBatch([]);
  };
  const updateDataset = (d: Dataset) =>
    setUploaded((old) =>
      old.some((x) => x.id === d.id)
        ? old.map((x) => (x.id === d.id ? d : x))
        : [...old, d],
    );
  return {
    preferences,
    setPreferences,
    catalog,
    datasets,
    dataset,
    selectDataset,
    experiments,
    error,
    loading,
    connected,
    refresh,
    api,
    latest,
    setLatest,
    batch,
    setBatch,
    updateDataset,
  };
}
type Context = ReturnType<typeof useWorkspaceValue>;
const WorkspaceContext = createContext<Context | null>(null);
export function WorkspaceProvider({ children }: { children: ReactNode }) {
  const value = useWorkspaceValue();
  return (
    <WorkspaceContext.Provider value={value}>
      {children}
    </WorkspaceContext.Provider>
  );
}
export function useWorkspace() {
  const c = useContext(WorkspaceContext);
  if (!c) throw Error("Workspace context missing");
  return c;
}
