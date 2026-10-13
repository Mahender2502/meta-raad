import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
export const number = (v: number | null | undefined, digits = 0) =>
  v == null
    ? "—"
    : v.toLocaleString(undefined, { maximumFractionDigits: digits });
const modelNames: Record<string, string> = {
  iforest: "Isolation Forest",
  lof: "LOF",
  ecod: "ECOD",
  autoencoder: "Autoencoder",
  vae: "VAE",
  deepsvdd: "DeepSVDD",
  lunar: "LUNAR",
  so_gaal: "SO-GAAL",
  knn: "KNN",
};
export const title = (s: string) =>
  modelNames[s] ??
  s.replaceAll("_", " ").replace(/\b\w/g, (c) => c.toUpperCase());
export function download(
  name: string,
  data: unknown,
  format: "json" | "csv" | "jsonl" = "json",
) {
  let text: string;
  if (format === "csv" && Array.isArray(data)) {
    const keys = Object.keys(data[0] ?? {});
    const cell = (v: unknown) =>
      '"' + String(v ?? "").replaceAll('"', '""') + '"';
    text = [
      keys.join(","),
      ...data.map((r) => keys.map((k) => cell(r[k])).join(",")),
    ].join("\r\n");
  } else
    text =
      format === "jsonl" && Array.isArray(data)
        ? data.map((r) => JSON.stringify(r)).join("\n")
        : JSON.stringify(data, null, 2);
  const url = URL.createObjectURL(
    new Blob([text], {
      type: format === "csv" ? "text/csv" : "application/json",
    }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = `${name}.${format}`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
