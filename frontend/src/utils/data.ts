import Papa from "papaparse";
import type { Row, Validation, Prediction } from "../types";
export const normalize = (s: string) =>
  s.toLowerCase().trim().replace(/\s+/g, " ");
export function parseFile(
  text: string,
  name: string,
): Record<string, unknown>[] {
  if (name.toLowerCase().endsWith(".jsonl"))
    return text
      .replace(/^\uFEFF/, "")
      .split(/\r?\n/)
      .filter((s) => s.trim())
      .map((s, i) => {
        try {
          const r = JSON.parse(s);
          if (!r || typeof r !== "object" || Array.isArray(r)) throw Error();
          return r;
        } catch {
          throw Error(`Invalid JSON object at line ${i + 1}.`);
        }
      });
  const result = Papa.parse<Record<string, unknown>>(text, {
    header: true,
    skipEmptyLines: true,
  });
  if (result.errors.length) throw Error(result.errors[0].message);
  return result.data;
}
export function mapRows(
  records: Record<string, unknown>[],
  textField: string,
  labelField: string,
  split: "train" | "test",
): Row[] {
  return records.map((r, i) => ({
    id: `${split}_${i.toString().padStart(5, "0")}`,
    text: typeof r[textField] === "string" ? (r[textField] as string) : "",
    label:
      r[labelField] === "" || r[labelField] == null
        ? undefined
        : typeof r[labelField] === "number" ||
            (typeof r[labelField] === "string" &&
              /^[01]$/.test((r[labelField] as string).trim()))
          ? Number(r[labelField])
          : NaN,
    category:
      typeof r.original_label === "string" ? r.original_label : undefined,
    split,
  }));
}
function jaccard(a: string, b: string) {
  const x = new Set(normalize(a).split(" ")),
    y = new Set(normalize(b).split(" "));
  let n = 0;
  x.forEach((v) => {
    if (y.has(v)) n++;
  });
  return n / (x.size + y.size - n || 1);
}
export function validateRows(
  rows: Row[],
  other: Row[] = [],
  nearThreshold = 0.9,
): Validation {
  const seen = new Set<string>(),
    otherTexts = new Set(other.map((r) => normalize(r.text)));
  let empty = 0,
    invalid = 0,
    missing = 0,
    duplicates = 0,
    overlap = 0;
  for (const r of rows) {
    const text = normalize(r.text);
    if (!text) empty++;
    if (r.label === undefined) missing++;
    else if (r.label !== 0 && r.label !== 1) invalid++;
    if (seen.has(text)) duplicates++;
    seen.add(text);
    if (otherTexts.has(text)) overlap++;
  }
  const normal = rows.filter((r) => r.label === 0).length,
    anomaly = rows.filter((r) => r.label === 1).length;
  const near: Validation["near"] = [];
  // Explicitly bounded review aid, not a semantic-embedding check or automatic deletion.
  for (const a of rows.slice(0, 100))
    for (const b of other.slice(0, 100)) {
      if (normalize(a.text) === normalize(b.text)) continue;
      const similarity = jaccard(a.text, b.text);
      if (similarity >= nearThreshold && near.length < 20)
        near.push({
          train: a.split === "train" ? a.id : b.id,
          test: a.split === "test" ? a.id : b.id,
          similarity,
        });
    }
  return {
    empty,
    invalid,
    missing,
    duplicates,
    overlap,
    near,
    normal,
    anomaly,
    blocked:
      empty > 0 ||
      invalid > 0 ||
      duplicates > 0 ||
      overlap > 0 ||
      (rows.some((r) => r.split === "train") && (anomaly > 0 || missing > 0)),
  };
}
export function evaluate(predictions: Prediction[], threshold = 0.5) {
  const valid = predictions.filter(
    (p) =>
      (p.label === 0 || p.label === 1) &&
      p.score !== null &&
      Number.isFinite(p.score),
  );
  const positives = valid.filter((p) => p.label === 1).length,
    negatives = valid.length - positives;
  let tp = 0,
    fp = 0,
    tn = 0,
    fn = 0;
  for (const p of valid) {
    if (p.score! >= threshold) {
      if (p.label === 1) tp++;
      else fp++;
    } else if (p.label === 1) fn++;
    else tn++;
  }
  const precision = tp + fp ? tp / (tp + fp) : 0,
    recall = tp + fn ? tp / (tp + fn) : 0;
  let auc = 0,
    ap = 0,
    lastFpr = 0,
    lastRecall = 0,
    cumTp = 0,
    cumFp = 0;
  const curve = [{ fpr: 0, recall: 0, precision: 1 }];
  const sorted = [...valid].sort((a, b) => b.score! - a.score!);
  for (let i = 0; i < sorted.length;) {
    let j = i;
    while (j < sorted.length && sorted[j].score === sorted[i].score) {
      if (sorted[j].label === 1) cumTp++;
      else cumFp++;
      j++;
    }
    const r = positives ? cumTp / positives : 0,
      f = negatives ? cumFp / negatives : 0,
      p = cumTp / (cumTp + cumFp);
    auc += ((f - lastFpr) * (r + lastRecall)) / 2;
    ap += (r - lastRecall) * p;
    curve.push({ fpr: f, recall: r, precision: p });
    lastFpr = f;
    lastRecall = r;
    i = j;
  }
  return {
    auroc: positives && negatives ? auc : null,
    auprc: positives && negatives ? ap : null,
    precision,
    recall,
    f1:
      precision + recall ? (2 * precision * recall) / (precision + recall) : 0,
    tp,
    fp,
    tn,
    fn,
    curve,
    valid: valid.length,
    total: predictions.length,
  };
}
