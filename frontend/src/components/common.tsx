import { useState, type ReactNode } from "react";
import {
  ArrowDown,
  ArrowUp,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  FileSearch,
  Info,
  Loader2,
  Search,
  ShieldCheck,
  TriangleAlert,
} from "lucide-react";
import { Button } from "./ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "./ui/dialog";
import { cn, number } from "../lib/utils";
import type { Reference } from "../types";
export { Button };
export function PageHeader({
  eyebrow,
  title,
  description,
  actions,
}: {
  eyebrow: string;
  title: string;
  description: string;
  actions?: ReactNode;
}) {
  return (
    <div className="page-heading">
      <div>
        <div className="eyebrow">{eyebrow}</div>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      <div className="heading-actions">{actions}</div>
    </div>
  );
}
export function Panel({
  title,
  description,
  action,
  children,
  className,
}: {
  title?: string;
  description?: string;
  action?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={cn("panel", className)}>
      {title && (
        <div className="panel-head">
          <div>
            <h2>{title}</h2>
            {description && <p>{description}</p>}
          </div>
          {action}
        </div>
      )}
      <div className="panel-body">{children}</div>
    </section>
  );
}
export function Status({
  children,
  tone = "neutral",
  dot = true,
}: {
  children: ReactNode;
  tone?: "green" | "amber" | "red" | "blue" | "neutral";
  dot?: boolean;
}) {
  return (
    <span className={`status ${tone}`}>
      {dot && <span />}
      {children}
    </span>
  );
}
export function Metric({
  label,
  value,
  note,
  icon,
  tone = "blue",
}: {
  label: string;
  value: ReactNode;
  note?: string;
  icon?: ReactNode;
  tone?: string;
}) {
  return (
    <div className="metric">
      <div className="metric-top">
        <span>{label}</span>
        <span className={`metric-icon ${tone}`}>{icon}</span>
      </div>
      <strong>{value}</strong>
      {note && <small>{note}</small>}
    </div>
  );
}
export function Notice({
  children,
  tone = "blue",
}: {
  children: ReactNode;
  tone?: "blue" | "amber" | "green" | "red";
}) {
  const Icon =
    tone === "red" || tone === "amber"
      ? TriangleAlert
      : tone === "green"
        ? ShieldCheck
        : Info;
  return (
    <div
      className={`notice ${tone}`}
      role={tone === "red" ? "alert" : undefined}
    >
      <Icon size={17} />
      <div>{children}</div>
    </div>
  );
}
export function Empty({
  title,
  description,
  action,
}: {
  title: string;
  description: string;
  action?: ReactNode;
}) {
  return (
    <div className="empty">
      <span>
        <FileSearch size={25} />
      </span>
      <h3>{title}</h3>
      <p>{description}</p>
      {action}
    </div>
  );
}
export function Loading({ text = "Loading workspace…" }: { text?: string }) {
  return (
    <div className="loading" role="status">
      <Loader2 className="spin" size={20} />
      {text}
    </div>
  );
}
export function Field({
  label,
  children,
  hint,
}: {
  label: string;
  children: ReactNode;
  hint?: string;
}) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
      {hint && <small>{hint}</small>}
    </label>
  );
}
export function Toggle({
  on,
  onChange,
  children,
}: {
  on: boolean;
  onChange: (v: boolean) => void;
  children: ReactNode;
}) {
  return (
    <label className="check">
      <input
        type="checkbox"
        checked={on}
        onChange={(e) => onChange(e.target.checked)}
      />
      <span>{children}</span>
    </label>
  );
}
export function Score({
  score,
  threshold = 0.5,
}: {
  score: number;
  threshold?: number;
}) {
  const anomaly = score >= threshold;
  return (
    <div className="score">
      <div className="score-head">
        <div>
          <span>ANOMALY SCORE</span>
          <strong className={anomaly ? "red-text" : "green-text"}>
            {score.toFixed(2)}
          </strong>
        </div>
        <Status tone={anomaly ? "red" : "green"}>
          {anomaly ? "Anomaly detected" : "Normal"}
        </Status>
      </div>
      <div className="score-track">
        <div
          style={{
            width: `${score * 100}%`,
            background: anomaly ? "#e35661" : "#239a74",
          }}
        />
        <i style={{ left: `${threshold * 100}%` }} />
      </div>
      <div className="score-labels">
        <span>0 · Normal</span>
        <span>Threshold {threshold.toFixed(2)}</span>
        <span>1 · Anomaly</span>
      </div>
    </div>
  );
}
export function DataTable<T extends object>({
  rows,
  columns,
  searchPlaceholder = "Search records…",
  onRow,
}: {
  rows: T[];
  columns: {
    key: string;
    label: string;
    render?: (row: T) => ReactNode;
    sortValue?: (row: T) => string | number;
  }[];
  searchPlaceholder?: string;
  onRow?: (row: T) => void;
}) {
  const [search, setSearch] = useState(""),
    [page, setPage] = useState(0),
    [sort, setSort] = useState({ key: "", asc: true });
  const filtered = rows
    .filter((r) =>
      JSON.stringify(r).toLowerCase().includes(search.toLowerCase()),
    )
    .sort((a, b) => {
      const col = columns.find((c) => c.key === sort.key);
      const x = col?.sortValue ? col.sortValue(a) : a[sort.key as keyof T],
        y = col?.sortValue ? col.sortValue(b) : b[sort.key as keyof T];
      return (
        (typeof x === "number" && typeof y === "number"
          ? x - y
          : String(x ?? "").localeCompare(String(y ?? ""))) *
        (sort.asc ? 1 : -1)
      );
    });
  const max = Math.max(0, Math.ceil(filtered.length / 8) - 1),
    safe = Math.min(page, max);
  return (
    <div>
      <div className="table-tools">
        <div className="search-input">
          <Search size={16} />
          <input
            aria-label={searchPlaceholder}
            placeholder={searchPlaceholder}
            value={search}
            onChange={(e) => {
              setSearch(e.target.value);
              setPage(0);
            }}
          />
        </div>
        <small>{number(filtered.length)} records</small>
      </div>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              {columns.map((c) => (
                <th key={c.key}>
                  <button
                    onClick={() =>
                      setSort({
                        key: c.key,
                        asc: sort.key === c.key ? !sort.asc : true,
                      })
                    }
                  >
                    {c.label}
                    {sort.key === c.key ? (
                      sort.asc ? (
                        <ArrowUp size={12} />
                      ) : (
                        <ArrowDown size={12} />
                      )
                    ) : null}
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filtered.slice(safe * 8, safe * 8 + 8).map((r, i) => (
              <tr key={i}>
                {columns.map((c, j) => (
                  <td key={c.key}>
                    {j === 0 && onRow ? (
                      <button className="text-link" onClick={() => onRow(r)}>
                        {c.render
                          ? c.render(r)
                          : String(r[c.key as keyof T] ?? "—")}
                      </button>
                    ) : c.render ? (
                      c.render(r)
                    ) : (
                      String(r[c.key as keyof T] ?? "—")
                    )}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
        {!filtered.length && <p className="no-rows">No matching records.</p>}
      </div>
      <div className="pagination">
        <small>
          Page {safe + 1} of {max + 1}
        </small>
        <div>
          <Button
            variant="outline"
            size="icon-sm"
            aria-label="Previous page"
            disabled={safe === 0}
            onClick={() => setPage(safe - 1)}
          >
            <ChevronLeft />
          </Button>
          <Button
            variant="outline"
            size="icon-sm"
            aria-label="Next page"
            disabled={safe === max}
            onClick={() => setPage(safe + 1)}
          >
            <ChevronRight />
          </Button>
        </div>
      </div>
    </div>
  );
}
export function References({
  references,
  citations = [],
}: {
  references: Reference[];
  citations?: number[];
}) {
  const [selected, setSelected] = useState<Reference | null>(null);
  return (
    <>
      <div className="reference-list">
        {references.map((r) => (
          <button
            key={r.doc_id}
            className={cn("reference", citations.includes(r.number) && "cited")}
            onClick={() => setSelected(r)}
          >
            <span className="rank">{r.number.toString().padStart(2, "0")}</span>
            <div>
              <div className="reference-title">
                <strong>{r.category || "Training reference"}</strong>
                <span>{((1 - r.distance) * 100).toFixed(1)}% similarity</span>
              </div>
              <p>{r.text}</p>
              <small>
                {r.doc_id} <b>·</b> TRAIN SPLIT{" "}
                {citations.includes(r.number) && (
                  <em>
                    <CheckCircle2 size={12} /> Cited
                  </em>
                )}
              </small>
            </div>
          </button>
        ))}
      </div>
      <Dialog open={!!selected} onOpenChange={(v) => !v && setSelected(null)}>
        <DialogContent className="detail-dialog">
          <DialogHeader>
            <DialogTitle>Reference {selected?.number}</DialogTitle>
            <DialogDescription>
              {selected?.doc_id} · Training split · {selected?.category}
            </DialogDescription>
          </DialogHeader>
          <p className="full-text">{selected?.text}</p>
          <Notice>
            Reference text returned by the service. Cosine distance:{" "}
            {selected?.distance.toFixed(5)}. Semantic similarity is not proof of
            normality.
          </Notice>
        </DialogContent>
      </Dialog>
    </>
  );
}
