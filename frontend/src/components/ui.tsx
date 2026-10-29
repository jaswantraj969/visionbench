import type { UseQueryResult } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { Link } from "react-router-dom";

import { ApiError, errorText } from "../api/client";

export const VAL_LABEL = "VALIDATION (selection-biased)";
export const ATTRIBUTION =
  "Images: EuroSAT (Helber et al., 2019; MIT license). Contains modified Copernicus Sentinel data " +
  "(acquisition year not stated by the dataset authors).";

export function Loading({ what = "data" }: { what?: string }) {
  return (
    <p className="state state-loading" role="status" aria-live="polite">
      Loading {what}…
    </p>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <p className="state state-empty">{children}</p>;
}

export function ErrorBox({ error, title = "Request failed" }: { error: unknown; title?: string }) {
  const code = error instanceof ApiError ? error.code : undefined;
  return (
    <div className="state state-error" role="alert">
      <strong>{title}</strong>
      <p>{errorText(error)}</p>
      {code ? <p className="muted">error code: {code}</p> : null}
    </div>
  );
}

/** Render a query's loading / error / empty / success states consistently. */
export function QueryState<T>({
  query,
  what,
  isEmpty,
  empty,
  children,
}: {
  query: UseQueryResult<T>;
  what: string;
  isEmpty?: (data: T) => boolean;
  empty?: ReactNode;
  children: (data: T) => ReactNode;
}) {
  if (query.isPending) return <Loading what={what} />;
  if (query.isError) return <ErrorBox error={query.error} title={`Could not load ${what}`} />;
  if (isEmpty && isEmpty(query.data)) return <Empty>{empty ?? `No ${what} yet.`}</Empty>;
  return <>{children(query.data)}</>;
}

export function SplitLabel({ split }: { split: string }) {
  const s = split.toLowerCase();
  const text = s === "val" ? VAL_LABEL : s === "test" ? "TEST" : s.toUpperCase();
  return <span className={`split-label split-${s}`}>{text}</span>;
}

export function StatusBadge({ status }: { status: string }) {
  return <span className={`badge status-${status}`}>{status}</span>;
}

export function Tag({ text }: { text: string }) {
  return <span className="tag">{text}</span>;
}

/** A control that cannot work here, shown as such with the reason (never a fake button). */
export function NotAvailable({ label, reason }: { label: string; reason: string }) {
  return (
    <p className="not-available">
      <span aria-hidden="true">⊘ </span>
      <strong>{label}</strong>: not available — {reason}
    </p>
  );
}

export function PageHeader({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <header className="page-header">
      <h1>{title}</h1>
      {children ? <div className="page-intro">{children}</div> : null}
    </header>
  );
}

export function Section({
  title,
  children,
  id,
}: {
  title: string;
  children: ReactNode;
  id?: string;
}) {
  return (
    <section className="card" aria-labelledby={id}>
      <h2 id={id}>{title}</h2>
      {children}
    </section>
  );
}

export function ExperimentLink({ id }: { id: string }) {
  return (
    <Link to={`/experiments/${id}`} className="mono">
      {id}
    </Link>
  );
}

export function JobLink({ id }: { id: string }) {
  return (
    <Link to={`/jobs/${id}`} className="mono">
      {id}
    </Link>
  );
}

export function KeyValues({ rows }: { rows: [string, ReactNode][] }) {
  return (
    <dl className="kv">
      {rows.map(([k, v]) => (
        <div key={k} className="kv-row">
          <dt>{k}</dt>
          <dd>{v}</dd>
        </div>
      ))}
    </dl>
  );
}

export function Attribution() {
  return <p className="attribution">{ATTRIBUTION}</p>;
}
