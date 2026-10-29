import { useState } from "react";
import { useSearchParams } from "react-router-dom";

import { useBenchmark, useBenchmarkDocument, useBenchmarks } from "../api/hooks";
import type { BenchmarkDetail, HashStatus } from "../api/types";
import {
  ExperimentLink,
  KeyValues,
  PageHeader,
  QueryState,
  Section,
  SplitLabel,
} from "../components/ui";
import { meanSd, num, seconds } from "../lib/format";
import { FinalReportSection } from "./FinalReportSection";

const DOC_LABELS: Record<string, string> = {
  benchmark_protocol: "Benchmark protocol",
  robustness_protocol: "Robustness protocol",
  final_analysis_plan: "Final analysis plan",
  robustness_val: "Robustness report (VALIDATION)",
  final_report: "Final report (TEST)",
};

function hashText(h: HashStatus): string {
  if (!h.present) return "not present";
  return h.ok
    ? `frozen — ${h.files.length} file(s), all hashes OK`
    : `HASH MISMATCH: ${h.files
        .filter((f) => !f.ok)
        .map((f) => f.name)
        .join(", ")}`;
}

export function BenchmarkPage() {
  const [params, setParams] = useSearchParams();
  const list = useBenchmarks();
  const name = params.get("name") ?? list.data?.[0]?.name;
  const bench = useBenchmark(name);
  return (
    <>
      <PageHeader title="Benchmark">
        Frozen, pre-registered benchmarks. Protocols and configs are hashed; VALIDATION numbers
        chose the checkpoints; TEST is evaluated once.
      </PageHeader>
      <QueryState
        query={list}
        what="benchmarks"
        isEmpty={(l) => l.length === 0}
        empty="No frozen benchmark found."
      >
        {(l) =>
          l.length > 1 ? (
            <div className="field">
              <label htmlFor="bm-name">Benchmark</label>
              <select
                id="bm-name"
                value={name}
                onChange={(e) => setParams({ name: e.target.value })}
              >
                {l.map((b) => (
                  <option key={b.name}>{b.name}</option>
                ))}
              </select>
            </div>
          ) : null
        }
      </QueryState>
      {name ? (
        <QueryState query={bench} what="benchmark">
          {(b) => <BenchmarkView b={b} />}
        </QueryState>
      ) : null}
    </>
  );
}

function BenchmarkView({ b }: { b: BenchmarkDetail }) {
  const docs = Object.entries(b.documents)
    .filter(([, present]) => present)
    .map(([k]) => k);
  const [doc, setDoc] = useState<string>("");
  const document = useBenchmarkDocument(b.name, doc || undefined);
  return (
    <>
      <Section title={`Protocol: ${b.name}`} id="protocol">
        <KeyValues
          rows={[
            ["benchmark protocol + configs", hashText(b.protocol_hashes)],
            ["robustness protocol + suite", hashText(b.robustness_hashes)],
            ["final analysis plan", hashText(b.final_analysis_plan_hashes)],
          ]}
        />
        <div className="field">
          <label htmlFor="bm-doc">Read a document</label>
          <select id="bm-doc" value={doc} onChange={(e) => setDoc(e.target.value)}>
            <option value="">— choose —</option>
            {docs.map((d) => (
              <option key={d} value={d}>
                {DOC_LABELS[d] ?? d}
              </option>
            ))}
          </select>
        </div>
        {doc ? (
          <QueryState query={document} what="document">
            {(d) => (
              <pre className="markdown" aria-label={DOC_LABELS[doc] ?? doc}>
                {d.markdown}
              </pre>
            )}
          </QueryState>
        ) : null}
      </Section>
      <Section title="Runs" id="runs">
        <div className="table-scroll">
          <table>
            <caption>
              Benchmark runs (val-selected checkpoints). Validation columns are{" "}
              <SplitLabel split="val" />.
            </caption>
            <thead>
              <tr>
                <th scope="col">model</th>
                <th scope="col" className="num">
                  seed
                </th>
                <th scope="col">experiment</th>
                <th scope="col">status</th>
                <th scope="col" className="num">
                  best epoch
                </th>
                <th scope="col" className="num">
                  epochs
                </th>
                <th scope="col">stopped</th>
                <th scope="col" className="num">
                  val bal. acc.
                </th>
                <th scope="col" className="num">
                  val macro F1
                </th>
                <th scope="col" className="num">
                  train time
                </th>
                <th scope="col" className="num">
                  TEST evals
                </th>
              </tr>
            </thead>
            <tbody>
              {b.runs.map((r) => {
                const val = r.val as
                  | { balanced_accuracy: number; macro_f1: number }
                  | null
                  | undefined;
                return (
                  <tr key={`${r.model}-${r.seed}`}>
                    <td>{String(r.model)}</td>
                    <td className="num">{String(r.seed)}</td>
                    <td>
                      {r.experiment_id ? <ExperimentLink id={String(r.experiment_id)} /> : "—"}
                    </td>
                    <td>{String(r.status ?? r.driver_status)}</td>
                    <td className="num">{String(r.best_epoch ?? "—")}</td>
                    <td className="num">{String(r.epochs_trained ?? "—")}</td>
                    <td>{String(r.stopped_reason ?? "—")}</td>
                    <td className="num">{num(val?.balanced_accuracy)}</td>
                    <td className="num">{num(val?.macro_f1)}</td>
                    <td className="num">{seconds(r.train_seconds as number | null)}</td>
                    <td className="num">{String(r.test_evaluations ?? 0)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p className="hint">Train time is indicative (one shared laptop GPU).</p>
      </Section>
      <Section title="Multi-seed summary — VALIDATION (selection-biased)" id="val-summary">
        <p>
          <SplitLabel split="val" /> {b.val_label}. Mean ± sample sd over seeds; the split is fixed,
          so the sd reflects training variance only.
        </p>
        <table>
          <caption>VALIDATION balanced accuracy and macro F1 per model</caption>
          <thead>
            <tr>
              <th scope="col">model</th>
              <th scope="col">balanced accuracy (mean ± sd)</th>
              <th scope="col">macro F1 (mean ± sd)</th>
              <th scope="col">per-seed balanced accuracy</th>
              <th scope="col">best epochs</th>
            </tr>
          </thead>
          <tbody>
            {b.val_summary.map((s) => {
              const ba = s.balanced_accuracy as {
                mean: number | null;
                sd: number | null;
                values: number[];
              };
              const f1 = s.macro_f1 as { mean: number | null; sd: number | null };
              return (
                <tr key={String(s.model)}>
                  <td>{String(s.model)}</td>
                  <td className="mono">{meanSd(ba.mean, ba.sd)}</td>
                  <td className="mono">{meanSd(f1.mean, f1.sd)}</td>
                  <td className="mono">{ba.values.map((v) => num(v)).join(" / ")}</td>
                  <td>{(s.best_epochs as (number | null)[]).join(" / ")}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </Section>
      <FinalReportSection report={b.final_report} />
    </>
  );
}
