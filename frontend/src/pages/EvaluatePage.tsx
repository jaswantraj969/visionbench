import { useState } from "react";
import type { FormEvent } from "react";
import { useSearchParams } from "react-router-dom";

import {
  useCompare,
  useEvaluation,
  useEvaluations,
  useExperiments,
  useStartEvaluation,
} from "../api/hooks";
import type { CI, ComparisonReport, EvaluationDetail } from "../api/types";
import { ConfusionMatrix } from "../components/ConfusionMatrix";
import { JobPanel } from "../components/JobPanel";
import {
  ErrorBox,
  ExperimentLink,
  NotAvailable,
  PageHeader,
  QueryState,
  Section,
  SplitLabel,
} from "../components/ui";
import { ci, num, signed } from "../lib/format";

export function EvaluatePage() {
  const [params, setParams] = useSearchParams();
  return (
    <>
      <PageHeader title="Evaluate & Compare">
        Evaluate a trained model on the validation split, inspect an evaluation, or compare recorded
        evaluations. Validation numbers are selection-biased: the validation split chose each
        checkpoint.
      </PageHeader>
      <StartEvaluation initial={params.get("experiment") ?? ""} />
      <EvaluationViewer
        evaluationId={params.get("evaluation") ?? ""}
        onPick={(id) => setParams(id ? { evaluation: id } : {})}
      />
      <Compare />
    </>
  );
}

function StartEvaluation({ initial }: { initial: string }) {
  const exps = useExperiments();
  const [exp, setExp] = useState(initial);
  const start = useStartEvaluation(exp);
  const [jobId, setJobId] = useState<string | null>(null);
  const completed = (exps.data ?? []).filter((e) => e.status === "completed");
  function submit(e: FormEvent) {
    e.preventDefault();
    if (!exp) return;
    start.mutate({ split: "val", checkpoint: "best" }, { onSuccess: (r) => setJobId(r.job.id) });
  }
  return (
    <Section title="Run a VAL evaluation" id="start-eval">
      <form onSubmit={submit}>
        <div className="field">
          <label htmlFor="ev-exp">Experiment (completed)</label>
          <select id="ev-exp" value={exp} onChange={(e) => setExp(e.target.value)}>
            <option value="">— choose —</option>
            {completed.map((e) => (
              <option key={e.id} value={e.id}>
                {e.id} · {e.model} · seed {e.seed}
                {e.tags.length ? ` · ${e.tags.join(", ")}` : ""}
              </option>
            ))}
          </select>
        </div>
        <button type="submit" disabled={!exp || start.isPending}>
          Start VAL evaluation (job)
        </button>
      </form>
      <NotAvailable
        label="TEST evaluation"
        reason="the test split is evaluated exactly once, at the final release, through the pre-registered procedure; it is never started from the UI"
      />
      {start.isError ? <ErrorBox error={start.error} title="Evaluation refused" /> : null}
      {jobId ? <JobPanel jobId={jobId} /> : null}
    </Section>
  );
}

function ciOf(doc: EvaluationDetail["metrics"], key: string): CI {
  return doc.bootstrap?.metrics?.[key] ?? { low: null, high: null };
}

export function EvaluationView({ ev }: { ev: EvaluationDetail }) {
  const m = ev.metrics.metrics;
  const undef = Object.entries(m.undefined ?? {});
  return (
    <>
      <p>
        <SplitLabel split={ev.evaluation.split} /> evaluation{" "}
        <span className="mono">{ev.evaluation.id}</span> of{" "}
        <ExperimentLink id={ev.evaluation.experiment_id} /> ({ev.metrics.model}, checkpoint{" "}
        {ev.evaluation.checkpoint}, epoch {ev.metrics.checkpoint?.epoch})
      </p>
      <table>
        <caption>
          Metrics with 95 % percentile bootstrap CIs ({ev.metrics.bootstrap?.resamples} resamples)
        </caption>
        <thead>
          <tr>
            <th scope="col">metric</th>
            <th scope="col">value [95 % CI]</th>
          </tr>
        </thead>
        <tbody>
          {(["accuracy", "balanced_accuracy", "macro_f1"] as const).map((k) => {
            const c = ciOf(ev.metrics, k);
            return (
              <tr key={k}>
                <td>{k.replace("_", " ")}</td>
                <td className="mono">{ci(m[k], c.low, c.high)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <h3>Per class</h3>
      <div className="table-scroll">
        <table>
          <caption>
            Per-class precision, recall and F1 (“undefined” = no support or no predictions; never 0)
          </caption>
          <thead>
            <tr>
              <th scope="col">class</th>
              <th scope="col" className="num">
                support
              </th>
              <th scope="col" className="num">
                predicted
              </th>
              <th scope="col" className="num">
                precision
              </th>
              <th scope="col" className="num">
                recall
              </th>
              <th scope="col" className="num">
                F1
              </th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(m.per_class).map(([cls, r]) => (
              <tr key={cls}>
                <td>{cls}</td>
                <td className="num">{r.support}</td>
                <td className="num">{r.predicted}</td>
                <td className="num">{num(r.precision)}</td>
                <td className="num">{num(r.recall)}</td>
                <td className="num">{num(r.f1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {undef.length ? (
        <>
          <h3>Undefined values</h3>
          <ul>
            {undef.map(([k, reason]) => (
              <li key={k}>
                <span className="mono">{k}</span>: undefined — {reason}
              </li>
            ))}
          </ul>
        </>
      ) : null}
      <h3>Confusion matrix</h3>
      <ConfusionMatrix
        labels={m.confusion_matrix.labels}
        counts={m.confusion_matrix.counts}
        rowNormalized={m.confusion_matrix.row_normalized}
      />
    </>
  );
}

function EvaluationViewer({
  evaluationId,
  onPick,
}: {
  evaluationId: string;
  onPick: (id: string) => void;
}) {
  const list = useEvaluations();
  const ev = useEvaluation(evaluationId || undefined);
  return (
    <Section title="Evaluation" id="eval-view">
      <div className="field">
        <label htmlFor="ev-pick">Recorded evaluation</label>
        <select id="ev-pick" value={evaluationId} onChange={(e) => onPick(e.target.value)}>
          <option value="">— choose —</option>
          {(list.data ?? []).map((r) => (
            <option key={r.id} value={r.id}>
              {r.id} · {r.split.toUpperCase()} · {r.experiment_id}
            </option>
          ))}
        </select>
      </div>
      {evaluationId ? (
        <QueryState query={ev} what="evaluation">
          {(d) => <EvaluationView ev={d} />}
        </QueryState>
      ) : (
        <p className="muted">Choose an evaluation to see its metrics and confusion matrix.</p>
      )}
    </Section>
  );
}

function Compare() {
  const evals = useEvaluations();
  const compare = useCompare();
  const [split, setSplit] = useState<"val" | "test">("val");
  const [chosen, setChosen] = useState<string[]>([]);
  const withSplit = [
    ...new Set((evals.data ?? []).filter((e) => e.split === split).map((e) => e.experiment_id)),
  ];
  function toggle(id: string) {
    setChosen((c) => (c.includes(id) ? c.filter((x) => x !== id) : [...c, id]));
  }
  function submit(e: FormEvent) {
    e.preventDefault();
    compare.mutate({
      experiment_ids: chosen,
      split,
      note: split === "val" ? "VALIDATION (selection-biased)" : "TEST",
    });
  }
  return (
    <Section title="Compare recorded evaluations" id="compare">
      <form onSubmit={submit}>
        <fieldset className="inline-options">
          <legend>Split</legend>
          {(["val", "test"] as const).map((s) => (
            <label key={s}>
              <input
                type="radio"
                name="cmp-split"
                checked={split === s}
                onChange={() => {
                  setSplit(s);
                  setChosen([]);
                }}
              />{" "}
              {s === "val" ? "VALIDATION" : "TEST"}
            </label>
          ))}
        </fieldset>
        {withSplit.length === 0 ? (
          <p className="muted">
            No experiment has a recorded {split === "val" ? "VALIDATION" : "TEST"} evaluation.
          </p>
        ) : (
          <fieldset>
            <legend>Experiments (at least 2)</legend>
            {withSplit.map((id) => (
              <label key={id} style={{ display: "block" }}>
                <input type="checkbox" checked={chosen.includes(id)} onChange={() => toggle(id)} />{" "}
                <span className="mono">{id}</span>
              </label>
            ))}
          </fieldset>
        )}
        <button type="submit" disabled={chosen.length < 2 || compare.isPending}>
          Compare
        </button>
      </form>
      {compare.isError ? <ErrorBox error={compare.error} title="Comparison failed" /> : null}
      {compare.data ? <ComparisonView report={compare.data} /> : null}
    </Section>
  );
}

export function ComparisonView({ report }: { report: ComparisonReport }) {
  const split = report.split;
  return (
    <div className="comparison">
      <h3>
        Report <span className="mono">{report.report_id}</span> · <SplitLabel split={split} /> ·{" "}
        {report.n_samples} images
      </h3>
      {report.selection_bias_note ? <p className="warning">{report.selection_bias_note}</p> : null}
      <div className="table-scroll">
        <table>
          <caption>
            {split === "val" ? "VALIDATION (selection-biased)" : "TEST"} metrics [95 % bootstrap CI]
          </caption>
          <thead>
            <tr>
              <th scope="col">experiment</th>
              <th scope="col">model</th>
              <th scope="col">balanced accuracy</th>
              <th scope="col">macro F1</th>
              <th scope="col">accuracy</th>
              <th scope="col">Δ balanced acc. vs reference</th>
              {split === "val" ? (
                <th scope="col" className="num">
                  VAL rank
                </th>
              ) : null}
            </tr>
          </thead>
          <tbody>
            {report.rows.map((r) => {
              const m = ((split === "test" ? r.test : r.val) ?? {}) as Record<
                string,
                number | CI | null
              >;
              const cell = (k: string) => {
                const c = (m[`${k}_ci`] ?? { low: null, high: null }) as CI;
                return ci(m[k] as number | null, c.low, c.high);
              };
              const d = r.paired_difference_vs_reference?.metrics?.balanced_accuracy;
              return (
                <tr key={r.experiment_id}>
                  <td>
                    <ExperimentLink id={r.experiment_id} />
                  </td>
                  <td>{r.model}</td>
                  <td className="mono">{cell("balanced_accuracy")}</td>
                  <td className="mono">{cell("macro_f1")}</td>
                  <td className="mono">{cell("accuracy")}</td>
                  <td className="mono">
                    {r.experiment_id === report.reference_experiment
                      ? "reference"
                      : d
                        ? `${signed(d.point)} [${signed(d.low)}, ${signed(d.high)}]`
                        : "undefined"}
                  </td>
                  {split === "val" ? (
                    <td className="num">{report.val_ranking?.ranks?.[r.experiment_id] ?? "—"}</td>
                  ) : null}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {split === "test" ? (
        <p className="muted">TEST results carry no winner label; ranking uses validation only.</p>
      ) : null}
      <h3>Confound flags</h3>
      {report.confounds.length === 0 ? (
        <p className="muted">none</p>
      ) : (
        <ul>
          {report.confounds.map((c) => (
            <li key={c.code}>
              <strong>{c.code}</strong>: {c.message}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
