import { useMemo, useState } from "react";
import type { FormEvent } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import {
  useBenchmark,
  useBenchmarks,
  useExperiments,
  useRobustness,
  useStartRobustness,
  useSuites,
} from "../api/hooks";
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
import { meanSd } from "../lib/format";

interface RunRow {
  model: string;
  seed: number;
  clean_balanced_accuracy: number;
  conditions: Record<string, { balanced_accuracy: number } | null>;
}

function stats(values: number[]): { mean: number | null; sd: number | null } {
  if (!values.length) return { mean: null, sd: null };
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  if (values.length < 2) return { mean, sd: null };
  const v = values.reduce((a, b) => a + (b - mean) ** 2, 0) / (values.length - 1);
  return { mean, sd: Math.sqrt(v) };
}

/** Per model: clean and per-condition balanced accuracy, mean ± sample sd over seeds. */
export function aggregate(runs: RunRow[], conditions: string[]) {
  const models = [...new Set(runs.map((r) => r.model))];
  return models.map((model) => {
    const rs = runs.filter((r) => r.model === model);
    const per: Record<string, { mean: number | null; sd: number | null; n: number }> = {};
    for (const c of conditions) {
      const vals = rs
        .map((r) => r.conditions[c]?.balanced_accuracy)
        .filter((v): v is number => typeof v === "number");
      per[c] = { ...stats(vals), n: vals.length };
    }
    return { model, clean: stats(rs.map((r) => r.clean_balanced_accuracy)), per, seeds: rs.length };
  });
}

const COLORS = ["#1d4e89", "#b35c00", "#0b5323", "#6b2d8f", "#7a7a7a", "#a11b1b"];

export function RobustnessPage() {
  const suites = useSuites();
  const benchmarks = useBenchmarks();
  const bench = useBenchmark(benchmarks.data?.[0]?.name);
  const rob = bench.data?.robustness_val as
    | {
        label: string;
        scope: string;
        suite: string;
        confounds: Record<string, string>;
        runs: RunRow[];
      }
    | null
    | undefined;
  const suite = (suites.data ?? []).find((s) => s.name === rob?.suite) ?? suites.data?.[0];
  const conditions = useMemo(
    () => (suite?.conditions ?? []).map((c) => `${c.corruption}/${c.severity}`),
    [suite],
  );
  const table = useMemo(() => (rob ? aggregate(rob.runs, conditions) : []), [rob, conditions]);
  const corruptions = [...new Set((suite?.conditions ?? []).map((c) => c.corruption))];

  return (
    <>
      <PageHeader title="Robustness">
        Pre-registered synthetic corruptions at fixed severities, applied identically to every
        model.
      </PageHeader>
      <Section title="Suite" id="suite">
        <QueryState
          query={suites}
          what="robustness suites"
          isEmpty={(s) => s.length === 0}
          empty="No robustness suite found."
        >
          {(list) => (
            <>
              {list.map((s) => (
                <div key={s.name}>
                  <p>
                    <strong>{s.name}</strong> · sha256{" "}
                    <span className="mono">{s.sha256.slice(0, 16)}…</span> · base seed {s.base_seed}{" "}
                    · {s.conditions.length} conditions
                  </p>
                  <table>
                    <caption>Corruptions and their fixed severities</caption>
                    <thead>
                      <tr>
                        <th scope="col">corruption</th>
                        <th scope="col">parameter</th>
                        <th scope="col">severities</th>
                      </tr>
                    </thead>
                    <tbody>
                      {[...new Set(s.conditions.map((c) => c.corruption))].map((name) => (
                        <tr key={name}>
                          <td>{name}</td>
                          <td>{s.conditions.find((c) => c.corruption === name)?.parameter}</td>
                          <td>
                            {s.conditions
                              .filter((c) => c.corruption === name)
                              .map((c) => c.severity)
                              .join(", ")}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ))}
              <p className="limitations">
                <strong>Scope.</strong>{" "}
                {rob?.scope ??
                  "Synthetic corruptions at fixed severities on one dataset: results describe sensitivity to these corruptions only, not general robustness."}
              </p>
            </>
          )}
        </QueryState>
      </Section>
      <Section title="Results per model and condition" id="results">
        {bench.isPending ? (
          <p className="muted">Loading benchmark results…</p>
        ) : !rob ? (
          <p className="muted">No robustness results recorded for the benchmark yet.</p>
        ) : (
          <>
            <p>
              <SplitLabel split="val" /> balanced accuracy, mean ± sample sd over{" "}
              {table[0]?.seeds ?? 0} seeds per model (benchmark {bench.data?.name}, suite{" "}
              {rob.suite}).
            </p>
            <div className="table-scroll">
              <table>
                <caption>{rob.label}</caption>
                <thead>
                  <tr>
                    <th scope="col">model</th>
                    <th scope="col">clean</th>
                    {conditions.map((c) => (
                      <th key={c} scope="col">
                        {c}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {table.map((r) => (
                    <tr key={r.model}>
                      <td>{r.model}</td>
                      <td className="mono">{meanSd(r.clean.mean, r.clean.sd)}</td>
                      {conditions.map((c) => (
                        <td key={c} className="mono">
                          {r.per[c]?.n ? meanSd(r.per[c]?.mean, r.per[c]?.sd) : "missing"}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="grid-2">
              {corruptions.map((name) => {
                const sevs = (suite?.conditions ?? [])
                  .filter((c) => c.corruption === name)
                  .map((c) => c.severity);
                const data = [
                  {
                    severity: "clean",
                    ...Object.fromEntries(table.map((r) => [r.model, r.clean.mean])),
                  },
                  ...sevs.map((sv) => ({
                    severity: sv,
                    ...Object.fromEntries(
                      table.map((r) => [r.model, r.per[`${name}/${sv}`]?.mean ?? null]),
                    ),
                  })),
                ];
                return (
                  <figure key={name} style={{ margin: 0 }}>
                    <figcaption>
                      <strong>{name}</strong>: VALIDATION balanced accuracy (3-seed mean) by
                      severity
                    </figcaption>
                    <div style={{ height: 240 }}>
                      <ResponsiveContainer>
                        <LineChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="severity" fontSize={11} />
                          <YAxis domain={[0, 1]} fontSize={11} />
                          <Tooltip />
                          <Legend wrapperStyle={{ fontSize: 11 }} />
                          {table.map((r, i) => (
                            <Line
                              key={r.model}
                              dataKey={r.model}
                              stroke={COLORS[i % COLORS.length]}
                              dot
                              isAnimationActive={false}
                            />
                          ))}
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </figure>
                );
              })}
            </div>
            <h3>Confounds</h3>
            <ul>
              {Object.entries(rob.confounds).map(([k, v]) => (
                <li key={k}>
                  <strong>{k}</strong>: {v}
                </li>
              ))}
            </ul>
          </>
        )}
      </Section>
      <StartRobustness suiteNames={(suites.data ?? []).map((s) => s.name)} />
    </>
  );
}

function StartRobustness({ suiteNames }: { suiteNames: string[] }) {
  const exps = useExperiments();
  const [exp, setExp] = useState("");
  const [suite, setSuite] = useState("");
  const chosenSuite = suite || suiteNames[0] || "";
  const start = useStartRobustness(exp);
  const recorded = useRobustness(exp || undefined);
  const [jobId, setJobId] = useState<string | null>(null);
  function submit(e: FormEvent) {
    e.preventDefault();
    if (!exp || !chosenSuite) return;
    start.mutate({ suite: chosenSuite, split: "val" }, { onSuccess: (r) => setJobId(r.job.id) });
  }
  return (
    <Section title="Run robustness for an experiment" id="start-robustness">
      <form onSubmit={submit}>
        <div className="field-row">
          <div className="field">
            <label htmlFor="rb-exp">Experiment</label>
            <select id="rb-exp" value={exp} onChange={(e) => setExp(e.target.value)}>
              <option value="">— choose —</option>
              {(exps.data ?? [])
                .filter((e) => e.status === "completed")
                .map((e) => (
                  <option key={e.id} value={e.id}>
                    {e.id} · {e.model} · seed {e.seed}
                  </option>
                ))}
            </select>
          </div>
          <div className="field">
            <label htmlFor="rb-suite">Suite</label>
            <select id="rb-suite" value={chosenSuite} onChange={(e) => setSuite(e.target.value)}>
              {suiteNames.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </div>
        </div>
        <p className="hint">
          Split: VALIDATION. The corrupted images must be materialized first (`visionbench
          robustness materialize`); conditions already evaluated are skipped.
        </p>
        <button type="submit" disabled={!exp || !chosenSuite || start.isPending}>
          Start VAL robustness (job)
        </button>
      </form>
      <NotAvailable
        label="TEST robustness"
        reason="the test split is evaluated once, at the final release, through the pre-registered procedure"
      />
      {start.isError ? <ErrorBox error={start.error} title="Robustness refused" /> : null}
      {jobId ? <JobPanel jobId={jobId} /> : null}
      {exp ? (
        <>
          <h3>Recorded robustness evaluations of this experiment</h3>
          <QueryState
            query={recorded}
            what="robustness evaluations"
            isEmpty={(r) => r.length === 0}
            empty="None yet."
          >
            {(rows) => (
              <table>
                <caption>
                  Robustness evaluations of <ExperimentLink id={exp} />
                </caption>
                <thead>
                  <tr>
                    <th scope="col">id</th>
                    <th scope="col">split</th>
                    <th scope="col">condition</th>
                    <th scope="col">clean evaluation (pinned)</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r) => (
                    <tr key={r.id}>
                      <td className="mono">{r.id}</td>
                      <td>
                        <SplitLabel split={r.split} />
                      </td>
                      <td>
                        {r.corruption}/{r.severity}
                      </td>
                      <td className="mono">{r.clean_evaluation_id}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </QueryState>
        </>
      ) : null}
    </Section>
  );
}
