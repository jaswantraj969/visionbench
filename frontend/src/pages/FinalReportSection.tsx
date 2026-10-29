import { Section, SplitLabel } from "../components/ui";

/** Mean ± sd over seeds, as stored in FINAL_REPORT.json. */
interface MeanSd {
  mean: number;
  sd: number | null;
  per_seed: number[];
}
interface Ci {
  point: number;
  low: number;
  high: number;
}
interface ModelEntry {
  balanced_accuracy: MeanSd;
  macro_f1: MeanSd;
  val: { balanced_accuracy: MeanSd; macro_f1: MeanSd };
}
interface Comparison {
  a: string;
  b: string;
  per_seed: Record<string, { balanced_accuracy: Ci; macro_f1: Ci }>;
  verdict: { balanced_accuracy: string; macro_f1: string };
}
interface RobustnessModel {
  clean: number[];
  mean_corrupted: number[];
  drop: number[];
}
interface FinalReport {
  label?: string;
  final_analysis_plan_sha256?: string;
  models?: Record<string, ModelEntry>;
  ranking?: { model: string; rank: number; rank_group: number }[];
  comparisons?: Comparison[];
  robustness?: { suite: string; summary: Record<string, RobustnessModel>; missing: unknown[] };
  constraints?: Record<string, string>;
}

const f4 = (x: number | null | undefined) => (x === null || x === undefined ? "—" : x.toFixed(4));
const ms = (d: MeanSd | undefined) =>
  d ? (d.sd === null ? f4(d.mean) : `${f4(d.mean)} ± ${f4(d.sd)}`) : "—";
const ci = (c: Ci | undefined) =>
  c ? `${c.point >= 0 ? "+" : ""}${f4(c.point)} [${f4(c.low)}, ${f4(c.high)}]` : "—";
const msList = (v: number[] | undefined) => {
  if (!v || v.length === 0) return "—";
  const mean = v.reduce((s, x) => s + x, 0) / v.length;
  const sd =
    v.length > 1 ? Math.sqrt(v.reduce((s, x) => s + (x - mean) ** 2, 0) / (v.length - 1)) : null;
  return sd === null ? f4(mean) : `${f4(mean)} ± ${f4(sd)}`;
};

/** The TEST section of the Benchmark page: the single pre-registered TEST evaluation, rendered
 * from results/FINAL_REPORT.json. Before it exists the page says so instead of showing anything
 * that looks like a TEST result. */
export function FinalReportSection({ report }: { report: Record<string, unknown> | null }) {
  if (!report) {
    return (
      <Section title="TEST results" id="test-results">
        <p>
          <SplitLabel split="test" /> Not evaluated yet. The test split is evaluated exactly once,
          for all runs, at the final release, following the committed final analysis plan.
        </p>
      </Section>
    );
  }
  const r = report as FinalReport;
  const models = r.models ?? {};
  const names = (r.ranking ?? []).map((x) => x.model);
  return (
    <Section title="TEST results" id="test-results">
      <p>
        <SplitLabel split="test" /> One clean TEST evaluation per run, following the committed final
        analysis plan
        {r.final_analysis_plan_sha256 ? (
          <>
            {" "}
            (sha256 <code>{r.final_analysis_plan_sha256.slice(0, 12)}…</code>)
          </>
        ) : null}
        . Mean ± sample sd over 3 seeds; the full tables and every limitation are in the final
        report document.
      </p>
      <h3>Ranking by 3-seed mean balanced accuracy (TEST)</h3>
      <div className="table-scroll">
        <table aria-label="TEST ranking">
          <thead>
            <tr>
              <th scope="col">rank</th>
              <th scope="col">model</th>
              <th scope="col" className="num">
                balanced accuracy (TEST)
              </th>
              <th scope="col" className="num">
                macro F1 (TEST)
              </th>
              <th scope="col" className="num">
                balanced accuracy (VALIDATION, selection-biased)
              </th>
              <th scope="col">rank group</th>
            </tr>
          </thead>
          <tbody>
            {(r.ranking ?? []).map((x) => (
              <tr key={x.model}>
                <td>{x.rank}</td>
                <td>{x.model}</td>
                <td className="num">{ms(models[x.model]?.balanced_accuracy)}</td>
                <td className="num">{ms(models[x.model]?.macro_f1)}</td>
                <td className="num">{ms(models[x.model]?.val.balanced_accuracy)}</td>
                <td>{x.rank_group}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="muted">
        Rank groups are descriptive: neighbours are separated only if all three seed-matched paired
        CIs exclude zero in the same direction.
      </p>
      <h3>Pre-registered comparisons (seed-matched, A − B, TEST)</h3>
      <div className="table-scroll">
        <table aria-label="Pre-registered comparisons">
          <thead>
            <tr>
              <th scope="col">A − B</th>
              <th scope="col">Δ balanced accuracy, seed 0 / 1 / 2 [95 % CI]</th>
              <th scope="col">decision (balanced accuracy)</th>
              <th scope="col">decision (macro F1)</th>
            </tr>
          </thead>
          <tbody>
            {(r.comparisons ?? []).map((c) => (
              <tr key={`${c.a}-${c.b}`}>
                <td>
                  {c.a} − {c.b}
                </td>
                <td>
                  {["0", "1", "2"].map((s) => (
                    <div key={s}>{ci(c.per_seed[s]?.balanced_accuracy)}</div>
                  ))}
                </td>
                <td>{c.verdict.balanced_accuracy}</td>
                <td>{c.verdict.macro_f1}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {r.robustness ? (
        <>
          <h3>Robustness on TEST ({r.robustness.suite}, mean over 12 conditions)</h3>
          <div className="table-scroll">
            <table aria-label="TEST robustness summary">
              <thead>
                <tr>
                  <th scope="col">model</th>
                  <th scope="col" className="num">
                    clean (TEST)
                  </th>
                  <th scope="col" className="num">
                    mean corrupted (TEST)
                  </th>
                  <th scope="col" className="num">
                    drop
                  </th>
                </tr>
              </thead>
              <tbody>
                {names.map((m) => {
                  const s = r.robustness?.summary[m];
                  return (
                    <tr key={m}>
                      <td>{m}</td>
                      <td className="num">{msList(s?.clean)}</td>
                      <td className="num">{msList(s?.mean_corrupted)}</td>
                      <td className="num">{msList(s?.drop)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="muted">
            Sensitivity to 4 synthetic corruptions at 3 fixed severities only; no model is called
            robust.
          </p>
        </>
      ) : null}
      {r.constraints ? (
        <>
          <h3>Interpretation constraints (fixed in the plan)</h3>
          <ul>
            {Object.entries(r.constraints).map(([code, text]) => (
              <li key={code}>
                <strong>{code}</strong>: {text}
              </li>
            ))}
          </ul>
        </>
      ) : null}
    </Section>
  );
}
