import { useMemo, useState } from "react";

import { useExperiments } from "../api/hooks";
import type { ExperimentSummary } from "../api/types";
import {
  ExperimentLink,
  PageHeader,
  QueryState,
  Section,
  StatusBadge,
  Tag,
} from "../components/ui";
import { num, seconds } from "../lib/format";

export function filterExperiments(
  list: ExperimentSummary[],
  f: { model: string; status: string; seed: string; tag: string },
): ExperimentSummary[] {
  return list.filter(
    (e) =>
      (f.model === "all" || e.model === f.model) &&
      (f.status === "all" || e.status === f.status) &&
      (f.seed === "all" || String(e.seed) === f.seed) &&
      (f.tag === "all" ||
        (f.tag === "untagged"
          ? e.tags.length === 0
          : e.tags.some((t) => t.startsWith(`${f.tag}:`)))),
  );
}

export function ExperimentsPage() {
  const exps = useExperiments();
  const [filters, setFilters] = useState({ model: "all", status: "all", seed: "all", tag: "all" });
  const options = useMemo(() => {
    const list = exps.data ?? [];
    return {
      model: ["all", ...new Set(list.map((e) => e.model))],
      status: ["all", ...new Set(list.map((e) => e.status))],
      seed: ["all", ...[...new Set(list.map((e) => e.seed))].sort((a, b) => a - b).map(String)],
      tag: ["all", "benchmark", "pilot", "repro-check", "untagged"],
    };
  }, [exps.data]);
  return (
    <>
      <PageHeader title="Experiments">
        Training runs. Benchmark runs, pilot runs and the repro-check run are tagged from the
        benchmark records; pilot and repro-check runs are excluded from all benchmark results.
      </PageHeader>
      <Section title="Experiments" id="experiments">
        <div className="field-row">
          {(Object.keys(filters) as (keyof typeof filters)[]).map((k) => (
            <div key={k} className="field">
              <label htmlFor={`exp-${k}`}>{k}</label>
              <select
                id={`exp-${k}`}
                value={filters[k]}
                onChange={(e) => setFilters((f) => ({ ...f, [k]: e.target.value }))}
              >
                {options[k].map((o) => (
                  <option key={o} value={o}>
                    {o}
                  </option>
                ))}
              </select>
            </div>
          ))}
        </div>
        <QueryState
          query={exps}
          what="experiments"
          isEmpty={(list) => filterExperiments(list, filters).length === 0}
          empty="No experiments match."
        >
          {(list) => (
            <div className="table-scroll">
              <table>
                <caption>
                  Experiments. Best value = the monitored validation metric at the best epoch
                  (VALIDATION).
                </caption>
                <thead>
                  <tr>
                    <th scope="col">experiment</th>
                    <th scope="col">model</th>
                    <th scope="col">weights</th>
                    <th scope="col" className="num">
                      seed
                    </th>
                    <th scope="col">status</th>
                    <th scope="col" className="num">
                      best epoch
                    </th>
                    <th scope="col" className="num">
                      best value
                    </th>
                    <th scope="col">stopped</th>
                    <th scope="col" className="num">
                      train time
                    </th>
                    <th scope="col">tags</th>
                  </tr>
                </thead>
                <tbody>
                  {filterExperiments(list, filters).map((e) => (
                    <tr key={e.id}>
                      <td>
                        <ExperimentLink id={e.id} />
                      </td>
                      <td>{e.model}</td>
                      <td>{e.model === "majority_baseline" ? "n/a" : (e.weights_source ?? "—")}</td>
                      <td className="num">{e.seed}</td>
                      <td>
                        <StatusBadge status={e.status} />
                      </td>
                      <td className="num">{e.best_epoch ?? "—"}</td>
                      <td className="num">{num(e.best_metric)}</td>
                      <td>{e.stopped_reason ?? "—"}</td>
                      <td className="num">{seconds(e.train_seconds)}</td>
                      <td>
                        {e.tags.map((t) => (
                          <Tag key={t} text={t} />
                        ))}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </QueryState>
      </Section>
    </>
  );
}
