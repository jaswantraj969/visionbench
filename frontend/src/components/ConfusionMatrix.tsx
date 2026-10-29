import { useId, useState } from "react";

import { num } from "../lib/format";

/** Accessible confusion-matrix heatmap: a real table (row = true class, column = predicted),
 * every cell shows its number; colour intensity only repeats what the number says. */
export function ConfusionMatrix({
  labels,
  counts,
  rowNormalized,
}: {
  labels: string[];
  counts: number[][];
  rowNormalized: (number | null)[][];
}) {
  const [mode, setMode] = useState<"counts" | "normalized">("counts");
  const id = useId();
  const max = Math.max(1, ...counts.flat());
  return (
    <div className="cm">
      <fieldset className="inline-options">
        <legend>Show</legend>
        <label>
          <input
            type="radio"
            name={`${id}-mode`}
            checked={mode === "counts"}
            onChange={() => setMode("counts")}
          />{" "}
          counts
        </label>
        <label>
          <input
            type="radio"
            name={`${id}-mode`}
            checked={mode === "normalized"}
            onChange={() => setMode("normalized")}
          />{" "}
          row-normalized (recall per true class)
        </label>
      </fieldset>
      <div className="table-scroll">
        <table className="cm-table">
          <caption>
            Confusion matrix ({mode === "counts" ? "counts" : "row-normalized"}): rows are true
            classes, columns are predicted classes.
          </caption>
          <thead>
            <tr>
              <th scope="col">true \ predicted</th>
              {labels.map((l) => (
                <th key={l} scope="col" className="cm-col">
                  {l}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {labels.map((row, i) => (
              <tr key={row}>
                <th scope="row">{row}</th>
                {labels.map((col, j) => {
                  const c = counts[i]?.[j] ?? 0;
                  const r = rowNormalized[i]?.[j] ?? null;
                  const intensity = mode === "counts" ? c / max : (r ?? 0);
                  const text = mode === "counts" ? String(c) : r === null ? "undefined" : num(r, 3);
                  return (
                    <td
                      key={col}
                      className={intensity > 0.55 ? "cm-cell dark" : "cm-cell"}
                      style={{ backgroundColor: `rgba(29, 78, 137, ${0.08 + 0.85 * intensity})` }}
                      aria-label={`true ${row}, predicted ${col}: ${text}`}
                    >
                      {text}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
