import { Link, useParams } from "react-router-dom";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { useExperiment } from "../api/hooks";
import type { EpochRow, ExperimentDetail } from "../api/types";
import {
  KeyValues,
  PageHeader,
  QueryState,
  Section,
  SplitLabel,
  StatusBadge,
  Tag,
} from "../components/ui";
import { bytes, num, seconds, shortSha } from "../lib/format";

function valueText(v: unknown): string {
  if (v === null || v === undefined) return "none";
  return typeof v === "object" ? JSON.stringify(v) : String(v);
}

export function ExperimentDetailPage() {
  const { experimentId = "" } = useParams();
  const exp = useExperiment(experimentId);
  return (
    <QueryState query={exp} what="experiment">
      {(e) => <ExperimentView e={e} />}
    </QueryState>
  );
}

function ExperimentView({ e }: { e: ExperimentDetail }) {
  const env = e.env ?? {};
  return (
    <>
      <PageHeader title={`Experiment ${e.id}`}>
        {e.model} · seed {e.seed} · <StatusBadge status={e.status} />{" "}
        {e.tags.map((t) => (
          <Tag key={t} text={t} />
        ))}
      </PageHeader>
      {e.error ? (
        <div className="state state-error" role="alert">
          <strong>Run error</strong>
          <p>{e.error}</p>
        </div>
      ) : null}
      <div className="grid-2">
        <Section title="Run" id="run">
          <KeyValues
            rows={[
              ["dataset / split", `${e.dataset_id} / ${e.split_id}`],
              [
                "weights source",
                e.model === "majority_baseline" ? "n/a" : (e.weights_source ?? "none"),
              ],
              ["weights", e.weights_enum ?? "none"],
              [
                "weights sha256",
                <span key="w" className="mono">
                  {e.weights_sha256 ?? "none"}
                </span>,
              ],
              ["best epoch (selected on VALIDATION)", String(e.best_epoch ?? "—")],
              ["best monitored value (VALIDATION)", num(e.best_metric)],
              ["stopped reason", e.stopped_reason ?? "—"],
              ["epochs trained", String(e.epochs.length)],
              ["train time", seconds(e.train_seconds)],
              ["created / finished", `${e.created_at} / ${e.finished_at ?? "—"}`],
            ]}
          />
        </Section>
        <Section title="Environment" id="env">
          <KeyValues
            rows={[
              "device",
              "device_name",
              "torch",
              "torchvision",
              "python",
              "platform",
              "visionbench",
            ].map((k): [string, string] => [k, valueText(env[k])])}
          />
        </Section>
      </div>
      <Curves epochs={e.epochs} bestEpoch={e.best_epoch} />
      <Section title="Configuration" id="config">
        <p className="hint">“n/a” marks settings this model does not use.</p>
        <KeyValues
          rows={Object.entries(e.config_display).map(([k, v]): [string, string] => [
            k,
            valueText(v),
          ])}
        />
      </Section>
      <Section title="Evaluations" id="evals">
        {e.evaluations.length === 0 ? (
          <p className="muted">
            No evaluations yet.{" "}
            <Link to={`/evaluate?experiment=${e.id}`}>Run a VAL evaluation</Link>.
          </p>
        ) : (
          <table>
            <caption>Recorded evaluations of this experiment</caption>
            <thead>
              <tr>
                <th scope="col">evaluation</th>
                <th scope="col">split</th>
                <th scope="col">checkpoint</th>
                <th scope="col">created</th>
                <th scope="col">forced</th>
              </tr>
            </thead>
            <tbody>
              {e.evaluations.map((ev) => (
                <tr key={ev.id}>
                  <td>
                    <Link to={`/evaluate?evaluation=${ev.id}`} className="mono">
                      {ev.id}
                    </Link>
                  </td>
                  <td>
                    <SplitLabel split={ev.split} />
                  </td>
                  <td>{ev.checkpoint}</td>
                  <td>{ev.created_at}</td>
                  <td>{ev.forced ? `yes: ${ev.reason}` : "no"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Section>
      <Section title="Artifacts" id="artifacts">
        <div className="table-scroll">
          <table>
            <caption>Files of this run (relative to the data root) with their sha256</caption>
            <thead>
              <tr>
                <th scope="col">kind</th>
                <th scope="col">path</th>
                <th scope="col" className="num">
                  size
                </th>
                <th scope="col">sha256</th>
              </tr>
            </thead>
            <tbody>
              {e.artifacts.map((a) => (
                <tr key={a.relpath}>
                  <td>{a.kind}</td>
                  <td className="mono">{a.relpath}</td>
                  <td className="num">{bytes(a.size_bytes)}</td>
                  <td className="mono" title={a.sha256}>
                    {shortSha(a.sha256)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>
    </>
  );
}

export function Curves({ epochs, bestEpoch }: { epochs: EpochRow[]; bestEpoch: number | null }) {
  if (epochs.length === 0) {
    return (
      <Section title="Training curves" id="curves">
        <p className="muted">No epochs recorded.</p>
      </Section>
    );
  }
  const best = bestEpoch ? (
    <ReferenceLine
      x={bestEpoch}
      stroke="#a11b1b"
      strokeDasharray="4 3"
      label={{ value: `best epoch ${bestEpoch}`, fontSize: 11, position: "insideTopRight" }}
    />
  ) : null;
  return (
    <Section title="Training curves" id="curves">
      <p>
        Validation curves are <SplitLabel split="val" />: the validation split chose the best epoch
        (dashed line).
      </p>
      <div className="grid-2">
        <div style={{ height: 260 }} aria-label="Loss per epoch chart">
          <ResponsiveContainer>
            <LineChart data={epochs} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="epoch" fontSize={11} />
              <YAxis fontSize={11} />
              <Tooltip />
              <Legend />
              <Line
                type="monotone"
                dataKey="train_loss"
                name="train loss"
                stroke="#1d4e89"
                dot={false}
                isAnimationActive={false}
              />
              <Line
                type="monotone"
                dataKey="val_loss"
                name="val loss"
                stroke="#b35c00"
                dot={false}
                isAnimationActive={false}
              />
              {best}
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div style={{ height: 260 }} aria-label="Validation metric per epoch chart">
          <ResponsiveContainer>
            <LineChart data={epochs} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="epoch" fontSize={11} />
              <YAxis domain={[0, 1]} fontSize={11} />
              <Tooltip />
              <Legend />
              <Line
                type="monotone"
                dataKey="val_balanced_accuracy"
                name="val balanced accuracy"
                stroke="#0b5323"
                dot={false}
                isAnimationActive={false}
              />
              <Line
                type="monotone"
                dataKey="val_accuracy"
                name="val accuracy"
                stroke="#6b3e00"
                dot={false}
                isAnimationActive={false}
              />
              {best}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
      <details>
        <summary>Per-epoch table</summary>
        <table>
          <caption>Per-epoch metrics (validation columns are VALIDATION)</caption>
          <thead>
            <tr>
              <th scope="col" className="num">
                epoch
              </th>
              <th scope="col" className="num">
                train loss
              </th>
              <th scope="col" className="num">
                val loss
              </th>
              <th scope="col" className="num">
                val balanced acc.
              </th>
              <th scope="col" className="num">
                seconds
              </th>
            </tr>
          </thead>
          <tbody>
            {epochs.map((r) => (
              <tr key={r.epoch}>
                <td className="num">{r.epoch}</td>
                <td className="num">{num(r.train_loss)}</td>
                <td className="num">{num(r.val_loss)}</td>
                <td className="num">{num(r.val_balanced_accuracy)}</td>
                <td className="num">{num(r.epoch_seconds, 1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </Section>
  );
}
