import { useState } from "react";

import { useCancelJob, useJob } from "../api/hooks";
import { num } from "../lib/format";
import { ErrorBox, JobLink, KeyValues, Loading, StatusBadge } from "./ui";

const KNOWN = new Set([
  "phase",
  "epoch",
  "epochs",
  "step",
  "max_steps",
  "latest",
  "experiment_id",
  "conditions_done",
  "conditions_total",
]);

/** Rows for the progress the worker reports (training: phase/epoch/step/latest val metrics;
 * robustness: conditions done; other jobs: whatever keys they report). */
export function describeProgress(p: Record<string, unknown>): [string, string][] {
  const rows: [string, string][] = [];
  if (p.phase !== undefined) rows.push(["phase", String(p.phase)]);
  if (p.experiment_id !== undefined) rows.push(["experiment", String(p.experiment_id)]);
  if (p.epoch !== undefined) rows.push(["epoch", `${p.epoch} / ${p.epochs ?? "?"}`]);
  if (p.step !== undefined) {
    rows.push(["step", p.max_steps ? `${p.step} / ${p.max_steps}` : String(p.step)]);
  }
  const latest = (p.latest ?? {}) as Record<string, number | null>;
  if (latest.val_loss !== undefined) rows.push(["latest val loss", num(latest.val_loss)]);
  if (latest.val_balanced_accuracy !== undefined) {
    rows.push(["latest val balanced accuracy (VALIDATION)", num(latest.val_balanced_accuracy)]);
  }
  if (p.conditions_done !== undefined) {
    rows.push(["conditions", `${p.conditions_done} / ${p.conditions_total}`]);
  }
  for (const [k, v] of Object.entries(p)) {
    if (!KNOWN.has(k)) rows.push([k, typeof v === "string" ? v : JSON.stringify(v)]);
  }
  return rows;
}

/** Live job status (polled while active), progress, result, error and a confirmed cancel. */
export function JobPanel({ jobId }: { jobId: string }) {
  const job = useJob(jobId);
  const cancel = useCancelJob();
  const [confirming, setConfirming] = useState(false);
  if (job.isPending) return <Loading what="job" />;
  if (job.isError) return <ErrorBox error={job.error} title="Could not load the job" />;
  const j = job.data;
  const active = j.status === "queued" || j.status === "running";
  const progressRows = describeProgress((j.progress ?? {}) as Record<string, unknown>);
  return (
    <div className="job-panel" aria-live="polite">
      <KeyValues
        rows={[
          ["job", <JobLink key="j" id={j.id} />],
          ["kind", j.kind],
          ["status", <StatusBadge key="s" status={j.status} />],
          ["created", j.created_at],
          ["started", j.started_at ?? "not yet"],
          ["finished", j.finished_at ?? (active ? "not yet" : "—")],
          ["cancel requested", j.cancel_requested ? "yes" : "no"],
        ]}
      />
      {progressRows.length ? (
        <>
          <h3>Progress</h3>
          <KeyValues rows={progressRows} />
        </>
      ) : active ? (
        <p className="muted">Waiting for the worker to report progress…</p>
      ) : null}
      {j.error ? (
        <div className="state state-error" role="alert">
          <strong>Job failed</strong>
          <p>{j.error}</p>
        </div>
      ) : null}
      {j.result ? (
        <>
          <h3>Result</h3>
          <pre className="json">{JSON.stringify(j.result, null, 2)}</pre>
        </>
      ) : null}
      {active ? (
        confirming ? (
          <div className="confirm" role="group" aria-label="Confirm cancel">
            <span>Cancel this {j.kind} job?</span>
            <button
              type="button"
              className="danger"
              onClick={() => cancel.mutate(j.id, { onSettled: () => setConfirming(false) })}
            >
              Yes, cancel the job
            </button>
            <button type="button" onClick={() => setConfirming(false)}>
              Keep running
            </button>
          </div>
        ) : (
          <button type="button" onClick={() => setConfirming(true)}>
            Cancel job…
          </button>
        )
      ) : null}
      {cancel.isError ? <ErrorBox error={cancel.error} title="Cancel failed" /> : null}
    </div>
  );
}
