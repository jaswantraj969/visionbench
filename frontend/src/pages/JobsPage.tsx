import { useState } from "react";
import { useParams } from "react-router-dom";

import { useJobs } from "../api/hooks";
import { JobPanel } from "../components/JobPanel";
import { JobLink, PageHeader, QueryState, Section, StatusBadge } from "../components/ui";

const STATUSES = ["all", "queued", "running", "completed", "failed", "cancelled"];

export function JobsPage() {
  const jobs = useJobs();
  const [status, setStatus] = useState("all");
  return (
    <>
      <PageHeader title="Jobs">
        Every import, split, training, evaluation, latency and robustness run is a job executed by
        the single worker, one at a time. The list refreshes while jobs are active.
      </PageHeader>
      <Section title="Jobs" id="jobs">
        <div className="field">
          <label htmlFor="job-status">Status</label>
          <select id="job-status" value={status} onChange={(e) => setStatus(e.target.value)}>
            {STATUSES.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </div>
        <QueryState
          query={jobs}
          what="jobs"
          isEmpty={(list) =>
            list.filter((j) => status === "all" || j.status === status).length === 0
          }
          empty="No jobs match."
        >
          {(list) => (
            <table>
              <caption>Jobs, newest first</caption>
              <thead>
                <tr>
                  <th scope="col">job</th>
                  <th scope="col">kind</th>
                  <th scope="col">status</th>
                  <th scope="col">created</th>
                  <th scope="col">finished</th>
                  <th scope="col">error</th>
                </tr>
              </thead>
              <tbody>
                {list
                  .filter((j) => status === "all" || j.status === status)
                  .map((j) => (
                    <tr key={j.id}>
                      <td>
                        <JobLink id={j.id} />
                      </td>
                      <td>{j.kind}</td>
                      <td>
                        <StatusBadge status={j.status} />
                      </td>
                      <td>{j.created_at}</td>
                      <td>{j.finished_at ?? "—"}</td>
                      <td>{j.error ? j.error.slice(0, 80) : ""}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          )}
        </QueryState>
      </Section>
    </>
  );
}

export function JobDetailPage() {
  const { jobId = "" } = useParams();
  return (
    <>
      <PageHeader title="Job" />
      <Section title="Status and progress" id="job-detail">
        <JobPanel jobId={jobId} />
      </Section>
    </>
  );
}
