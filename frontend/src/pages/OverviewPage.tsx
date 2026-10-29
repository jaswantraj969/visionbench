import { Link } from "react-router-dom";

import { useHealth, useOverview, useReady } from "../api/hooks";
import {
  ErrorBox,
  KeyValues,
  Loading,
  PageHeader,
  QueryState,
  Section,
  StatusBadge,
} from "../components/ui";

export function OverviewPage() {
  const health = useHealth();
  const ready = useReady();
  const overview = useOverview();
  return (
    <>
      <PageHeader title="Overview">
        Local computer-vision benchmarking: datasets, training jobs, evaluations, explanations and
        robustness, all through the VisionBench API.
      </PageHeader>
      <div className="grid-2">
        <Section title="API status" id="api-status">
          {health.isPending ? (
            <Loading what="health" />
          ) : health.isError ? (
            <ErrorBox error={health.error} title="API not reachable" />
          ) : (
            <KeyValues
              rows={[
                [
                  "health",
                  <StatusBadge
                    key="h"
                    status={health.data.status === "ok" ? "completed" : "failed"}
                  />,
                ],
                ["version", health.data.version],
                [
                  "ready",
                  ready.isPending
                    ? "checking…"
                    : ready.isError
                      ? `not ready: ${ready.error.message}`
                      : `ready (${Object.entries(ready.data.checks)
                          .map(([k, v]) => `${k}: ${v}`)
                          .join(", ")})`,
                ],
              ]}
            />
          )}
        </Section>
        <Section title="Counts" id="counts">
          <QueryState query={overview} what="overview">
            {(o) => (
              <KeyValues
                rows={[
                  [
                    "datasets",
                    <Link key="d" to="/datasets">
                      {o.datasets}
                    </Link>,
                  ],
                  [
                    "experiments",
                    <span key="e">
                      <Link to="/experiments">{o.experiments}</Link>{" "}
                      <span className="muted">
                        (
                        {Object.entries(o.experiments_by_status)
                          .map(([k, v]) => `${k} ${v}`)
                          .join(", ") || "none"}
                        )
                      </span>
                    </span>,
                  ],
                  [
                    "jobs",
                    <span key="j">
                      <Link to="/jobs">
                        {Object.values(o.jobs_by_status).reduce((a, b) => a + b, 0)}
                      </Link>{" "}
                      <span className="muted">
                        (
                        {Object.entries(o.jobs_by_status)
                          .map(([k, v]) => `${k} ${v}`)
                          .join(", ") || "none"}
                        )
                      </span>
                    </span>,
                  ],
                  ["inference device", o.inference_device],
                ]}
              />
            )}
          </QueryState>
        </Section>
      </div>
      <Section title="Test split lock" id="test-lock">
        <QueryState query={overview} what="overview">
          {(o) =>
            o.test_images_unlocked ? (
              <p>
                <strong>Unlocked.</strong> TEST images may be viewed and explained. This is only set
                after the single final test evaluation.
              </p>
            ) : (
              <p>
                <strong>Locked.</strong> TEST images cannot be listed, predicted or explained, and
                robustness on TEST is refused, until the single pre-registered final evaluation.
              </p>
            )
          }
        </QueryState>
      </Section>
      <Section title="Benchmarks" id="benchmarks">
        <QueryState
          query={overview}
          what="benchmarks"
          isEmpty={(o) => o.benchmarks.length === 0}
          empty="No frozen benchmark folders found."
        >
          {(o) => (
            <table>
              <caption>Frozen benchmark protocols and their hash status</caption>
              <thead>
                <tr>
                  <th scope="col">benchmark</th>
                  <th scope="col">protocol</th>
                  <th scope="col">robustness protocol</th>
                  <th scope="col" className="num">
                    runs
                  </th>
                </tr>
              </thead>
              <tbody>
                {o.benchmarks.map((b) => (
                  <tr key={b.name}>
                    <td>
                      <Link to={`/benchmark?name=${b.name}`}>{b.name}</Link>
                    </td>
                    <td>{b.protocol_hashes.ok ? "frozen, hashes OK" : "HASH MISMATCH"}</td>
                    <td>
                      {!b.robustness_hashes.present
                        ? "none"
                        : b.robustness_hashes.ok
                          ? "frozen, hashes OK"
                          : "HASH MISMATCH"}
                    </td>
                    <td className="num">{b.runs}</td>
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
