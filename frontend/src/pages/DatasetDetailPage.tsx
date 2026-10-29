import { useState } from "react";
import type { FormEvent } from "react";
import { useParams } from "react-router-dom";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { apiUrl } from "../api/client";
import { useAudit, useCreateSplit, useDataset, useSplit, useSplits } from "../api/hooks";
import type { DatasetDetail } from "../api/types";
import { JobPanel } from "../components/JobPanel";
import {
  Attribution,
  ErrorBox,
  KeyValues,
  PageHeader,
  QueryState,
  Section,
} from "../components/ui";
import { bytes, num } from "../lib/format";

export function DatasetDetailPage() {
  const { datasetId = "" } = useParams();
  const dataset = useDataset(datasetId);
  return (
    <QueryState query={dataset} what="dataset">
      {(d) => (
        <>
          <PageHeader title={`Dataset: ${d.name}`}>
            <span className="mono">{d.id}</span> · {d.total_images} images · {d.classes.length}{" "}
            classes · fingerprint <span className="mono">{d.fingerprint.slice(0, 16)}…</span>
          </PageHeader>
          <div className="grid-2">
            <ClassDistribution d={d} />
            <Summary d={d} />
          </div>
          <Issues d={d} />
          <Samples d={d} />
          <Splits datasetId={d.id} />
          <CreateSplit datasetId={d.id} />
        </>
      )}
    </QueryState>
  );
}

function ClassDistribution({ d }: { d: DatasetDetail }) {
  const data = Object.entries(d.class_counts).map(([cls, n]) => ({ cls, n }));
  return (
    <Section title="Class distribution" id="class-dist">
      <div style={{ width: "100%", height: 260 }} aria-hidden="true">
        <ResponsiveContainer>
          <BarChart data={data} margin={{ top: 4, right: 8, bottom: 60, left: 4 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="cls" angle={-40} textAnchor="end" interval={0} fontSize={11} />
            <YAxis fontSize={11} />
            <Tooltip />
            <Bar dataKey="n" name="images" fill="#1d4e89" />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <table>
        <caption>Images per class (imbalance ratio {num(d.report.imbalance_ratio, 2)})</caption>
        <thead>
          <tr>
            <th scope="col">class</th>
            <th scope="col" className="num">
              images
            </th>
          </tr>
        </thead>
        <tbody>
          {data.map((r) => (
            <tr key={r.cls}>
              <td>{r.cls}</td>
              <td className="num">{r.n}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Section>
  );
}

function Summary({ d }: { d: DatasetDetail }) {
  const r = d.report;
  const dup = r.duplicates;
  const lowInfo = dup.perceptual_check_skipped_low_information as Record<string, unknown>;
  return (
    <Section title="Images and duplicates" id="summary">
      <KeyValues
        rows={[
          ["width (min / median / max)", `${r.width.min} / ${r.width.median} / ${r.width.max} px`],
          [
            "height (min / median / max)",
            `${r.height.min} / ${r.height.median} / ${r.height.max} px`,
          ],
          [
            "modes",
            Object.entries(r.mode_counts)
              .map(([k, v]) => `${k} ${v}`)
              .join(", "),
          ],
          [
            "formats",
            Object.entries(r.format_counts)
              .map(([k, v]) => `${k} ${v}`)
              .join(", "),
          ],
          ["total size", bytes(r.total_bytes)],
          ["dHash threshold", String(dup.dhash_threshold)],
          ["exact duplicate files", String(dup.exact_duplicate_files)],
          [
            "multi-member duplicate groups",
            `${dup.multi_member_groups} (${dup.files_in_multi_member_groups} files)`,
          ],
          ["label-conflict groups", String(dup.label_conflict_groups.length)],
          [
            "low-information (perceptual check skipped)",
            lowInfo && typeof lowInfo === "object"
              ? Object.entries(lowInfo)
                  .map(([k, v]) => `${k}: ${typeof v === "object" ? JSON.stringify(v) : String(v)}`)
                  .join("; ")
              : "none",
          ],
        ]}
      />
    </Section>
  );
}

function Issues({ d }: { d: DatasetDetail }) {
  const rows = Object.entries(d.report.issues);
  return (
    <Section title="Issues found at import" id="issues">
      <table>
        <caption>Files that were not imported, by category (nothing is dropped silently)</caption>
        <thead>
          <tr>
            <th scope="col">category</th>
            <th scope="col" className="num">
              files
            </th>
            <th scope="col">examples</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([cat, v]) => (
            <tr key={cat}>
              <td>{cat}</td>
              <td className="num">{v.count}</td>
              <td className="mono">{(v.relpaths ?? []).slice(0, 3).join(", ") || "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Section>
  );
}

function Samples({ d }: { d: DatasetDetail }) {
  const s = d.report.samples;
  return (
    <Section title="Sample images per class" id="samples">
      <p className="hint">
        Seeded sample recorded at import (seed {s.seed}, {s.per_class} per class), not hand-picked.
      </p>
      {Object.entries(s.relpaths).map(([cls, rels]) => (
        <div key={cls}>
          <h3>{cls}</h3>
          <div className="image-grid">
            {rels.slice(0, 6).map((rel) => (
              <figure key={rel}>
                <img
                  src={apiUrl(`/api/datasets/${d.id}/images`, { relpath: rel })}
                  alt={`${cls} sample ${rel}`}
                  loading="lazy"
                />
                <figcaption className="mono" title={rel}>
                  {rel.split("/").pop()}
                </figcaption>
              </figure>
            ))}
          </div>
        </div>
      ))}
      <Attribution />
    </Section>
  );
}

function Splits({ datasetId }: { datasetId: string }) {
  const splits = useSplits(datasetId);
  const [selected, setSelected] = useState<string | null>(null);
  return (
    <Section title="Splits" id="splits">
      <QueryState
        query={splits}
        what="splits"
        isEmpty={(s) => s.length === 0}
        empty="No splits yet."
      >
        {(list) => (
          <>
            <table>
              <caption>
                Splits of this dataset (group-aware; the leakage check counts duplicates across
                splits)
              </caption>
              <thead>
                <tr>
                  <th scope="col">split</th>
                  <th scope="col">created</th>
                  <th scope="col" className="num">
                    train
                  </th>
                  <th scope="col" className="num">
                    val
                  </th>
                  <th scope="col" className="num">
                    test
                  </th>
                  <th scope="col">leakage check</th>
                  <th scope="col">details</th>
                </tr>
              </thead>
              <tbody>
                {list.map((s) => (
                  <tr key={s.split_id}>
                    <td className="mono">{s.split_id}</td>
                    <td>{s.created_at}</td>
                    <td className="num">{s.split_totals.train}</td>
                    <td className="num">{s.split_totals.val}</td>
                    <td className="num">{s.split_totals.test}</td>
                    <td>
                      {s.leakage_check.groups_in_multiple_splits === 0 &&
                      s.leakage_check.sha256_in_multiple_splits === 0
                        ? "0 groups / 0 files across splits"
                        : `${s.leakage_check.groups_in_multiple_splits} groups / ${s.leakage_check.sha256_in_multiple_splits} files across splits`}
                    </td>
                    <td>
                      <button
                        type="button"
                        className="secondary"
                        onClick={() => setSelected(s.split_id)}
                        aria-pressed={selected === s.split_id}
                      >
                        Show
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {selected ? <SplitDetailPanel datasetId={datasetId} splitId={selected} /> : null}
          </>
        )}
      </QueryState>
    </Section>
  );
}

function SplitDetailPanel({ datasetId, splitId }: { datasetId: string; splitId: string }) {
  const split = useSplit(datasetId, splitId);
  const audit = useAudit(datasetId, splitId);
  return (
    <QueryState query={split} what="split">
      {(s) => (
        <>
          <h3>
            Split <span className="mono">{s.split_id}</span> ({s.algorithm})
          </h3>
          <KeyValues
            rows={[
              [
                "config",
                <span key="c" className="mono">
                  {JSON.stringify(s.config)}
                </span>,
              ],
              [
                "manifest sha256",
                <span key="m" className="mono">
                  {s.manifest_sha256}
                </span>,
              ],
              ["entries", String(s.entries_total)],
              ["warnings", s.warnings.length ? JSON.stringify(s.warnings) : "none"],
              ["excluded groups", String(s.excluded_groups.length)],
            ]}
          />
          <table>
            <caption>Images per class and split</caption>
            <thead>
              <tr>
                <th scope="col">class</th>
                <th scope="col" className="num">
                  train
                </th>
                <th scope="col" className="num">
                  val
                </th>
                <th scope="col" className="num">
                  test
                </th>
                <th scope="col" className="num">
                  total
                </th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(s.counts).map(([cls, c]) => (
                <tr key={cls}>
                  <td>{cls}</td>
                  <td className="num">{c.train}</td>
                  <td className="num">{c.val}</td>
                  <td className="num">{c.test}</td>
                  <td className="num">{c.total}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <h3>Split audit</h3>
          {audit.isPending ? (
            <p className="muted">Loading audit…</p>
          ) : audit.isError ? (
            <p className="muted">
              No audit was run for this split (`visionbench dataset audit-split`).
            </p>
          ) : (
            <AuditSummary audit={audit.data} />
          )}
        </>
      )}
    </QueryState>
  );
}

function AuditSummary({ audit }: { audit: Record<string, unknown> }) {
  const counts = (audit.counts ?? {}) as Record<string, number>;
  const exact = (audit.cross_split_exact_sha256 ?? {}) as { count?: number };
  const pixel = (audit.cross_split_pixel_identical ?? {}) as { count?: number };
  return (
    <KeyValues
      rows={[
        ["entries", String(counts.entries ?? "—")],
        ["low-information images", String(counts.low_information ?? "—")],
        ["cross-split exact (sha256) matches", String(exact.count ?? "—")],
        ["cross-split pixel-identical matches", String(pixel.count ?? "—")],
        [
          "note",
          "pHash nearest neighbours are evidence for review, not a duplicate decision (unreliable on dark, low-texture tiles).",
        ],
      ]}
    />
  );
}

function CreateSplit({ datasetId }: { datasetId: string }) {
  const create = useCreateSplit(datasetId);
  const [seed, setSeed] = useState("0");
  const [train, setTrain] = useState("0.7");
  const [val, setVal] = useState("0.15");
  const [test, setTest] = useState("0.15");
  const [jobId, setJobId] = useState<string | null>(null);
  const fr = [train, val, test].map(Number);
  const seedN = Number(seed);
  const error =
    !Number.isInteger(seedN) || seedN < 0
      ? "Seed must be a non-negative integer."
      : fr.some((f) => !(f > 0 && f < 1))
        ? "Each fraction must be between 0 and 1."
        : Math.abs(fr.reduce((a, b) => a + b, 0) - 1) > 1e-6
          ? "Fractions must sum to 1."
          : null;
  function submit(e: FormEvent) {
    e.preventDefault();
    if (error) return;
    create.mutate({ seed: seedN, fractions: fr }, { onSuccess: (r) => setJobId(r.job.id) });
  }
  return (
    <Section title="Create a split" id="create-split">
      <form onSubmit={submit} noValidate>
        <div className="field-row">
          <div className="field">
            <label htmlFor="sp-seed">Seed</label>
            <input
              id="sp-seed"
              inputMode="numeric"
              value={seed}
              onChange={(e) => setSeed(e.target.value)}
            />
          </div>
          <div className="field">
            <label htmlFor="sp-train">Train fraction</label>
            <input id="sp-train" value={train} onChange={(e) => setTrain(e.target.value)} />
          </div>
          <div className="field">
            <label htmlFor="sp-val">Val fraction</label>
            <input id="sp-val" value={val} onChange={(e) => setVal(e.target.value)} />
          </div>
          <div className="field">
            <label htmlFor="sp-test">Test fraction</label>
            <input id="sp-test" value={test} onChange={(e) => setTest(e.target.value)} />
          </div>
        </div>
        {error ? <p className="field-error">{error}</p> : null}
        <button type="submit" disabled={Boolean(error) || create.isPending}>
          Create split (job)
        </button>
      </form>
      {create.isError ? <ErrorBox error={create.error} title="Split refused" /> : null}
      {jobId ? <JobPanel jobId={jobId} /> : null}
    </Section>
  );
}
