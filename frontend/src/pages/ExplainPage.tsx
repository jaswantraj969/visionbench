import { useMemo, useState } from "react";

import { apiUrl } from "../api/client";
import {
  useBenchmark,
  useBenchmarks,
  useExperiment,
  useExperiments,
  useGradcam,
  useOverview,
  usePredict,
  useSplitSample,
} from "../api/hooks";
import type { ImageInput } from "../api/hooks";
import type { GradcamResult, PredictResult } from "../api/types";
import {
  Attribution,
  ErrorBox,
  KeyValues,
  NotAvailable,
  PageHeader,
  Section,
  SplitLabel,
} from "../components/ui";
import { num, pct } from "../lib/format";

// Same text as visionbench.explain.gradcam.LIMITATIONS (shown before any request, too).
export const LIMITATIONS =
  "Grad-CAM shows which regions of the final convolutional feature map most increased the score " +
  "of the target class. It is coarse (feature-map resolution stated above), is not a causal " +
  "explanation, and does not show that the prediction is correct.";

export function ExplainPage() {
  const exps = useExperiments();
  const benchmarks = useBenchmarks();
  const bench = useBenchmark(benchmarks.data?.[0]?.name);
  const seed0 = useMemo(
    () =>
      (bench.data?.runs ?? [])
        .filter((r) => r.seed === 0 && r.model !== "majority_baseline")
        .map((r) => ({ id: String(r.experiment_id), model: String(r.model) })),
    [bench.data],
  );
  const [expId, setExpId] = useState("");
  const chosen = expId || seed0[0]?.id || "";
  const exp = useExperiment(chosen || undefined);
  const overview = useOverview();
  const [source, setSource] = useState<"val" | "test" | "upload">("val");
  const [seed, setSeed] = useState(0);
  const [relpath, setRelpath] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [target, setTarget] = useState("");
  // TEST images are offered only once the split's single final TEST evaluation is complete.
  const testOpen =
    !!exp.data &&
    (overview.data?.test_images_unlocked === true ||
      (overview.data?.final_test_splits ?? []).includes(exp.data.split_id));
  const imageSplit = source === "test" && testOpen ? "test" : "val";
  const sample = useSplitSample(exp.data?.dataset_id, exp.data?.split_id, imageSplit, 1, seed);
  const predict = usePredict(chosen);
  const gradcam = useGradcam(chosen);
  const classNames = (
    (exp.data?.config.resolved?.class_names as string[] | undefined) ?? []
  ).slice();
  const input: ImageInput | null =
    source !== "upload"
      ? relpath && exp.data
        ? { kind: "dataset", dataset_id: exp.data.dataset_id, relpath }
        : null
      : file
        ? { kind: "upload", file }
        : null;
  const others = (exps.data ?? []).filter(
    (e) => e.status === "completed" && !seed0.some((s) => s.id === e.id),
  );

  return (
    <>
      <PageHeader title="Explain">
        Predict one image and show its Grad-CAM. Images come from the VALIDATION split (seeded
        browser, not hand-picked), from the TEST split once its single final evaluation is complete,
        or from an upload.
      </PageHeader>
      <p className="limitations">
        <strong>Limitations.</strong> {LIMITATIONS}
      </p>
      <div className="grid-2">
        <Section title="Model" id="model">
          <div className="field">
            <label htmlFor="xp-exp">Experiment</label>
            <select
              id="xp-exp"
              value={chosen}
              onChange={(e) => {
                setExpId(e.target.value);
                setRelpath("");
                setTarget("");
              }}
            >
              {seed0.length ? (
                <optgroup label="Benchmark, seed 0 of each protocol model">
                  {seed0.map((s) => (
                    <option key={s.id} value={s.id}>
                      {s.model} · {s.id}
                    </option>
                  ))}
                </optgroup>
              ) : null}
              <optgroup label="Other completed experiments">
                {others.map((e) => (
                  <option key={e.id} value={e.id}>
                    {e.model} · seed {e.seed} · {e.id}
                  </option>
                ))}
              </optgroup>
            </select>
          </div>
          <div className="field">
            <label htmlFor="xp-target">Grad-CAM target class</label>
            <select id="xp-target" value={target} onChange={(e) => setTarget(e.target.value)}>
              <option value="">predicted class (default)</option>
              {classNames.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </div>
        </Section>
        <Section title="Image" id="image">
          <fieldset className="inline-options">
            <legend>Source</legend>
            <label>
              <input
                type="radio"
                name="xp-src"
                checked={source === "val"}
                onChange={() => {
                  setSource("val");
                  setRelpath("");
                }}
              />{" "}
              validation image
            </label>
            {testOpen ? (
              <label>
                <input
                  type="radio"
                  name="xp-src"
                  checked={source === "test"}
                  onChange={() => {
                    setSource("test");
                    setRelpath("");
                  }}
                />{" "}
                TEST image (final evaluation done)
              </label>
            ) : null}
            <label>
              <input
                type="radio"
                name="xp-src"
                checked={source === "upload"}
                onChange={() => setSource("upload")}
              />{" "}
              upload
            </label>
          </fieldset>
          {source === "upload" ? (
            <div className="field">
              <label htmlFor="xp-file">Image file (JPEG, PNG, BMP, WEBP; at most 10 MB)</label>
              <input
                id="xp-file"
                type="file"
                accept="image/jpeg,image/png,image/bmp,image/webp"
                onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              />
            </div>
          ) : (
            <>
              <div className="field">
                <label htmlFor="xp-seed">Browser seed (one image per class)</label>
                <input
                  id="xp-seed"
                  type="number"
                  min={0}
                  value={seed}
                  onChange={(e) => {
                    setSeed(Math.max(0, Number(e.target.value) || 0));
                    setRelpath("");
                  }}
                />
              </div>
              {sample.isError ? (
                <ErrorBox error={sample.error} title="Could not list images" />
              ) : null}
              <div
                className="image-grid"
                role="group"
                aria-label={imageSplit === "test" ? "TEST images" : "Validation images"}
              >
                {(sample.data?.images ?? []).map((img) => (
                  <button
                    key={img.relpath}
                    type="button"
                    className="image-pick"
                    aria-pressed={relpath === img.relpath}
                    onClick={() => setRelpath(img.relpath)}
                  >
                    <figure>
                      <img
                        src={apiUrl(`/api/datasets/${exp.data?.dataset_id}/images`, {
                          relpath: img.relpath,
                        })}
                        alt={`${img.label}: ${img.relpath}`}
                        loading="lazy"
                      />
                      <figcaption>{img.label}</figcaption>
                    </figure>
                  </button>
                ))}
              </div>
              {testOpen ? null : (
                <NotAvailable
                  label="TEST images"
                  reason="the test split is locked until the single final test evaluation"
                />
              )}
            </>
          )}
          <div className="actions">
            <button
              type="button"
              disabled={!input || !chosen || predict.isPending}
              onClick={() => input && predict.mutate(input)}
            >
              Predict
            </button>
            <button
              type="button"
              disabled={!input || !chosen || gradcam.isPending}
              onClick={() => input && gradcam.mutate({ input, target: target || undefined })}
            >
              Grad-CAM
            </button>
          </div>
        </Section>
      </div>
      {predict.isError ? <ErrorBox error={predict.error} title="Prediction failed" /> : null}
      {predict.data ? <Prediction r={predict.data} /> : null}
      {gradcam.isError ? <ErrorBox error={gradcam.error} title="Grad-CAM failed" /> : null}
      {gradcam.data ? <Gradcam r={gradcam.data} /> : null}
      {(source === "val" && relpath) || predict.data || gradcam.data ? <Attribution /> : null}
    </>
  );
}

function Messages({ warnings, notes }: { warnings: string[]; notes: string[] }) {
  return (
    <>
      {warnings.map((w) => (
        <p key={w} className="warning" role="alert">
          Warning: {w}
        </p>
      ))}
      {notes.length ? (
        <ul className="muted">
          {notes.map((n) => (
            <li key={n}>{n}</li>
          ))}
        </ul>
      ) : null}
    </>
  );
}

function splitOf(r: { input: Record<string, unknown> }) {
  const s = r.input.split;
  return typeof s === "string" ? (
    <SplitLabel split={s} />
  ) : (
    <span className="muted">upload (no split)</span>
  );
}

export function Prediction({ r }: { r: PredictResult }) {
  return (
    <Section title="Prediction" id="prediction">
      <p>
        Predicted <strong>{r.predicted_class}</strong> for {splitOf(r)} image (
        {r.input.original_size.join("x")} px) by {r.model}{" "}
        <span className="mono">{r.experiment_id}</span>. A probability is the model's score, not a
        guarantee of correctness.
      </p>
      <Messages warnings={r.warnings} notes={r.notes} />
      <table>
        <caption>All class probabilities, highest first</caption>
        <thead>
          <tr>
            <th scope="col">class</th>
            <th scope="col" className="num">
              probability
            </th>
          </tr>
        </thead>
        <tbody>
          {r.probabilities.map((p) => (
            <tr key={p.class}>
              <td>{p.class}</td>
              <td className="num">{num(p.probability)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="hint">
        forward {r.timing_ms.forward === null ? "n/a" : `${num(r.timing_ms.forward ?? null, 1)} ms`}{" "}
        · total {num(r.timing_ms.total ?? null, 1)} ms
      </p>
    </Section>
  );
}

export function Gradcam({ r }: { r: GradcamResult }) {
  return (
    <Section title="Grad-CAM" id="gradcam">
      <Messages warnings={r.warnings} notes={r.notes} />
      <div className="grid-2">
        <img
          className="overlay-img"
          src={r.overlay.png}
          alt={`Grad-CAM overlay for target class ${r.target_class}; regions not seen by the model are hatched`}
        />
        <KeyValues
          rows={[
            ["image", <span key="s">{splitOf(r)}</span>],
            ["target class", `${r.target_class} (p = ${num(r.target_probability)})`],
            ["predicted class", `${r.predicted_class} (p = ${num(r.predicted_probability)})`],
            ["CAM resolution", `${r.cam.resolution} at ${r.cam.layer}`],
            [
              "CAM",
              r.cam.all_zero
                ? "all zeros: no positive evidence (not rescaled)"
                : "normalised by its maximum",
            ],
            ["not seen by the model", pct(r.geometry.unseen_fraction)],
            ["Grad-CAM version", r.gradcam_version],
          ]}
        />
      </div>
      <p className="limitations">{r.limitations}</p>
    </Section>
  );
}
