import { useState } from "react";
import type { FormEvent, ReactNode } from "react";

import { useDatasets, useSplits, useStartTraining } from "../api/hooks";
import { JobPanel } from "../components/JobPanel";
import { ErrorBox, PageHeader, Section } from "../components/ui";

const MODELS = ["small_cnn", "resnet18", "efficientnet_b0", "majority_baseline"] as const;

export interface TrainForm {
  dataset_id: string;
  split_id: string;
  model: string;
  weights: string;
  finetune: string;
  input_size: string;
  epochs: string;
  max_steps: string;
  batch_size: string;
  optimizer: string;
  lr: string;
  weight_decay: string;
  momentum: string;
  scheduler: string;
  monitor: string;
  patience: string;
  min_delta: string;
  early_stopping: boolean;
  class_weighted_loss: boolean;
  seed: string;
  device: string;
  amp: boolean;
  num_workers: string;
  horizontal_flip: boolean;
  random_resized_crop: boolean;
  allow_weight_fallback: boolean;
}

export const DEFAULT_FORM: TrainForm = {
  dataset_id: "",
  split_id: "",
  model: "small_cnn",
  weights: "scratch",
  finetune: "full",
  input_size: "",
  epochs: "10",
  max_steps: "",
  batch_size: "32",
  optimizer: "adamw",
  lr: "0.001",
  weight_decay: "0.0001",
  momentum: "0.9",
  scheduler: "none",
  monitor: "val_loss",
  patience: "5",
  min_delta: "0",
  early_stopping: true,
  class_weighted_loss: false,
  seed: "0",
  device: "auto",
  amp: false,
  num_workers: "0",
  horizontal_flip: true,
  random_resized_crop: true,
  allow_weight_fallback: false,
};

const isInt = (s: string) => /^\d+$/.test(s.trim());
const isNum = (s: string) => s.trim() !== "" && Number.isFinite(Number(s));

/** Client-side validation with the same limits as the API's TrainConfig schema. */
export function validateTrainForm(f: TrainForm): Record<string, string> {
  const e: Record<string, string> = {};
  if (!f.dataset_id) e.dataset_id = "Choose a dataset.";
  if (!f.split_id) e.split_id = "Choose a split.";
  if (f.input_size.trim() && (!isInt(f.input_size) || Number(f.input_size) < 16)) {
    e.input_size = "An integer ≥ 16, or empty for the model default.";
  }
  if (!isInt(f.epochs) || Number(f.epochs) < 1) e.epochs = "An integer ≥ 1.";
  if (f.max_steps.trim() && (!isInt(f.max_steps) || Number(f.max_steps) < 1)) {
    e.max_steps = "An integer ≥ 1, or empty for no cap.";
  }
  if (!isInt(f.batch_size) || Number(f.batch_size) < 1) e.batch_size = "An integer ≥ 1.";
  if (!isNum(f.lr) || Number(f.lr) < 0) e.lr = "A number ≥ 0.";
  if (!isNum(f.weight_decay) || Number(f.weight_decay) < 0) e.weight_decay = "A number ≥ 0.";
  if (!isNum(f.momentum) || Number(f.momentum) < 0) e.momentum = "A number ≥ 0.";
  if (!isInt(f.patience) || Number(f.patience) < 1) e.patience = "An integer ≥ 1.";
  if (!isNum(f.min_delta) || Number(f.min_delta) < 0) e.min_delta = "A number ≥ 0.";
  if (!isInt(f.seed) || Number(f.seed) >= 2 ** 32) e.seed = "An integer from 0 to 4294967295.";
  if (!isInt(f.num_workers)) e.num_workers = "An integer ≥ 0.";
  if (f.amp && f.device === "cpu") e.amp = "AMP needs a CUDA device.";
  if (f.model === "small_cnn" && f.weights !== "scratch")
    e.weights = "small_cnn has no pretrained weights.";
  if (f.finetune === "head_only" && f.weights !== "imagenet")
    e.finetune = "head_only needs ImageNet weights.";
  return e;
}

export function toTrainConfig(f: TrainForm): Record<string, unknown> {
  return {
    dataset_id: f.dataset_id,
    split_id: f.split_id,
    model: f.model,
    weights: f.weights,
    finetune: f.finetune,
    input_size: f.input_size.trim() ? Number(f.input_size) : null,
    epochs: Number(f.epochs),
    max_steps: f.max_steps.trim() ? Number(f.max_steps) : null,
    batch_size: Number(f.batch_size),
    optimizer: f.optimizer,
    lr: Number(f.lr),
    weight_decay: Number(f.weight_decay),
    momentum: Number(f.momentum),
    scheduler: f.scheduler,
    early_stopping: {
      enabled: f.early_stopping,
      monitor: f.monitor,
      patience: Number(f.patience),
      min_delta: Number(f.min_delta),
    },
    class_weighted_loss: f.class_weighted_loss,
    seed: Number(f.seed),
    device: f.device,
    amp: f.amp,
    num_workers: Number(f.num_workers),
    augmentation: {
      horizontal_flip: f.horizontal_flip,
      random_resized_crop: f.random_resized_crop,
    },
    allow_weight_fallback: f.allow_weight_fallback,
  };
}

function Field({
  id,
  label,
  error,
  hint,
  children,
}: {
  id: string;
  label: string;
  error?: string;
  hint?: string;
  children: ReactNode;
}) {
  return (
    <div className="field">
      <label htmlFor={id}>{label}</label>
      {children}
      {hint ? <span className="hint">{hint}</span> : null}
      {error ? (
        <span className="field-error" id={`${id}-err`}>
          {error}
        </span>
      ) : null}
    </div>
  );
}

export function TrainingPage() {
  const datasets = useDatasets();
  const [f, setF] = useState<TrainForm>(DEFAULT_FORM);
  const splits = useSplits(f.dataset_id || undefined);
  const start = useStartTraining();
  const [submitted, setSubmitted] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const errors = validateTrainForm(f);
  const show = (k: string) => (submitted ? errors[k] : undefined);
  const set = <K extends keyof TrainForm>(k: K, v: TrainForm[K]) =>
    setF((old) => ({ ...old, [k]: v }));
  const text = (k: keyof TrainForm) => ({
    id: `tr-${k}`,
    value: f[k] as string,
    onChange: (e: { target: { value: string } }) => set(k, e.target.value as never),
    "aria-invalid": Boolean(show(k)),
    "aria-describedby": show(k) ? `tr-${k}-err` : undefined,
  });
  const check = (k: keyof TrainForm, label: string) => (
    <label key={k}>
      <input
        type="checkbox"
        checked={f[k] as boolean}
        onChange={(e) => set(k, e.target.checked as never)}
      />{" "}
      {label}
    </label>
  );

  function submit(e: FormEvent) {
    e.preventDefault();
    setSubmitted(true);
    if (Object.keys(errors).length) return;
    start.mutate(toTrainConfig(f), { onSuccess: (r) => setJobId(r.job.id) });
  }

  return (
    <>
      <PageHeader title="Training">
        Configure a training run. Nothing starts until you press “Start training”; the run then
        executes as a job in the worker. Model selection and early stopping use the validation split
        only.
      </PageHeader>
      <Section title="Training configuration" id="train-form">
        <form onSubmit={submit} noValidate>
          <fieldset>
            <legend>Data</legend>
            <div className="field-row">
              <Field id="tr-dataset_id" label="Dataset" error={show("dataset_id")}>
                <select
                  {...text("dataset_id")}
                  onChange={(e) =>
                    setF((o) => ({ ...o, dataset_id: e.target.value, split_id: "" }))
                  }
                >
                  <option value="">— choose —</option>
                  {(datasets.data ?? []).map((d) => (
                    <option key={d.id} value={d.id}>
                      {d.name} ({d.id})
                    </option>
                  ))}
                </select>
              </Field>
              <Field id="tr-split_id" label="Split" error={show("split_id")}>
                <select {...text("split_id")} disabled={!f.dataset_id}>
                  <option value="">— choose —</option>
                  {(splits.data ?? []).map((s) => (
                    <option key={s.split_id} value={s.split_id}>
                      {s.split_id} (train {s.split_totals.train} / val {s.split_totals.val})
                    </option>
                  ))}
                </select>
              </Field>
            </div>
          </fieldset>
          <fieldset>
            <legend>Model</legend>
            <div className="field-row">
              <Field id="tr-model" label="Model">
                <select {...text("model")}>
                  {MODELS.map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
              </Field>
              <Field id="tr-weights" label="Weights" error={show("weights")}>
                <select {...text("weights")} disabled={f.model === "majority_baseline"}>
                  <option value="scratch">scratch</option>
                  <option value="imagenet">imagenet</option>
                </select>
              </Field>
              <Field id="tr-finetune" label="Fine-tune" error={show("finetune")}>
                <select {...text("finetune")} disabled={f.model === "majority_baseline"}>
                  <option value="full">full</option>
                  <option value="head_only">head_only</option>
                </select>
              </Field>
              <Field
                id="tr-input_size"
                label="Input size (px)"
                error={show("input_size")}
                hint="empty = model default"
              >
                <input {...text("input_size")} inputMode="numeric" />
              </Field>
            </div>
            {f.model === "majority_baseline" ? (
              <p className="hint">
                The majority baseline is fitted from TRAIN class counts; the training settings below
                are not used.
              </p>
            ) : null}
          </fieldset>
          <fieldset>
            <legend>Optimisation</legend>
            <div className="field-row">
              <Field id="tr-epochs" label="Epochs" error={show("epochs")}>
                <input {...text("epochs")} inputMode="numeric" />
              </Field>
              <Field
                id="tr-max_steps"
                label="Max steps"
                error={show("max_steps")}
                hint="empty = no cap"
              >
                <input {...text("max_steps")} inputMode="numeric" />
              </Field>
              <Field id="tr-batch_size" label="Batch size" error={show("batch_size")}>
                <input {...text("batch_size")} inputMode="numeric" />
              </Field>
              <Field id="tr-optimizer" label="Optimizer">
                <select {...text("optimizer")}>
                  <option value="adamw">adamw</option>
                  <option value="sgd">sgd</option>
                </select>
              </Field>
              <Field id="tr-lr" label="Learning rate" error={show("lr")}>
                <input {...text("lr")} inputMode="decimal" />
              </Field>
              <Field id="tr-weight_decay" label="Weight decay" error={show("weight_decay")}>
                <input {...text("weight_decay")} inputMode="decimal" />
              </Field>
              <Field id="tr-momentum" label="Momentum (sgd)" error={show("momentum")}>
                <input {...text("momentum")} inputMode="decimal" />
              </Field>
              <Field id="tr-scheduler" label="Scheduler">
                <select {...text("scheduler")}>
                  <option value="none">none</option>
                  <option value="cosine">cosine</option>
                </select>
              </Field>
            </div>
          </fieldset>
          <fieldset>
            <legend>Early stopping (on the validation split)</legend>
            <div className="field-row">
              <Field id="tr-monitor" label="Monitor">
                <select {...text("monitor")}>
                  <option value="val_loss">val_loss</option>
                  <option value="val_balanced_accuracy">val_balanced_accuracy</option>
                </select>
              </Field>
              <Field id="tr-patience" label="Patience" error={show("patience")}>
                <input {...text("patience")} inputMode="numeric" />
              </Field>
              <Field id="tr-min_delta" label="Min delta" error={show("min_delta")}>
                <input {...text("min_delta")} inputMode="decimal" />
              </Field>
            </div>
            {check("early_stopping", "early stopping enabled")}
          </fieldset>
          <fieldset>
            <legend>Run</legend>
            <div className="field-row">
              <Field id="tr-seed" label="Seed" error={show("seed")}>
                <input {...text("seed")} inputMode="numeric" />
              </Field>
              <Field id="tr-device" label="Device">
                <select {...text("device")}>
                  <option value="auto">auto</option>
                  <option value="cpu">cpu</option>
                  <option value="cuda">cuda</option>
                </select>
              </Field>
              <Field id="tr-num_workers" label="Data-loader workers" error={show("num_workers")}>
                <input {...text("num_workers")} inputMode="numeric" />
              </Field>
            </div>
            <div className="actions">
              {check("horizontal_flip", "horizontal flip")}
              {check("random_resized_crop", "random resized crop")}
              {check("class_weighted_loss", "class-weighted loss")}
              {check("amp", "AMP (CUDA only)")}
              {check(
                "allow_weight_fallback",
                "allow fallback to scratch if weights are unavailable",
              )}
            </div>
            {show("amp") ? <p className="field-error">{show("amp")}</p> : null}
          </fieldset>
          {submitted && Object.keys(errors).length ? (
            <p className="field-error" role="alert">
              Fix the highlighted fields ({Object.keys(errors).length}) before starting.
            </p>
          ) : null}
          <button type="submit" disabled={start.isPending}>
            Start training
          </button>
        </form>
        {start.isError ? <ErrorBox error={start.error} title="Training refused" /> : null}
      </Section>
      {jobId ? (
        <Section title="Training job" id="train-job">
          <JobPanel jobId={jobId} />
        </Section>
      ) : null}
    </>
  );
}
