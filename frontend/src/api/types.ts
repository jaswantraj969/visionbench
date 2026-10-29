// Types used by the pages. Where the API declares a Pydantic response model, the type comes from
// the generated OpenAPI schema; endpoints that return free-form JSON documents (recorded
// metrics.json, comparison reports) are described here by the fields the UI reads.
import type { components } from "./schema";

type S = components["schemas"];

export type Overview = S["Overview"];
export type BenchmarkListItem = S["BenchmarkListItem"];
export type BenchmarkDetail = S["BenchmarkDetail"];
export type BenchmarkDocument = S["BenchmarkDocument"];
export type HashStatus = S["HashStatus"];
export type DatasetSummary = S["DatasetSummary"];
export type JobOut = S["JobOut"];
export type JobAccepted = S["JobAccepted"];
export type SplitSample = S["SplitSample"];
export type SuiteOut = S["SuiteOut"];
export type TrainConfig = S["TrainConfig"];

export type Split = "train" | "val" | "test";

export interface MinMedMax {
  min: number;
  median: number;
  max: number;
}

export interface DatasetDetail extends DatasetSummary {
  visionbench_version: string;
  import_config: Record<string, unknown>;
  validation: { accepted: number; issue_counts: Record<string, number> };
  report: {
    imbalance_ratio: number;
    width: MinMedMax;
    height: MinMedMax;
    mode_counts: Record<string, number>;
    format_counts: Record<string, number>;
    total_bytes: number;
    issues: Record<string, { count: number; relpaths?: string[] }>;
    duplicates: {
      dhash_threshold: number;
      exact_duplicate_files: number;
      files_in_multi_member_groups: number;
      groups_total: number;
      label_conflict_groups: unknown[];
      multi_member_groups: number;
      perceptual_check_skipped_low_information: Record<string, unknown>;
      perceptual_edges: number;
    };
    samples: { per_class: number; seed: number; relpaths: Record<string, string[]> };
  };
}

export interface SplitCounts {
  train: number;
  val: number;
  test: number;
  total: number;
}

export interface SplitSummaryOut {
  split_id: string;
  created_at: string;
  manifest_sha256: string;
  algorithm: string;
  config: Record<string, unknown>;
  split_totals: { train: number; val: number; test: number };
  leakage_check: {
    groups_checked: number;
    groups_in_multiple_splits: number;
    sha256_checked: number;
    sha256_in_multiple_splits: number;
  };
}

export interface SplitDetail extends SplitSummaryOut {
  counts: Record<string, SplitCounts>;
  target_counts: Record<string, Record<string, number>>;
  warnings: unknown[];
  excluded: unknown[];
  excluded_groups: unknown[];
  low_information_files: unknown;
  entries_total: number;
}

export interface ExperimentSummary {
  id: string;
  status: string;
  error: string | null;
  model: string;
  weights_source: string | null;
  dataset_id: string;
  split_id: string;
  seed: number;
  best_epoch: number | null;
  best_metric: number | null;
  stopped_reason: string | null;
  train_seconds: number | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  tags: string[];
}

export interface EpochRow {
  epoch: number;
  train_loss: number | null;
  val_loss: number;
  val_accuracy: number;
  val_balanced_accuracy: number;
  lr: number | null;
  epoch_seconds: number;
  steps: number;
}

export interface EvaluationRow {
  id: string;
  experiment_id: string;
  split: string;
  checkpoint: string;
  created_at: string;
  metrics_path: string;
  forced: boolean;
  reason: string | null;
}

export interface ArtifactRow {
  kind: string;
  relpath: string;
  sha256: string;
  size_bytes: number;
}

export interface ExperimentDetail extends ExperimentSummary {
  run_dir: string;
  config: { config: Record<string, unknown>; resolved?: Record<string, unknown> };
  config_display: Record<string, unknown>;
  env: Record<string, unknown> | null;
  weights_enum: string | null;
  weights_sha256: string | null;
  epochs: EpochRow[];
  artifacts: ArtifactRow[];
  evaluations: EvaluationRow[];
}

export interface CI {
  low: number | null;
  high: number | null;
  resamples_defined?: number;
}

export interface PerClass {
  support: number;
  predicted: number;
  precision: number | null;
  recall: number | null;
  f1: number | null;
}

export interface MetricsDoc {
  experiment_id: string;
  model: string;
  split: string;
  checkpoint: { name: string; sha256: string; epoch: number };
  metrics: {
    accuracy: number | null;
    balanced_accuracy: number | null;
    macro_f1: number | null;
    class_names: string[];
    per_class: Record<string, PerClass>;
    confusion_matrix: { labels: string[]; counts: number[][]; row_normalized: (number | null)[][] };
    undefined: Record<string, string>;
  };
  bootstrap: { metrics: Record<string, CI>; resamples: number; seed: number };
  test_use: { test_evaluation_number: number | null; forced: boolean; reason: string | null };
}

export interface EvaluationDetail {
  evaluation: EvaluationRow;
  metrics: MetricsDoc;
}

export interface ComparisonRow {
  experiment_id: string;
  model: string;
  weights_source: string;
  checkpoint: string;
  eval_id: string;
  best_epoch: number | null;
  input_size: number | null;
  params_total: number | null;
  train_seconds: number | null;
  latency_bs1_median_ms: number | null;
  latency_note: string;
  test_evaluations: number;
  val?: Record<string, number | CI | null>;
  test?: Record<string, number | CI | null>;
  paired_difference_vs_reference?: {
    metrics: Record<string, { point: number | null; low: number | null; high: number | null }>;
  } | null;
}

export interface ComparisonReport {
  report_id: string;
  split: Split;
  note: string | null;
  selection_bias_note?: string | null;
  reference_experiment: string;
  class_names: string[];
  n_samples: number;
  rows: ComparisonRow[];
  confounds: { code: string; message: string; values: Record<string, unknown> }[];
  val_ranking?: { basis: string; order: string[]; ranks: Record<string, number> } | null;
  paths: { json: string; markdown: string };
}

export interface ProbabilityRow {
  class: string;
  probability: number;
}

export interface PredictResult {
  experiment_id: string;
  model: string;
  checkpoint: { name: string; sha256: string; epoch: number };
  input: Record<string, unknown> & { original_size: [number, number]; split?: string | null };
  preprocessing: Record<string, unknown> | null;
  predicted_class: string;
  probabilities: ProbabilityRow[];
  timing_ms: Record<string, number | null>;
  warnings: string[];
  notes: string[];
}

export interface GradcamResult extends Omit<PredictResult, "probabilities" | "predicted_class"> {
  gradcam_version: string;
  target_class: string;
  target_probability: number;
  predicted_class: string;
  predicted_probability: number;
  cam: {
    layer: string;
    feature_map_size: [number, number];
    resolution: string;
    values: number[][];
    all_zero: boolean;
  };
  geometry: {
    unseen_fraction: number;
    input_size: number;
    seen_box_in_original: Record<string, number>;
  };
  overlay: { png: string; display_size: { width: number; height: number } };
  limitations: string;
}

export interface RobustnessRow {
  id: string;
  experiment_id: string;
  split: string;
  suite: string;
  corruption: string;
  severity: string;
  clean_evaluation_id: string;
  created_at: string;
}
