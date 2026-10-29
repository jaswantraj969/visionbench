// Small, self-consistent API fixtures for component tests (shapes follow the real API).

export const DS = "ds-0123456789ab";
export const SP = "sp-0123456789ab";
export const EXP = "exp-20260101-000000-aaaaaa";
export const BASE = "exp-20260101-000001-bbbbbb";
export const JOB = "job-20260101-000000-cccccc";
export const EV = "ev-20260101-000000-dddddd";

const hashes = (ok = true) => ({
  present: true,
  ok,
  files: [{ name: "BENCHMARK_PROTOCOL.md", expected: "a".repeat(64), actual: "a".repeat(64), ok }],
});

export const overview = {
  version: "0.0.1",
  datasets: 1,
  experiments: 2,
  experiments_by_status: { completed: 2 },
  jobs_by_status: { completed: 3 },
  test_images_unlocked: false,
  inference_device: "cpu",
  benchmarks: [{ name: "toy-v1", protocol_hashes: hashes(), robustness_hashes: hashes(), runs: 2 }],
};

export const datasets = [
  {
    id: DS,
    name: "shapes",
    created_at: "2026-01-01T00:00:00+00:00",
    classes: ["circle", "square"],
    class_counts: { circle: 10, square: 12 },
    total_images: 22,
    fingerprint: "f".repeat(64),
  },
];

export const datasetDetail = {
  ...datasets[0],
  visionbench_version: "0.0.1",
  import_config: {},
  validation: { accepted: 22, issue_counts: {} },
  report: {
    imbalance_ratio: 1.2,
    width: { min: 32, median: 32, max: 32 },
    height: { min: 32, median: 32, max: 32 },
    mode_counts: { RGB: 22 },
    format_counts: { PNG: 22 },
    total_bytes: 40960,
    issues: { zero_byte: { count: 1, relpaths: ["circle/empty.png"] } },
    duplicates: {
      dhash_threshold: 5,
      exact_duplicate_files: 0,
      files_in_multi_member_groups: 0,
      groups_total: 22,
      label_conflict_groups: [],
      multi_member_groups: 0,
      perceptual_check_skipped_low_information: { count: 0 },
      perceptual_edges: 0,
    },
    samples: {
      per_class: 2,
      seed: 0,
      relpaths: { circle: ["circle/c1.png"], square: ["square/s1.png"] },
    },
  },
};

export const splits = [
  {
    split_id: SP,
    created_at: "2026-01-01T00:00:00+00:00",
    manifest_sha256: "m".repeat(64),
    algorithm: "vb-split-v2",
    config: { seed: 0 },
    split_totals: { train: 12, val: 5, test: 5 },
    leakage_check: {
      groups_checked: 22,
      groups_in_multiple_splits: 0,
      sha256_checked: 22,
      sha256_in_multiple_splits: 0,
    },
  },
];

export const splitDetail = {
  ...splits[0],
  counts: {
    circle: { train: 6, val: 2, test: 2, total: 10 },
    square: { train: 6, val: 3, test: 3, total: 12 },
  },
  target_counts: {},
  warnings: [],
  excluded: [],
  excluded_groups: [],
  low_information_files: [],
  entries_total: 22,
};

const summary = (id: string, model: string, tags: string[]) => ({
  id,
  status: "completed",
  error: null,
  model,
  weights_source: model === "majority_baseline" ? "none" : "scratch",
  dataset_id: DS,
  split_id: SP,
  seed: 0,
  best_epoch: 2,
  best_metric: 0.5,
  stopped_reason: model === "majority_baseline" ? "baseline_fit" : "epochs_completed",
  train_seconds: 12.5,
  created_at: "2026-01-01T00:00:00+00:00",
  started_at: "2026-01-01T00:00:00+00:00",
  finished_at: "2026-01-01T00:00:10+00:00",
  tags,
});

export const experiments = [
  summary(EXP, "small_cnn", ["benchmark:toy-v1"]),
  summary(BASE, "majority_baseline", ["pilot:toy-v1"]),
];

export const experimentDetail = {
  ...experiments[0],
  run_dir: `experiments/${EXP}`,
  config: {
    config: { model: "small_cnn", epochs: 2 },
    resolved: { class_names: ["circle", "square"] },
  },
  config_display: { model: "small_cnn", epochs: 2, batch_size: 8 },
  env: { device: "cpu", torch: "2.x" },
  weights_enum: null,
  weights_sha256: null,
  epochs: [
    {
      epoch: 1,
      train_loss: 0.9,
      val_loss: 0.8,
      val_accuracy: 0.6,
      val_balanced_accuracy: 0.55,
      lr: 0.01,
      epoch_seconds: 1.2,
      steps: 3,
    },
    {
      epoch: 2,
      train_loss: 0.7,
      val_loss: 0.6,
      val_accuracy: 0.7,
      val_balanced_accuracy: 0.65,
      lr: 0.01,
      epoch_seconds: 1.1,
      steps: 6,
    },
  ],
  artifacts: [
    {
      kind: "checkpoint",
      relpath: `experiments/${EXP}/checkpoints/best.pt`,
      sha256: "b".repeat(64),
      size_bytes: 2048,
    },
  ],
  evaluations: [
    {
      id: EV,
      experiment_id: EXP,
      split: "val",
      checkpoint: "best",
      created_at: "2026-01-01T00:00:20+00:00",
      metrics_path: "x",
      forced: false,
      reason: null,
    },
  ],
};

export const baselineDetail = {
  ...experiments[1],
  run_dir: `experiments/${BASE}`,
  config: { config: { model: "majority_baseline", epochs: 10 } },
  config_display: { model: "majority_baseline", epochs: "n/a", batch_size: "n/a", weights: "n/a" },
  env: null,
  weights_enum: null,
  weights_sha256: null,
  epochs: [],
  artifacts: [],
  evaluations: [],
};

export const job = (status: string, extra: Record<string, unknown> = {}) => ({
  id: JOB,
  kind: "train",
  status,
  params: {},
  result: status === "completed" ? { experiment_id: EXP } : null,
  error: status === "failed" ? "RuntimeError: boom" : null,
  progress: {
    phase: "training",
    epoch: 1,
    epochs: 2,
    step: 3,
    latest: { val_loss: 0.8, val_balanced_accuracy: 0.55 },
  },
  cancel_requested: false,
  heartbeat_at: null,
  created_at: "2026-01-01T00:00:00+00:00",
  started_at: "2026-01-01T00:00:01+00:00",
  finished_at: status === "running" ? null : "2026-01-01T00:00:09+00:00",
  ...extra,
});

export const evaluationDetail = {
  evaluation: experimentDetail.evaluations[0],
  metrics: {
    experiment_id: EXP,
    model: "small_cnn",
    split: "val",
    checkpoint: { name: "best", sha256: "b".repeat(64), epoch: 2 },
    metrics: {
      accuracy: 0.7,
      balanced_accuracy: 0.65,
      macro_f1: null,
      class_names: ["circle", "square"],
      per_class: {
        circle: { support: 2, predicted: 3, precision: 0.6667, recall: 1.0, f1: 0.8 },
        square: { support: 3, predicted: 0, precision: null, recall: 0.0, f1: null },
      },
      confusion_matrix: {
        labels: ["circle", "square"],
        counts: [
          [2, 0],
          [3, 0],
        ],
        row_normalized: [
          [1, 0],
          [1, 0],
        ],
      },
      undefined: { "per_class.square.precision": "no predictions of this class" },
    },
    bootstrap: {
      metrics: { accuracy: { low: 0.5, high: 0.9 }, balanced_accuracy: { low: 0.4, high: 0.8 } },
      resamples: 50,
      seed: 0,
    },
    test_use: { test_evaluation_number: null, forced: false, reason: null },
  },
};

export const comparison = {
  report_id: "rp-20260101-000000-eeeeee",
  split: "val",
  note: "VALIDATION (selection-biased)",
  selection_bias_note:
    "validation metrics are selection-biased: the same validation data chose each model's checkpoint",
  reference_experiment: BASE,
  class_names: ["circle", "square"],
  n_samples: 5,
  rows: [
    {
      experiment_id: EXP,
      model: "small_cnn",
      weights_source: "scratch",
      checkpoint: "best",
      eval_id: EV,
      best_epoch: 2,
      input_size: 32,
      params_total: 100,
      train_seconds: 12.5,
      latency_bs1_median_ms: null,
      latency_note: "",
      test_evaluations: 0,
      val: {
        balanced_accuracy: 0.65,
        balanced_accuracy_ci: { low: 0.4, high: 0.8 },
        macro_f1: 0.6,
        macro_f1_ci: { low: 0.3, high: 0.8 },
        accuracy: 0.7,
        accuracy_ci: { low: 0.5, high: 0.9 },
      },
      paired_difference_vs_reference: {
        metrics: { balanced_accuracy: { point: 0.15, low: 0.0, high: 0.3 } },
      },
    },
    {
      experiment_id: BASE,
      model: "majority_baseline",
      weights_source: "none",
      checkpoint: "baseline",
      eval_id: "ev-x",
      best_epoch: 1,
      input_size: null,
      params_total: null,
      train_seconds: 0.1,
      latency_bs1_median_ms: null,
      latency_note: "",
      test_evaluations: 0,
      val: {
        balanced_accuracy: 0.5,
        balanced_accuracy_ci: { low: 0.5, high: 0.5 },
        macro_f1: 0.33,
        macro_f1_ci: { low: 0.3, high: 0.4 },
        accuracy: 0.6,
        accuracy_ci: { low: 0.4, high: 0.8 },
      },
      paired_difference_vs_reference: null,
    },
  ],
  confounds: [{ code: "single_seed_per_model", message: "one seed per model", values: {} }],
  val_ranking: {
    basis: "val balanced accuracy",
    order: [EXP, BASE],
    ranks: { [EXP]: 1, [BASE]: 2 },
  },
  paths: { json: "reports/x/comparison.json", markdown: "reports/x/comparison.md" },
};

export const benchmarkDetail = {
  name: "toy-v1",
  protocol_hashes: hashes(),
  robustness_hashes: hashes(),
  final_analysis_plan_hashes: { present: false, ok: false, files: [] },
  documents: {
    benchmark_protocol: true,
    robustness_protocol: true,
    final_analysis_plan: false,
    robustness_val: true,
    final_report: false,
  },
  runs: [
    {
      model: "small_cnn",
      seed: 0,
      experiment_id: EXP,
      driver_status: "completed",
      status: "completed",
      best_epoch: 2,
      epochs_trained: 2,
      stopped_reason: "epochs_completed",
      train_seconds: 12.5,
      test_evaluations: 0,
      val: { balanced_accuracy: 0.65, macro_f1: 0.6, accuracy: 0.7 },
    },
    {
      model: "small_cnn",
      seed: 1,
      experiment_id: "exp-20260101-000002-ffffff",
      driver_status: "completed",
      status: "completed",
      best_epoch: 1,
      epochs_trained: 2,
      stopped_reason: "early_stopping",
      train_seconds: 11.0,
      test_evaluations: 0,
      val: { balanced_accuracy: 0.6, macro_f1: 0.55, accuracy: 0.66 },
    },
  ],
  val_label: "VALIDATION (selection-biased) — not test results",
  val_summary: [
    {
      model: "small_cnn",
      balanced_accuracy: { mean: 0.625, sd: 0.0354, n: 2, values: [0.65, 0.6] },
      macro_f1: { mean: 0.575, sd: 0.0354, n: 2, values: [0.6, 0.55] },
      best_epochs: [2, 1],
      seeds: [0, 1],
    },
  ],
  robustness_val: {
    label: "VALIDATION (selection-biased) — not test results",
    scope: "Scope: synthetic corruptions at 3 fixed severities on one dataset.",
    confounds: { resolution_upsampling: "corruption and content are scaled together" },
    suite: "toy-c-v1",
    suite_sha256: "c".repeat(64),
    summary: {},
    runs: [
      {
        model: "small_cnn",
        seed: 0,
        experiment_id: EXP,
        clean_balanced_accuracy: 0.65,
        conditions: { "gaussian_noise/0.04": { balanced_accuracy: 0.4, paired: {} } },
      },
      {
        model: "small_cnn",
        seed: 1,
        experiment_id: "exp-20260101-000002-ffffff",
        clean_balanced_accuracy: 0.6,
        conditions: { "gaussian_noise/0.04": { balanced_accuracy: 0.3, paired: {} } },
      },
    ],
  },
  final_report: null,
};

export const suites = [
  {
    name: "toy-c-v1",
    sha256: "c".repeat(64),
    base_seed: 0,
    conditions: [{ corruption: "gaussian_noise", parameter: "sigma", severity: "0.04" }],
  },
];

export const sample = {
  dataset_id: DS,
  split_id: SP,
  split: "val",
  seed: 0,
  per_class: 1,
  rule: "numpy default_rng(seed)",
  images: [
    { relpath: "circle/c1.png", label: "circle", split: "val" },
    { relpath: "square/s1.png", label: "square", split: "val" },
  ],
};

export const predictResult = {
  experiment_id: EXP,
  model: "small_cnn",
  checkpoint: { name: "best", sha256: "b".repeat(64), epoch: 2 },
  input: { type: "dataset", relpath: "circle/c1.png", split: "val", original_size: [32, 32] },
  preprocessing: null,
  predicted_class: "circle",
  probabilities: [
    { class: "circle", probability: 0.9 },
    { class: "square", probability: 0.1 },
  ],
  timing_ms: { forward: 1.5, model_load: null, total: 3.2 },
  warnings: [],
  notes: ["VALIDATION image: val chose this model's checkpoint (selection-biased)"],
};

export const gradcamResult = {
  ...predictResult,
  gradcam_version: "vb-gradcam-2",
  target_class: "circle",
  target_probability: 0.9,
  predicted_class: "circle",
  predicted_probability: 0.9,
  cam: {
    layer: "features.block4.relu",
    feature_map_size: [2, 2],
    resolution: "2x2",
    values: [
      [0, 1],
      [0.5, 0],
    ],
    all_zero: false,
  },
  geometry: { unseen_fraction: 0.234375, input_size: 224, seen_box_in_original: {} },
  overlay: { png: "data:image/png;base64,AAAA", display_size: { width: 256, height: 256 } },
  warnings: [],
  limitations:
    "Grad-CAM shows which regions of the final convolutional feature map most increased the score of the target class. It is coarse (feature-map resolution stated above), is not a causal explanation, and does not show that the prediction is correct.",
};
