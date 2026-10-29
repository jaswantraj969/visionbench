// Data fetching with TanStack Query. Mutations never start on their own: every one is called
// from an explicit user action (button, form submit).
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { request, upload } from "./client";
import type {
  BenchmarkDetail,
  BenchmarkDocument,
  BenchmarkListItem,
  ComparisonReport,
  DatasetDetail,
  DatasetSummary,
  EvaluationDetail,
  EvaluationRow,
  ExperimentDetail,
  ExperimentSummary,
  GradcamResult,
  JobAccepted,
  JobOut,
  Overview,
  PredictResult,
  RobustnessRow,
  SplitDetail,
  SplitSample,
  SplitSummaryOut,
  SuiteOut,
} from "./types";

const ACTIVE = new Set(["queued", "running"]);

export const useHealth = () =>
  useQuery({
    queryKey: ["health"],
    queryFn: () => request<{ status: string; version: string }>("/api/health"),
  });

export const useReady = () =>
  useQuery({
    queryKey: ["ready"],
    queryFn: () => request<{ status: string; checks: Record<string, string> }>("/api/ready"),
    retry: false,
  });

export const useOverview = () =>
  useQuery({ queryKey: ["overview"], queryFn: () => request<Overview>("/api/overview") });

export const useDatasets = () =>
  useQuery({ queryKey: ["datasets"], queryFn: () => request<DatasetSummary[]>("/api/datasets") });

export const useDataset = (id: string | undefined) =>
  useQuery({
    queryKey: ["dataset", id],
    queryFn: () => request<DatasetDetail>(`/api/datasets/${id}`),
    enabled: Boolean(id),
  });

export const useSplits = (datasetId: string | undefined) =>
  useQuery({
    queryKey: ["splits", datasetId],
    queryFn: () => request<SplitSummaryOut[]>(`/api/datasets/${datasetId}/splits`),
    enabled: Boolean(datasetId),
  });

export const useSplit = (datasetId: string | undefined, splitId: string | undefined) =>
  useQuery({
    queryKey: ["split", datasetId, splitId],
    queryFn: () => request<SplitDetail>(`/api/datasets/${datasetId}/splits/${splitId}`),
    enabled: Boolean(datasetId && splitId),
  });

export const useSplitSample = (
  datasetId: string | undefined,
  splitId: string | undefined,
  split: "train" | "val",
  perClass: number,
  seed: number,
) =>
  useQuery({
    queryKey: ["sample", datasetId, splitId, split, perClass, seed],
    queryFn: () =>
      request<SplitSample>(`/api/datasets/${datasetId}/splits/${splitId}/sample`, {
        query: { split, per_class: perClass, seed },
      }),
    enabled: Boolean(datasetId && splitId),
  });

export const useAudit = (datasetId: string | undefined, splitId: string | undefined) =>
  useQuery({
    queryKey: ["audit", datasetId, splitId],
    queryFn: () =>
      request<Record<string, unknown>>(`/api/datasets/${datasetId}/splits/${splitId}/audit`),
    enabled: Boolean(datasetId && splitId),
    retry: false,
  });

export const useExperiments = () =>
  useQuery({
    queryKey: ["experiments"],
    queryFn: () => request<ExperimentSummary[]>("/api/experiments"),
  });

export const useExperiment = (id: string | undefined) =>
  useQuery({
    queryKey: ["experiment", id],
    queryFn: () => request<ExperimentDetail>(`/api/experiments/${id}`),
    enabled: Boolean(id),
  });

export const useEvaluations = (experimentId?: string) =>
  useQuery({
    queryKey: ["evaluations", experimentId ?? "all"],
    queryFn: () =>
      request<EvaluationRow[]>("/api/evaluations", { query: { experiment_id: experimentId } }),
  });

export const useEvaluation = (id: string | undefined) =>
  useQuery({
    queryKey: ["evaluation", id],
    queryFn: () => request<EvaluationDetail>(`/api/evaluations/${id}`),
    enabled: Boolean(id),
  });

export const useJobs = (pollMs = 2000) =>
  useQuery({
    queryKey: ["jobs"],
    queryFn: () => request<JobOut[]>("/api/jobs", { query: { limit: 200 } }),
    refetchInterval: (q) =>
      (q.state.data ?? []).some((j) => ACTIVE.has(j.status)) ? pollMs : false,
  });

/** One job, polled every second while it is queued or running. */
export const useJob = (id: string | undefined, pollMs = 1000) =>
  useQuery({
    queryKey: ["job", id],
    queryFn: () => request<JobOut>(`/api/jobs/${id}`),
    enabled: Boolean(id),
    refetchInterval: (q) => (q.state.data && ACTIVE.has(q.state.data.status) ? pollMs : false),
  });

export const useBenchmarks = () =>
  useQuery({
    queryKey: ["benchmarks"],
    queryFn: () => request<BenchmarkListItem[]>("/api/benchmarks"),
  });

export const useBenchmark = (name: string | undefined) =>
  useQuery({
    queryKey: ["benchmark", name],
    queryFn: () => request<BenchmarkDetail>(`/api/benchmarks/${name}`),
    enabled: Boolean(name),
  });

export const useBenchmarkDocument = (name: string | undefined, doc: string | undefined) =>
  useQuery({
    queryKey: ["benchmark-doc", name, doc],
    queryFn: () => request<BenchmarkDocument>(`/api/benchmarks/${name}/documents/${doc}`),
    enabled: Boolean(name && doc),
    retry: false,
  });

export const useSuites = () =>
  useQuery({ queryKey: ["suites"], queryFn: () => request<SuiteOut[]>("/api/robustness-suites") });

export const useRobustness = (experimentId?: string) =>
  useQuery({
    queryKey: ["robustness", experimentId ?? "all"],
    queryFn: () =>
      request<RobustnessRow[]>("/api/robustness", { query: { experiment_id: experimentId } }),
  });

function useInvalidate() {
  const qc = useQueryClient();
  return (...keys: string[]) =>
    Promise.all(keys.map((k) => qc.invalidateQueries({ queryKey: [k] })));
}

export function useUploadDataset() {
  const invalidate = useInvalidate();
  return useMutation({
    mutationFn: (v: { file: File; name: string; onProgress?: (f: number) => void }) => {
      const form = new FormData();
      form.append("name", v.name);
      form.append("file", v.file);
      return upload<JobAccepted>("/api/datasets/uploads", form, v.onProgress);
    },
    onSuccess: () => invalidate("jobs"),
  });
}

export function useCreateSplit(datasetId: string) {
  const invalidate = useInvalidate();
  return useMutation({
    mutationFn: (body: Record<string, unknown>) =>
      request<JobAccepted>(`/api/datasets/${datasetId}/splits`, { json: body }),
    onSuccess: () => invalidate("jobs"),
  });
}

export function useStartTraining() {
  const invalidate = useInvalidate();
  return useMutation({
    mutationFn: (config: Record<string, unknown>) =>
      request<JobAccepted>("/api/experiments", { json: config }),
    onSuccess: () => invalidate("jobs"),
  });
}

export function useCancelJob() {
  const invalidate = useInvalidate();
  return useMutation({
    mutationFn: (id: string) => request<JobOut>(`/api/jobs/${id}/cancel`, { method: "POST" }),
    onSuccess: () => invalidate("jobs", "job"),
  });
}

export function useStartEvaluation(experimentId: string) {
  const invalidate = useInvalidate();
  return useMutation({
    mutationFn: (body: Record<string, unknown>) =>
      request<JobAccepted>(`/api/experiments/${experimentId}/evaluations`, { json: body }),
    onSuccess: () => invalidate("jobs"),
  });
}

export function useCompare() {
  return useMutation({
    mutationFn: (body: Record<string, unknown>) =>
      request<ComparisonReport>("/api/comparisons", { json: body }),
  });
}

export type ImageInput =
  | { kind: "dataset"; dataset_id: string; relpath: string }
  | { kind: "upload"; file: File };

function inferenceBody(input: ImageInput, target?: string): { json?: unknown; body?: BodyInit } {
  if (input.kind === "dataset") {
    const json: Record<string, string> = { dataset_id: input.dataset_id, relpath: input.relpath };
    if (target) json.target_class = target;
    return { json };
  }
  const form = new FormData();
  form.append("file", input.file);
  if (target) form.append("target_class", target);
  return { body: form };
}

export function usePredict(experimentId: string) {
  return useMutation({
    mutationFn: (input: ImageInput) =>
      request<PredictResult>(`/api/experiments/${experimentId}/predict`, {
        method: "POST",
        ...inferenceBody(input),
      }),
  });
}

export function useGradcam(experimentId: string) {
  return useMutation({
    mutationFn: (v: { input: ImageInput; target?: string }) =>
      request<GradcamResult>(`/api/experiments/${experimentId}/gradcam`, {
        method: "POST",
        ...inferenceBody(v.input, v.target),
      }),
  });
}

export function useStartRobustness(experimentId: string) {
  const invalidate = useInvalidate();
  return useMutation({
    mutationFn: (body: { suite: string; split: "val" | "test" }) =>
      request<JobAccepted>(`/api/experiments/${experimentId}/robustness`, { json: body }),
    onSuccess: () => invalidate("jobs"),
  });
}
