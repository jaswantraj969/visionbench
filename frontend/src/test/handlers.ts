import { http, HttpResponse } from "msw";

import * as F from "./fixtures";

/** Default handlers: a small, healthy backend. Tests override single routes with server.use(). */
export const handlers = [
  http.get("/api/health", () => HttpResponse.json({ status: "ok", version: "0.0.1" })),
  http.get("/api/ready", () =>
    HttpResponse.json({ status: "ready", checks: { data_root: "ok", torch: "ok" } }),
  ),
  http.get("/api/overview", () => HttpResponse.json(F.overview)),
  http.get("/api/datasets", () => HttpResponse.json(F.datasets)),
  http.get(`/api/datasets/${F.DS}`, () => HttpResponse.json(F.datasetDetail)),
  http.get(`/api/datasets/${F.DS}/splits`, () => HttpResponse.json(F.splits)),
  http.get(`/api/datasets/${F.DS}/splits/${F.SP}`, () => HttpResponse.json(F.splitDetail)),
  http.get(`/api/datasets/${F.DS}/splits/${F.SP}/sample`, () => HttpResponse.json(F.sample)),
  http.get(`/api/datasets/${F.DS}/splits/${F.SP}/audit`, () =>
    HttpResponse.json({ error: { code: "audit_not_found", message: "no audit" } }, { status: 404 }),
  ),
  http.get("/api/experiments", () => HttpResponse.json(F.experiments)),
  http.get(`/api/experiments/${F.EXP}`, () => HttpResponse.json(F.experimentDetail)),
  http.get(`/api/experiments/${F.BASE}`, () => HttpResponse.json(F.baselineDetail)),
  http.get("/api/jobs", () => HttpResponse.json([F.job("completed")])),
  http.get(`/api/jobs/${F.JOB}`, () => HttpResponse.json(F.job("completed"))),
  http.get("/api/evaluations", () => HttpResponse.json(F.experimentDetail.evaluations)),
  http.get(`/api/evaluations/${F.EV}`, () => HttpResponse.json(F.evaluationDetail)),
  http.get("/api/benchmarks", () => HttpResponse.json(F.overview.benchmarks)),
  http.get("/api/benchmarks/toy-v1", () => HttpResponse.json(F.benchmarkDetail)),
  http.get("/api/benchmarks/toy-v1/documents/benchmark_protocol", () =>
    HttpResponse.json({
      benchmark: "toy-v1",
      document: "benchmark_protocol",
      path: "BENCHMARK_PROTOCOL.md",
      markdown: "# Toy protocol",
    }),
  ),
  http.get("/api/robustness-suites", () => HttpResponse.json(F.suites)),
  http.get("/api/robustness", () => HttpResponse.json([])),
];
