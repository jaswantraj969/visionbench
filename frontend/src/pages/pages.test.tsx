import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import type { JsonBodyType } from "msw";
import { describe, expect, it } from "vitest";

import { CLIENT_HEADER } from "../api/client";
import * as F from "../test/fixtures";
import { renderAt } from "../test/render";
import type { Seen } from "../test/render";
import { server } from "../test/server";
import { aggregate } from "./RobustnessPage";
import { DEFAULT_FORM, toTrainConfig, validateTrainForm } from "./TrainingPage";
import { filterExperiments } from "./ExperimentsPage";
import { describeProgress } from "../components/JobPanel";

const failing = (path: string, status = 500, method: "get" | "post" = "get") =>
  http[method](path, () =>
    HttpResponse.json(
      { error: { code: "internal_error", message: "internal server error" } },
      { status },
    ),
  );

function recordPosts(path: string, response: JsonBodyType, status = 202) {
  const seen: Seen[] = [];
  server.use(
    http.post(path, async ({ request }) => {
      const ct = request.headers.get("content-type") ?? "";
      const body = ct.includes("json")
        ? await request.json()
        : ct.includes("multipart")
          ? (await request.arrayBuffer(), "multipart")
          : null;
      seen.push({ method: "POST", path, header: request.headers.get(CLIENT_HEADER), body });
      return HttpResponse.json(response, { status });
    }),
  );
  return seen;
}

describe("Overview", () => {
  it("shows health, counts, the test lock and benchmark hash status", async () => {
    renderAt("/");
    expect(screen.getByText(/Loading health/)).toBeInTheDocument();
    expect(await screen.findByText("ready (data_root: ok, torch: ok)")).toBeInTheDocument();
    expect(await screen.findByText("Locked.")).toBeInTheDocument();
    expect(screen.getAllByText("frozen, hashes OK")).toHaveLength(2);
    expect(screen.getByRole("link", { name: "toy-v1" })).toBeInTheDocument();
  });

  it("shows an error when the API is not reachable", async () => {
    server.use(failing("/api/health"), failing("/api/overview"));
    renderAt("/");
    expect(await screen.findByText("API not reachable")).toBeInTheDocument();
    expect(await screen.findAllByText(/Could not load overview/)).not.toHaveLength(0);
  });

  it("shows the empty benchmark state", async () => {
    server.use(
      http.get("/api/overview", () => HttpResponse.json({ ...F.overview, benchmarks: [] })),
    );
    renderAt("/");
    expect(await screen.findByText("No frozen benchmark folders found.")).toBeInTheDocument();
  });
});

describe("Datasets", () => {
  it("lists datasets and validates the upload form before sending", async () => {
    const user = userEvent.setup();
    const seen = recordPosts("/api/datasets/uploads", { job: F.job("queued") });
    renderAt("/datasets");
    expect(await screen.findByRole("link", { name: "shapes" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Upload and import" }));
    expect(screen.getByText("A name is required.")).toBeInTheDocument();
    expect(screen.getByText("Choose a ZIP file.")).toBeInTheDocument();
    expect(seen).toHaveLength(0);
  });

  it("shows the empty and error states", async () => {
    server.use(http.get("/api/datasets", () => HttpResponse.json([])));
    renderAt("/datasets");
    expect(await screen.findByText("No datasets yet. Upload a ZIP below.")).toBeInTheDocument();
    server.use(failing("/api/datasets"));
    renderAt("/datasets");
    expect(await screen.findByText("Could not load datasets")).toBeInTheDocument();
  });

  it("shows a dataset with class counts, samples, attribution and splits", async () => {
    const user = userEvent.setup();
    renderAt(`/datasets/${F.DS}`);
    expect(await screen.findByText("Dataset: shapes")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /circle sample circle\/c1.png/ })).toHaveAttribute(
      "src",
      `/api/datasets/${F.DS}/images?relpath=circle%2Fc1.png`,
    );
    expect(
      screen.getAllByText(/Contains modified Copernicus Sentinel data/).length,
    ).toBeGreaterThan(0);
    expect(screen.getByText("zero_byte")).toBeInTheDocument();
    await user.click(await screen.findByRole("button", { name: "Show" }));
    expect(await screen.findByText("Images per class and split")).toBeInTheDocument();
    expect(await screen.findByText(/No audit was run/)).toBeInTheDocument();
  });

  it("shows a not-found dataset as an error", async () => {
    server.use(failing(`/api/datasets/${F.DS}`, 404));
    renderAt(`/datasets/${F.DS}`);
    expect(await screen.findByText("Could not load dataset")).toBeInTheDocument();
  });

  it("creates a split only from a valid form", async () => {
    const user = userEvent.setup();
    const seen = recordPosts(`/api/datasets/${F.DS}/splits`, { job: F.job("queued") });
    renderAt(`/datasets/${F.DS}`);
    await screen.findByText("Dataset: shapes");
    const val = screen.getByLabelText("Val fraction");
    await user.clear(val);
    await user.type(val, "0.5");
    expect(screen.getByText("Fractions must sum to 1.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Create split (job)" })).toBeDisabled();
    await user.clear(val);
    await user.type(val, "0.15");
    await user.click(screen.getByRole("button", { name: "Create split (job)" }));
    await waitFor(() => expect(seen).toHaveLength(1));
    expect(seen[0]).toMatchObject({ header: "1", body: { seed: 0, fractions: [0.7, 0.15, 0.15] } });
  });
});

describe("Training", () => {
  it("validates with the schema limits and never starts on its own", async () => {
    const user = userEvent.setup();
    const seen = recordPosts("/api/experiments", { job: F.job("queued") });
    renderAt("/training");
    await screen.findByRole("option", { name: /shapes/ });
    expect(seen).toHaveLength(0);
    await user.click(screen.getByRole("button", { name: "Start training" }));
    expect(screen.getByText("Choose a dataset.")).toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent(/Fix the highlighted fields/);
    expect(seen).toHaveLength(0);
  });

  it("submits a valid config with the client header and shows the job", async () => {
    const user = userEvent.setup();
    const seen = recordPosts("/api/experiments", { job: F.job("queued") });
    server.use(http.get(`/api/jobs/${F.JOB}`, () => HttpResponse.json(F.job("running"))));
    renderAt("/training");
    await screen.findByRole("option", { name: /shapes/ });
    await user.selectOptions(screen.getByLabelText("Dataset"), F.DS);
    await user.selectOptions(
      await screen.findByLabelText("Split"),
      await screen.findByRole("option", { name: new RegExp(F.SP) }),
    );
    const epochs = screen.getByLabelText("Epochs");
    await user.clear(epochs);
    await user.type(epochs, "0");
    await user.click(screen.getByRole("button", { name: "Start training" }));
    expect(screen.getByText("An integer ≥ 1.")).toBeInTheDocument();
    await user.clear(epochs);
    await user.type(epochs, "2");
    await user.click(screen.getByRole("button", { name: "Start training" }));
    await waitFor(() => expect(seen).toHaveLength(1));
    expect(seen[0]?.header).toBe("1");
    expect(seen[0]?.body).toMatchObject({
      dataset_id: F.DS,
      split_id: F.SP,
      epochs: 2,
      model: "small_cnn",
    });
    expect(await screen.findByText("Training job")).toBeInTheDocument();
    expect(await screen.findByText("1 / 2")).toBeInTheDocument();
  });

  it("maps form values to a TrainConfig", () => {
    const f = { ...DEFAULT_FORM, dataset_id: F.DS, split_id: F.SP, input_size: "", max_steps: "5" };
    expect(validateTrainForm(f)).toEqual({});
    expect(toTrainConfig(f)).toMatchObject({
      input_size: null,
      max_steps: 5,
      early_stopping: { patience: 5 },
    });
    expect(validateTrainForm({ ...f, input_size: "8" }).input_size).toMatch(/≥ 16/);
    expect(validateTrainForm({ ...f, amp: true, device: "cpu" }).amp).toMatch(/CUDA/);
    expect(validateTrainForm({ ...f, weights: "imagenet" }).weights).toMatch(/small_cnn/);
  });
});

describe("Jobs", () => {
  it("lists jobs, filters, and shows the empty state", async () => {
    const user = userEvent.setup();
    renderAt("/jobs");
    expect(await screen.findByRole("link", { name: F.JOB })).toBeInTheDocument();
    await user.selectOptions(screen.getByLabelText("Status"), "failed");
    expect(screen.getByText("No jobs match.")).toBeInTheDocument();
  });

  it("shows progress and cancels a running job only after confirmation", async () => {
    const user = userEvent.setup();
    server.use(http.get(`/api/jobs/${F.JOB}`, () => HttpResponse.json(F.job("running"))));
    const seen = recordPosts(
      `/api/jobs/${F.JOB}/cancel`,
      F.job("running", { cancel_requested: true }),
      200,
    );
    renderAt(`/jobs/${F.JOB}`);
    expect(
      await screen.findByText("latest val balanced accuracy (VALIDATION)"),
    ).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Cancel job…" }));
    expect(seen).toHaveLength(0);
    await user.click(screen.getByRole("button", { name: "Yes, cancel the job" }));
    await waitFor(() => expect(seen).toHaveLength(1));
    expect(seen[0]?.header).toBe("1");
  });

  it("shows a failed job's error", async () => {
    server.use(http.get(`/api/jobs/${F.JOB}`, () => HttpResponse.json(F.job("failed"))));
    renderAt(`/jobs/${F.JOB}`);
    expect(await screen.findByText("RuntimeError: boom")).toBeInTheDocument();
  });

  it("shows an error when the jobs cannot be loaded", async () => {
    server.use(failing("/api/jobs"));
    renderAt("/jobs");
    expect(await screen.findByText("Could not load jobs")).toBeInTheDocument();
  });

  it("describes progress of other job kinds", () => {
    expect(describeProgress({ conditions_done: 3, conditions_total: 12, last: "jpeg/10" })).toEqual(
      [
        ["conditions", "3 / 12"],
        ["last", "jpeg/10"],
      ],
    );
  });
});

describe("Experiments", () => {
  it("lists experiments with tags and filters them", async () => {
    const user = userEvent.setup();
    renderAt("/experiments");
    expect(await screen.findByText("benchmark:toy-v1")).toBeInTheDocument();
    expect(screen.getByText("pilot:toy-v1")).toBeInTheDocument();
    await user.selectOptions(screen.getByLabelText("tag"), "pilot");
    expect(screen.queryByText("benchmark:toy-v1")).not.toBeInTheDocument();
    expect(
      filterExperiments(F.experiments, {
        model: "all",
        status: "all",
        seed: "all",
        tag: "untagged",
      }),
    ).toEqual([]);
  });

  it("shows the empty and error states", async () => {
    server.use(http.get("/api/experiments", () => HttpResponse.json([])));
    renderAt("/experiments");
    expect(await screen.findByText("No experiments match.")).toBeInTheDocument();
    server.use(failing("/api/experiments"));
    renderAt("/experiments");
    expect(await screen.findByText("Could not load experiments")).toBeInTheDocument();
  });

  it("shows an experiment with curves, best epoch, config and evaluations", async () => {
    renderAt(`/experiments/${F.EXP}`);
    expect(await screen.findByText(`Experiment ${F.EXP}`)).toBeInTheDocument();
    expect(screen.getByText("epochs_completed")).toBeInTheDocument();
    expect(screen.getByText("Training curves")).toBeInTheDocument();
    expect(screen.getAllByText("VALIDATION (selection-biased)").length).toBeGreaterThan(0);
    expect(screen.getByRole("link", { name: F.EV })).toBeInTheDocument();
  });

  it("shows n/a for the majority baseline's unused settings", async () => {
    renderAt(`/experiments/${F.BASE}`);
    expect(await screen.findByText(`Experiment ${F.BASE}`)).toBeInTheDocument();
    const config = screen.getByRole("region", { name: "Configuration" });
    expect(within(config).getAllByText("n/a").length).toBeGreaterThanOrEqual(3);
    expect(screen.getByText("No epochs recorded.")).toBeInTheDocument();
  });
});

describe("Evaluate & Compare", () => {
  it("renders undefined metrics as undefined with the reason, never 0", async () => {
    renderAt(`/evaluate?evaluation=${F.EV}`);
    const view = await screen.findByRole("region", { name: "Evaluation" });
    expect(await within(view).findByText("macro f1")).toBeInTheDocument();
    expect(within(view).getAllByText("undefined").length).toBeGreaterThanOrEqual(2);
    expect(within(view).getByText(/no predictions of this class/)).toBeInTheDocument();
    expect(within(view).getByText("VALIDATION (selection-biased)")).toBeInTheDocument();
    expect(within(view).getByRole("table", { name: /Confusion matrix/ })).toBeInTheDocument();
    expect(within(view).getByLabelText("true square, predicted circle: 3")).toBeInTheDocument();
  });

  it("starts a VAL evaluation with the client header; TEST is not offered", async () => {
    const user = userEvent.setup();
    const seen = recordPosts(`/api/experiments/${F.EXP}/evaluations`, { job: F.job("queued") });
    renderAt("/evaluate");
    await screen.findByRole("option", { name: new RegExp(F.EXP) });
    await user.selectOptions(screen.getByLabelText("Experiment (completed)"), F.EXP);
    await user.click(screen.getByRole("button", { name: "Start VAL evaluation (job)" }));
    await waitFor(() => expect(seen).toHaveLength(1));
    expect(seen[0]).toMatchObject({ header: "1", body: { split: "val", checkpoint: "best" } });
    expect(screen.getByText(/TEST evaluation/)).toBeInTheDocument();
  });

  it("compares experiments and shows CIs, paired differences, confounds and the bias note", async () => {
    const user = userEvent.setup();
    server.use(
      http.get("/api/evaluations", () =>
        HttpResponse.json([
          F.experimentDetail.evaluations[0],
          { ...F.experimentDetail.evaluations[0], id: "ev-x", experiment_id: F.BASE },
        ]),
      ),
    );
    const seen = recordPosts("/api/comparisons", F.comparison, 201);
    renderAt("/evaluate");
    const cmp = await screen.findByRole("region", { name: "Compare recorded evaluations" });
    await user.click(await within(cmp).findByRole("checkbox", { name: F.EXP }));
    await user.click(within(cmp).getByRole("checkbox", { name: F.BASE }));
    await user.click(within(cmp).getByRole("button", { name: "Compare" }));
    await waitFor(() => expect(seen).toHaveLength(1));
    expect(seen[0]?.body).toMatchObject({ split: "val", experiment_ids: [F.EXP, F.BASE] });
    expect(
      await within(cmp).findByText(/selection-biased: the same validation data/),
    ).toBeInTheDocument();
    expect(within(cmp).getByText("0.6500 [0.4000, 0.8000]")).toBeInTheDocument();
    expect(within(cmp).getByText("+0.1500 [+0.0000, +0.3000]")).toBeInTheDocument();
    expect(within(cmp).getByText("single_seed_per_model")).toBeInTheDocument();
  });
});

describe("Explain", () => {
  it("always shows the limitations text and offers no TEST images", async () => {
    renderAt("/explain");
    expect(
      screen.getByText(
        /is not a causal explanation, and does not show that the prediction is correct/,
      ),
    ).toBeInTheDocument();
    expect(
      await screen.findByRole("button", { name: /circle: circle\/c1.png/ }),
    ).toBeInTheDocument();
    expect(screen.getByText("TEST images", { selector: "strong" })).toBeInTheDocument();
  });

  it("predicts and explains a VAL image with the client header", async () => {
    const user = userEvent.setup();
    const p = recordPosts(`/api/experiments/${F.EXP}/predict`, F.predictResult, 200);
    const g = recordPosts(`/api/experiments/${F.EXP}/gradcam`, F.gradcamResult, 200);
    renderAt("/explain");
    await user.click(await screen.findByRole("button", { name: /circle: circle\/c1.png/ }));
    await user.click(screen.getByRole("button", { name: "Predict" }));
    await waitFor(() => expect(p).toHaveLength(1));
    expect(p[0]).toMatchObject({
      header: "1",
      body: { dataset_id: F.DS, relpath: "circle/c1.png" },
    });
    expect(await screen.findByText("0.9000")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Grad-CAM" }));
    await waitFor(() => expect(g).toHaveLength(1));
    expect(await screen.findByText("2x2 at features.block4.relu")).toBeInTheDocument();
    expect(screen.getByText("23.4 %")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /Grad-CAM overlay/ })).toBeInTheDocument();
  });

  it("explains a test_split_locked refusal", async () => {
    const user = userEvent.setup();
    server.use(
      http.post(`/api/experiments/${F.EXP}/predict`, () =>
        HttpResponse.json(
          { error: { code: "test_split_locked", message: "locked" } },
          { status: 403 },
        ),
      ),
    );
    renderAt("/explain");
    await user.click(await screen.findByRole("button", { name: /circle: circle\/c1.png/ }));
    await user.click(screen.getByRole("button", { name: "Predict" }));
    expect(
      await screen.findByText(/TEST images are locked until the single final test evaluation/),
    ).toBeInTheDocument();
  });

  it("shows an inference_busy refusal distinctly", async () => {
    const user = userEvent.setup();
    server.use(
      http.post(`/api/experiments/${F.EXP}/gradcam`, () =>
        HttpResponse.json({ error: { code: "inference_busy", message: "busy" } }, { status: 429 }),
      ),
    );
    renderAt("/explain");
    await user.click(await screen.findByRole("button", { name: /circle: circle\/c1.png/ }));
    await user.click(screen.getByRole("button", { name: "Grad-CAM" }));
    expect(await screen.findByText(/busy with another request/)).toBeInTheDocument();
  });
});

describe("Robustness", () => {
  it("shows the suite, scope, VALIDATION table with mean ± sd, and starts a job", async () => {
    const user = userEvent.setup();
    const seen = recordPosts(`/api/experiments/${F.EXP}/robustness`, { job: F.job("queued") });
    renderAt("/robustness");
    expect(await screen.findByText(/Scope: synthetic corruptions/)).toBeInTheDocument();
    const results = await screen.findByRole("region", { name: "Results per model and condition" });
    expect(await within(results).findByText("0.3500 ± 0.0707")).toBeInTheDocument();
    expect(within(results).getByText("VALIDATION (selection-biased)")).toBeInTheDocument();
    await user.selectOptions(screen.getByLabelText("Experiment"), F.EXP);
    await user.click(screen.getByRole("button", { name: "Start VAL robustness (job)" }));
    await waitFor(() => expect(seen).toHaveLength(1));
    expect(seen[0]).toMatchObject({ header: "1", body: { suite: "toy-c-v1", split: "val" } });
    expect(screen.getByText(/TEST robustness/)).toBeInTheDocument();
  });

  it("aggregates seeds per model and condition", () => {
    const rows = aggregate(
      [
        {
          model: "m",
          seed: 0,
          clean_balanced_accuracy: 0.9,
          conditions: { a: { balanced_accuracy: 0.5 } },
        },
        { model: "m", seed: 1, clean_balanced_accuracy: 0.8, conditions: { a: null } },
      ],
      ["a"],
    );
    expect(rows[0]?.per.a).toEqual({ mean: 0.5, sd: null, n: 1 });
    expect(rows[0]?.clean.mean).toBeCloseTo(0.85);
  });

  it("shows the empty state without results", async () => {
    server.use(
      http.get("/api/benchmarks/toy-v1", () =>
        HttpResponse.json({ ...F.benchmarkDetail, robustness_val: null }),
      ),
    );
    renderAt("/robustness");
    expect(
      await screen.findByText("No robustness results recorded for the benchmark yet."),
    ).toBeInTheDocument();
  });
});

describe("Benchmark", () => {
  it("shows hashes, runs, the VALIDATION summary and no TEST results yet", async () => {
    const user = userEvent.setup();
    renderAt("/benchmark");
    expect(await screen.findByText("Protocol: toy-v1")).toBeInTheDocument();
    expect(screen.getAllByText(/all hashes OK/)).toHaveLength(2);
    expect(screen.getByText("0.6250 ± 0.0354")).toBeInTheDocument();
    expect(screen.getAllByText("VALIDATION (selection-biased)").length).toBeGreaterThan(0);
    expect(screen.getByText(/Not evaluated yet/)).toBeInTheDocument();
    expect(screen.getByText("TEST")).toBeInTheDocument();
    await user.selectOptions(screen.getByLabelText("Read a document"), "benchmark_protocol");
    expect(await screen.findByText("# Toy protocol")).toBeInTheDocument();
  });

  it("shows a hash mismatch and an error state", async () => {
    server.use(
      http.get("/api/benchmarks/toy-v1", () =>
        HttpResponse.json({
          ...F.benchmarkDetail,
          protocol_hashes: {
            present: true,
            ok: false,
            files: [{ name: "a.yaml", expected: "x", actual: "y", ok: false }],
          },
        }),
      ),
    );
    renderAt("/benchmark");
    expect(await screen.findByText("HASH MISMATCH: a.yaml")).toBeInTheDocument();
    server.use(failing("/api/benchmarks"));
    renderAt("/benchmark");
    expect(await screen.findByText("Could not load benchmarks")).toBeInTheDocument();
  });
});

describe("Routing", () => {
  it("shows a not-found page", async () => {
    renderAt("/nowhere");
    expect(await screen.findByText("Page not found")).toBeInTheDocument();
  });
});
