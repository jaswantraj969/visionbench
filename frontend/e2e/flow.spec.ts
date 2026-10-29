import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { expect, test } from "@playwright/test";
import type { APIRequestContext, Page } from "@playwright/test";

// Full user flow against the real backend started by playwright.config.ts (synthetic data).
interface Info {
  data_root: string;
  zip: string;
  suite: string;
  python: string;
}

function must<T>(value: T | undefined, what: string): T {
  if (value === undefined) throw new Error(`missing ${what}`);
  return value;
}

function info(): Info {
  const root = join(process.env.E2E_ROOT as string, "run");
  return JSON.parse(readFileSync(join(root, "e2e.json"), "utf-8")) as Info;
}

async function waitForJob(page: Page, status = "completed") {
  // The job panel polls the API; wait for the final status badge.
  await expect(page.locator(".job-panel .badge").first()).toHaveText(status, {
    timeout: 5 * 60_000,
  });
}

async function apiJson<T>(request: APIRequestContext, path: string): Promise<T> {
  const r = await request.get(path);
  expect(r.ok()).toBeTruthy();
  return (await r.json()) as T;
}

async function train(
  page: Page,
  ds: string,
  sp: string,
  model: string,
  extra: Record<string, string>,
) {
  await page.goto("/training");
  await page.getByLabel("Dataset").selectOption(ds);
  await page.getByLabel("Split").selectOption(sp);
  await page.getByLabel("Model").selectOption(model);
  for (const [label, value] of Object.entries(extra))
    await page.getByLabel(label, { exact: true }).fill(value);
  await page.getByLabel("Device").selectOption("cpu");
  await page.getByRole("button", { name: "Start training" }).click();
  await expect(page.getByRole("heading", { name: "Training job" })).toBeVisible();
}

test("upload, split, train, evaluate, compare, explain, robustness, cancel and test lock", async ({
  page,
  request,
}) => {
  const env = info();

  // 1. Upload a ZIP -> import job
  await page.goto("/datasets");
  await expect(page.getByText("No datasets yet. Upload a ZIP below.")).toBeVisible();
  await page.getByLabel("Dataset name").fill("e2e-shapes");
  await page.getByLabel("ZIP file").setInputFiles(env.zip);
  await page.getByRole("button", { name: "Upload and import" }).click();
  await expect(page.getByRole("heading", { name: "Import job" })).toBeVisible();
  await waitForJob(page);
  const datasets = await apiJson<{ id: string }[]>(request, "/api/datasets");
  const ds = must(datasets[0], "dataset").id;

  // 2. Dataset detail -> create a split
  await page.goto(`/datasets/${ds}`);
  await expect(page.getByRole("heading", { name: "Dataset: e2e-shapes" })).toBeVisible();
  await page.getByRole("button", { name: "Create split (job)" }).click();
  await waitForJob(page);
  const splits = await apiJson<{ split_id: string }[]>(request, `/api/datasets/${ds}/splits`);
  const sp = must(splits[0], "split").split_id;
  await page.reload();
  await page.getByRole("button", { name: "Show" }).click();
  await expect(page.getByText("Images per class and split")).toBeVisible();

  // 3. Train (tiny, CPU, max_steps) with visible progress; plus a majority baseline
  await train(page, ds, sp, "small_cnn", {
    "Input size (px)": "32",
    Epochs: "2",
    "Max steps": "4",
    "Batch size": "8",
  });
  await expect(page.getByText("phase")).toBeVisible({ timeout: 120_000 });
  await waitForJob(page);
  await expect(page.locator(".job-panel")).toContainText("epoch");
  await train(page, ds, sp, "majority_baseline", {});
  await waitForJob(page);
  const exps = await apiJson<{ id: string; model: string }[]>(request, "/api/experiments");
  const cnn = must(
    exps.find((e) => e.model === "small_cnn"),
    "small_cnn run",
  ).id;
  const base = must(
    exps.find((e) => e.model === "majority_baseline"),
    "baseline run",
  ).id;

  // experiment detail with curves
  await page.goto(`/experiments/${cnn}`);
  await expect(page.getByRole("heading", { name: "Training curves" })).toBeVisible();
  await expect(page.getByRole("region", { name: "Run" }).getByText("max_steps")).toBeVisible();

  // 4. VAL evaluation of both (jobs)
  for (const exp of [cnn, base]) {
    await page.goto(`/evaluate?experiment=${exp}`);
    await page.getByLabel("Experiment (completed)").selectOption(exp);
    await page.getByRole("button", { name: "Start VAL evaluation (job)" }).click();
    await waitForJob(page);
  }

  // 5. Compare (VALIDATION)
  await page.goto("/evaluate");
  const cmp = page.getByRole("region", { name: "Compare recorded evaluations" });
  await cmp.getByRole("checkbox", { name: cnn }).check();
  await cmp.getByRole("checkbox", { name: base }).check();
  await cmp.getByRole("button", { name: "Compare" }).click();
  await expect(cmp.getByText(/selection-biased/).first()).toBeVisible();
  await expect(cmp.getByText("Confound flags")).toBeVisible();

  // evaluation viewer with confusion matrix
  const evals = await apiJson<{ id: string; experiment_id: string }[]>(
    request,
    `/api/evaluations?experiment_id=${cnn}`,
  );
  await page.goto(`/evaluate?evaluation=${must(evals[0], "evaluation").id}`);
  await expect(page.getByRole("table", { name: /Confusion matrix/ })).toBeVisible();

  // 6. Predict + Grad-CAM on a VAL image
  await page.goto("/explain");
  await expect(page.getByText(/is not a causal explanation/).first()).toBeVisible();
  await page.getByLabel("Experiment").selectOption(cnn);
  await page.locator(".image-pick").first().click();
  await page.getByRole("button", { name: "Predict" }).click();
  await expect(page.getByRole("heading", { name: "Prediction" })).toBeVisible({ timeout: 120_000 });
  await page.getByRole("button", { name: "Grad-CAM" }).click();
  await expect(page.getByRole("img", { name: /Grad-CAM overlay/ })).toBeVisible({
    timeout: 120_000,
  });
  await expect(page.getByText(/at features\.block4\.relu/)).toBeVisible();

  // 7. TEST image (identical bytes uploaded) -> locked message
  const manifest = JSON.parse(
    readFileSync(join(env.data_root, "datasets", ds, "splits", sp, "manifest.json"), "utf-8"),
  ) as { entries: { relpath: string; split: string }[] };
  const testEntry = must(
    manifest.entries.find((e) => e.split === "test"),
    "test entry",
  );
  await page.getByLabel("upload").check();
  await page
    .getByLabel(/Image file/)
    .setInputFiles(join(env.data_root, "datasets", ds, "raw", ...testEntry.relpath.split("/")));
  await page.getByRole("button", { name: "Predict" }).click();
  await expect(
    page.getByText(/TEST images are locked until the single final test evaluation/),
  ).toBeVisible();

  // 8. Robustness on a small synthetic suite (VAL); corrupted images are materialized by the CLI
  execFileSync(
    env.python,
    [
      "-m",
      "visionbench.cli",
      "robustness",
      "materialize",
      ds,
      sp,
      "--suite",
      env.suite,
      "--split",
      "val",
    ],
    {
      env: { ...process.env, VISIONBENCH_DATA_ROOT: env.data_root },
      stdio: "pipe",
    },
  );
  await page.goto("/robustness");
  await expect(page.getByText("e2e-c-v1").first()).toBeVisible();
  await page.getByLabel("Experiment", { exact: true }).selectOption(cnn);
  await page.getByRole("button", { name: "Start VAL robustness (job)" }).click();
  await waitForJob(page);
  await expect(page.locator(".job-panel")).toContainText('"evaluated": 2');

  // 9. Cancel a running job
  await train(page, ds, sp, "small_cnn", {
    "Input size (px)": "32",
    Epochs: "200",
    "Batch size": "2",
  });
  await expect(page.locator(".job-panel .badge").first()).toHaveText("running", {
    timeout: 120_000,
  });
  await page.getByRole("button", { name: "Cancel job…" }).click();
  await page.getByRole("button", { name: "Yes, cancel the job" }).click();
  await waitForJob(page, "cancelled");

  // 10. Overview shows the lock and the counts; every page renders
  await page.goto("/");
  await expect(page.getByText("Locked.")).toBeVisible();
  for (const path of ["/jobs", "/experiments", "/benchmark"]) {
    await page.goto(path);
    await expect(page.locator("h1")).toBeVisible();
  }
});
