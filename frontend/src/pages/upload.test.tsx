import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { describe, expect, it, vi } from "vitest";

import * as F from "../test/fixtures";
import { renderAt } from "../test/render";
import { server } from "../test/server";

// Only the XHR transport is replaced (jsdom + MSW cannot complete a FormData XHR); the page,
// hooks and the job panel are real. upload() itself is tested in client.test.ts.
const uploadSpy = vi.fn();
vi.mock("../api/client", async (importOriginal) => {
  const mod = await importOriginal<typeof import("../api/client")>();
  return {
    ...mod,
    upload: (path: string, form: FormData, onProgress?: (f: number) => void) => {
      uploadSpy(path, form);
      onProgress?.(1);
      return Promise.resolve({ job: F.job("queued") });
    },
  };
});

describe("Dataset upload", () => {
  it("uploads the chosen ZIP with its name and tracks the import job", async () => {
    const user = userEvent.setup();
    server.use(
      http.get(`/api/jobs/${F.JOB}`, () =>
        HttpResponse.json(F.job("running", { kind: "dataset_import" })),
      ),
    );
    renderAt("/datasets");
    await screen.findByRole("link", { name: "shapes" });
    await user.type(screen.getByLabelText("Dataset name"), "toy");
    await user.upload(
      screen.getByLabelText("ZIP file"),
      new File(["PK"], "toy.zip", { type: "application/zip" }),
    );
    await user.click(screen.getByRole("button", { name: "Upload and import" }));
    await waitFor(() => expect(uploadSpy).toHaveBeenCalledTimes(1));
    const [path, form] = uploadSpy.mock.calls[0] as [string, FormData];
    expect(path).toBe("/api/datasets/uploads");
    expect(form.get("name")).toBe("toy");
    expect((form.get("file") as File).name).toBe("toy.zip");
    expect(await screen.findByText("Import job")).toBeInTheDocument();
    expect(await screen.findByText("dataset_import")).toBeInTheDocument();
  });

  it("rejects a non-ZIP file before sending", async () => {
    const user = userEvent.setup({ applyAccept: false });
    uploadSpy.mockClear();
    renderAt("/datasets");
    await screen.findByRole("link", { name: "shapes" });
    await user.type(screen.getByLabelText("Dataset name"), "toy");
    await user.upload(
      screen.getByLabelText("ZIP file"),
      new File(["x"], "notes.txt", { type: "text/plain" }),
    );
    await user.click(screen.getByRole("button", { name: "Upload and import" }));
    expect(screen.getByText("The file must be a .zip archive.")).toBeInTheDocument();
    expect(uploadSpy).not.toHaveBeenCalled();
  });
});
