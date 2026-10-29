import { http, HttpResponse } from "msw";
import { describe, expect, it } from "vitest";

import { server } from "../test/server";
import { ApiError, CLIENT_HEADER, request, userMessage } from "./client";

describe("API client", () => {
  it("sends the client header on mutations only", async () => {
    const seen: (string | null)[] = [];
    server.use(
      http.get("/api/x", ({ request: r }) => {
        seen.push(r.headers.get(CLIENT_HEADER));
        return HttpResponse.json({ ok: 1 });
      }),
      http.post("/api/x", ({ request: r }) => {
        seen.push(r.headers.get(CLIENT_HEADER));
        return HttpResponse.json({ ok: 2 });
      }),
      http.delete("/api/x", ({ request: r }) => {
        seen.push(r.headers.get(CLIENT_HEADER));
        return HttpResponse.json({ ok: 3 });
      }),
    );
    await request("/api/x");
    await request("/api/x", { json: { a: 1 } });
    await request("/api/x", { method: "DELETE" });
    expect(seen).toEqual([null, "1", "1"]);
  });

  it("maps the API error schema to ApiError with a user message", async () => {
    server.use(
      http.post("/api/locked", () =>
        HttpResponse.json(
          { error: { code: "test_split_locked", message: "locked" } },
          { status: 403 },
        ),
      ),
      http.get(
        "/api/plain",
        () => new HttpResponse("oops", { status: 500, statusText: "Server Error" }),
      ),
    );
    const err = await request("/api/locked", { json: {} }).catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).code).toBe("test_split_locked");
    expect((err as ApiError).status).toBe(403);
    expect((err as ApiError).message).toMatch(/TEST images are locked/);
    const plain = (await request("/api/plain").catch((e: unknown) => e)) as ApiError;
    expect(plain.code).toBe("http_500");
  });

  it("handles each important status and code distinctly", () => {
    expect(userMessage(429, "inference_busy", "busy")).toMatch(/busy with another request/);
    expect(userMessage(429, "queue_full", "full")).toMatch(/Too many jobs/);
    expect(userMessage(409, "experiment_not_completed", "running")).toMatch(/not completed/);
    expect(userMessage(409, "conflict", "x")).toBe("Conflict: x");
    expect(userMessage(413, "upload_too_large", "exceeds 10 MB")).toMatch(
      /too large: exceeds 10 MB/,
    );
    expect(userMessage(415, "unsupported_image", "not an image")).toMatch(/not supported/);
    expect(userMessage(422, "validation_error", "bad field")).toBe("Invalid input: bad field");
    expect(userMessage(403, "missing_client_header", "x")).toMatch(/client header/);
    expect(userMessage(422, "explanation_not_supported", "baseline")).toMatch(/No Grad-CAM/);
    expect(userMessage(404, "not_found", "gone")).toBe("Not found: gone");
    expect(userMessage(500, "internal_error", "")).toBe("Request failed (HTTP 500)");
  });

  it("adds query parameters and skips empty ones", async () => {
    let url = "";
    server.use(
      http.get("/api/q", ({ request: r }) => {
        url = r.url;
        return HttpResponse.json([]);
      }),
    );
    await request("/api/q", { query: { a: 1, b: "", c: undefined, d: "x y" } });
    expect(url).toMatch(/\/api\/q\?a=1&d=x\+y$/);
  });
});

describe("upload (XHR with progress)", () => {
  // jsdom + MSW cannot complete an XHR whose body is FormData (the request never resolves), so
  // this test drives upload() against a stub XMLHttpRequest; the e2e test covers a real browser.
  class FakeXHR {
    static last: FakeXHR | null = null;
    static status = 202;
    static response = JSON.stringify({ job: { id: "job-1" } });
    headers: Record<string, string> = {};
    method = "";
    url = "";
    status = 0;
    responseText = "";
    statusText = "Accepted";
    sent: unknown = null;
    upload: {
      onprogress:
        | ((e: { lengthComputable: boolean; loaded: number; total: number }) => void)
        | null;
    } = { onprogress: null };
    onload: (() => void) | null = null;
    onerror: (() => void) | null = null;
    open(method: string, url: string) {
      this.method = method;
      this.url = url;
    }
    setRequestHeader(k: string, v: string) {
      this.headers[k] = v;
    }
    send(body: unknown) {
      this.sent = body;
      FakeXHR.last = this;
      this.upload.onprogress?.({ lengthComputable: true, loaded: 5, total: 10 });
      this.status = FakeXHR.status;
      this.responseText = FakeXHR.response;
      this.onload?.();
    }
  }

  it("sends the client header, reports progress and returns the JSON body", async () => {
    const original = globalThis.XMLHttpRequest;
    globalThis.XMLHttpRequest = FakeXHR as unknown as typeof XMLHttpRequest;
    try {
      const { upload } = await import("./client");
      const progress: number[] = [];
      const form = new FormData();
      form.append("name", "x");
      const r = await upload<{ job: { id: string } }>("/api/datasets/uploads", form, (f) =>
        progress.push(f),
      );
      expect(r.job.id).toBe("job-1");
      expect(FakeXHR.last?.method).toBe("POST");
      expect(FakeXHR.last?.headers[CLIENT_HEADER]).toBe("1");
      expect(FakeXHR.last?.sent).toBe(form);
      expect(progress).toEqual([0.5]);
      FakeXHR.status = 413;
      FakeXHR.response = JSON.stringify({
        error: { code: "upload_too_large", message: "upload exceeds 200 MB" },
      });
      const err = (await upload("/api/datasets/uploads", form).catch(
        (e: unknown) => e,
      )) as ApiError;
      expect(err.code).toBe("upload_too_large");
      expect(err.message).toMatch(/too large: upload exceeds 200 MB/);
    } finally {
      globalThis.XMLHttpRequest = original;
    }
  });
});
