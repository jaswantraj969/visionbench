// The one API client. Every request goes to `${API_BASE}/api/...` (same origin by default: Vite
// proxies /api in development, `visionbench serve` serves the built app in production).
// Every state-changing request carries `X-VisionBench-Request: 1` (the API's CSRF defence).

export const API_BASE: string = import.meta.env.VITE_API_BASE ?? "";
export const CLIENT_HEADER = "X-VisionBench-Request";
const UNSAFE = new Set(["POST", "PUT", "PATCH", "DELETE"]);

/** An API error with the server's code and a message meant for people. */
export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly serverMessage: string;

  constructor(status: number, code: string, serverMessage: string) {
    super(userMessage(status, code, serverMessage));
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.serverMessage = serverMessage;
  }
}

/** Map the API error schema {"error": {"code", "message"}} to a user-facing message. */
export function userMessage(status: number, code: string, message: string): string {
  switch (code) {
    case "test_split_locked":
      return (
        "TEST images are locked until the single final test evaluation (M8), so that the " +
        "test split is never inspected or used for decisions before then."
      );
    case "inference_busy":
      return "The model is busy with another request. Try again in a moment.";
    case "queue_full":
      return "Too many jobs are queued. Wait for some to finish, then try again.";
    case "missing_client_header":
      return "The request was refused because the client header was missing.";
    case "upload_too_large":
    case "image_too_large":
      return `The file is too large: ${message}`;
    case "unsupported_image":
    case "unsupported_media_type":
      return `This file type is not supported: ${message}`;
    case "corruptions_not_materialized":
      return `The corrupted images for this suite are not prepared yet: ${message}`;
    case "experiment_not_completed":
      return `The experiment is not completed: ${message}`;
    case "test_reevaluation_requires_force":
      return "This experiment was already evaluated on TEST; a repeat is not allowed here.";
    case "explanation_not_supported":
      return `No Grad-CAM for this model: ${message}`;
  }
  switch (status) {
    case 403:
      return `Not allowed: ${message}`;
    case 404:
      return `Not found: ${message}`;
    case 409:
      return `Conflict: ${message}`;
    case 413:
      return `Too large: ${message}`;
    case 415:
      return `Unsupported type: ${message}`;
    case 422:
      return `Invalid input: ${message}`;
    case 429:
      return `Busy: ${message}`;
  }
  return message || `Request failed (HTTP ${status})`;
}

async function parseError(res: Response): Promise<ApiError> {
  let code = `http_${res.status}`;
  let message = res.statusText;
  try {
    const body = (await res.json()) as { error?: { code?: string; message?: string } };
    if (body.error?.code) code = body.error.code;
    if (body.error?.message) message = body.error.message;
  } catch {
    // not JSON: keep the HTTP status text
  }
  return new ApiError(res.status, code, message);
}

export interface RequestOptions {
  method?: string;
  json?: unknown;
  body?: BodyInit;
  query?: Record<string, string | number | boolean | undefined | null>;
  signal?: AbortSignal;
}

export function apiUrl(path: string, query?: RequestOptions["query"]): string {
  const params = new URLSearchParams();
  for (const [k, v] of Object.entries(query ?? {})) {
    if (v !== undefined && v !== null && v !== "") params.set(k, String(v));
  }
  const qs = params.toString();
  return `${API_BASE}${path}${qs ? `?${qs}` : ""}`;
}

export async function request<T>(path: string, opts: RequestOptions = {}): Promise<T> {
  const method = (
    opts.method ?? (opts.json !== undefined || opts.body ? "POST" : "GET")
  ).toUpperCase();
  const headers: Record<string, string> = { Accept: "application/json" };
  if (UNSAFE.has(method)) headers[CLIENT_HEADER] = "1";
  let body = opts.body;
  if (opts.json !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(opts.json);
  }
  const res = await fetch(apiUrl(path, opts.query), { method, headers, body, signal: opts.signal });
  if (!res.ok) throw await parseError(res);
  return (await res.json()) as T;
}

/** Multipart upload with progress (XHR: fetch has no upload progress events). */
export function upload<T>(
  path: string,
  form: FormData,
  onProgress?: (fraction: number) => void,
): Promise<T> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", apiUrl(path));
    xhr.setRequestHeader(CLIENT_HEADER, "1");
    xhr.setRequestHeader("Accept", "application/json");
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total);
    };
    xhr.onerror = () => reject(new ApiError(0, "network_error", "network error during upload"));
    xhr.onload = () => {
      let parsed: unknown = null;
      try {
        parsed = JSON.parse(xhr.responseText);
      } catch {
        parsed = null;
      }
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(parsed as T);
        return;
      }
      const err = (parsed as { error?: { code?: string; message?: string } } | null)?.error;
      reject(
        new ApiError(xhr.status, err?.code ?? `http_${xhr.status}`, err?.message ?? xhr.statusText),
      );
    };
    xhr.send(form);
  });
}

export function errorText(error: unknown): string {
  if (error instanceof ApiError) return error.message;
  if (error instanceof Error) return error.message;
  return String(error);
}
