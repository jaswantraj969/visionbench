import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render } from "@testing-library/react";
import { RouterProvider, createMemoryRouter } from "react-router-dom";

import { routes } from "../App";

/** Render the real app routes at `path`, with a fresh query client (no retries in tests). */
export function renderAt(path: string) {
  const client = new QueryClient({
    defaultOptions: {
      queries: { retry: false, refetchOnWindowFocus: false },
      mutations: { retry: false },
    },
  });
  const router = createMemoryRouter(routes, { initialEntries: [path] });
  return {
    ...render(
      <QueryClientProvider client={client}>
        <RouterProvider router={router} />
      </QueryClientProvider>,
    ),
    router,
  };
}

/** Collect requests that reach the mock server (method, path, client header, body). */
export interface Seen {
  method: string;
  path: string;
  header: string | null;
  body: unknown;
}
