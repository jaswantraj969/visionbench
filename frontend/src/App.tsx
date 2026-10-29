import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { NavLink, Outlet, RouterProvider, createBrowserRouter } from "react-router-dom";
import type { RouteObject } from "react-router-dom";

import { ATTRIBUTION } from "./components/ui";
import { BenchmarkPage } from "./pages/BenchmarkPage";
import { DatasetDetailPage } from "./pages/DatasetDetailPage";
import { DatasetsPage } from "./pages/DatasetsPage";
import { EvaluatePage } from "./pages/EvaluatePage";
import { ExperimentDetailPage } from "./pages/ExperimentDetailPage";
import { ExperimentsPage } from "./pages/ExperimentsPage";
import { ExplainPage } from "./pages/ExplainPage";
import { JobDetailPage, JobsPage } from "./pages/JobsPage";
import { NotFoundPage } from "./pages/NotFoundPage";
import { OverviewPage } from "./pages/OverviewPage";
import { RobustnessPage } from "./pages/RobustnessPage";
import { TrainingPage } from "./pages/TrainingPage";

export const NAV: [string, string][] = [
  ["/", "Overview"],
  ["/datasets", "Datasets"],
  ["/training", "Training"],
  ["/jobs", "Jobs"],
  ["/experiments", "Experiments"],
  ["/evaluate", "Evaluate & Compare"],
  ["/explain", "Explain"],
  ["/robustness", "Robustness"],
  ["/benchmark", "Benchmark"],
];

export function Layout() {
  return (
    <div className="layout">
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <nav className="sidebar" aria-label="Main">
        <div className="brand">VisionBench</div>
        <ul>
          {NAV.map(([to, label]) => (
            <li key={to}>
              <NavLink to={to} end={to === "/"}>
                {label}
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>
      <main id="main" tabIndex={-1}>
        <Outlet />
        <footer className="footer">
          <p>
            VisionBench — local benchmarking. Every metric is labelled with its split: VALIDATION
            numbers are selection-biased (they chose the checkpoints); TEST numbers come from the
            single pre-registered final evaluation.
          </p>
          <p>{ATTRIBUTION}</p>
        </footer>
      </main>
    </div>
  );
}

export const routes: RouteObject[] = [
  {
    path: "/",
    element: <Layout />,
    children: [
      { index: true, element: <OverviewPage /> },
      { path: "datasets", element: <DatasetsPage /> },
      { path: "datasets/:datasetId", element: <DatasetDetailPage /> },
      { path: "training", element: <TrainingPage /> },
      { path: "jobs", element: <JobsPage /> },
      { path: "jobs/:jobId", element: <JobDetailPage /> },
      { path: "experiments", element: <ExperimentsPage /> },
      { path: "experiments/:experimentId", element: <ExperimentDetailPage /> },
      { path: "evaluate", element: <EvaluatePage /> },
      { path: "explain", element: <ExplainPage /> },
      { path: "robustness", element: <RobustnessPage /> },
      { path: "benchmark", element: <BenchmarkPage /> },
      { path: "*", element: <NotFoundPage /> },
    ],
  },
];

export function makeQueryClient() {
  return new QueryClient({
    defaultOptions: { queries: { retry: 1, refetchOnWindowFocus: false, staleTime: 5_000 } },
  });
}

export function App() {
  const router = createBrowserRouter(routes);
  return (
    <QueryClientProvider client={makeQueryClient()}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  );
}
