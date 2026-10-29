import { Section, SplitLabel } from "../components/ui";

/** The TEST section of the Benchmark page. Before the single final evaluation there is no TEST
 * result, and the page says so instead of showing anything that looks like one. */
export function FinalReportSection({ report }: { report: Record<string, unknown> | null }) {
  if (!report) {
    return (
      <Section title="TEST results" id="test-results">
        <p>
          <SplitLabel split="test" /> Not evaluated yet. The test split is evaluated exactly once,
          for all runs, at the final release, following the committed final analysis plan.
        </p>
      </Section>
    );
  }
  return (
    <Section title="TEST results" id="test-results">
      <p>
        <SplitLabel split="test" /> {String(report.label ?? "TEST")} — see the final report document
        for every table and the limitations.
      </p>
      <pre className="json">{JSON.stringify(report.summary ?? report, null, 2)}</pre>
    </Section>
  );
}
