import { useState } from "react";
import type { FormEvent } from "react";
import { Link } from "react-router-dom";

import { useDatasets, useUploadDataset } from "../api/hooks";
import { JobPanel } from "../components/JobPanel";
import { ErrorBox, PageHeader, QueryState, Section } from "../components/ui";

export function DatasetsPage() {
  const datasets = useDatasets();
  return (
    <>
      <PageHeader title="Datasets">
        Imported datasets are immutable snapshots. Upload a ZIP of class folders to import a new
        one; the import runs as a job.
      </PageHeader>
      <Section title="Datasets" id="dataset-list">
        <QueryState
          query={datasets}
          what="datasets"
          isEmpty={(d) => d.length === 0}
          empty="No datasets yet. Upload a ZIP below."
        >
          {(list) => (
            <table>
              <caption>Imported datasets</caption>
              <thead>
                <tr>
                  <th scope="col">name</th>
                  <th scope="col">id</th>
                  <th scope="col" className="num">
                    images
                  </th>
                  <th scope="col" className="num">
                    classes
                  </th>
                  <th scope="col">created</th>
                </tr>
              </thead>
              <tbody>
                {list.map((d) => (
                  <tr key={d.id}>
                    <td>
                      <Link to={`/datasets/${d.id}`}>{d.name}</Link>
                    </td>
                    <td className="mono">{d.id}</td>
                    <td className="num">{d.total_images}</td>
                    <td className="num">{d.classes.length}</td>
                    <td>{d.created_at}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </QueryState>
      </Section>
      <UploadForm />
    </>
  );
}

function UploadForm() {
  const uploadMut = useUploadDataset();
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState("");
  const [progress, setProgress] = useState<number | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [touched, setTouched] = useState(false);
  const nameError = !name.trim()
    ? "A name is required."
    : name.length > 100
      ? "At most 100 characters."
      : null;
  const fileError = !file
    ? "Choose a ZIP file."
    : !file.name.toLowerCase().endsWith(".zip")
      ? "The file must be a .zip archive."
      : null;

  function submit(e: FormEvent) {
    e.preventDefault();
    setTouched(true);
    if (nameError || fileError || !file) return;
    setProgress(0);
    uploadMut.mutate(
      { file, name: name.trim(), onProgress: setProgress },
      { onSuccess: (r) => setJobId(r.job.id) },
    );
  }

  return (
    <Section title="Upload a dataset (ZIP)" id="upload">
      <form onSubmit={submit} noValidate>
        <p className="hint">
          The ZIP must contain one folder per class (at its root or inside one top folder). The
          archive is checked before the import job is queued.
        </p>
        <div className="field-row">
          <div className="field">
            <label htmlFor="ds-name">Dataset name</label>
            <input
              id="ds-name"
              value={name}
              maxLength={100}
              onChange={(e) => setName(e.target.value)}
              aria-invalid={touched && Boolean(nameError)}
              aria-describedby="ds-name-err"
            />
            {touched && nameError ? (
              <span id="ds-name-err" className="field-error">
                {nameError}
              </span>
            ) : null}
          </div>
          <div className="field">
            <label htmlFor="ds-file">ZIP file</label>
            <input
              id="ds-file"
              type="file"
              accept=".zip,application/zip"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              aria-invalid={touched && Boolean(fileError)}
              aria-describedby="ds-file-err"
            />
            {touched && fileError ? (
              <span id="ds-file-err" className="field-error">
                {fileError}
              </span>
            ) : null}
          </div>
        </div>
        <div className="actions">
          <button type="submit" disabled={uploadMut.isPending}>
            {uploadMut.isPending ? "Uploading…" : "Upload and import"}
          </button>
          {progress !== null && uploadMut.isPending ? (
            <progress className="progress" value={progress} max={1} aria-label="Upload progress" />
          ) : null}
        </div>
      </form>
      {uploadMut.isError ? <ErrorBox error={uploadMut.error} title="Upload refused" /> : null}
      {jobId ? (
        <>
          <h3>Import job</h3>
          <JobPanel jobId={jobId} />
        </>
      ) : null}
    </Section>
  );
}
