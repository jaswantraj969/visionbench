// Number formatting. Missing or undefined values are never shown as 0.

export function num(value: number | null | undefined, digits = 4): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "undefined";
  return value.toFixed(digits);
}

export function pct(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "undefined";
  return `${(value * 100).toFixed(digits)} %`;
}

export function meanSd(mean: number | null | undefined, sd: number | null | undefined): string {
  if (mean === null || mean === undefined) return "undefined";
  return sd === null || sd === undefined ? num(mean) : `${num(mean)} ± ${num(sd)}`;
}

export function ci(
  point: number | null | undefined,
  low: number | null | undefined,
  high: number | null | undefined,
  digits = 4,
): string {
  if (point === null || point === undefined) return "undefined";
  if (low === null || low === undefined || high === null || high === undefined) {
    return `${num(point, digits)} [CI undefined]`;
  }
  return `${num(point, digits)} [${num(low, digits)}, ${num(high, digits)}]`;
}

export function signed(value: number | null | undefined, digits = 4): string {
  if (value === null || value === undefined) return "undefined";
  return `${value >= 0 ? "+" : ""}${value.toFixed(digits)}`;
}

export function seconds(value: number | null | undefined): string {
  if (value === null || value === undefined) return "undefined";
  if (value < 60) return `${value.toFixed(1)} s`;
  return `${(value / 60).toFixed(1)} min`;
}

export function bytes(value: number | null | undefined): string {
  if (value === null || value === undefined) return "undefined";
  if (value < 1024) return `${value} B`;
  if (value < 1024 ** 2) return `${(value / 1024).toFixed(1)} KiB`;
  if (value < 1024 ** 3) return `${(value / 1024 ** 2).toFixed(1)} MiB`;
  return `${(value / 1024 ** 3).toFixed(2)} GiB`;
}

export function shortSha(sha: string | null | undefined): string {
  return sha ? `${sha.slice(0, 12)}…` : "none";
}
