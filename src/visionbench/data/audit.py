"""Independent split audit: ``visionbench dataset audit-split <dataset_id> <split_id>``.

The split was built from duplicate groups (SHA-256 + dHash within a threshold, low-information
images excluded from perceptual matching). The audit does NOT trust that grouping. It decodes
every image of the split again and checks leakage with:

- an independent detector, the DCT pHash (``visionbench.data.phash``), computed for ALL images,
  low-information ones included;
- the dHash, recomputed over ALL images (low-information ones included and flagged);
- exact SHA-256 matches and decoded-pixel matches (SHA-256 of the decoded RGB bytes, which
  catches re-encoded identical images) across any two splits.

For every val and test image the nearest train image is found by pHash distance and by dHash
distance. The report contains:
- distance histograms (0..16, informative vs low-information query images);
- the consistency check "no informative val/test image is within the dHash threshold of an
  informative train image" (true by construction of the split);
- near-miss counts (threshold+1 .. threshold+3);
- group-size statistics;
- seeded samples of image pairs, written as ``pairs.csv`` and labelled contact sheets for
  visual inspection.

Outputs go to ``<data_root>/datasets/<id>/audits/<split_id>/`` (never into the write-once
split directory): ``audit.json``, ``audit.md``, ``pairs.csv``, ``sheets/*.png``. An existing
audit of the same split is replaced.
"""

from __future__ import annotations

import csv
import hashlib
import math
import re
import shutil
import uuid
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from visionbench import __version__
from visionbench.core.errors import SplitError
from visionbench.core.jsonio import read_json, write_json
from visionbench.core.seeding import derived_rng
from visionbench.data.grouping import is_low_information, low_information_reasons
from visionbench.data.hashing import HASH_VERSION, popcount64
from visionbench.data.imageio import ORIENTATION_POLICY, load_rgb
from visionbench.data.phash import PHASH_VERSION, phash
from visionbench.data.records import FileRecord
from visionbench.data.report import sample_relpaths
from visionbench.data.split import group_size_stats
from visionbench.data.store import dataset_dir, load_dataset, load_records

AUDIT_FORMAT_VERSION = 1
SPLIT_ID_RE = re.compile(r"^sp-[0-9a-f]{12}$")
HIST_MAX = 16
NEAR_MISS_WIDTH = 3
N_CLOSEST = 60
N_PER_DISTANCE = 40
N_NEAR_MISS = 40
N_LOW_INFO_PER_CLASS = 20
PER_SHEET = 20
TILE = 128
NO_MATCH = 255  # distance reported when there is nothing to compare against

Progress = Callable[[str], None]


@dataclass(frozen=True)
class AuditResult:
    path: Path
    audit: dict[str, Any]
    sheets: list[Path]


# --- per-image facts ------------------------------------------------------------------------


def pixel_digest(image: Image.Image) -> str:
    """SHA-256 of the decoded RGB pixels (and size): equal for re-encoded identical images."""
    rgb = image.convert("RGB")
    head = f"{rgb.width}x{rgb.height}\n".encode()
    return hashlib.sha256(head + rgb.tobytes()).hexdigest()


def image_facts(
    raw_root: Path, relpaths: list[str], progress: Progress | None = None
) -> tuple[np.ndarray, list[str]]:
    """(pHash per image as uint64, decoded-pixel digest per image), in ``relpaths`` order."""
    hashes = np.zeros(len(relpaths), dtype=np.uint64)
    digests: list[str] = []
    for i, rel in enumerate(relpaths):
        img = load_rgb(raw_root.joinpath(*rel.split("/")))
        hashes[i] = phash(img)
        digests.append(pixel_digest(img))
        if progress is not None and (i + 1) % 5000 == 0:
            progress(f"  decoded {i + 1}/{len(relpaths)} images")
    return hashes, digests


# --- distance search ------------------------------------------------------------------------


def nearest(
    queries: np.ndarray, refs: np.ndarray, *, chunk_elements: int = 1 << 22
) -> tuple[np.ndarray, np.ndarray]:
    """For each query hash: (Hamming distance to the nearest ref, index of that ref).

    Ties go to the lowest ref index, so results are deterministic. Without refs the distance
    is :data:`NO_MATCH` and the index -1.
    """
    q = np.ascontiguousarray(queries, dtype=np.uint64)
    r = np.ascontiguousarray(refs, dtype=np.uint64)
    dist = np.full(q.size, NO_MATCH, dtype=np.int64)
    idx = np.full(q.size, -1, dtype=np.int64)
    if r.size == 0 or q.size == 0:
        return dist, idx
    rows = max(1, chunk_elements // r.size)
    for start in range(0, q.size, rows):
        d = popcount64(q[start : start + rows, None] ^ r[None, :]).astype(np.int64)
        best = d.argmin(axis=1)
        idx[start : start + rows] = best
        dist[start : start + rows] = d[np.arange(best.size), best]
    return dist, idx


def pairs_in_band(
    queries: np.ndarray, refs: np.ndarray, lo: int, hi: int, *, chunk_elements: int = 1 << 22
) -> np.ndarray:
    """All (query index, ref index, distance) with ``lo <= distance <= hi``; int64 (k, 3)."""
    q = np.ascontiguousarray(queries, dtype=np.uint64)
    r = np.ascontiguousarray(refs, dtype=np.uint64)
    out: list[np.ndarray] = []
    if r.size and q.size:
        rows = max(1, chunk_elements // r.size)
        for start in range(0, q.size, rows):
            d = popcount64(q[start : start + rows, None] ^ r[None, :]).astype(np.int64)
            qi, ri = np.nonzero((d >= lo) & (d <= hi))
            if qi.size:
                out.append(np.stack([qi + start, ri, d[qi, ri]], axis=1))
    return np.concatenate(out) if out else np.empty((0, 3), dtype=np.int64)


def histogram(distances: np.ndarray) -> dict[str, int]:
    counts = {str(k): int((distances == k).sum()) for k in range(HIST_MAX + 1)}
    counts[f">{HIST_MAX}"] = int((distances > HIST_MAX).sum())
    return counts


# --- contact sheets -------------------------------------------------------------------------


def _font(size: int = 13) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    try:
        return ImageFont.load_default(size=size)
    except (TypeError, OSError):  # Pillow without FreeType
        return ImageFont.load_default()


def _tile(img: Image.Image, size: int = TILE) -> Image.Image:
    img = img.convert("RGB")
    scale = size / max(img.size)
    new = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
    resample = Image.Resampling.NEAREST if scale >= 1 else Image.Resampling.LANCZOS
    canvas = Image.new("RGB", (size, size), (235, 235, 235))
    canvas.paste(img.resize(new, resample), ((size - new[0]) // 2, (size - new[1]) // 2))
    return canvas


def _short(text: str, n: int = 40) -> str:
    return text if len(text) <= n else "…" + text[-(n - 1) :]


def render_pair_sheet(pairs: list[dict[str, Any]], raw_root: Path, path: Path, title: str) -> None:
    """Up to :data:`PER_SHEET` pairs, two columns; each pair: image A, image B, labels."""
    font, head_font = _font(13), _font(16)
    text_w, pad = 330, 10
    cell_w, cell_h = 2 * TILE + 3 * pad + text_w, TILE + 2 * pad
    cols = 2
    rows = math.ceil(len(pairs) / cols)
    sheet = Image.new("RGB", (cols * cell_w, 36 + rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((pad, 8), title, fill="black", font=head_font)
    for n, p in enumerate(pairs):
        x0, y0 = (n % cols) * cell_w, 36 + (n // cols) * cell_h
        for k, side in enumerate("ab"):
            img = load_rgb(raw_root.joinpath(*p[f"relpath_{side}"].split("/")))
            sheet.paste(_tile(img), (x0 + pad + k * (TILE + pad), y0 + pad))
        lines = [
            f"{p['pair_id']}   pHash {p['phash_distance']}   dHash {p['dhash_distance']}",
            f"A: {p['split_a']:<5} {p['class_a']}{'  [low-info]' if p['low_info_a'] else ''}",
            f"   {_short(p['relpath_a'])}",
            f"B: {p['split_b']:<5} {p['class_b']}{'  [low-info]' if p['low_info_b'] else ''}",
            f"   {_short(p['relpath_b'])}",
            f"same group: {'yes' if p['same_group'] else 'no'}",
        ]
        tx = x0 + 2 * TILE + 3 * pad
        for i, line in enumerate(lines):
            draw.text((tx, y0 + pad + i * 19), line, fill="black", font=font)
        draw.rectangle([x0, y0, x0 + cell_w - 1, y0 + cell_h - 1], outline=(200, 200, 200))
    sheet.save(path, format="PNG", optimize=True)


def render_image_sheet(items: list[dict[str, Any]], raw_root: Path, path: Path, title: str) -> None:
    """Single images (low-information samples), 5 per row, captioned."""
    font, head_font = _font(12), _font(16)
    pad, cols, caption = 10, 5, 34
    cell_w, cell_h = TILE + 2 * pad, TILE + caption + pad
    rows = math.ceil(len(items) / cols)
    sheet = Image.new("RGB", (cols * cell_w, 36 + rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((pad, 8), title, fill="black", font=head_font)
    for n, it in enumerate(items):
        x0, y0 = (n % cols) * cell_w + pad, 36 + (n // cols) * cell_h
        sheet.paste(_tile(load_rgb(raw_root.joinpath(*it["relpath"].split("/")))), (x0, y0))
        draw.text((x0, y0 + TILE + 2), _short(it["relpath"], 22), fill="black", font=font)
        draw.text(
            (x0, y0 + TILE + 17),
            f"{it['split']} {'/'.join(it['reasons'])}",
            fill="black",
            font=font,
        )
    sheet.save(path, format="PNG", optimize=True)


# --- the audit ------------------------------------------------------------------------------


def _low_info_reasons(r: FileRecord, cfg: dict[str, Any]) -> list[str]:
    return low_information_reasons(
        r, cfg["low_info_std_threshold"], cfg["low_info_popcount_min"], cfg["low_info_popcount_max"]
    )


def audit_split(
    datasets_dir: Path,
    dataset_id: str,
    split_id: str,
    *,
    seed: int = 0,
    progress: Progress | None = None,
) -> AuditResult:
    if not SPLIT_ID_RE.match(split_id):
        raise SplitError(f"invalid split id {split_id!r} (expected sp-<12 hex>)")
    meta = load_dataset(datasets_dir, dataset_id)
    root = dataset_dir(datasets_dir, dataset_id)
    split_dir = root / "splits" / split_id
    if not (split_dir / "manifest.json").is_file():
        raise SplitError(f"split {split_id} of dataset {dataset_id} not found")
    manifest = read_json(split_dir / "manifest.json")
    if manifest["dataset_fingerprint"] != meta["fingerprint"]:
        raise SplitError(f"split {split_id} was made for another version of {dataset_id}")
    cfg = manifest["config"]
    threshold = int(cfg["dhash_threshold"])
    records = {r.relpath: r for r in load_records(datasets_dir, dataset_id)}
    entries = manifest["entries"]  # sorted by relpath
    rel = [e["relpath"] for e in entries]
    split = np.array([e["split"] for e in entries])
    labels = [e["label"] for e in entries]
    groups = [e["group_id"] for e in entries]
    recs = [records[p] for p in rel]
    low = np.array([is_low_information_cfg(r, cfg) for r in recs], dtype=bool)
    dh = np.array([r.dhash_int for r in recs], dtype=np.uint64)

    if progress:
        progress(f"decoding {len(rel)} images (pHash + decoded-pixel digest)")
    ph, pixel = image_facts(root / "raw", rel, progress)

    train = np.flatnonzero(split == "train")
    query = np.flatnonzero(split != "train")  # val and test
    if train.size == 0:
        raise SplitError(f"split {split_id} has no train images; nothing to audit against")
    d_dist, d_idx = nearest(dh[query], dh[train])
    p_dist, p_idx = nearest(ph[query], ph[train])
    q_inf = query[~low[query]]
    t_inf = train[~low[train]]
    di_dist, _ = nearest(dh[q_inf], dh[t_inf])
    band = pairs_in_band(dh[q_inf], dh[t_inf], threshold + 1, threshold + NEAR_MISS_WIDTH)
    q_low = low[query]
    n_low = low[train][d_idx]

    def by_category(dist: np.ndarray) -> dict[str, dict[str, int]]:
        return {
            "informative": histogram(dist[~q_low]),
            "low_information": histogram(dist[q_low]),
        }

    within = d_dist <= threshold
    consistency = int((di_dist <= threshold).sum())

    def cross_split(values: list[str]) -> list[dict[str, Any]]:
        where: dict[str, list[int]] = defaultdict(list)
        for i, v in enumerate(values):
            where[v].append(i)
        out = []
        for _, idxs in sorted(where.items()):
            if len({split[i] for i in idxs}) > 1:
                out.append({"members": [{"relpath": rel[i], "split": str(split[i])} for i in idxs]})
        return out

    exact = cross_split([r.sha256 for r in recs])
    pixel_matches = cross_split(pixel)
    gstats = group_size_stats(manifest)

    def pair(set_name: str, n: int, a: int, b: int) -> dict[str, Any]:
        return {
            "pair_id": f"{set_name}-{n:03d}",
            "set": set_name,
            "relpath_a": rel[a],
            "class_a": labels[a],
            "split_a": str(split[a]),
            "low_info_a": bool(low[a]),
            "relpath_b": rel[b],
            "class_b": labels[b],
            "split_b": str(split[b]),
            "low_info_b": bool(low[b]),
            "phash_distance": int((int(ph[a]) ^ int(ph[b])).bit_count()),
            "dhash_distance": int((int(dh[a]) ^ int(dh[b])).bit_count()),
            "same_group": groups[a] == groups[b],
        }

    # (a) closest cross-split pairs by pHash (each val/test image with its nearest train image)
    order = sorted(
        range(query.size),
        key=lambda k: (int(p_dist[k]), (int(dh[query[k]]) ^ int(dh[train[p_idx[k]]])).bit_count(),
                       rel[query[k]]),
    )  # fmt: skip
    set_a = [
        pair("a", n + 1, int(query[k]), int(train[p_idx[k]]))
        for n, k in enumerate(order[:N_CLOSEST])
    ]

    # (b) within-group near-duplicate edges per dHash distance 1..threshold (seeded sample)
    members: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        if not low[i]:
            members[g].append(i)
    edges_by_d: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for idxs in members.values():
        if len(idxs) < 2:
            continue
        arr = np.array(idxs)
        d = popcount64(dh[arr][:, None] ^ dh[arr][None, :]).astype(np.int64)
        ii, jj = np.nonzero(np.triu((d >= 1) & (d <= threshold), k=1))
        for i, j in zip(ii.tolist(), jj.tolist(), strict=True):
            edges_by_d[int(d[i, j])].append((int(arr[i]), int(arr[j])))
    set_b: dict[int, list[dict[str, Any]]] = {}
    for dist_value in range(1, threshold + 1):
        edges = sorted(edges_by_d.get(dist_value, []))
        if len(edges) > N_PER_DISTANCE:
            pick = derived_rng(seed, f"within-group-d{dist_value}").choice(
                len(edges), size=N_PER_DISTANCE, replace=False
            )
            edges = [edges[k] for k in sorted(pick)]
        set_b[dist_value] = [pair(f"b{dist_value}", n + 1, a, b) for n, (a, b) in enumerate(edges)]

    # (c) random cross-split pairs from the near-miss band (informative images only)
    band_rows = band.tolist()
    if len(band_rows) > N_NEAR_MISS:
        pick = derived_rng(seed, "near-miss").choice(len(band_rows), N_NEAR_MISS, replace=False)
        band_rows = [band_rows[k] for k in sorted(pick)]
    set_c = [
        pair("c", n + 1, int(q_inf[qi]), int(t_inf[ri])) for n, (qi, ri, _) in enumerate(band_rows)
    ]

    # (d) low-information samples per class
    set_d: dict[str, list[dict[str, Any]]] = {}
    pos = {p: i for i, p in enumerate(rel)}
    for label in sorted(set(labels)):
        low_rel = [rel[i] for i in range(len(rel)) if low[i] and labels[i] == label]
        chosen = sample_relpaths(low_rel, N_LOW_INFO_PER_CLASS, seed, f"low-info|{label}")
        set_d[label] = [
            {
                "relpath": p,
                "split": str(split[pos[p]]),
                "reasons": _low_info_reasons(records[p], cfg),
            }
            for p in chosen
        ]

    audit: dict[str, Any] = {
        "format_version": AUDIT_FORMAT_VERSION,
        "visionbench_version": __version__,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset_id": dataset_id,
        "split_id": split_id,
        "split_config": cfg,
        "seed": seed,
        "methods": {
            "orientation": ORIENTATION_POLICY,
            "dhash": f"hash_version {HASH_VERSION} (as recorded in files.json)",
            "phash": PHASH_VERSION,
            "pixel_digest": "sha256 of decoded RGB bytes (with size)",
            "nearest_neighbour": "each val/test image vs all train images (low-information "
            "images included and flagged); ties -> first train image by relpath",
        },
        "counts": {
            "entries": len(rel),
            "train": int(train.size),
            "val": int((split == "val").sum()),
            "test": int((split == "test").sum()),
            "excluded_label_conflicts": len(manifest["excluded"]),
            "low_information": int(low.sum()),
            "low_information_val_test": int(q_low.sum()),
        },
        "nearest_train_histograms": {
            "dhash": by_category(d_dist),
            "phash": by_category(p_dist),
        },
        "within_dhash_threshold": {
            "threshold": threshold,
            "consistency_check": {
                "description": "informative val/test images whose nearest INFORMATIVE train "
                "image is within the dHash threshold; 0 by construction of the split",
                "count": consistency,
                "ok": consistency == 0,
            },
            "any_nearest_train_within_threshold": int(within.sum()),
            "by_low_information": {
                "query_informative_neighbour_informative": int((within & ~q_low & ~n_low).sum()),
                "query_informative_neighbour_low_info": int((within & ~q_low & n_low).sum()),
                "query_low_info_neighbour_informative": int((within & q_low & ~n_low).sum()),
                "query_low_info_neighbour_low_info": int((within & q_low & n_low).sum()),
            },
            "phash_nearest_within_same_value": int((p_dist <= threshold).sum()),
        },
        "near_miss_band": {
            "range": [threshold + 1, threshold + NEAR_MISS_WIDTH],
            "informative_queries_with_nearest_informative_train_in_band": int(
                ((di_dist > threshold) & (di_dist <= threshold + NEAR_MISS_WIDTH)).sum()
            ),
            "informative_pairs_in_band": int(len(band)),
            "informative_pairs_in_band_by_distance": dict(
                sorted(Counter(str(int(x)) for x in band[:, 2]).items())
            ),
        },
        "cross_split_exact_sha256": {"count": len(exact), "items": exact},
        "cross_split_pixel_identical": {"count": len(pixel_matches), "items": pixel_matches},
        "groups": gstats,
        "within_group_pairs_by_dhash_distance": {
            str(k): len(edges_by_d.get(k, [])) for k in range(1, threshold + 1)
        },
    }

    pairs: list[dict[str, Any]] = [*set_a]
    for k in range(1, threshold + 1):
        pairs.extend(set_b[k])
    pairs.extend(set_c)
    return _write(root, split_id, audit, pairs, set_a, set_b, set_c, set_d, progress)


def is_low_information_cfg(r: FileRecord, cfg: dict[str, Any]) -> bool:
    return is_low_information(
        r, cfg["low_info_std_threshold"], cfg["low_info_popcount_min"], cfg["low_info_popcount_max"]
    )


def _write(
    root: Path,
    split_id: str,
    audit: dict[str, Any],
    pairs: list[dict[str, Any]],
    set_a: list[dict[str, Any]],
    set_b: dict[int, list[dict[str, Any]]],
    set_c: list[dict[str, Any]],
    set_d: dict[str, list[dict[str, Any]]],
    progress: Progress | None,
) -> AuditResult:
    audits = root / "audits"
    audits.mkdir(exist_ok=True)
    staging = audits / f".tmp-audit-{uuid.uuid4().hex}"
    (staging / "sheets").mkdir(parents=True)
    raw = root / "raw"
    sheets: list[str] = []

    def chunks(items: list[dict[str, Any]], stem: str, title: str) -> None:
        for n in range(0, len(items), PER_SHEET):
            part = items[n : n + PER_SHEET]
            name = f"{stem}_{n // PER_SHEET + 1:02d}.png"
            for p in part:
                p["sheet"] = f"sheets/{name}"
            render_pair_sheet(part, raw, staging / "sheets" / name, f"{title} ({name})")
            sheets.append(f"sheets/{name}")

    try:
        if progress:
            progress("rendering contact sheets")
        thr = audit["within_dhash_threshold"]["threshold"]
        chunks(set_a, "a_closest_phash", f"{split_id}: closest cross-split pairs by pHash")
        for k, items in set_b.items():
            chunks(items, f"b_within_group_d{k}", f"{split_id}: within-group pairs, dHash = {k}")
        lo, hi = thr + 1, thr + NEAR_MISS_WIDTH
        chunks(set_c, "c_near_miss", f"{split_id}: cross-split near misses, dHash {lo}..{hi}")
        for label, items in set_d.items():
            if items:
                name = f"d_low_info_{_safe(label)}.png"
                render_image_sheet(
                    items, raw, staging / "sheets" / name, f"{split_id}: low-information, {label}"
                )
                sheets.append(f"sheets/{name}")

        audit["pairs"] = {
            "a_closest_phash": len(set_a),
            "b_within_group_by_distance": {str(k): len(v) for k, v in set_b.items()},
            "c_near_miss": len(set_c),
            "d_low_info_samples": {k: len(v) for k, v in set_d.items()},
        }
        audit["low_information_samples"] = set_d
        audit["sheets"] = sheets
        fields = [
            "pair_id",
            "set",
            "relpath_a",
            "class_a",
            "split_a",
            "low_info_a",
            "relpath_b",
            "class_b",
            "split_b",
            "low_info_b",
            "phash_distance",
            "dhash_distance",
            "same_group",
            "sheet",
        ]
        with open(staging / "pairs.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(pairs)
        write_json(staging / "audit.json", audit)
        (staging / "audit.md").write_text(render_markdown(audit), encoding="utf-8", newline="\n")
        final = audits / split_id
        if final.exists():
            shutil.rmtree(final)
        staging.rename(final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return AuditResult(final, audit, [final / s for s in sheets])


def _safe(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", label)


def _hist_table(h: dict[str, dict[str, int]]) -> list[str]:
    keys = [*map(str, range(HIST_MAX + 1)), f">{HIST_MAX}"]
    lines = ["| query images | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for cat, counts in h.items():
        lines.append(f"| {cat} | " + " | ".join(str(counts[k]) for k in keys) + " |")
    return lines


def render_markdown(a: dict[str, Any]) -> str:
    c, w, nm = a["counts"], a["within_dhash_threshold"], a["near_miss_band"]
    g = a["groups"]
    cfg = a["split_config"]
    verdict = "OK" if w["consistency_check"]["ok"] else "FAILED"
    out = [
        f"# Split audit: {a['dataset_id']} / {a['split_id']}",
        "",
        f"Created {a['created_at']} by VisionBench {a['visionbench_version']}; seed {a['seed']}.",
        f"Split config: dHash threshold {cfg['dhash_threshold']}, low-info std "
        f"{cfg['low_info_std_threshold']}, popcount [{cfg['low_info_popcount_min']}, "
        f"{cfg['low_info_popcount_max']}], fractions {cfg['fractions']}, seed {cfg['seed']}.",
        f"Methods: pHash {a['methods']['phash']} (independent of the grouping), dHash "
        f"(recomputed over all images), orientation {a['methods']['orientation']}.",
        "",
        "## Counts",
        "",
        f"- entries {c['entries']} (train {c['train']}, val {c['val']}, test {c['test']}); "
        f"excluded label-conflict files {c['excluded_label_conflicts']}",
        f"- low-information images {c['low_information']} (of which val/test "
        f"{c['low_information_val_test']})",
        "",
        "## Nearest train image of each val/test image",
        "",
        "dHash distance:",
        "",
        *_hist_table(a["nearest_train_histograms"]["dhash"]),
        "",
        "pHash distance:",
        "",
        *_hist_table(a["nearest_train_histograms"]["phash"]),
        "",
        f"## Within the dHash threshold ({w['threshold']})",
        "",
        f"- consistency check (informative vs informative, must be 0): "
        f"**{w['consistency_check']['count']}** -> {verdict}",
        f"- val/test images whose nearest train image (low-info included) is within the "
        f"threshold: {w['any_nearest_train_within_threshold']}",
    ]
    for k, v in w["by_low_information"].items():
        out.append(f"  - {k.replace('_', ' ')}: {v}")
    out += [
        f"- val/test images whose nearest pHash train distance is <= {w['threshold']}: "
        f"{w['phash_nearest_within_same_value']}",
        "",
        f"## Near-miss band (dHash {nm['range'][0]}..{nm['range'][1]}, informative images)",
        "",
        f"- val/test images whose nearest informative train image is in the band: "
        f"{nm['informative_queries_with_nearest_informative_train_in_band']}",
        f"- all cross-split (val/test x train) pairs in the band: {nm['informative_pairs_in_band']}"
        f" (by distance: {nm['informative_pairs_in_band_by_distance']})",
        "",
        "## Exact and pixel-identical matches across splits",
        "",
        f"- exact SHA-256 matches across splits: **{a['cross_split_exact_sha256']['count']}** "
        "(must be 0)",
        f"- decoded-pixel-identical matches across splits: "
        f"**{a['cross_split_pixel_identical']['count']}**",
        "",
        "## Groups",
        "",
        "Group-size distribution (size: groups): "
        + ", ".join(f"{k}: {v}" for k, v in g["size_distribution"].items()),
        "",
        "| class | images | largest group | val target | test target |",
        "|---|---|---|---|---|",
    ]
    for label, s in g["per_class"].items():
        out.append(
            f"| {label} | {s['class_total']} | {s['largest_size']} | {s['val_target']} | "
            f"{s['test_target']} |"
        )
    out += ["", f"Oversized groups (> val target or > 2% of the class): {len(g['oversized'])}"]
    for o in g["oversized"][:50]:
        out.append(
            f"- {o['label']} {o['group_id']}: {o['size']} images ({o['share_of_class']:.1%} of "
            f"{o['class_total']}; val target {o['val_target']}; {', '.join(o['reasons'])})"
        )
    if len(g["oversized"]) > 50:
        out.append(f"- ... {len(g['oversized']) - 50} more in audit.json")
    out += [
        "",
        "## Within-group pairs by dHash distance (informative members)",
        "",
        ", ".join(f"d={k}: {v}" for k, v in a["within_group_pairs_by_dhash_distance"].items())
        or "(threshold 0: none)",
        "",
        "## Pairs for visual inspection",
        "",
        f"- (a) closest cross-split pairs by pHash: {a['pairs']['a_closest_phash']}",
        f"- (b) within-group pairs per dHash distance: {a['pairs']['b_within_group_by_distance']}",
        f"- (c) near-miss cross-split pairs: {a['pairs']['c_near_miss']}",
        f"- (d) low-information samples per class: {a['pairs']['d_low_info_samples']}",
        "",
        "Contact sheets: " + ", ".join(a["sheets"]),
        "",
    ]
    return "\n".join(out)
