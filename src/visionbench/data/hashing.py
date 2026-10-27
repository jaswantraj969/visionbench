"""64-bit difference hash (dHash) and vectorized near-duplicate search.

dHash: convert to grayscale, resize to 9x8 (width x height), and for each of the 8 rows emit
8 bits ``pixel[x + 1] > pixel[x]``. Bits are packed row-major, most significant bit first.

:data:`HASH_VERSION` identifies what the stored hashes were computed from: 1 (M1) the stored
pixel orientation; 2 (D1) the EXIF-transposed image (``visionbench.data.imageio``).
"""

from __future__ import annotations

import numpy as np
from PIL import Image

HASH_VERSION = 2
HASH_BITS = 64
ALL_ONES = (1 << HASH_BITS) - 1

_POPCOUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
HAS_BITWISE_COUNT = hasattr(np, "bitwise_count")  # NumPy >= 2.0


def dhash_from_gray(gray: np.ndarray) -> int:
    """dHash of an already-resized 8x9 (rows x cols) grayscale array."""
    a = np.asarray(gray)
    if a.shape != (8, 9):
        raise ValueError(f"expected an 8x9 grayscale array, got shape {a.shape}")
    a = a.astype(np.int32)
    bits = a[:, 1:] > a[:, :-1]
    return int.from_bytes(np.packbits(bits.ravel()).tobytes(), "big")


def dhash(image: Image.Image) -> int:
    gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    return dhash_from_gray(np.asarray(gray))


def format_hash(value: int) -> str:
    return f"{value:016x}"


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def popcount64(values: np.ndarray, *, use_lut: bool = False) -> np.ndarray:
    """Per-element popcount of a uint64 array (``np.bitwise_count`` or a byte lookup table)."""
    x = np.ascontiguousarray(values, dtype=np.uint64)
    if HAS_BITWISE_COUNT and not use_lut:
        return np.bitwise_count(x)
    return _POPCOUNT_LUT[x.view(np.uint8)].reshape(*x.shape, 8).sum(axis=-1, dtype=np.uint8)


def near_duplicate_edges(
    hashes: np.ndarray,
    threshold: int,
    *,
    chunk_elements: int = 1 << 23,
    use_lut: bool = False,
) -> np.ndarray:
    """Edges ``(i, j)``, i < j, whose connected components equal those of the graph
    "Hamming distance <= threshold".

    Identical hashes are collapsed first (each extra copy gets one edge to its first occurrence),
    then distinct values are compared pairwise in row chunks of about ``chunk_elements`` XORs, so
    memory stays bounded. Returns an int64 array of shape (k, 2).
    """
    if not 0 <= threshold <= HASH_BITS:
        raise ValueError(f"threshold must be in [0, {HASH_BITS}], got {threshold}")
    h = np.ascontiguousarray(hashes, dtype=np.uint64).ravel()
    if h.size < 2:
        return np.empty((0, 2), dtype=np.int64)

    uniq, first_idx, inverse = np.unique(h, return_index=True, return_inverse=True)
    edges: list[np.ndarray] = []

    # Copies of an identical hash: link each to the first occurrence of its value.
    dup = np.flatnonzero(first_idx[inverse] != np.arange(h.size))
    if dup.size:
        edges.append(np.stack([first_idx[inverse[dup]], dup], axis=1))

    n = uniq.size
    if threshold > 0 and n > 1:
        rows_per_chunk = max(1, chunk_elements // n)
        for start in range(0, n, rows_per_chunk):
            stop = min(n, start + rows_per_chunk)
            # Compare rows [start, stop) with columns [start, n); keep only col > row.
            dist = popcount64(uniq[start:stop, None] ^ uniq[None, start:], use_lut=use_lut)
            rows = np.arange(start, stop)[:, None]
            cols = np.arange(start, n)[None, :]
            r, c = np.nonzero((dist <= threshold) & (cols > rows))
            if r.size:
                edges.append(np.stack([first_idx[r + start], first_idx[c + start]], axis=1))

    if not edges:
        return np.empty((0, 2), dtype=np.int64)
    out = np.concatenate(edges).astype(np.int64)
    return np.sort(out, axis=1)
