import itertools
import time

import numpy as np
import pytest
from PIL import Image

from visionbench.data.hashing import (
    ALL_ONES,
    HAS_BITWISE_COUNT,
    dhash,
    dhash_from_gray,
    format_hash,
    hamming,
    near_duplicate_edges,
    popcount64,
)


def test_dhash_constant_rows_is_zero() -> None:
    assert dhash_from_gray(np.full((8, 9), 128, dtype=np.uint8)) == 0


def test_dhash_increasing_rows_is_all_ones() -> None:
    gray = np.tile(np.arange(9, dtype=np.uint8) * 10, (8, 1))
    assert dhash_from_gray(gray) == ALL_ONES


def test_dhash_bit_order_is_row_major_msb_first() -> None:
    gray = np.zeros((8, 9), dtype=np.uint8)
    gray[0] = np.arange(9) * 10  # first row increasing -> top byte set
    assert dhash_from_gray(gray) == 0xFF00_0000_0000_0000
    gray = np.zeros((8, 9), dtype=np.uint8)
    gray[7, 8] = 255  # last comparison of the last row -> lowest bit
    assert dhash_from_gray(gray) == 1
    gray = np.zeros((8, 9), dtype=np.uint8)
    gray[0, 1] = 255  # first comparison of the first row -> highest bit
    assert dhash_from_gray(gray) == 1 << 63


def test_dhash_alternating_pattern() -> None:
    row = np.array([0, 9, 0, 9, 0, 9, 0, 9, 0], dtype=np.uint8)  # up, down, up, ...
    assert dhash_from_gray(np.tile(row, (8, 1))) == int("10101010" * 8, 2)


def test_dhash_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="8x9"):
        dhash_from_gray(np.zeros((8, 8)))


def test_dhash_of_image_matches_manual_resize() -> None:
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 256, (40, 50), dtype=np.uint8))
    manual = np.asarray(img.resize((9, 8), Image.Resampling.LANCZOS))
    assert dhash(img) == dhash_from_gray(manual)
    assert dhash(Image.new("RGB", (30, 30), (5, 200, 90))) == 0


def test_format_and_hamming() -> None:
    assert format_hash(255) == "00000000000000ff"
    assert hamming(0, ALL_ONES) == 64
    assert hamming(0b1011, 0b0001) == 2


def test_popcount_lut_matches_builtin() -> None:
    rng = np.random.default_rng(1)
    x = np.frombuffer(rng.bytes(8 * 1000), dtype=np.uint64)
    expected = np.array([int(v).bit_count() for v in x])
    assert np.array_equal(popcount64(x, use_lut=True), expected)
    assert np.array_equal(popcount64(x), expected)
    assert np.array_equal(popcount64(x.reshape(10, 100), use_lut=True), expected.reshape(10, 100))


def _components(n: int, edges: np.ndarray) -> set[frozenset[int]]:
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            i = parent[i]
        return i

    for a, b in edges:
        parent[find(int(a))] = find(int(b))
    comps: dict[int, set[int]] = {}
    for i in range(n):
        comps.setdefault(find(i), set()).add(i)
    return {frozenset(c) for c in comps.values()}


@pytest.mark.parametrize("use_lut", [False, True])
@pytest.mark.parametrize("chunk_elements", [7, 1 << 23])
def test_near_duplicate_edges_matches_brute_force(use_lut: bool, chunk_elements: int) -> None:
    rng = np.random.default_rng(2)
    base = [int(v) for v in np.frombuffer(rng.bytes(8 * 30), dtype=np.uint64)]
    hashes = list(base)
    for i, flips in enumerate([0, 1, 3, 5, 6, 12]):  # planted variants at known distances
        h = base[i]
        for bit in rng.choice(64, size=flips, replace=False):
            h ^= 1 << int(bit)
        hashes.append(h)
    arr = np.array(hashes, dtype=np.uint64)
    threshold = 5
    edges = near_duplicate_edges(arr, threshold, chunk_elements=chunk_elements, use_lut=use_lut)
    assert edges.dtype == np.int64 and (edges[:, 0] < edges[:, 1]).all()
    brute = np.array(
        [
            (i, j)
            for i, j in itertools.combinations(range(len(hashes)), 2)
            if hamming(hashes[i], hashes[j]) <= threshold
        ],
        dtype=np.int64,
    ).reshape(-1, 2)
    assert _components(len(hashes), edges) == _components(len(hashes), brute)
    # the planted 0/1/3/5-bit variants are linked; the 6- and 12-bit ones are not
    comps = _components(len(hashes), edges)
    for i, linked in enumerate([True, True, True, True, False, False]):
        assert (frozenset({i, 30 + i}) <= next(c for c in comps if i in c)) is linked


def test_near_duplicate_edges_threshold_zero_and_small_inputs() -> None:
    arr = np.array([5, 5, 7, 5], dtype=np.uint64)
    assert _components(4, near_duplicate_edges(arr, 0)) == {frozenset({0, 1, 3}), frozenset({2})}
    assert near_duplicate_edges(np.array([1], dtype=np.uint64), 5).shape == (0, 2)
    assert near_duplicate_edges(np.array([1, 1 << 40], dtype=np.uint64), 0).shape == (0, 2)
    with pytest.raises(ValueError):
        near_duplicate_edges(arr, 65)


@pytest.mark.parametrize("use_lut", [False, True])
def test_near_duplicate_search_20k_performance(use_lut: bool, report_line) -> None:
    if not use_lut and not HAS_BITWISE_COUNT:
        pytest.skip("np.bitwise_count unavailable (NumPy < 2.0)")
    rng = np.random.default_rng(3)
    hashes = np.frombuffer(rng.bytes(8 * 20_000), dtype=np.uint64).copy()
    hashes[19_999] = hashes[0] ^ np.uint64(0b10101)  # planted pair at distance 3
    start = time.perf_counter()
    edges = near_duplicate_edges(hashes, 5, use_lut=use_lut)
    elapsed = time.perf_counter() - start
    method = "byte LUT" if use_lut else "np.bitwise_count"
    report_line(
        f"PERF near-duplicate search, 20,000 random hashes, threshold 5, {method}: "
        f"{elapsed:.2f} s, {len(edges)} edge(s)"
    )
    assert [0, 19_999] in edges.tolist()
    assert elapsed < 60
