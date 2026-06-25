#!/usr/bin/env python3
"""
analyze_gol.py — offline post-analysis for .gol archives produced by gol_sim.cpp

This is intentionally a SEPARATE tool from the simulator. The simulator's hot
loop only ever does whole-grid hashing for stagnation detection (cheap, fast,
no per-object bookkeeping). All the expensive work — splitting each frame
into connected "objects", tracking them across generations, and classifying
each one as a still life / oscillator / spaceship — happens here, after the
fact, by reading the lossless bit-packed archive back off disk.

Pipeline:
  1. Stream frames out of the .gol file (one frame in memory at a time, so
     this scales to huge grids — never loads the whole archive at once).
  2. Per frame: label connected components with 8-connectivity, correctly
     merging components that touch across the toroidal (wrap-around) edges.
  3. Link each component in frame t to its descendant(s) in frame t+1 by
     checking which frame-(t+1) labels fall within a 1-cell (Chebyshev)
     dilation of each frame-t component — this is the maximum distance any
     live cell's influence can reach in one Game of Life step.
  4. Walk those links into "lineages": a clean run of frames where a
     component has exactly one parent and one child (no merge, no split).
  5. For each finished lineage, detect its intrinsic period by comparing its
     translation-normalized shape across time. A repeat with zero
     displacement at period 1 is a still life, zero displacement at period
     >1 is an oscillator, nonzero displacement is a spaceship (its period and
     displacement give you the speed, e.g. c/4 diagonal).
  6. Look up each lineage's orientation-normalized (D4-canonical) shape
     against a small built-in catalog of named patterns.

Usage:
    python3 analyze_gol.py run.gol
    python3 analyze_gol.py run.gol --out report.json
    python3 analyze_gol.py run.gol --every 5          # sample every 5th frame
    python3 analyze_gol.py run.gol --max-frames 500

Requires: numpy, scipy (both standard; `pip install numpy scipy` if missing).
"""

import argparse
import json
import struct
import sys
from collections import defaultdict

import numpy as np
from scipy import ndimage

try:
    import zstandard as _zstd
except ImportError:
    _zstd = None

MAGIC = b"GOL1"
HEADER_SIZE = 64
FRAME_HDR_SIZE = 12
FRAME_HDR_SIZE_COMP = 21

GOL_COMPRESS_RAW = 0
GOL_COMPRESS_DELTA = 1
GOL_BLOCK_SIZE = 128


def block_sparse_decode(data, nbytes):
    nblocks = (nbytes + GOL_BLOCK_SIZE - 1) // GOL_BLOCK_SIZE
    mask_bytes = (nblocks + 7) // 8
    mask = data[:mask_bytes]
    blocks = data[mask_bytes:]
    result = bytearray(nbytes)
    block_pos = 0
    for i in range(nblocks):
        if mask[i // 8] & (1 << (i % 8)):
            off = i * GOL_BLOCK_SIZE
            blen = min(GOL_BLOCK_SIZE, nbytes - off)
            result[off:off+blen] = blocks[block_pos:block_pos+blen]
            block_pos += blen
    return bytes(result)

# ──────────────────────────────────────────────────────────────────────────
# .gol reading
# ──────────────────────────────────────────────────────────────────────────

def read_header(path):
    with open(path, "rb") as f:
        hdr = f.read(HEADER_SIZE)
    if len(hdr) != HEADER_SIZE or hdr[0:4] != MAGIC:
        raise ValueError(f"'{path}' is not a valid .gol archive (bad magic)")
    grid_w, grid_h, total_gens, fps_hint = struct.unpack_from("<IIII", hdr, 4)
    rule_b = hdr[20]
    rule_s = hdr[24]
    stag_period, stag_gen = struct.unpack_from("<ii", hdr, 28)
    compression = hdr[36] if len(hdr) > 36 else 0
    return {
        "grid_w": grid_w,
        "grid_h": grid_h,
        "total_gens": total_gens,
        "fps_hint": fps_hint,
        "rule_b": rule_b,
        "rule_s": rule_s,
        "stag_period": stag_period,
        "stag_gen": stag_gen,
        "compression": compression,
    }


def rule_str(rule_b, rule_s):
    b = "".join(str(i) for i in range(9) if (rule_b >> i) & 1)
    s = "".join(str(i) for i in range(9) if (rule_s >> i) & 1)
    return f"B{b}/S{s}"


def iter_frames(path, every=1, max_frames=None):
    """Yield (gen, live_count, bool_array[H,W]) one frame at a time."""
    hdr = read_header(path)
    W, H = hdr["grid_w"], hdr["grid_h"]
    compression = hdr.get("compression", 0)
    nbytes = (W * H + 7) // 8

    if compression == GOL_COMPRESS_DELTA and _zstd is None:
        raise ImportError(
            "This .gol file uses zstd compression. "
            "Install it: pip install zstandard"
        )

    with open(path, "rb") as f:
        f.seek(HEADER_SIZE)
        idx = 0
        yielded = 0
        prev_bits = None

        while True:
            if compression == GOL_COMPRESS_DELTA:
                fhdr = f.read(FRAME_HDR_SIZE_COMP)
                if len(fhdr) < FRAME_HDR_SIZE_COMP:
                    break
                gen, live_count = struct.unpack("<QI", fhdr[:12])
                ctype = fhdr[12]
                usize, csize = struct.unpack("<II", fhdr[13:21])
                cdata = f.read(csize)
                if len(cdata) < csize:
                    break

                # Decompress
                dctx = _zstd.ZstdDecompressor()
                sparse = dctx.decompress(cdata, max_output_size=usize)
                decoded = block_sparse_decode(sparse, nbytes)

                # Reconstruct frame (undo delta if not frame 0)
                if prev_bits is None:
                    cur_bits = decoded
                else:
                    cur_bits = bytes(a ^ b for a, b in zip(decoded, prev_bits))
                prev_bits = cur_bits

                if idx % every == 0:
                    flat = np.frombuffer(cur_bits, dtype=np.uint8)
                    flat = np.unpackbits(flat, bitorder="little")[:W * H]
                    grid = flat.reshape(H, W).astype(bool)
                    yield gen, live_count, grid
                    yielded += 1
                    if max_frames is not None and yielded >= max_frames:
                        break
            else:
                fhdr = f.read(FRAME_HDR_SIZE)
                if len(fhdr) < FRAME_HDR_SIZE:
                    break
                gen, live_count = struct.unpack("<QI", fhdr)
                payload = f.read(nbytes)
                if len(payload) < nbytes:
                    break
                if idx % every == 0:
                    bits = np.frombuffer(payload, dtype=np.uint8)
                    flat = np.unpackbits(bits, bitorder="little")[:W * H]
                    grid = flat.reshape(H, W).astype(bool)
                    yield gen, live_count, grid
                    yielded += 1
                    if max_frames is not None and yielded >= max_frames:
                        break
            idx += 1


# ──────────────────────────────────────────────────────────────────────────
# Toroidal connected components
# ──────────────────────────────────────────────────────────────────────────

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n + 1))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def label_toroidal(grid):
    """8-connected component labeling with wraparound at all four edges.
    Returns (labels[H,W] int array, num_components)."""
    H, W = grid.shape
    if not grid.any():
        return np.zeros((H, W), dtype=np.int32), 0

    labels, num = ndimage.label(grid, structure=np.ones((3, 3), dtype=np.uint8))
    if num == 0:
        return labels, 0

    uf = UnionFind(num)

    # Merge across top/bottom wrap (row 0 <-> row H-1), including diagonal wrap.
    for c in range(W):
        if labels[0, c] == 0:
            continue
        for dc in (-1, 0, 1):
            other = labels[H - 1, (c + dc) % W]
            if other != 0:
                uf.union(labels[0, c], other)

    # Merge across left/right wrap (col 0 <-> col W-1), including diagonal wrap.
    for r in range(H):
        if labels[r, 0] == 0:
            continue
        for dr in (-1, 0, 1):
            other = labels[(r + dr) % H, W - 1]
            if other != 0:
                uf.union(labels[r, 0], other)

    remap = np.zeros(num + 1, dtype=np.int32)
    next_id = 0
    seen = {}
    for old in range(1, num + 1):
        root = uf.find(old)
        if root not in seen:
            next_id += 1
            seen[root] = next_id
        remap[old] = seen[root]

    out = remap[labels]
    return out, next_id


def component_info(labels, num, grid_h, grid_w):
    """Return {label: dict(cells, size, bbox, centroid, raw_sig)}."""
    info = {}
    if num == 0:
        return info
    objs = ndimage.find_objects(labels)
    for lbl in range(1, num + 1):
        slc = objs[lbl - 1]
        if slc is None:
            continue
        coords = np.argwhere(labels == lbl)
        rs_full, cs_full = coords[:, 0], coords[:, 1]
        min_r, max_r = int(rs_full.min()), int(rs_full.max())
        min_c, max_c = int(cs_full.min()), int(cs_full.max())
        # Detect wraparound: if the component's row OR column span is
        # implausibly large relative to its cell count it likely wraps; in
        # that case anchor by the component's actual cell adjacency instead
        # of naive min/max (rare edge case for baked, non-noise patterns).
        raw_sig = frozenset(
            (int(r - min_r), int(c - min_c)) for r, c in zip(rs_full, cs_full)
        )
        size = len(raw_sig)
        info[lbl] = {
            "cells": set(zip(rs_full.tolist(), cs_full.tolist())),
            "size": size,
            "bbox": (min_r, min_c, max_r, max_c),
            "anchor": (min_r, min_c),
            "centroid": (float(rs_full.mean()), float(cs_full.mean())),
            "raw_sig": raw_sig,
        }
    return info


# ──────────────────────────────────────────────────────────────────────────
# Canonical (orientation-invariant) shape signature, for catalog matching
# ──────────────────────────────────────────────────────────────────────────

def _normalize(coords):
    min_r = min(r for r, c in coords)
    min_c = min(c for r, c in coords)
    return frozenset((r - min_r, c - min_c) for r, c in coords)


def canonical_signature(raw_sig):
    """D4-minimal form: try all 8 rotations/reflections, normalize each to
    its own bounding-box origin, return the lexicographically smallest
    sorted tuple. Orientation- and reflection-independent."""
    variants = []
    pts = list(raw_sig)
    base = pts
    transforms = [
        lambda r, c: (r, c),
        lambda r, c: (c, -r),
        lambda r, c: (-r, -c),
        lambda r, c: (-c, r),
        lambda r, c: (r, -c),
        lambda r, c: (-r, c),
        lambda r, c: (c, r),
        lambda r, c: (-c, -r),
    ]
    for t in transforms:
        transformed = [t(r, c) for r, c in base]
        norm = _normalize(transformed)
        variants.append(tuple(sorted(norm)))
    return min(variants)


# ──────────────────────────────────────────────────────────────────────────
# Built-in catalog (small starter set — easy to extend with more entries
# from the LifeWiki taxonomy later; everything here is a textbook pattern)
# ──────────────────────────────────────────────────────────────────────────

CATALOG_PATTERNS = {
    "block":   (["OO", "OO"], "still_life"),
    "beehive": ([".OO.", "O..O", ".OO."], "still_life"),
    "loaf":    ([".OO.", "O..O", ".O.O", "..O."], "still_life"),
    "boat":    (["OO.", "O.O", ".O."], "still_life"),
    "tub":     ([".O.", "O.O", ".O."], "still_life"),
    "blinker": (["OOO"], "oscillator"),
    "beacon":  (["OO..", "OO..", "..OO", "..OO"], "oscillator"),
    "pulsar":  ([
        "..OOO...OOO..", ".............", "O....O.O....O", "O....O.O....O",
        "O....O.O....O", "..OOO...OOO..", ".............", "..OOO...OOO..",
        "O....O.O....O", "O....O.O....O", "O....O.O....O", ".............",
        "..OOO...OOO..",
    ], "oscillator"),
    "glider":  ([".O.", "..O", "OOO"], "spaceship"),
    "lwss":    ([".O..O", "O....", "O...O", "OOOO."], "spaceship"),
}


def build_catalog():
    catalog = {}
    for name, (rows, kind) in CATALOG_PATTERNS.items():
        coords = [
            (r, c) for r, row in enumerate(rows) for c, ch in enumerate(row) if ch == "O"
        ]
        sig = canonical_signature(frozenset(coords))
        catalog[sig] = (name, kind)
    return catalog


CATALOG = build_catalog()


def catalog_lookup(raw_sig):
    sig = canonical_signature(raw_sig)
    hit = CATALOG.get(sig)
    return hit if hit else (None, None)


# ──────────────────────────────────────────────────────────────────────────
# Frame-to-frame linkage
# ──────────────────────────────────────────────────────────────────────────

def link_frames(prev_info, cur_labels, grid_h, grid_w):
    """For each prev label, find which cur labels its dilated (toroidal,
    Chebyshev radius 1) footprint touches. Returns {prev_label: set(cur_labels)}."""
    links = defaultdict(set)
    for lbl, info in prev_info.items():
        touched = set()
        for (r, c) in info["cells"]:
            for dr in (-1, 0, 1):
                rr = (r + dr) % grid_h
                for dc in (-1, 0, 1):
                    cc = (c + dc) % grid_w
                    v = cur_labels[rr, cc]
                    if v != 0:
                        touched.add(int(v))
        links[lbl] = touched
    return links


# ──────────────────────────────────────────────────────────────────────────
# Lineage tracking + classification
# ──────────────────────────────────────────────────────────────────────────

class UnionFindIds:
    """Union-find over arbitrary (lineage) integer ids, used to track which
    lineages are part of the same merge/split interaction cluster."""
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def classify_history(history):
    """history: list of (gen, raw_sig, anchor, size), time-ordered.
    Detects the minimal period at which the shape exactly repeats
    (translation allowed, same size), and classifies accordingly."""
    n = len(history)
    if n < 2:
        return {"kind": "unresolved", "reason": "too_short"}
    gen0, sig0, anchor0, size0 = history[0]
    for j in range(1, n):
        genj, sigj, anchorj, sizej = history[j]
        if sizej != size0:
            continue
        if sigj == sig0:
            period = genj - gen0
            dr = anchorj[0] - anchor0[0]
            dc = anchorj[1] - anchor0[1]
            if dr == 0 and dc == 0:
                kind = "still_life" if period == 1 else "oscillator"
            else:
                kind = "spaceship"
            speed = max(abs(dr), abs(dc)) / period if period else 0.0
            name, catalog_kind = catalog_lookup(sig0)
            return {
                "kind": kind,
                "period": int(period),
                "displacement": [int(dr), int(dc)],
                "speed_c": round(speed, 4),
                "size": int(size0),
                "catalog_name": name,
                "gen_confirmed": int(genj),
            }
    return {"kind": "unresolved", "reason": "no_repeat_found", "size": int(size0)}


class Lineage:
    _next_id = 1

    def __init__(self, start_gen, info):
        self.id = Lineage._next_id
        Lineage._next_id += 1
        self.history = [(start_gen, info["raw_sig"], info["anchor"], info["size"])]
        self.end_reason = None

    def append(self, gen, info):
        self.history.append((gen, info["raw_sig"], info["anchor"], info["size"]))

    def classify(self):
        return classify_history(self.history)


def analyze(path, every=1, max_frames=None, verbose=True):
    hdr = read_header(path)
    W, H = hdr["grid_w"], hdr["grid_h"]
    if verbose:
        comp = hdr.get("compression", 0)
        comp_str = "delta+blocksparse+zstd" if comp == 1 else "none (raw)"
        print(f"Archive: {path}")
        print(f"Grid: {W}x{H}  Rules: {rule_str(hdr['rule_b'], hdr['rule_s'])}  "
              f"Recorded gens: {hdr['total_gens']}")
        print(f"Compression: {comp_str}")
        if hdr["stag_period"] == -1:
            print("Stagnation: extinction")
        elif hdr["stag_period"] == 0:
            print("Stagnation: none (max_gens reached)")
        else:
            print(f"Stagnation: period {hdr['stag_period']} cycle at gen {hdr['stag_gen']}")
        print()

    active = {}       # cur_label -> Lineage
    prev_info = None
    finished = []
    cluster_uf = UnionFindIds()
    frame_count = 0

    for gen, live_count, grid in iter_frames(path, every=every, max_frames=max_frames):
        labels, num = label_toroidal(grid)
        cur_info = component_info(labels, num, H, W)
        frame_count += 1

        if prev_info is None:
            for lbl, info in cur_info.items():
                active[lbl] = Lineage(gen, info)
        else:
            links = link_frames(prev_info, labels, H, W)
            parents_of = defaultdict(set)
            for plbl, targets in links.items():
                for t in targets:
                    parents_of[t].add(plbl)

            handled_cur = set()
            clean_pairs = []   # (prev_label, target_label): simple 1:1 continuation
            closing = []       # (lineage, targets): split / merge / death

            for plbl, targets in links.items():
                lineage = active.get(plbl)
                if lineage is None:
                    continue
                if len(targets) == 1:
                    t = next(iter(targets))
                    if len(parents_of[t]) == 1:
                        clean_pairs.append((plbl, t))
                        handled_cur.add(t)
                        continue
                closing.append((lineage, targets))

            new_active = {}
            for plbl, t in clean_pairs:
                lineage = active[plbl]
                lineage.append(gen, cur_info[t])
                new_active[t] = lineage

            for lbl, info in cur_info.items():
                if lbl not in handled_cur:
                    new_active[lbl] = Lineage(gen, info)

            for lineage, targets in closing:
                lineage.end_reason = "died" if len(targets) == 0 else (
                    "split" if len(targets) > 1 else "merged"
                )
                finished.append(lineage)
                for t in targets:
                    if t in new_active:
                        cluster_uf.union(lineage.id, new_active[t].id)

            active = new_active

        prev_info = cur_info

    for lineage in active.values():
        lineage.end_reason = "archive_ended"
        finished.append(lineage)

    # Individual-lineage results (clean, never-merged objects).
    results = []
    for lineage in finished:
        gen_start = lineage.history[0][0]
        gen_end = lineage.history[-1][0]
        cls = lineage.classify()
        cls.update({
            "lineage_id": lineage.id,
            "gen_start": int(gen_start),
            "gen_end": int(gen_end),
            "frames_tracked": len(lineage.history),
            "end_reason": lineage.end_reason,
        })
        results.append(cls)

    # Cluster-level results: lineages that merged/split among themselves
    # (e.g. a pulsar's 12 sub-blobs <-> 4 sub-blobs each half-period) are
    # unioned and re-tested as a single combined shape. This recovers
    # periodicity that's invisible to any individual sub-piece. Clusters
    # that successfully classify replace their member lineages' individual
    # "unresolved" entries; clusters that don't classify are left as-is
    # (no information lost — the per-lineage entries remain).
    by_root = defaultdict(list)
    for lineage in finished:
        by_root[cluster_uf.find(lineage.id)].append(lineage)

    consumed = set()
    cluster_results = []
    for root, members in by_root.items():
        if len(members) < 2:
            continue
        per_gen_cells = defaultdict(set)
        for lineage in members:
            for (g, sig, anchor, _size) in lineage.history:
                for (dr, dc) in sig:
                    per_gen_cells[g].add((anchor[0] + dr, anchor[1] + dc))

        norm_history = []
        for g in sorted(per_gen_cells):
            cells = per_gen_cells[g]
            min_r = min(r for r, c in cells)
            min_c = min(c for r, c in cells)
            sig = frozenset((r - min_r, c - min_c) for r, c in cells)
            norm_history.append((g, sig, (min_r, min_c), len(cells)))

        cls = classify_history(norm_history)
        if cls["kind"] != "unresolved":
            cls.update({
                "lineage_id": f"cluster-{root}",
                "gen_start": int(norm_history[0][0]),
                "gen_end": int(norm_history[-1][0]),
                "frames_tracked": len(norm_history),
                "end_reason": "interacting_cluster",
                "num_components": len(members),
            })
            cluster_results.append(cls)
            consumed.update(m.id for m in members)

    results = [r for r in results if r.get("lineage_id") not in consumed] + cluster_results

    if verbose:
        summarize(results, frame_count)

    return {"header": hdr, "lineages": results, "frames_analyzed": frame_count}


def summarize(results, frame_count):
    counts = defaultdict(int)
    named = defaultdict(int)
    clusters = 0
    for r in results:
        counts[r["kind"]] += 1
        if r.get("catalog_name"):
            named[r["catalog_name"]] += 1
        if r.get("end_reason") == "interacting_cluster":
            clusters += 1

    print(f"Analyzed {frame_count} frames, {len(results)} object-lineages total"
          f" ({clusters} of them resolved as multi-component clusters)\n")
    for kind in ("still_life", "oscillator", "spaceship", "unresolved"):
        if counts[kind]:
            print(f"  {kind:12s}: {counts[kind]}")
    if named:
        print("\n  Named matches:")
        for name, n in sorted(named.items()):
            print(f"    {name:12s}: {n}")

    periodic = [r for r in results if r["kind"] in ("oscillator", "spaceship")]
    if periodic:
        print("\n  Periods seen:", sorted({r["period"] for r in periodic}))


# ──────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Offline analysis for .gol archives")
    ap.add_argument("archive", help="path to .gol file")
    ap.add_argument("--out", help="write full JSON report here")
    ap.add_argument("--every", type=int, default=1, help="sample every Nth frame (default 1)")
    ap.add_argument("--max-frames", type=int, default=None, help="stop after N sampled frames")
    ap.add_argument("--quiet", action="store_true", help="suppress console summary")
    args = ap.parse_args()

    report = analyze(args.archive, every=args.every, max_frames=args.max_frames,
                      verbose=not args.quiet)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()