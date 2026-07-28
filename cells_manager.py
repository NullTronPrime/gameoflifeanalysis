#!/usr/bin/env python3
"""
cells_manager.py — Unified Conway's Game of Life Pattern Manager

Subcommands:
  convert     Convert RLE/LIF → .cells files
  index       Generate/update patterns_index.json
  catalog     Pre-compute canonical signatures for analyse_gol.py
  compose     Place multiple .cells patterns on a grid → combined .cells
  config      Generate .cfg files for gol_sim.cpp
  list        List available patterns
  download    Download pattern sources (jslife, guns, etc.)
"""

import argparse
import json
import os
import re
import shutil
import sys
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
BUILTIN_DIR = SCRIPT_DIR / "built_in_patterns"
INDEX_PATH = BUILTIN_DIR / "patterns_index.json"
CATALOG_PATH = BUILTIN_DIR / "catalog_signatures.json"
CONWAYLIFE_DIR = BUILTIN_DIR / "conwaylife"

MAX_CELLS_DIMS = 5000
MAX_SPARSE_DIMS = 10000
MAX_SPARSE_CELLS = 2000000
MAX_CATALOG_CELLS = 100

CATEGORY_RULES = [
    ("still_life", r"still\s*life", 50),
    ("oscillator", r"p\d+", 30),
    ("spaceship", r"spaceship|ship|c/\d+|(\d+)p(\d+)h(\d+)v(\d+)", 30),
    ("gun", r"gun", 40),
    ("puffer", r"puffer|rake|breeder", 30),
    ("eater", r"eater", 40),
    ("reflector", r"reflector", 40),
    ("synthesis", r"synth", 10),
    ("reaction", r"reaction|interaction", 20),
    ("shuttle", r"shuttle", 30),
    ("hassler", r"hassler", 30),
    ("loop", r"loop", 20),
    ("wick", r"wick", 20),
    ("fuse", r"fuse", 20),
    ("tagalong", r"tagalong|pushalong", 20),
    ("quadratic_growth", r"quadratic\s*growth|infinite\s*growth|breeder", 30),
    ("methuselah", r"methuselah|diehard", 30),
    ("other", r".*", 0),
]

# ---- RLE/LIF Parser ----


def parse_rle(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()

    name = Path(filepath).stem
    author = ""
    comments = []
    header_line = None
    data_lines = []
    metadata = {"source_file": Path(filepath).name}

    for line in lines:
        line = line.rstrip()
        if line.startswith("#N"):
            val = line[3:].strip()
            if val:
                name = val
        elif line.startswith("#O"):
            author = line[3:].strip()
        elif line.startswith("#C") or line.startswith("#D"):
            comments.append(line[1:].strip())
        elif line.startswith("#P"):
            header_line = line
        elif line.startswith("x ="):
            header_line = line
        elif header_line or (not header_line and line and not line.startswith("#")):
            data_lines.append(line)

    metadata["name"] = name
    if author:
        metadata["author"] = author
    if comments:
        metadata["comments"] = comments

    if not header_line:
        rle_str = "".join(line.strip() for line in data_lines)
        width, height = _guess_rle_dims(rle_str)
        if width == 0 or height == 0:
            return name, comments, [], 0, 0, metadata
        metadata["rule"] = "B3/S23"
        grid = _decode_rle(rle_str, width, height)
        return name, comments, grid, width, height, metadata

    if header_line.startswith("#P"):
        return _parse_life106(lines, metadata)

    m = re.match(r"x\s*=\s*(\d+)\s*,\s*y\s*=\s*(\d+)(?:\s*,\s*rule\s*=\s*(\S+))?", header_line)
    if not m:
        return name, comments, [], 0, 0, metadata
    width, height = int(m.group(1)), int(m.group(2))
    if m.group(3):
        metadata["rule"] = m.group(3)

    rle_str = "".join(line.strip() for line in data_lines)
    metadata["rle_raw_len"] = len(rle_str)

    if width <= MAX_CELLS_DIMS and height <= MAX_CELLS_DIMS:
        return name, comments, _decode_rle(rle_str, width, height), width, height, metadata

    metadata["too_large"] = True
    metadata["actual_width"] = width
    metadata["actual_height"] = height
    coords = _decode_rle_sparse(rle_str)
    if coords is None:
        return name, comments, [], width, height, metadata
    live_count = len(coords)
    metadata["cells"] = live_count
    if live_count > MAX_SPARSE_CELLS:
        return name, comments, [], width, height, metadata
    xs = [x for x, y in coords]
    ys = [y for x, y in coords]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    bw = max_x - min_x + 1
    bh = max_y - min_y + 1
    metadata["bounding_width"] = bw
    metadata["bounding_height"] = bh
    if bw > MAX_SPARSE_DIMS or bh > MAX_SPARSE_DIMS:
        metadata["too_wide"] = True
        return name, comments, [], width, height, metadata
    grid = [["."] * bw for _ in range(bh)]
    for x, y in coords:
        grid[y - min_y][x - min_x] = "O"
    grid_strs = ["".join(row) for row in grid]
    return name, comments, grid_strs, bw, bh, metadata


def _parse_life106(lines, metadata):
    coords = set()
    px = py = 0
    for line in lines:
        line = line.rstrip()
        if line.startswith("#P"):
            parts = line.split()
            if len(parts) >= 3:
                px, py = int(parts[1]), int(parts[2])
        elif line and not line.startswith("#"):
            for x, ch in enumerate(line):
                if ch in ("*", "O", "o"):
                    coords.add((px + x, py))
    if not coords:
        return metadata["name"], [], [], 0, 0, metadata
    xs = [x for x, y in coords]
    ys = [y for x, y in coords]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    bw = max_x - min_x + 1
    bh = max_y - min_y + 1
    metadata["cells"] = len(coords)
    metadata["bounding_width"] = bw
    metadata["bounding_height"] = bh
    metadata["format"] = "Life 1.06"
    if bw > MAX_SPARSE_DIMS or bh > MAX_SPARSE_DIMS:
        metadata["too_wide"] = True
        return metadata["name"], [], [], bw, bh, metadata
    grid = [["."] * bw for _ in range(bh)]
    for x, y in coords:
        grid[y - min_y][x - min_x] = "O"
    grid_strs = ["".join(row) for row in grid]
    return metadata["name"], [], grid_strs, bw, bh, metadata


def _guess_rle_dims(rle_str):
    rows = rle_str.split("$")
    max_row_len = 0
    row_count = 0
    for row in rows:
        row_count += 1
        idx = row.find("!")
        if idx >= 0:
            row = row[:idx]
        count = 0
        for ch in row:
            if ch.isdigit():
                count = count * 10 + int(ch)
            elif ch in ("b", "B", "o", "O"):
                count = count if count else 1
                max_row_len += count
                count = 0
        if count:
            max_row_len += count
    return max_row_len, row_count


def _decode_rle(rle_str, width, height):
    if width > 2000000 or height > 2000000:
        return []
    rows = []
    cur_row = []
    cur_count = 0
    i = 0
    while i < len(rle_str):
        ch = rle_str[i]
        if ch.isdigit():
            cur_count = cur_count * 10 + int(ch)
            i += 1
            continue
        if cur_count == 0:
            cur_count = 1
        if ch in ("b", "B"):
            cur_row.extend(["."] * min(cur_count, max(0, width + 1 - len(cur_row))))
        elif ch in ("o", "O"):
            cur_row.extend(["O"] * min(cur_count, max(0, width + 1 - len(cur_row))))
        elif ch == "$":
            rows.append("".join(cur_row))
            cur_row = []
        elif ch == "!":
            break
        cur_count = 0
        i += 1
    if cur_row:
        rows.append("".join(cur_row))
    while len(rows) < height:
        rows.append("")
    for i in range(len(rows)):
        if len(rows[i]) < width:
            rows[i] = rows[i] + "." * (width - len(rows[i]))
        rows[i] = rows[i][:width]
    trimmed = _trim_empty_rows_and_cols(rows)
    return trimmed if trimmed else rows


def _decode_rle_sparse(rle_str):
    coords = set()
    x = y = 0
    i = cur_count = 0
    while i < len(rle_str):
        ch = rle_str[i]
        if ch.isdigit():
            cur_count = cur_count * 10 + int(ch)
            i += 1
            continue
        if cur_count == 0:
            cur_count = 1
        if ch in ("b", "B"):
            x += cur_count
        elif ch in ("o", "O"):
            for _ in range(cur_count):
                coords.add((x, y))
                x += 1
        elif ch == "$":
            y += cur_count if cur_count > 1 else 1
            x = 0
        elif ch == "!":
            break
        cur_count = 0
        i += 1
    return coords


def _trim_empty_rows_and_cols(rows):
    if not rows:
        return rows
    non_empty_rows = [r for r in rows if any(c == "O" for c in r)]
    if not non_empty_rows:
        return rows
    min_col = min((c for r in rows for c, ch in enumerate(r) if ch == "O"), default=0)
    max_col = max((c for r in rows for c, ch in enumerate(r) if ch == "O"), default=0)
    return [r[min_col: max_col + 1] for r in non_empty_rows]


# ---- Classification ----


def classify_pattern(name, comments, src_path):
    text = name.lower() + " " + " ".join(c.lower() for c in comments)
    text += " " + src_path.stem.lower()
    for cat, pattern, priority in sorted(CATEGORY_RULES, key=lambda x: -x[2]):
        if re.search(pattern, text, re.IGNORECASE):
            return cat
    return "other"


def classify_by_period(name, comments):
    text = name.lower() + " " + " ".join(c.lower() for c in comments)
    m = re.search(r"period\s*(\d+)", text)
    if m:
        return int(m.group(1))
    m = re.search(r"p(\d+)", Path(name).stem.lower())
    if m:
        return int(m.group(1))
    return 0


# ---- Subcommand: convert ----


def cmd_convert(args):
    """Convert RLE/LIF → .cells from source directories."""
    sources = []
    if args.source:
        for s in args.source:
            sp = Path(s)
            if sp.is_dir():
                sources.append((sp.name, sp))
            else:
                print(f"  WARNING: source directory not found: {s}")
    else:
        extracted = SCRIPT_DIR / "pattern_source" / "extracted"
        if extracted.exists():
            for d in sorted(extracted.iterdir()):
                if d.is_dir():
                    sources.append((d.name, d))
        if not sources:
            print("No source directories specified and pattern_source/extracted/ not found.")
            print("Use: cells_manager.py convert --source <dir>")
            return

    existing_cells = {}
    if CONWAYLIFE_DIR.exists():
        for cp in CONWAYLIFE_DIR.glob("*.cells"):
            existing_cells[cp.stem.lower()] = cp.name

    all_converted = 0
    all_existing = 0
    all_errors = 0
    new_entries = {}

    for src_name, src_dir in sources:
        files = []
        for ext in ("*.rle", "*.RLE", "*.lif", "*.LIF"):
            files.extend(src_dir.rglob(ext))

        out_dir = BUILTIN_DIR / src_name
        out_dir.mkdir(parents=True, exist_ok=True)

        for fpath in files:
            stem = fpath.stem
            target_name = f"{src_name}_{stem}" if stem.lower() in existing_cells else stem
            cells_path = out_dir / f"{target_name}.cells"
            if cells_path.exists():
                all_existing += 1
                continue

            try:
                name, comments, grid, w, h, meta = parse_rle(fpath)
            except Exception as e:
                print(f"  ERROR {src_name}/{fpath.name}: {e}")
                all_errors += 1
                continue

            if not grid:
                reason = meta.get("too_wide", "") or meta.get("too_large", "") or "unparseable"
                print(f"  SKIP {src_name}/{fpath.name}: {reason}")
                all_errors += 1
                continue

            with open(cells_path, "w") as f:
                f.write(f"!Name: {name}\n")
                if "author" in meta:
                    f.write(f"!Author: {meta['author']}\n")
                if "comments" in meta:
                    for c in meta["comments"]:
                        f.write(f"!{c}\n")
                if src_name != "jslife-20121230":
                    f.write(f"!Source: {src_name}\n")
                f.write("!\n")
                for row in grid:
                    f.write(row + "\n")

            all_converted += 1

            cat = classify_pattern(name, comments, fpath)
            period = classify_by_period(name, comments)
            live_cells = sum(row.count("O") for row in grid)
            bw = max(len(row) for row in grid) if grid else 0
            bh = len(grid)
            new_entries[target_name] = {
                "name": meta.get("name", target_name),
                "category": cat,
                "width": bw,
                "height": bh,
                "cells": live_cells,
                "period": period if period else None,
                "author": meta.get("author", ""),
                "comments": meta.get("comments", []),
                "source": src_name,
            }

        print(f"  {src_name}: {len(list(out_dir.glob('*.cells')))} .cells files")

    print(f"Converted: {all_converted}, Existing: {all_existing}, Errors: {all_errors}")

    if new_entries and args.update_index:
        if INDEX_PATH.exists():
            with open(INDEX_PATH) as f:
                index = json.load(f)
        else:
            index = {}
        index.update(new_entries)
        with open(INDEX_PATH, "w") as f:
            json.dump(index, f, indent=2)
        print(f"Index updated: {len(index)} total patterns")

    # Organize into category folders
    if args.organize:
        organized = 0
        for stem, entry in new_entries.items():
            cells_path = next(BUILTIN_DIR.rglob(f"{stem}.cells"), None)
            if cells_path is None:
                continue
            dst = BUILTIN_DIR / entry["category"] / cells_path.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(cells_path, dst)
                organized += 1
        print(f"Organized {organized} patterns into category folders")


# ---- Subcommand: index ----


def cmd_index(args):
    """Build/update patterns_index.json from all .cells files."""
    index = {}
    all_cells = list(BUILTIN_DIR.rglob("*.cells"))
    for cells_path in all_cells:
        if cells_path.parent.name in ("conwaylife",) or cells_path.parent == BUILTIN_DIR:
            continue
        stem = cells_path.stem
        with open(cells_path) as f:
            lines = f.readlines()
        comment_lines = []
        w = h = live_cells = 0
        for line in lines:
            if line.startswith("!"):
                comment_lines.append(line[1:].strip())
            elif line.strip():
                live_cells += line.strip().count("O")
                h += 1
                w = max(w, len(line.strip()))
        cat = classify_pattern(stem, comment_lines, Path(stem))
        period = classify_by_period(stem, comment_lines)
        index[stem] = {
            "name": stem,
            "category": cat,
            "width": w,
            "height": h,
            "cells": live_cells,
            "period": period if period else None,
            "author": "",
            "comments": comment_lines,
        }

    with open(INDEX_PATH, "w") as f:
        json.dump(index, f, indent=2)
    print(f"Index: {len(index)} patterns -> {INDEX_PATH}")

    # Report category counts
    counts = defaultdict(int)
    for stem, entry in index.items():
        counts[entry["category"]] += 1
    for cat, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")


# ---- Subcommand: catalog ----


def cmd_catalog(args):
    """Pre-compute canonical signatures for analyse_gol.py."""
    sys.path.insert(0, str(SCRIPT_DIR))
    try:
        from analyse_gol import canonical_signature
    except ImportError:
        print("ERROR: analyse_gol.py not found. Cannot compute signatures.")
        return

    if not INDEX_PATH.exists():
        print("No index found. Run 'cells_manager.py index' first.")
        return

    with open(INDEX_PATH) as f:
        index = json.load(f)

    catalog = {}
    skipped = 0
    for stem, info in sorted(index.items()):
        cells_count = info.get("cells", 0)
        if cells_count > MAX_CATALOG_CELLS or cells_count == 0:
            skipped += 1
            continue
        cells_path = next(BUILTIN_DIR.rglob(f"{stem}.cells"), None)
        if cells_path is None:
            skipped += 1
            continue
        coords = []
        with open(cells_path) as f:
            for r, line in enumerate(f):
                if line.startswith("!"):
                    continue
                line = line.rstrip()
                for c, ch in enumerate(line):
                    if ch == "O":
                        coords.append((r, c))
        if not coords:
            skipped += 1
            continue
        sig = canonical_signature(frozenset(coords))
        sig_key = ";".join(f"{r},{c}" for r, c in sig)
        catalog[sig_key] = {
            "name": info.get("name", stem),
            "category": info.get("category", "other"),
            "stem": stem,
            "cells": cells_count,
            "period": info.get("period"),
        }

    with open(CATALOG_PATH, "w") as f:
        json.dump(catalog, f, indent=2)
    print(f"Catalog: {len(catalog)} signatures -> {CATALOG_PATH}")
    print(f"Skipped {skipped} patterns (too large or no cells)")


# ---- Subcommand: compose ----


def load_cells(path):
    coords = []
    with open(path) as f:
        for r, line in enumerate(f):
            line = line.rstrip()
            if line.startswith("!"):
                continue
            for c, ch in enumerate(line):
                if ch == "O":
                    coords.append((c, r))
    return coords


def load_pattern(name):
    if Path(name).exists():
        return load_cells(Path(name))
    for d in [CONWAYLIFE_DIR] + sorted(BUILTIN_DIR.iterdir()) + [BUILTIN_DIR]:
        if d.is_dir():
            p = d / f"{name}.cells"
            if p.exists():
                return load_cells(p)
    raise FileNotFoundError(f"Pattern '{name}' not found.")


def make_grid(width, height):
    return [["." for _ in range(width)] for _ in range(height)]


def place_pattern(grid, coords, ox, oy):
    if not coords:
        return (ox, oy, ox, oy)
    pmin_x = pmin_y = float("inf")
    pmax_x = pmax_y = float("-inf")
    for c, r in coords:
        x, y = ox + c, oy + r
        if 0 <= y < len(grid) and 0 <= x < len(grid[0]):
            grid[y][x] = "O"
            pmin_x = min(pmin_x, x)
            pmax_x = max(pmax_x, x)
            pmin_y = min(pmin_y, y)
            pmax_y = max(pmax_y, y)
    return (pmin_x, pmin_y, pmax_x, pmax_y)


def grid_to_cells(grid, path, comments=None, trim=True):
    if trim:
        rows_used = [i for i, row in enumerate(grid) if any(c == "O" for c in row)]
        if rows_used:
            cols_used = [j for i, row in enumerate(grid) for j, c in enumerate(row) if c == "O"]
            min_r, max_r = min(rows_used), max(rows_used)
            min_c, max_c = min(cols_used), max(cols_used)
            grid = [row[min_c:max_c + 1] for row in grid[min_r:max_r + 1]]
    with open(path, "w") as f:
        if comments:
            for c in comments:
                f.write(f"!{c}\n")
            f.write("!\n")
        for row in grid:
            f.write("".join(row) + "\n")


def load_index():
    if INDEX_PATH.exists():
        with open(INDEX_PATH) as f:
            return json.load(f)
    return {}


def cmd_compose(args):
    index = load_index()

    if args.list is not None:
        names = sorted(index.keys())
        if args.list:
            names = [n for n in names if args.list.lower() in n.lower()]
        print(f"Patterns matching '{args.list or '*'}': {len(names)}")
        for n in names[:50]:
            info = index.get(n, {})
            print(f"  {n:40s} {info.get('category',''):15s} {info.get('cells',0):4d}c {info.get('width',0):4d}x{info.get('height',0):4d}")
        if len(names) > 50:
            print(f"  ... and {len(names) - 50} more")
        return

    placements = []
    if args.json:
        with open(args.json) as f:
            placements = json.load(f)
    elif args.placements:
        i = 0
        while i < len(args.placements):
            name = args.placements[i]
            i += 1
            if i < len(args.placements) and "," in args.placements[i]:
                parts = args.placements[i].split(",")
                ox, oy = int(parts[0]), int(parts[1])
                i += 1
            else:
                ox = oy = -1
            placements.append({"name": name, "x": ox, "y": oy})

    grid_w_str, grid_h_str = args.grid.split("x")
    grid_w, grid_h = int(grid_w_str), int(grid_h_str)
    grid = make_grid(grid_w, grid_h)
    comments = []
    for p in placements:
        name = p["name"]
        ox = p.get("x", -1)
        oy = p.get("y", -1)
        try:
            coords = load_pattern(name)
        except FileNotFoundError:
            print(f"  WARNING: pattern '{name}' not found, skipping")
            continue
        if ox < 0 or oy < 0:
            ox = (grid_w - (max(c for c, r in coords) + 1 if coords else 0)) // 2
            oy = (grid_h - (max(r for c, r in coords) + 1 if coords else 0)) // 2
        place_pattern(grid, coords, ox, oy)
        info = index.get(name, {})
        comments.append(f"Placed '{info.get('name', name)}' at ({ox},{oy})")
        print(f"  Placed '{name}' at ({ox},{oy})")

    grid_to_cells(grid, args.output, comments=comments, trim=not args.no_trim)
    print(f"Written: {args.output}")


# ---- Subcommand: config ----


def cmd_config(args):
    index = load_index()

    if args.list is not None:
        names = sorted(index.keys())
        if args.list:
            names = [n for n in names if args.list.lower() in n.lower()]
        print(f"Patterns matching '{args.list or '*'}': {len(names)}")
        for n in names[:60]:
            info = index.get(n, {})
            ps = str(info.get("period") or "")
            print(f"  {n:40s} {info.get('category',''):15s} {info.get('cells',0):4d}c p{ps}")
        if len(names) > 60:
            print(f"  ... and {len(names) - 60} more")
        return

    grid_w, grid_h = args.grid.split("x")
    lines = [
        f"# Generated by cells_manager.py config",
        f"grid_w          = {grid_w}",
        f"grid_h          = {grid_h}",
        f"rules           = {args.rules}",
        f"threads         = {args.threads}",
        f"max_gens        = {args.gens}",
    ]

    if args.density > 0:
        lines.append(f"random_density  = {args.density}")
        lines.append(f"random_seed     = {args.seed}")

    if args.archive:
        lines.append(f"archive_file    = {args.archive}")
        lines.append(f"archive_every   = {args.archive_every}")

    if args.video_w and args.video_h:
        lines.append(f"video_w         = {args.video_w}")
        lines.append(f"video_h         = {args.video_h}")
        lines.append(f"video_fps       = {args.video_fps}")

    if args.final_cells:
        lines.append(f"final_cells_file = {args.final_cells}")

    if args.place:
        for placement in args.place:
            name = placement[0]
            if len(placement) >= 3:
                lines.append(f"seed = {name} {placement[1]} {placement[2]}")
            elif len(placement) == 2:
                lines.append(f"seed = {name} {placement[1]}")
            else:
                lines.append(f"seed = {name}")

    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Written: {args.out}")
    print(f"  Grid: {grid_w}x{grid_h}, Rules: {args.rules}, Gens: {args.gens}")

    if args.place:
        missing = []
        for p in args.place:
            name = p[0]
            if name not in index:
                p_path = CONWAYLIFE_DIR / f"{name}.cells"
                if not p_path.exists():
                    missing.append(name)
        if missing:
            print(f"  WARNING: patterns not found: {missing}")


# ---- Subcommand: download ----


ENTROPYMINE_URLS = [
    ("jslife-20121230.zip", "https://entropymine.com/jason/life/p/jslife-20121230.zip"),
    ("jslife-oversize-20100113.zip", "https://entropymine.com/jason/life/oversize/jslife-oversize-20100113.zip"),
    ("guns1j-20121211.zip", "https://entropymine.com/jason/life/dpguns/guns1j-20121211.zip"),
    ("guns2j-20090906.zip", "https://entropymine.com/jason/life/dpguns/guns2j-20090906.zip"),
    ("Life-DL-98.zip", "https://entropymine.com/jason/life/Life-DL-98.zip"),
    ("Life-PR-98.zip", "https://entropymine.com/jason/life/Life-PR-98.zip"),
    ("2c5puff.zip", "https://entropymine.com/jason/life/2c5puff.zip"),
    ("ptpuffer.zip", "https://entropymine.com/jason/life/ptpuffer.zip"),
    ("c5rakes.zip", "https://entropymine.com/jason/life/c5rakes.zip"),
    ("hppatterns.zip", "https://entropymine.com/jason/life/hppatterns.zip"),
]

GITHUB_ZIPS = [
    ("jslife-moving.zip", "https://github.com/Matthias-Merzenich/jslife-moving/archive/master.zip"),
]


def try_download_lifewiki(target_dir):
    """Download LifeWiki pattern archive from conwaylife.com."""
    urls = [
        "https://conwaylife.com/patterns/all.zip",
        "https://conwaylife.com/w/images/patterns/all.zip",
    ]
    for url in urls:
        try:
            print(f"  Trying {url} ...")
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
            )
            zip_path = target_dir / "all.zip"
            urllib.request.urlretrieve(url, zip_path)
            print(f"  Downloaded {zip_path} ({zip_path.stat().st_size / 1024 / 1024:.1f} MB)")
            extract_dir = target_dir / "lifewiki_raw"
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(extract_dir)
            print(f"  Extracted to {extract_dir}")
            return extract_dir
        except Exception as e:
            print(f"  Failed: {e}")
    return None


def cmd_download(args):
    out_dir = SCRIPT_DIR / "pattern_source"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.lifewiki:
        try_download_lifewiki(out_dir)
        return

    all_urls = []
    if args.all or args.source == "entropymine":
        all_urls.extend(ENTROPYMINE_URLS)
    if args.all or args.source == "github":
        all_urls.extend(GITHUB_ZIPS)
    if args.url:
        all_urls.append((Path(args.url).name, args.url))

    if not all_urls:
        print("Specify --source entropymine|github --lifewiki or --url <direct_url>")
        return

    for fname, url in all_urls:
        out_path = out_dir / fname
        if out_path.exists() and not args.force:
            print(f"  Already exists: {fname} (use --force to re-download)")
            continue
        print(f"  Downloading {fname} ...")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as res:
                with open(out_path, "wb") as f:
                    f.write(res.read())
            sz = out_path.stat().st_size
            print(f"    OK ({sz / 1024:.1f} KB)")
        except Exception as e:
            print(f"    FAIL: {e}")


# ---- Main ----


def main():
    ap = argparse.ArgumentParser(description="Conway's Game of Life Pattern Manager")
    sub = ap.add_subparsers(dest="command", required=True)

    # convert
    p = sub.add_parser("convert", help="Convert RLE/LIF → .cells")
    p.add_argument("--source", "-s", action="append", help="Source directory (repeatable)")
    p.add_argument("--no-index", dest="update_index", action="store_false", default=True, help="Skip index update")
    p.add_argument("--no-organize", dest="organize", action="store_false", default=True, help="Skip category organization")
    p.set_defaults(func=cmd_convert)

    # index
    p = sub.add_parser("index", help="Build/update patterns_index.json")
    p.set_defaults(func=cmd_index)

    # catalog
    p = sub.add_parser("catalog", help="Build canonical signature catalog")
    p.set_defaults(func=cmd_catalog)

    # compose
    p = sub.add_parser("compose", help="Place patterns on a grid → combined .cells")
    p.add_argument("output", help="Output .cells file")
    p.add_argument("placements", nargs="*", help="<name> <x,y> [<name> <x,y> ...]")
    p.add_argument("--grid", default="500x500", help="Grid size WxH")
    p.add_argument("--json", help="Load placements from JSON")
    p.add_argument("--list", nargs="?", const="", help="List available patterns")
    p.add_argument("--no-trim", action="store_true", help="Don't trim to bounding box")
    p.set_defaults(func=cmd_compose)

    # config
    p = sub.add_parser("config", help="Generate gol_sim.cpp config file")
    p.add_argument("--out", default="sim.cfg", help="Output path")
    p.add_argument("--grid", default="1920x1080", help="Grid WxH")
    p.add_argument("--rules", default="B3/S23", help="Rule string")
    p.add_argument("--gens", type=int, default=100000, help="Max generations")
    p.add_argument("--threads", type=int, default=0, help="Threads (0=auto)")
    p.add_argument("--place", action="append", nargs="+", help="Pattern placement")
    p.add_argument("--density", type=float, default=0, help="Random fill density")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--archive", help="Archive output path (.gol)")
    p.add_argument("--archive-every", type=int, default=1)
    p.add_argument("--video-w", type=int, default=0)
    p.add_argument("--video-h", type=int, default=0)
    p.add_argument("--video-fps", type=int, default=60)
    p.add_argument("--final-cells", help="Write final state .cells")
    p.add_argument("--list", nargs="?", const="", help="List patterns")
    p.set_defaults(func=cmd_config)

    # download
    p = sub.add_parser("download", help="Download pattern sources")
    p.add_argument("--source", choices=["entropymine", "github"], default="entropymine", help="Source to download from")
    p.add_argument("--all", action="store_true", help="Download all known sources")
    p.add_argument("--url", help="Download a specific URL")
    p.add_argument("--force", action="store_true", help="Re-download if exists")
    p.add_argument("--lifewiki", action="store_true", help="Download LifeWiki pattern archive")
    p.set_defaults(func=cmd_download)

    # list (quick access from compose/config)
    p = sub.add_parser("list", help="List available patterns (alias for --list)")
    p.add_argument("filter", nargs="?", default="", help="Optional filter string")
    p.set_defaults(func=lambda a: cmd_compose(argparse.Namespace(list=a.filter, output="", placements=[], grid="500x500", json=None, no_trim=False)))

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
