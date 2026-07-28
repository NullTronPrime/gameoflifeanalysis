#!/usr/bin/env python3
"""
Sparse GOL renderer — clusters active blocks in 2D, sizes panels by area,
composites into output frame. Pipe to ffmpeg.

Usage:
  python sparse_render.py <file.gol> <out_w> <out_h> [max_frames] [cluster_gap]
"""
import struct, sys
import numpy as np
import zstandard as zstd

GOL_PATH    = sys.argv[1] if len(sys.argv) > 1 else 'all_patterns_15k.gol'
OUT_W       = int(sys.argv[2]) if len(sys.argv) > 2 else 1920
OUT_H       = int(sys.argv[3]) if len(sys.argv) > 3 else 1080
MAX_FRAMES  = int(sys.argv[4]) if len(sys.argv) > 4 else 0
CLUSTER_GAP = int(sys.argv[5]) if len(sys.argv) > 5 else 100  # cells
PADDING     = 6
MARGIN      = 8   # extra cells around each cluster bbox
MIN_PANEL   = 16  # minimum panel dimension in output pixels

with open(GOL_PATH, 'rb') as fh:
    raw_hdr = fh.read(64)

grid_w  = struct.unpack_from('<I', raw_hdr, 4)[0]
grid_h  = struct.unpack_from('<I', raw_hdr, 8)[0]
nbytes  = (grid_w * grid_h + 7) // 8
BS      = 128
BPB     = BS * 8   # 1024 cells per block
nblocks = (nbytes + BS - 1) // BS
mask_bytes = (nblocks + 7) // 8

print(f"Grid {grid_w}x{grid_h}, output {OUT_W}x{OUT_H}", file=sys.stderr)

# Age colour LUT
age_lut = np.zeros((129, 3), dtype=np.uint8)
for a in range(1, 129):
    t = (a-1)/127.0
    if   t<0.15: s=t/0.15;        age_lut[a]=[255,255,int(255*(1-s))]
    elif t<0.38: s=(t-0.15)/0.23; age_lut[a]=[255,int(255*(1-s*0.45)),0]
    elif t<0.60: s=(t-0.38)/0.22; age_lut[a]=[255,int(140+60*(1-s)),0]
    elif t<0.78: s=(t-0.60)/0.18; age_lut[a]=[int(255-255*s),200,int(60+195*s)]
    else:        s=(t-0.78)/0.22; age_lut[a]=[0,int(80-60*s),int(255-175*s)]

def decode_frame(f, prev_bits):
    """Read and decode one frame. Returns (gen, live_count, mask, frame_bytes)
    or (None,None,None,None) on EOF. Uses correct delta decoding."""
    fhdr = f.read(21)
    if len(fhdr) < 21:
        return None, None, None, None
    g        = struct.unpack_from('<Q', fhdr, 0)[0]
    live     = struct.unpack_from('<I', fhdr, 8)[0]
    usize    = struct.unpack_from('<I', fhdr, 13)[0]
    csize    = struct.unpack_from('<I', fhdr, 17)[0]
    cdata    = f.read(csize)
    if len(cdata) < csize:
        return None, None, None, None

    sparse = dctx.decompress(cdata, max_output_size=usize)
    mask   = sparse[:mask_bytes]
    bdata  = sparse[mask_bytes:]

    # Start with zeros, overlay delta at non-zero blocks
    cur = bytearray(nbytes)
    bp = 0
    for i in range(nblocks):
        if mask[i//8] & (1 << (i%8)):
            off = i * BS
            blen = min(BS, nbytes - off)
            cur[off:off+blen] = bdata[bp:bp+blen]
            bp += blen

    # XOR with previous frame to reconstruct current
    if g > 0 and prev_bits is not None:
        cur = bytearray(a ^ b for a, b in zip(cur, prev_bits))

    return g, live, mask, bytes(cur)

def get_tight_clusters(cur_bits, mask_data):
    """2D cluster active blocks, merge overlapping/nearby clusters,
    return list of (r0,c0,r1,c1, area) sorted by area descending."""
    # Collect active block positions as (row, col) in cell coordinates
    active = []
    for i in range(nblocks):
        if mask_data[i//8] & (1 << (i%8)):
            start_bit = i * BPB
            r = start_bit // grid_w
            c = start_bit % grid_w
            active.append((r, c, i))

    if not active:
        return []

    # Greedy 2D clustering by Chebyshev distance
    active.sort(key=lambda x: (x[0], x[1]))
    raw_clusters = []  # each: {'blocks':[], 'r0':r, 'c0':c, 'r1':r, 'c1':c}
    for r, c, i in active:
        placed = False
        for cl in raw_clusters:
            dx = max(cl['r0'] - r, r - cl['r1'], 0)
            dy = max(cl['c0'] - c, c - cl['c1'], 0)
            if dx <= CLUSTER_GAP and dy <= CLUSTER_GAP:
                cl['blocks'].append((r, c, i))
                cl['r0'] = min(cl['r0'], r)
                cl['c0'] = min(cl['c0'], c)
                cl['r1'] = max(cl['r1'], r)
                cl['c1'] = max(cl['c1'], c)
                placed = True
                break
        if not placed:
            raw_clusters.append({
                'blocks': [(r, c, i)],
                'r0': r, 'c0': c, 'r1': r, 'c1': c,
            })

    # Merge overlapping clusters (repeat until stable)
    changed = True
    while changed:
        changed = False
        merged = []
        used = set()
        for i, a in enumerate(raw_clusters):
            if i in used:
                continue
            for j, b in enumerate(raw_clusters):
                if j <= i or j in used:
                    continue
                dx = max(a['r0'] - b['r1'], b['r0'] - a['r1'], 0)
                dy = max(a['c0'] - b['c1'], b['c0'] - a['c1'], 0)
                if dx <= CLUSTER_GAP and dy <= CLUSTER_GAP:
                    a['r0'] = min(a['r0'], b['r0'])
                    a['c0'] = min(a['c0'], b['c0'])
                    a['r1'] = max(a['r1'], b['r1'])
                    a['c1'] = max(a['c1'], b['c1'])
                    a['blocks'].extend(b['blocks'])
                    used.add(j)
                    changed = True
            merged.append(a)
            used.add(i)
        raw_clusters = merged

    # Tighten bboxes to actual live cells
    arr = np.frombuffer(cur_bits, dtype=np.uint8)
    bits = np.unpackbits(arr, bitorder='little')[:grid_w * grid_h].reshape(grid_h, grid_w)

    clusters = []
    for cl in raw_clusters:
        r0 = max(0, cl['r0'] - MARGIN)
        r1 = min(grid_h, cl['r1'] + BPB // grid_w + 2 + MARGIN)
        c0 = max(0, cl['c0'] - MARGIN)
        c1 = min(grid_w, cl['c1'] + BPB + MARGIN)
        band = bits[r0:r1, c0:c1]
        live_rows, live_cols = np.where(band)
        if len(live_rows) == 0:
            continue
        rr0 = r0 + int(live_rows.min())
        rr1 = r0 + int(live_rows.max()) + 1
        cc0 = c0 + int(live_cols.min())
        cc1 = c0 + int(live_cols.max()) + 1
        area = (rr1 - rr0) * (cc1 - cc0)
        clusters.append((rr0, cc0, rr1, cc1, area))

    # Sort by area descending (largest clusters first)
    clusters.sort(key=lambda x: -x[4])
    return clusters


def render_panel(ages, r0, c0, r1, c1, max_w, max_h):
    """Render a cluster panel to RGB24, fitting within max_w x max_h."""
    patch = ages[r0:r1, c0:c1]
    ph, pw = patch.shape
    scale = min(max_w / max(pw, 1), max_h / max(ph, 1))
    ow = max(MIN_PANEL, int(pw * scale))
    oh = max(MIN_PANEL, int(ph * scale))
    ys = (np.arange(oh) * ph / oh).astype(int)
    xs = (np.arange(ow) * pw / ow).astype(int)
    return age_lut[patch[np.ix_(ys, xs)]]


ages      = np.zeros((grid_h, grid_w), dtype=np.uint8)
prev_bits = None
frames_written = 0
dctx = zstd.ZstdDecompressor()
out  = sys.stdout.buffer

with open(GOL_PATH, 'rb') as f:
    f.seek(64)
    while True:
        g, live, mask, cur = decode_frame(f, prev_bits)
        if cur is None:
            break

        arr  = np.frombuffer(cur, dtype=np.uint8)
        bits = np.unpackbits(arr, bitorder='little')[:grid_w * grid_h].reshape(grid_h, grid_w)
        alive = bits.astype(bool)
        ages[alive]  = np.minimum(ages[alive].astype(np.int16) + 1, 128).astype(np.uint8)
        ages[~alive] = 0

        clusters = get_tight_clusters(cur, mask)

        canvas = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
        if clusters:
            total_area = max(sum(c[4] for c in clusters), 1)
            avail_w = OUT_W - PADDING * 2
            avail_h = OUT_H - PADDING * (len(clusters) + 1)
            y = PADDING
            for r0, c0, r1, c1, area in clusters:
                if y + MIN_PANEL >= OUT_H:
                    break
                # Proportional vertical slot
                frac = area / total_area
                slot_h = max(MIN_PANEL, int(avail_h * frac))
                slot_h = min(slot_h, OUT_H - y - PADDING)
                if slot_h < MIN_PANEL:
                    break
                panel = render_panel(ages, r0, c0, r1, c1, avail_w, slot_h)
                ph, pw = panel.shape[:2]
                # Center panel horizontally
                x_off = PADDING + (avail_w - pw) // 2
                canvas[y:y + ph, x_off:x_off + pw] = panel
                y += slot_h + PADDING

        out.write(canvas.tobytes())
        frames_written += 1
        prev_bits = cur

        if frames_written % 60 == 0 or frames_written == 1:
            print(f"  Frame {frames_written} gen {g} live={live} clusters={len(clusters)}", file=sys.stderr)

        if MAX_FRAMES and frames_written >= MAX_FRAMES:
            break

print(f"Done: {frames_written} frames", file=sys.stderr)
