#!/usr/bin/env python3
"""
Sparse GOL renderer — finds tight bounding boxes per cluster, renders each
as a panel, composites into output frame. Pipe to ffmpeg.

Usage:
  python sparse_render.py <file.gol> <out_w> <out_h> [max_frames] [cluster_gap]
"""
import struct, sys, zstandard as zstd
import numpy as np

GOL_PATH    = sys.argv[1] if len(sys.argv) > 1 else 'all_patterns_15k.gol'
OUT_W       = int(sys.argv[2]) if len(sys.argv) > 2 else 1920
OUT_H       = int(sys.argv[3]) if len(sys.argv) > 3 else 1080
MAX_FRAMES  = int(sys.argv[4]) if len(sys.argv) > 4 else 0
CLUSTER_GAP = int(sys.argv[5]) if len(sys.argv) > 5 else 100  # cells
PADDING     = 6
MARGIN      = 8   # extra cells around each cluster bbox

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

def decode_frame(f, prev_bits, gen):
    fhdr = f.read(21)
    if len(fhdr) < 21: return None, None, None, None
    g        = struct.unpack_from('<Q', fhdr, 0)[0]
    live     = struct.unpack_from('<I', fhdr, 8)[0]
    usize    = struct.unpack_from('<I', fhdr, 13)[0]
    csize    = struct.unpack_from('<I', fhdr, 17)[0]
    cdata    = f.read(csize)
    if len(cdata) < csize: return None, None, None, None

    sparse = dctx.decompress(cdata)
    mask   = sparse[:mask_bytes]
    bdata  = sparse[mask_bytes:]

    cur = bytearray(prev_bits)
    bp  = 0
    for i in range(nblocks):
        blen = BS if (i*BS+BS <= nbytes) else (nbytes - i*BS)
        if mask[i//8] & (1 << (i%8)):
            off = i*BS
            cur[off:off+blen] = bdata[bp:bp+blen]
            bp += blen

    if g > 0:
        cur = bytearray(a^b for a,b in zip(cur, prev_bits))

    return g, live, mask, cur

def get_tight_clusters(cur_bits, mask_data):
    """Find tight (r0,c0,r1,c1) bboxes per cluster of live cells."""
    # Step 1: get rows with active blocks from mask
    active_row_set = set()
    for i in range(nblocks):
        if mask_data[i//8] & (1<<(i%8)):
            cs = i * BPB
            ce = min(cs + BPB - 1, grid_w*grid_h - 1)
            active_row_set.add(cs // grid_w)
            active_row_set.add(ce // grid_w)

    if not active_row_set:
        return []

    # Step 2: merge rows into row-bands
    sorted_rows = sorted(active_row_set)
    bands = []
    s = sorted_rows[0]; p = sorted_rows[0]
    for r in sorted_rows[1:]:
        if r - p > CLUSTER_GAP:
            bands.append((s, p))
            s = r
        p = r
    bands.append((s, p))

    # Step 3: for each band, find tight col range from actual live cells
    arr  = np.frombuffer(cur_bits, dtype=np.uint8)
    bits = np.unpackbits(arr, bitorder='little')[:grid_w*grid_h].reshape(grid_h, grid_w)

    clusters = []
    for r0, r1 in bands:
        r0e = max(0, r0 - MARGIN)
        r1e = min(grid_h-1, r1 + MARGIN)
        band = bits[r0e:r1e+1, :]
        live_rows, live_cols = np.where(band)
        if len(live_cols) == 0:
            continue
        c0 = max(0,        live_cols.min() - MARGIN)
        c1 = min(grid_w-1, live_cols.max() + MARGIN)
        r0f = max(0,        r0e + live_rows.min() - MARGIN)
        r1f = min(grid_h-1, r0e + live_rows.max() + MARGIN)
        clusters.append((r0f, c0, r1f, c1))

    return clusters

def render_panel(ages, r0, c0, r1, c1, max_w, max_h):
    patch = ages[r0:r1+1, c0:c1+1]
    ph, pw = patch.shape
    scale = min(max_w/max(pw,1), max_h/max(ph,1))
    ow = max(1, int(pw*scale))
    oh = max(1, int(ph*scale))
    ys = (np.arange(oh)*ph/oh).astype(int)
    xs = (np.arange(ow)*pw/ow).astype(int)
    return age_lut[patch[np.ix_(ys,xs)]]

ages      = np.zeros((grid_h, grid_w), dtype=np.uint8)
prev_bits = bytearray(nbytes)
frames_written = 0
dctx = zstd.ZstdDecompressor()
out  = sys.stdout.buffer

with open(GOL_PATH, 'rb') as f:
    f.read(64)
    while True:
        g, live, mask, cur = decode_frame(f, prev_bits, frames_written)
        if cur is None: break

        arr  = np.frombuffer(cur, dtype=np.uint8)
        bits = np.unpackbits(arr, bitorder='little')[:grid_w*grid_h].reshape(grid_h, grid_w)
        alive = bits.astype(bool)
        ages[alive]  = np.minimum(ages[alive].astype(np.int16)+1, 128).astype(np.uint8)
        ages[~alive] = 0

        clusters = get_tight_clusters(cur, mask)

        canvas = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
        if clusters:
            n      = len(clusters)
            slot_h = max(1, (OUT_H - PADDING*(n+1)) // n)
            slot_w = OUT_W - PADDING*2
            y = PADDING
            for r0,c0,r1,c1 in clusters:
                if y + 1 >= OUT_H: break
                avail_h = min(slot_h, OUT_H - y - PADDING)
                panel   = render_panel(ages, r0,c0,r1,c1, slot_w, avail_h)
                ph, pw  = panel.shape[:2]
                canvas[y:y+ph, PADDING:PADDING+pw] = panel
                y += slot_h + PADDING

        out.write(canvas.tobytes())
        frames_written += 1

        if frames_written % 60 == 0 or frames_written == 1:
            print(f"  Frame {frames_written} gen {g} live={live} clusters={len(clusters)}", file=sys.stderr)

        prev_bits = bytearray(cur)
        if MAX_FRAMES and frames_written >= MAX_FRAMES:
            break

print(f"Done: {frames_written} frames", file=sys.stderr)
