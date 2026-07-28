#!/usr/bin/env python3
"""
gol_to_video.py — Render .gol archive frames → raw RGB24 → pipe to ffmpeg.

Reads a compressed .gol archive frame by frame, renders (downscaled) full-grid
images, and writes raw RGB24 frames to stdout for ffmpeg to encode into a video.

Pipe to ffmpeg for encoding:
    python gol_to_video.py archive.gol --out 1920x1080 | ffmpeg <args>

Usage:
    python gol_to_video.py <file.gol> [options]

Options:
    --out WxH          Output resolution (default: 1920x1080)
    --fps N            Framerate for output (default: 30)
    --every N          Process every Nth frame (default: 1)
    --max-frames N     Stop after N frames
    --age              Color-coded age map (warmer = older live cells)
    --dry-run          Just print archive info, skip rendering

Examples (PowerShell, ^ = line continuation):
    # 4K archive → 1080p H.264 MP4 (Windows Photos / YouTube)
    python gol_to_video.py longrun_4k.gol --out 1920x1080 --fps 30 | ^
      ffmpeg -f rawvideo -pixel_format rgb24 -video_size 1920x1080 ^
      -framerate 30 -i - -c:v libx264 -crf 20 -pix_fmt yuv420p output.mp4

    # Smaller file with x265 (better compression for YouTube)
    python gol_to_video.py longrun_4k.gol --out 1280x720 --fps 24 --every 2 | ^
      ffmpeg -f rawvideo -pixel_format rgb24 -video_size 1280x720 ^
      -framerate 24 -i - -c:v libx265 -crf 24 -pix_fmt yuv420p output.mp4

    # Age-colored rendering
    python gol_to_video.py longrun_4k.gol --out 1920x1080 --age | ^
      ffmpeg -f rawvideo -pixel_format rgb24 -video_size 1920x1080 ^
      -framerate 30 -i - -c:v libx264 -crf 20 -pix_fmt yuv420p age_map.mp4

    # Sparse archive: skip frames for 510K-frame 8K archive
    python gol_to_video.py 8k_convergence.gol --out 1920x1080 --fps 30 --every 10 | ^
      ffmpeg -f rawvideo -pixel_format rgb24 -video_size 1920x1080 ^
      -framerate 30 -i - -c:v libx265 -crf 24 -pix_fmt yuv420p 8k_archive.mp4

    # Archive info only
    python gol_to_video.py 16k_convergence.gol --dry-run

Requirements: numpy, zstandard

File size estimates for video output (very rough):
    ~3-8 GB per hour of 1080p x264 CRF 20 (for GoL content)
    ~2-5 GB per hour of 1080p x265 CRF 24
    Use --every N to reduce frame count proportionally.
"""

import argparse
import struct
import sys
import time

import numpy as np

try:
    import zstandard as zstd
except ImportError:
    zstd = None

MAGIC = b"GOL1"
HEADER_SIZE = 64
FRAME_HDR_COMP = 21
GOL_BLOCK_SIZE = 128


# ── .gol reading ──

def read_header(path):
    with open(path, "rb") as f:
        hdr = f.read(HEADER_SIZE)
    if len(hdr) != HEADER_SIZE or hdr[:4] != MAGIC:
        raise ValueError("Not a valid .gol archive (bad magic)")
    w, h, total, fps = struct.unpack_from("<IIII", hdr, 4)
    rule_b, rule_s = hdr[20], hdr[24]
    stag_period, stag_gen = struct.unpack_from("<ii", hdr, 28)
    compression = hdr[36] if len(hdr) > 36 else 0
    return dict(grid_w=w, grid_h=h, total_gens=total, fps_hint=fps,
                rule_b=rule_b, rule_s=rule_s, stag_period=stag_period,
                stag_gen=stag_gen, compression=compression)


def rule_str(b, s):
    return f"B{''.join(str(i) for i in range(9) if (b>>i)&1)}/S{''.join(str(i) for i in range(9) if (s>>i)&1)}"


def block_sparse_decode(data, nbytes):
    nblocks = (nbytes + GOL_BLOCK_SIZE - 1) // GOL_BLOCK_SIZE
    mask_bytes = (nblocks + 7) // 8
    mask = data[:mask_bytes]
    blocks = data[mask_bytes:]
    result = bytearray(nbytes)
    bp = 0
    for i in range(nblocks):
        if mask[i // 8] & (1 << (i % 8)):
            off = i * GOL_BLOCK_SIZE
            blen = min(GOL_BLOCK_SIZE, nbytes - off)
            result[off:off + blen] = blocks[bp:bp + blen]
            bp += blen
    return bytes(result)


def iter_frames(path):
    """Yield (gen, live_count, bits_bytes) for each frame in the archive."""
    hdr = read_header(path)
    W, H = hdr["grid_w"], hdr["grid_h"]
    nbytes = (W * H + 7) // 8
    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as f:
        f.seek(HEADER_SIZE)
        prev_bits = None
        while True:
            fhdr = f.read(FRAME_HDR_COMP)
            if len(fhdr) < FRAME_HDR_COMP:
                break
            gen, live = struct.unpack("<QI", fhdr[:12])
            usize, csize = struct.unpack("<II", fhdr[13:21])
            cdata = f.read(csize)
            if len(cdata) < csize:
                break
            sparse = dctx.decompress(cdata, max_output_size=usize)
            decoded = block_sparse_decode(sparse, nbytes)
            if prev_bits is None:
                cur_bits = decoded
            else:
                cur_bits = bytes(a ^ b for a, b in zip(decoded, prev_bits))
            prev_bits = cur_bits
            yield gen, live, cur_bits


# ── Rendering ──

class AgeTracker:
    def __init__(self, H, W, max_age=128):
        self.ages = np.zeros((H, W), dtype=np.uint8)
        self.max_age = max_age
        # Build colour LUT: cool (new) → warm (old)
        self.lut = np.zeros((max_age + 1, 3), dtype=np.uint8)
        for a in range(max_age + 1):
            t = a / max_age
            if t < 0.15:
                s = t / 0.15
                self.lut[a] = [255, 255, int(255 * (1 - s))]
            elif t < 0.38:
                s = (t - 0.15) / 0.23
                self.lut[a] = [255, int(255 * (1 - s * 0.45)), 0]
            elif t < 0.60:
                s = (t - 0.38) / 0.22
                self.lut[a] = [255, int(140 + 60 * (1 - s)), 0]
            elif t < 0.78:
                s = (t - 0.60) / 0.18
                self.lut[a] = [int(255 - 255 * s), 200, int(60 + 195 * s)]
            else:
                s = (t - 0.78) / 0.22
                self.lut[a] = [0, int(80 - 60 * s), int(255 - 175 * s)]

    def update(self, bits, W, H):
        flat = np.frombuffer(bits, dtype=np.uint8)
        alive = np.unpackbits(flat, bitorder='little')[:W * H].reshape(H, W).astype(bool)
        self.ages[alive] = np.minimum(self.ages[alive].astype(np.uint16) + 1, self.max_age).astype(np.uint8)
        self.ages[~alive] = 0

    def render(self, out_h, out_w, W, H):
        """Downscale age map and apply colour LUT."""
        out_ages = _downscale_grid(self.ages, out_h, out_w, W, H)
        return self.lut[out_ages.astype(np.uint8)]


def _downscale_grid(src, out_h, out_w, W, H):
    """Downscale a 2D array to target dimensions using block averaging.

    Uses integer block averaging when possible (clean downscale),
    falls back to nearest-neighbour for non-integer factors.
    """
    if W == out_w and H == out_h:
        return src.copy()

    # Try integer block average first
    if H % out_h == 0 and W % out_w == 0:
        sh = H // out_h
        sw = W // out_w
        trimmed = src[:out_h * sh, :out_w * sw]
        reshaped = trimmed.reshape(out_h, sh, out_w, sw)
        binned = reshaped.mean(axis=(1, 3))
        if src.dtype == np.bool_ or src.dtype == np.uint8:
            return binned.astype(np.uint8)
        return binned
    # Nearest-neighbour sampling for non-integer
    ys = np.linspace(0, H - 1, out_h).astype(np.int32)
    xs = np.linspace(0, W - 1, out_w).astype(np.int32)
    return src[np.ix_(ys, xs)]


def render_binary(bits, out_h, out_w, W, H):
    """Render a binary grid frame to RGB24 array.

    Downscales by block max (live if any cell in block is live).
    """
    if W == out_w and H == out_h:
        flat = np.frombuffer(bits, dtype=np.uint8)
        full = np.unpackbits(flat, bitorder='little')[:W * H].reshape(H, W)
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        rgb[full > 0] = [255, 255, 255]
        return rgb

    flat = np.frombuffer(bits, dtype=np.uint8)
    full = np.unpackbits(flat, bitorder='little')[:W * H].reshape(H, W)

    if H % out_h == 0 and W % out_w == 0:
        sh = H // out_h
        sw = W // out_w
        trimmed = full[:out_h * sh, :out_w * sw]
        blocks = trimmed.reshape(out_h, sh, out_w, sw)
        out = blocks.max(axis=(1, 3))
    else:
        ys = np.linspace(0, H - 1, out_h).astype(np.int32)
        xs = np.linspace(0, W - 1, out_w).astype(np.int32)
        out = full[np.ix_(ys, xs)]

    rgb = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    rgb[out > 0] = [255, 255, 255]
    return rgb


# ── Main ──

def main():
    ap = argparse.ArgumentParser(
        description="Render .gol archive frames → raw RGB24 → pipe to ffmpeg",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("archive", help="Path to .gol archive")
    ap.add_argument("--out", default="1920x1080", help="Output resolution WxH (default: 1920x1080)")
    ap.add_argument("--fps", type=int, default=30, help="Output framerate (default: 30)")
    ap.add_argument("--every", type=int, default=1, help="Process every Nth frame (default: 1)")
    ap.add_argument("--max-frames", type=int, default=0, help="Stop after N frames")
    ap.add_argument("--age", action="store_true", help="Age-coloured rendering")
    ap.add_argument("--dry-run", action="store_true", help="Print archive info and exit")
    args = ap.parse_args()

    if zstd is None:
        print("ERROR: 'zstandard' package required. Install: pip install zstandard", file=sys.stderr)
        sys.exit(1)

    try:
        hdr = read_header(args.archive)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        out_w, out_h = map(int, args.out.lower().split("x"))
    except ValueError:
        print(f"ERROR: invalid --out format '{args.out}'. Use WxH (e.g. 1920x1080)", file=sys.stderr)
        sys.exit(1)

    W, H = hdr["grid_w"], hdr["grid_h"]
    total = hdr["total_gens"]
    stag = hdr["stag_period"]
    stag_str = f"period {stag} cycle" if stag > 0 else ("extinction" if stag == -1 else "max_gens reached")

    print(f"Archive: {args.archive}", file=sys.stderr)
    print(f"  Grid:     {W}x{H}", file=sys.stderr)
    print(f"  Rules:    {rule_str(hdr['rule_b'], hdr['rule_s'])}", file=sys.stderr)
    print(f"  Frames:   {total:,}  ({'until ' + stag_str if stag else 'complete'})", file=sys.stderr)
    print(f"  Output:   {out_w}x{out_h} @ {args.fps} fps", file=sys.stderr)
    if args.every > 1:
        print(f"  Sampling: every {args.every} frame (≈{total // args.every:,} output frames)", file=sys.stderr)
    if args.age:
        print(f"  Colour:   age map", file=sys.stderr)
    if args.dry_run:
        return

    # Pipeline: read frames → render → write RGB24 to stdout
    out = sys.stdout.buffer
    frame_count = 0
    written = 0
    t0 = time.time()

    age_tracker = AgeTracker(H, W) if args.age else None

    for gen, live, bits in iter_frames(args.archive):
        frame_count += 1

        if frame_count % args.every != 0:
            continue

        if args.age:
            age_tracker.update(bits, W, H)
            rgb = age_tracker.render(out_h, out_w, W, H)
        else:
            rgb = render_binary(bits, out_h, out_w, W, H)

        out.write(rgb.tobytes())
        written += 1

        if written % 100 == 0 or written == 1:
            elapsed = time.time() - t0
            rate = written / elapsed if elapsed > 0 else 0
            remaining = (total // args.every - written) / rate if rate > 0 else 0
            print(f"  [{written} frames rendered, {rate:.1f} fps, ETA {remaining:.0f}s]  "
                  f"gen={gen:,} live={live:,}", file=sys.stderr)

        if args.max_frames and written >= args.max_frames:
            print(f"  Reached --max-frames={args.max_frames}", file=sys.stderr)
            break

    elapsed = time.time() - t0
    print(f"Done: {written} frames in {elapsed:.1f}s ({written / elapsed:.1f} fps)", file=sys.stderr)


if __name__ == "__main__":
    main()
