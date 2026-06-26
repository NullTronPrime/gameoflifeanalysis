#!/usr/bin/env python3
"""Generate self-contained HTML viewer from a .bin coordinate file.

Usage:
  python make_viewer.py <coords.bin> <grid_w> <grid_h> [output.html]
"""
import struct, sys, base64, os

coords_path = sys.argv[1]
grid_w = int(sys.argv[2])
grid_h = int(sys.argv[3])
out_path = sys.argv[4] if len(sys.argv) > 4 else 'viewer.html'

with open(coords_path, 'rb') as f:
    raw = f.read()

# Read uint16 LE pairs
cells = [(raw[i] | raw[i+1] << 8, raw[i+2] | raw[i+3] << 8)
         for i in range(0, len(raw), 4)]

b64 = base64.b64encode(raw).decode()

html = f'''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GoL 32K Viewer</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: #111; overflow: hidden; font-family: monospace; }}
canvas {{ display: block; cursor: grab; }}
canvas:active {{ cursor: grabbing; }}
#info {{
  position: fixed; bottom: 8px; left: 8px; color: #888; font-size: 12px;
  background: rgba(0,0,0,0.7); padding: 4px 8px; border-radius: 4px;
  pointer-events: none; user-select: none;
}}
</style>
</head>
<body>
<canvas id="c"></canvas>
<div id="info">{len(cells):,} live cells  |  Grid {grid_w}x{grid_h}  |  Drag to pan, scroll to zoom</div>
<script>
const GRID_W = {grid_w}, GRID_H = {grid_h};
const CELLS_B64 = "{b64}";
const CELL_COUNT = {len(cells)};

// Decode base64 to Uint16Array
function b64ToUint16(b64) {{
  const bin = atob(b64);
  const buf = new ArrayBuffer(bin.length);
  const view = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) view[i] = bin.charCodeAt(i);
  return new Uint16Array(buf);
}}

const raw = b64ToUint16(CELLS_B64);
// Transpose (x,y) pairs into two parallel arrays for faster access
const cx = new Uint16Array(CELL_COUNT);
const cy = new Uint16Array(CELL_COUNT);
for (let i = 0; i < CELL_COUNT; i++) {{
  cx[i] = raw[i * 2];
  cy[i] = raw[i * 2 + 1];
}}

const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');

let W, H;
let scale = 1;
let panX = 0, panY = 0;
let dragX, dragY, dragPanX, dragPanY, dragging = false;

function resize() {{
  W = canvas.width = window.innerWidth;
  H = canvas.height = window.innerHeight;
}}

// Clamp pan so grid stays in reasonable view
function clampPan() {{
  const gw = GRID_W * scale;
  const gh = GRID_H * scale;
  panX = Math.min(panX, W * 0.5);
  panX = Math.max(panX, W - gw - W * 0.5);
  panY = Math.min(panY, H * 0.5);
  panY = Math.max(panY, H - gh - H * 0.5);
}}

function draw() {{
  ctx.fillStyle = '#111';
  ctx.fillRect(0, 0, W, H);
  ctx.save();
  ctx.translate(panX, panY);
  ctx.scale(scale, scale);

  if (scale > 0.5) {{
    // High zoom: draw each cell as a pixel or small rect
    ctx.fillStyle = '#fff';
    const vpL = -panX / scale;
    const vpT = -panY / scale;
    const vpR = vpL + W / scale;
    const vpB = vpT + H / scale;
    const minX = Math.max(0, Math.floor(vpL));
    const minY = Math.max(0, Math.floor(vpT));
    const maxX = Math.min(GRID_W, Math.ceil(vpR));
    const maxY = Math.min(GRID_H, Math.ceil(vpB));
    const sz = Math.max(1, scale * 0.8);

    // Binary search or iterate — for sparsity, iterate all in view
    for (let i = 0; i < CELL_COUNT; i++) {{
      const x = cx[i], y = cy[i];
      if (x >= minX && x < maxX && y >= minY && y < maxY) {{
        ctx.fillRect(x, y, sz, sz);
      }}
    }}
  }} else {{
    // Low zoom: draw density heatmap
    const step = Math.max(1, Math.floor(1 / scale));
    const binW = Math.ceil(GRID_W / step);
    const binH = Math.ceil(GRID_H / step);
    const counts = new Uint16Array(Math.max(1, binW * binH));
    for (let i = 0; i < CELL_COUNT; i++) {{
      const bx = Math.floor(cx[i] / step);
      const by = Math.floor(cy[i] / step);
      counts[by * binW + bx]++;
    }}
    const maxC = Math.max(1, Math.max(...counts));
    for (let by = 0; by < binH; by++) {{
      for (let bx = 0; bx < binW; bx++) {{
        const v = counts[by * binW + bx];
        if (v > 0) {{
          const t = Math.min(v / maxC, 1);
          const r = Math.floor(255 * (1 - t));
          const g = Math.floor(255 * (1 - t * 0.7));
          ctx.fillStyle = `rgb(${{r}},${{g}},255)`;
          ctx.fillRect(bx * step, by * step, step, step);
        }}
      }}
    }}
  }}

  ctx.restore();
}}

canvas.onwheel = e => {{
  e.preventDefault();
  const oldScale = scale;
  const mx = e.clientX, my = e.clientY;
  scale *= (e.deltaY > 0 ? 0.85 : 1.18);
  scale = Math.max(0.005, Math.min(scale, 64));
  panX = mx - (mx - panX) * (scale / oldScale);
  panY = my - (my - panY) * (scale / oldScale);
  clampPan();
  draw();
}};

canvas.onmousedown = e => {{
  dragging = true;
  dragX = e.clientX; dragY = e.clientY;
  dragPanX = panX; dragPanY = panY;
}};

canvas.onmousemove = e => {{
  if (dragging) {{
    panX = dragPanX + (e.clientX - dragX);
    panY = dragPanY + (e.clientY - dragY);
    draw();
  }}
}};

canvas.onmouseup = canvas.onmouseleave = () => {{ dragging = false; }};

window.onresize = () => {{ resize(); draw(); }};

// Fit grid to screen initially
function fitToScreen() {{
  resize();
  const sx = W / GRID_W, sy = H / GRID_H;
  scale = Math.min(sx, sy) * 0.9;
  panX = (W - GRID_W * scale) / 2;
  panY = (H - GRID_H * scale) / 2;
  draw();
}}

fitToScreen();
document.title = `GoL Viewer (${{CELL_COUNT.toLocaleString()}} cells)`;
</script>
</body>
</html>'''

with open(out_path, 'w') as f:
    f.write(html)

print(f'Viewer written to {out_path}  ({len(html):,} bytes, {len(cells):,} cells embedded)')
