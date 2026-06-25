/**
 * Conway's Game of Life — Sim Core + Age-Colored Video Export
 *
 * Build:
 *   g++ -O3 -march=native -mavx2 -fopenmp -std=c++17 -o gol gol_sim.cpp
 *
 * Modes:
 *   gol bench <grid_w> <grid_h> <gens> [threads]
 *   gol video <grid_w> <grid_h> <gens> <fps> <render_w> <render_h> [threads]
 *
 * Video pipe (copy-paste ready):
 *   gol video 15000 15000 600 60 1920 1080 8 | ffmpeg -f rawvideo -pixel_format rgb24 -video_size 1920x1080 -framerate 60 -i - -c:v libx264 -profile:v baseline -level 4.0 -pix_fmt yuv420p -preset fast -crf 18 life.mp4
 *
 * Age color ramp (gen 1 → 128+):
 *   1   → white
 *   ~20 → yellow
 *   ~50 → orange
 *   ~80 → green
 *   ~100→ blue
 *   128+→ dark blue
 *   dead→ black
 */

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <string>
#include <vector>
#include <algorithm>
#include <immintrin.h>
#include <omp.h>

#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Bitpacked grid for fast simulation (1 bit per cell)
// Age grid (uint8_t per cell, 0=dead, 1–255=alive gens, capped at 128)
// ─────────────────────────────────────────────────────────────────────────────

struct Grid {
    // Bitpacked sim state
    uint64_t* bits;
    // Age grid: 0 = dead, N = alive for N gens (capped at 128)
    uint8_t*  age;

    int W, H;
    int words_per_row; // ceil(W/64) + 2 pad words

    Grid(int w, int h) : W(w), H(h) {
        words_per_row = (w + 63) / 64 + 2;
        size_t bit_total = (size_t)(h + 2) * words_per_row;
        size_t age_total = (size_t)h * w;

#ifdef _WIN32
        bits = (uint64_t*)_aligned_malloc(bit_total * sizeof(uint64_t), 32);
        age  = (uint8_t*) _aligned_malloc(age_total * sizeof(uint8_t),  32);
#else
        bits = (uint64_t*)aligned_alloc(32, bit_total * sizeof(uint64_t));
        age  = (uint8_t*) aligned_alloc(32, age_total * sizeof(uint8_t));
#endif
        memset(bits, 0, bit_total * sizeof(uint64_t));
        memset(age,  0, age_total * sizeof(uint8_t));
    }

    ~Grid() {
#ifdef _WIN32
        _aligned_free(bits);
        _aligned_free(age);
#else
        free(bits);
        free(age);
#endif
    }

    // Bit row accessors (with ghost padding)
    inline uint64_t* brow(int r) {
        return bits + (size_t)(r + 1) * words_per_row + 1;
    }
    inline const uint64_t* brow(int r) const {
        return bits + (size_t)(r + 1) * words_per_row + 1;
    }

    // Age accessors (no padding needed, computed directly)
    inline uint8_t* arow(int r) {
        return age + (size_t)r * W;
    }
    inline const uint8_t* arow(int r) const {
        return age + (size_t)r * W;
    }

    inline int  get(int r, int c) const { return (brow(r)[c/64] >> (c%64)) & 1; }
    inline void set(int r, int c, int v) {
        uint64_t& w = brow(r)[c/64];
        w = (w & ~(1ULL << (c%64))) | ((uint64_t)v << (c%64));
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Border sync (toroidal)
// ─────────────────────────────────────────────────────────────────────────────

void sync_borders(Grid& g) {
    int H = g.H, wpr = g.words_per_row;
    int rw = (g.W + 63) / 64;

    memcpy(g.brow(-1), g.brow(H-1), wpr * sizeof(uint64_t));
    memcpy(g.brow(H),  g.brow(0),   wpr * sizeof(uint64_t));

    for (int r = -1; r <= H; r++) {
        uint64_t* rp = g.brow(r);
        rp[-1] = rp[rw-1];
        rp[rw] = rp[0];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Bitwise neighbour counting (Abrash carry-save)
// ─────────────────────────────────────────────────────────────────────────────

#define HALF_ADD(s,c,a,b)     { s=(a)^(b); c=(a)&(b); }
#define FULL_ADD(s,c,a,b,cin) { uint64_t _t=(a)^(b); s=_t^(cin); c=((a)&(b))|((_t)&(cin)); }

inline uint64_t next_word(
    uint64_t tl, uint64_t tc, uint64_t tr,
    uint64_t ml, uint64_t mc, uint64_t mr,
    uint64_t bl, uint64_t bc, uint64_t br)
{
    auto hor = [](uint64_t L, uint64_t C, uint64_t R, uint64_t& s, uint64_t& c) {
        uint64_t left  = (C << 1) | (L >> 63);
        uint64_t right = (C >> 1) | (R << 63);
        HALF_ADD(s, c, left, right);
    };
    uint64_t ts,tc2, ms,mc2, bs,bc2;
    hor(tl,tc,tr, ts,tc2);
    hor(ml,mc,mr, ms,mc2);
    hor(bl,bc,br, bs,bc2);

    uint64_t s0,c0,s1,c1;
    HALF_ADD(s0,c0,ts,ms);
    FULL_ADD(s1,c1,s0,bs,0ULL);

    uint64_t h0,hc0,h1,carry;
    HALF_ADD(h0,hc0,tc2,mc2);
    h1 = h0 ^ bc2;

    uint64_t a0,a1,a2;
    a0 = s1;
    HALF_ADD(a1,carry,c1,h0);
    a2 = carry ^ h1 ^ hc0;

    uint64_t count2 = ~a2 & a1 & ~a0;
    uint64_t count3 = ~a2 & a1 &  a0;
    return count3 | (count2 & mc);
}

// ─────────────────────────────────────────────────────────────────────────────
// Step: advance one generation, update bit grid + age grid
// ─────────────────────────────────────────────────────────────────────────────

void step(Grid& cur, Grid& nxt) {
    int rw = (cur.W + 63) / 64;

    // Pass 1: update bitpacked state (fast AVX2/bitwise path)
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < cur.H; r++) {
        const uint64_t* top = cur.brow(r-1);
        const uint64_t* mid = cur.brow(r);
        const uint64_t* bot = cur.brow(r+1);
        uint64_t*       out = nxt.brow(r);
        for (int w = 0; w < rw; w++) {
            out[w] = next_word(
                top[w-1], top[w], top[w+1],
                mid[w-1], mid[w], mid[w+1],
                bot[w-1], bot[w], bot[w+1]);
        }
    }

    // Pass 2: update age grid from new bit state (separate cache-friendly pass)
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < cur.H; r++) {
        const uint64_t* nbit = nxt.brow(r);
        const uint8_t*  cage = cur.arow(r);
        uint8_t*        nage = nxt.arow(r);
        int W = cur.W;

        for (int w = 0; w < rw; w++) {
            uint64_t alive = nbit[w];
            int base  = w * 64;
            int limit = std::min(64, W - base);
            if (alive == 0ULL) {
                memset(nage + base, 0, limit);
                continue;
            }
            for (int b = 0; b < limit; b++) {
                int cell = base + b;
                if ((alive >> b) & 1) {
                    uint8_t a = cage[cell];
                    nage[cell] = a >= 128 ? 128 : a + 1;
                } else {
                    nage[cell] = 0;
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Random fill
// ─────────────────────────────────────────────────────────────────────────────

void random_fill(Grid& g, float density = 0.5f) {
    srand(42);
    for (int r = 0; r < g.H; r++)
        for (int c = 0; c < g.W; c++) {
            int alive = (rand() / (float)RAND_MAX) < density ? 1 : 0;
            g.set(r, c, alive);
            g.arow(r)[c] = alive ? 1 : 0;
        }
}

long long count_live(const Grid& g) {
    long long total = 0;
    int rw = (g.W + 63) / 64;
    for (int r = 0; r < g.H; r++) {
        const uint64_t* rp = g.brow(r);
        for (int w = 0; w < rw; w++)
            total += __builtin_popcountll(rp[w]);
    }
    return total;
}

// ─────────────────────────────────────────────────────────────────────────────
// Age → color ramp
// age 0   = dead  → black
// age 1   → white
// age ~20 → yellow
// age ~50 → orange
// age ~80 → green
// age ~100→ blue
// age 128+→ dark blue
// ─────────────────────────────────────────────────────────────────────────────

struct RGB { uint8_t r, g, b; };

// Precomputed lookup table for ages 0–128
static RGB age_lut[129];

static void build_age_lut() {
    struct Stop { float t; uint8_t r, g, b; };
    // t is normalized age 0–1 (maps to age 1–128 for alive cells)
    static const Stop stops[] = {
        {0.00f, 255, 255, 255},  // age 1   : white
        {0.15f, 255, 255,   0},  // age ~20 : yellow
        {0.38f, 255, 140,   0},  // age ~50 : orange
        {0.60f,   0, 200,  60},  // age ~78 : green
        {0.78f,   0,  80, 255},  // age ~100: blue
        {1.00f,   0,  20,  80},  // age 128 : dark blue
    };
    static const int N = 6;

    age_lut[0] = {0, 0, 0}; // dead = black

    for (int age = 1; age <= 128; age++) {
        float t = (age - 1) / 127.0f; // 0..1
        RGB c = {255, 255, 255};
        for (int i = 1; i < N; i++) {
            if (t <= stops[i].t) {
                float s = (t - stops[i-1].t) / (stops[i].t - stops[i-1].t);
                c.r = (uint8_t)(stops[i-1].r + s * (stops[i].r - stops[i-1].r));
                c.g = (uint8_t)(stops[i-1].g + s * (stops[i].g - stops[i-1].g));
                c.b = (uint8_t)(stops[i-1].b + s * (stops[i].b - stops[i-1].b));
                break;
            }
        }
        age_lut[age] = c;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Renderer: grid → RGB24 frame
//
// If render_w == grid_w and render_h == grid_h: 1:1 pixel mapping
// Otherwise: each output pixel covers a ceil block of grid cells.
//   Color = age_lut[max age in block]  (oldest cell dominates, looks best)
// ─────────────────────────────────────────────────────────────────────────────

void render_frame(const Grid& g, int rW, int rH, std::vector<uint8_t>& buf) {
    buf.resize((size_t)rW * rH * 3);

    #pragma omp parallel for schedule(static)
    for (int py = 0; py < rH; py++) {
        int gr0 = (int)((double)py       * g.H / rH);
        int gr1 = (int)((double)(py + 1) * g.H / rH);
        if (gr1 <= gr0) gr1 = gr0 + 1;
        if (gr1 > g.H)  gr1 = g.H;

        uint8_t* out_row = buf.data() + (size_t)py * rW * 3;

        for (int px = 0; px < rW; px++) {
            int gc0 = (int)((double)px       * g.W / rW);
            int gc1 = (int)((double)(px + 1) * g.W / rW);
            if (gc1 <= gc0) gc1 = gc0 + 1;
            if (gc1 > g.W)  gc1 = g.W;

            // Find max age in block (oldest cell drives the color)
            uint8_t max_age = 0;
            for (int gr = gr0; gr < gr1 && max_age < 128; gr++) {
                const uint8_t* arow = g.arow(gr);
                for (int gc = gc0; gc < gc1; gc++) {
                    if (arow[gc] > max_age) max_age = arow[gc];
                }
            }

            RGB c = age_lut[max_age];
            out_row[px * 3 + 0] = c.r;
            out_row[px * 3 + 1] = c.g;
            out_row[px * 3 + 2] = c.b;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark mode
// ─────────────────────────────────────────────────────────────────────────────

void run_bench(int W, int H, int GENS, int THREADS) {
    omp_set_num_threads(THREADS);

    fprintf(stderr, "=== GoL Benchmark ===\n");
    fprintf(stderr, "Grid: %d x %d  (%.2f MB bits + %.2f MB age)\n",
            W, H,
            (double)W*H/8.0/1024/1024,
            (double)W*H/1024/1024);
    fprintf(stderr, "Gens: %d  Threads: %d\n\n", GENS, THREADS);

    Grid A(W, H), B(W, H);
    random_fill(A, 0.5f);
    sync_borders(A);

    fprintf(stderr, "Warmup (10 gens)...\n");
    for (int i = 0; i < 10; i++) {
        step(A, B); sync_borders(B); std::swap(A.bits, B.bits); std::swap(A.age, B.age);
    }

    fprintf(stderr, "Benchmarking...\n");
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < GENS; i++) {
        step(A, B); sync_borders(B); std::swap(A.bits, B.bits); std::swap(A.age, B.age);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double>(t1-t0).count();
    double gps   = GENS / elapsed;
    double gcups = (double)W*H*GENS / elapsed / 1e9;

    fprintf(stderr, "\n=== Results ===\n");
    fprintf(stderr, "Time:  %.3f s\n", elapsed);
    fprintf(stderr, "GPS:   %.1f  (gens/sec)\n", gps);
    fprintf(stderr, "GCUPS: %.2f  (billion cell-updates/sec)\n\n", gcups);

    fprintf(stderr, "=== Max square grid @ target GPS ===\n");
    for (double tgt : {240.0, 144.0, 60.0, 30.0})
        fprintf(stderr, "  @ %3.0f GPS → ~%.0f x %.0f cells\n",
                tgt, sqrt(gcups*1e9/tgt), sqrt(gcups*1e9/tgt));
}

// ─────────────────────────────────────────────────────────────────────────────
// Video mode
// ─────────────────────────────────────────────────────────────────────────────

void run_video(int W, int H, int GENS, int FPS, int rW, int rH, int THREADS) {
    omp_set_num_threads(THREADS);
    build_age_lut();

#ifdef _WIN32
    _setmode(_fileno(stdout), _O_BINARY);
#endif

    double cells_per_px_x = (double)W / rW;
    double cells_per_px_y = (double)H / rH;

    fprintf(stderr, "=== GoL Video Export ===\n");
    fprintf(stderr, "Grid:          %d x %d  (%.2f MB bits + %.2f MB age)\n",
            W, H, (double)W*H/8.0/1024/1024, (double)W*H/1024/1024);
    fprintf(stderr, "Render:        %d x %d @ %d fps\n", rW, rH, FPS);
    fprintf(stderr, "Coverage:      %.2f x %.2f cells per pixel\n",
            cells_per_px_x, cells_per_px_y);
    fprintf(stderr, "Duration:      %d gens = %.1f sec video\n\n", GENS, (double)GENS/FPS);
    fprintf(stderr, "FFmpeg command:\n");
    fprintf(stderr,
        "  [this] | ffmpeg -f rawvideo -pixel_format rgb24 \\\n"
        "      -video_size %dx%d -framerate %d -i - \\\n"
        "      -c:v libx264 -profile:v baseline -level 4.0 \\\n"
        "      -pix_fmt yuv420p -preset fast -crf 18 life.mp4\n\n",
        rW, rH, FPS);

    Grid A(W, H), B(W, H);
    random_fill(A, 0.5f);
    sync_borders(A);

    std::vector<uint8_t> frame_buf;
    size_t frame_bytes = (size_t)rW * rH * 3;

    auto wall0 = std::chrono::high_resolution_clock::now();

    for (int gen = 0; gen < GENS; gen++) {
        render_frame(A, rW, rH, frame_buf);
        fwrite(frame_buf.data(), 1, frame_bytes, stdout);
        fflush(stdout);

        step(A, B);
        sync_borders(B);
        std::swap(A.bits, B.bits);
        std::swap(A.age,  B.age);

        if (gen % 50 == 0 || gen == GENS-1) {
            auto now = std::chrono::high_resolution_clock::now();
            double el  = std::chrono::duration<double>(now - wall0).count();
            double fps = (gen+1) / el;
            double eta = (GENS - gen - 1) / fps;
            fprintf(stderr, "  Frame %5d/%d  %.1f fps  ETA %.0fs\n",
                    gen+1, GENS, fps, eta);
        }
    }

    auto wall1 = std::chrono::high_resolution_clock::now();
    double total = std::chrono::duration<double>(wall1-wall0).count();
    fprintf(stderr, "\nDone. %.1fs encode time for %.1fs of video (%.1fx realtime)\n",
            total, (double)GENS/FPS, (double)GENS/FPS/total);
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

void usage(const char* prog) {
    fprintf(stderr,
        "Usage:\n"
        "  %s bench <grid_w> <grid_h> <gens> [threads]\n"
        "  %s video <grid_w> <grid_h> <gens> <fps> <render_w> <render_h> [threads]\n"
        "\nExamples:\n"
        "  %s bench 15000 15000 200 8\n"
        "\n"
        "  # 1080p60, 10sec, 15K grid:\n"
        "  %s video 15000 15000 600 60 1920 1080 8 | ffmpeg -f rawvideo -pixel_format rgb24 -video_size 1920x1080 -framerate 60 -i - -c:v libx264 -profile:v baseline -level 4.0 -pix_fmt yuv420p -preset fast -crf 18 life.mp4\n"
        "\n"
        "  # 4K144, 6sec, 30K grid:\n"
        "  %s video 30000 30000 864 144 3840 2160 8 | ffmpeg -f rawvideo -pixel_format rgb24 -video_size 3840x2160 -framerate 144 -i - -c:v libx264 -profile:v baseline -level 4.0 -pix_fmt yuv420p -preset fast -crf 18 life_4k.mp4\n"
        "\n"
        "  # 1:1 pixel mapping (grid == render res):\n"
        "  %s video 1920 1080 600 60 1920 1080 8 | ffmpeg ...\n",
        prog,prog,prog,prog,prog,prog);
}

int main(int argc, char** argv) {
    if (argc < 2) { usage(argv[0]); return 1; }
    std::string mode = argv[1];

    if (mode == "bench") {
        if (argc < 5) { usage(argv[0]); return 1; }
        run_bench(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]),
                  argc > 5 ? atoi(argv[5]) : omp_get_max_threads());

    } else if (mode == "video") {
        if (argc < 8) { usage(argv[0]); return 1; }
        run_video(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]),
                  atoi(argv[5]), atoi(argv[6]), atoi(argv[7]),
                  argc > 8 ? atoi(argv[8]) : omp_get_max_threads());

    } else {
        usage(argv[0]); return 1;
    }
    return 0;
}
