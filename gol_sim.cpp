/**
 * Conway's Game of Life — Sim Core + Age-Colored Video Export
 *
 * Build:
 *   g++ -O3 -march=native -mavx2 -fopenmp -std=c++17 -o gol gol_sim.cpp
 *
 * Modes:
 *   bench <grid_w> <grid_h> <gens> [threads]
 *   video <grid_w> <grid_h> <gens> <fps> <render_w> <render_h> [threads] [state_file] [rules]
 *   sim   <config_file> [threads]
 *
 * Config file format (key = value, # comments):
 *   grid_w = 15000
 *   grid_h = 15000
 *   rules = B3/S23
 *   state_file = start.cells
 *   video_w = 1920
 *   video_h = 1080
 *   video_fps = 60
 *   max_gens = 100000
 *   stagnation = 1
 *   detect_extinction = 1
 *   detect_cycle = 1
 *   threads = 8
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
#include <unordered_map>
#include <unordered_set>
#include <immintrin.h>
#include <omp.h>

#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#include <filesystem>
namespace fs = std::filesystem;
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

struct Config {
    int gw = 1000, gh = 1000;
    uint8_t B = 8, S = 12;           // B3/S23 by default
    std::string state_file;
    int vw = 0, vh = 0, vfps = 60;
    int max_gens = 100000;
    bool stag_enable = true;
    bool stag_extinct = true;
    bool stag_cycle = true;
    int threads = 0;                 // 0 = auto

    bool has_video() const { return vw > 0 && vh > 0; }
};

static bool parse_rules(const std::string& s, uint8_t& B, uint8_t& S) {
    B = S = 0;
    size_t slash = s.find('/');
    if (slash == std::string::npos) return false;

    auto parse_set = [](const std::string& part, uint8_t& mask) {
        mask = 0;
        for (size_t i = 0; i < part.size(); i++) {
            if (part[i] >= '0' && part[i] <= '8')
                mask |= (1 << (part[i] - '0'));
        }
    };

    std::string Bpart = s.substr(0, slash);
    std::string Spart = s.substr(slash + 1);

    if (!Bpart.empty() && (Bpart[0] == 'B' || Bpart[0] == 'b'))
        parse_set(Bpart.substr(1), B);
    else
        parse_set(Bpart, B);

    if (!Spart.empty() && (Spart[0] == 'S' || Spart[0] == 's'))
        parse_set(Spart.substr(1), S);
    else
        parse_set(Spart, S);

    return true;
}

static bool parse_config(const std::string& path, Config& cfg) {
    FILE* f = fopen(path.c_str(), "r");
    if (!f) { fprintf(stderr, "Error: cannot open config '%s'\n", path.c_str()); return false; }

    char line[1024];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
        char key[256] = {0}, val[768] = {0};
        if (sscanf(line, " %255s = %767[^\n\r]", key, val) < 2) continue;

        std::string k(key), v(val);

        if (k == "grid_w") cfg.gw = atoi(v.c_str());
        else if (k == "grid_h") cfg.gh = atoi(v.c_str());
        else if (k == "rules") parse_rules(v, cfg.B, cfg.S);
        else if (k == "state_file") cfg.state_file = v;
        else if (k == "video_w") cfg.vw = atoi(v.c_str());
        else if (k == "video_h") cfg.vh = atoi(v.c_str());
        else if (k == "video_fps") cfg.vfps = atoi(v.c_str());
        else if (k == "max_gens") cfg.max_gens = atoi(v.c_str());
        else if (k == "stagnation") cfg.stag_enable = atoi(v.c_str()) != 0;
        else if (k == "detect_extinction") cfg.stag_extinct = atoi(v.c_str()) != 0;
        else if (k == "detect_cycle") cfg.stag_cycle = atoi(v.c_str()) != 0;
        else if (k == "threads") cfg.threads = atoi(v.c_str());
    }
    fclose(f);

    if (cfg.threads <= 0) cfg.threads = omp_get_max_threads();
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Bitpacked grid + age grid
// ─────────────────────────────────────────────────────────────────────────────

struct Grid {
    uint64_t* bits;
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

    inline uint64_t* brow(int r) {
        return bits + (size_t)(r + 1) * words_per_row + 1;
    }
    inline const uint64_t* brow(int r) const {
        return bits + (size_t)(r + 1) * words_per_row + 1;
    }

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
    inline void set_with_age(int r, int c, int v) {
        set(r, c, v);
        arow(r)[c] = v ? 1 : 0;
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
// Generalised bitwise neighbour counting + rule application
// ─────────────────────────────────────────────────────────────────────────────

#define HALF_ADD(s,c,a,b)     { s=(a)^(b); c=(a)&(b); }
#define FULL_ADD(s,c,a,b,cin) { uint64_t _t=(a)^(b); s=_t^(cin); c=((a)&(b))|((_t)&(cin)); }

static inline uint64_t next_word_rules(
    uint64_t tl, uint64_t tc, uint64_t tr,
    uint64_t ml, uint64_t mc, uint64_t mr,
    uint64_t bl, uint64_t bc, uint64_t br,
    uint8_t B, uint8_t S)
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

    // Decode all 8 count masks from 3-bit binary (a2,a1,a0)
    uint64_t cnt[8];
    cnt[0] = ~a2 & ~a1 & ~a0;
    cnt[1] = ~a2 & ~a1 &  a0;
    cnt[2] = ~a2 &  a1 & ~a0;
    cnt[3] = ~a2 &  a1 &  a0;
    cnt[4] =  a2 & ~a1 & ~a0;
    cnt[5] =  a2 & ~a1 &  a0;
    cnt[6] =  a2 &  a1 & ~a0;
    cnt[7] =  a2 &  a1 &  a0;

    uint64_t birth = 0, survive = 0;
    if (B & 1)   birth   |= cnt[0];
    if (B & 2)   birth   |= cnt[1];
    if (B & 4)   birth   |= cnt[2];
    if (B & 8)   birth   |= cnt[3];
    if (B & 16)  birth   |= cnt[4];
    if (B & 32)  birth   |= cnt[5];
    if (B & 64)  birth   |= cnt[6];
    if (B & 128) birth   |= cnt[7];

    if (S & 1)   survive |= cnt[0];
    if (S & 2)   survive |= cnt[1];
    if (S & 4)   survive |= cnt[2];
    if (S & 8)   survive |= cnt[3];
    if (S & 16)  survive |= cnt[4];
    if (S & 32)  survive |= cnt[5];
    if (S & 64)  survive |= cnt[6];
    if (S & 128) survive |= cnt[7];

    return (birth & ~mc) | (survive & mc);
}

// ─────────────────────────────────────────────────────────────────────────────
// Step: advance one generation with arbitrary B/S rules
// ─────────────────────────────────────────────────────────────────────────────

struct StepResult {
    long long live_count;
};

StepResult step_rules(Grid& cur, Grid& nxt, uint8_t rule_B, uint8_t rule_S) {
    int rw = (cur.W + 63) / 64;

    StepResult res = {0};
    long long live_total = 0;

    // Pass 1: update bitpacked state
    #pragma omp parallel for schedule(static) reduction(+:live_total)
    for (int r = 0; r < cur.H; r++) {
        const uint64_t* top = cur.brow(r-1);
        const uint64_t* mid = cur.brow(r);
        const uint64_t* bot = cur.brow(r+1);
        uint64_t*       out = nxt.brow(r);
        for (int w = 0; w < rw; w++) {
            out[w] = next_word_rules(
                top[w-1], top[w], top[w+1],
                mid[w-1], mid[w], mid[w+1],
                bot[w-1], bot[w], bot[w+1],
                rule_B, rule_S);
            live_total += __builtin_popcountll(out[w]);
        }
    }
    res.live_count = live_total;

    // Pass 2: update age grid (separate cache-friendly pass)
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

    return res;
}

// ─────────────────────────────────────────────────────────────────────────────
// Random fill
// ─────────────────────────────────────────────────────────────────────────────

void random_fill(Grid& g, float density = 0.5f) {
    srand(42);
    for (int r = 0; r < g.H; r++)
        for (int c = 0; c < g.W; c++) {
            int alive = (rand() / (float)RAND_MAX) < density ? 1 : 0;
            g.set_with_age(r, c, alive);
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
// Age -> color ramp
// ─────────────────────────────────────────────────────────────────────────────

struct RGB { uint8_t r, g, b; };

static RGB age_lut[129];

static void build_age_lut() {
    struct Stop { float t; uint8_t r, g, b; };
    static const Stop stops[] = {
        {0.00f, 255, 255, 255},
        {0.15f, 255, 255,   0},
        {0.38f, 255, 140,   0},
        {0.60f,   0, 200,  60},
        {0.78f,   0,  80, 255},
        {1.00f,   0,  20,  80},
    };
    static const int N = 6;

    age_lut[0] = {0, 0, 0};

    for (int age = 1; age <= 128; age++) {
        float t = (age - 1) / 127.0f;
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
// Renderer: grid -> RGB24 frame
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
// State file loader (.cells plaintext format)
// ─────────────────────────────────────────────────────────────────────────────

bool load_cells(Grid& g, const std::string& path, int off_x = 0, int off_y = 0) {
    FILE* f = fopen(path.c_str(), "r");
    if (!f) { fprintf(stderr, "Error: cannot open state file '%s'\n", path.c_str()); return false; }

    char line[4096];
    int r = off_y;
    while (fgets(line, sizeof(line), f) && r < g.H) {
        if (line[0] == '!' || line[0] == '#') continue;
        if (line[0] == 'x' && line[1] == ' ') continue; // RLE header, ignore

        int len = (int)strlen(line);
        for (int c = off_x, i = 0; c < g.W && i < len; i++) {
            char ch = line[i];
            if (ch == 'O' || ch == 'o')  { g.set_with_age(r, c, 1); c++; }
            else if (ch == '.')           { g.set_with_age(r, c, 0); c++; }
            else if (ch == '$')           break;
        }
        r++;
    }
    fclose(f);
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Fast hash for stagnation detection (word-wise XOR with golden ratio mixing)
// ─────────────────────────────────────────────────────────────────────────────

static uint64_t xor_hash(const Grid& g) {
    int rw = (g.W + 63) / 64;
    uint64_t h = 0;
    for (int r = 0; r < g.H; r++) {
        const uint64_t* row = g.brow(r);
        for (int w = 0; w < rw; w++) {
            h ^= row[w];
            h *= 0x9E3779B97F4A7C15ULL;
        }
    }
    return h;
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
        step_rules(A, B, 8, 12);
        sync_borders(B);
        std::swap(A.bits, B.bits);
        std::swap(A.age,  B.age);
    }

    fprintf(stderr, "Benchmarking...\n");
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < GENS; i++) {
        step_rules(A, B, 8, 12);
        sync_borders(B);
        std::swap(A.bits, B.bits);
        std::swap(A.age,  B.age);
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
        fprintf(stderr, "  @ %3.0f GPS -> ~%.0f x %.0f cells\n",
                tgt, sqrt(gcups*1e9/tgt), sqrt(gcups*1e9/tgt));
}

// ─────────────────────────────────────────────────────────────────────────────
// Video mode (extended: optional state_file + rules)
// ─────────────────────────────────────────────────────────────────────────────

void run_video(int W, int H, int GENS, int FPS, int rW, int rH,
               int THREADS, const std::string& state_file,
               uint8_t rule_B, uint8_t rule_S)
{
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
    fprintf(stderr, "Rules:         ");
    for (int i = 0; i <= 8; i++) if (rule_B & (1<<i)) fprintf(stderr, "%d", i);
    fprintf(stderr, "/");
    for (int i = 0; i <= 8; i++) if (rule_S & (1<<i)) fprintf(stderr, "%d", i);
    fprintf(stderr, "\n");
    fprintf(stderr, "Render:        %d x %d @ %d fps\n", rW, rH, FPS);
    fprintf(stderr, "Coverage:      %.2f x %.2f cells per pixel\n",
            cells_per_px_x, cells_per_px_y);
    fprintf(stderr, "Duration:      %d gens = %.1f sec video\n\n", GENS, (double)GENS/FPS);
    fprintf(stderr, "FFmpeg:\n");
    fprintf(stderr,
        "  [this] | ffmpeg -f rawvideo -pixel_format rgb24 \\\n"
        "      -video_size %dx%d -framerate %d -i - \\\n"
        "      -c:v libx264 -profile:v baseline -level 4.0 \\\n"
        "      -pix_fmt yuv420p -preset fast -crf 18 life.mp4\n\n",
        rW, rH, FPS);

    Grid A(W, H), N(W, H);

    if (!state_file.empty()) {
        fprintf(stderr, "Loading state: %s\n", state_file.c_str());
        load_cells(A, state_file);
    } else {
        random_fill(A, 0.5f);
    }

    long long initial_live = count_live(A);
    fprintf(stderr, "Initial live cells: %lld (%.1f%%)\n", initial_live, 100.0 * initial_live / ((long long)W*H));
    sync_borders(A);

    std::vector<uint8_t> frame_buf;
    size_t frame_bytes = (size_t)rW * rH * 3;

    auto wall0 = std::chrono::high_resolution_clock::now();

    for (int gen = 0; gen < GENS; gen++) {
        render_frame(A, rW, rH, frame_buf);
        fwrite(frame_buf.data(), 1, frame_bytes, stdout);
        fflush(stdout);

        step_rules(A, N, rule_B, rule_S);
        sync_borders(N);
        std::swap(A.bits, N.bits);
        std::swap(A.age,  N.age);

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
// Sim mode: config-driven, runs until stagnation
// ─────────────────────────────────────────────────────────────────────────────

int run_sim(const Config& cfg) {
    omp_set_num_threads(cfg.threads);
    build_age_lut();

    fprintf(stderr, "=== GoL Simulation ===\n");
    fprintf(stderr, "Grid:   %d x %d  (%.2f MB bits + %.2f MB age)\n",
            cfg.gw, cfg.gh,
            (double)cfg.gw*cfg.gh/8.0/1024/1024,
            (double)cfg.gw*cfg.gh/1024/1024);
    fprintf(stderr, "Rules:  ");
    for (int i = 0; i <= 8; i++) if (cfg.B & (1<<i)) fprintf(stderr, "%d", i);
    fprintf(stderr, "/");
    for (int i = 0; i <= 8; i++) if (cfg.S & (1<<i)) fprintf(stderr, "%d", i);
    fprintf(stderr, "\n");
    fprintf(stderr, "Max:    %d gens\n", cfg.max_gens);
    fprintf(stderr, "Stag:   %s  cycle=%s  extinct=%s\n",
            cfg.stag_enable ? "on" : "off",
            cfg.stag_cycle ? "on" : "off",
            cfg.stag_extinct ? "on" : "off");
    fprintf(stderr, "Threads: %d\n", cfg.threads);

    if (cfg.has_video()) {
        fprintf(stderr, "Video:  %dx%d @ %d fps\n", cfg.vw, cfg.vh, cfg.vfps);
#ifdef _WIN32
        _setmode(_fileno(stdout), _O_BINARY);
#endif
    }

    Grid A(cfg.gw, cfg.gh), B(cfg.gw, cfg.gh);

    if (!cfg.state_file.empty()) {
        fprintf(stderr, "\nLoading state: %s\n", cfg.state_file.c_str());
        if (!load_cells(A, cfg.state_file)) return 1;
    } else {
        random_fill(A, 0.5f);
    }

    long long initial_live = count_live(A);
    fprintf(stderr, "Initial live: %lld (%.1f%%)\n",
            initial_live, 100.0 * initial_live / ((long long)cfg.gw * cfg.gh));
    sync_borders(A);

    bool has_vid = cfg.has_video();
    std::vector<uint8_t> frame_buf;
    size_t frame_bytes = 0;
    if (has_vid) {
        frame_bytes = (size_t)cfg.vw * cfg.vh * 3;
    }

    // Stagnation detection state
    std::unordered_map<uint64_t, int> hash_to_gen;
    hash_to_gen.reserve(std::min(cfg.max_gens, 10000));

    int stop_gen = -1;
    std::string stop_reason;
    auto last_report = std::chrono::high_resolution_clock::now();

    // Render initial frame
    if (has_vid) {
        render_frame(A, cfg.vw, cfg.vh, frame_buf);
        fwrite(frame_buf.data(), 1, frame_bytes, stdout);
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int gen = 1; gen <= cfg.max_gens; gen++) {
        StepResult res = step_rules(A, B, cfg.B, cfg.S);
        sync_borders(B);

        // Stagnation checks
        if (cfg.stag_enable) {
            // Extinction
            if (cfg.stag_extinct && res.live_count == 0) {
                stop_gen = gen;
                stop_reason = "Extinction: all cells dead";
                // Render final frame
                if (has_vid) {
                    render_frame(B, cfg.vw, cfg.vh, frame_buf);
                    fwrite(frame_buf.data(), 1, frame_bytes, stdout);
                    fflush(stdout);
                }
                std::swap(A.bits, B.bits);
                std::swap(A.age,  B.age);
                break;
            }

            // Cycle detection via hash
            if (cfg.stag_cycle) {
                uint64_t h = xor_hash(B);
                auto it = hash_to_gen.find(h);
                if (it != hash_to_gen.end()) {
                    stop_gen = gen;
                    int period = gen - it->second;
                    char buf[128];
                    snprintf(buf, sizeof(buf),
                             "Cycle detected: period %d (gen %d = gen %d)",
                             period, gen, it->second);
                    stop_reason = buf;
                    // Render final frame
                    if (has_vid) {
                        render_frame(B, cfg.vw, cfg.vh, frame_buf);
                        fwrite(frame_buf.data(), 1, frame_bytes, stdout);
                        fflush(stdout);
                    }
                    std::swap(A.bits, B.bits);
                    std::swap(A.age,  B.age);
                    break;
                }
                hash_to_gen[h] = gen;
            }
        }

        std::swap(A.bits, B.bits);
        std::swap(A.age,  B.age);

        // Render current state (gen 1, 2, ...)
        if (has_vid) {
            render_frame(A, cfg.vw, cfg.vh, frame_buf);
            fwrite(frame_buf.data(), 1, frame_bytes, stdout);
            fflush(stdout);
        }

        // Progress report
        auto now = std::chrono::high_resolution_clock::now();
        double since_last = std::chrono::duration<double>(now - last_report).count();
        if (since_last >= 1.0 || gen == cfg.max_gens) {
            double elapsed = std::chrono::duration<double>(now - t0).count();
            double gps = gen / elapsed;
            fprintf(stderr, "  Gen %6d/%d  %7.1f GPS  live: %lld\r",
                    gen, cfg.max_gens, gps, (long long)res.live_count);
            last_report = now;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1-t0).count();
    int total_gens = stop_gen > 0 ? stop_gen : cfg.max_gens;
    double gps = total_gens / elapsed;

    fprintf(stderr, "\n\n=== Simulation Complete ===\n");
    fprintf(stderr, "Gens run: %d  Time: %.3fs  GPS: %.1f\n",
            total_gens, elapsed, gps);
    if (!stop_reason.empty())
        fprintf(stderr, "Stop: %s\n", stop_reason.c_str());
    else
        fprintf(stderr, "Stop: max_gens reached (%d)\n", cfg.max_gens);

    if (has_vid) {
        fprintf(stderr, "\nVideo: %d frames sent to stdout (%.1fs at %d fps)\n",
                total_gens + 1, (double)(total_gens + 1) / cfg.vfps, cfg.vfps);
    }

    // Debug dump for small grids
    if (cfg.gw <= 64 && cfg.gh <= 64) {
        fprintf(stderr, "\nFinal grid state:\n");
        for (int r = 0; r < cfg.gh && r < 16; r++) {
            fprintf(stderr, "  ");
            for (int c = 0; c < cfg.gw && c < 64; c++) {
                fprintf(stderr, "%c", A.get(r, c) ? 'O' : '.');
            }
            fprintf(stderr, "\n");
        }
    }

    return 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

void usage(const char* prog) {
    fprintf(stderr,
        "Usage:\n"
        "  %s bench <grid_w> <grid_h> <gens> [threads]\n"
        "  %s video <grid_w> <grid_h> <gens> <fps> <render_w> <render_h> [threads] [state_file] [rules]\n"
        "  %s sim   <config_file> [threads]\n"
        "\n"
        "Video examples:\n"
        "  %s video 15000 15000 600 60 1920 1080 8 | ffmpeg -f rawvideo -pixel_format rgb24 -video_size 1920x1080 -framerate 60 -i - -c:v libx264 -profile:v baseline -level 4.0 -pix_fmt yuv420p -preset fast -crf 18 life.mp4\n"
        "  %s video 15000 15000 600 60 1920 1080 8 start.cells B3/S23 | ffmpeg ...\n"
        "\n"
        "Sim with config file:\n"
        "  %s sim my_config.txt\n"
        "  %s sim my_config.txt 8\n",
        prog,prog,prog,prog,prog,prog,prog);
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

        std::string state_file;
        uint8_t B = 8, S = 12;

        if (argc > 8) {
            std::string arg8 = argv[8];
            // Check if it looks like a file path (has extension or contains path separators)
            if (arg8.find('.') != std::string::npos ||
                arg8.find('/') != std::string::npos ||
                arg8.find('\\') != std::string::npos) {
                state_file = arg8;
                if (argc > 9) parse_rules(argv[9], B, S);
            } else {
                // It's a thread count (or rules string starting with B)
                if (arg8[0] == 'B' || arg8[0] == 'b')
                    parse_rules(arg8, B, S);
            }
        }
        if (argc > 9) {
            std::string arg9 = argv[9];
            if (arg9[0] == 'B' || arg9[0] == 'b')
                parse_rules(arg9, B, S);
        }

        int threads = argc > 10 ? atoi(argv[10]) : omp_get_max_threads();

        run_video(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]),
                  atoi(argv[5]), atoi(argv[6]), atoi(argv[7]),
                  threads, state_file, B, S);

    } else if (mode == "sim") {
        if (argc < 3) { usage(argv[0]); return 1; }

        Config cfg;
        if (!parse_config(argv[2], cfg)) return 1;
        if (argc > 3) cfg.threads = atoi(argv[3]);

        return run_sim(cfg);

    } else {
        usage(argv[0]); return 1;
    }
    return 0;
}
