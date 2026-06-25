/**
 * Conway's Game of Life — High-Performance Backend
 *
 * Build:
 *   g++ -O3 -march=native -mavx2 -fopenmp -std=c++17 -o gol gol_sim.cpp
 *
 * Modes:
 *   gol bench <grid_w> <grid_h> <gens> [threads]
 *   gol video <grid_w> <grid_h> <gens> <fps> <render_w> <render_h> [threads] [state.cells] [B3/S23]
 *   gol sim   <config.cfg> [threads]
 *   gol init  <config.cfg>           -- write a starter config file
 *
 * Config file (key = value, # comments):
 *   grid_w         = 15000
 *   grid_h         = 15000
 *   threads        = 8
 *   rules          = B3/S23
 *   state_file     = start.cells     # omit for random fill
 *   random_seed    = 42
 *   random_density = 0.5
 *   video_w        = 1920            # omit to disable video output
 *   video_h        = 1080
 *   video_fps      = 60
 *   max_gens       = 100000          # hard cap
 *   stagnation     = 1               # enable stagnation detection
 *   detect_extinction = 1
 *   detect_cycle   = 1
 *
 * Video pipe (Windows-compatible):
 *   gol video 15000 15000 600 60 1920 1080 8 | ffmpeg ^
 *     -f rawvideo -pixel_format rgb24 -video_size 1920x1080 -framerate 60 -i - ^
 *     -c:v libx264 -profile:v baseline -level 4.0 -pix_fmt yuv420p ^
 *     -preset fast -crf 18 life.mp4
 *
 *   (Linux: replace ^ with \)
 *
 * .cells format: lines of '.' and 'O', '!' = comment. Pattern is centered on grid.
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
#include <immintrin.h>
#include <omp.h>

#ifdef _WIN32
  #include <io.h>
  #include <fcntl.h>
  #define SET_STDOUT_BINARY() _setmode(_fileno(stdout), _O_BINARY)
#else
  #define SET_STDOUT_BINARY() ((void)0)
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────────────

struct Config {
    int   grid_w         = 1920;
    int   grid_h         = 1080;
    int   threads        = 0;        // 0 = auto
    uint8_t rule_B       = 8;        // B3
    uint8_t rule_S       = 12;       // S23
    std::string state_file;          // empty = random
    int   random_seed    = 42;
    float random_density = 0.5f;
    int   video_w        = 0;        // 0 = no video
    int   video_h        = 0;
    int   video_fps      = 60;
    int   max_gens       = 100000;
    bool  stag_enable    = true;
    bool  stag_extinct   = true;
    bool  stag_cycle     = true;

    bool has_video() const { return video_w > 0 && video_h > 0; }
};

static bool parse_rules(const std::string& s, uint8_t& B, uint8_t& S) {
    B = S = 0;
    // Accept: "B3/S23", "3/23", "b3/s23"
    size_t slash = s.find('/');
    if (slash == std::string::npos) return false;

    auto fill = [](const std::string& part, uint8_t& mask) {
        mask = 0;
        for (char ch : part)
            if (ch >= '0' && ch <= '8') mask |= (uint8_t)(1 << (ch - '0'));
    };

    std::string bp = s.substr(0, slash);
    std::string sp = s.substr(slash + 1);
    if (!bp.empty() && (bp[0]=='B'||bp[0]=='b')) bp = bp.substr(1);
    if (!sp.empty() && (sp[0]=='S'||sp[0]=='s')) sp = sp.substr(1);
    fill(bp, B);
    fill(sp, S);
    return true;
}

static std::string rules_str(uint8_t B, uint8_t S) {
    std::string r = "B";
    for (int i=0;i<=8;i++) if ((B>>i)&1) r += (char)('0'+i);
    r += "/S";
    for (int i=0;i<=8;i++) if ((S>>i)&1) r += (char)('0'+i);
    return r;
}

static bool parse_config(const std::string& path, Config& cfg) {
    FILE* f = fopen(path.c_str(), "r");
    if (!f) { fprintf(stderr, "Error: cannot open config '%s'\n", path.c_str()); return false; }
    char line[1024];
    while (fgets(line, sizeof(line), f)) {
        if (line[0]=='#'||line[0]=='\n'||line[0]=='\r') continue;
        char key[256]={}, val[768]={};
        if (sscanf(line, " %255[^= \t] = %767[^\n\r]", key, val) < 2) continue;
        // trim trailing spaces from val
        int vlen = (int)strlen(val);
        while (vlen > 0 && (val[vlen-1]==' '||val[vlen-1]=='\t')) val[--vlen]=0;

        std::string k(key), v(val);
        if      (k=="grid_w")          cfg.grid_w         = atoi(v.c_str());
        else if (k=="grid_h")          cfg.grid_h         = atoi(v.c_str());
        else if (k=="threads")         cfg.threads        = atoi(v.c_str());
        else if (k=="rules")           parse_rules(v, cfg.rule_B, cfg.rule_S);
        else if (k=="state_file")      cfg.state_file     = v;
        else if (k=="random_seed")     cfg.random_seed    = atoi(v.c_str());
        else if (k=="random_density")  cfg.random_density = (float)atof(v.c_str());
        else if (k=="video_w")         cfg.video_w        = atoi(v.c_str());
        else if (k=="video_h")         cfg.video_h        = atoi(v.c_str());
        else if (k=="video_fps")       cfg.video_fps      = atoi(v.c_str());
        else if (k=="max_gens")        cfg.max_gens       = atoi(v.c_str());
        else if (k=="stagnation")      cfg.stag_enable    = atoi(v.c_str())!=0;
        else if (k=="detect_extinction") cfg.stag_extinct = atoi(v.c_str())!=0;
        else if (k=="detect_cycle")    cfg.stag_cycle     = atoi(v.c_str())!=0;
    }
    fclose(f);
    if (cfg.threads <= 0) cfg.threads = omp_get_max_threads();
    return true;
}

static void write_default_config(const std::string& path) {
    FILE* f = fopen(path.c_str(), "w");
    if (!f) { fprintf(stderr, "Cannot write config: %s\n", path.c_str()); return; }
    fprintf(f,
        "# Conway's Game of Life — config file\n"
        "# Edit and pass to: gol sim <this_file>\n\n"
        "grid_w          = 1920\n"
        "grid_h          = 1080\n"
        "threads         = 0          # 0 = auto-detect\n\n"
        "rules           = B3/S23     # standard Conway\n"
        "# other rules: B36/S23 (HighLife), B2/S (Seeds), B3/S12345 (Maze)\n\n"
        "state_file      =            # leave empty for random fill\n"
        "random_seed     = 42\n"
        "random_density  = 0.5\n\n"
        "# video output (pipe stdout to ffmpeg — see top of gol_sim.cpp)\n"
        "video_w         = 1920\n"
        "video_h         = 1080\n"
        "video_fps       = 60\n\n"
        "max_gens        = 100000\n"
        "stagnation      = 1\n"
        "detect_extinction = 1\n"
        "detect_cycle    = 1\n"
    );
    fclose(f);
    fprintf(stderr, "Wrote default config: %s\n", path.c_str());
}

// ─────────────────────────────────────────────────────────────────────────────
// Grid
// ─────────────────────────────────────────────────────────────────────────────

struct Grid {
    uint64_t* bits;
    uint8_t*  age;
    int W, H;
    int words_per_row; // ceil(W/64) + 2 pad

    Grid(int w, int h) : W(w), H(h) {
        words_per_row = (w + 63) / 64 + 2;
        size_t bsz = (size_t)(h + 2) * words_per_row * sizeof(uint64_t);
        // age_total must be multiple of 32 for aligned_alloc on Linux
        size_t asz = ((size_t)h * w + 31) & ~(size_t)31;
#ifdef _WIN32
        bits = (uint64_t*)_aligned_malloc(bsz, 32);
        age  = (uint8_t*) _aligned_malloc(asz, 32);
#else
        bits = (uint64_t*)aligned_alloc(32, (bsz + 31) & ~(size_t)31);
        age  = (uint8_t*) aligned_alloc(32, asz);
#endif
        memset(bits, 0, bsz);
        memset(age,  0, asz);
    }
    ~Grid() {
#ifdef _WIN32
        _aligned_free(bits); _aligned_free(age);
#else
        free(bits); free(age);
#endif
    }

    inline uint64_t* brow(int r)             { return bits + (size_t)(r+1)*words_per_row + 1; }
    inline const uint64_t* brow(int r) const { return bits + (size_t)(r+1)*words_per_row + 1; }
    inline uint8_t* arow(int r)              { return age + (size_t)r * W; }
    inline const uint8_t* arow(int r) const  { return age + (size_t)r * W; }

    inline int  get(int r, int c) const { return (brow(r)[c/64] >> (c%64)) & 1; }
    inline void set(int r, int c, int v) {
        uint64_t& w = brow(r)[c/64];
        w = (w & ~(1ULL<<(c%64))) | ((uint64_t)v<<(c%64));
    }
    inline void set_with_age(int r, int c, int v) {
        set(r, c, v);
        arow(r)[c] = v ? 1 : 0;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Border sync (toroidal wrapping)
// ─────────────────────────────────────────────────────────────────────────────

void sync_borders(Grid& g) {
    int rw = (g.W+63)/64;
    memcpy(g.brow(-1),   g.brow(g.H-1), g.words_per_row*sizeof(uint64_t));
    memcpy(g.brow(g.H),  g.brow(0),     g.words_per_row*sizeof(uint64_t));
    for (int r=-1; r<=g.H; r++) {
        uint64_t* rp = g.brow(r);
        rp[-1]  = rp[rw-1];
        rp[rw]  = rp[0];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Bitwise neighbour counting (Abrash carry-save, full 4-bit count for rules 0–8)
// ─────────────────────────────────────────────────────────────────────────────

#define HALF_ADD(s,c,a,b)     { s=(a)^(b); c=(a)&(b); }
#define FULL_ADD(s,c,a,b,cin) { uint64_t _t=(a)^(b); s=_t^(cin); c=((a)&(b))|((_t)&(cin)); }

static inline uint64_t next_word_rules(
    uint64_t tl, uint64_t tc, uint64_t tr,
    uint64_t ml, uint64_t mc, uint64_t mr,
    uint64_t bl, uint64_t bc, uint64_t br,
    uint8_t rule_B, uint8_t rule_S)
{
    // Horizontal pair sums for each of the 3 rows
    auto hor = [](uint64_t L, uint64_t C, uint64_t R, uint64_t& s, uint64_t& c) {
        uint64_t left  = (C<<1)|(L>>63);
        uint64_t right = (C>>1)|(R<<63);
        HALF_ADD(s, c, left, right);
    };
    uint64_t ts,tc2, ms,mc2, bs,bc2;
    hor(tl,tc,tr, ts,tc2);
    hor(ml,mc,mr, ms,mc2);
    hor(bl,bc,br, bs,bc2);

    // Vertical sum of three 2-bit row-sums → 4-bit count [a3:a2:a1:a0]
    // Low bits
    uint64_t s0,c0,s1,c1;
    HALF_ADD(s0,c0, ts,ms);
    FULL_ADD(s1,c1, s0,bs, 0ULL);

    // High bits (carries from horizontal sums × 2)
    uint64_t h0,hc0,h1,hc1,carry;
    HALF_ADD(h0,hc0, tc2,mc2);
    FULL_ADD(h1,hc1, h0,bc2, 0ULL);
    // (hc0 and hc1 would give bit 3+, which can only be non-zero if count≥8)

    // Combine: count = 2*(tc2+mc2+bc2) + (ts+ms+bs)
    uint64_t a0,a1,a2,a3;
    a0 = s1;
    HALF_ADD(a1, carry, c1, h0);
    uint64_t c2a; HALF_ADD(a2, c2a, carry, h1);
    a3 = c2a ^ hc0 ^ hc1;  // bit 3: count=8 only

    // Decode count masks 0..8 from 4-bit a3:a2:a1:a0
    // count = 8*a3 + 4*a2 + 2*a1 + a0
    uint64_t cnt[9];
    cnt[0] = ~a3 & ~a2 & ~a1 & ~a0;
    cnt[1] = ~a3 & ~a2 & ~a1 &  a0;
    cnt[2] = ~a3 & ~a2 &  a1 & ~a0;
    cnt[3] = ~a3 & ~a2 &  a1 &  a0;
    cnt[4] = ~a3 &  a2 & ~a1 & ~a0;
    cnt[5] = ~a3 &  a2 & ~a1 &  a0;
    cnt[6] = ~a3 &  a2 &  a1 & ~a0;
    cnt[7] = ~a3 &  a2 &  a1 &  a0;
    cnt[8] =  a3 & ~a2 & ~a1 & ~a0;  // all 8 neighbours alive

    uint64_t birth = 0, survive = 0;
    for (int n = 0; n <= 8; n++) {
        if ((rule_B >> n) & 1) birth   |= cnt[n];
        if ((rule_S >> n) & 1) survive |= cnt[n];
    }
    return (birth & ~mc) | (survive & mc);
}

// ─────────────────────────────────────────────────────────────────────────────
// Step: one generation — two passes (bits then age)
// ─────────────────────────────────────────────────────────────────────────────

struct StepResult { long long live_count; };

StepResult step_rules(Grid& cur, Grid& nxt, uint8_t rule_B, uint8_t rule_S) {
    int rw = (cur.W+63)/64;
    long long live_total = 0;

    // Pass 1: bit update
    #pragma omp parallel for schedule(static) reduction(+:live_total)
    for (int r=0; r<cur.H; r++) {
        const uint64_t *top=cur.brow(r-1), *mid=cur.brow(r), *bot=cur.brow(r+1);
        uint64_t* out = nxt.brow(r);
        for (int w=0; w<rw; w++) {
            out[w] = next_word_rules(
                top[w-1],top[w],top[w+1],
                mid[w-1],mid[w],mid[w+1],
                bot[w-1],bot[w],bot[w+1],
                rule_B, rule_S);
            live_total += __builtin_popcountll(out[w]);
        }
    }

    // Pass 2: age update (separate pass = better cache behaviour)
    #pragma omp parallel for schedule(static)
    for (int r=0; r<cur.H; r++) {
        const uint64_t* nbit = nxt.brow(r);
        const uint8_t*  cage = cur.arow(r);
        uint8_t*        nage = nxt.arow(r);
        int W = cur.W;
        for (int w=0; w<rw; w++) {
            uint64_t alive = nbit[w];
            int base = w*64, limit = std::min(64, W-base);
            if (alive == 0ULL) { memset(nage+base, 0, limit); continue; }
            for (int b=0; b<limit; b++) {
                int cell = base+b;
                nage[cell] = (alive>>b)&1 ? (cage[cell]>=128 ? 128 : cage[cell]+1) : 0;
            }
        }
    }

    return {live_total};
}

// ─────────────────────────────────────────────────────────────────────────────
// Fill helpers
// ─────────────────────────────────────────────────────────────────────────────

void random_fill(Grid& g, float density, int seed) {
    srand((unsigned)seed);
    for (int r=0; r<g.H; r++)
        for (int c=0; c<g.W; c++) {
            int alive = (rand()/(float)RAND_MAX) < density ? 1 : 0;
            g.set_with_age(r, c, alive);
        }
}

long long count_live(const Grid& g) {
    long long t=0; int rw=(g.W+63)/64;
    for (int r=0; r<g.H; r++) {
        const uint64_t* rp=g.brow(r);
        for (int w=0; w<rw; w++) t+=__builtin_popcountll(rp[w]);
    }
    return t;
}

// ─────────────────────────────────────────────────────────────────────────────
// .cells loader — centers pattern on grid
// ─────────────────────────────────────────────────────────────────────────────

bool load_cells(Grid& g, const std::string& path) {
    FILE* f = fopen(path.c_str(), "r");
    if (!f) { fprintf(stderr, "Error: cannot open '%s'\n", path.c_str()); return false; }

    // First pass: collect lines (skip comments)
    std::vector<std::string> lines;
    char buf[65536];
    while (fgets(buf, sizeof(buf), f)) {
        if (buf[0]=='!'||buf[0]=='#') continue;
        // strip \r\n
        int len=(int)strlen(buf);
        while (len>0 && (buf[len-1]=='\r'||buf[len-1]=='\n')) buf[--len]=0;
        lines.emplace_back(buf);
    }
    fclose(f);

    // Pattern dimensions
    int pat_h = (int)lines.size();
    int pat_w = 0;
    for (auto& l : lines) pat_w = std::max(pat_w, (int)l.size());

    // Center on grid
    int start_r = (g.H - pat_h) / 2;
    int start_c = (g.W - pat_w) / 2;

    for (int pr=0; pr<pat_h; pr++) {
        int gr = start_r + pr;
        if (gr<0||gr>=g.H) continue;
        const std::string& ln = lines[pr];
        for (int pc=0; pc<(int)ln.size(); pc++) {
            int gc = start_c + pc;
            if (gc<0||gc>=g.W) continue;
            char ch = ln[pc];
            if (ch=='O'||ch=='o'||ch=='*') g.set_with_age(gr, gc, 1);
            else if (ch=='.') g.set_with_age(gr, gc, 0);
        }
    }

    fprintf(stderr, "Loaded '%s': %dx%d pattern, centered at (%d,%d)\n",
            path.c_str(), pat_w, pat_h, start_c, start_r);
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Stagnation hash — positional XOR so translationally distinct states differ
// Uses a per-word position mixing so hash(grid A) != hash(shifted A)
// ─────────────────────────────────────────────────────────────────────────────

static uint64_t grid_hash(const Grid& g) {
    int rw = (g.W+63)/64;
    uint64_t h = 0xcbf29ce484222325ULL; // FNV offset basis
    for (int r=0; r<g.H; r++) {
        const uint64_t* row = g.brow(r);
        uint64_t row_mix = (uint64_t)(r+1) * 0x9e3779b97f4a7c15ULL;
        for (int w=0; w<rw; w++) {
            uint64_t word = row[w] ^ row_mix ^ ((uint64_t)(w+1)*0x517cc1b727220a95ULL);
            // FNV-1a step
            h ^= word;
            h *= 0x00000100000001B3ULL;
        }
    }
    return h;
}

// ─────────────────────────────────────────────────────────────────────────────
// Age → RGB color LUT
// 0=dead(black), 1=white, ~20=yellow, ~50=orange, ~80=green, ~100=blue, 128=dark blue
// ─────────────────────────────────────────────────────────────────────────────

struct RGB { uint8_t r,g,b; };
static RGB age_lut[129];

static void build_age_lut() {
    struct Stop { float t; uint8_t r,g,b; };
    static const Stop stops[] = {
        {0.00f, 255,255,255},  // age 1   white
        {0.15f, 255,255,  0},  // ~20     yellow
        {0.38f, 255,140,  0},  // ~50     orange
        {0.60f,   0,200, 60},  // ~78     green
        {0.78f,   0, 80,255},  // ~100    blue
        {1.00f,   0, 20, 80},  // 128     dark blue
    };
    age_lut[0] = {0,0,0};
    for (int age=1; age<=128; age++) {
        float t = (age-1)/127.0f;
        RGB c = {255,255,255};
        for (int i=1; i<6; i++) {
            if (t<=stops[i].t) {
                float s=(t-stops[i-1].t)/(stops[i].t-stops[i-1].t);
                c.r=(uint8_t)(stops[i-1].r+s*(stops[i].r-stops[i-1].r));
                c.g=(uint8_t)(stops[i-1].g+s*(stops[i].g-stops[i-1].g));
                c.b=(uint8_t)(stops[i-1].b+s*(stops[i].b-stops[i-1].b));
                break;
            }
        }
        age_lut[age] = c;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Renderer: grid → RGB24, handles any ratio of grid:render pixels
// ─────────────────────────────────────────────────────────────────────────────

void render_frame(const Grid& g, int rW, int rH, std::vector<uint8_t>& buf) {
    buf.resize((size_t)rW*rH*3);
    #pragma omp parallel for schedule(static)
    for (int py=0; py<rH; py++) {
        int gr0=(int)((double)py    *g.H/rH), gr1=(int)((double)(py+1)*g.H/rH);
        if (gr1<=gr0) gr1=gr0+1;
        if (gr1>g.H)  gr1=g.H;
        uint8_t* out_row = buf.data()+(size_t)py*rW*3;
        for (int px=0; px<rW; px++) {
            int gc0=(int)((double)px    *g.W/rW), gc1=(int)((double)(px+1)*g.W/rW);
            if (gc1<=gc0) gc1=gc0+1;
            if (gc1>g.W)  gc1=g.W;
            uint8_t max_age=0;
            for (int gr=gr0; gr<gr1&&max_age<128; gr++) {
                const uint8_t* ar=g.arow(gr);
                for (int gc=gc0; gc<gc1; gc++)
                    if (ar[gc]>max_age) max_age=ar[gc];
            }
            RGB c=age_lut[max_age];
            out_row[px*3]=c.r; out_row[px*3+1]=c.g; out_row[px*3+2]=c.b;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark mode
// ─────────────────────────────────────────────────────────────────────────────

void run_bench(int W, int H, int GENS, int THREADS) {
    omp_set_num_threads(THREADS);
    fprintf(stderr,"=== GoL Benchmark ===\n");
    fprintf(stderr,"Grid: %dx%d  (%.2fMB bits + %.2fMB age)\n",
            W,H,(double)W*H/8/1024/1024,(double)W*H/1024/1024);
    fprintf(stderr,"Gens: %d  Threads: %d\n\n",GENS,THREADS);

    Grid A(W,H),B(W,H);
    random_fill(A,0.5f,42);
    sync_borders(A);

    fprintf(stderr,"Warmup (10 gens)...\n");
    for (int i=0;i<10;i++){step_rules(A,B,8,12);sync_borders(B);std::swap(A.bits,B.bits);std::swap(A.age,B.age);}

    auto t0=std::chrono::high_resolution_clock::now();
    for (int i=0;i<GENS;i++){step_rules(A,B,8,12);sync_borders(B);std::swap(A.bits,B.bits);std::swap(A.age,B.age);}
    double el=std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-t0).count();

    double gps=GENS/el, gcups=(double)W*H*GENS/el/1e9;
    fprintf(stderr,"\nTime: %.3fs  GPS: %.1f  GCUPS: %.2f\n\n",el,gps,gcups);
    for (double t:{240.0,144.0,60.0,30.0})
        fprintf(stderr,"  @ %3.0f GPS -> ~%.0f x %.0f cells\n",t,sqrt(gcups*1e9/t),sqrt(gcups*1e9/t));
}

// ─────────────────────────────────────────────────────────────────────────────
// Video mode — raw RGB24 to stdout, pipe to ffmpeg
//
// Windows-compatible ffmpeg command (run in PowerShell or cmd, ^ = line continuation):
//   gol video 15000 15000 600 60 1920 1080 8 | ffmpeg ^
//     -f rawvideo -pixel_format rgb24 -video_size 1920x1080 -framerate 60 -i - ^
//     -c:v libx264 -profile:v baseline -level 4.0 -pix_fmt yuv420p ^
//     -preset fast -crf 18 life.mp4
//
// In MSYS2/bash (\ = line continuation):
//   ./gol video 15000 15000 600 60 1920 1080 8 | ffmpeg \
//     -f rawvideo -pixel_format rgb24 -video_size 1920x1080 -framerate 60 -i - \
//     -c:v libx264 -profile:v baseline -level 4.0 -pix_fmt yuv420p \
//     -preset fast -crf 18 life.mp4
// ─────────────────────────────────────────────────────────────────────────────

void run_video(int W, int H, int GENS, int FPS, int rW, int rH,
               int THREADS, const std::string& state_file,
               uint8_t rule_B, uint8_t rule_S, int seed, float density)
{
    omp_set_num_threads(THREADS);
    build_age_lut();
    SET_STDOUT_BINARY();

    fprintf(stderr,"=== GoL Video Export ===\n");
    fprintf(stderr,"Grid:     %dx%d  (%.2fMB bits + %.2fMB age)\n",
            W,H,(double)W*H/8/1024/1024,(double)W*H/1024/1024);
    fprintf(stderr,"Rules:    %s\n", rules_str(rule_B,rule_S).c_str());
    fprintf(stderr,"Render:   %dx%d @ %d fps\n",rW,rH,FPS);
    fprintf(stderr,"Coverage: %.2f x %.2f cells/pixel\n",(double)W/rW,(double)H/rH);
    fprintf(stderr,"Duration: %d gens = %.1fs video\n\n",GENS,(double)GENS/FPS);
    // Print the exact ffmpeg command for this run
    fprintf(stderr,"FFmpeg command (MSYS2/bash):\n");
    fprintf(stderr,"  ./gol video %d %d %d %d %d %d %d | ffmpeg \\\n",W,H,GENS,FPS,rW,rH,THREADS);
    fprintf(stderr,"    -f rawvideo -pixel_format rgb24 -video_size %dx%d -framerate %d -i - \\\n",rW,rH,FPS);
    fprintf(stderr,"    -c:v libx264 -profile:v baseline -level 4.0 -pix_fmt yuv420p \\\n");
    fprintf(stderr,"    -preset fast -crf 18 life.mp4\n\n");

    Grid A(W,H),B(W,H);
    if (!state_file.empty()) { if (!load_cells(A,state_file)) return; }
    else                       random_fill(A, density, seed);

    fprintf(stderr,"Initial live: %lld (%.1f%%)\n",
            count_live(A), 100.0*count_live(A)/((long long)W*H));
    sync_borders(A);

    std::vector<uint8_t> frame_buf;
    size_t frame_bytes=(size_t)rW*rH*3;
    auto wall0=std::chrono::high_resolution_clock::now();

    for (int gen=0; gen<GENS; gen++) {
        render_frame(A,rW,rH,frame_buf);
        fwrite(frame_buf.data(),1,frame_bytes,stdout);
        // Don't fflush every frame — let the OS buffer it.
        // ffmpeg reads from the pipe continuously; flushing per frame
        // adds ~5% overhead and is unnecessary with a pipe buffer.
        if (gen % 100 == 99) fflush(stdout); // flush every 100 frames

        step_rules(A,B,rule_B,rule_S);
        sync_borders(B);
        std::swap(A.bits,B.bits); std::swap(A.age,B.age);

        if (gen%50==0||gen==GENS-1) {
            auto now=std::chrono::high_resolution_clock::now();
            double el=std::chrono::duration<double>(now-wall0).count();
            double fps=(gen+1)/el, eta=(GENS-gen-1)/fps;
            fprintf(stderr,"  Frame %5d/%d  %.1f fps  ETA %.0fs\n",gen+1,GENS,fps,eta);
        }
    }
    fflush(stdout); // final flush before closing pipe
    double total=std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-wall0).count();
    fprintf(stderr,"\nDone. %.1fs for %.1fs of video (%.1fx realtime)\n",
            total,(double)GENS/FPS,(double)GENS/FPS/total);
}

// ─────────────────────────────────────────────────────────────────────────────
// Sim mode — config-driven, runs to stagnation, optionally pipes video
// ─────────────────────────────────────────────────────────────────────────────

int run_sim(const Config& cfg) {
    omp_set_num_threads(cfg.threads);
    build_age_lut();

    if (cfg.has_video()) {
        SET_STDOUT_BINARY();
        // Print the ffmpeg command the user needs to run with this config
        fprintf(stderr,"Video output enabled. Pipe command (MSYS2/bash):\n");
        fprintf(stderr,"  gol sim <config> | ffmpeg \\\n");
        fprintf(stderr,"    -f rawvideo -pixel_format rgb24 -video_size %dx%d -framerate %d -i - \\\n",
                cfg.video_w,cfg.video_h,cfg.video_fps);
        fprintf(stderr,"    -c:v libx264 -profile:v baseline -level 4.0 -pix_fmt yuv420p \\\n");
        fprintf(stderr,"    -preset fast -crf 18 life.mp4\n\n");
    }

    fprintf(stderr,"=== GoL Simulation ===\n");
    fprintf(stderr,"Grid:    %dx%d  (%.2fMB bits + %.2fMB age)\n",
            cfg.grid_w,cfg.grid_h,
            (double)cfg.grid_w*cfg.grid_h/8/1024/1024,
            (double)cfg.grid_w*cfg.grid_h/1024/1024);
    fprintf(stderr,"Rules:   %s\n", rules_str(cfg.rule_B,cfg.rule_S).c_str());
    fprintf(stderr,"Max:     %d gens\n",cfg.max_gens);
    fprintf(stderr,"Stag:    %s  cycle=%s  extinct=%s\n",
            cfg.stag_enable?"on":"off",
            cfg.stag_cycle?"on":"off",
            cfg.stag_extinct?"on":"off");
    fprintf(stderr,"Threads: %d\n",cfg.threads);
    if (cfg.has_video())
        fprintf(stderr,"Video:   %dx%d @ %d fps\n",cfg.video_w,cfg.video_h,cfg.video_fps);
    fprintf(stderr,"\n");

    Grid A(cfg.grid_w,cfg.grid_h), B(cfg.grid_w,cfg.grid_h);

    if (!cfg.state_file.empty()) {
        if (!load_cells(A,cfg.state_file)) return 1;
    } else {
        random_fill(A, cfg.random_density, cfg.random_seed);
    }

    fprintf(stderr,"Initial live: %lld (%.1f%%)\n",
            count_live(A), 100.0*count_live(A)/((long long)cfg.grid_w*cfg.grid_h));
    sync_borders(A);

    std::vector<uint8_t> frame_buf;
    size_t frame_bytes = cfg.has_video() ? (size_t)cfg.video_w*cfg.video_h*3 : 0;

    // Stagnation: store hash → gen map (bounded by max_gens cap)
    std::unordered_map<uint64_t,int> hash_seen;
    hash_seen.reserve(std::min(cfg.max_gens, 16384));

    int   stop_gen = -1;
    std::string stop_reason;
    auto last_report = std::chrono::high_resolution_clock::now();
    auto t0 = last_report;

    // Render gen 0
    if (cfg.has_video()) {
        render_frame(A,cfg.video_w,cfg.video_h,frame_buf);
        fwrite(frame_buf.data(),1,frame_bytes,stdout);
    }

    for (int gen=1; gen<=cfg.max_gens; gen++) {
        StepResult res = step_rules(A,B,cfg.rule_B,cfg.rule_S);
        sync_borders(B);

        if (cfg.stag_enable) {
            // Extinction
            if (cfg.stag_extinct && res.live_count==0) {
                stop_gen=gen; stop_reason="Extinction (all cells dead)";
                std::swap(A.bits,B.bits); std::swap(A.age,B.age);
                if (cfg.has_video()) {
                    render_frame(A,cfg.video_w,cfg.video_h,frame_buf);
                    fwrite(frame_buf.data(),1,frame_bytes,stdout);
                    fflush(stdout);
                }
                break;
            }
            // Cycle detection
            if (cfg.stag_cycle) {
                uint64_t h = grid_hash(B);
                auto it = hash_seen.find(h);
                if (it != hash_seen.end()) {
                    char tmp[128];
                    snprintf(tmp,sizeof(tmp),"Cycle detected: period %d (gen %d = gen %d)",
                             gen - it->second, gen, it->second);
                    stop_gen=gen; stop_reason=tmp;
                    std::swap(A.bits,B.bits); std::swap(A.age,B.age);
                    if (cfg.has_video()) {
                        render_frame(A,cfg.video_w,cfg.video_h,frame_buf);
                        fwrite(frame_buf.data(),1,frame_bytes,stdout);
                        fflush(stdout);
                    }
                    break;
                }
                hash_seen[h]=gen;
            }
        }

        std::swap(A.bits,B.bits); std::swap(A.age,B.age);

        if (cfg.has_video()) {
            render_frame(A,cfg.video_w,cfg.video_h,frame_buf);
            fwrite(frame_buf.data(),1,frame_bytes,stdout);
            if (gen%100==0) fflush(stdout); // don't fflush every frame
        }

        // Progress (stderr, \r overwrite)
        auto now=std::chrono::high_resolution_clock::now();
        if (std::chrono::duration<double>(now-last_report).count() >= 1.0 || gen==cfg.max_gens) {
            double elapsed=std::chrono::duration<double>(now-t0).count();
            fprintf(stderr,"  Gen %6d/%d  %7.1f GPS  live: %lld        \r",
                    gen,cfg.max_gens,(double)gen/elapsed,(long long)res.live_count);
            last_report=now;
        }
    }

    fflush(stdout); // ensure pipe is drained before exit
    double elapsed=std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-t0).count();
    int total_gens = stop_gen>0 ? stop_gen : cfg.max_gens;

    fprintf(stderr,"\n\n=== Complete ===\n");
    fprintf(stderr,"Gens: %d  Time: %.3fs  GPS: %.1f\n",total_gens,elapsed,(double)total_gens/elapsed);
    fprintf(stderr,"Stop: %s\n", stop_reason.empty() ? "max_gens reached" : stop_reason.c_str());
    if (cfg.has_video())
        fprintf(stderr,"Video: %d frames (%.1fs @ %d fps)\n",
                total_gens+1,(double)(total_gens+1)/cfg.video_fps,cfg.video_fps);

    return 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

void usage(const char* p) {
    fprintf(stderr,
        "Conway's Game of Life — High-Performance Backend\n\n"
        "Build:  g++ -O3 -march=native -mavx2 -fopenmp -std=c++17 -o gol gol_sim.cpp\n\n"
        "Modes:\n"
        "  %s bench <grid_w> <grid_h> <gens> [threads]\n"
        "  %s video <grid_w> <grid_h> <gens> <fps> <render_w> <render_h> [threads] [state.cells] [B3/S23]\n"
        "  %s sim   <config.cfg> [threads]\n"
        "  %s init  <config.cfg>           (write starter config)\n\n"
        "Video pipe (MSYS2/bash):\n"
        "  ./gol video 15000 15000 600 60 1920 1080 8 | ffmpeg \\\n"
        "    -f rawvideo -pixel_format rgb24 -video_size 1920x1080 -framerate 60 -i - \\\n"
        "    -c:v libx264 -profile:v baseline -level 4.0 -pix_fmt yuv420p \\\n"
        "    -preset fast -crf 18 life.mp4\n\n"
        "Video pipe (PowerShell/cmd, use ^ for line continuation):\n"
        "  .\\gol video 15000 15000 600 60 1920 1080 8 | ffmpeg ^\n"
        "    -f rawvideo -pixel_format rgb24 -video_size 1920x1080 -framerate 60 -i - ^\n"
        "    -c:v libx264 -profile:v baseline -level 4.0 -pix_fmt yuv420p ^\n"
        "    -preset fast -crf 18 life.mp4\n\n"
        "Video with custom state + rules:\n"
        "  ./gol video 15000 15000 600 60 1920 1080 8 start.cells B36/S23\n\n"
        "Notable rules:\n"
        "  B3/S23    Conway (default)\n"
        "  B36/S23   HighLife\n"
        "  B3/S12345 Maze\n"
        "  B2/S      Seeds (explosive)\n"
        "  B3/S45678 Coral\n",
        p,p,p,p);
}

int main(int argc, char** argv) {
    if (argc < 2) { usage(argv[0]); return 1; }
    std::string mode = argv[1];

    if (mode == "init") {
        if (argc < 3) { fprintf(stderr,"Usage: %s init <config.cfg>\n",argv[0]); return 1; }
        write_default_config(argv[2]);
        return 0;
    }

    if (mode == "bench") {
        if (argc < 5) { usage(argv[0]); return 1; }
        run_bench(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]),
                  argc>5 ? atoi(argv[5]) : omp_get_max_threads());
        return 0;
    }

    if (mode == "video") {
        if (argc < 8) { usage(argv[0]); return 1; }
        int W=atoi(argv[2]),H=atoi(argv[3]),GENS=atoi(argv[4]);
        int FPS=atoi(argv[5]),rW=atoi(argv[6]),rH=atoi(argv[7]);

        // Optional positional args after rH:
        //   [threads]      — integer, default auto
        //   [state.cells]  — path containing '.' or '/' or '\' or ending in .cells
        //   [B../S..]      — starts with 'B' or 'b'
        int threads = omp_get_max_threads();
        std::string state_file;
        uint8_t rule_B=8, rule_S=12;
        int seed=42; float density=0.5f;

        for (int i=8; i<argc; i++) {
            std::string a = argv[i];
            if (a[0]=='B'||a[0]=='b') {
                parse_rules(a, rule_B, rule_S);
            } else if (a.find('.')!=std::string::npos ||
                       a.find('/')!=std::string::npos ||
                       a.find('\\')!=std::string::npos) {
                state_file = a;
            } else {
                int v = atoi(a.c_str());
                if (v > 0) threads = v;
            }
        }
        run_video(W,H,GENS,FPS,rW,rH,threads,state_file,rule_B,rule_S,seed,density);
        return 0;
    }

    if (mode == "sim") {
        if (argc < 3) { usage(argv[0]); return 1; }
        Config cfg;
        if (!parse_config(argv[2], cfg)) return 1;
        if (argc > 3) cfg.threads = atoi(argv[3]);
        return run_sim(cfg);
    }

    usage(argv[0]); return 1;
}
