/**
 * Conway's Game of Life — v2: Analysis & Archive Edition
 *
 * Build:
 *   g++ -O3 -march=native -mavx2 -fopenmp -std=c++17 -o gol gol_sim.cpp
 *   (omit -mavx2 if your CPU lacks it; -fopenmp optional but recommended)
 *
 * ─────────────────────────────── MODES ────────────────────────────────────
 *
 *   gol bench  <grid_w> <grid_h> <gens> [threads]
 *       Benchmark: no video, measures gens/sec.
 *
 *   gol video  <grid_w> <grid_h> <gens> <fps> <render_w> <render_h>
 *              [threads] [state.cells|state.gol] [B3/S23]
 *       Pipe raw RGB24 to ffmpeg (see examples below).
 *
 *   gol sim    <config.cfg> [threads]
 *       Config-driven run with stagnation detection and optional archive.
 *
 *   gol init   <config.cfg>
 *       Write a default config file.
 *
 *   gol archive-info  <file.gol>
 *       Print metadata from a .gol archive without running the sim.
 *
 * ─────────────────────────────── .gol FORMAT ──────────────────────────────
 *
 *   A compact, lossless binary archive of every generation's bit-grid.
 *   Layout (all little-endian):
 *
 *   Header (64 bytes, fixed):
 *     [0..3]   magic        "GOL1"
 *     [4..7]   grid_w       uint32
 *     [8..11]  grid_h       uint32
 *     [12..15] total_gens   uint32  (filled at close)
 *     [16..19] fps_hint     uint32  (0 = not a video capture)
 *     [20..23] rule_B       uint8 * 4  (B byte, then 3 padding bytes)
 *     [24..27] rule_S       uint8 * 4
 *     [28..31] stag_period  int32   (-1 = extinction, 0 = max_gens, N = cycle period)
 *     [32..35] stag_gen     int32   (generation at which stagnation was first detected)
 *     [36..63] reserved     (28 bytes, zero)
 *
 *   Frame records (one per generation, gen 0 first):
 *     [0..7]   gen          uint64
 *     [8..11]  live_count   uint32
 *     [12..]   bits         ceil(W*H/8) bytes, row-major, LSB-first within byte
 *
 *   The bit grid is NOT compressed in the raw frame record. However, the
 *   entire .gol file compresses extremely well with gzip / zstd because
 *   the bit density drops and still-life patterns are highly repetitive.
 *   Recommended: gol_archive.gol.zst  (zstd -19 typically 10-50x ratio)
 *
 * ─────────────────────────────── STAGNATION ───────────────────────────────
 *
 *   Three detection levels, all on by default:
 *     1. Extinction   – live count reaches 0.
 *     2. Still life   – consecutive hash match (period 1).
 *     3. Cycle        – hash seen before (period N). The sim stores a
 *                       rolling window of the last `cycle_window` hashes
 *                       (default 1024). Periods up to that window are found.
 *
 *   The hash is a position-sensitive 64-bit FNV-1a over all live cell words,
 *   so translational duplicates are not confused.
 *
 * ─────────────────────────────── SEED / INPUT ─────────────────────────────
 *
 *   Pass a .cells file (Golly plain-text) or a .gol archive (any gen).
 *   Omit for random fill with density + seed from config.
 *   Use the config `seed_pattern` field to bake a starting pattern by name
 *   (built-in: glider, blinker, block, r-pentomino, acorn, pulsar, lwss,
 *    diehard, gosper-gun).
 *
 * ─────────────────────────────── CONFIG ───────────────────────────────────
 *
 *   grid_w          = 1920
 *   grid_h          = 1080
 *   threads         = 0         # 0 = auto
 *   rules           = B3/S23
 *   state_file      =           # .cells or .gol; empty = random or seed_pattern
 *   seed_pattern    =           # built-in named pattern (overrides random)
 *   random_seed     = 42
 *   random_density  = 0.3
 *   video_w         = 1920
 *   video_h         = 1080
 *   video_fps       = 60
 *   max_gens        = 100000
 *   stagnation      = 1
 *   detect_extinction = 1
 *   detect_still    = 1
 *   detect_cycle    = 1
 *   cycle_window    = 1024      # max period detectable
 *   archive_file    =           # path to write .gol; empty = no archive
 *   archive_every   = 1        # save every N gens (1 = all)
 *
 * ─────────────────────────────── VIDEO PIPE ───────────────────────────────
 *
 *   Windows (PowerShell, ^ = line continuation):
 *     .\gol video 3840 2160 600 60 1920 1080 8 | ffmpeg ^
 *       -f rawvideo -pixel_format rgb24 -video_size 1920x1080 -framerate 60 -i - ^
 *       -c:v libx264 -preset veryslow -crf 10 -pix_fmt yuv420p life.mp4
 *
 *   Linux/bash (\ = line continuation):
 *     ./gol video 3840 2160 600 60 1920 1080 8 | ffmpeg \
 *       -f rawvideo -pixel_format rgb24 -video_size 1920x1080 -framerate 60 -i - \
 *       -c:v libx264 -preset veryslow -crf 10 -pix_fmt yuv420p life.mp4
 */

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <string>
#include <vector>
#include <deque>
#include <algorithm>
#include <unordered_map>
#include <cassert>
#include <omp.h>

#ifdef _WIN32
  #include <io.h>
  #include <fcntl.h>
  #define SET_STDOUT_BINARY() _setmode(_fileno(stdout), _O_BINARY)
#else
  #define SET_STDOUT_BINARY() ((void)0)
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Rules
// ─────────────────────────────────────────────────────────────────────────────

static bool parse_rules(const std::string& s, uint8_t& B, uint8_t& S) {
    B = S = 0;
    size_t slash = s.find('/');
    if (slash == std::string::npos) return false;
    auto fill = [](const std::string& part, uint8_t& mask) {
        mask = 0;
        for (char ch : part)
            if (ch >= '0' && ch <= '8') mask |= (uint8_t)(1 << (ch - '0'));
    };
    std::string bp = s.substr(0, slash), sp = s.substr(slash + 1);
    if (!bp.empty() && (bp[0]=='B'||bp[0]=='b')) bp = bp.substr(1);
    if (!sp.empty() && (sp[0]=='S'||sp[0]=='s')) sp = sp.substr(1);
    fill(bp, B); fill(sp, S);
    return true;
}

static std::string rules_str(uint8_t B, uint8_t S) {
    std::string r = "B";
    for (int i=0;i<=8;i++) if ((B>>i)&1) r += (char)('0'+i);
    r += "/S";
    for (int i=0;i<=8;i++) if ((S>>i)&1) r += (char)('0'+i);
    return r;
}

// ─────────────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────────────

struct Config {
    int   grid_w          = 1920;
    int   grid_h          = 1080;
    int   threads         = 0;
    uint8_t rule_B        = 8;   // B3
    uint8_t rule_S        = 12;  // S23
    std::string state_file;
    std::string seed_pattern;
    int   random_seed     = 42;
    float random_density  = 0.3f;
    int   video_w         = 0;
    int   video_h         = 0;
    int   video_fps       = 60;
    int   max_gens        = 100000;
    bool  stag_enable     = true;
    bool  stag_extinct    = true;
    bool  stag_still      = true;
    bool  stag_cycle      = true;
    int   cycle_window    = 1024;
    std::string archive_file;
    int   archive_every   = 1;

    bool has_video() const { return video_w > 0 && video_h > 0; }
    bool has_archive() const { return !archive_file.empty(); }
};

static bool parse_config(const std::string& path, Config& cfg) {
    FILE* f = fopen(path.c_str(), "r");
    if (!f) { fprintf(stderr, "Error: cannot open config '%s'\n", path.c_str()); return false; }
    char line[1024];
    while (fgets(line, sizeof(line), f)) {
        if (line[0]=='#'||line[0]=='\n'||line[0]=='\r') continue;
        char key[256]={}, val[768]={};
        if (sscanf(line, " %255[^= \t] = %767[^\n\r]", key, val) < 2) continue;
        int vlen=(int)strlen(val);
        while (vlen>0 && (val[vlen-1]==' '||val[vlen-1]=='\t')) val[--vlen]=0;
        std::string k(key), v(val);
        if      (k=="grid_w")           cfg.grid_w          = atoi(v.c_str());
        else if (k=="grid_h")           cfg.grid_h          = atoi(v.c_str());
        else if (k=="threads")          cfg.threads         = atoi(v.c_str());
        else if (k=="rules")            parse_rules(v, cfg.rule_B, cfg.rule_S);
        else if (k=="state_file")       cfg.state_file      = v;
        else if (k=="seed_pattern")     cfg.seed_pattern    = v;
        else if (k=="random_seed")      cfg.random_seed     = atoi(v.c_str());
        else if (k=="random_density")   cfg.random_density  = (float)atof(v.c_str());
        else if (k=="video_w")          cfg.video_w         = atoi(v.c_str());
        else if (k=="video_h")          cfg.video_h         = atoi(v.c_str());
        else if (k=="video_fps")        cfg.video_fps       = atoi(v.c_str());
        else if (k=="max_gens")         cfg.max_gens        = atoi(v.c_str());
        else if (k=="stagnation")       cfg.stag_enable     = atoi(v.c_str())!=0;
        else if (k=="detect_extinction")cfg.stag_extinct    = atoi(v.c_str())!=0;
        else if (k=="detect_still")     cfg.stag_still      = atoi(v.c_str())!=0;
        else if (k=="detect_cycle")     cfg.stag_cycle      = atoi(v.c_str())!=0;
        else if (k=="cycle_window")     cfg.cycle_window    = atoi(v.c_str());
        else if (k=="archive_file")     cfg.archive_file    = v;
        else if (k=="archive_every")    cfg.archive_every   = atoi(v.c_str());
    }
    fclose(f);
    if (cfg.threads <= 0) cfg.threads = omp_get_max_threads();
    return true;
}

static void write_default_config(const std::string& path) {
    FILE* f = fopen(path.c_str(), "w");
    if (!f) { fprintf(stderr, "Cannot write config: %s\n", path.c_str()); return; }
    fprintf(f,
        "# Conway's Game of Life v2 — config\n\n"
        "grid_w          = 1920\n"
        "grid_h          = 1080\n"
        "threads         = 0           # 0 = auto\n\n"
        "rules           = B3/S23\n\n"
        "# Input: state_file (path to .cells or .gol), seed_pattern (built-in name),\n"
        "# or neither (random fill).\n"
        "state_file      =\n"
        "seed_pattern    =             # glider blinker block r-pentomino acorn pulsar lwss\n"
        "random_seed     = 42\n"
        "random_density  = 0.3\n\n"
        "# Video output (pipe stdout to ffmpeg)\n"
        "video_w         = 1920\n"
        "video_h         = 1080\n"
        "video_fps       = 60\n\n"
        "max_gens        = 100000\n\n"
        "# Stagnation detection\n"
        "stagnation      = 1\n"
        "detect_extinction = 1\n"
        "detect_still    = 1           # still life (period 1)\n"
        "detect_cycle    = 1           # repeating cycle\n"
        "cycle_window    = 1024        # max detectable period\n\n"
        "# Archive: lossless binary record of every generation\n"
        "archive_file    =             # e.g. run.gol\n"
        "archive_every   = 1          # 1 = every gen, 10 = every 10th, etc.\n"
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
    int words_per_row;

    Grid(int w, int h) : W(w), H(h) {
        words_per_row = (w + 63) / 64 + 2;
        size_t bsz = (size_t)(h + 2) * words_per_row * sizeof(uint64_t);
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

    inline uint64_t* brow(int r)            { return bits + (size_t)(r+1)*words_per_row + 1; }
    inline const uint64_t* brow(int r) const{ return bits + (size_t)(r+1)*words_per_row + 1; }
    inline uint8_t* arow(int r)             { return age + (size_t)r * W; }
    inline const uint8_t* arow(int r) const { return age + (size_t)r * W; }

    inline int  get(int r, int c) const { return (brow(r)[c/64] >> (c%64)) & 1; }
    inline void set(int r, int c, int v) {
        uint64_t& w = brow(r)[c/64];
        w = (w & ~(1ULL<<(c%64))) | ((uint64_t)v<<(c%64));
    }
    inline void set_with_age(int r, int c, int v) {
        set(r,c,v);
        arow(r)[c] = v ? 1 : 0;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Border sync (toroidal wrapping)
// ─────────────────────────────────────────────────────────────────────────────

static void sync_borders(Grid& g) {
    int rw = (g.W+63)/64;
    memcpy(g.brow(-1),  g.brow(g.H-1), g.words_per_row*sizeof(uint64_t));
    memcpy(g.brow(g.H), g.brow(0),     g.words_per_row*sizeof(uint64_t));
    for (int r=-1; r<=g.H; r++) {
        uint64_t* rp = g.brow(r);
        rp[-1] = rp[rw-1];
        rp[rw] = rp[0];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Step: 8-neighbour bit-serial adder
// ─────────────────────────────────────────────────────────────────────────────

static inline uint64_t next_word_rules(
    uint64_t tl, uint64_t tc, uint64_t tr,
    uint64_t ml, uint64_t mc, uint64_t mr,
    uint64_t bl, uint64_t bc, uint64_t br,
    uint8_t rule_B, uint8_t rule_S)
{
    uint64_t n0=0, n1=0, n2=0, n3=0;

    auto add = [&](uint64_t b) {
        uint64_t c = n0 & b; n0 ^= b;
        b = n1 & c;          n1 ^= c;
        c = n2 & b;          n2 ^= b;
                             n3 ^= c;
    };

    add((tc<<1)|(tl>>63));
    add(tc);
    add((tc>>1)|(tr<<63));
    add((mc<<1)|(ml>>63));
    add((mc>>1)|(mr<<63));
    add((bc<<1)|(bl>>63));
    add(bc);
    add((bc>>1)|(br<<63));

    uint64_t cnt[9];
    cnt[0] = ~n3 & ~n2 & ~n1 & ~n0;
    cnt[1] = ~n3 & ~n2 & ~n1 &  n0;
    cnt[2] = ~n3 & ~n2 &  n1 & ~n0;
    cnt[3] = ~n3 & ~n2 &  n1 &  n0;
    cnt[4] = ~n3 &  n2 & ~n1 & ~n0;
    cnt[5] = ~n3 &  n2 & ~n1 &  n0;
    cnt[6] = ~n3 &  n2 &  n1 & ~n0;
    cnt[7] = ~n3 &  n2 &  n1 &  n0;
    cnt[8] =  n3 & ~n2 & ~n1 & ~n0;

    uint64_t birth = 0, survive = 0;
    for (int n = 0; n <= 8; n++) {
        if ((rule_B >> n) & 1) birth   |= cnt[n];
        if ((rule_S >> n) & 1) survive |= cnt[n];
    }
    return (birth & ~mc) | (survive & mc);
}

struct StepResult { long long live_count; };

static StepResult step_rules(Grid& cur, Grid& nxt, uint8_t rule_B, uint8_t rule_S) {
    int rw = (cur.W+63)/64;
    long long live_total = 0;

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

    #pragma omp parallel for schedule(static)
    for (int r=0; r<cur.H; r++) {
        const uint64_t* nbit = nxt.brow(r);
        const uint8_t*  cage = cur.arow(r);
        uint8_t*        nage = nxt.arow(r);
        int W = cur.W;
        for (int w=0; w<rw; w++) {
            uint64_t alive = nbit[w];
            int base=w*64, limit=std::min(64,W-base);
            if (alive==0ULL) { memset(nage+base,0,limit); continue; }
            for (int b=0; b<limit; b++) {
                int cell=base+b;
                nage[cell] = (alive>>b)&1 ? (cage[cell]>=128 ? 128 : cage[cell]+1) : 0;
            }
        }
    }

    return {live_total};
}

// ─────────────────────────────────────────────────────────────────────────────
// Stagnation: rolling-window hash detector
// ─────────────────────────────────────────────────────────────────────────────

struct StagnationResult {
    bool triggered   = false;
    int  period      = -1;
    int  detect_gen  = -1;
    int  cycle_start = -1;
};

static uint64_t grid_hash(const Grid& g) {
    int rw = (g.W+63)/64;
    uint64_t h = 0xcbf29ce484222325ULL;
    for (int r=0; r<g.H; r++) {
        const uint64_t* row = g.brow(r);
        uint64_t mix = (uint64_t)(r+1) * 0x9e3779b97f4a7c15ULL;
        for (int w=0; w<rw; w++) {
            uint64_t word = row[w] ^ mix ^ ((uint64_t)(w+1)*0x517cc1b727220a95ULL);
            h ^= word;
            h *= 0x00000100000001B3ULL;
        }
    }
    return h;
}

struct StagnationDetector {
    bool enable_extinct, enable_still, enable_cycle;
    int  window;
    std::deque<std::pair<int,uint64_t>> history;
    std::unordered_map<uint64_t,int> hash_map;

    explicit StagnationDetector(const Config& cfg)
        : enable_extinct(cfg.stag_extinct)
        , enable_still(cfg.stag_still)
        , enable_cycle(cfg.stag_cycle)
        , window(cfg.cycle_window)
    {}

    StagnationResult check(int gen, long long live_count, const Grid& nxt) {
        if (enable_extinct && live_count == 0)
            return {true, -1, gen, -1};

        uint64_t h = grid_hash(nxt);

        if (enable_still || enable_cycle) {
            auto it = hash_map.find(h);
            if (it != hash_map.end()) {
                int old_gen = it->second;
                int period  = gen - old_gen;
                return {true, period, gen, old_gen};
            }
            history.push_back({gen, h});
            hash_map[h] = gen;
            if ((int)history.size() > window) {
                auto& front = history.front();
                hash_map.erase(front.second);
                history.pop_front();
            }
        }

        return {false};
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// .gol Archive writer
// ─────────────────────────────────────────────────────────────────────────────

struct GolArchive {
    FILE*    fp       = nullptr;
    uint32_t grid_w   = 0;
    uint32_t grid_h   = 0;
    uint32_t gen_count= 0;
    uint8_t  rule_B   = 8;
    uint8_t  rule_S   = 12;
    uint32_t fps_hint = 0;
    size_t   bytes_per_frame = 0;

    static const size_t HEADER_SIZE = 64;

    bool open(const std::string& path, int W, int H,
              uint8_t B, uint8_t S, uint32_t fps=0) {
        fp = fopen(path.c_str(), "wb");
        if (!fp) { fprintf(stderr,"Archive: cannot open '%s'\n",path.c_str()); return false; }
        grid_w=W; grid_h=H; rule_B=B; rule_S=S; fps_hint=fps;
        bytes_per_frame = 12 + (size_t)((W*H+7)/8);
        uint8_t hdr[HEADER_SIZE] = {};
        hdr[0]='G'; hdr[1]='O'; hdr[2]='L'; hdr[3]='1';
        write_u32(hdr+4,  W);
        write_u32(hdr+8,  H);
        write_u32(hdr+12, 0);
        write_u32(hdr+16, fps);
        hdr[20]=B; hdr[24]=S;
        write_i32(hdr+28, -1);
        write_i32(hdr+32, -1);
        fwrite(hdr, 1, HEADER_SIZE, fp);
        return true;
    }

    void write_frame(int gen, long long live_count, const Grid& g) {
        if (!fp) return;
        uint8_t fhdr[12];
        write_u64(fhdr+0, (uint64_t)gen);
        write_u32(fhdr+8, (uint32_t)live_count);
        fwrite(fhdr, 1, 12, fp);
        size_t nbytes = (size_t)((g.W*g.H+7)/8);
        std::vector<uint8_t> buf(nbytes, 0);
        long long bit_pos = 0;
        for (int r=0; r<g.H; r++) {
            const uint64_t* row = g.brow(r);
            for (int c=0; c<g.W; c++) {
                int alive = (row[c/64] >> (c%64)) & 1;
                if (alive) buf[bit_pos/8] |= (1 << (bit_pos%8));
                bit_pos++;
            }
        }
        fwrite(buf.data(), 1, nbytes, fp);
        gen_count++;
    }

    void close(int stag_period=-1, int stag_gen=-1) {
        if (!fp) return;
        fseek(fp, 12, SEEK_SET);
        uint8_t tmp[8];
        write_u32(tmp, gen_count);  fwrite(tmp, 1, 4, fp);
        fseek(fp, 28, SEEK_SET);
        write_i32(tmp, stag_period); fwrite(tmp, 1, 4, fp);
        write_i32(tmp, stag_gen);    fwrite(tmp, 1, 4, fp);
        fclose(fp); fp=nullptr;
    }

    ~GolArchive() { if (fp) close(); }

    static void write_u32(uint8_t* p, uint32_t v) {
        p[0]=v&0xff; p[1]=(v>>8)&0xff; p[2]=(v>>16)&0xff; p[3]=(v>>24)&0xff;
    }
    static void write_i32(uint8_t* p, int32_t v) { write_u32(p, (uint32_t)v); }
    static void write_u64(uint8_t* p, uint64_t v) {
        for(int i=0;i<8;i++) p[i]=(v>>(i*8))&0xff;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// .gol Archive info reader
// ─────────────────────────────────────────────────────────────────────────────

static void archive_info(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr,"Cannot open '%s'\n",path.c_str()); return; }
    uint8_t hdr[64];
    if (fread(hdr,1,64,f)!=64 || hdr[0]!='G'||hdr[1]!='O'||hdr[2]!='L'||hdr[3]!='1') {
        fprintf(stderr,"Not a valid .gol file\n"); fclose(f); return;
    }
    auto ru32=[&](uint8_t* p){ return (uint32_t)p[0]|((uint32_t)p[1]<<8)|((uint32_t)p[2]<<16)|((uint32_t)p[3]<<24); };
    auto ri32=[&](uint8_t* p){ return (int32_t)ru32(p); };
    uint32_t W    = ru32(hdr+4);
    uint32_t H    = ru32(hdr+8);
    uint32_t gens = ru32(hdr+12);
    uint32_t fps  = ru32(hdr+16);
    uint8_t  B    = hdr[20], S = hdr[24];
    int32_t  sp   = ri32(hdr+28);
    int32_t  sg   = ri32(hdr+32);

    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fclose(f);

    fprintf(stderr,"=== .gol Archive Info ===\n");
    fprintf(stderr,"File:        %s  (%.2f MB)\n", path.c_str(), fsize/1048576.0);
    fprintf(stderr,"Grid:        %dx%d\n", W, H);
    fprintf(stderr,"Rules:       %s\n", rules_str(B,S).c_str());
    fprintf(stderr,"Generations: %u\n", gens);
    if (fps>0) fprintf(stderr,"FPS hint:    %u\n", fps);
    if (sp==-1)       fprintf(stderr,"Stagnation:  extinction\n");
    else if (sp==0)   fprintf(stderr,"Stagnation:  max_gens reached (none detected)\n");
    else              fprintf(stderr,"Stagnation:  period %d cycle, detected at gen %d\n", sp, sg);
    size_t bytes_per = 12 + (size_t)((W*H+7)/8);
    fprintf(stderr,"Frame size:  %zu bytes (header 12 + %zu bits payload)\n",
            bytes_per, (size_t)W*H);
    if (gens>0) fprintf(stderr,"Ratio:       %.2f bytes/cell/gen\n",
            (double)bytes_per/(W*H));
}

// ─────────────────────────────────────────────────────────────────────────────
// Fill helpers
// ─────────────────────────────────────────────────────────────────────────────

static void random_fill(Grid& g, float density, int seed) {
    srand((unsigned)seed);
    for (int r=0; r<g.H; r++)
        for (int c=0; c<g.W; c++) {
            int alive = (rand()/(float)RAND_MAX) < density ? 1 : 0;
            g.set_with_age(r,c,alive);
        }
}

static long long count_live(const Grid& g) {
    long long t=0; int rw=(g.W+63)/64;
    for (int r=0; r<g.H; r++) {
        const uint64_t* rp=g.brow(r);
        for (int w=0; w<rw; w++) t+=__builtin_popcountll(rp[w]);
    }
    return t;
}

// ─────────────────────────────────────────────────────────────────────────────
// Named seed patterns (centered on grid)
// ─────────────────────────────────────────────────────────────────────────────

struct CellsPattern {
    std::string name;
    std::vector<std::string> rows;
};

static const CellsPattern BUILT_IN_PATTERNS[] = {
    {"glider",      {".O.", "..O", "OOO"}},
    {"blinker",     {"OOO"}},
    {"block",       {"OO","OO"}},
    {"r-pentomino", {".OO","OO.",".O."}},
    {"acorn",       {".O.....","...O...","OO..OOO"}},
    {"pulsar",      {"..OOO...OOO..",
                     ".............",
                     "O....O.O....O",
                     "O....O.O....O",
                     "O....O.O....O",
                     "..OOO...OOO..",
                     ".............",
                     "..OOO...OOO..",
                     "O....O.O....O",
                     "O....O.O....O",
                     "O....O.O....O",
                     ".............",
                     "..OOO...OOO.."}},
    {"lwss",        {".O..O","O....","O...O","OOOO."}},
    {"diehard",     {"......O.","OO......",".O...OOO"}},
    {"gosper-gun",  {
        "........................O...........",
        "......................O.O...........",
        "............OO......OO............OO",
        "...........O...O....OO............OO",
        "OO........O.....O...OO..............",
        "OO........O...O.OO....O.O...........",
        "..........O.....O.......O...........",
        "...........O...O....................",
        "............OO......................"
    }},
};

static bool is_builtin_pattern(const std::string& name) {
    for (auto& p : BUILT_IN_PATTERNS)
        if (p.name == name) return true;
    return false;
}

static bool load_named_pattern(Grid& g, const std::string& name) {
    const CellsPattern* pat = nullptr;
    for (auto& p : BUILT_IN_PATTERNS)
        if (p.name == name) { pat = &p; break; }
    if (!pat) {
        fprintf(stderr,"Unknown seed pattern '%s'. Available: ", name.c_str());
        for (auto& p : BUILT_IN_PATTERNS) fprintf(stderr,"%s ", p.name.c_str());
        fprintf(stderr,"\n");
        return false;
    }
    int ph = (int)pat->rows.size();
    int pw = 0;
    for (auto& r : pat->rows) pw = std::max(pw,(int)r.size());
    int sr = (g.H - ph) / 2, sc = (g.W - pw) / 2;
    for (int pr=0; pr<ph; pr++) {
        int gr = sr+pr; if (gr<0||gr>=g.H) continue;
        for (int pc=0; pc<(int)pat->rows[pr].size(); pc++) {
            int gc = sc+pc; if (gc<0||gc>=g.W) continue;
            char ch = pat->rows[pr][pc];
            if (ch=='O'||ch=='o'||ch=='*') g.set_with_age(gr,gc,1);
        }
    }
    fprintf(stderr,"Loaded pattern '%s' (%dx%d) centered at (%d,%d)\n",
            name.c_str(),pw,ph,sc,sr);
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// .cells / .gol loader
// ─────────────────────────────────────────────────────────────────────────────

static bool load_cells(Grid& g, const std::string& path) {
    FILE* f = fopen(path.c_str(), "r");
    if (!f) { fprintf(stderr,"Error: cannot open '%s'\n",path.c_str()); return false; }
    std::vector<std::string> lines;
    char buf[65536];
    while (fgets(buf,sizeof(buf),f)) {
        if (buf[0]=='!'||buf[0]=='#') continue;
        int len=(int)strlen(buf);
        while (len>0&&(buf[len-1]=='\r'||buf[len-1]=='\n')) buf[--len]=0;
        lines.emplace_back(buf);
    }
    fclose(f);
    int pat_h=(int)lines.size(), pat_w=0;
    for (auto& l:lines) pat_w=std::max(pat_w,(int)l.size());
    int sr=(g.H-pat_h)/2, sc=(g.W-pat_w)/2;
    for (int pr=0; pr<pat_h; pr++) {
        int gr=sr+pr; if (gr<0||gr>=g.H) continue;
        for (int pc=0; pc<(int)lines[pr].size(); pc++) {
            int gc=sc+pc; if (gc<0||gc>=g.W) continue;
            char ch=lines[pr][pc];
            if (ch=='O'||ch=='o'||ch=='*') g.set_with_age(gr,gc,1);
            else if (ch=='.') g.set_with_age(gr,gc,0);
        }
    }
    fprintf(stderr,"Loaded '%s': %dx%d pattern, centered at (%d,%d)\n",
            path.c_str(),pat_w,pat_h,sc,sr);
    return true;
}

static bool load_gol_frame(Grid& g, const std::string& path, int target_gen=0) {
    FILE* f = fopen(path.c_str(),"rb");
    if (!f) { fprintf(stderr,"Cannot open '%s'\n",path.c_str()); return false; }
    uint8_t hdr[64];
    if (fread(hdr,1,64,f)!=64||hdr[0]!='G'||hdr[1]!='O'||hdr[2]!='L'||hdr[3]!='1') {
        fprintf(stderr,"Not a valid .gol file\n"); fclose(f); return false;
    }
    auto ru32=[](uint8_t* p){return (uint32_t)p[0]|((uint32_t)p[1]<<8)|((uint32_t)p[2]<<16)|((uint32_t)p[3]<<24);};
    uint32_t W=ru32(hdr+4), H=ru32(hdr+8);
    if ((int)W!=g.W||(int)H!=g.H) {
        fprintf(stderr,".gol grid %dx%d doesn't match target %dx%d\n",W,H,g.W,g.H);
        fclose(f); return false;
    }
    size_t nbytes=(size_t)((W*H+7)/8);
    uint8_t fhdr[12];
    for (int gen=0;;gen++) {
        if (fread(fhdr,1,12,f)!=12) {
            fprintf(stderr,".gol: gen %d not found\n",target_gen); fclose(f); return false;
        }
        if (gen==target_gen) break;
        fseek(f,(long)nbytes,SEEK_CUR);
    }
    std::vector<uint8_t> buf(nbytes);
    fread(buf.data(),1,nbytes,f);
    fclose(f);
    long long bit_pos=0;
    for (int r=0;r<g.H;r++)
        for (int c=0;c<g.W;c++) {
            int alive=(buf[bit_pos/8]>>(bit_pos%8))&1;
            g.set_with_age(r,c,alive);
            bit_pos++;
        }
    fprintf(stderr,"Loaded gen %d from '%s'\n",target_gen,path.c_str());
    return true;
}

static bool load_state(Grid& g, const std::string& path) {
    if (path.size()>4 && path.substr(path.size()-4)==".gol")
        return load_gol_frame(g, path, 0);
    return load_cells(g, path);
}

// ─────────────────────────────────────────────────────────────────────────────
// Renderer (age -> RGB)
// ─────────────────────────────────────────────────────────────────────────────

struct RGB { uint8_t r,g,b; };
static RGB age_lut[129];

static void build_age_lut() {
    struct Stop { float t; uint8_t r,g,b; };
    static const Stop stops[] = {
        {0.00f, 255,255,255},
        {0.15f, 255,255,  0},
        {0.38f, 255,140,  0},
        {0.60f,   0,200, 60},
        {0.78f,   0, 80,255},
        {1.00f,   0, 20, 80},
    };
    age_lut[0]={0,0,0};
    for (int age=1; age<=128; age++) {
        float t=(age-1)/127.0f;
        RGB c={255,255,255};
        for (int i=1; i<6; i++) {
            if (t<=stops[i].t) {
                float s=(t-stops[i-1].t)/(stops[i].t-stops[i-1].t);
                c.r=(uint8_t)(stops[i-1].r+s*(stops[i].r-stops[i-1].r));
                c.g=(uint8_t)(stops[i-1].g+s*(stops[i].g-stops[i-1].g));
                c.b=(uint8_t)(stops[i-1].b+s*(stops[i].b-stops[i-1].b));
                break;
            }
        }
        age_lut[age]=c;
    }
}

static void render_frame(const Grid& g, int rW, int rH, std::vector<uint8_t>& buf) {
    buf.resize((size_t)rW*rH*3);
    #pragma omp parallel for schedule(static)
    for (int py=0; py<rH; py++) {
        int gr0=(int)((double)py*g.H/rH), gr1=(int)((double)(py+1)*g.H/rH);
        if (gr1<=gr0) gr1=gr0+1; if (gr1>g.H) gr1=g.H;
        uint8_t* out_row=buf.data()+(size_t)py*rW*3;
        for (int px=0; px<rW; px++) {
            int gc0=(int)((double)px*g.W/rW), gc1=(int)((double)(px+1)*g.W/rW);
            if (gc1<=gc0) gc1=gc0+1; if (gc1>g.W) gc1=g.W;
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

static void run_bench(int W, int H, int GENS, int THREADS, const std::string& pattern="") {
    omp_set_num_threads(THREADS);
    fprintf(stderr,"=== GoL Benchmark ===\nGrid: %dx%d  Gens: %d  Threads: %d\n\n",W,H,GENS,THREADS);
    Grid A(W,H),B(W,H);
    if (!pattern.empty()) load_named_pattern(A, pattern);
    else                  random_fill(A,0.3f,42);
    sync_borders(A);
    fprintf(stderr,"Warmup...\n");
    for (int i=0;i<10;i++){step_rules(A,B,8,12);sync_borders(B);std::swap(A.bits,B.bits);std::swap(A.age,B.age);}
    auto t0=std::chrono::high_resolution_clock::now();
    for (int i=0;i<GENS;i++){step_rules(A,B,8,12);sync_borders(B);std::swap(A.bits,B.bits);std::swap(A.age,B.age);}
    double el=std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-t0).count();
    fprintf(stderr,"Time: %.3fs  GPS: %.1f  GCUPS: %.3f\n",el,GENS/el,(double)W*H*GENS/el/1e9);
}

// ─────────────────────────────────────────────────────────────────────────────
// Video mode
// ─────────────────────────────────────────────────────────────────────────────

static void run_video(int W, int H, int GENS, int FPS, int rW, int rH,
                      int THREADS, const std::string& state_file,
                      const std::string& seed_pattern,
                      uint8_t rule_B, uint8_t rule_S, int seed, float density)
{
    omp_set_num_threads(THREADS);
    build_age_lut();
    SET_STDOUT_BINARY();

    fprintf(stderr,"=== GoL Video Export ===\n");
    fprintf(stderr,"Grid: %dx%d  Rules: %s  Render: %dx%d @ %d fps\n",
            W,H,rules_str(rule_B,rule_S).c_str(),rW,rH,FPS);
    fprintf(stderr,"Coverage: %.2f x %.2f cells/pixel\n",(double)W/rW,(double)H/rH);
    fprintf(stderr,"Duration: %d gens = %.1fs\n\n",GENS,(double)GENS/FPS);

    Grid A(W,H),B(W,H);
    if (!state_file.empty()) {
        if (!load_state(A,state_file)) return;
    } else if (!seed_pattern.empty()) {
        if (!load_named_pattern(A, seed_pattern)) return;
    } else {
        random_fill(A,density,seed);
    }
    sync_borders(A);

    std::vector<uint8_t> frame_buf;
    size_t frame_bytes=(size_t)rW*rH*3;
    auto wall0=std::chrono::high_resolution_clock::now();

    for (int gen=0; gen<GENS; gen++) {
        render_frame(A,rW,rH,frame_buf);
        fwrite(frame_buf.data(),1,frame_bytes,stdout);
        if (gen%100==99) fflush(stdout);

        step_rules(A,B,rule_B,rule_S);
        sync_borders(B);
        std::swap(A.bits,B.bits); std::swap(A.age,B.age);

        if (gen%50==0||gen==GENS-1) {
            auto now=std::chrono::high_resolution_clock::now();
            double el=std::chrono::duration<double>(now-wall0).count();
            fprintf(stderr,"  Frame %5d/%d  %.1f fps  ETA %.0fs\n",
                    gen+1,GENS,(gen+1)/el,(GENS-gen-1)/((gen+1)/el));
        }
    }
    fflush(stdout);
    double total=std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-wall0).count();
    fprintf(stderr,"\nDone. %.1fs for %.1fs of video (%.1fx)\n",total,(double)GENS/FPS,(double)GENS/FPS/total);
}

// ─────────────────────────────────────────────────────────────────────────────
// Sim mode (config-driven, stagnation detection, archive)
// ─────────────────────────────────────────────────────────────────────────────

static int run_sim(const Config& cfg) {
    omp_set_num_threads(cfg.threads);
    build_age_lut();

    if (cfg.has_video()) SET_STDOUT_BINARY();

    fprintf(stderr,"=== GoL Simulation ===\n");
    fprintf(stderr,"Grid:    %dx%d\n",cfg.grid_w,cfg.grid_h);
    fprintf(stderr,"Rules:   %s\n",rules_str(cfg.rule_B,cfg.rule_S).c_str());
    fprintf(stderr,"Max:     %d gens\n",cfg.max_gens);
    fprintf(stderr,"Stag:    %s  still=%s  cycle=%s  window=%d  extinct=%s\n",
            cfg.stag_enable?"on":"off",
            cfg.stag_still?"on":"off",
            cfg.stag_cycle?"on":"off",
            cfg.cycle_window,
            cfg.stag_extinct?"on":"off");
    if (cfg.has_video())
        fprintf(stderr,"Video:   %dx%d @ %d fps\n",cfg.video_w,cfg.video_h,cfg.video_fps);
    if (cfg.has_archive())
        fprintf(stderr,"Archive: %s (every %d gen)\n",cfg.archive_file.c_str(),cfg.archive_every);
    fprintf(stderr,"Threads: %d\n\n",cfg.threads);

    Grid A(cfg.grid_w,cfg.grid_h), B(cfg.grid_w,cfg.grid_h);

    if (!cfg.state_file.empty()) {
        if (!load_state(A, cfg.state_file)) return 1;
    } else if (!cfg.seed_pattern.empty()) {
        if (!load_named_pattern(A, cfg.seed_pattern)) return 1;
    } else {
        random_fill(A, cfg.random_density, cfg.random_seed);
    }

    fprintf(stderr,"Initial live: %lld (%.2f%%)\n",
            count_live(A), 100.0*count_live(A)/((long long)cfg.grid_w*cfg.grid_h));
    sync_borders(A);

    GolArchive archive;
    if (cfg.has_archive()) {
        archive.open(cfg.archive_file, cfg.grid_w, cfg.grid_h,
                     cfg.rule_B, cfg.rule_S, (uint32_t)cfg.video_fps);
    }

    StagnationDetector stag(cfg);
    StagnationResult stag_result;

    std::vector<uint8_t> frame_buf;
    size_t frame_bytes = cfg.has_video() ? (size_t)cfg.video_w*cfg.video_h*3 : 0;
    auto last_report = std::chrono::high_resolution_clock::now();
    auto t0 = last_report;

    if (cfg.has_archive() && cfg.archive_every > 0)
        archive.write_frame(0, count_live(A), A);
    if (cfg.has_video()) {
        render_frame(A,cfg.video_w,cfg.video_h,frame_buf);
        fwrite(frame_buf.data(),1,frame_bytes,stdout);
    }

    int final_gen = cfg.max_gens;
    for (int gen=1; gen<=cfg.max_gens; gen++) {
        StepResult res = step_rules(A,B,cfg.rule_B,cfg.rule_S);
        sync_borders(B);
        std::swap(A.bits,B.bits); std::swap(A.age,B.age);

        if (cfg.has_archive() && gen % cfg.archive_every == 0)
            archive.write_frame(gen, res.live_count, A);

        if (cfg.stag_enable) {
            stag_result = stag.check(gen, res.live_count, A);
            if (stag_result.triggered) {
                final_gen = gen;
                if (cfg.has_video()) {
                    render_frame(A,cfg.video_w,cfg.video_h,frame_buf);
                    fwrite(frame_buf.data(),1,frame_bytes,stdout);
                    fflush(stdout);
                }
                break;
            }
        }

        if (cfg.has_video()) {
            render_frame(A,cfg.video_w,cfg.video_h,frame_buf);
            fwrite(frame_buf.data(),1,frame_bytes,stdout);
            if (gen%100==0) fflush(stdout);
        }

        auto now=std::chrono::high_resolution_clock::now();
        if (std::chrono::duration<double>(now-last_report).count()>=1.0||gen==cfg.max_gens) {
            double elapsed=std::chrono::duration<double>(now-t0).count();
            fprintf(stderr,"  Gen %6d/%d  %7.1f GPS  live: %lld        \r",
                    gen,cfg.max_gens,(double)gen/elapsed,(long long)res.live_count);
            last_report=now;
        }
    }

    fflush(stdout);

    int sp = stag_result.triggered ? stag_result.period : 0;
    int sg = stag_result.triggered ? stag_result.detect_gen : -1;
    archive.close(sp, sg);

    double elapsed=std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-t0).count();
    fprintf(stderr,"\n\n=== Complete ===\n");
    fprintf(stderr,"Gens: %d  Time: %.3fs  GPS: %.1f\n",
            final_gen,elapsed,(double)final_gen/elapsed);

    if (stag_result.triggered) {
        if (stag_result.period == -1)
            fprintf(stderr,"Stop: Extinction (all cells dead) at gen %d\n",stag_result.detect_gen);
        else if (stag_result.period == 1)
            fprintf(stderr,"Stop: Still life detected at gen %d\n",stag_result.detect_gen);
        else
            fprintf(stderr,"Stop: Cycle period %d, detected at gen %d (cycle starts gen %d)\n",
                    stag_result.period,stag_result.detect_gen,stag_result.cycle_start);
    } else {
        fprintf(stderr,"Stop: max_gens reached (no stagnation detected in window=%d)\n",
                cfg.cycle_window);
    }

    if (cfg.has_archive())
        fprintf(stderr,"Archive: %s  (%d frames)\n",cfg.archive_file.c_str(),archive.gen_count);

    return 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Usage / main
// ─────────────────────────────────────────────────────────────────────────────

static void usage(const char* p) {
    fprintf(stderr,
        "Conway's Game of Life v2 — Analysis & Archive Edition\n\n"
        "Build:  g++ -O3 -march=native -mavx2 -fopenmp -std=c++17 -o gol %s\n\n"
        "Modes:\n"
        "  %s bench  <W> <H> <gens> [threads] [pattern]\n"
        "  %s video  <W> <H> <gens> <fps> <rW> <rH> [threads] [state|pattern] [B3/S23]\n"
        "  %s sim    <config.cfg> [threads]\n"
        "  %s init   <config.cfg>\n"
        "  %s archive-info <file.gol>\n\n"
        "Pattern names (use directly in place of state file):\n"
        "  glider blinker block r-pentomino acorn pulsar lwss diehard gosper-gun\n\n"
        "Examples:\n"
        "  ./gol video 3840 2160 600 60 1920 1080 8 gosper-gun\n"
        "  ./gol video 1920 1080 500 60 1920 1080 8 glider B3/S23\n"
        "  ./gol bench 3840 2160 1000 8 glider\n\n"
        "Video pipe:\n"
        "  ./gol video 3840 2160 600 60 1920 1080 8 | ffmpeg \\\n"
        "    -f rawvideo -pixel_format rgb24 -video_size 1920x1080 -framerate 60 -i - \\\n"
        "    -c:v libx264 -preset veryslow -crf 10 -pix_fmt yuv420p life.mp4\n\n"
        "Sim with archive:\n"
        "  # In config: archive_file = run.gol; seed_pattern = gosper-gun\n"
        "  ./gol sim life.cfg\n"
        "  ./gol archive-info run.gol\n\n"
        "Rules: B3/S23 (Conway)  B36/S23 (HighLife)  B3/S12345 (Maze)  B2/S (Seeds)\n",
        p,p,p,p,p,p);
}

int main(int argc, char** argv) {
    if (argc < 2) { usage(argv[0]); return 1; }
    std::string mode = argv[1];

    if (mode=="init") {
        if (argc<3) { fprintf(stderr,"Usage: %s init <config>\n",argv[0]); return 1; }
        write_default_config(argv[2]); return 0;
    }
    if (mode=="archive-info") {
        if (argc<3) { fprintf(stderr,"Usage: %s archive-info <file.gol>\n",argv[0]); return 1; }
        archive_info(argv[2]); return 0;
    }
    if (mode=="bench") {
        if (argc<5) { usage(argv[0]); return 1; }
        std::string bench_pattern;
        if (argc>5 && is_builtin_pattern(argv[5])) bench_pattern=argv[5];
        run_bench(atoi(argv[2]),atoi(argv[3]),atoi(argv[4]),
                  argc>5 && bench_pattern.empty() ? atoi(argv[5]) : omp_get_max_threads(),
                  bench_pattern);
        return 0;
    }
    if (mode=="video") {
        if (argc<8) { usage(argv[0]); return 1; }
        int W=atoi(argv[2]),H=atoi(argv[3]),GENS=atoi(argv[4]);
        int FPS=atoi(argv[5]),rW=atoi(argv[6]),rH=atoi(argv[7]);
        int threads=omp_get_max_threads();
        std::string state, pattern; uint8_t rule_B=8,rule_S=12;
        int seed=42; float density=0.3f;
        for (int i=8;i<argc;i++) {
            std::string a=argv[i];
            if (a[0]=='B'||a[0]=='b') parse_rules(a,rule_B,rule_S);
            else if (a.find('.')!=std::string::npos||a.find('/')!=std::string::npos
                     ||a.find('\\')!=std::string::npos) state=a;
            else if (is_builtin_pattern(a)) pattern=a;
            else { int v=atoi(a.c_str()); if (v>0) threads=v; }
        }
        run_video(W,H,GENS,FPS,rW,rH,threads,state,pattern,rule_B,rule_S,seed,density);
        return 0;
    }
    if (mode=="sim") {
        if (argc<3) { usage(argv[0]); return 1; }
        Config cfg;
        if (!parse_config(argv[2],cfg)) return 1;
        if (argc>3) cfg.threads=atoi(argv[3]);
        return run_sim(cfg);
    }

    usage(argv[0]); return 1;
}
