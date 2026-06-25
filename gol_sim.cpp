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
#include <zstd.h>

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

struct SeedPlacement {
    std::string name;   // built-in pattern name or file path
    int cx, cy;         // center on grid (-1,-1 = auto-center)
    bool is_file;       // true if name is a file path
};

struct Config {
    int   grid_w          = 1920;
    int   grid_h          = 1080;
    int   threads         = 0;
    uint8_t rule_B        = 8;   // B3
    uint8_t rule_S        = 12;  // S23
    std::vector<SeedPlacement> seeds;
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
    std::string final_cells_file;

    bool has_video() const { return video_w > 0 && video_h > 0; }
    bool has_archive() const { return !archive_file.empty(); }
    bool has_seeds() const { return !seeds.empty(); }
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
        else if (k=="state_file")       cfg.seeds.push_back({v, -1, -1, true});
        else if (k=="seed_pattern")     cfg.seeds.push_back({v, -1, -1, false});
        else if (k=="seed") {
            std::vector<std::string> tok;
            char tmp[1024]; strncpy(tmp,v.c_str(),sizeof(tmp)-1);
            char* t = strtok(tmp," \t");
            while (t) { tok.push_back(t); t=strtok(nullptr," \t"); }
            int cx=-1, cy=-1, pos = (int)tok.size();
            auto is_int = [](const std::string& s) {
                if (s.empty()) return false;
                size_t i=0; if (s[i]=='-'||s[i]=='+') i++;
                return i<s.size() && s.find_first_not_of("0123456789",i)==std::string::npos;
            };
            if (pos>=3 && is_int(tok[pos-2]) && is_int(tok[pos-1])) {
                cx = atoi(tok[pos-2].c_str()); cy = atoi(tok[pos-1].c_str());
                pos -= 2;
            } else if (pos>=2 && is_int(tok[pos-1])) {
                cx = atoi(tok[pos-1].c_str());
                pos -= 1;
            }
            std::string name;
            for (int i=0;i<pos;i++) { if (i) name+=' '; name+=tok[i]; }
            bool is_file = name.find('.')!=std::string::npos
                        || name.find('/')!=std::string::npos
                        || name.find('\\')!=std::string::npos;
            cfg.seeds.push_back({name, cx, cy, is_file});
        }
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
        else if (k=="final_cells_file") cfg.final_cells_file = v;
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
        "# Seeds: place one or more patterns on the initial grid.\n"
        "#   seed = <name_or_file> [cx] [cy]   (cx,cy = center coords; omit = auto-center)\n"
        "#   seed_pattern = <name>              (backward compat, single; looks in built_in_patterns/)\n"
        "#   state_file    = <path>              (backward compat, single)\n"
        "# If no seeds are given, random fill is used.\n"
        "seed = gosper-gun                      # auto-centered\n"
        "seed = glider 100 200                  # placed at center (100,200)\n"
        "seed = block 500 500\n"
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
        "archive_every   = 1          # 1 = every gen, 10 = every 10th, etc.\n\n"
        "# Optional: dump the final (stagnated) state as a .cells file for a\n"
        "# quick look in Golly / LifeViewer. Trims to the live-cell bounding box.\n"
        "final_cells_file =            # e.g. final_state.cells\n"
    );
    fclose(f);
    fprintf(stderr, "Wrote default config: %s\n", path.c_str());
}

// ─────────────────────────────────────────────────────────────────────────────
// Grid
// ─────────────────────────────────────────────────────────────────────────────

struct Grid {
    uint64_t* bits;
    int W, H;
    int words_per_row;

    Grid(int w, int h) : W(w), H(h) {
        words_per_row = (w + 63) / 64 + 2;
        size_t bsz = (size_t)(h + 2) * words_per_row * sizeof(uint64_t);
#ifdef _WIN32
        bits = (uint64_t*)_aligned_malloc(bsz, 32);
#else
        bits = (uint64_t*)aligned_alloc(32, (bsz + 31) & ~(size_t)31);
#endif
        memset(bits, 0, bsz);
    }
    ~Grid() {
#ifdef _WIN32
        _aligned_free(bits);
#else
        free(bits);
#endif
    }

    inline uint64_t* brow(int r)            { return bits + (size_t)(r+1)*words_per_row + 1; }
    inline const uint64_t* brow(int r) const{ return bits + (size_t)(r+1)*words_per_row + 1; }

    inline int  get(int r, int c) const { return (brow(r)[c/64] >> (c%64)) & 1; }
    inline void set(int r, int c, int v) {
        uint64_t& w = brow(r)[c/64];
        w = (w & ~(1ULL<<(c%64))) | ((uint64_t)v<<(c%64));
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
            // window <= 0 means unbounded: keep every hash for the whole
            // run. Memory cost is trivial (~24 bytes/gen) compared to the
            // value of not silently missing long-period cycles.
            if (window > 0 && (int)history.size() > window) {
                auto& front = history.front();
                hash_map.erase(front.second);
                history.pop_front();
            }
        }

        return {false};
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Compression helpers (.gol archive)
// ─────────────────────────────────────────────────────────────────────────────

enum { GOL_COMPRESS_RAW = 0, GOL_COMPRESS_DELTA = 1 };
enum { GOL_BLOCK_LOG2 = 7, GOL_BLOCK_SIZE = 1 << GOL_BLOCK_LOG2 }; // 128 bytes

// Block-sparse encode: split `data` (nbytes) into blocks, write bitmask + kept blocks.
// Returns compressed data (caller free). Sets *out_size.
static uint8_t* block_sparse_encode(const uint8_t* data, size_t nbytes, size_t* out_size) {
    size_t nblocks = (nbytes + GOL_BLOCK_SIZE - 1) / GOL_BLOCK_SIZE;
    size_t mask_bytes = (nblocks + 7) / 8;
    std::vector<uint8_t> mask(mask_bytes, 0);
    std::vector<uint8_t> blocks;
    blocks.reserve(nbytes); // worst case

    for (size_t i = 0; i < nblocks; i++) {
        size_t off = i * GOL_BLOCK_SIZE;
        size_t remain = nbytes - off;
        size_t blen = (remain < GOL_BLOCK_SIZE) ? remain : GOL_BLOCK_SIZE;
        bool all_zero = true;
        for (size_t j = 0; j < blen; j++) {
            if (data[off + j]) { all_zero = false; break; }
        }
        if (!all_zero) {
            mask[i / 8] |= (uint8_t)(1 << (i % 8));
            blocks.insert(blocks.end(), data + off, data + off + blen);
        }
    }

    *out_size = mask_bytes + blocks.size();
    uint8_t* result = (uint8_t*)malloc(*out_size);
    if (!result) return nullptr;
    memcpy(result, mask.data(), mask_bytes);
    if (!blocks.empty()) memcpy(result + mask_bytes, blocks.data(), blocks.size());
    return result;
}

// Block-sparse decode: inverse of above.
static uint8_t* block_sparse_decode(const uint8_t* compressed, size_t csize, size_t nbytes, size_t* out_size) {
    size_t nblocks = (nbytes + GOL_BLOCK_SIZE - 1) / GOL_BLOCK_SIZE;
    size_t mask_bytes = (nblocks + 7) / 8;
    if (csize < mask_bytes) return nullptr;

    uint8_t* result = (uint8_t*)calloc(1, nbytes);
    if (!result) return nullptr;
    *out_size = nbytes;

    const uint8_t* mask = compressed;
    const uint8_t* blocks = compressed + mask_bytes;
    size_t block_pos = 0;

    for (size_t i = 0; i < nblocks; i++) {
        size_t off = i * GOL_BLOCK_SIZE;
        size_t blen = ((off + GOL_BLOCK_SIZE) <= nbytes) ? GOL_BLOCK_SIZE : (nbytes - off);
        if (mask[i / 8] & (1 << (i % 8))) {
            memcpy(result + off, blocks + block_pos, blen);
            block_pos += blen;
        }
    }
    return result;
}

// Zstd wrapper: compress src (size) → *dst, *dst_size allocated by malloc.
static bool zstd_compress(const uint8_t* src, size_t size, uint8_t** dst, size_t* dst_size, int level) {
    size_t bound = ZSTD_compressBound(size);
    *dst = (uint8_t*)malloc(bound);
    if (!*dst) return false;
    *dst_size = ZSTD_compress(*dst, bound, src, size, level);
    if (ZSTD_isError(*dst_size)) { free(*dst); *dst = nullptr; return false; }
    // Shrink allocation
    uint8_t* shrunk = (uint8_t*)realloc(*dst, *dst_size);
    if (shrunk) *dst = shrunk;
    return true;
}

static bool zstd_decompress(const uint8_t* src, size_t src_size, uint8_t** dst, size_t dst_size) {
    *dst = (uint8_t*)malloc(dst_size);
    if (!*dst) return false;
    size_t res = ZSTD_decompress(*dst, dst_size, src, src_size);
    if (ZSTD_isError(res) || res != dst_size) { free(*dst); *dst = nullptr; return false; }
    return true;
}

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
    uint8_t  compression = GOL_COMPRESS_DELTA;
    int      zstd_level  = 19;
    size_t   nbytes      = 0;
    uint8_t* prev_bits   = nullptr;

    static const size_t HEADER_SIZE = 64;

    bool open(const std::string& path, int W, int H,
              uint8_t B, uint8_t S, uint32_t fps=0,
              uint8_t comp=GOL_COMPRESS_DELTA, int zl=19) {
        fp = fopen(path.c_str(), "wb");
        if (!fp) { fprintf(stderr,"Archive: cannot open '%s'\n",path.c_str()); return false; }
        grid_w=W; grid_h=H; rule_B=B; rule_S=S; fps_hint=fps;
        compression=comp; zstd_level=zl;
        nbytes = (size_t)((size_t)W*H+7)/8;
        prev_bits = (uint8_t*)calloc(1, nbytes);
        if (!prev_bits) { fclose(fp); fp=nullptr; return false; }
        uint8_t hdr[HEADER_SIZE] = {};
        hdr[0]='G'; hdr[1]='O'; hdr[2]='L'; hdr[3]='1';
        write_u32(hdr+4,  W);
        write_u32(hdr+8,  H);
        write_u32(hdr+12, 0);
        write_u32(hdr+16, fps);
        hdr[20]=B; hdr[24]=S;
        write_i32(hdr+28, -1);
        write_i32(hdr+32, -1);
        hdr[36] = compression;
        hdr[37] = (uint8_t)zstd_level;
        fwrite(hdr, 1, HEADER_SIZE, fp);
        return true;
    }

    void write_frame(int gen, long long live_count, const Grid& g) {
        if (!fp) return;
        // Pack grid bits into byte array
        std::vector<uint8_t> cur(nbytes, 0);
        long long bit_pos = 0;
        for (int r=0; r<g.H; r++) {
            const uint64_t* row = g.brow(r);
            for (int c=0; c<g.W; c++) {
                if ((row[c/64] >> (c%64)) & 1) cur[bit_pos/8] |= (1 << (bit_pos%8));
                bit_pos++;
            }
        }

        if (compression == GOL_COMPRESS_DELTA && gen > 0) {
            // Delta encode: XOR with previous frame
            std::vector<uint8_t> delta(nbytes);
            for (size_t i = 0; i < nbytes; i++) delta[i] = cur[i] ^ prev_bits[i];

            // Block-sparse encode the delta
            size_t sparse_size = 0;
            uint8_t* sparse = block_sparse_encode(delta.data(), nbytes, &sparse_size);
            if (!sparse) { /* fall through to raw */ goto write_raw; }

            // zstd compress the sparse data
            size_t csize = 0;
            uint8_t* cdata = nullptr;
            if (!zstd_compress(sparse, sparse_size, &cdata, &csize, zstd_level)) {
                free(sparse); goto write_raw;
            }

            // Write compressed frame header: 8+4+1+4+4 + csize
            uint8_t fhdr[21];
            write_u64(fhdr+0, (uint64_t)gen);
            write_u32(fhdr+8, (uint32_t)live_count);
            fhdr[12] = GOL_COMPRESS_DELTA;
            write_u32(fhdr+13, (uint32_t)sparse_size);
            write_u32(fhdr+17, (uint32_t)csize);
            fwrite(fhdr, 1, 21, fp);
            fwrite(cdata, 1, csize, fp);
            free(sparse); free(cdata);
        } else {
            // Frame 0 or raw mode: block-sparse + zstd
            write_raw: {
                size_t sparse_size = 0;
                uint8_t* sparse = block_sparse_encode(cur.data(), nbytes, &sparse_size);
                size_t csize = 0;
                uint8_t* cdata = nullptr;
                if (sparse && zstd_compress(sparse, sparse_size, &cdata, &csize, zstd_level)) {
                    uint8_t fhdr[21];
                    write_u64(fhdr+0, (uint64_t)gen);
                    write_u32(fhdr+8, (uint32_t)live_count);
                    fhdr[12] = GOL_COMPRESS_DELTA;
                    write_u32(fhdr+13, (uint32_t)sparse_size);
                    write_u32(fhdr+17, (uint32_t)csize);
                    fwrite(fhdr, 1, 21, fp);
                    fwrite(cdata, 1, csize, fp);
                    free(sparse); free(cdata);
                } else {
                    free(sparse);
                    // Fallback: raw uncompressed frame
                    uint8_t fhdr[12];
                    write_u64(fhdr+0, (uint64_t)gen);
                    write_u32(fhdr+8, (uint32_t)live_count);
                    fwrite(fhdr, 1, 12, fp);
                    fwrite(cur.data(), 1, nbytes, fp);
                }
            }
        }

        // Save as previous for next delta
        memcpy(prev_bits, cur.data(), nbytes);
        gen_count++;
    }

    void close(int stag_period=-1, int stag_gen=-1) {
        if (!fp) return;
        fseek(fp, 12, SEEK_SET);
        uint8_t tmp[8];
        write_u32(tmp, gen_count);  fwrite(tmp, 1, 4, fp);
        fseek(fp, 28, SEEK_SET);
        write_i32(tmp, stag_period); fwrite(tmp, 1, 4, fp);
        fclose(fp); fp=nullptr;
        free(prev_bits); prev_bits=nullptr;
    }

    ~GolArchive() { if (fp) close(); else free(prev_bits); }

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

    uint8_t comp    = hdr[36];
    int     zl      = hdr[37];

    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fclose(f);

    fprintf(stderr,"=== .gol Archive Info ===\n");
    fprintf(stderr,"File:        %s  (%.2f MB)\n", path.c_str(), fsize/1048576.0);
    fprintf(stderr,"Grid:        %dx%d\n", W, H);
    fprintf(stderr,"Rules:       %s\n", rules_str(B,S).c_str());
    fprintf(stderr,"Compression: %s (zstd level %d)\n",
            comp==0?"none":"delta+blocksparse+zstd", zl);
    fprintf(stderr,"Generations: %u\n", gens);
    if (fps>0) fprintf(stderr,"FPS hint:    %u\n", fps);
    if (sp==-1)       fprintf(stderr,"Stagnation:  extinction\n");
    else if (sp==0)   fprintf(stderr,"Stagnation:  max_gens reached (none detected)\n");
    else              fprintf(stderr,"Stagnation:  period %d cycle, detected at gen %d\n", sp, sg);
    size_t raw_per = 12 + (size_t)((W*H+7)/8);
    double ratio = (double)raw_per * gens / fsize;
    fprintf(stderr,"Raw frame:   %zu bytes  |  Compressed: %.2f MB total  |  Ratio: %.1fx\n",
            raw_per, fsize/1048576.0, ratio);
}

// ─────────────────────────────────────────────────────────────────────────────
// Fill helpers
// ─────────────────────────────────────────────────────────────────────────────

static void random_fill(Grid& g, float density, int seed) {
    srand((unsigned)seed);
    for (int r=0; r<g.H; r++)
        for (int c=0; c<g.W; c++) {
            int alive = (rand()/(float)RAND_MAX) < density ? 1 : 0;
            g.set(r,c,alive);
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
// Named seed patterns (loaded from built_in_patterns/*.cells)
// ─────────────────────────────────────────────────────────────────────────────

static bool load_cells(Grid& g, const std::string& path, int cx, int cy);

static std::string pattern_path(const std::string& name) {
    return std::string("built_in_patterns/") + name + ".cells";
}

static bool is_builtin_pattern(const std::string& name) {
    FILE* f = fopen(pattern_path(name).c_str(), "r");
    if (f) { fclose(f); return true; }
    return false;
}

static bool load_named_pattern(Grid& g, const std::string& name, int cx=-1, int cy=-1) {
    return load_cells(g, pattern_path(name), cx, cy);
}

// ─────────────────────────────────────────────────────────────────────────────
// .cells / .gol loader
// ─────────────────────────────────────────────────────────────────────────────

static bool load_cells(Grid& g, const std::string& path, int cx=-1, int cy=-1) {
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
    int sr=(cx<0)?(g.H-pat_h)/2:cy-pat_h/2;
    int sc=(cx<0)?(g.W-pat_w)/2:cx-pat_w/2;
    for (int pr=0; pr<pat_h; pr++) {
        int gr=sr+pr; if (gr<0||gr>=g.H) continue;
        for (int pc=0; pc<(int)lines[pr].size(); pc++) {
            int gc=sc+pc; if (gc<0||gc>=g.W) continue;
            char ch=lines[pr][pc];
            if (ch=='O'||ch=='o'||ch=='*') g.set(gr,gc,1);
            else if (ch=='.') g.set(gr,gc,0);
        }
    }
    fprintf(stderr,"Loaded '%s': %dx%d pattern, placed at (%d,%d)\n",
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
    uint8_t compression = hdr[36];
    size_t nbytes=(size_t)((W*H+7)/8);
    std::vector<uint8_t> prev_bits(nbytes, 0);
    std::vector<uint8_t> cur_bits(nbytes, 0);

    for (int gen=0; gen<=target_gen; gen++) {
        if (compression == GOL_COMPRESS_DELTA) {
            uint8_t fhdr[21];
            size_t got = fread(fhdr,1,21,f);
            if (got<21) { fprintf(stderr,".gol: frame %d truncated\n",gen); fclose(f); return false; }
            uint8_t  ctype  = fhdr[12];
            (void)ctype;
            uint32_t usize  = (uint32_t)fhdr[13]|((uint32_t)fhdr[14]<<8)|((uint32_t)fhdr[15]<<16)|((uint32_t)fhdr[16]<<24);
            uint32_t csize  = (uint32_t)fhdr[17]|((uint32_t)fhdr[18]<<8)|((uint32_t)fhdr[19]<<16)|((uint32_t)fhdr[20]<<24);
            std::vector<uint8_t> cdata(csize);
            if (fread(cdata.data(),1,csize,f)!=csize) {
                fprintf(stderr,".gol: frame %d data truncated\n",gen); fclose(f); return false;
            }
            uint8_t* sparse = nullptr;
            if (!zstd_decompress(cdata.data(),csize,&sparse,usize)) {
                fprintf(stderr,".gol: frame %d decompress failed\n",gen); fclose(f); return false;
            }
            size_t decoded_nbytes=0;
            uint8_t* decoded = block_sparse_decode(sparse,usize,nbytes,&decoded_nbytes);
            free(sparse);
            if (!decoded) { fprintf(stderr,".gol: frame %d block decode failed\n",gen); fclose(f); return false; }
            if (gen==0) {
                memcpy(cur_bits.data(),decoded,nbytes);
            } else {
                for (size_t i=0;i<nbytes;i++) cur_bits[i] = decoded[i] ^ prev_bits[i];
            }
            if (gen<target_gen) memcpy(prev_bits.data(),cur_bits.data(),nbytes);
            free(decoded);
        } else {
            uint8_t fhdr[12];
            if (fread(fhdr,1,12,f)!=12) {
                fprintf(stderr,".gol: gen %d not found\n",target_gen); fclose(f); return false;
            }
            if (gen==target_gen) {
                if (fread(cur_bits.data(),1,nbytes,f)!=(size_t)nbytes) {
                    fprintf(stderr,".gol: frame %d data truncated\n",gen); fclose(f); return false;
                }
            } else {
                fseek(f,(long)nbytes,SEEK_CUR);
            }
        }
    }
    fclose(f);

    long long bit_pos=0;
    for (int r=0;r<g.H;r++)
        for (int c=0;c<g.W;c++) {
            int alive=(cur_bits[bit_pos/8]>>(bit_pos%8))&1;
            g.set(r,c,alive);
            bit_pos++;
        }
    fprintf(stderr,"Loaded gen %d from '%s'\n",target_gen,path.c_str());
    return true;
}

static bool load_state(Grid& g, const std::string& path, int cx=-1, int cy=-1) {
    if (path.size()>4 && path.substr(path.size()-4)==".gol")
        return load_gol_frame(g, path, 0);
    return load_cells(g, path, cx, cy);
}

// ─────────────────────────────────────────────────────────────────────────────
// .cells writer — trims to the minimal bounding box of live cells (Golly
// plaintext convention) so the exported pattern can be re-loaded by load_cells,
// opened in Golly/LifeViewer, or diffed against a known pattern signature.
// ─────────────────────────────────────────────────────────────────────────────

static bool write_cells(const Grid& g, const std::string& path,
                         const std::string& comment = "") {
    int min_r=g.H, max_r=-1, min_c=g.W, max_c=-1;
    for (int r=0; r<g.H; r++) {
        const uint64_t* row = g.brow(r);
        for (int c=0; c<g.W; c++) {
            if ((row[c/64]>>(c%64))&1) {
                if (r<min_r) min_r=r;
                if (r>max_r) max_r=r;
                if (c<min_c) min_c=c;
                if (c>max_c) max_c=c;
            }
        }
    }

    FILE* f = fopen(path.c_str(), "w");
    if (!f) { fprintf(stderr,"Error: cannot write '%s'\n",path.c_str()); return false; }
    if (!comment.empty()) fprintf(f, "!%s\n", comment.c_str());

    if (max_r < 0) {
        fprintf(f, "!Extinct: 0 live cells\n");
        fclose(f);
        fprintf(stderr,"Wrote '%s': extinct (no live cells)\n",path.c_str());
        return true;
    }

    int bw = max_c-min_c+1, bh = max_r-min_r+1;
    for (int r=min_r; r<=max_r; r++) {
        const uint64_t* row = g.brow(r);
        std::string line(bw, '.');
        for (int c=min_c; c<=max_c; c++)
            if ((row[c/64]>>(c%64))&1) line[c-min_c]='O';
        fprintf(f, "%s\n", line.c_str());
    }
    fclose(f);
    fprintf(stderr,"Wrote '%s': %dx%d bounding box at (%d,%d)\n",
            path.c_str(),bw,bh,min_c,min_r);
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// extract mode: pull a single generation out of a .gol archive as .cells
// ─────────────────────────────────────────────────────────────────────────────

static bool run_extract(const std::string& gol_path, int target_gen, const std::string& out_path) {
    FILE* f = fopen(gol_path.c_str(), "rb");
    if (!f) { fprintf(stderr,"Cannot open '%s'\n",gol_path.c_str()); return false; }
    uint8_t hdr[64];
    if (fread(hdr,1,64,f)!=64||hdr[0]!='G'||hdr[1]!='O'||hdr[2]!='L'||hdr[3]!='1') {
        fprintf(stderr,"Not a valid .gol file\n"); fclose(f); return false;
    }
    auto ru32=[](uint8_t* p){return (uint32_t)p[0]|((uint32_t)p[1]<<8)|((uint32_t)p[2]<<16)|((uint32_t)p[3]<<24);};
    uint32_t W=ru32(hdr+4), H=ru32(hdr+8), gens=ru32(hdr+12);
    fclose(f);

    if (target_gen < 0 || (gens > 0 && (uint32_t)target_gen >= gens)) {
        fprintf(stderr,"Warning: gen %d may be outside recorded range [0,%u)\n",target_gen,gens);
    }

    Grid g((int)W,(int)H);
    if (!load_gol_frame(g, gol_path, target_gen)) return false;

    char namebuf[320];
    snprintf(namebuf,sizeof(namebuf),"Extracted from %s, gen %d",gol_path.c_str(),target_gen);
    return write_cells(g, out_path, namebuf);
}

// ─────────────────────────────────────────────────────────────────────────────
// Streaming .gol frame reader (sequential only, for render-from-archive)
// ─────────────────────────────────────────────────────────────────────────────

struct GolFrameReader {
    FILE*    fp       = nullptr;
    uint8_t  compression = 0;
    size_t   nbytes      = 0;
    uint8_t* prev_bits   = nullptr;
    int      cur_gen     = -1;

    bool open(const std::string& path) {
        fp = fopen(path.c_str(), "rb");
        if (!fp) { fprintf(stderr, "Cannot open '%s'\n", path.c_str()); return false; }
        uint8_t hdr[64];
        if (fread(hdr, 1, 64, fp) != 64 || hdr[0]!='G'||hdr[1]!='O'||hdr[2]!='L'||hdr[3]!='1') {
            fprintf(stderr, "Not a valid .gol file\n"); fclose(fp); fp=nullptr; return false;
        }
        auto ru32 = [](uint8_t* p){ return (uint32_t)p[0]|((uint32_t)p[1]<<8)|((uint32_t)p[2]<<16)|((uint32_t)p[3]<<24); };
        uint32_t W = ru32(hdr+4), H = ru32(hdr+8);
        nbytes = (size_t)((size_t)W*H+7)/8;
        compression = hdr[36];
        prev_bits = (uint8_t*)calloc(1, nbytes);
        if (!prev_bits) { fclose(fp); fp=nullptr; return false; }
        cur_gen = 0;
        return true;
    }

    // Read next frame. Returns false at end of file.
    // Sets *live_count and fills `bits_out` with nbytes of bit-packed row-major data.
    bool read_next(uint32_t& live_count, std::vector<uint8_t>& bits_out) {
        if (!fp) return false;
        bits_out.resize(nbytes);

        if (compression == GOL_COMPRESS_DELTA) {
            uint8_t fhdr[21];
            if (fread(fhdr, 1, 21, fp) != 21) return false;
            live_count = (uint32_t)fhdr[8]|((uint32_t)fhdr[9]<<8)|((uint32_t)fhdr[10]<<16)|((uint32_t)fhdr[11]<<24);
            uint8_t  ctype  = fhdr[12]; (void)ctype;
            uint32_t usize  = (uint32_t)fhdr[13]|((uint32_t)fhdr[14]<<8)|((uint32_t)fhdr[15]<<16)|((uint32_t)fhdr[16]<<24);
            uint32_t csize  = (uint32_t)fhdr[17]|((uint32_t)fhdr[18]<<8)|((uint32_t)fhdr[19]<<16)|((uint32_t)fhdr[20]<<24);
            std::vector<uint8_t> cdata(csize);
            if (fread(cdata.data(), 1, csize, fp) != csize) return false;

            uint8_t* sparse = nullptr;
            if (!zstd_decompress(cdata.data(), csize, &sparse, usize)) return false;
            size_t decoded_nbytes = 0;
            uint8_t* decoded = block_sparse_decode(sparse, usize, nbytes, &decoded_nbytes);
            free(sparse);
            if (!decoded) return false;

            if (cur_gen == 0) {
                memcpy(bits_out.data(), decoded, nbytes);
                memcpy(prev_bits, decoded, nbytes);
            } else {
                for (size_t i = 0; i < nbytes; i++) {
                    uint8_t v = decoded[i] ^ prev_bits[i];
                    prev_bits[i] = v;
                    bits_out[i] = v;
                }
            }
            free(decoded);
        } else {
            uint8_t fhdr[12];
            if (fread(fhdr, 1, 12, fp) != 12) return false;
            live_count = (uint32_t)fhdr[8]|((uint32_t)fhdr[9]<<8)|((uint32_t)fhdr[10]<<16)|((uint32_t)fhdr[11]<<24);
            if (fread(bits_out.data(), 1, nbytes, fp) != nbytes) return false;
        }
        cur_gen++;
        return true;
    }

    void close() { if (fp) { fclose(fp); fp=nullptr; } free(prev_bits); prev_bits=nullptr; }
    ~GolFrameReader() { close(); }
};

// ─────────────────────────────────────────────────────────────────────────────
// Renderer: grid -> RGB24 (alive=white, dead=black)
// ─────────────────────────────────────────────────────────────────────────────

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
            int alive=0;
            for (int gr=gr0; gr<gr1&&!alive; gr++) {
                const uint64_t* row=g.brow(gr);
                for (int gc=gc0; gc<gc1&&!alive; gc++)
                    if ((row[gc/64]>>(gc%64))&1) alive=1;
            }
            out_row[px*3]=alive*255; out_row[px*3+1]=alive*255; out_row[px*3+2]=alive*255;
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
    for (int i=0;i<10;i++){step_rules(A,B,8,12);sync_borders(B);std::swap(A.bits,B.bits);}
    auto t0=std::chrono::high_resolution_clock::now();
    for (int i=0;i<GENS;i++){step_rules(A,B,8,12);sync_borders(B);std::swap(A.bits,B.bits);}
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
    SET_STDOUT_BINARY();

    fprintf(stderr,"=== GoL Video Export ===\n");
    fprintf(stderr,"Grid: %dx%d  Rules: %s  Render: %dx%d @ %d fps\n",
            W,H,rules_str(rule_B,rule_S).c_str(),rW,rH,FPS);
    fprintf(stderr,"Coverage: %.2f x %.2f cells/pixel\n",(double)W/rW,(double)H/rH);
    fprintf(stderr,"Duration: %d gens = %.1fs\n\n",GENS,(double)GENS/FPS);

    Grid A(W,H),B(W,H);
    if (!state_file.empty()) {
        if (!load_state(A, state_file)) return;
    } else if (!seed_pattern.empty()) {
        if (!load_named_pattern(A, seed_pattern)) return;
    } else {
        random_fill(A, density, seed);
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
        std::swap(A.bits,B.bits);

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
// Render mode — render .gol archive to RGB24 video (stdout pipe to ffmpeg)
// ─────────────────────────────────────────────────────────────────────────────

static void run_render(const std::string& gol_path, int rW, int rH, int FPS, int THREADS) {
    omp_set_num_threads(THREADS);
    SET_STDOUT_BINARY();

    // Get header info for progress reporting
    GolFrameReader reader;
    if (!reader.open(gol_path)) return;
    size_t nbytes = reader.nbytes;
    reader.close();

    // Re-open for streaming
    GolFrameReader stream;
    if (!stream.open(gol_path)) return;

    // Age-tracking buffer: same dimensions as bit grid
    // We allocate one row at a time lazily — for huge grids, don't allocate full age array
    int grid_w, grid_h;
    {
        FILE* f = fopen(gol_path.c_str(), "rb");
        uint8_t hdr[64]; fread(hdr,1,64,f); fclose(f);
        auto ru32=[](uint8_t* p){return (uint32_t)p[0]|((uint32_t)p[1]<<8)|((uint32_t)p[2]<<16)|((uint32_t)p[3]<<24);};
        grid_w = ru32(hdr+4); grid_h = ru32(hdr+8);
    }

    // Age tracking — we update age cell-by-cell as we stream
    // Use uint8_t for age (0=dead, 1-127=age, 127+ capped)
    size_t total_cells = (size_t)grid_w * grid_h;
    uint8_t* ages = (uint8_t*)calloc(1, total_cells);
    std::vector<uint8_t> prev_row_bits;

    fprintf(stderr,"=== GoL Render from Archive ===\n");
    fprintf(stderr,"Archive: %s  Render: %dx%d @ %d fps\n", gol_path.c_str(), rW, rH, FPS);
    fprintf(stderr,"Grid:    %dx%d  Coverage: %.2f x %.2f cells/pixel\n",
            grid_w, grid_h, (double)grid_w/rW, (double)grid_h/rH);
    fprintf(stderr,"\n");

    std::vector<uint8_t> frame_buf;
    size_t frame_bytes = (size_t)rW * rH * 3;
    auto wall0 = std::chrono::high_resolution_clock::now();
    int total_frames = 0;

    std::vector<uint8_t> bits;
    uint32_t live_count;
    // Read gen 0
    if (!stream.read_next(live_count, bits)) {
        free(ages); return;
    }
    total_frames++;

    // Compute ages for gen 0: alive = age 1
    {
        long long bp = 0;
        for (int r = 0; r < grid_h; r++) {
            for (int c = 0; c < grid_w; c++) {
                int alive = (bits[bp/8] >> (bp%8)) & 1;
                ages[r * (size_t)grid_w + c] = alive ? 1 : 0;
                bp++;
            }
        }
    }

    // Build age color LUT for render-from-archive (white->yellow->orange->green->blue)
    uint8_t age_lut_r[129], age_lut_g[129], age_lut_b[129];
    age_lut_r[0]=age_lut_g[0]=age_lut_b[0]=0;
    for (int a=1; a<=128; a++) {
        float t=(a-1)/127.0f;
        if (t<0.15f)      { float s=t/0.15f; age_lut_r[a]=255;           age_lut_g[a]=255;           age_lut_b[a]=(uint8_t)(255*(1-s)); }
        else if (t<0.38f) { float s=(t-0.15f)/0.23f; age_lut_r[a]=255;  age_lut_g[a]=(uint8_t)(255*(1-s*0.45f)); age_lut_b[a]=0; }
        else if (t<0.60f) { float s=(t-0.38f)/0.22f; age_lut_r[a]=255;  age_lut_g[a]=(uint8_t)(140+60*(1-s)); age_lut_b[a]=0; }
        else if (t<0.78f) { float s=(t-0.60f)/0.18f; age_lut_r[a]=(uint8_t)(255-255*s); age_lut_g[a]=200; age_lut_b[a]=(uint8_t)(60+195*s); }
        else              { float s=(t-0.78f)/0.22f; age_lut_r[a]=0;    age_lut_g[a]=(uint8_t)(80-60*s); age_lut_b[a]=(uint8_t)(255-175*s); }
    }

    // Helper: render row range with age coloring
    auto render_row = [&](int py, const uint8_t* bits, const uint8_t* ages, uint8_t* out) {
        int gr0=(int)((double)py*grid_h/rH), gr1=(int)((double)(py+1)*grid_h/rH);
        if (gr1<=gr0) gr1=gr0+1; if (gr1>grid_h) gr1=grid_h;
        for (int px=0; px<rW; px++) {
            int gc0=(int)((double)px*grid_w/rW), gc1=(int)((double)(px+1)*grid_w/rW);
            if (gc1<=gc0) gc1=gc0+1; if (gc1>grid_w) gc1=grid_w;
            uint8_t max_age=0;
            for (int gr=gr0; gr<gr1&&max_age<128; gr++) {
                for (int gc=gc0; gc<gc1; gc++) {
                    size_t idx = (size_t)gr*grid_w+gc;
                    uint8_t a = ages[idx];
                    if (a>max_age) max_age=a;
                }
            }
            out[px*3]=age_lut_r[max_age]; out[px*3+1]=age_lut_g[max_age]; out[px*3+2]=age_lut_b[max_age];
        }
    };

    // Render gen 0
    if (rW>0 && rH>0) {
        frame_buf.resize(frame_bytes);
        for (int py=0; py<rH; py++)
            render_row(py, bits.data(), ages, frame_buf.data()+py*rW*3);
        fwrite(frame_buf.data(), 1, frame_bytes, stdout);
    }

    for (int gen = 1; ; gen++) {
        if (!stream.read_next(live_count, bits)) break;
        total_frames++;

        // Update ages: XOR with prev bits tells us births and deaths
        long long bp = 0;
        for (size_t i = 0; i < total_cells; i++) {
            int alive = (bits[bp/8] >> (bp%8)) & 1;
            if (alive) {
                ages[i] = (ages[i] >= 127) ? 127 : ages[i] + 1;
            } else {
                ages[i] = 0;
            }
            bp++;
        }

        // Render using live bits + ages
        if (rW>0 && rH>0) {
            for (int py=0; py<rH; py++)
                render_row(py, bits.data(), ages, frame_buf.data()+py*rW*3);
            fwrite(frame_buf.data(), 1, frame_bytes, stdout);
        }

        if (gen % 100 == 0 || gen % 1000 == 0 || gen == 1) fflush(stdout);

        auto now = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(now - wall0).count();
        if (elapsed >= 1.0 || gen == 1) {
            fprintf(stderr,"  Frame %5d  %.1f fps           \r",
                    gen, (double)gen / elapsed);
        }
    }

    fflush(stdout);
    double total = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - wall0).count();
    fprintf(stderr,"\nDone. %d frames in %.1fs (%.1f fps)\n", total_frames, total, total_frames/total);
    free(ages);
}

// ─────────────────────────────────────────────────────────────────────────────
// Sim mode (config-driven, stagnation detection, archive)
// ─────────────────────────────────────────────────────────────────────────────

static int run_sim(const Config& cfg) {
    omp_set_num_threads(cfg.threads);

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

    if (cfg.has_seeds()) {
        for (auto& seed : cfg.seeds) {
            bool ok;
            if (seed.is_file) ok = load_state(A, seed.name, seed.cx, seed.cy);
            else ok = load_named_pattern(A, seed.name, seed.cx, seed.cy);
            if (!ok) return 1;
        }
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
        std::swap(A.bits,B.bits);

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

    if (!cfg.final_cells_file.empty()) {
        char namebuf[320];
        snprintf(namebuf,sizeof(namebuf),"Final state at gen %d (%s)",
                 final_gen, stag_result.triggered ? "stagnated" : "max_gens reached");
        write_cells(A, cfg.final_cells_file, namebuf);
    }

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
        "  %s archive-info <file.gol>\n"
        "  %s extract <file.gol> <gen> <out.cells>\n"
        "  %s render  <file.gol> <rW> <rH> <fps> [threads]\n\n"
        "Pattern names (use directly in place of state file):\n"
        "  any *.cells file in built_in_patterns/\n\n"
        "Examples:\n"
        "  ./gol video 3840 2160 600 60 1920 1080 8 gosper-gun\n"
        "  ./gol video 1920 1080 500 60 1920 1080 8 glider B3/S23\n"
        "  ./gol bench 3840 2160 1000 8 glider\n"
        "  ./gol sim life.cfg                        # saves run.gol\n"
        "  ./gol render run.gol 1920 1080 60 | ffmpeg ...  # render archive to video\n\n"
        "Video pipe:\n"
        "  ./gol video 3840 2160 600 60 1920 1080 8 | ffmpeg \\\n"
        "    -f rawvideo -pixel_format rgb24 -video_size 1920x1080 -framerate 60 -i - \\\n"
        "    -c:v libx264 -preset veryslow -crf 10 -pix_fmt yuv420p life.mp4\n\n"
        "Sim with archive:\n"
        "  # In config: archive_file = run.gol; seed_pattern = gosper-gun\n"
        "  ./gol sim life.cfg\n"
        "  ./gol archive-info run.gol\n\n"
        "Rules: B3/S23 (Conway)  B36/S23 (HighLife)  B3/S12345 (Maze)  B2/S (Seeds)\n",
        p,p,p,p,p,p,p,p);
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
    if (mode=="extract") {
        if (argc<5) { fprintf(stderr,"Usage: %s extract <file.gol> <gen> <out.cells>\n",argv[0]); return 1; }
        bool ok = run_extract(argv[2], atoi(argv[3]), argv[4]);
        return ok ? 0 : 1;
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
    if (mode=="render") {
        if (argc<6) { usage(argv[0]); return 1; }
        int rW=atoi(argv[3]), rH=atoi(argv[4]), FPS=atoi(argv[5]);
        int threads = argc>6 ? atoi(argv[6]) : omp_get_max_threads();
        run_render(argv[2], rW, rH, FPS, threads);
        return 0;
    }

    usage(argv[0]); return 1;
}
