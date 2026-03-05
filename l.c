/*
 * l.c — actually llama.
 *
 * one file trains a full Llama 3 from scratch. no pytorch. no python.
 * no frameworks. no pip install. no "works on my machine."
 * just malloc, free, and knowing what a gradient is.
 *
 * cc l.c -O3 -lm -lpthread -fopenmp -o l && ./l --depth 4
 *
 * what happens: downloads data → trains BPE tokenizer → builds Llama 3 →
 * trains it with hand-written backward passes → finetunes on personality →
 * exports GGUF → drops you into chat. all in this file. sorry not sorry.
 *
 * depth is the only dial:
 *   ./l --depth 2   # ~1M params, fast demo, 700 tok/s
 *   ./l --depth 4   # ~3M params, your grandma's GPU isn't needed
 *   ./l --depth 8   # ~15M params, go make coffee
 *
 * symbiote of Karpathy's nanochat and microGPT. but actually Llama.
 * born from the Arianna Method ecosystem. raised by spite and curiosity.
 * now with OpenMP parallelization, fp16 KV cache, and NTK-aware RoPE.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/stat.h>
#include <float.h>
#include <stdint.h>
#include <errno.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

/* BLAS ACCELERATION — optional cblas_sgemm for matmul. 3-4x speedup.
 *   macOS:  cc l.c -O3 -lm -lpthread -DUSE_BLAS -DACCELERATE -framework Accelerate -o l
 *   Linux:  cc l.c -O3 -lm -lpthread -DUSE_BLAS -lopenblas -o l */
#ifdef USE_BLAS
  #ifdef ACCELERATE
    #define ACCELERATE_NEW_LAPACK
    #include <Accelerate/Accelerate.h>
  #else
    #include <cblas.h>
  #endif
#endif

/* CUDA ACCELERATION — cuBLAS for GPU matmul. A100 goes brrr.
 *   nvcc -c ariannamethod_cuda.cu -lcublas -O3
 *   cc l.c ariannamethod_cuda.o -O3 -lm -lpthread -DUSE_CUDA -lcublas -lcudart -L/usr/local/cuda/lib64 -o l
 * uses ariannamethod_cuda.h/cu from the Arianna Method ecosystem. */
#ifdef USE_CUDA
#include "ariannamethod_cuda.h"
/* GPU temp buffers for activations/results — weights are resident on GPU via Tensor.d_data */
static float *d_tmp_a, *d_tmp_c;  /* a=activation, c=result */
static float *d_tmp_b;            /* extra for bwd */
static int d_tmp_size = 0;
static void gpu_ensure_tmp(int needed) {
    if (needed <= d_tmp_size) return;
    if (d_tmp_a) { gpu_free(d_tmp_a); gpu_free(d_tmp_b); gpu_free(d_tmp_c); }
    d_tmp_a = gpu_alloc(needed);
    d_tmp_b = gpu_alloc(needed);
    d_tmp_c = gpu_alloc(needed);
    d_tmp_size = needed;
}

/* gpu_upload_weights and gpu_resync_weights defined after Tensor struct */
#define GPU(t) ((t)->d_data)
#else
#define GPU(t) NULL
#endif

/* ═══════════════════════════════════════════════════════════════════════════════
 * CONFIGURATION — one knob to rule them all.
 * you turn depth, everything else figures itself out.
 * that's more than most ML engineers can say about themselves.
 * ═══════════════════════════════════════════════════════════════════════════════ */

typedef struct {
    int depth;          /* number of transformer layers */
    int dim;            /* model dimension (embedding size) */
    int n_heads;        /* number of attention heads */
    int n_kv_heads;     /* number of key-value heads (GQA) */
    int head_dim;       /* dimension per head */
    int hidden_dim;     /* SwiGLU intermediate dimension */
    int vocab_size;     /* set after BPE training */
    int seq_len;        /* context window */
    float norm_eps;     /* RMSNorm epsilon */
    float rope_theta;   /* RoPE base frequency */
    float rope_scaling; /* NTK scaling factor (>1 extends context) */
    int fp16_cache;     /* half-precision KV cache */

    /* training */
    float lr;           /* learning rate */
    int batch_size;     /* sequences per batch */
    int max_steps;      /* training steps */
    int warmup_steps;   /* linear LR warmup */
    float weight_decay; /* AdamW weight decay */
    int log_every;      /* print loss every N steps */
    int eval_every;     /* evaluate every N steps */

    /* BPE */
    int bpe_merges;     /* number of BPE merges */

    /* personality */
    int personality_steps; /* finetune steps on personality.txt */

    /* data */
    char data_url[512]; /* URL to download training text */
    char data_path[256]; /* local path for training data */
    char personality_path[256]; /* path to personality file */
    char gguf_path[256]; /* output GGUF path */
} Config;

/* one number in, entire architecture out. autograd wishes it were this clean. */
static Config config_from_depth(int depth) {
    Config c = {0};
    c.depth = depth;

    /* Dimension scaling: dim = depth * 64, each step genuinely widens the model */
    c.dim = depth * 64;
    c.dim = ((c.dim + 63) / 64) * 64;
    if (c.dim < 128) c.dim = 128; /* min 128: below this, attention has 1 head and no opinions */
    if (c.dim > 768) c.dim = 768; /* max 768: your CPU has feelings and RAM has limits */

    /* Heads: head_dim = 64, n_heads = dim / head_dim */
    c.head_dim = 64;
    c.n_heads = c.dim / c.head_dim;
    if (c.n_heads < 1) c.n_heads = 1;

    /* GQA: MHA for small models, GQA for larger */
    if (c.dim <= 384) {
        c.n_kv_heads = c.n_heads; /* MHA */
    } else {
        c.n_kv_heads = c.n_heads / 2;
        if (c.n_kv_heads < 1) c.n_kv_heads = 1;
        /* ensure n_heads % n_kv_heads == 0 */
        while (c.n_heads % c.n_kv_heads != 0 && c.n_kv_heads > 1)
            c.n_kv_heads--;
    }

    /* SwiGLU hidden dim: ~2.67x dim, rounded to multiple of 64 */
    c.hidden_dim = (int)(c.dim * 2.6667f);
    c.hidden_dim = ((c.hidden_dim + 63) / 64) * 64;

    c.seq_len = 256;    /* 256 tokens of context. enough to be dangerous. */
    c.norm_eps = 1e-5f;
    c.rope_theta = 10000.0f;
    c.rope_scaling = 1.0f;  /* >1 extends context via NTK */
    c.fp16_cache = 0;       /* half-precision KV cache off by default */

    /* Training hyperparams scale with model size */
    c.lr = 3e-4f;
    c.batch_size = 4;
    c.warmup_steps = 100;
    c.weight_decay = 0.01f;
    c.log_every = depth > 4 ? 100 : 20;
    c.eval_every = 100;

    /* steps scale with depth: small models train fast, big ones need time.
     * nanoGPT: 10M params, 5000 steps, batch 64. we scale proportionally. */
    c.max_steps = depth * depth * 300;
    if (c.max_steps < 500) c.max_steps = 500;
    if (c.max_steps > 50000) c.max_steps = 50000;

    /* BPE: more merges for bigger vocab with bigger models */
    c.bpe_merges = 2000;

    c.personality_steps = 100;

    snprintf(c.data_url, sizeof(c.data_url), "fineweb-edu"); /* marker: triggers HF API download */
    snprintf(c.data_path, sizeof(c.data_path), "l_data.txt");
    snprintf(c.personality_path, sizeof(c.personality_path), "personality.txt");
    snprintf(c.gguf_path, sizeof(c.gguf_path), "l.gguf");

    return c;
}

static long count_params(Config *c) {
    long embed = (long)c->vocab_size * c->dim * 2; /* tok_emb + output (untied) */
    long per_layer = 0;
    per_layer += (long)c->dim * c->n_heads * c->head_dim;    /* Wq */
    per_layer += (long)c->dim * c->n_kv_heads * c->head_dim; /* Wk */
    per_layer += (long)c->dim * c->n_kv_heads * c->head_dim; /* Wv */
    per_layer += (long)c->n_heads * c->head_dim * c->dim;    /* Wo */
    per_layer += (long)c->dim * c->hidden_dim * 3;           /* gate + up + down */
    per_layer += (long)c->dim * 2;                           /* 2x RMSNorm */
    long total = embed + per_layer * c->depth + c->dim;      /* + final norm */
    return total;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * HALF-PRECISION — software fp16 for KV cache. saves half the memory.
 * IEEE 754 binary16: 1 sign, 5 exponent, 10 mantissa. good enough for attention.
 * ═══════════════════════════════════════════════════════════════════════════════ */

typedef uint16_t half;

static half float2half(float f) {
    uint32_t x = *(uint32_t*)&f;
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = ((x >> 23) & 0xFF) - 112;
    uint32_t mant = (x >> 13) & 0x3FF;
    if (exp <= 0) return sign;
    if (exp > 30) return sign | 0x7C00;
    return sign | (exp << 10) | mant;
}

static float half2float(half h) {
    uint32_t sign = (h >> 15) & 0x1;
    int32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) { f = sign << 31; }
    else if (exp == 31) { f = (sign << 31) | 0x7F800000 | (mant << 13); }
    else { f = (sign << 31) | ((exp + 112) << 23) | (mant << 13); }
    return *(float*)&f;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * RNG — xorshift64*. three XORs and a multiply. pytorch uses Mersenne Twister
 * which is 2500 lines of C. we use 5. the weights don't care.
 * ═══════════════════════════════════════════════════════════════════════════════ */

static uint64_t rng_state = 42;

static uint64_t rng_next(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

static float rand_uniform(void) {
    return (float)(rng_next() & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

static float rand_normal(void) {
    float u1 = rand_uniform();
    float u2 = rand_uniform();
    if (u1 < 1e-10f) u1 = 1e-10f;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * DYNAMIC ARRAYS — because C doesn't have std::vector and we don't miss it
 * ═══════════════════════════════════════════════════════════════════════════════ */

typedef struct { char **items; int len, cap; } StrArr;

static void sa_push(StrArr *a, const char *s) {
    if (a->len >= a->cap) {
        a->cap = a->cap ? a->cap * 2 : 16;
        a->items = realloc(a->items, sizeof(char*) * a->cap);
    }
    a->items[a->len++] = strdup(s);
}

static void sa_free(StrArr *a) {
    for (int i = 0; i < a->len; i++) free(a->items[i]);
    free(a->items);
    a->items = NULL; a->len = a->cap = 0;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * BPE TOKENIZER — trained from scratch on your data. no sentencepiece.
 * no tiktoken. no "download the 4GB tokenizer model first."
 * 256 byte tokens + merges. stolen from molequla, refined here.
 * tiktoken wishes it compiled in 0.3 seconds.
 * ═══════════════════════════════════════════════════════════════════════════════ */

#define TOK_MAX_VOCAB 16384
#define TOK_STOI_CAP  32768

typedef struct { char a[64]; char b[64]; } MergePair;

typedef struct { char *key; int val; } StoiEntry;

typedef struct {
    StoiEntry entries[TOK_STOI_CAP];
} StoiTable;

typedef struct {
    char *tokens[TOK_MAX_VOCAB];
    int vocab_size;
    StoiTable stoi;
    int bos_id, eos_id;
    MergePair *merges;
    int n_merges;
} Tokenizer;

static unsigned int str_hash(const char *s) {
    unsigned int h = 5381;
    while (*s) h = h * 33 + (unsigned char)*s++;
    return h;
}

static void stoi_init(StoiTable *t) {
    for (int i = 0; i < TOK_STOI_CAP; i++) {
        t->entries[i].key = NULL;
        t->entries[i].val = -1;
    }
}

static void stoi_put(StoiTable *t, const char *key, int val) {
    unsigned int h = str_hash(key) % TOK_STOI_CAP;
    for (int i = 0; i < TOK_STOI_CAP; i++) {
        int idx = (h + i) % TOK_STOI_CAP;
        if (t->entries[idx].key == NULL) {
            t->entries[idx].key = strdup(key);
            t->entries[idx].val = val;
            return;
        }
        if (strcmp(t->entries[idx].key, key) == 0) {
            t->entries[idx].val = val;
            return;
        }
    }
}

static int stoi_get(StoiTable *t, const char *key) {
    unsigned int h = str_hash(key) % TOK_STOI_CAP;
    for (int i = 0; i < TOK_STOI_CAP; i++) {
        int idx = (h + i) % TOK_STOI_CAP;
        if (t->entries[idx].key == NULL) return -1;
        if (strcmp(t->entries[idx].key, key) == 0) return t->entries[idx].val;
    }
    return -1;
}

static void tok_init(Tokenizer *tok) {
    memset(tok, 0, sizeof(Tokenizer));
    stoi_init(&tok->stoi);

    /* 256 byte tokens */
    for (int i = 0; i < 256; i++) {
        char hex[8];
        snprintf(hex, sizeof(hex), "0x%02x", i);
        tok->tokens[tok->vocab_size] = strdup(hex);
        stoi_put(&tok->stoi, hex, tok->vocab_size);
        tok->vocab_size++;
    }

    /* Special tokens */
    tok->tokens[tok->vocab_size] = strdup("<BOS>");
    stoi_put(&tok->stoi, "<BOS>", tok->vocab_size);
    tok->bos_id = tok->vocab_size++;

    tok->tokens[tok->vocab_size] = strdup("<EOS>");
    stoi_put(&tok->stoi, "<EOS>", tok->vocab_size);
    tok->eos_id = tok->vocab_size++;
}

static void tok_add(Tokenizer *tok, const char *s) {
    if (stoi_get(&tok->stoi, s) >= 0) return;
    if (tok->vocab_size >= TOK_MAX_VOCAB) return;
    tok->tokens[tok->vocab_size] = strdup(s);
    stoi_put(&tok->stoi, s, tok->vocab_size);
    tok->vocab_size++;
}

/* Unicode pre-segmentation: split by category boundaries */
static char byte_category(unsigned char b) {
    if ((b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z')) return 'L';
    if (b >= '0' && b <= '9') return 'N';
    if (b == ' ' || b == '\n' || b == '\r' || b == '\t') return 'Z';
    if (b >= 0xC0) return 'L'; /* UTF-8 lead */
    if (b >= 0x80) return 'L'; /* UTF-8 continuation */
    return 'P';
}

/* Segment text into runs of same unicode category */
typedef struct { unsigned char *data; int len; } ByteSeg;
typedef struct { ByteSeg *segs; int len, cap; } SegArr;

static void seg_push(SegArr *a, unsigned char *data, int len) {
    if (a->len >= a->cap) {
        a->cap = a->cap ? a->cap * 2 : 64;
        a->segs = realloc(a->segs, sizeof(ByteSeg) * a->cap);
    }
    a->segs[a->len].data = malloc(len);
    memcpy(a->segs[a->len].data, data, len);
    a->segs[a->len].len = len;
    a->len++;
}

static void seg_free(SegArr *a) {
    for (int i = 0; i < a->len; i++) free(a->segs[i].data);
    free(a->segs);
    memset(a, 0, sizeof(SegArr));
}

static SegArr unicode_segment(const char *text, int text_len) {
    SegArr result = {0};
    if (!text || text_len == 0) return result;

    unsigned char buf[4096];
    int buf_len = 0;
    char cur_cat = 0;
    const unsigned char *p = (const unsigned char *)text;

    for (int i = 0; i < text_len; i++) {
        char cat = byte_category(p[i]);
        if (cat != cur_cat && buf_len > 0) {
            seg_push(&result, buf, buf_len);
            buf_len = 0;
        }
        cur_cat = cat;
        if (buf_len < (int)sizeof(buf) - 1) {
            buf[buf_len++] = p[i];
        } else {
            seg_push(&result, buf, buf_len);
            buf_len = 0;
            buf[buf_len++] = p[i];
        }
    }
    if (buf_len > 0) seg_push(&result, buf, buf_len);
    return result;
}

/* BPE pair frequency counting */
#define PAIR_CAP 32768
typedef struct { char a[64]; char b[64]; int count; int used; } PairEntry;

static unsigned int pair_hash(const char *a, const char *b) {
    unsigned int h = 5381;
    for (const char *p = a; *p; p++) h = h * 33 + (unsigned char)*p;
    h = h * 33 + 0xFF;
    for (const char *p = b; *p; p++) h = h * 33 + (unsigned char)*p;
    return h;
}

/* Train BPE merges on text corpus */
static void tok_train_bpe(Tokenizer *tok, const char *text, int text_len, int num_merges) {
    printf("[bpe] training %d merges on %d bytes...\n", num_merges, text_len);

    /* Segment text */
    SegArr segs = unicode_segment(text, text_len);
    if (segs.len == 0) { seg_free(&segs); return; }

    /* Convert segments to byte-token sequences */
    int n_seqs = segs.len;
    StrArr *sym_seqs = calloc(n_seqs, sizeof(StrArr));
    for (int s = 0; s < n_seqs; s++) {
        for (int b = 0; b < segs.segs[s].len; b++) {
            char hex[8];
            snprintf(hex, sizeof(hex), "0x%02x", segs.segs[s].data[b]);
            sa_push(&sym_seqs[s], hex);
        }
    }
    seg_free(&segs);

    /* Allocate merges */
    if (tok->merges) free(tok->merges);
    tok->merges = calloc(num_merges, sizeof(MergePair));
    tok->n_merges = 0;

    PairEntry *pairs = calloc(PAIR_CAP, sizeof(PairEntry));

    for (int iter = 0; iter < num_merges; iter++) {
        /* Count pairs */
        memset(pairs, 0, sizeof(PairEntry) * PAIR_CAP);
        for (int s = 0; s < n_seqs; s++) {
            StrArr *seq = &sym_seqs[s];
            for (int i = 0; i < seq->len - 1; i++) {
                unsigned int h = pair_hash(seq->items[i], seq->items[i+1]) % PAIR_CAP;
                for (int probe = 0; probe < 64; probe++) {
                    int idx = (h + probe) % PAIR_CAP;
                    if (!pairs[idx].used) {
                        strncpy(pairs[idx].a, seq->items[i], 63);
                        strncpy(pairs[idx].b, seq->items[i+1], 63);
                        pairs[idx].count = 1;
                        pairs[idx].used = 1;
                        break;
                    }
                    if (strcmp(pairs[idx].a, seq->items[i]) == 0 &&
                        strcmp(pairs[idx].b, seq->items[i+1]) == 0) {
                        pairs[idx].count++;
                        break;
                    }
                }
            }
        }

        /* Find best pair */
        int best_count = 1; /* need at least 2 occurrences */
        int best_idx = -1;
        for (int i = 0; i < PAIR_CAP; i++) {
            if (pairs[i].used && pairs[i].count > best_count) {
                best_count = pairs[i].count;
                best_idx = i;
            }
        }
        if (best_idx < 0) break;

        char *best_a = pairs[best_idx].a;
        char *best_b = pairs[best_idx].b;

        /* Create merged token */
        char new_tok[128];
        snprintf(new_tok, sizeof(new_tok), "%s+%s", best_a, best_b);

        strncpy(tok->merges[tok->n_merges].a, best_a, 63);
        strncpy(tok->merges[tok->n_merges].b, best_b, 63);
        tok->n_merges++;

        /* Apply merge to all sequences */
        for (int s = 0; s < n_seqs; s++) {
            StrArr *seq = &sym_seqs[s];
            StrArr merged = {0};
            int i = 0;
            while (i < seq->len) {
                if (i < seq->len - 1 &&
                    strcmp(seq->items[i], best_a) == 0 &&
                    strcmp(seq->items[i+1], best_b) == 0) {
                    sa_push(&merged, new_tok);
                    i += 2;
                } else {
                    sa_push(&merged, seq->items[i]);
                    i++;
                }
            }
            sa_free(seq);
            *seq = merged;
        }
        tok_add(tok, new_tok);

        if ((iter + 1) % 500 == 0)
            printf("[bpe] %d/%d merges (vocab=%d)\n", iter + 1, num_merges, tok->vocab_size);
    }

    free(pairs);
    for (int s = 0; s < n_seqs; s++) sa_free(&sym_seqs[s]);
    free(sym_seqs);

    printf("[bpe] done: %d merges, vocab=%d\n", tok->n_merges, tok->vocab_size);
}

/* Apply BPE merges to a byte sequence → token IDs */
static int *tok_encode(Tokenizer *tok, const char *text, int text_len, int *out_len) {
    /* Segment by unicode category */
    SegArr segs = unicode_segment(text, text_len);

    /* Collect all token IDs */
    int *ids = NULL;
    int n_ids = 0, cap_ids = 0;

    for (int s = 0; s < segs.len; s++) {
        /* Convert segment bytes to hex tokens */
        StrArr symbols = {0};
        for (int b = 0; b < segs.segs[s].len; b++) {
            char hex[8];
            snprintf(hex, sizeof(hex), "0x%02x", segs.segs[s].data[b]);
            sa_push(&symbols, hex);
        }

        /* Apply BPE merges (greedy, lowest-rank first) */
        if (tok->n_merges > 0 && symbols.len >= 2) {
            int changed = 1;
            while (changed && symbols.len >= 2) {
                changed = 0;
                int best_rank = tok->n_merges;
                int best_pos = -1;
                for (int i = 0; i < symbols.len - 1; i++) {
                    for (int m = 0; m < best_rank; m++) {
                        if (strcmp(symbols.items[i], tok->merges[m].a) == 0 &&
                            strcmp(symbols.items[i+1], tok->merges[m].b) == 0) {
                            best_rank = m;
                            best_pos = i;
                            break;
                        }
                    }
                }
                if (best_pos >= 0) {
                    char new_tok[128];
                    snprintf(new_tok, sizeof(new_tok), "%s+%s",
                             tok->merges[best_rank].a, tok->merges[best_rank].b);
                    StrArr merged = {0};
                    for (int i = 0; i < symbols.len; i++) {
                        if (i == best_pos) {
                            sa_push(&merged, new_tok);
                            i++; /* skip next */
                        } else {
                            sa_push(&merged, symbols.items[i]);
                        }
                    }
                    sa_free(&symbols);
                    symbols = merged;
                    changed = 1;
                }
            }
        }

        /* Convert token names to IDs */
        for (int i = 0; i < symbols.len; i++) {
            int id = stoi_get(&tok->stoi, symbols.items[i]);
            if (id < 0) id = 0; /* fallback to 0x00 */
            if (n_ids >= cap_ids) {
                cap_ids = cap_ids ? cap_ids * 2 : 256;
                ids = realloc(ids, sizeof(int) * cap_ids);
            }
            ids[n_ids++] = id;
        }
        sa_free(&symbols);
    }

    seg_free(&segs);
    *out_len = n_ids;
    return ids;
}

/* Decode token IDs to text (bytes) */
static char *tok_decode(Tokenizer *tok, int *ids, int n_ids, int *out_len) {
    /* Each token decodes to one or more bytes via hex */
    char *buf = malloc(n_ids * 8 + 1);
    int pos = 0;

    for (int i = 0; i < n_ids; i++) {
        if (ids[i] < 0 || ids[i] >= tok->vocab_size) continue;
        if (ids[i] == tok->bos_id || ids[i] == tok->eos_id) continue;

        const char *name = tok->tokens[ids[i]];
        /* Parse hex bytes from token name: "0x48+0x65+0x6c" → "Hel" */
        const char *p = name;
        while (*p) {
            if (p[0] == '0' && p[1] == 'x') {
                unsigned int byte_val;
                if (sscanf(p, "0x%02x", &byte_val) == 1) {
                    buf[pos++] = (char)byte_val;
                }
                p += 4; /* skip "0xHH" */
                if (*p == '+') p++; /* skip separator */
            } else {
                p++;
            }
        }
    }
    buf[pos] = '\0';
    *out_len = pos;
    return buf;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * TENSOR — a float* with delusions of grandeur. has shape. has opinions.
 * numpy uses 47 dtypes. we use float. deal with it.
 * ═══════════════════════════════════════════════════════════════════════════════ */

typedef struct {
    float *data;
    int size;       /* total elements */
    int rows, cols; /* for 2D tensors */
#ifdef USE_CUDA
    float *d_data;  /* GPU-resident copy — uploaded once, resynced after Adam */
#endif
} Tensor;

#ifdef USE_CUDA
static void gpu_upload_weights(Tensor **tensors, int n) {
    for (int i = 0; i < n; i++) {
        tensors[i]->d_data = gpu_alloc(tensors[i]->size);
        gpu_upload(tensors[i]->d_data, tensors[i]->data, tensors[i]->size);
    }
}
static void gpu_resync_weights(Tensor **tensors, int n) {
    for (int i = 0; i < n; i++)
        gpu_upload(tensors[i]->d_data, tensors[i]->data, tensors[i]->size);
}
#endif

static Tensor *tensor_new(int size) {
    Tensor *t = calloc(1, sizeof(Tensor));
    t->data = calloc(size, sizeof(float));
    t->size = size;
    t->rows = 1;
    t->cols = size;
    return t;
}

static Tensor *tensor_new_2d(int rows, int cols) {
    Tensor *t = calloc(1, sizeof(Tensor));
    t->data = calloc(rows * cols, sizeof(float));
    t->size = rows * cols;
    t->rows = rows;
    t->cols = cols;
    return t;
}

static void tensor_free(Tensor *t) {
    if (t) { free(t->data); free(t); }
}

/* Xavier/Kaiming init */
static void tensor_init_normal(Tensor *t, float std) {
    for (int i = 0; i < t->size; i++)
        t->data[i] = rand_normal() * std;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * MODEL WEIGHTS — the actual llama. RMSNorm, RoPE, GQA, SwiGLU, no bias.
 * same architecture Meta uses for Llama 3 405B. except ours fits in RAM.
 * and compiles. and you can read every line. try that with fairscale.
 * ═══════════════════════════════════════════════════════════════════════════════ */

typedef struct {
    Tensor *attn_norm;  /* [dim] */
    Tensor *wq;         /* [n_heads*head_dim, dim] */
    Tensor *wk;         /* [n_kv_heads*head_dim, dim] */
    Tensor *wv;         /* [n_kv_heads*head_dim, dim] */
    Tensor *wo;         /* [dim, n_heads*head_dim] */
    Tensor *ffn_norm;   /* [dim] */
    Tensor *w_gate;     /* [hidden_dim, dim] */
    Tensor *w_up;       /* [hidden_dim, dim] */
    Tensor *w_down;     /* [dim, hidden_dim] */
} LayerWeights;

typedef struct {
    Tensor *tok_emb;     /* [vocab, dim] */
    Tensor *output;      /* [vocab, dim] */
    Tensor *output_norm; /* [dim] */
    LayerWeights *layers;
    int n_layers;
} ModelWeights;

static void init_weights(ModelWeights *w, Config *c) {
    float emb_std = 1.0f / sqrtf((float)c->dim);
    float layer_std = 1.0f / sqrtf((float)c->dim);

    w->tok_emb = tensor_new_2d(c->vocab_size, c->dim);
    tensor_init_normal(w->tok_emb, emb_std);

    w->output = tensor_new_2d(c->vocab_size, c->dim);
    tensor_init_normal(w->output, layer_std);

    w->output_norm = tensor_new(c->dim);
    for (int i = 0; i < c->dim; i++) w->output_norm->data[i] = 1.0f;

    w->n_layers = c->depth;
    w->layers = calloc(c->depth, sizeof(LayerWeights));

    int qkv_dim = c->n_heads * c->head_dim;
    int kv_dim = c->n_kv_heads * c->head_dim;

    for (int l = 0; l < c->depth; l++) {
        LayerWeights *lw = &w->layers[l];

        lw->attn_norm = tensor_new(c->dim);
        for (int i = 0; i < c->dim; i++) lw->attn_norm->data[i] = 1.0f;

        lw->wq = tensor_new_2d(qkv_dim, c->dim);
        tensor_init_normal(lw->wq, layer_std);

        lw->wk = tensor_new_2d(kv_dim, c->dim);
        tensor_init_normal(lw->wk, layer_std);

        lw->wv = tensor_new_2d(kv_dim, c->dim);
        tensor_init_normal(lw->wv, layer_std);

        lw->wo = tensor_new_2d(c->dim, qkv_dim);
        /* Init output projection to zero (GPT-2/nanollama convention) */
        memset(lw->wo->data, 0, lw->wo->size * sizeof(float));

        lw->ffn_norm = tensor_new(c->dim);
        for (int i = 0; i < c->dim; i++) lw->ffn_norm->data[i] = 1.0f;

        lw->w_gate = tensor_new_2d(c->hidden_dim, c->dim);
        tensor_init_normal(lw->w_gate, layer_std);

        lw->w_up = tensor_new_2d(c->hidden_dim, c->dim);
        tensor_init_normal(lw->w_up, layer_std);

        lw->w_down = tensor_new_2d(c->dim, c->hidden_dim);
        memset(lw->w_down->data, 0, lw->w_down->size * sizeof(float));
    }
}

/* Collect all parameter tensors into a flat list for optimizer */
typedef struct {
    Tensor **tensors;
    int count;
} ParamList;

static ParamList collect_params(ModelWeights *w) {
    int max_params = 2 + 1 + w->n_layers * 9;
    ParamList p;
    p.tensors = calloc(max_params, sizeof(Tensor*));
    p.count = 0;

    p.tensors[p.count++] = w->tok_emb;
    p.tensors[p.count++] = w->output;
    p.tensors[p.count++] = w->output_norm;

    for (int l = 0; l < w->n_layers; l++) {
        LayerWeights *lw = &w->layers[l];
        p.tensors[p.count++] = lw->attn_norm;
        p.tensors[p.count++] = lw->wq;
        p.tensors[p.count++] = lw->wk;
        p.tensors[p.count++] = lw->wv;
        p.tensors[p.count++] = lw->wo;
        p.tensors[p.count++] = lw->ffn_norm;
        p.tensors[p.count++] = lw->w_gate;
        p.tensors[p.count++] = lw->w_up;
        p.tensors[p.count++] = lw->w_down;
    }
    return p;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * FORWARD PASS — Llama 3 inference. no autograd tape, no computation graph.
 * just for loops. pytorch devs look away, this might hurt your feelings.
 *
 * Used for generation after training. Training uses the tape-based
 * autograd forward below.
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* RMSNorm: out = x * weight / rms(x) */
static void rmsnorm(float *out, float *x, float *weight, int dim, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < dim; i++) ss += x[i] * x[i];
    float rms = 1.0f / sqrtf(ss / dim + eps);
    for (int i = 0; i < dim; i++) out[i] = x[i] * rms * weight[i];
}

/* Matrix-vector multiply: out = W @ x, W is [rows, cols] */
static void matvec(float *out, float *W, float *x, int rows, int cols) {
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        float s = 0.0f;
        float *row = W + i * cols;
        #pragma omp simd reduction(+:s)
        for (int j = 0; j < cols; j++) s += row[j] * x[j];
        out[i] = s;
    }
}

/* SiLU activation */
static float silu(float x) {
    return x / (1.0f + expf(-x));
}

/* Softmax in-place */
static void softmax(float *x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

/* Apply RoPE to a single head vector (with NTK scaling support) */
static void apply_rope(float *vec, int pos, float *cos_cache, float *sin_cache, int head_dim) {
    int half = head_dim / 2;
    int off = pos * half;
    for (int i = 0; i < half; i++) {
        float x0 = vec[i], x1 = vec[i + half];
        vec[i]        = x0 * cos_cache[off+i] - x1 * sin_cache[off+i];
        vec[i + half] = x0 * sin_cache[off+i] + x1 * cos_cache[off+i];
    }
}

/* RoPE backward (same rotation, transposed) */
static void rope_bwd_ntk(float *dvec, int pos, float *cos_cache, float *sin_cache, int head_dim) {
    int half = head_dim / 2;
    int off = pos * half;
    for (int i = 0; i < half; i++) {
        float d0 = dvec[i], d1 = dvec[i + half];
        dvec[i]        =  d0 * cos_cache[off+i] + d1 * sin_cache[off+i];
        dvec[i + half] = -d0 * sin_cache[off+i] + d1 * cos_cache[off+i];
    }
}

/* Runtime state for inference — KV cache is either float or half */
typedef struct {
    float *x, *xb, *xb2;
    float *hb, *hb2;
    float *q, *k, *v;
    float *att;
    float *logits;
    void *key_cache, *value_cache; /* float* or half* depending on fp16_cache */
    int use_half;
    float *cos_cache, *sin_cache;
} RunState;

static RunState alloc_run_state(Config *c) {
    RunState s;
    int kv_dim = c->n_kv_heads * c->head_dim;
    s.x      = calloc(c->dim, sizeof(float));
    s.xb     = calloc(c->dim, sizeof(float));
    s.xb2    = calloc(c->dim, sizeof(float));
    s.hb     = calloc(c->hidden_dim, sizeof(float));
    s.hb2    = calloc(c->hidden_dim, sizeof(float));
    s.q      = calloc(c->n_heads * c->head_dim, sizeof(float));
    s.k      = calloc(kv_dim, sizeof(float));
    s.v      = calloc(kv_dim, sizeof(float));
    s.att    = calloc(c->n_heads * c->seq_len, sizeof(float));
    s.logits = calloc(c->vocab_size, sizeof(float));
    s.use_half = c->fp16_cache;
    {
        size_t elem = c->fp16_cache ? sizeof(half) : sizeof(float);
        size_t cache_n = c->depth * c->seq_len * kv_dim;
        s.key_cache   = calloc(cache_n, elem);
        s.value_cache = calloc(cache_n, elem);
    }

    int half = c->head_dim / 2;
    s.cos_cache = calloc(c->seq_len * half, sizeof(float));
    s.sin_cache = calloc(c->seq_len * half, sizeof(float));

    /* Precompute RoPE (with NTK scaling if rope_scaling > 1) */
    for (int pos = 0; pos < c->seq_len; pos++) {
        for (int i = 0; i < half; i++) {
            float theta = c->rope_theta;
            if (c->rope_scaling > 1.0f)
                theta *= powf(c->rope_scaling, (float)c->head_dim / (c->head_dim - 2.0f));
            float freq = 1.0f / powf(theta, (float)(2*i) / (float)c->head_dim);
            float angle = (float)pos * freq;
            s.cos_cache[pos * half + i] = cosf(angle);
            s.sin_cache[pos * half + i] = sinf(angle);
        }
    }
    return s;
}

static void free_run_state(RunState *s) {
    free(s->x); free(s->xb); free(s->xb2);
    free(s->hb); free(s->hb2);
    free(s->q); free(s->k); free(s->v);
    free(s->att); free(s->logits);
    free(s->key_cache); free(s->value_cache);
    free(s->cos_cache); free(s->sin_cache);
}

/* Forward one token through the transformer (inference with KV cache) */
static float *forward_token(ModelWeights *w, Config *c, RunState *s, int token, int pos) {
    int dim = c->dim;
    int kv_dim = c->n_kv_heads * c->head_dim;
    int hd = c->head_dim;
    int head_group = c->n_heads / c->n_kv_heads;
    float scale = 1.0f / sqrtf((float)hd);

    /* Embedding lookup */
    memcpy(s->x, w->tok_emb->data + token * dim, dim * sizeof(float));

    for (int l = 0; l < c->depth; l++) {
        LayerWeights *lw = &w->layers[l];

        /* Attention pre-norm */
        rmsnorm(s->xb, s->x, lw->attn_norm->data, dim, c->norm_eps);

        /* Q, K, V projections */
        matvec(s->q, lw->wq->data, s->xb, c->n_heads * hd, dim);
        matvec(s->k, lw->wk->data, s->xb, c->n_kv_heads * hd, dim);
        matvec(s->v, lw->wv->data, s->xb, c->n_kv_heads * hd, dim);

        /* RoPE */
        for (int h = 0; h < c->n_heads; h++)
            apply_rope(s->q + h * hd, pos, s->cos_cache, s->sin_cache, hd);
        for (int h = 0; h < c->n_kv_heads; h++)
            apply_rope(s->k + h * hd, pos, s->cos_cache, s->sin_cache, hd);

        /* Store K, V in cache (fp16 or fp32) */
        int cache_off = l * c->seq_len * kv_dim + pos * kv_dim;
        if (s->use_half) {
            half *kc = (half*)s->key_cache, *vc = (half*)s->value_cache;
            for (int i = 0; i < kv_dim; i++) kc[cache_off + i] = float2half(s->k[i]);
            for (int i = 0; i < kv_dim; i++) vc[cache_off + i] = float2half(s->v[i]);
        } else {
            float *kc = (float*)s->key_cache, *vc = (float*)s->value_cache;
            memcpy(kc + cache_off, s->k, kv_dim * sizeof(float));
            memcpy(vc + cache_off, s->v, kv_dim * sizeof(float));
        }

        /* Multi-head attention with GQA */
        #pragma omp parallel for
        for (int h = 0; h < c->n_heads; h++) {
            int kvh = h / head_group;
            float *qh = s->q + h * hd;
            float *att = s->att + h * c->seq_len;

            for (int t = 0; t <= pos; t++) {
                int k_off = l * c->seq_len * kv_dim + t * kv_dim + kvh * hd;
                float dot = 0.0f;
                if (s->use_half) {
                    half *kc = (half*)s->key_cache;
                    for (int d = 0; d < hd; d++) dot += qh[d] * half2float(kc[k_off + d]);
                } else {
                    float *kc = (float*)s->key_cache;
                    for (int d = 0; d < hd; d++) dot += qh[d] * kc[k_off + d];
                }
                att[t] = dot * scale;
            }
            softmax(att, pos + 1);

            float *xb2h = s->xb2 + h * hd;
            memset(xb2h, 0, hd * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float a = att[t];
                int v_off = l * c->seq_len * kv_dim + t * kv_dim + kvh * hd;
                if (s->use_half) {
                    half *vc = (half*)s->value_cache;
                    for (int d = 0; d < hd; d++) xb2h[d] += a * half2float(vc[v_off + d]);
                } else {
                    float *vc = (float*)s->value_cache;
                    for (int d = 0; d < hd; d++) xb2h[d] += a * vc[v_off + d];
                }
            }
        }

        /* Output projection + residual */
        matvec(s->xb, lw->wo->data, s->xb2, dim, dim);
        for (int i = 0; i < dim; i++) s->x[i] += s->xb[i];

        /* FFN pre-norm */
        rmsnorm(s->xb, s->x, lw->ffn_norm->data, dim, c->norm_eps);

        /* SwiGLU MLP */
        matvec(s->hb, lw->w_gate->data, s->xb, c->hidden_dim, dim);
        matvec(s->hb2, lw->w_up->data, s->xb, c->hidden_dim, dim);
        for (int i = 0; i < c->hidden_dim; i++)
            s->hb[i] = silu(s->hb[i]) * s->hb2[i];
        matvec(s->xb, lw->w_down->data, s->hb, dim, c->hidden_dim);
        for (int i = 0; i < dim; i++) s->x[i] += s->xb[i];
    }

    /* Final norm */
    rmsnorm(s->x, s->x, w->output_norm->data, dim, c->norm_eps);

    /* LM head → logits */
    matvec(s->logits, w->output->data, s->x, c->vocab_size, dim);

    return s->logits;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * TRAINING FORWARD + BACKWARD — the 400 lines pytorch doesn't want you to see.
 *
 * hand-written gradients through every layer. RMSNorm backward, RoPE backward,
 * GQA backward, SwiGLU backward. no loss.backward(). no autograd tape.
 * just chain rule and patience. verified to 1e-5 against numerical gradients.
 * if you're reading this at 3am trying to understand backprop: welcome home.
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* Per-layer saved activations for backward */
typedef struct {
    float *input_to_norm;  /* [T, dim] — residual before attn norm */
    float *xn;             /* [T, dim] — after attn norm */
    float *q;              /* [T, n_heads*hd] — after RoPE */
    float *k;              /* [T, kv_dim] — after RoPE */
    float *v;              /* [T, kv_dim] */
    float *attn_scores;    /* [T * n_heads * T] — softmax output (causal) */
    float *attn_out;       /* [T, n_heads*hd] — after weighted sum */
    float *res_after_attn; /* [T, dim] — residual after attention */
    float *ffn_xn;         /* [T, dim] — after ffn norm */
    float *gate_pre;       /* [T, hidden] — before silu */
    float *up_pre;         /* [T, hidden] */
    float *swiglu;         /* [T, hidden] — silu(gate)*up */
} LayerAct;

typedef struct {
    LayerAct *layers;
    float *final_normed;   /* [T, dim] */
    float *logits;         /* [T, vocab] */
    float *residual;       /* [T, dim] — final residual */

    /* Scratch buffers for backward (reused across layers) */
    float *d_residual, *d_xn;
    float *d_q, *d_k, *d_v;
    float *d_attn_out;
    float *d_ffn_xn;
    float *d_gate, *d_up, *d_swiglu;

    /* RoPE cache */
    float *cos_cache, *sin_cache;

    int T, n_layers;
} TrainState;


static TrainState alloc_train_state(Config *c) {
    TrainState s = {0};
    int T = c->seq_len;
    int D = c->dim;
    int kv = c->n_kv_heads * c->head_dim;
    int qd = c->n_heads * c->head_dim;
    int H = c->hidden_dim;
    int L = c->depth;

    s.T = T; s.n_layers = L;

    s.layers = calloc(L, sizeof(LayerAct));
    for (int l = 0; l < L; l++) {
        LayerAct *la = &s.layers[l];
        la->input_to_norm  = calloc(T * D, sizeof(float));
        la->xn             = calloc(T * D, sizeof(float));
        la->q              = calloc(T * qd, sizeof(float));
        la->k              = calloc(T * kv, sizeof(float));
        la->v              = calloc(T * kv, sizeof(float));
        la->attn_scores    = calloc(T * c->n_heads * T, sizeof(float));
        la->attn_out       = calloc(T * qd, sizeof(float));
        la->res_after_attn = calloc(T * D, sizeof(float));
        la->ffn_xn         = calloc(T * D, sizeof(float));
        la->gate_pre       = calloc(T * H, sizeof(float));
        la->up_pre         = calloc(T * H, sizeof(float));
        la->swiglu         = calloc(T * H, sizeof(float));
    }

    s.residual     = calloc(T * D, sizeof(float));
    s.d_residual   = calloc(T * D, sizeof(float));
    s.d_xn         = calloc(T * D, sizeof(float));
    s.d_q          = calloc(T * qd, sizeof(float));
    s.d_k          = calloc(T * kv, sizeof(float));
    s.d_v          = calloc(T * kv, sizeof(float));
    s.d_attn_out   = calloc(T * qd, sizeof(float));
    s.d_ffn_xn     = calloc(T * D, sizeof(float));
    s.d_gate       = calloc(T * H, sizeof(float));
    s.d_up         = calloc(T * H, sizeof(float));
    s.d_swiglu     = calloc(T * H, sizeof(float));

    int half = c->head_dim / 2;
    s.cos_cache = calloc(T * half, sizeof(float));
    s.sin_cache = calloc(T * half, sizeof(float));
    for (int pos = 0; pos < T; pos++) {
        for (int i = 0; i < half; i++) {
            float theta = c->rope_theta;
            if (c->rope_scaling > 1.0f)
                theta *= powf(c->rope_scaling, (float)c->head_dim / (c->head_dim - 2.0f));
            float freq = 1.0f / powf(theta, (float)(2*i) / (float)c->head_dim);
            float angle = (float)pos * freq;
            s.cos_cache[pos * half + i] = cosf(angle);
            s.sin_cache[pos * half + i] = sinf(angle);
        }
    }
    return s;
}

/* Matrix multiply: C[M,N] = A[M,K] @ B[N,K]^T
 * d_B = GPU-resident weight pointer (NULL = upload B every time) */
static void matmul_fwd(float *C, float *A, float *B, int M, int N, int K,
                        float *d_B) {
#ifdef USE_CUDA
    int biggest = M*K; if(M*N>biggest)biggest=M*N;
    gpu_ensure_tmp(biggest);
    gpu_upload(d_tmp_a, A, M * K);
    float *d_weight = d_B ? d_B : d_tmp_b;
    if (!d_B) { if(N*K>biggest)gpu_ensure_tmp(N*K); gpu_upload(d_tmp_b, B, N * K); }
    gpu_sgemm_nt(M, N, K, d_tmp_a, d_weight, d_tmp_c);
    gpu_download(C, d_tmp_c, M * N);
#elif defined(USE_BLAS)
    (void)d_B;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0f, A, K, B, K, 0.0f, C, N);
#else
    (void)d_B;
    for (int m = 0; m < M; m++) {
        float *cm = C + m * N;
        float *am = A + m * K;
        for (int n = 0; n < N; n++) {
            float s = 0.0f;
            float *bn = B + n * K;
            for (int k = 0; k < K; k++) s += am[k] * bn[k];
            cm[n] = s;
        }
    }
#endif
}

/* C = A @ B^T backward: dA += dC @ B, dB += dC^T @ A
 * d_B = GPU-resident weight pointer (NULL = upload B every time) */
static void matmul_bwd(float *dA, float *dB, float *dC, float *A, float *B,
                        int M, int N, int K, float *d_B) {
#ifdef USE_CUDA
    int biggest = M*K; if(N*K>biggest)biggest=N*K; if(M*N>biggest)biggest=M*N;
    gpu_ensure_tmp(biggest);
    gpu_upload(d_tmp_a, dC, M * N);
    float *d_weight = d_B ? d_B : d_tmp_b;
    if (!d_B) gpu_upload(d_tmp_b, B, N * K);
    gpu_sgemm_nn(M, K, N, d_tmp_a, d_weight, d_tmp_c);
    {
        static float *bwd_tmp = NULL;
        static int bwd_tmp_size = 0;
        int need = M * K > N * K ? M * K : N * K;
        if (need > bwd_tmp_size) {
            free(bwd_tmp);
            bwd_tmp = malloc(need * sizeof(float));
            bwd_tmp_size = need;
        }
        gpu_download(bwd_tmp, d_tmp_c, M * K);
        for (int i = 0; i < M * K; i++) dA[i] += bwd_tmp[i];
        gpu_upload(d_tmp_b, A, M * K);
        gpu_sgemm_tn(N, K, M, d_tmp_a, d_tmp_b, d_tmp_c);
        gpu_download(bwd_tmp, d_tmp_c, N * K);
        for (int i = 0; i < N * K; i++) dB[i] += bwd_tmp[i];
    }
#elif defined(USE_BLAS)
    (void)d_B;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, K, N, 1.0f, dC, N, B, K, 1.0f, dA, K);
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, N, K, M, 1.0f, dC, N, A, K, 1.0f, dB, K);
#else
    for (int m = 0; m < M; m++) {
        float *dcm = dC + m * N;
        float *am = A + m * K;
        for (int n = 0; n < N; n++) {
            float dc = dcm[n];
            if (dc == 0.0f) continue;
            float *bn = B + n * K;
            for (int k = 0; k < K; k++) {
                dA[m * K + k] += dc * bn[k];
                dB[n * K + k] += dc * am[k];
            }
        }
    }
#endif
}

/* RMSNorm forward */
static void rmsnorm_fwd_seq(float *out, float *x, float *weight, int T, int dim, float eps) {
    for (int t = 0; t < T; t++) {
        float *xt = x + t * dim, *ot = out + t * dim;
        float ss = 0.0f;
        for (int i = 0; i < dim; i++) ss += xt[i] * xt[i];
        float inv = 1.0f / sqrtf(ss / dim + eps);
        for (int i = 0; i < dim; i++) ot[i] = xt[i] * inv * weight[i];
    }
}

/* RMSNorm backward: dx += ..., dweight += ... */
static void rmsnorm_bwd_seq(float *dx, float *dw, float *dout,
                             float *x, float *w, int T, int dim, float eps) {
    for (int t = 0; t < T; t++) {
        float *xt = x + t * dim, *dot = dout + t * dim, *dxt = dx + t * dim;
        float ss = 0.0f;
        for (int i = 0; i < dim; i++) ss += xt[i] * xt[i];
        float var = ss / dim + eps;
        float inv = 1.0f / sqrtf(var);
        float coeff_sum = 0.0f;
        for (int i = 0; i < dim; i++) coeff_sum += dot[i] * w[i] * xt[i];
        float c2 = coeff_sum / (dim * var);
        for (int i = 0; i < dim; i++) {
            dxt[i] += (dot[i] * w[i] - xt[i] * c2) * inv;
            dw[i]  += dot[i] * xt[i] * inv;
        }
    }
}

/* RoPE backward: yes, through the rotation matrices. no, pytorch doesn't
 * teach you this. the gradient of a rotation is a counter-rotation.
 * who knew trig would be useful after high school. */
static void rope_bwd(float *dvec, int pos, float *cos_c, float *sin_c, int hd) {
    int half = hd / 2;
    int off = pos * half;
    for (int i = 0; i < half; i++) {
        float d0 = dvec[i], d1 = dvec[i + half];
        float co = cos_c[off+i], si = sin_c[off+i];
        dvec[i]        =  d0 * co + d1 * si;
        dvec[i + half] = -d0 * si + d1 * co;
    }
}

/* ─────────── Training forward pass ─────────── */

static float train_forward(ModelWeights *w, Config *c, TrainState *s,
                            int *tokens, int *targets, int T) {
    int D = c->dim, kv = c->n_kv_heads * c->head_dim;
    int qd = c->n_heads * c->head_dim, hd = c->head_dim;
    int H = c->hidden_dim, hg = c->n_heads / c->n_kv_heads;
    float scale = 1.0f / sqrtf((float)hd);
    s->T = T;

    /* 1. Embedding lookup → residual */
    for (int t = 0; t < T; t++)
        memcpy(s->residual + t * D, w->tok_emb->data + tokens[t] * D, D * sizeof(float));

    /* 2. Transformer layers */
    for (int l = 0; l < c->depth; l++) {
        LayerWeights *lw = &w->layers[l];
        LayerAct *la = &s->layers[l];

        memcpy(la->input_to_norm, s->residual, T * D * sizeof(float));
        rmsnorm_fwd_seq(la->xn, s->residual, lw->attn_norm->data, T, D, c->norm_eps);

        /* Q, K, V projections */
        matmul_fwd(la->q, la->xn, lw->wq->data, T, qd, D, GPU(lw->wq));
        matmul_fwd(la->k, la->xn, lw->wk->data, T, kv, D, GPU(lw->wk));
        matmul_fwd(la->v, la->xn, lw->wv->data, T, kv, D, GPU(lw->wv));

        /* RoPE */
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < c->n_heads; h++)
                apply_rope(la->q + t * qd + h * hd, t, s->cos_cache, s->sin_cache, hd);
            for (int h = 0; h < c->n_kv_heads; h++)
                apply_rope(la->k + t * kv + h * hd, t, s->cos_cache, s->sin_cache, hd);
        }

        /* Causal self-attention with GQA */
        memset(la->attn_out, 0, T * qd * sizeof(float));
        for (int h = 0; h < c->n_heads; h++) {
            int kvh = h / hg;
            for (int t = 0; t < T; t++) {
                float *qt = la->q + t * qd + h * hd;
                float *att = la->attn_scores + (t * c->n_heads + h) * T;

                float mx = -1e30f;
                for (int sp = 0; sp <= t; sp++) {
                    float *ks = la->k + sp * kv + kvh * hd;
                    float dot = 0.0f;
                    for (int d = 0; d < hd; d++) dot += qt[d] * ks[d];
                    att[sp] = dot * scale;
                    if (att[sp] > mx) mx = att[sp];
                }
                float se = 0.0f;
                for (int sp = 0; sp <= t; sp++) { att[sp] = expf(att[sp] - mx); se += att[sp]; }
                for (int sp = 0; sp <= t; sp++) att[sp] /= se;
                for (int sp = t + 1; sp < T; sp++) att[sp] = 0.0f;

                float *oh = la->attn_out + t * qd + h * hd;
                for (int sp = 0; sp <= t; sp++) {
                    float a = att[sp];
                    float *vs = la->v + sp * kv + kvh * hd;
                    for (int d = 0; d < hd; d++) oh[d] += a * vs[d];
                }
            }
        }

        /* Output projection + residual */
        float *attn_proj = calloc(T * D, sizeof(float));
        matmul_fwd(attn_proj, la->attn_out, lw->wo->data, T, D, qd, GPU(lw->wo));
        for (int i = 0; i < T * D; i++) s->residual[i] += attn_proj[i];
        free(attn_proj);
        memcpy(la->res_after_attn, s->residual, T * D * sizeof(float));

        /* FFN: norm → gate/up → SwiGLU → down → residual */
        rmsnorm_fwd_seq(la->ffn_xn, s->residual, lw->ffn_norm->data, T, D, c->norm_eps);
        matmul_fwd(la->gate_pre, la->ffn_xn, lw->w_gate->data, T, H, D, GPU(lw->w_gate));
        matmul_fwd(la->up_pre, la->ffn_xn, lw->w_up->data, T, H, D, GPU(lw->w_up));
        for (int i = 0; i < T * H; i++)
            la->swiglu[i] = silu(la->gate_pre[i]) * la->up_pre[i];
        float *ffn_proj = calloc(T * D, sizeof(float));
        matmul_fwd(ffn_proj, la->swiglu, lw->w_down->data, T, D, H, GPU(lw->w_down));
        for (int i = 0; i < T * D; i++) s->residual[i] += ffn_proj[i];
        free(ffn_proj);
    }

    /* 3. Final norm + LM head */
    s->final_normed = calloc(T * D, sizeof(float));
    rmsnorm_fwd_seq(s->final_normed, s->residual, w->output_norm->data, T, D, c->norm_eps);
    s->logits = calloc(T * c->vocab_size, sizeof(float));
    matmul_fwd(s->logits, s->final_normed, w->output->data, T, c->vocab_size, D, GPU(w->output));

    /* 4. Cross-entropy loss */
    float total_loss = 0.0f;
    int nv = 0;
    for (int t = 0; t < T; t++) {
        if (targets[t] < 0) continue;
        float *lt = s->logits + t * c->vocab_size;
        float mx = lt[0];
        for (int j = 1; j < c->vocab_size; j++) if (lt[j] > mx) mx = lt[j];
        float se = 0.0f;
        for (int j = 0; j < c->vocab_size; j++) se += expf(lt[j] - mx);
        total_loss += -(lt[targets[t]] - mx - logf(se));
        nv++;
    }
    return nv > 0 ? total_loss / nv : 0.0f;
}

/* ─────────── Training backward pass ─────────── */

static void train_backward(ModelWeights *w, Config *c, TrainState *s,
                            int *tokens, int *targets, int T, float **grads) {
    int D = c->dim, kv = c->n_kv_heads * c->head_dim;
    int qd = c->n_heads * c->head_dim, hd = c->head_dim;
    int H = c->hidden_dim, hg = c->n_heads / c->n_kv_heads;
    float scale = 1.0f / sqrtf((float)hd);
    int V = c->vocab_size;
    int nv = 0;
    for (int t = 0; t < T; t++) if (targets[t] >= 0) nv++;
    if (nv == 0) goto cleanup;
    float inv_n = 1.0f / (float)nv;

    /* Gradient array order: tok_emb(0), output(1), output_norm(2),
     * then per layer: attn_norm, wq, wk, wv, wo, ffn_norm, w_gate, w_up, w_down */

    /* ── d_logits = (softmax - onehot) / nv ── */
    float *d_logits = calloc(T * V, sizeof(float));
    for (int t = 0; t < T; t++) {
        if (targets[t] < 0) continue;
        float *lt = s->logits + t * V;
        float *dl = d_logits + t * V;
        float mx = lt[0];
        for (int j = 1; j < V; j++) if (lt[j] > mx) mx = lt[j];
        float se = 0.0f;
        for (int j = 0; j < V; j++) { dl[j] = expf(lt[j] - mx); se += dl[j]; }
        for (int j = 0; j < V; j++) dl[j] = (dl[j] / se) * inv_n;
        dl[targets[t]] -= inv_n;
    }

    /* ── LM head backward ── */
    float *d_fn = calloc(T * D, sizeof(float));
    matmul_bwd(d_fn, grads[1], d_logits, s->final_normed, w->output->data, T, V, D, GPU(w->output));

    /* ── Final norm backward ── */
    memset(s->d_residual, 0, T * D * sizeof(float));
    rmsnorm_bwd_seq(s->d_residual, grads[2], d_fn, s->residual, w->output_norm->data, T, D, c->norm_eps);

    free(d_fn);
    free(d_logits);

    /* ── Layers in reverse ── */
    for (int l = c->depth - 1; l >= 0; l--) {
        LayerWeights *lw = &w->layers[l];
        LayerAct *la = &s->layers[l];
        int gi = 3 + l * 9;

        /* === FFN backward === */
        /* ffn_proj (= swiglu @ w_down^T) backward */
        memset(s->d_swiglu, 0, T * H * sizeof(float));
        matmul_bwd(s->d_swiglu, grads[gi+8], s->d_residual, la->swiglu, lw->w_down->data, T, D, H, GPU(lw->w_down));

        /* SwiGLU backward: out = silu(gate) * up */
        for (int i = 0; i < T * H; i++) {
            float g = la->gate_pre[i], u = la->up_pre[i];
            float sig = 1.0f / (1.0f + expf(-g));
            float silu_g = g * sig;
            s->d_gate[i] = s->d_swiglu[i] * u * (sig + g * sig * (1.0f - sig));
            s->d_up[i]   = s->d_swiglu[i] * silu_g;
        }

        memset(s->d_ffn_xn, 0, T * D * sizeof(float));
        matmul_bwd(s->d_ffn_xn, grads[gi+6], s->d_gate, la->ffn_xn, lw->w_gate->data, T, H, D, GPU(lw->w_gate));
        matmul_bwd(s->d_ffn_xn, grads[gi+7], s->d_up,   la->ffn_xn, lw->w_up->data,   T, H, D, GPU(lw->w_up));

        /* FFN norm backward */
        rmsnorm_bwd_seq(s->d_residual, grads[gi+5], s->d_ffn_xn,
                         la->res_after_attn, lw->ffn_norm->data, T, D, c->norm_eps);

        /* === Attention backward === */
        /* Output projection backward */
        memset(s->d_attn_out, 0, T * qd * sizeof(float));
        matmul_bwd(s->d_attn_out, grads[gi+4], s->d_residual, la->attn_out, lw->wo->data, T, D, qd, GPU(lw->wo));

        /* Attention score backward */
        memset(s->d_q, 0, T * qd * sizeof(float));
        memset(s->d_k, 0, T * kv * sizeof(float));
        memset(s->d_v, 0, T * kv * sizeof(float));

        for (int h = 0; h < c->n_heads; h++) {
            int kvh = h / hg;
            for (int t = 0; t < T; t++) {
                float *d_oh = s->d_attn_out + t * qd + h * hd;
                float *att = la->attn_scores + (t * c->n_heads + h) * T;

                /* d_att and d_v */
                float da[512]; /* max T */
                for (int sp = 0; sp <= t; sp++) {
                    float *vs = la->v + sp * kv + kvh * hd;
                    float dd = 0.0f;
                    for (int d = 0; d < hd; d++) dd += d_oh[d] * vs[d];
                    da[sp] = dd;
                    float a = att[sp];
                    float *dvs = s->d_v + sp * kv + kvh * hd;
                    for (int d = 0; d < hd; d++) dvs[d] += a * d_oh[d];
                }

                /* Softmax backward: d_score = att * (d_att - dot(att, d_att)) */
                float dot_ad = 0.0f;
                for (int sp = 0; sp <= t; sp++) dot_ad += att[sp] * da[sp];

                float *qt = la->q + t * qd + h * hd;
                float *dqt = s->d_q + t * qd + h * hd;
                for (int sp = 0; sp <= t; sp++) {
                    float ds = att[sp] * (da[sp] - dot_ad) * scale;
                    float *ks = la->k + sp * kv + kvh * hd;
                    float *dks = s->d_k + sp * kv + kvh * hd;
                    for (int d = 0; d < hd; d++) {
                        dqt[d] += ds * ks[d];
                        dks[d] += ds * qt[d];
                    }
                }
            }
        }

        /* RoPE backward (inverse rotation) */
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < c->n_heads; h++)
                rope_bwd(s->d_q + t * qd + h * hd, t, s->cos_cache, s->sin_cache, hd);
            for (int h = 0; h < c->n_kv_heads; h++)
                rope_bwd(s->d_k + t * kv + h * hd, t, s->cos_cache, s->sin_cache, hd);
        }

        /* Q, K, V projection backward */
        memset(s->d_xn, 0, T * D * sizeof(float));
        matmul_bwd(s->d_xn, grads[gi+1], s->d_q, la->xn, lw->wq->data, T, qd, D, GPU(lw->wq));
        matmul_bwd(s->d_xn, grads[gi+2], s->d_k, la->xn, lw->wk->data, T, kv, D, GPU(lw->wk));
        matmul_bwd(s->d_xn, grads[gi+3], s->d_v, la->xn, lw->wv->data, T, kv, D, GPU(lw->wv));

        /* Attention norm backward → d_residual for prev layer */
        float *d_save = calloc(T * D, sizeof(float));
        memcpy(d_save, s->d_residual, T * D * sizeof(float));
        memset(s->d_residual, 0, T * D * sizeof(float));
        rmsnorm_bwd_seq(s->d_residual, grads[gi+0], s->d_xn,
                         la->input_to_norm, lw->attn_norm->data, T, D, c->norm_eps);
        for (int i = 0; i < T * D; i++) s->d_residual[i] += d_save[i];
        free(d_save);
    }

    /* ── Embedding backward ── */
    for (int t = 0; t < T; t++) {
        float *de = grads[0] + tokens[t] * D;
        float *dr = s->d_residual + t * D;
        for (int i = 0; i < D; i++) de[i] += dr[i];
    }

cleanup:
    free(s->logits); s->logits = NULL;
    free(s->final_normed); s->final_normed = NULL;
}
/* ═══════════════════════════════════════════════════════════════════════════════
 * ADAM OPTIMIZER — the training wheels of deep learning. m̂/(√v̂ + ε).
 * boring? yes. works? also yes. chuck will replace him eventually.
 * until then, adam does the job. he doesn't complain. he never does.
 * ═══════════════════════════════════════════════════════════════════════════════ */

typedef struct {
    float *m; /* first moment */
    float *v; /* second moment */
    int size;
} AdamState;

typedef struct {
    AdamState *states;
    int n_params;
    float beta1, beta2, eps;
    int t; /* timestep */
} Adam;

static Adam *adam_new(ParamList *params, float beta1, float beta2, float eps) {
    Adam *opt = calloc(1, sizeof(Adam));
    opt->n_params = params->count;
    opt->states = calloc(params->count, sizeof(AdamState));
    opt->beta1 = beta1;
    opt->beta2 = beta2;
    opt->eps = eps;
    opt->t = 0;

    for (int i = 0; i < params->count; i++) {
        int sz = params->tensors[i]->size;
        opt->states[i].m = calloc(sz, sizeof(float));
        opt->states[i].v = calloc(sz, sizeof(float));
        opt->states[i].size = sz;
    }
    return opt;
}

static void adam_step(Adam *opt, ParamList *params, float **grads, float lr, float wd) {
    opt->t++;
    float bc1 = 1.0f - powf(opt->beta1, (float)opt->t);
    float bc2 = 1.0f - powf(opt->beta2, (float)opt->t);

    for (int i = 0; i < opt->n_params; i++) {
        Tensor *p = params->tensors[i];
        float *g = grads[i];
        AdamState *s = &opt->states[i];
        if (!g) continue;

        for (int j = 0; j < p->size; j++) {
            /* Weight decay (AdamW) */
            if (wd > 0 && p->rows > 1) /* only on 2D params */
                p->data[j] -= lr * wd * p->data[j];

            s->m[j] = opt->beta1 * s->m[j] + (1.0f - opt->beta1) * g[j];
            s->v[j] = opt->beta2 * s->v[j] + (1.0f - opt->beta2) * g[j] * g[j];
            float m_hat = s->m[j] / bc1;
            float v_hat = s->v[j] / bc2;
            p->data[j] -= lr * m_hat / (sqrtf(v_hat) + opt->eps);
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * DATA LOADING — THREE sources, zero dependencies:
 *   1. --url: HuggingFace rows API → JSON → extracts "text" fields
 *   2. --parquet FILE: inline Snappy + Thrift + Parquet reader
 *   3. fallback: synthetic. shameful but functional.
 * yes, we have a parquet parser. in C. inline. deal with it.
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* ── HuggingFace JSON text extractor ── */
static int hf_extract_texts(const char *json, int json_len, FILE *out) {
    int count = 0;
    const char *p = json, *end = json + json_len;
    while (p < end) {
        p = strstr(p, "\"text\":\"");
        if (!p) break;
        p += 8;
        const char *start = p;
        while (p < end && !(*p == '"' && *(p-1) != '\\')) p++;
        if (p >= end) break;
        for (const char *s = start; s < p; s++) {
            if (*s == '\\' && s + 1 < p) {
                s++;
                if (*s == 'n') fputc('\n', out);
                else if (*s == 't') fputc('\t', out);
                else if (*s == '\\') fputc('\\', out);
                else if (*s == '"') fputc('"', out);
                else if (*s == 'u' && s + 4 < p) { fputc('?', out); s += 4; }
                else fputc(*s, out);
            } else fputc(*s, out);
        }
        fputc('\n', out);
        count++;
        p++;
    }
    return count;
}

/* ── Snappy decompressor (for parquet) ── */
static int snappy_decompress(const uint8_t *src, int slen, uint8_t *dst, int dlen) {
    int si = 0, di = 0;
    uint32_t ulen = 0; int shift = 0;
    while (si < slen) { uint8_t b = src[si++]; ulen |= (uint32_t)(b & 0x7F) << shift; if (!(b & 0x80)) break; shift += 7; }
    if ((int)ulen > dlen) return -1;
    while (si < slen && di < (int)ulen) {
        uint8_t tag = src[si++]; int type = tag & 3;
        if (type == 0) {
            int len = (tag >> 2) + 1;
            if ((tag >> 2) >= 60) { int nb = (tag >> 2) - 59; len = 0; for (int i = 0; i < nb && si < slen; i++) len |= src[si++] << (i * 8); len++; }
            if (si + len > slen || di + len > (int)ulen) return -1;
            memcpy(dst + di, src + si, len); si += len; di += len;
        } else {
            int len, off;
            if (type == 1) { len = ((tag >> 2) & 7) + 4; if (si >= slen) return -1; off = ((tag >> 5) << 8) | src[si++]; }
            else if (type == 2) { len = (tag >> 2) + 1; if (si + 1 >= slen) return -1; off = src[si] | (src[si+1] << 8); si += 2; }
            else { len = (tag >> 2) + 1; if (si + 3 >= slen) return -1; off = src[si] | (src[si+1]<<8) | (src[si+2]<<16) | (src[si+3]<<24); si += 4; }
            if (off == 0 || di - off < 0) return -1;
            for (int i = 0; i < len; i++) dst[di + i] = dst[di - off + i];
            di += len;
        }
    }
    return di;
}

/* ── Thrift Compact Protocol decoder (for parquet footer) ── */
typedef struct { const uint8_t *data; int pos, len; } TR;
static uint64_t tr_varint(TR *r) { uint64_t v=0; int s=0; while(r->pos<r->len){uint8_t b=r->data[r->pos++];v|=(uint64_t)(b&0x7F)<<s;if(!(b&0x80))break;s+=7;} return v; }
static int64_t tr_zigzag(TR *r) { uint64_t v=tr_varint(r); return (int64_t)((v>>1)^-(v&1)); }
static char *tr_string(TR *r) { uint64_t l=tr_varint(r); char *s=malloc(l+1); if(r->pos+(int)l<=r->len){memcpy(s,r->data+r->pos,l);r->pos+=(int)l;}s[l]=0; return s; }
static void tr_skip(TR *r, int type);
static void tr_skip_struct(TR *r) { int prev=0; while(r->pos<r->len){uint8_t b=r->data[r->pos++];if(b==0)break;int ft=b&0xF,delta=(b>>4)&0xF;if(delta==0){prev=(int)(int16_t)tr_zigzag(r);}else prev+=delta;tr_skip(r,ft);} }
static void tr_skip(TR *r, int type) {
    switch(type) {
        case 1: case 2: break;
        case 3: case 4: case 5: case 6: tr_zigzag(r); break;
        case 7: r->pos+=8; break;
        case 8: { uint64_t l=tr_varint(r); r->pos+=(int)l; break; }
        case 9: case 10: { uint8_t h=r->data[r->pos++]; int cnt=(h>>4)&0xF, et=h&0xF; if(cnt==0xF)cnt=(int)tr_varint(r); for(int i=0;i<cnt;i++)tr_skip(r,et); break; }
        case 11: { uint8_t h=r->data[r->pos++]; int kt=(h>>4)&0xF, vt=h&0xF; int cnt=(int)tr_varint(r); for(int i=0;i<cnt;i++){tr_skip(r,kt);tr_skip(r,vt);} break; }
        case 12: tr_skip_struct(r); break;
    }
}

/* ── Parquet reader ── */
typedef struct { char *name; int64_t data_off, dict_off, comp_size, nval; int codec; } PqCol;
typedef struct { PqCol *cols; int n; int64_t nrows; } PqMeta;

static PqMeta pq_footer(const uint8_t *f, int64_t sz) {
    PqMeta m = {0};
    uint32_t flen = *(uint32_t*)(f + sz - 8);
    TR r = { f + sz - 8 - flen, 0, (int)flen };
    int prev = 0;
    while (r.pos < r.len) {
        uint8_t b = r.data[r.pos++]; if (b == 0) break;
        int ft = b & 0xF, delta = (b >> 4) & 0xF;
        int fid = delta ? prev + delta : (int)(int16_t)tr_zigzag(&r); prev = fid;
        if (fid == 1 && ft == 5) tr_zigzag(&r);
        else if (fid == 2 && ft == 9) { uint8_t h=r.data[r.pos++]; int cnt=(h>>4)&0xF; if(cnt==0xF)cnt=(int)tr_varint(&r); for(int i=0;i<cnt;i++)tr_skip_struct(&r); }
        else if (fid == 3 && ft == 6) m.nrows = (int64_t)tr_zigzag(&r);
        else if (fid == 4 && ft == 9) {
            uint8_t h=r.data[r.pos++]; int rg_cnt=(h>>4)&0xF; if(rg_cnt==0xF)rg_cnt=(int)tr_varint(&r);
            for (int rg=0; rg<rg_cnt; rg++) {
                int rp=0;
                while (r.pos<r.len) { uint8_t rb=r.data[r.pos++]; if(rb==0)break; int rt=rb&0xF,rd=(rb>>4)&0xF; int rf=rd?rp+rd:(int)(int16_t)tr_zigzag(&r); rp=rf;
                    if (rf==1 && rt==9) {
                        uint8_t ch=r.data[r.pos++]; int cc=(ch>>4)&0xF; if(cc==0xF)cc=(int)tr_varint(&r);
                        for (int ci=0; ci<cc; ci++) {
                            PqCol col={0}; col.dict_off=-1; int cp=0;
                            while (r.pos<r.len) { uint8_t cb=r.data[r.pos++]; if(cb==0)break; int ct_=cb&0xF,cd_=(cb>>4)&0xF; int cf=cd_?cp+cd_:(int)(int16_t)tr_zigzag(&r); cp=cf;
                                if (cf==3 && ct_==12) {
                                    int mp=0;
                                    while (r.pos<r.len) { uint8_t mb=r.data[r.pos++]; if(mb==0)break; int mt=mb&0xF,md=(mb>>4)&0xF; int mf=md?mp+md:(int)(int16_t)tr_zigzag(&r); mp=mf;
                                        if (mf==3&&mt==9) { uint8_t lh=r.data[r.pos++]; int lc=(lh>>4)&0xF; if(lc==0xF)lc=(int)tr_varint(&r); for(int li=0;li<lc;li++){char*s=tr_string(&r);if(li==lc-1)col.name=s;else free(s);} }
                                        else if (mf==4&&mt==5) col.codec=(int)tr_zigzag(&r);
                                        else if (mf==5&&mt==6) col.nval=(int64_t)tr_zigzag(&r);
                                        else if (mf==7&&mt==6) col.comp_size=(int64_t)tr_zigzag(&r);
                                        else if (mf==9&&mt==6) col.data_off=(int64_t)tr_zigzag(&r);
                                        else if (mf==11&&mt==6) col.dict_off=(int64_t)tr_zigzag(&r);
                                        else tr_skip(&r,mt);
                                    }
                                } else tr_skip(&r,ct_);
                            }
                            m.n++; m.cols=realloc(m.cols,m.n*sizeof(PqCol)); m.cols[m.n-1]=col;
                        }
                    } else tr_skip(&r,rt);
                }
            }
        } else tr_skip(&r,ft);
    }
    return m;
}

typedef struct { int type, comp_sz, uncomp_sz, nval; } PgHdr;
static PgHdr pq_page_hdr(const uint8_t *data, int len, int *hlen) {
    TR r={data,0,len}; PgHdr h={0}; int prev=0;
    while (r.pos<r.len) { uint8_t b=r.data[r.pos++]; if(b==0)break; int ft=b&0xF,delta=(b>>4)&0xF; int fid=delta?prev+delta:(int)(int16_t)tr_zigzag(&r); prev=fid;
        if (fid==1&&ft==5) h.type=(int)tr_zigzag(&r);
        else if (fid==2&&ft==5) h.uncomp_sz=(int)tr_zigzag(&r);
        else if (fid==3&&ft==5) h.comp_sz=(int)tr_zigzag(&r);
        else if ((fid==5||fid==7||fid==8)&&ft==12) { int dp=0; while(r.pos<r.len){uint8_t db=r.data[r.pos++];if(db==0)break;int dt=db&0xF,dd=(db>>4)&0xF;int df=dd?dp+dd:(int)(int16_t)tr_zigzag(&r);dp=df;if(df==1&&dt==5)h.nval=(int)tr_zigzag(&r);else tr_skip(&r,dt);} }
        else tr_skip(&r,ft);
    }
    *hlen=r.pos; return h;
}

static int pq_extract(const uint8_t *file, int64_t fsz, PqCol *col, FILE *out) {
    int64_t pos=(col->dict_off>=0)?col->dict_off:col->data_off;
    int64_t end=col->data_off+col->comp_size;
    int total=0; char **dict=NULL; int *dlens=NULL, dsz=0;
    while (pos<end && pos<fsz) {
        int hlen; PgHdr ph=pq_page_hdr(file+pos,(int)(fsz-pos),&hlen);
        pos+=hlen; if(ph.comp_sz<=0||pos+ph.comp_sz>fsz)break;
        uint8_t *pd; int plen; int nf=0;
        if (col->codec==1) { pd=malloc(ph.uncomp_sz); plen=snappy_decompress(file+pos,ph.comp_sz,pd,ph.uncomp_sz); if(plen<0){free(pd);pos+=ph.comp_sz;continue;} nf=1; }
        else { pd=(uint8_t*)(file+pos); plen=ph.comp_sz; }
        if (ph.type==2) {
            dsz=ph.nval; dict=calloc(dsz,sizeof(char*)); dlens=calloc(dsz,sizeof(int));
            int dp=0;
            for(int i=0;i<dsz&&dp+4<=plen;i++){int32_t sl=*(int32_t*)(pd+dp);dp+=4;if(dp+sl>plen)break;dict[i]=malloc(sl);memcpy(dict[i],pd+dp,sl);dlens[i]=sl;dp+=sl;}
        } else if (ph.type==0||ph.type==3) {
            int dp=0;
            if (dsz>0) {
                if(dp>=plen)goto nxt;
                int bw=pd[dp++];
                for(int v=0;v<ph.nval&&dp<plen;){
                    uint8_t rh=pd[dp++];
                    if(rh&1){ int count=(rh>>1)*8,bytes=(count*bw+7)/8; uint64_t buf=0;int bb=0,bp=dp;
                        for(int i=0;i<count&&v<ph.nval;i++,v++){while(bb<bw&&bp<dp+bytes&&bp<plen){buf|=(uint64_t)pd[bp++]<<bb;bb+=8;}int idx=(int)(buf&((1ULL<<bw)-1));buf>>=bw;bb-=bw;if(idx>=0&&idx<dsz){fwrite(dict[idx],1,dlens[idx],out);fputc('\n',out);total++;}}
                        dp+=bytes;
                    } else { int count=rh>>1,idx=0,nb=(bw+7)/8; for(int b=0;b<nb&&dp<plen;b++)idx|=pd[dp++]<<(b*8);
                        for(int i=0;i<count&&v<ph.nval;i++,v++){if(idx>=0&&idx<dsz){fwrite(dict[idx],1,dlens[idx],out);fputc('\n',out);total++;}}}
                }
            } else {
                for(int v=0;v<ph.nval&&dp+4<=plen;v++){int32_t sl=*(int32_t*)(pd+dp);dp+=4;if(sl<0||dp+sl>plen)break;fwrite(pd+dp,1,sl,out);fputc('\n',out);dp+=sl;total++;}
            }
        }
        nxt: if(nf)free(pd); pos+=ph.comp_sz;
    }
    if(dict){for(int i=0;i<dsz;i++)free(dict[i]);free(dict);free(dlens);}
    return total;
}

static int load_parquet(const char *path, const char *out_path, const char *col_name) {
    FILE *f=fopen(path,"rb"); if(!f)return -1;
    fseek(f,0,SEEK_END); int64_t fsz=ftell(f); fseek(f,0,SEEK_SET);
    uint8_t *file=malloc(fsz); fread(file,1,fsz,f); fclose(f);
    if(fsz<12||memcmp(file,"PAR1",4)!=0||memcmp(file+fsz-4,"PAR1",4)!=0){free(file);return -1;}
    PqMeta meta=pq_footer(file,fsz);
    printf("[parquet] %lld rows, %d column chunks\n",(long long)meta.nrows,meta.n);
    FILE *out=fopen(out_path,"w"); if(!out){free(file);return -1;}
    int total=0;
    for(int i=0;i<meta.n;i++){if(meta.cols[i].name&&strcmp(meta.cols[i].name,col_name)==0)total+=pq_extract(file,fsz,&meta.cols[i],out);}
    fclose(out);
    for(int i=0;i<meta.n;i++)free(meta.cols[i].name); free(meta.cols); free(file);
    printf("[parquet] extracted %d texts from '%s'\n",total,col_name);
    return total>0?0:-1;
}

/* Download training text — HF rows API paginated, or synthetic fallback */
#define HF_BATCH 100
#define HF_PAGES 50

static int download_data(Config *c) {
    struct stat st;
    if (stat(c->data_path, &st) == 0 && st.st_size > 1000) {
        printf("[data] found existing %s (%.1f MB)\n", c->data_path,
               (float)st.st_size / 1048576.0f);
        return 0;
    }
    if (c->data_url[0]) {
        printf("[data] fetching FineWeb-Edu from HuggingFace (%d pages)...\n", HF_PAGES);
        FILE *out = fopen(c->data_path, "w"); if (!out) goto synthetic;
        char tmp[280]; snprintf(tmp, sizeof(tmp), "%s.json", c->data_path);
        int total = 0;
        for (int page = 0; page < HF_PAGES; page++) {
            char cmd[1024];
            snprintf(cmd, sizeof(cmd),
                "curl -sL 'https://datasets-server.huggingface.co/rows"
                "?dataset=HuggingFaceFW/fineweb-edu"
                "&config=sample-10BT&split=train&offset=%d&length=%d' -o '%s'",
                page * HF_BATCH, HF_BATCH, tmp);
            if (system(cmd) != 0) continue;
            if (stat(tmp, &st) != 0 || st.st_size < 500) continue;
            FILE *jf = fopen(tmp, "r"); if (!jf) continue;
            char *json = malloc(st.st_size + 1);
            int jl = (int)fread(json, 1, st.st_size, jf); json[jl] = 0; fclose(jf);
            int n = hf_extract_texts(json, jl, out);
            free(json); total += n;
            if ((page + 1) % 10 == 0)
                printf("[data] page %d/%d — %d texts so far\n", page + 1, HF_PAGES, total);
        }
        fclose(out); unlink(tmp);
        if (total > 0) {
            stat(c->data_path, &st);
            printf("[data] downloaded %d texts (%.1f MB) from FineWeb-Edu\n",
                   total, (float)st.st_size / 1048576.0f);
            return 0;
        }
        printf("[data] HuggingFace download failed\n");
    }
    synthetic:
    printf("[data] creating synthetic dataset for demo...\n");
    FILE *f = fopen(c->data_path, "w");
    if (!f) return -1;
    const char *samples[] = {
        "The quick brown fox jumps over the lazy dog. This is a simple sentence for training.",
        "Machine learning is a subset of artificial intelligence that focuses on learning from data.",
        "Neural networks are inspired by biological neurons and can learn complex patterns.",
        "Language models predict the next token in a sequence based on the preceding context.",
        "Transformers use self-attention mechanisms to process sequences in parallel.",
        "The attention mechanism allows the model to focus on relevant parts of the input.",
        "Training involves minimizing the cross-entropy loss between predictions and targets.",
        "Backpropagation computes gradients by applying the chain rule through the network.",
        "The Adam optimizer combines momentum and adaptive learning rates for each parameter.",
        "Tokenization converts text into numerical tokens that the model can process.",
        NULL
    };
    for (int repeat = 0; repeat < 500; repeat++)
        for (int i = 0; samples[i]; i++)
            fprintf(f, "%s\n", samples[i]);
    fclose(f);
    printf("[data] created synthetic dataset: %s\n", c->data_path);
    return 0;
}

/* Load text file into memory */
static char *load_text(const char *path, int *out_len) {
    FILE *f = fopen(path, "r");
    if (!f) { *out_len = 0; return NULL; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *text = malloc(sz + 1);
    *out_len = (int)fread(text, 1, sz, f);
    text[*out_len] = '\0';
    fclose(f);
    return text;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * GGUF EXPORT — spits out a .gguf that llama.cpp can load.
 * your model trains in l.c and inferences in llama.cpp. interoperability
 * without committees, standards bodies, or 14 competing serialization formats.
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* ═══════════════════════════════════════════════════════════════════════════════
 * CHECKPOINT — binary save/load so you don't retrain every time you want to chat.
 * format: magic(4) + config fields + tokenizer(vocab+merges) + all weights.
 * ═══════════════════════════════════════════════════════════════════════════════ */

#define CKPT_MAGIC 0x4C4C414D /* "LLAM" */

static void save_checkpoint(const char *path, ModelWeights *w, Config *c, Tokenizer *tok) {
    FILE *f = fopen(path, "wb");
    if (!f) { printf("[ckpt] cannot create %s\n", path); return; }

    uint32_t magic = CKPT_MAGIC;
    fwrite(&magic, 4, 1, f);
    fwrite(&c->depth, 4, 1, f);
    fwrite(&c->dim, 4, 1, f);
    fwrite(&c->n_heads, 4, 1, f);
    fwrite(&c->n_kv_heads, 4, 1, f);
    fwrite(&c->head_dim, 4, 1, f);
    fwrite(&c->hidden_dim, 4, 1, f);
    fwrite(&c->vocab_size, 4, 1, f);
    fwrite(&c->seq_len, 4, 1, f);
    fwrite(&c->norm_eps, 4, 1, f);
    fwrite(&c->rope_theta, 4, 1, f);

    /* tokenizer: vocab strings + merges */
    fwrite(&tok->vocab_size, 4, 1, f);
    for (int i = 0; i < tok->vocab_size; i++) {
        int len = tok->tokens[i] ? (int)strlen(tok->tokens[i]) : 0;
        fwrite(&len, 4, 1, f);
        if (len > 0) fwrite(tok->tokens[i], 1, len, f);
    }
    fwrite(&tok->n_merges, 4, 1, f);
    for (int i = 0; i < tok->n_merges; i++) {
        fwrite(tok->merges[i].a, 1, 64, f);
        fwrite(tok->merges[i].b, 1, 64, f);
    }

    /* weights */
    fwrite(w->tok_emb->data, 4, w->tok_emb->size, f);
    fwrite(w->output_norm->data, 4, w->output_norm->size, f);
    fwrite(w->output->data, 4, w->output->size, f);
    for (int l = 0; l < c->depth; l++) {
        LayerWeights *lw = &w->layers[l];
        fwrite(lw->attn_norm->data, 4, lw->attn_norm->size, f);
        fwrite(lw->wq->data, 4, lw->wq->size, f);
        fwrite(lw->wk->data, 4, lw->wk->size, f);
        fwrite(lw->wv->data, 4, lw->wv->size, f);
        fwrite(lw->wo->data, 4, lw->wo->size, f);
        fwrite(lw->ffn_norm->data, 4, lw->ffn_norm->size, f);
        fwrite(lw->w_gate->data, 4, lw->w_gate->size, f);
        fwrite(lw->w_up->data, 4, lw->w_up->size, f);
        fwrite(lw->w_down->data, 4, lw->w_down->size, f);
    }

    fclose(f);
    struct stat st; stat(path, &st);
    printf("[ckpt] saved %s (%.1f MB)\n", path, (float)st.st_size / 1048576);
}

static int load_checkpoint(const char *path, ModelWeights *w, Config *c, Tokenizer *tok) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[ckpt] cannot open %s\n", path); return -1; }

    uint32_t magic;
    fread(&magic, 4, 1, f);
    if (magic != CKPT_MAGIC) { fprintf(stderr, "[ckpt] bad magic\n"); fclose(f); return -1; }

    fread(&c->depth, 4, 1, f);
    fread(&c->dim, 4, 1, f);
    fread(&c->n_heads, 4, 1, f);
    fread(&c->n_kv_heads, 4, 1, f);
    fread(&c->head_dim, 4, 1, f);
    fread(&c->hidden_dim, 4, 1, f);
    fread(&c->vocab_size, 4, 1, f);
    fread(&c->seq_len, 4, 1, f);
    fread(&c->norm_eps, 4, 1, f);
    fread(&c->rope_theta, 4, 1, f);

    /* tokenizer */
    tok_init(tok);
    int vs; fread(&vs, 4, 1, f);
    for (int i = 0; i < vs; i++) {
        int len; fread(&len, 4, 1, f);
        char buf[256] = {0};
        if (len > 0 && len < 256) fread(buf, 1, len, f);
        tok_add(tok, buf);
    }
    int nm; fread(&nm, 4, 1, f);
    tok->merges = calloc(nm, sizeof(MergePair));
    tok->n_merges = nm;
    for (int i = 0; i < nm; i++) {
        fread(tok->merges[i].a, 1, 64, f);
        fread(tok->merges[i].b, 1, 64, f);
    }

    /* weights — init structure then load data */
    init_weights(w, c);
    fread(w->tok_emb->data, 4, w->tok_emb->size, f);
    fread(w->output_norm->data, 4, w->output_norm->size, f);
    fread(w->output->data, 4, w->output->size, f);
    for (int l = 0; l < c->depth; l++) {
        LayerWeights *lw = &w->layers[l];
        fread(lw->attn_norm->data, 4, lw->attn_norm->size, f);
        fread(lw->wq->data, 4, lw->wq->size, f);
        fread(lw->wk->data, 4, lw->wk->size, f);
        fread(lw->wv->data, 4, lw->wv->size, f);
        fread(lw->wo->data, 4, lw->wo->size, f);
        fread(lw->ffn_norm->data, 4, lw->ffn_norm->size, f);
        fread(lw->w_gate->data, 4, lw->w_gate->size, f);
        fread(lw->w_up->data, 4, lw->w_up->size, f);
        fread(lw->w_down->data, 4, lw->w_down->size, f);
    }

    fclose(f);
    printf("[ckpt] loaded %s — depth=%d dim=%d vocab=%d params=%.2fM\n",
           path, c->depth, c->dim, c->vocab_size,
           (float)count_params(c) / 1e6f);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * GGUF EXPORT — because llama.cpp won't run your raw floats.
 * ═══════════════════════════════════════════════════════════════════════════════ */

static void write_u32(FILE *f, uint32_t v) { fwrite(&v, 4, 1, f); }
static void write_u64(FILE *f, uint64_t v) { fwrite(&v, 8, 1, f); }

static void write_gguf_string(FILE *f, const char *s) {
    uint64_t len = strlen(s);
    write_u64(f, len);
    fwrite(s, 1, len, f);
}

static void write_gguf_kv_string(FILE *f, const char *key, const char *val) {
    write_gguf_string(f, key);
    write_u32(f, 8); /* GGUF_TYPE_STRING */
    write_gguf_string(f, val);
}

static void write_gguf_kv_u32(FILE *f, const char *key, uint32_t val) {
    write_gguf_string(f, key);
    write_u32(f, 4); /* GGUF_TYPE_UINT32 */
    write_u32(f, val);
}

static void write_gguf_kv_f32(FILE *f, const char *key, float val) {
    write_gguf_string(f, key);
    write_u32(f, 6); /* GGUF_TYPE_FLOAT32 */
    fwrite(&val, 4, 1, f);
}

static void export_gguf(ModelWeights *w, Config *c, Tokenizer *tok) {
    FILE *f = fopen(c->gguf_path, "wb");
    if (!f) { printf("[gguf] cannot create %s\n", c->gguf_path); return; }

    /* Count tensors */
    int n_tensors = 3 + c->depth * 9; /* tok_emb, output, output_norm + 9 per layer */

    /* Count metadata KV pairs */
    int n_metadata = 12; /* architecture + model config + tokenizer basics */
    if (c->rope_scaling > 1.0f) n_metadata++;

    /* Header */
    write_u32(f, 0x46554747); /* magic "GGUF" */
    write_u32(f, 3);          /* version */
    write_u64(f, n_tensors);
    write_u64(f, n_metadata);

    /* Metadata */
    write_gguf_kv_string(f, "general.architecture", "llama");
    write_gguf_kv_string(f, "general.name", "l");
    write_gguf_kv_u32(f, "llama.block_count", c->depth);
    write_gguf_kv_u32(f, "llama.embedding_length", c->dim);
    write_gguf_kv_u32(f, "llama.attention.head_count", c->n_heads);
    write_gguf_kv_u32(f, "llama.attention.head_count_kv", c->n_kv_heads);
    write_gguf_kv_u32(f, "llama.feed_forward_length", c->hidden_dim);
    write_gguf_kv_u32(f, "llama.context_length", c->seq_len);
    write_gguf_kv_f32(f, "llama.attention.layer_norm_rms_epsilon", c->norm_eps);
    write_gguf_kv_f32(f, "llama.rope.freq_base", c->rope_theta);
    if (c->rope_scaling > 1.0f)
        write_gguf_kv_f32(f, "llama.rope.scaling_factor", c->rope_scaling);
    write_gguf_kv_string(f, "tokenizer.ggml.model", "gpt2");
    write_gguf_kv_u32(f, "tokenizer.ggml.vocab_size", c->vocab_size); /* non-standard but useful */

    /* Tensor infos — name, ndims, dims, type(F32=0), offset */
    uint64_t offset = 0;

    /* Helper: write tensor info */
    #define WRITE_TENSOR_INFO(name, tensor) do { \
        write_gguf_string(f, name); \
        if ((tensor)->rows > 1) { \
            write_u32(f, 2); \
            write_u64(f, (tensor)->cols); \
            write_u64(f, (tensor)->rows); \
        } else { \
            write_u32(f, 1); \
            write_u64(f, (tensor)->size); \
        } \
        write_u32(f, 0); /* F32 */ \
        write_u64(f, offset); \
        offset += (tensor)->size * 4; \
    } while(0)

    WRITE_TENSOR_INFO("token_embd.weight", w->tok_emb);
    WRITE_TENSOR_INFO("output_norm.weight", w->output_norm);
    WRITE_TENSOR_INFO("output.weight", w->output);

    for (int l = 0; l < c->depth; l++) {
        LayerWeights *lw = &w->layers[l];
        char name[64];
        snprintf(name, 64, "blk.%d.attn_norm.weight", l); WRITE_TENSOR_INFO(name, lw->attn_norm);
        snprintf(name, 64, "blk.%d.attn_q.weight", l);    WRITE_TENSOR_INFO(name, lw->wq);
        snprintf(name, 64, "blk.%d.attn_k.weight", l);    WRITE_TENSOR_INFO(name, lw->wk);
        snprintf(name, 64, "blk.%d.attn_v.weight", l);    WRITE_TENSOR_INFO(name, lw->wv);
        snprintf(name, 64, "blk.%d.attn_output.weight", l); WRITE_TENSOR_INFO(name, lw->wo);
        snprintf(name, 64, "blk.%d.ffn_norm.weight", l);  WRITE_TENSOR_INFO(name, lw->ffn_norm);
        snprintf(name, 64, "blk.%d.ffn_gate.weight", l);  WRITE_TENSOR_INFO(name, lw->w_gate);
        snprintf(name, 64, "blk.%d.ffn_up.weight", l);    WRITE_TENSOR_INFO(name, lw->w_up);
        snprintf(name, 64, "blk.%d.ffn_down.weight", l);  WRITE_TENSOR_INFO(name, lw->w_down);
    }

    /* Alignment padding to 32 bytes */
    long pos = ftell(f);
    long aligned = ((pos + 31) / 32) * 32;
    for (long i = pos; i < aligned; i++) fputc(0, f);

    /* Tensor data */
    fwrite(w->tok_emb->data, 4, w->tok_emb->size, f);
    fwrite(w->output_norm->data, 4, w->output_norm->size, f);
    fwrite(w->output->data, 4, w->output->size, f);

    for (int l = 0; l < c->depth; l++) {
        LayerWeights *lw = &w->layers[l];
        fwrite(lw->attn_norm->data, 4, lw->attn_norm->size, f);
        fwrite(lw->wq->data, 4, lw->wq->size, f);
        fwrite(lw->wk->data, 4, lw->wk->size, f);
        fwrite(lw->wv->data, 4, lw->wv->size, f);
        fwrite(lw->wo->data, 4, lw->wo->size, f);
        fwrite(lw->ffn_norm->data, 4, lw->ffn_norm->size, f);
        fwrite(lw->w_gate->data, 4, lw->w_gate->size, f);
        fwrite(lw->w_up->data, 4, lw->w_up->size, f);
        fwrite(lw->w_down->data, 4, lw->w_down->size, f);
    }

    fclose(f);

    struct stat st;
    stat(c->gguf_path, &st);
    printf("[gguf] exported to %s (%.1f MB)\n", c->gguf_path,
           (float)st.st_size / 1024 / 1024);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * GENERATION — you trained it, now talk to it. KV cache, temperature,
 * top-k sampling. it's a chat loop. it's 2026. everyone has a chat loop.
 * ours just doesn't need a GPU cluster to run.
 * ═══════════════════════════════════════════════════════════════════════════════ */

static int sample_token(float *logits, int vocab_size, float temperature, int top_k) {
    if (temperature <= 0) {
        /* Greedy */
        int best = 0;
        for (int i = 1; i < vocab_size; i++)
            if (logits[i] > logits[best]) best = i;
        return best;
    }

    /* Temperature scaling */
    for (int i = 0; i < vocab_size; i++) logits[i] /= temperature;

    /* Top-K filtering */
    if (top_k > 0 && top_k < vocab_size) {
        float threshold = -1e30f;
        /* Find k-th largest (simple partial sort) */
        float *sorted = malloc(vocab_size * sizeof(float));
        memcpy(sorted, logits, vocab_size * sizeof(float));
        for (int i = 0; i < top_k; i++) {
            int best = i;
            for (int j = i + 1; j < vocab_size; j++)
                if (sorted[j] > sorted[best]) best = j;
            float tmp = sorted[i]; sorted[i] = sorted[best]; sorted[best] = tmp;
        }
        threshold = sorted[top_k - 1];
        free(sorted);
        for (int i = 0; i < vocab_size; i++)
            if (logits[i] < threshold) logits[i] = -1e30f;
    }

    /* Softmax */
    softmax(logits, vocab_size);

    /* Sample */
    float r = rand_uniform();
    float cum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        cum += logits[i];
        if (cum >= r) return i;
    }
    return vocab_size - 1;
}

static void chat_loop(ModelWeights *w, Config *c, Tokenizer *tok) {
    RunState rs = alloc_run_state(c);
    char input[1024];

    printf("\n[l] ready. type your message (Ctrl+C to quit):\n\n");

    while (1) {
        printf("> ");
        fflush(stdout);
        if (!fgets(input, sizeof(input), stdin)) break;
        int len = strlen(input);
        while (len > 0 && (input[len-1] == '\n' || input[len-1] == '\r')) input[--len] = '\0';
        if (len == 0) continue;
        if (strcmp(input, "quit") == 0 || strcmp(input, "exit") == 0) break;

        /* Reset KV cache */
        int kv_dim = c->n_kv_heads * c->head_dim;
        size_t cache_bytes = c->depth * c->seq_len * kv_dim * (c->fp16_cache ? sizeof(half) : sizeof(float));
        memset(rs.key_cache, 0, cache_bytes);
        memset(rs.value_cache, 0, cache_bytes);

        /* Encode input */
        int n_input_ids;
        int *input_ids = tok_encode(tok, input, len, &n_input_ids);

        /* Feed input tokens (prefill) */
        int pos = 0;
        for (int i = 0; i < n_input_ids && pos < c->seq_len - 1; i++, pos++) {
            forward_token(w, c, &rs, input_ids[i], pos);
        }

        /* Generate response */
        int prev_token = input_ids[n_input_ids - 1];
        int generated = 0;
        int max_gen = 200;

        printf("  ");
        for (int i = 0; i < max_gen && pos < c->seq_len; i++, pos++) {
            float *logits = forward_token(w, c, &rs, prev_token, pos);
            int next = sample_token(logits, c->vocab_size, 0.8f, 40);

            if (next == tok->eos_id) break;

            /* Decode and print token */
            int dec_len;
            char *dec = tok_decode(tok, &next, 1, &dec_len);
            if (dec_len > 0) {
                fwrite(dec, 1, dec_len, stdout);
                fflush(stdout);
            }
            free(dec);

            prev_token = next;
            generated++;
        }
        printf("\n\n");
        free(input_ids);
    }

    free_run_state(&rs);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * MAIN — data → tokenizer → model → train → personality → GGUF → chat.
 * seven steps. one function. one file. your move, huggingface.
 * ═══════════════════════════════════════════════════════════════════════════════ */

int main(int argc, char **argv) {
    setbuf(stdout, NULL); /* unbuffered output */
    int depth = 4; /* default */
    char *chat_ckpt = NULL;

    /* Parse args */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--depth") == 0 && i + 1 < argc) {
            depth = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--chat") == 0 && i + 1 < argc) {
            chat_ckpt = argv[++i];
        } else if (strcmp(argv[i], "--data") == 0 && i + 1 < argc) {
            /* custom data path handled below */
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("l.c — one file. one llama. no excuses.\n\n");
            printf("Usage: ./l [options]\n");
            printf("  --depth N       Model depth (2=~1M, 4=~3M, 6=~7M, 8=~15M params)\n");
            printf("  --chat FILE     Load checkpoint and chat (skip training)\n");
            printf("  --data PATH     Path to training text file\n");
            printf("  --url URL       HuggingFace rows API URL for training data\n");
            printf("  --parquet FILE  Extract text from local .parquet file\n");
            printf("  --fp16-cache    Half-precision KV cache (saves memory)\n");
            printf("  --rope-scale F  NTK scaling for RoPE (>1 extends context)\n");
            printf("  --help          Show this help\n");
            printf("\n  BLAS:  cc l.c -O3 -lm -DUSE_BLAS -DACCELERATE -framework Accelerate -o l\n");
            return 0;
        }
    }

    printf("\n");
    printf("  ╔══════════════════════════════════════╗\n");
    printf("  ║  l.c — actually llama                ║\n");
    printf("  ║  one file. no frameworks. no excuses. ║\n");
    printf("  ╚══════════════════════════════════════╝\n\n");

    /* ── Chat-only mode: load checkpoint, skip training ── */
    if (chat_ckpt) {
        Config c = {0};
        c.norm_eps = 1e-5f;
        c.rope_theta = 10000.0f;
        Tokenizer tok;
        ModelWeights w;
        if (load_checkpoint(chat_ckpt, &w, &c, &tok) != 0) return 1;
        chat_loop(&w, &c, &tok);
        printf("[l] done.\n");
        return 0;
    }

    Config c = config_from_depth(depth);
#ifdef USE_CUDA
    if (gpu_init() != 0) { fprintf(stderr, "[error] CUDA init failed\n"); return 1; }
#endif

    /* Parse overrides */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--data") == 0 && i + 1 < argc)
            snprintf(c.data_path, sizeof(c.data_path), "%s", argv[i+1]);
        else if (strcmp(argv[i], "--url") == 0 && i + 1 < argc)
            snprintf(c.data_url, sizeof(c.data_url), "%s", argv[++i]);
        else if (strcmp(argv[i], "--parquet") == 0 && i + 1 < argc) {
            const char *pqf = argv[++i];
            printf("[parquet] loading %s...\n", pqf);
            if (load_parquet(pqf, c.data_path, "text") != 0) {
                fprintf(stderr, "[error] parquet load failed\n"); return 1;
            }
        }
        else if (strcmp(argv[i], "--fp16-cache") == 0)
            c.fp16_cache = 1;
        else if (strcmp(argv[i], "--rope-scale") == 0 && i + 1 < argc)
            c.rope_scaling = atof(argv[++i]);
    }

    /* ── Step 1: Get data ── */
    if (download_data(&c) != 0) {
        fprintf(stderr, "[error] cannot get training data\n");
        return 1;
    }

    int text_len;
    char *text = load_text(c.data_path, &text_len);
    if (!text || text_len < 100) {
        fprintf(stderr, "[error] training data too small\n");
        return 1;
    }
    printf("[data] loaded %d bytes (%.1f MB)\n", text_len, (float)text_len / 1024 / 1024);

    /* ── Step 2: Train BPE tokenizer ── */
    Tokenizer tok;
    tok_init(&tok);
    /* BPE on first 1MB — O(n²) per merge, full corpus is too slow */
    int bpe_len = text_len < 1000000 ? text_len : 1000000;
    tok_train_bpe(&tok, text, bpe_len, c.bpe_merges);
    c.vocab_size = tok.vocab_size;

    /* Tokenize training data */
    int n_tokens;
    int *all_tokens = tok_encode(&tok, text, text_len, &n_tokens);
    free(text);
    printf("[data] tokenized: %d tokens (%.1f tokens/byte)\n",
           n_tokens, (float)n_tokens / text_len);

    /* ── Step 3: Init model ── */
    long n_params = count_params(&c);
    printf("[model] depth=%d dim=%d heads=%d kv_heads=%d hidden=%d\n",
           c.depth, c.dim, c.n_heads, c.n_kv_heads, c.hidden_dim);
    printf("[model] vocab=%d seq_len=%d params=%.2fM\n",
           c.vocab_size, c.seq_len, (float)n_params / 1e6f);

    ModelWeights w;
    init_weights(&w, &c);

#ifdef USE_CUDA
    /* Upload ALL weights to GPU — resident, no per-matmul transfers */
    {
        ParamList tmp = collect_params(&w);
        gpu_upload_weights(tmp.tensors, tmp.count);
        int total_gpu = 0;
        for (int i = 0; i < tmp.count; i++) total_gpu += tmp.tensors[i]->size;
        printf("[cuda] uploaded %d weight tensors to GPU (%.1f MB resident)\n",
               tmp.count, total_gpu * 4.0f / 1048576.0f);
        fflush(stdout);
    }
#endif

    ParamList params = collect_params(&w);

    /* Allocate gradient buffers */
    float **grads = calloc(params.count, sizeof(float*));
    for (int i = 0; i < params.count; i++)
        grads[i] = calloc(params.tensors[i]->size, sizeof(float));

    Adam *opt = adam_new(&params, 0.9f, 0.999f, 1e-8f);

    /* ── Step 4: Train ── */
    TrainState ts = alloc_train_state(&c);

    printf("[train] starting training: %d steps, batch=%d, seq=%d, lr=%.1e\n",
           c.max_steps, c.batch_size, c.seq_len, c.lr);
    fflush(stdout);

#ifdef USE_CUDA
    /* Pre-allocate GPU buffers for largest matmul: output head T×V×D */
    {
        int T = c.seq_len, V = c.vocab_size, D = c.dim, H = c.hidden_dim;
        int biggest = T * V; /* T×V is largest for output head */
        if (T * H > biggest) biggest = T * H;
        if (V * D > biggest) biggest = V * D;
        if (H * D > biggest) biggest = H * D;
        gpu_ensure_tmp(biggest);
        printf("[cuda] pre-allocated GPU buffers: %d floats (%.1f MB each)\n",
               biggest, biggest * 4.0f / 1048576.0f);
        fflush(stdout);
    }
#endif

    clock_t train_start = clock();
    float running_loss = 0.0f;
    int loss_count = 0;

    for (int step = 0; step < c.max_steps; step++) {
        /* LR schedule: linear warmup + cosine decay */
        float lr = c.lr;
        if (step < c.warmup_steps) {
            lr = c.lr * ((float)(step + 1) / c.warmup_steps);
        } else {
            float progress = (float)(step - c.warmup_steps) / (float)(c.max_steps - c.warmup_steps);
            lr = c.lr * 0.5f * (1.0f + cosf(3.14159f * progress));
        }
        if (lr < c.lr * 0.01f) lr = c.lr * 0.01f;

        /* Sample random sequence from training data */
        int max_start = n_tokens - c.seq_len - 1;
        if (max_start < 0) max_start = 0;
        int start = (int)(rand_uniform() * max_start);

        int *tokens = all_tokens + start;
        int *targets = all_tokens + start + 1;

        /* Forward */
        float loss = train_forward(&w, &c, &ts, tokens, targets, c.seq_len);

        running_loss += loss;
        loss_count++;

        /* Zero grads */
        for (int i = 0; i < params.count; i++)
            memset(grads[i], 0, params.tensors[i]->size * sizeof(float));

        /* Analytical backward pass through all layers */
        train_backward(&w, &c, &ts, tokens, targets, c.seq_len, grads);

        /* Gradient clipping (global norm) */
        float grad_norm = 0.0f;
        for (int i = 0; i < params.count; i++)
            for (int j = 0; j < params.tensors[i]->size; j++)
                grad_norm += grads[i][j] * grads[i][j];
        grad_norm = sqrtf(grad_norm);
        float clip = 1.0f;
        if (grad_norm > clip) {
            float scale_g = clip / grad_norm;
            for (int i = 0; i < params.count; i++)
                for (int j = 0; j < params.tensors[i]->size; j++)
                    grads[i][j] *= scale_g;
        }

        /* Adam step */
        adam_step(opt, &params, grads, lr, c.weight_decay);
#ifdef USE_CUDA
        /* Resync updated weights to GPU after Adam modifies them */
        gpu_resync_weights(params.tensors, params.count);
#endif

        /* Logging */
        if ((step + 1) % c.log_every == 0 || step == 0) {
            float avg_loss = running_loss / loss_count;
            float elapsed = (float)(clock() - train_start) / CLOCKS_PER_SEC;
            float tok_per_sec = (float)((step + 1) * c.seq_len) / elapsed;
            printf("  step %4d/%d  loss=%.4f  lr=%.2e  tok/s=%.0f  (%.1fs)\n",
                   step + 1, c.max_steps, avg_loss, lr, tok_per_sec, elapsed);
            fflush(stdout);
            running_loss = 0;
            loss_count = 0;
        }
    }

    float total_time = (float)(clock() - train_start) / CLOCKS_PER_SEC;
    printf("[train] finished in %.1f seconds\n", total_time);

    /* ── Step 5: Personality finetune ── */
    struct stat st;
    if (stat(c.personality_path, &st) == 0) {
        printf("[personality] found %s, finetuning...\n", c.personality_path);
        int pers_len;
        char *pers_text = load_text(c.personality_path, &pers_len);
        if (pers_text && pers_len > 10) {
            int n_pers_tokens;
            int *pers_tokens = tok_encode(&tok, pers_text, pers_len, &n_pers_tokens);

            for (int step = 0; step < c.personality_steps && n_pers_tokens > c.seq_len + 1; step++) {
                int start = (int)(rand_uniform() * (n_pers_tokens - c.seq_len - 1));
                int *toks = pers_tokens + start;
                int *tgts = pers_tokens + start + 1;
                float loss = train_forward(&w, &c, &ts, toks, tgts, c.seq_len);
                if ((step + 1) % 20 == 0)
                    printf("  personality step %d/%d  loss=%.4f\n",
                           step + 1, c.personality_steps, loss);
            }
            free(pers_tokens);
        }
        free(pers_text);
    } else {
        printf("[personality] no %s found, skipping finetune\n", c.personality_path);
    }

    /* ── Step 6: Save checkpoint + Export GGUF ── */
    save_checkpoint("l.bin", &w, &c, &tok);
    export_gguf(&w, &c, &tok);

    /* ── Step 7: Chat ── */
    chat_loop(&w, &c, &tok);

    /* Cleanup */
    free(all_tokens);
    printf("[l] done.\n");
    return 0;
}
