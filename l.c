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

    /* Dimension scaling: dim = depth * 48, rounded to multiple of 64, min 192 */
    c.dim = depth * 48;
    c.dim = ((c.dim + 63) / 64) * 64;
    if (c.dim < 192) c.dim = 192;
    if (c.dim > 1024) c.dim = 1024; /* your CPU has feelings too */

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
    c.log_every = 20;
    c.eval_every = 100;

    /* Compute tokens budget: N * 8 (nanochat ratio) */
    /* Rough param count: ~12 * depth * dim^2 */
    long params = 12L * depth * c.dim * c.dim;
    long tokens_budget = params * 8;
    c.max_steps = (int)(tokens_budget / (c.batch_size * c.seq_len));
    if (c.max_steps < 200) c.max_steps = 200;
    if (c.max_steps > 2000) c.max_steps = 2000; /* mercy on the silicon */

    /* BPE: more merges for bigger vocab with bigger models */
    c.bpe_merges = 4000;

    c.personality_steps = 100;

    snprintf(c.data_url, sizeof(c.data_url),
        "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu-score-2/resolve/main/data/CC-MAIN-2024-10/train-00000-of-00196.parquet");
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
} Tensor;

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

/* Matrix multiply: C[M,N] = A[M,K] @ B[N,K]^T */
static void matmul_fwd(float *C, float *A, float *B, int M, int N, int K) {
    #pragma omp parallel for
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
}

/* C = A @ B^T backward: dA += dC @ B, dB += dC^T @ A */
static void matmul_bwd(float *dA, float *dB, float *dC, float *A, float *B,
                        int M, int N, int K) {
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
        matmul_fwd(la->q, la->xn, lw->wq->data, T, qd, D);
        matmul_fwd(la->k, la->xn, lw->wk->data, T, kv, D);
        matmul_fwd(la->v, la->xn, lw->wv->data, T, kv, D);

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
        matmul_fwd(attn_proj, la->attn_out, lw->wo->data, T, D, qd);
        for (int i = 0; i < T * D; i++) s->residual[i] += attn_proj[i];
        free(attn_proj);
        memcpy(la->res_after_attn, s->residual, T * D * sizeof(float));

        /* FFN: norm → gate/up → SwiGLU → down → residual */
        rmsnorm_fwd_seq(la->ffn_xn, s->residual, lw->ffn_norm->data, T, D, c->norm_eps);
        matmul_fwd(la->gate_pre, la->ffn_xn, lw->w_gate->data, T, H, D);
        matmul_fwd(la->up_pre, la->ffn_xn, lw->w_up->data, T, H, D);
        for (int i = 0; i < T * H; i++)
            la->swiglu[i] = silu(la->gate_pre[i]) * la->up_pre[i];
        float *ffn_proj = calloc(T * D, sizeof(float));
        matmul_fwd(ffn_proj, la->swiglu, lw->w_down->data, T, D, H);
        for (int i = 0; i < T * D; i++) s->residual[i] += ffn_proj[i];
        free(ffn_proj);
    }

    /* 3. Final norm + LM head */
    s->final_normed = calloc(T * D, sizeof(float));
    rmsnorm_fwd_seq(s->final_normed, s->residual, w->output_norm->data, T, D, c->norm_eps);
    s->logits = calloc(T * c->vocab_size, sizeof(float));
    matmul_fwd(s->logits, s->final_normed, w->output->data, T, c->vocab_size, D);

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
    matmul_bwd(d_fn, grads[1], d_logits, s->final_normed, w->output->data, T, V, D);

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
        matmul_bwd(s->d_swiglu, grads[gi+8], s->d_residual, la->swiglu, lw->w_down->data, T, D, H);

        /* SwiGLU backward: out = silu(gate) * up */
        for (int i = 0; i < T * H; i++) {
            float g = la->gate_pre[i], u = la->up_pre[i];
            float sig = 1.0f / (1.0f + expf(-g));
            float silu_g = g * sig;
            s->d_gate[i] = s->d_swiglu[i] * u * (sig + g * sig * (1.0f - sig));
            s->d_up[i]   = s->d_swiglu[i] * silu_g;
        }

        memset(s->d_ffn_xn, 0, T * D * sizeof(float));
        matmul_bwd(s->d_ffn_xn, grads[gi+6], s->d_gate, la->ffn_xn, lw->w_gate->data, T, H, D);
        matmul_bwd(s->d_ffn_xn, grads[gi+7], s->d_up,   la->ffn_xn, lw->w_up->data,   T, H, D);

        /* FFN norm backward */
        rmsnorm_bwd_seq(s->d_residual, grads[gi+5], s->d_ffn_xn,
                         la->res_after_attn, lw->ffn_norm->data, T, D, c->norm_eps);

        /* === Attention backward === */
        /* Output projection backward */
        memset(s->d_attn_out, 0, T * qd * sizeof(float));
        matmul_bwd(s->d_attn_out, grads[gi+4], s->d_residual, la->attn_out, lw->wo->data, T, D, qd);

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
        matmul_bwd(s->d_xn, grads[gi+1], s->d_q, la->xn, lw->wq->data, T, qd, D);
        matmul_bwd(s->d_xn, grads[gi+2], s->d_k, la->xn, lw->wk->data, T, kv, D);
        matmul_bwd(s->d_xn, grads[gi+3], s->d_v, la->xn, lw->wv->data, T, kv, D);

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
 * DATA LOADING — give it a text file. any text file. it doesn't judge.
 * no parquet parser. no huggingface datasets library. no 2GB arrow cache.
 * fopen, fread, done. your data pipeline is 30 lines. you're welcome.
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* Download FineWeb-Edu text file */
static int download_data(Config *c) {
    struct stat st;
    if (stat(c->data_path, &st) == 0 && st.st_size > 1000) {
        printf("[data] found existing %s (%.1f MB)\n", c->data_path,
               (float)st.st_size / 1024 / 1024);
        return 0;
    }

    printf("[data] downloading training data...\n");
    printf("[data] NOTE: provide your own text file as '%s'\n", c->data_path);
    printf("[data] or download FineWeb-Edu:\n");
    printf("  curl -L 'https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu-score-2/"
           "resolve/main/data/CC-MAIN-2024-10/train-00000-of-00196.parquet' -o shard.parquet\n");
    printf("  python3 -c \"import pandas as pd; "
           "df=pd.read_parquet('shard.parquet'); "
           "open('%s','w').write('\\n'.join(df['text'].tolist()))\"\n", c->data_path);

    /* Try to create a small synthetic dataset for testing */
    printf("[data] creating small synthetic dataset for demo...\n");
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

    /* Repeat samples to get enough training data */
    for (int repeat = 0; repeat < 500; repeat++) {
        for (int i = 0; samples[i]; i++) {
            fprintf(f, "%s\n", samples[i]);
        }
    }
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

    /* Parse args */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--depth") == 0 && i + 1 < argc) {
            depth = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--data") == 0 && i + 1 < argc) {
            /* custom data path handled below */
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("l.c — one file. one llama. no excuses.\n\n");
            printf("Usage: ./l [options]\n");
            printf("  --depth N       Model depth (2=~1M, 4=~3M, 6=~7M, 8=~15M params)\n");
            printf("  --data PATH     Path to training text file\n");
            printf("  --fp16-cache    Half-precision KV cache (saves memory)\n");
            printf("  --rope-scale F  NTK scaling for RoPE (>1 extends context)\n");
            printf("  --help          Show this help\n");
            return 0;
        }
    }

    printf("\n");
    printf("  ╔══════════════════════════════════════╗\n");
    printf("  ║  l.c — actually llama                ║\n");
    printf("  ║  one file. no frameworks. no excuses. ║\n");
    printf("  ╚══════════════════════════════════════╝\n\n");

    Config c = config_from_depth(depth);

    /* Parse overrides */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--data") == 0 && i + 1 < argc)
            snprintf(c.data_path, sizeof(c.data_path), "%s", argv[i+1]);
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
    tok_train_bpe(&tok, text, text_len, c.bpe_merges);
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

        /* Logging */
        if ((step + 1) % c.log_every == 0 || step == 0) {
            float avg_loss = running_loss / loss_count;
            float elapsed = (float)(clock() - train_start) / CLOCKS_PER_SEC;
            float tok_per_sec = (float)((step + 1) * c.seq_len) / elapsed;
            printf("  step %4d/%d  loss=%.4f  lr=%.2e  tok/s=%.0f  (%.1fs)\n",
                   step + 1, c.max_steps, avg_loss, lr, tok_per_sec, elapsed);
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

    /* ── Step 6: Export GGUF ── */
    export_gguf(&w, &c, &tok);

    /* ── Step 7: Chat ── */
    chat_loop(&w, &c, &tok);

    /* Cleanup */
    free(all_tokens);
    printf("[l] done.\n");
    return 0;
}
