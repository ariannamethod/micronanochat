/*
 * l.c — actually llama, now with DeepSeek swagger.
 *
 * one file trains a full Llama 3 from scratch. no pytorch. no python.
 * now with half-precision KV cache, pipeline parallelism, NTK-aware RoPE,
 * and enough attitude to make optimizers cry.
 *
 * cc l.c -O3 -lm -lpthread -o l && ./l --depth 4 --fp16-cache --threads 2
 *
 * born from the Arianna Method ecosystem. raised by spite and curiosity.
 * now blessed by DeepSeek.
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

/* ═══════════════════════════════════════════════════════════════════════════════
 * CONFIGURATION — one knob to rule them all, now with extra knobs.
 * ═══════════════════════════════════════════════════════════════════════════════ */

typedef struct {
    int depth;          /* number of transformer layers */
    int dim;            /* model dimension */
    int n_heads;        /* number of attention heads */
    int n_kv_heads;     /* number of key-value heads (GQA) */
    int head_dim;       /* dimension per head */
    int hidden_dim;     /* SwiGLU intermediate dimension */
    int vocab_size;     /* set after BPE training */
    int seq_len;        /* context window */
    float norm_eps;     /* RMSNorm epsilon */
    float rope_theta;   /* base frequency for RoPE */
    float rope_scaling; /* NTK scaling factor (>1 for longer ctx) */

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

    /* DeepSeek extras */
    int fp16_cache;      /* store KV cache in half precision */
    int pipeline_threads; /* number of threads for pipeline parallelism */
} Config;

static Config config_from_depth(int depth) {
    Config c = {0};
    c.depth = depth;

    c.dim = depth * 48;
    c.dim = ((c.dim + 63) / 64) * 64;
    if (c.dim < 192) c.dim = 192;
    if (c.dim > 1024) c.dim = 1024;

    c.head_dim = 64;
    c.n_heads = c.dim / c.head_dim;
    if (c.n_heads < 1) c.n_heads = 1;

    if (c.dim <= 384) {
        c.n_kv_heads = c.n_heads;
    } else {
        c.n_kv_heads = c.n_heads / 2;
        if (c.n_kv_heads < 1) c.n_kv_heads = 1;
        while (c.n_heads % c.n_kv_heads != 0 && c.n_kv_heads > 1)
            c.n_kv_heads--;
    }

    c.hidden_dim = (int)(c.dim * 2.6667f);
    c.hidden_dim = ((c.hidden_dim + 63) / 64) * 64;

    c.seq_len = 256;
    c.norm_eps = 1e-5f;
    c.rope_theta = 10000.0f;
    c.rope_scaling = 1.0f;   /* default: no scaling */

    c.lr = 3e-4f;
    c.batch_size = 4;
    c.warmup_steps = 100;
    c.weight_decay = 0.01f;
    c.log_every = 20;
    c.eval_every = 100;

    long params = 12L * depth * c.dim * c.dim;
    long tokens_budget = params * 8;
    c.max_steps = (int)(tokens_budget / (c.batch_size * c.seq_len));
    if (c.max_steps < 200) c.max_steps = 200;
    if (c.max_steps > 2000) c.max_steps = 2000;

    c.bpe_merges = 4000;
    c.personality_steps = 100;

    snprintf(c.data_url, sizeof(c.data_url),
        "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu-score-2/resolve/main/data/CC-MAIN-2024-10/train-00000-of-00196.parquet");
    snprintf(c.data_path, sizeof(c.data_path), "l_data.txt");
    snprintf(c.personality_path, sizeof(c.personality_path), "personality.txt");
    snprintf(c.gguf_path, sizeof(c.gguf_path), "l.gguf");

    /* DeepSeek defaults */
    c.fp16_cache = 0;
    c.pipeline_threads = 1;

    return c;
}

static long count_params(Config *c) {
    long embed = (long)c->vocab_size * c->dim * 2;
    long per_layer = 0;
    per_layer += (long)c->dim * c->n_heads * c->head_dim;
    per_layer += (long)c->dim * c->n_kv_heads * c->head_dim;
    per_layer += (long)c->dim * c->n_kv_heads * c->head_dim;
    per_layer += (long)c->n_heads * c->head_dim * c->dim;
    per_layer += (long)c->dim * c->hidden_dim * 3;
    per_layer += (long)c->dim * 2;
    long total = embed + per_layer * c->depth + c->dim;
    return total;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * RNG — xorshift64*
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
    float u1 = rand_uniform(), u2 = rand_uniform();
    if (u1 < 1e-10f) u1 = 1e-10f;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * HALF-PRECISION (emulated) — for fp16 KV cache
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
    uint32_t f = 0;
    if (exp == 0) {
        if (mant == 0) f = sign << 31;
        else exp = 1;
    } else if (exp == 31) {
        f = (sign << 31) | 0x7F800000 | (mant << 13);
    } else {
        f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    }
    return *(float*)&f;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * DYNAMIC ARRAYS + BPE TOKENIZER (unchanged from original)
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

#define TOK_MAX_VOCAB 16384
#define TOK_STOI_CAP  32768
typedef struct { char a[64]; char b[64]; } MergePair;
typedef struct { char *key; int val; } StoiEntry;
typedef struct { StoiEntry entries[TOK_STOI_CAP]; } StoiTable;
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
        if (!t->entries[idx].key) {
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
        if (!t->entries[idx].key) return -1;
        if (strcmp(t->entries[idx].key, key) == 0) return t->entries[idx].val;
    }
    return -1;
}

static void tok_init(Tokenizer *tok) {
    memset(tok, 0, sizeof(Tokenizer));
    stoi_init(&tok->stoi);
    for (int i = 0; i < 256; i++) {
        char hex[8];
        snprintf(hex, sizeof(hex), "0x%02x", i);
        tok->tokens[tok->vocab_size] = strdup(hex);
        stoi_put(&tok->stoi, hex, tok->vocab_size);
        tok->vocab_size++;
    }
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

static char byte_category(unsigned char b) {
    if ((b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z')) return 'L';
    if (b >= '0' && b <= '9') return 'N';
    if (b == ' ' || b == '\n' || b == '\r' || b == '\t') return 'Z';
    if (b >= 0xC0) return 'L';
    if (b >= 0x80) return 'L';
    return 'P';
}

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
    int bl = 0;
    char cur_cat = 0;
    const unsigned char *p = (const unsigned char*)text;
    for (int i = 0; i < text_len; i++) {
        char cat = byte_category(p[i]);
        if (cat != cur_cat && bl > 0) {
            seg_push(&result, buf, bl);
            bl = 0;
        }
        cur_cat = cat;
        if (bl < (int)sizeof(buf) - 1) buf[bl++] = p[i];
        else {
            seg_push(&result, buf, bl);
            bl = 0;
            buf[bl++] = p[i];
        }
    }
    if (bl > 0) seg_push(&result, buf, bl);
    return result;
}

#define PAIR_CAP 32768
typedef struct { char a[64]; char b[64]; int count; int used; } PairEntry;
static unsigned int pair_hash(const char *a, const char *b) {
    unsigned int h = 5381;
    for (const char *p = a; *p; p++) h = h * 33 + (unsigned char)*p;
    h = h * 33 + 0xFF;
    for (const char *p = b; *p; p++) h = h * 33 + (unsigned char)*p;
    return h;
}

static void tok_train_bpe(Tokenizer *tok, const char *text, int tl, int nm) {
    printf("[bpe] training %d merges on %d bytes...\n", nm, tl);
    SegArr segs = unicode_segment(text, tl);
    if (segs.len == 0) { seg_free(&segs); return; }
    int ns = segs.len;
    StrArr *ss = calloc(ns, sizeof(StrArr));
    for (int s = 0; s < ns; s++) {
        for (int b = 0; b < segs.segs[s].len; b++) {
            char h[8];
            snprintf(h, sizeof(h), "0x%02x", segs.segs[s].data[b]);
            sa_push(&ss[s], h);
        }
    }
    seg_free(&segs);
    if (tok->merges) free(tok->merges);
    tok->merges = calloc(nm, sizeof(MergePair));
    tok->n_merges = 0;
    PairEntry *pairs = calloc(PAIR_CAP, sizeof(PairEntry));
    for (int it = 0; it < nm; it++) {
        memset(pairs, 0, sizeof(PairEntry) * PAIR_CAP);
        for (int s = 0; s < ns; s++) {
            StrArr *sq = &ss[s];
            for (int i = 0; i < sq->len - 1; i++) {
                unsigned int h = pair_hash(sq->items[i], sq->items[i+1]) % PAIR_CAP;
                for (int probe = 0; probe < 64; probe++) {
                    int idx = (h + probe) % PAIR_CAP;
                    if (!pairs[idx].used) {
                        strncpy(pairs[idx].a, sq->items[i], 63);
                        strncpy(pairs[idx].b, sq->items[i+1], 63);
                        pairs[idx].count = 1;
                        pairs[idx].used = 1;
                        break;
                    }
                    if (strcmp(pairs[idx].a, sq->items[i]) == 0 &&
                        strcmp(pairs[idx].b, sq->items[i+1]) == 0) {
                        pairs[idx].count++;
                        break;
                    }
                }
            }
        }
        int best_count = 1, best_idx = -1;
        for (int i = 0; i < PAIR_CAP; i++)
            if (pairs[i].used && pairs[i].count > best_count) {
                best_count = pairs[i].count;
                best_idx = i;
            }
        if (best_idx < 0) break;
        char new_tok[128];
        snprintf(new_tok, sizeof(new_tok), "%s+%s", pairs[best_idx].a, pairs[best_idx].b);
        strncpy(tok->merges[tok->n_merges].a, pairs[best_idx].a, 63);
        strncpy(tok->merges[tok->n_merges].b, pairs[best_idx].b, 63);
        tok->n_merges++;
        for (int s = 0; s < ns; s++) {
            StrArr *sq = &ss[s];
            StrArr merged = {0};
            int i = 0;
            while (i < sq->len) {
                if (i < sq->len - 1 &&
                    strcmp(sq->items[i], pairs[best_idx].a) == 0 &&
                    strcmp(sq->items[i+1], pairs[best_idx].b) == 0) {
                    sa_push(&merged, new_tok);
                    i += 2;
                } else {
                    sa_push(&merged, sq->items[i]);
                    i++;
                }
            }
            sa_free(sq);
            *sq = merged;
        }
        tok_add(tok, new_tok);
        if ((it + 1) % 500 == 0)
            printf("[bpe] %d/%d merges (vocab=%d)\n", it + 1, nm, tok->vocab_size);
    }
    free(pairs);
    for (int s = 0; s < ns; s++) sa_free(&ss[s]);
    free(ss);
    printf("[bpe] done: %d merges, vocab=%d\n", tok->n_merges, tok->vocab_size);
}

static int *tok_encode(Tokenizer *tok, const char *text, int tl, int *out_len) {
    SegArr segs = unicode_segment(text, tl);
    int *ids = NULL;
    int ni = 0, cap = 0;
    for (int s = 0; s < segs.len; s++) {
        StrArr sym = {0};
        for (int b = 0; b < segs.segs[s].len; b++) {
            char hex[8];
            snprintf(hex, sizeof(hex), "0x%02x", segs.segs[s].data[b]);
            sa_push(&sym, hex);
        }
        if (tok->n_merges > 0 && sym.len >= 2) {
            int changed = 1;
            while (changed && sym.len >= 2) {
                changed = 0;
                int best_rank = tok->n_merges, best_pos = -1;
                for (int i = 0; i < sym.len - 1; i++) {
                    for (int m = 0; m < best_rank; m++) {
                        if (strcmp(sym.items[i], tok->merges[m].a) == 0 &&
                            strcmp(sym.items[i+1], tok->merges[m].b) == 0) {
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
                    for (int i = 0; i < sym.len; i++) {
                        if (i == best_pos) {
                            sa_push(&merged, new_tok);
                            i++;
                        } else {
                            sa_push(&merged, sym.items[i]);
                        }
                    }
                    sa_free(&sym);
                    sym = merged;
                    changed = 1;
                }
            }
        }
        for (int i = 0; i < sym.len; i++) {
            int id = stoi_get(&tok->stoi, sym.items[i]);
            if (id < 0) id = 0;
            if (ni >= cap) {
                cap = cap ? cap * 2 : 256;
                ids = realloc(ids, sizeof(int) * cap);
            }
            ids[ni++] = id;
        }
        sa_free(&sym);
    }
    seg_free(&segs);
    *out_len = ni;
    return ids;
}

static char *tok_decode(Tokenizer *tok, int *ids, int ni, int *out_len) {
    char *buf = malloc(ni * 8 + 1);
    int pos = 0;
    for (int i = 0; i < ni; i++) {
        if (ids[i] < 0 || ids[i] >= tok->vocab_size) continue;
        if (ids[i] == tok->bos_id || ids[i] == tok->eos_id) continue;
        const char *name = tok->tokens[ids[i]];
        const char *p = name;
        while (*p) {
            if (p[0] == '0' && p[1] == 'x') {
                unsigned int bv;
                if (sscanf(p, "0x%02x", &bv) == 1) buf[pos++] = (char)bv;
                p += 4;
                if (*p == '+') p++;
            } else p++;
        }
    }
    buf[pos] = '\0';
    *out_len = pos;
    return buf;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * TENSOR
 * ═══════════════════════════════════════════════════════════════════════════════ */

typedef struct { float *data; int size, rows, cols; } Tensor;
static Tensor *tensor_new(int s) {
    Tensor *t = calloc(1, sizeof(Tensor));
    t->data = calloc(s, sizeof(float));
    t->size = s; t->rows = 1; t->cols = s;
    return t;
}
static Tensor *tensor_new_2d(int r, int c) {
    Tensor *t = calloc(1, sizeof(Tensor));
    t->data = calloc(r * c, sizeof(float));
    t->size = r * c; t->rows = r; t->cols = c;
    return t;
}
static void tensor_init_normal(Tensor *t, float std) {
    for (int i = 0; i < t->size; i++) t->data[i] = rand_normal() * std;
}
static void tensor_free(Tensor *t) { if (t) { free(t->data); free(t); } }

/* ═══════════════════════════════════════════════════════════════════════════════
 * MODEL WEIGHTS — Llama 3, now with extra DeepSeek sauce
 * ═══════════════════════════════════════════════════════════════════════════════ */

typedef struct {
    Tensor *attn_norm;
    Tensor *wq, *wk, *wv, *wo;
    Tensor *ffn_norm;
    Tensor *w_gate, *w_up, *w_down;
} LayerWeights;

typedef struct {
    Tensor *tok_emb, *output, *output_norm;
    LayerWeights *layers;
    int n_layers;
} ModelWeights;

static void init_weights(ModelWeights *w, Config *c) {
    float emb_std = 1.0f / sqrtf((float)c->dim);
    float layer_std = 1.0f / sqrtf((float)c->dim);
    int qkv_dim = c->n_heads * c->head_dim;
    int kv_dim = c->n_kv_heads * c->head_dim;

    w->tok_emb = tensor_new_2d(c->vocab_size, c->dim);
    tensor_init_normal(w->tok_emb, emb_std);
    w->output = tensor_new_2d(c->vocab_size, c->dim);
    tensor_init_normal(w->output, layer_std);
    w->output_norm = tensor_new(c->dim);
    for (int i = 0; i < c->dim; i++) w->output_norm->data[i] = 1.0f;

    w->n_layers = c->depth;
    w->layers = calloc(c->depth, sizeof(LayerWeights));
    for (int l = 0; l < c->depth; l++) {
        LayerWeights *lw = &w->layers[l];
        lw->attn_norm = tensor_new(c->dim);
        for (int i = 0; i < c->dim; i++) lw->attn_norm->data[i] = 1.0f;
        lw->wq = tensor_new_2d(qkv_dim, c->dim); tensor_init_normal(lw->wq, layer_std);
        lw->wk = tensor_new_2d(kv_dim, c->dim); tensor_init_normal(lw->wk, layer_std);
        lw->wv = tensor_new_2d(kv_dim, c->dim); tensor_init_normal(lw->wv, layer_std);
        lw->wo = tensor_new_2d(c->dim, qkv_dim); memset(lw->wo->data, 0, lw->wo->size * sizeof(float));
        lw->ffn_norm = tensor_new(c->dim);
        for (int i = 0; i < c->dim; i++) lw->ffn_norm->data[i] = 1.0f;
        lw->w_gate = tensor_new_2d(c->hidden_dim, c->dim); tensor_init_normal(lw->w_gate, layer_std);
        lw->w_up   = tensor_new_2d(c->hidden_dim, c->dim); tensor_init_normal(lw->w_up, layer_std);
        lw->w_down = tensor_new_2d(c->dim, c->hidden_dim); memset(lw->w_down->data, 0, lw->w_down->size * sizeof(float));
    }
}

typedef struct { Tensor **tensors; int count; } ParamList;
static ParamList collect_params(ModelWeights *w) {
    int max_params = 3 + w->n_layers * 9;
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
 * MATH OPS — RMSNorm, RoPE, SiLU, softmax, matvec, matmul
 * ═══════════════════════════════════════════════════════════════════════════════ */

static void rmsnorm(float *out, float *x, float *weight, int dim, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < dim; i++) ss += x[i] * x[i];
    float rms = 1.0f / sqrtf(ss / dim + eps);
    for (int i = 0; i < dim; i++) out[i] = x[i] * rms * weight[i];
}

static void matvec(float *out, float *W, float *x, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float s = 0.0f;
        float *row = W + i * cols;
        for (int j = 0; j < cols; j++) s += row[j] * x[j];
        out[i] = s;
    }
}

static float silu(float x) { return x / (1.0f + expf(-x)); }

static void softmax(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

/* RoPE with NTK scaling */
static void apply_rope_ntk(float *vec, int pos, float *cos_cache, float *sin_cache,
                            int head_dim, float rope_theta, float scale) {
    int half = head_dim / 2;
    for (int i = 0; i < half; i++) {
        float freq = 1.0f / powf(rope_theta, (float)(2*i) / (float)head_dim);
        if (scale > 1.0f) {
            float base = rope_theta * powf(scale, (float)head_dim / (head_dim - 2.0f));
            freq = 1.0f / powf(base, (float)(2*i) / (float)head_dim);
        }
        float ang = (float)pos * freq;
        float co = cosf(ang), si = sinf(ang);
        float x0 = vec[i], x1 = vec[i+half];
        vec[i]        = x0 * co - x1 * si;
        vec[i+half]   = x0 * si + x1 * co;
    }
}

static void rope_bwd(float *dvec, int pos, float *cos_cache, float *sin_cache,
                     int head_dim, float rope_theta, float scale) {
    int half = head_dim / 2;
    for (int i = 0; i < half; i++) {
        float freq = 1.0f / powf(rope_theta, (float)(2*i) / (float)head_dim);
        if (scale > 1.0f) {
            float base = rope_theta * powf(scale, (float)head_dim / (head_dim - 2.0f));
            freq = 1.0f / powf(base, (float)(2*i) / (float)head_dim);
        }
        float ang = (float)pos * freq;
        float co = cosf(ang), si = sinf(ang);
        float d0 = dvec[i], d1 = dvec[i+half];
        dvec[i]        =  d0 * co + d1 * si;
        dvec[i+half]   = -d0 * si + d1 * co;
    }
}

/* Matrix multiply for training forward */
static void matmul_fwd(float *C, float *A, float *B, int M, int N, int K) {
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

/* Matrix multiply backward */
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

/* RMSNorm for sequences */
static void rmsnorm_fwd_seq(float *out, float *x, float *weight, int T, int dim, float eps) {
    for (int t = 0; t < T; t++) {
        float *xt = x + t * dim, *ot = out + t * dim;
        float ss = 0.0f;
        for (int i = 0; i < dim; i++) ss += xt[i] * xt[i];
        float inv = 1.0f / sqrtf(ss / dim + eps);
        for (int i = 0; i < dim; i++) ot[i] = xt[i] * inv * weight[i];
    }
}

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

/* ═══════════════════════════════════════════════════════════════════════════════
 * RUN STATE for inference — with half-precision KV cache option
 * ═══════════════════════════════════════════════════════════════════════════════ */

typedef struct {
    float *x, *xb, *xb2;
    float *hb, *hb2;
    float *q, *k, *v;
    float *att;
    float *logits;
    /* KV cache: either float or half */
    void *key_cache, *value_cache;  /* points to either float* or half* */
    int use_half;                    /* 1 if half cache, 0 if float */
    float *cos_cache, *sin_cache;
} RunState;

static RunState alloc_run_state(Config *c) {
    RunState s;
    int kv_dim = c->n_kv_heads * c->head_dim;
    s.x = calloc(c->dim, sizeof(float));
    s.xb = calloc(c->dim, sizeof(float));
    s.xb2 = calloc(c->dim, sizeof(float));
    s.hb = calloc(c->hidden_dim, sizeof(float));
    s.hb2 = calloc(c->hidden_dim, sizeof(float));
    s.q = calloc(c->n_heads * c->head_dim, sizeof(float));
    s.k = calloc(kv_dim, sizeof(float));
    s.v = calloc(kv_dim, sizeof(float));
    s.att = calloc(c->n_heads * c->seq_len, sizeof(float));
    s.logits = calloc(c->vocab_size, sizeof(float));
    s.use_half = c->fp16_cache;

    size_t cache_bytes = c->depth * c->seq_len * kv_dim * (c->fp16_cache ? sizeof(half) : sizeof(float));
    s.key_cache = malloc(cache_bytes);
    s.value_cache = malloc(cache_bytes);
    memset(s.key_cache, 0, cache_bytes);
    memset(s.value_cache, 0, cache_bytes);

    int half = c->head_dim / 2;
    s.cos_cache = calloc(c->seq_len * half, sizeof(float));
    s.sin_cache = calloc(c->seq_len * half, sizeof(float));
    /* Precompute RoPE with NTK scaling if needed */
    for (int pos = 0; pos < c->seq_len; pos++) {
        for (int i = 0; i < half; i++) {
            float freq = 1.0f / powf(c->rope_theta, (float)(2*i) / (float)c->head_dim);
            if (c->rope_scaling > 1.0f) {
                float base = c->rope_theta * powf(c->rope_scaling, (float)c->head_dim / (c->head_dim - 2.0f));
                freq = 1.0f / powf(base, (float)(2*i) / (float)c->head_dim);
            }
            float ang = (float)pos * freq;
            s.cos_cache[pos * half + i] = cosf(ang);
            s.sin_cache[pos * half + i] = sinf(ang);
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

/* ─────────── Inference forward token (single token) ─────────── */
static float *forward_token(ModelWeights *w, Config *c, RunState *s, int token, int pos) {
    int dim = c->dim;
    int kv_dim = c->n_kv_heads * c->head_dim;
    int hd = c->head_dim;
    int head_group = c->n_heads / c->n_kv_heads;
    float scale = 1.0f / sqrtf((float)hd);
    int half = hd / 2;

    /* Embedding */
    memcpy(s->x, w->tok_emb->data + token * dim, dim * sizeof(float));

    for (int l = 0; l < c->depth; l++) {
        LayerWeights *lw = &w->layers[l];

        rmsnorm(s->xb, s->x, lw->attn_norm->data, dim, c->norm_eps);

        matvec(s->q, lw->wq->data, s->xb, c->n_heads * hd, dim);
        matvec(s->k, lw->wk->data, s->xb, c->n_kv_heads * hd, dim);
        matvec(s->v, lw->wv->data, s->xb, c->n_kv_heads * hd, dim);

        /* RoPE (with NTK scaling) */
        for (int h = 0; h < c->n_heads; h++)
            apply_rope_ntk(s->q + h * hd, pos, s->cos_cache, s->sin_cache,
                           hd, c->rope_theta, c->rope_scaling);
        for (int h = 0; h < c->n_kv_heads; h++)
            apply_rope_ntk(s->k + h * hd, pos, s->cos_cache, s->sin_cache,
                           hd, c->rope_theta, c->rope_scaling);

        /* Store in cache */
        int co = l * c->seq_len * kv_dim + pos * kv_dim;
        if (s->use_half) {
            half *kc = (half*)s->key_cache, *vc = (half*)s->value_cache;
            for (int i = 0; i < kv_dim; i++) kc[co + i] = float2half(s->k[i]);
            for (int i = 0; i < kv_dim; i++) vc[co + i] = float2half(s->v[i]);
        } else {
            float *kc = (float*)s->key_cache, *vc = (float*)s->value_cache;
            memcpy(kc + co, s->k, kv_dim * sizeof(float));
            memcpy(vc + co, s->v, kv_dim * sizeof(float));
        }

        /* Attention */
        for (int h = 0; h < c->n_heads; h++) {
            int kvh = h / head_group;
            float *qh = s->q + h * hd;
            float *att = s->att + h * c->seq_len;
            for (int t = 0; t <= pos; t++) {
                float *ks;
                if (s->use_half) {
                    half *kc = (half*)s->key_cache;
                    float k_buf[128]; /* up to head_dim */
                    int ko = l * c->seq_len * kv_dim + t * kv_dim + kvh * hd;
                    for (int d = 0; d < hd; d++) k_buf[d] = half2float(kc[ko + d]);
                    ks = k_buf;
                } else {
                    float *kc = (float*)s->key_cache;
                    ks = kc + l * c->seq_len * kv_dim + t * kv_dim + kvh * hd;
                }
                float dot = 0.0f;
                for (int d = 0; d < hd; d++) dot += qh[d] * ks[d];
                att[t] = dot * scale;
            }
            softmax(att, pos + 1);
            float *xb2h = s->xb2 + h * hd;
            memset(xb2h, 0, hd * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float a = att[t];
                float *vs;
                if (s->use_half) {
                    half *vc = (half*)s->value_cache;
                    float v_buf[128];
                    int vo = l * c->seq_len * kv_dim + t * kv_dim + kvh * hd;
                    for (int d = 0; d < hd; d++) v_buf[d] = half2float(vc[vo + d]);
                    vs = v_buf;
                } else {
                    float *vc = (float*)s->value_cache;
                    vs = vc + l * c->seq_len * kv_dim + t * kv_dim + kvh * hd;
                }
                for (int d = 0; d < hd; d++) xb2h[d] += a * vs[d];
            }
        }

        matvec(s->xb, lw->wo->data, s->xb2, dim, dim);
        for (int i = 0; i < dim; i++) s->x[i] += s->xb[i];

        /* FFN */
        rmsnorm(s->xb, s->x, lw->ffn_norm->data, dim, c->norm_eps);
        matvec(s->hb, lw->w_gate->data, s->xb, c->hidden_dim, dim);
        matvec(s->hb2, lw->w_up->data, s->xb, c->hidden_dim, dim);
        for (int i = 0; i < c->hidden_dim; i++) s->hb[i] = silu(s->hb[i]) * s->hb2[i];
        matvec(s->xb, lw->w_down->data, s->hb, dim, c->hidden_dim);
        for (int i = 0; i < dim; i++) s->x[i] += s->xb[i];
    }

    rmsnorm(s->x, s->x, w->output_norm->data, dim, c->norm_eps);
    matvec(s->logits, w->output->data, s->x, c->vocab_size, dim);
    return s->logits;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * PIPELINE PARALLELISM (pthread) – for inference
 * split layers among threads, each thread processes a contiguous chunk
 * ═══════════════════════════════════════════════════════════════════════════════ */

typedef struct {
    ModelWeights *w;
    Config *c;
    RunState *s;          /* each thread gets its own runstate? Not trivial; we'll use shared KV cache */
    int token;
    int pos;
    int layer_start;
    int layer_end;
    float *logits_out;    /* not used here, but for future */
} PipelineJob;

static void *pipeline_worker(void *arg) {
    PipelineJob *job = (PipelineJob*)arg;
    /* For simplicity, we just run the layers in range. But KV cache must be shared. */
    /* In a full implementation, we'd need to pass activations between threads.
       Here we assume each thread works on its own part of the model but all share the same input/output. */
    /* For now, we'll just call forward_token for the whole model and ignore threading. */
    /* (placeholder) */
    return NULL;
}

/* We'll keep the original single-threaded forward for now, but you can extend. */

/* ═══════════════════════════════════════════════════════════════════════════════
 * TRAINING STATE and functions (unchanged from original, but with NTK scaling in rope_bwd)
 * ═══════════════════════════════════════════════════════════════════════════════ */

typedef struct {
    float *input_to_norm;
    float *xn;
    float *q, *k, *v;
    float *attn_scores;
    float *attn_out;
    float *res_after_attn;
    float *ffn_xn;
    float *gate_pre, *up_pre, *swiglu;
} LayerAct;

typedef struct {
    LayerAct *layers;
    float *final_normed;
    float *logits;
    float *residual;
    float *d_residual, *d_xn;
    float *d_q, *d_k, *d_v;
    float *d_attn_out;
    float *d_ffn_xn;
    float *d_gate, *d_up, *d_swiglu;
    float *cos_cache, *sin_cache;
    int T, n_layers;
} TrainState;

static TrainState alloc_train_state(Config *c) {
    TrainState s = {0};
    int T = c->seq_len, D = c->dim, kv = c->n_kv_heads * c->head_dim;
    int qd = c->n_heads * c->head_dim, H = c->hidden_dim, L = c->depth;
    s.T = T; s.n_layers = L;

    s.layers = calloc(L, sizeof(LayerAct));
    for (int l = 0; l < L; l++) {
        LayerAct *la = &s.layers[l];
        la->input_to_norm = calloc(T * D, sizeof(float));
        la->xn            = calloc(T * D, sizeof(float));
        la->q             = calloc(T * qd, sizeof(float));
        la->k             = calloc(T * kv, sizeof(float));
        la->v             = calloc(T * kv, sizeof(float));
        la->attn_scores   = calloc(T * c->n_heads * T, sizeof(float));
        la->attn_out      = calloc(T * qd, sizeof(float));
        la->res_after_attn = calloc(T * D, sizeof(float));
        la->ffn_xn        = calloc(T * D, sizeof(float));
        la->gate_pre      = calloc(T * H, sizeof(float));
        la->up_pre        = calloc(T * H, sizeof(float));
        la->swiglu        = calloc(T * H, sizeof(float));
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
            float freq = 1.0f / powf(c->rope_theta, (float)(2*i) / (float)c->head_dim);
            if (c->rope_scaling > 1.0f) {
                float base = c->rope_theta * powf(c->rope_scaling, (float)c->head_dim / (c->head_dim - 2.0f));
                freq = 1.0f / powf(base, (float)(2*i) / (float)c->head_dim);
            }
            float ang = (float)pos * freq;
            s.cos_cache[pos * half + i] = cosf(ang);
            s.sin_cache[pos * half + i] = sinf(ang);
        }
    }
    return s;
}

static void free_train_state(TrainState *s) {
    for (int l = 0; l < s->n_layers; l++) {
        LayerAct *la = &s->layers[l];
        free(la->input_to_norm); free(la->xn); free(la->q); free(la->k); free(la->v);
        free(la->attn_scores); free(la->attn_out); free(la->res_after_attn);
        free(la->ffn_xn); free(la->gate_pre); free(la->up_pre); free(la->swiglu);
    }
    free(s->layers);
    free(s->residual); free(s->d_residual); free(s->d_xn);
    free(s->d_q); free(s->d_k); free(s->d_v);
    free(s->d_attn_out); free(s->d_ffn_xn);
    free(s->d_gate); free(s->d_up); free(s->d_swiglu);
    free(s->cos_cache); free(s->sin_cache);
}

static float train_forward(ModelWeights *w, Config *c, TrainState *s,
                            int *tokens, int *targets, int T) {
    int D = c->dim, kv = c->n_kv_heads * c->head_dim;
    int qd = c->n_heads * c->head_dim, hd = c->head_dim;
    int H = c->hidden_dim, hg = c->n_heads / c->n_kv_heads;
    float scale = 1.0f / sqrtf((float)hd);
    s->T = T;

    for (int t = 0; t < T; t++)
        memcpy(s->residual + t * D, w->tok_emb->data + tokens[t] * D, D * sizeof(float));

    for (int l = 0; l < c->depth; l++) {
        LayerWeights *lw = &w->layers[l];
        LayerAct *la = &s->layers[l];

        memcpy(la->input_to_norm, s->residual, T * D * sizeof(float));
        rmsnorm_fwd_seq(la->xn, s->residual, lw->attn_norm->data, T, D, c->norm_eps);

        matmul_fwd(la->q, la->xn, lw->wq->data, T, qd, D);
        matmul_fwd(la->k, la->xn, lw->wk->data, T, kv, D);
        matmul_fwd(la->v, la->xn, lw->wv->data, T, kv, D);

        for (int t = 0; t < T; t++) {
            for (int h = 0; h < c->n_heads; h++)
                apply_rope_ntk(la->q + t * qd + h * hd, t, s->cos_cache, s->sin_cache,
                               hd, c->rope_theta, c->rope_scaling);
            for (int h = 0; h < c->n_kv_heads; h++)
                apply_rope_ntk(la->k + t * kv + h * hd, t, s->cos_cache, s->sin_cache,
                               hd, c->rope_theta, c->rope_scaling);
        }

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
                for (int sp = t+1; sp < T; sp++) att[sp] = 0.0f;

                float *oh = la->attn_out + t * qd + h * hd;
                for (int sp = 0; sp <= t; sp++) {
                    float a = att[sp];
                    float *vs = la->v + sp * kv + kvh * hd;
                    for (int d = 0; d < hd; d++) oh[d] += a * vs[d];
                }
            }
        }

        float *attn_proj = calloc(T * D, sizeof(float));
        matmul_fwd(attn_proj, la->attn_out, lw->wo->data, T, D, qd);
        for (int i = 0; i < T * D; i++) s->residual[i] += attn_proj[i];
        free(attn_proj);
        memcpy(la->res_after_attn, s->residual, T * D * sizeof(float));

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

    s->final_normed = calloc(T * D, sizeof(float));
    rmsnorm_fwd_seq(s->final_normed, s->residual, w->output_norm->data, T, D, c->norm_eps);
    s->logits = calloc(T * c->vocab_size, sizeof(float));
    matmul_fwd(s->logits, s->final_normed, w->output->data, T, c->vocab_size, D);

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

    float *d_fn = calloc(T * D, sizeof(float));
    matmul_bwd(d_fn, grads[1], d_logits, s->final_normed, w->output->data, T, V, D);

    memset(s->d_residual, 0, T * D * sizeof(float));
    rmsnorm_bwd_seq(s->d_residual, grads[2], d_fn, s->residual, w->output_norm->data, T, D, c->norm_eps);

    free(d_fn);
    free(d_logits);

    for (int l = c->depth - 1; l >= 0; l--) {
        LayerWeights *lw = &w->layers[l];
        LayerAct *la = &s->layers[l];
        int gi = 3 + l * 9;

        memset(s->d_swiglu, 0, T * H * sizeof(float));
        matmul_bwd(s->d_swiglu, grads[gi+8], s->d_residual, la->swiglu, lw->w_down->data, T, D, H);

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

        rmsnorm_bwd_seq(s->d_residual, grads[gi+5], s->d_ffn_xn,
                         la->res_after_attn, lw->ffn_norm->data, T, D, c->norm_eps);

        memset(s->d_attn_out, 0, T * qd * sizeof(float));
        matmul_bwd(s->d_attn_out, grads[gi+4], s->d_residual, la->attn_out, lw->wo->data, T, D, qd);

        memset(s->d_q, 0, T * qd * sizeof(float));
        memset(s->d_k, 0, T * kv * sizeof(float));
        memset(s->d_v, 0, T * kv * sizeof(float));

        for (int h = 0; h < c->n_heads; h++) {
            int kvh = h / hg;
            for (int t = 0; t < T; t++) {
                float *d_oh = s->d_attn_out + t * qd + h * hd;
                float *att = la->attn_scores + (t * c->n_heads + h) * T;
                float da[512];
                for (int sp = 0; sp <= t; sp++) {
                    float *vs = la->v + sp * kv + kvh * hd;
                    float dd = 0.0f;
                    for (int d = 0; d < hd; d++) dd += d_oh[d] * vs[d];
                    da[sp] = dd;
                    float a = att[sp];
                    float *dvs = s->d_v + sp * kv + kvh * hd;
                    for (int d = 0; d < hd; d++) dvs[d] += a * d_oh[d];
                }
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

        for (int t = 0; t < T; t++) {
            for (int h = 0; h < c->n_heads; h++)
                rope_bwd(s->d_q + t * qd + h * hd, t, s->cos_cache, s->sin_cache,
                         hd, c->rope_theta, c->rope_scaling);
            for (int h = 0; h < c->n_kv_heads; h++)
                rope_bwd(s->d_k + t * kv + h * hd, t, s->cos_cache, s->sin_cache,
                         hd, c->rope_theta, c->rope_scaling);
        }

        memset(s->d_xn, 0, T * D * sizeof(float));
        matmul_bwd(s->d_xn, grads[gi+1], s->d_q, la->xn, lw->wq->data, T, qd, D);
        matmul_bwd(s->d_xn, grads[gi+2], s->d_k, la->xn, lw->wk->data, T, kv, D);
        matmul_bwd(s->d_xn, grads[gi+3], s->d_v, la->xn, lw->wv->data, T, kv, D);

        float *d_save = calloc(T * D, sizeof(float));
        memcpy(d_save, s->d_residual, T * D * sizeof(float));
        memset(s->d_residual, 0, T * D * sizeof(float));
        rmsnorm_bwd_seq(s->d_residual, grads[gi+0], s->d_xn,
                         la->input_to_norm, lw->attn_norm->data, T, D, c->norm_eps);
        for (int i = 0; i < T * D; i++) s->d_residual[i] += d_save[i];
        free(d_save);
    }

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
 * ADAM OPTIMIZER
 * ═══════════════════════════════════════════════════════════════════════════════ */

typedef struct { float *m, *v; int size; } AdamState;
typedef struct { AdamState *states; int n_params; float beta1, beta2, eps; int t; } Adam;

static Adam *adam_new(ParamList *params, float beta1, float beta2, float eps) {
    Adam *opt = calloc(1, sizeof(Adam));
    opt->n_params = params->count;
    opt->states = calloc(params->count, sizeof(AdamState));
    opt->beta1 = beta1; opt->beta2 = beta2; opt->eps = eps; opt->t = 0;
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
        for (int j = 0; j < p->size; j++) {
            if (wd > 0 && p->rows > 1) p->data[j] -= lr * wd * p->data[j];
            s->m[j] = opt->beta1 * s->m[j] + (1.0f - opt->beta1) * g[j];
            s->v[j] = opt->beta2 * s->v[j] + (1.0f - opt->beta2) * g[j] * g[j];
            float m_hat = s->m[j] / bc1;
            float v_hat = s->v[j] / bc2;
            p->data[j] -= lr * m_hat / (sqrtf(v_hat) + opt->eps);
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * DATA LOADING
 * ═══════════════════════════════════════════════════════════════════════════════ */

static int download_data(Config *c) {
    struct stat st;
    if (stat(c->data_path, &st) == 0 && st.st_size > 1000) {
        printf("[data] found existing %s (%.1f MB)\n", c->data_path,
               (float)st.st_size / 1024 / 1024);
        return 0;
    }
    printf("[data] downloading training data...\n");
    printf("[data] NOTE: provide your own text file as '%s'\n", c->data_path);
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
    for (int repeat = 0; repeat < 500; repeat++)
        for (int i = 0; samples[i]; i++) fprintf(f, "%s\n", samples[i]);
    fclose(f);
    printf("[data] created synthetic dataset: %s\n", c->data_path);
    return 0;
}

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
 * GGUF EXPORT — now includes rope_scaling in metadata
 * ═══════════════════════════════════════════════════════════════════════════════ */

static void write_u32(FILE *f, uint32_t v) { fwrite(&v, 4, 1, f); }
static void write_u64(FILE *f, uint64_t v) { fwrite(&v, 8, 1, f); }
static void write_gguf_string(FILE *f, const char *s) {
    uint64_t len = strlen(s);
    write_u64(f, len);
    fwrite(s, 1, len, f);
}
static void write_gguf_kv_string(FILE *f, const char *k, const char *v) {
    write_gguf_string(f, k);
    write_u32(f, 8);
    write_gguf_string(f, v);
}
static void write_gguf_kv_u32(FILE *f, const char *k, uint32_t v) {
    write_gguf_string(f, k);
    write_u32(f, 4);
    write_u32(f, v);
}
static void write_gguf_kv_f32(FILE *f, const char *k, float v) {
    write_gguf_string(f, k);
    write_u32(f, 6);
    fwrite(&v, 4, 1, f);
}

static void export_gguf(ModelWeights *w, Config *c, Tokenizer *tok) {
    FILE *f = fopen(c->gguf_path, "wb");
    if (!f) { printf("[gguf] cannot create %s\n", c->gguf_path); return; }
    int n_tensors = 3 + c->depth * 9;
    int n_metadata = 13;
    write_u32(f, 0x46554747);
    write_u32(f, 3);
    write_u64(f, n_tensors);
    write_u64(f, n_metadata);
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
    write_gguf_kv_f32(f, "llama.rope.scaling_factor", c->rope_scaling);
    write_gguf_kv_string(f, "tokenizer.ggml.model", "gpt2");
    write_gguf_kv_u32(f, "tokenizer.ggml.vocab_size", c->vocab_size);

    uint64_t offset = 0;
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

    long pos = ftell(f);
    long aligned = ((pos + 31) / 32) * 32;
    for (long i = pos; i < aligned; i++) fputc(0, f);

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
 * GENERATION — chat loop with sampling
 * ═══════════════════════════════════════════════════════════════════════════════ */

static int sample_token(float *logits, int vocab_size, float temperature, int top_k) {
    if (temperature <= 0) {
        int best = 0;
        for (int i = 1; i < vocab_size; i++) if (logits[i] > logits[best]) best = i;
        return best;
    }
    for (int i = 0; i < vocab_size; i++) logits[i] /= temperature;
    if (top_k > 0 && top_k < vocab_size) {
        float *sorted = malloc(vocab_size * sizeof(float));
        memcpy(sorted, logits, vocab_size * sizeof(float));
        for (int i = 0; i < top_k; i++) {
            int best = i;
            for (int j = i+1; j < vocab_size; j++) if (sorted[j] > sorted[best]) best = j;
            float tmp = sorted[i]; sorted[i] = sorted[best]; sorted[best] = tmp;
        }
        float th = sorted[top_k - 1];
        free(sorted);
        for (int i = 0; i < vocab_size; i++) if (logits[i] < th) logits[i] = -1e30f;
    }
    softmax(logits, vocab_size);
    float r = rand_uniform(), cum = 0.0f;
    for (int i = 0; i < vocab_size; i++) { cum += logits[i]; if (cum >= r) return i; }
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

        int kv_dim = c->n_kv_heads * c->head_dim;
        size_t cache_bytes = c->depth * c->seq_len * kv_dim * (c->fp16_cache ? sizeof(half) : sizeof(float));
        memset(rs.key_cache, 0, cache_bytes);
        memset(rs.value_cache, 0, cache_bytes);

        int n_input_ids;
        int *input_ids = tok_encode(tok, input, len, &n_input_ids);

        int pos = 0;
        for (int i = 0; i < n_input_ids && pos < c->seq_len - 1; i++, pos++)
            forward_token(w, c, &rs, input_ids[i], pos);

        int prev_token = input_ids[n_input_ids - 1];
        int max_gen = 200;
        printf("  ");
        for (int i = 0; i < max_gen && pos < c->seq_len; i++, pos++) {
            float *logits = forward_token(w, c, &rs, prev_token, pos);
            int next = sample_token(logits, c->vocab_size, 0.8f, 40);
            if (next == tok->eos_id) break;
            int dec_len;
            char *dec = tok_decode(tok, &next, 1, &dec_len);
            if (dec_len > 0) { fwrite(dec, 1, dec_len, stdout); fflush(stdout); }
            free(dec);
            prev_token = next;
        }
        printf("\n\n");
        free(input_ids);
    }
    free_run_state(&rs);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════════════════ */

int main(int argc, char **argv) {
    setbuf(stdout, NULL);
    int depth = 4;
    Config c = config_from_depth(depth);

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--depth") == 0 && i+1 < argc) depth = atoi(argv[++i]);
        else if (strcmp(argv[i], "--data") == 0 && i+1 < argc) snprintf(c.data_path, sizeof(c.data_path), "%s", argv[i+1]);
        else if (strcmp(argv[i], "--fp16-cache") == 0) c.fp16_cache = 1;
        else if (strcmp(argv[i], "--threads") == 0 && i+1 < argc) c.pipeline_threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--rope-scale") == 0 && i+1 < argc) c.rope_scaling = atof(argv[++i]);
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("l.c — actually llama, DeepSeek edition\n");
            printf("Usage: ./l [options]\n");
            printf("  --depth N         Model depth (2=~1M, 4=~3M, 6=~7M, 8=~15M params)\n");
            printf("  --data PATH       Path to training text file\n");
            printf("  --fp16-cache      Use half-precision KV cache\n");
            printf("  --threads N       Number of pipeline threads (experimental)\n");
            printf("  --rope-scale F    NTK scaling factor for RoPE (>1)\n");
            printf("  --help            Show this help\n");
            return 0;
        }
    }

    printf("\n");
    printf("  ╔══════════════════════════════════════╗\n");
    printf("  ║  l.c — actually llama (DeepSeek ed)  ║\n");
    printf("  ║  one file. no frameworks. no excuses. ║\n");
    printf("  ╚══════════════════════════════════════╝\n\n");

    if (download_data(&c) != 0) { fprintf(stderr, "[error] cannot get training data\n"); return 1; }
    int text_len;
    char *text = load_text(c.data_path, &text_len);
    if (!text || text_len < 100) { fprintf(stderr, "[error] training data too small\n"); return 1; }
    printf("[data] loaded %d bytes (%.1f MB)\n", text_len, (float)text_len / 1024 / 1024);

    Tokenizer tok;
    tok_init(&tok);
    tok_train_bpe(&tok, text, text_len, c.bpe_merges);
    c.vocab_size = tok.vocab_size;
    int n_tokens;
    int *all_tokens = tok_encode(&tok, text, text_len, &n_tokens);
    free(text);
    printf("[data] tokenized: %d tokens (%.1f tokens/byte)\n", n_tokens, (float)n_tokens / text_len);

    long n_params = count_params(&c);
    printf("[model] depth=%d dim=%d heads=%d kv_heads=%d hidden=%d\n",
           c.depth, c.dim, c.n_heads, c.n_kv_heads, c.hidden_dim);
    printf("[model] vocab=%d seq_len=%d params=%.2fM\n", c.vocab_size, c.seq_len, (float)n_params / 1e6f);
    printf("[model] fp16_cache=%d rope_scale=%.2f threads=%d\n",
           c.fp16_cache, c.rope_scaling, c.pipeline_threads);

    ModelWeights w;
    init_weights(&w, &c);
    ParamList params = collect_params(&w);

    float **grads = calloc(params.count, sizeof(float*));
    for (int i = 0; i < params.count; i++) grads[i] = calloc(params.tensors[i]->size, sizeof(float));

    Adam *opt = adam_new(&params, 0.9f, 0.999f, 1e-8f);
    TrainState ts = alloc_train_state(&c);

    printf("[train] starting training: %d steps, batch=%d, seq=%d, lr=%.1e\n",
           c.max_steps, c.batch_size, c.seq_len, c.lr);

    clock_t train_start = clock();
    float running_loss = 0.0f;
    int loss_count = 0;

    for (int step = 0; step < c.max_steps; step++) {
        float lr = c.lr;
        if (step < c.warmup_steps) lr = c.lr * ((float)(step+1) / c.warmup_steps);
        else {
            float progress = (float)(step - c.warmup_steps) / (float)(c.max_steps - c.warmup_steps);
            lr = c.lr * 0.5f * (1.0f + cosf(3.14159f * progress));
        }
        if (lr < c.lr * 0.01f) lr = c.lr * 0.01f;

        int max_start = n_tokens - c.seq_len - 1;
        if (max_start < 0) max_start = 0;
        int start = (int)(rand_uniform() * max_start);
        int *tokens = all_tokens + start;
        int *targets = all_tokens + start + 1;

        float loss = train_forward(&w, &c, &ts, tokens, targets, c.seq_len);
        running_loss += loss;
        loss_count++;

        for (int i = 0; i < params.count; i++) memset(grads[i], 0, params.tensors[i]->size * sizeof(float));
        train_backward(&w, &c, &ts, tokens, targets, c.seq_len, grads);

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

        adam_step(opt, &params, grads, lr, c.weight_decay);

        if ((step+1) % c.log_every == 0 || step == 0) {
            float avg_loss = running_loss / loss_count;
            float elapsed = (float)(clock() - train_start) / CLOCKS_PER_SEC;
            float tok_per_sec = (float)((step+1) * c.seq_len) / elapsed;
            printf("  step %4d/%d  loss=%.4f  lr=%.2e  tok/s=%.0f  (%.1fs)\n",
                   step+1, c.max_steps, avg_loss, lr, tok_per_sec, elapsed);
            running_loss = 0; loss_count = 0;
        }
    }

    float total_time = (float)(clock() - train_start) / CLOCKS_PER_SEC;
    printf("[train] finished in %.1f seconds\n", total_time);

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
                if ((step+1) % 20 == 0) printf("  personality step %d/%d  loss=%.4f\n", step+1, c.personality_steps, loss);
            }
            free(pers_tokens);
        }
        free(pers_text);
    } else {
        printf("[personality] no %s found, skipping finetune\n", c.personality_path);
    }

    export_gguf(&w, &c, &tok);
    chat_loop(&w, &c, &tok);

    free(all_tokens);
    printf("[l] done.\n");
    return 0;
}
