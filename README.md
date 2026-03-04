```
  ██╗          ██████╗
  ██║         ██╔════╝
  ██║         ██║
  ██║         ██║
  ███████╗    ╚██████╗
  ╚══════╝     ╚═════╝
```

# l.c

one file. one llama. no excuses.

**by [Arianna Method](https://github.com/ariannamethod)**

---

```
cc l.c -O3 -lm -lpthread -o l && ./l --depth 4
```

that's it. that's the whole framework.

---

## what

1907 lines of C. compiles in 0.3 seconds. trains a full Llama 3 from scratch.
finetunes on any personality. exports GGUF. runs interactive chat.

no Python. no PyTorch. no pip. no conda. no venv. no docker.
no requirements.txt. no "works on my machine."

a C compiler. `-lm`. `-lpthread`. done.

## why

[Karpathy](https://github.com/karpathy) made [nanoGPT](https://github.com/karpathy/nanoGPT) — beautiful, but Python + PyTorch.
Karpathy made [llama2.c](https://github.com/karpathy/llama2.c) — beautiful, but inference only.
Karpathy made [nanochat](https://github.com/karpathy/nanochat) — same deal, Python + PyTorch.

somebody had to close the loop.

**l.c trains a Llama 3 in one C file. no dependencies. no GPU required.**

symbiote of nanochat and microGPT. but actually llama.

## how

```
cc l.c -O3 -lm -lpthread -o l

./l --depth 2    # ~1.1M params — fast demo, 700 tok/s on CPU
./l --depth 4    # ~3M params   — decent
./l --depth 6    # ~7M params   — good
./l --depth 8    # ~15M params  — best single-CPU quality
```

`--depth` is the only knob. everything else auto-scales.

## what happens when you run it

1. downloads training data (or generates synthetic demo)
2. trains a byte-level BPE tokenizer from scratch
3. initializes a Llama 3 transformer
4. trains it with hand-written analytical backward passes
5. finetunes on `personality.txt` (if present)
6. exports `l.gguf`
7. drops you into interactive chat

all of this. one file. one binary.

## architecture

not a toy GPT. this is **Llama 3**.

```
Token Embedding (untied)
  ┌─────────────────────────────────────┐
  │  RMSNorm                            │
  │  RoPE                               │
  │  Grouped Query Attention (GQA)      │ × depth
  │  Residual                           │
  │  RMSNorm                            │
  │  SwiGLU FFN (gate · up · down)      │
  │  Residual                           │
  └─────────────────────────────────────┘
RMSNorm → LM Head → Softmax → Token
```

every component has a hand-written forward **and** backward pass.
no autograd. no tape. no computation graph. ~400 lines of pure gradient math.

- RMSNorm backward — through normalization, through variance
- RoPE backward — through rotation matrices
- GQA backward — through grouped KV heads, through attention scores
- SwiGLU backward — through gating, through both projections
- softmax + cross-entropy — fused, numerically stable

Karpathy uses `loss.backward()`. we use `float *dout`.

## personality

drop any text file as `personality.txt`. the model trains on data first,
then finetunes on personality.

```bash
cp dubrovsky.txt personality.txt    # absurdist philosopher
cp wtforacle.txt personality.txt    # angry redditor
echo "arr matey" > personality.txt  # pirate
./l --depth 4
```

## proof (actual output, depth 2, Mac CPU)

```
  ╔══════════════════════════════════════╗
  ║         l.c v1.0                     ║
  ║   One file. Pure C. No frameworks.   ║
  ║   Full Llama 3 from scratch.         ║
  ╚══════════════════════════════════════╝

[bpe] done: 292 merges, vocab=550
[model] depth=2 dim=192 heads=3 kv_heads=3 hidden=512 params=1.10M
[train] 2000 steps, lr=3.0e-04
  step    1  loss=6.8521  tok/s=770
  step  100  loss=0.0698  tok/s=705
  step 1000  loss=0.0098  tok/s=393
  step 2000  loss=0.0088  tok/s=371
[train] finished in 1381s
[gguf] exported to l.gguf (4.2 MB)
```

**6.85 → 0.009.** CPU. 23 minutes. from scratch.

## vs everything else

| | **l.c** | nanoGPT | nanochat | llama2.c |
|---|---|---|---|---|
| lang | **C** | Python | Python | C |
| trains | **yes** | yes | yes | no |
| arch | **Llama 3** | GPT-2 | GPT-2 | Llama 2 |
| tokenizer | **from scratch** | tiktoken | tiktoken | pretrained |
| backward | **analytical** | autograd | autograd | — |
| deps | **0** | PyTorch, numpy... | PyTorch, numpy... | 0 |
| files | **1** | ~10 | ~10 | 1 |
| GGUF export | **yes** | no | no | — |
| personality | **built-in** | no | no | — |

llama2.c is inference. nanoGPT needs PyTorch. l.c does everything in one file.

## family

- **[AML](https://github.com/ariannamethod/ariannamethod.ai)** — the language that started it all
- **[Janus](https://github.com/ariannamethod/ariannamethod.ai/tree/main/janus)** — AML transformer, same C training lineage
- **[Chuck](https://github.com/ariannamethod/chuck.optimizer)** — self-aware optimizer (replacing Adam here, soon)
- **[Dubrovsky](https://github.com/ariannamethod/dubrovsky)** — absurdist personality dataset
- **[WTForacle](https://github.com/ariannamethod/WTForacle)** — angry oracle personality dataset

## credits

**Oleg** and **Claude**. one session. no GPU harmed.

---

<div align="center">

`cc l.c -O3 -lm -lpthread -o l && ./l --depth 4`

*one file. one llama. no excuses.*

**הרזוננס לא נשבר**

</div>
