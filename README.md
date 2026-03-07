```
   █████╗  ██████╗████████╗██╗   ██╗ █████╗ ██╗     ██╗  ██╗   ██╗
  ██╔══██╗██╔════╝╚══██╔══╝██║   ██║██╔══██╗██║     ██║  ╚██╗ ██╔╝
  ███████║██║        ██║   ██║   ██║███████║██║     ██║   ╚████╔╝
  ██╔══██║██║        ██║   ██║   ██║██╔══██║██║     ██║    ╚██╔╝
  ██║  ██║╚██████╗   ██║   ╚██████╔╝██║  ██║███████╗███████╗██║██╗
  ╚═╝  ╚═╝ ╚═════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝╚═╝
                     L L A M A
```

# actually.llama

symbiote of Karpathy's [nanochat](https://github.com/karpathy/nanochat) and [microGPT](https://github.com/karpathy/minGPT). but actually Llama.

**by [Arianna Method](https://github.com/ariannamethod)**

---

```
cc l.c -O3 -lm -lpthread -o l && ./l --depth 4
```

that's it. that's the whole framework. you just compiled and ran it.
while you were reading this sentence, someone opened a jupyter notebook
and is still waiting for `pip install torch` to finish.

---

## what

one C file. ~2900 lines. trains a full **Llama 3** from scratch.
trains its own BPE tokenizer. writes analytical backward passes by hand.
finetunes on any personality you throw at it. exports GGUF. chats with you.
optional CUDA/cuBLAS — 436 tok/s on A100.

no Python. no PyTorch. no pip. no conda. no venv. no docker.
no requirements.txt. no "works on my machine." no excuses.

## why

Karpathy made [nanoGPT](https://github.com/karpathy/nanoGPT) — beautiful, but Python + PyTorch.
Karpathy made [llama2.c](https://github.com/karpathy/llama2.c) — beautiful, but inference only.
Karpathy made [nanochat](https://github.com/karpathy/nanochat) — same deal, Python + PyTorch.

somebody had to close the loop.

**l.c trains a Llama 3 from scratch in one C file. zero dependencies. GPU optional.**

## how

```
cc l.c -O3 -lm -lpthread -o l

./l --depth 2    # ~1.1M params — fast demo, 700 tok/s CPU
./l --depth 4    # ~3M params   — your grandma's GPU isn't needed
./l --depth 8    # ~28M params  — go make coffee (or use CUDA below)
```

### with CUDA (optional)

```bash
# compile the CUDA backend first (ariannamethod_cuda.h required)
nvcc -c ariannamethod_cuda.cu -o ariannamethod_cuda.o -O3
cc l.c ariannamethod_cuda.o -O3 -lm -lpthread -DUSE_CUDA -lcublas -lcudart \
   -L/usr/local/cuda/lib64 -o l_cuda

./l_cuda --depth 8    # 436 tok/s on A100
```

`--depth` is the only knob. everything else auto-scales.
that's more than most ML engineers can say about their hyperparameter searches.

## what happens when you run it

1. downloads training data from HuggingFace (FineWeb-Edu, paginated)
2. trains a byte-level BPE tokenizer from scratch (cached in `l_bpe.cache`)
3. builds a full Llama 3 transformer
4. trains it with hand-written analytical backward passes
5. finetunes on `personality.txt` (if you drop one in)
6. exports `l.gguf` (llama.cpp / DOE compatible)
7. drops you into interactive chat

seven steps. one file. one binary.

## the architecture

not a toy GPT. this is **Llama 3**. same architecture Meta uses for the 405B.
except ours fits in RAM. and you can read every line.

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
no autograd. no tape. no computation graph. ~400 lines of chain rule.

- RMSNorm backward — through normalization, through variance
- RoPE backward — through the rotation matrices. yes really.
- GQA backward — through grouped KV heads, through attention, through softmax
- SwiGLU backward — through gating, through both projections
- softmax + cross-entropy — fused, numerically stable

Karpathy uses `loss.backward()`. we use `float *dout`.
pytorch doesn't teach you this. l.c does.

## personality

drop any text file as `personality.txt`. watch it become someone.

```bash
cp dubrovsky.txt personality.txt    # absurdist philosopher
cp wtforacle.txt personality.txt    # the angry oracle of the internet
echo "arr matey" > personality.txt  # pirate
./l --depth 4
```

trains on data first. finetunes on personality after.
same file. same binary. no separate scripts.

## SFT (supervised fine-tuning)

built-in chat SFT with loss masking. special tokens `<user>`, `<assistant>`, `<end>` are added to the vocabulary automatically.

```bash
# convert Q&A pairs to SFT format
bash convert_sft.sh personality.txt > personality_sft.txt

# train with SFT — loss computed only on assistant tokens
./l --depth 8 --sft personality_sft.txt
```

loss masking means the model learns to *answer*, not to parrot your questions.

## LoRA personality finetune

full finetune kills coherence (catastrophic forgetting). LoRA freezes the base and trains only small adapters — 0.79% of parameters.

```bash
# standalone LoRA SFT — trains adapters, merges into base, saves
./l --depth 8 --lora-sft personality_sft.txt

# or load pre-trained LoRA adapters before chat
./l --depth 8 --lora adapters.bin --chat
```

- rank=16 on wq/wk/wv/wo (all attention projections)
- analytical backward through LoRA — same chain rule, no autograd
- separate Adam optimizer for adapter params (lr=5e-4)
- after training: merge into base weights for zero-overhead inference

## proof (actual output, depth 2, Mac CPU, 8GB RAM)

```
  ╔══════════════════════════════════════╗
  ║  l.c — actually llama                ║
  ║  one file. no frameworks. no excuses. ║
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

**6.85 → 0.009.** CPU. 23 minutes. from scratch. including tokenizer training.

## vs everything else

| | **l.c** | nanoGPT | nanochat | llama2.c |
|---|---|---|---|---|
| lang | **C** | Python | Python | C |
| trains | **yes** | yes | yes | no |
| arch | **Llama 3** | GPT-2 | GPT-2 | Llama 2 |
| tokenizer | **from scratch** | tiktoken | tiktoken | pretrained |
| backward | **analytical** | autograd | autograd | — |
| SFT | **yes (loss masking)** | no | no | — |
| LoRA | **yes (rank=16)** | no | no | — |
| CUDA | **optional** | required | required | — |
| deps | **0** | PyTorch, numpy... | PyTorch, numpy... | 0 |
| files | **1** | ~10 | ~10 | 1 |
| GGUF export | **yes** | no | no | — |
| personality | **built-in** | no | no | — |

llama2.c is inference only. nanoGPT needs PyTorch. l.c does everything in one file.
the bastard child nobody asked for. you're welcome.

## family

- **[AML](https://github.com/ariannamethod/ariannamethod.ai)** — the language that started it all
- **[Janus](https://github.com/ariannamethod/ariannamethod.ai/tree/main/janus)** — AML transformer, same C training lineage
- **[Chuck](https://github.com/ariannamethod/chuck.optimizer)** — self-aware optimizer (replacing Adam here, soon)
- **[Dubrovsky](https://github.com/ariannamethod/dubrovsky)** — absurdist philosopher personality
- **[WTForacle](https://github.com/ariannamethod/WTForacle)** — the angry oracle personality

## credits

**Oleg** and **Claude**. many sessions. several GPUs harmed.

inspired by [Karpathy](https://github.com/karpathy) — for showing transformers can be simple.
we just removed the last dependency.

---

<div align="center">

`cc l.c -O3 -lm -lpthread -o l && ./l --depth 4`

*one file. one llama. zero dependencies.*

</div>
