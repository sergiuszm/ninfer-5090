# Qwen3.8-27B on the RTX 5090: NInfer vs llama.cpp

Measured 2026-08-15 on the host `nuntius`. This document records the full prefill and decode
comparison between this repository's NInfer build and llama.cpp on the same card, plus the
fp16-accumulate change that closed the prefill gap at depth.

## Summary

- llama.cpp wins shallow prefill by about 20%. The crossover sits just under a 64K-token prompt.
- NInfer wins prefill at depth: +20% at 128K, +33% at 200K.
- NInfer wins decode without speculation at every depth. The lead grows from +5% at zero depth
  to +60% at 200K.
- NInfer wins speculative decode at depth by about 2x at comparable draft acceptance.

## Test setup

| Component | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, 32 GB, power limit 400 W (deliberate, default 600 W) |
| Host | AMD Ryzen 9950X3D, 192 GB RAM, Ubuntu 26.04 |
| NInfer | commit `4483c820` on `nuntius-serve`, CUDA 13.1, `sm_120a` |
| NInfer artifact | `qwen3_8_27b.ninfer`, groupwise-int, 16.96 GiB |
| llama.cpp | build `1692f9e`, image `llamacpp-0814:server-cuda` |
| llama.cpp model | `Qwen3.8-27B-UD-Q4_K_XL.gguf` (unsloth), 16.68 GiB |

Shared settings: batch 4096, micro-batch 1024, flash attention on, single request. NInfer uses
INT8 group-64 KV and a 262144-token context. llama.cpp uses `q8_0` KV for both tensors. Each
engine runs its production quantization format, so this is a deployment comparison, not a
same-weights comparison.

## Prefill

llama.cpp rows come from `llama-bench` (`-p N`, 2 repetitions). NInfer rows come from the
`ninfer-serve` `/metrics` counters on needle-in-a-haystack prompts; every needle retrieval was
exact. The `/metrics` numbers include request handling, which costs NInfer a few percent on the
shortest prompt.

| Prompt tokens | llama.cpp | NInfer | NInfer vs llama.cpp |
|---:|---:|---:|---|
| 1,024 | 3,222 tok/s | 2,689 tok/s | -17% |
| 7,671 | not measured | 2,647 tok/s | - |
| 63,619 / 65,536 | 2,289 tok/s | 2,299 tok/s | even |
| 127,925 / 131,072 | 1,642 tok/s | 1,964 tok/s | **+20%** |
| 200,618 / 200,704 | 1,260 tok/s | 1,682 tok/s | **+33%** |

llama.cpp loses 61% of its shallow rate by 200K. NInfer loses 37%.

## Decode without speculation

llama.cpp rows come from `llama-bench` (`tg64 -d N`). The NInfer rows at 0 and 64K come from
the same-day bench-corpus run; the rows at 128K and 200K come from a temporary `ninfer-serve`
instance without `--spec`, generating 1,200 tokens of Python after a long prompt. The two
NInfer methods agree at 128K (61.1 vs 61.4 tok/s).

| Depth | llama.cpp | NInfer | NInfer vs llama.cpp |
|---:|---:|---:|---|
| 0 | 71.3 tok/s | 75.1 tok/s | +5% |
| 64K | 54.3 tok/s | 67.3 tok/s | +24% |
| 128K | 43.0 tok/s | 61.4 tok/s | +43% |
| 200K | 34.9 tok/s | 55.9 tok/s | **+60%** |

## Decode with speculation

Both engines received the same request: a long journal prompt followed by a code task, greedy,
thinking disabled, 1,200 generated tokens. llama.cpp ran its `draft-mtp` speculation; NInfer ran
MTP with 3 draft tokens and `--lm-head-draft`. Rates come from the llama.cpp response `timings`
object and the NInfer `/metrics` counters.

| Depth | llama.cpp draft-mtp | NInfer MTP3 | NInfer vs llama.cpp |
|---:|---:|---:|---|
| 128K | 77.5 tok/s (84.0% accepted) | 152.9 tok/s (88.1% accepted) | **2.0x** |
| 200K | 65.4 tok/s (85.4% accepted) | 133.1 tok/s (81.7% accepted) | **2.0x** |

Draft acceptance is nearly equal, so the gap is engine efficiency: the llama.cpp verify step
pays far more per token at depth.

## The fp16-accumulate change

Commit `4483c820` accumulates each 64-key PV tile of the INT8 attention prefill kernel in
packed fp16 and folds it into the fp32 running accumulator once per tile. Consumer GPUs run
f32-accumulate HMMA at half rate, and the kernel sat at that ceiling before the change.

Kernel throughput, `d256-h24-kv4` INT8 append, 2,048 new tokens:

| Context | Before | After |
|---:|---:|---:|
| 8K | 185.9 TFLOP/s | 226.3 TFLOP/s |
| 64K | 191.9 TFLOP/s | 236.0 TFLOP/s |
| 128K | 192.5 TFLOP/s | 229.6 TFLOP/s |

Serve prefill on identical payloads: 2,247 to 2,299 tok/s at 64K, 1,896 to 1,964 at 128K,
1,607 to 1,682 at 200K. Validation: exact needle retrieval at 64K, 128K, and 200K; 83 of 84
suite tests pass, with the one failure being the known `frontend_test` tokenizer-path issue
that predates the change. The same change shipped earlier on the RTX 4090 port
([sergiuszm/ninfer-4090](https://github.com/sergiuszm/ninfer-4090)), where it was the largest
single win of a five-part retune.

## Reproduction

llama.cpp prefill and decode sweeps:

```bash
docker run --rm --gpus all -v /home/serchio/Models:/models \
  --entrypoint /app/llama llamacpp-0814:server-cuda bench \
  -m /models/unsloth/Qwen3.8-27B/Qwen3.8-27B-UD-Q4_K_XL.gguf \
  -ngl 999 -ctk q8_0 -ctv q8_0 -fa on -ub 1024 -b 4096 \
  -p 1024,65536,131072,200704 -n 0 -r 2

# decode: swap the last line for
  -p 0 -n 64 -d 0,65536,131072,200704 -r 2
```

NInfer numbers: send a fixed prompt to `ninfer-serve`, snapshot `GET /metrics` before and
after, and difference `llamacpp:prompt_tokens_total` / `llamacpp:prompt_seconds_total` for
prefill or `llamacpp:tokens_predicted_total` / `llamacpp:tokens_predicted_seconds_total` for
decode. Draft acceptance comes from the `ninfer:draft_tokens_total` and
`ninfer:draft_accepted_tokens_total` counters.
