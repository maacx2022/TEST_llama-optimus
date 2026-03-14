# TEST_llama-optimus

Inference tuning for `llama.cpp`, upgraded from a simple Optuna tuner into a hardware-aware orchestration layer.

It benchmarks real `llama.cpp` runs, adapts to your machine, detects model architecture, and prints a ready-to-use launch command for local inference.

<p align="center">
  <img src="assets/llama.optimus_logo_PyPi_Optuna_Llama.png" width="560" alt="TEST_llama-optimus">
</p>

## Origin

This repository is inspired by the original [`llama-optimus`](https://github.com/BrunoArsioli/llama-optimus) project.

That original tool was a useful starting point, but it has not kept pace with newer `llama.cpp` inference patterns, newer hardware assumptions, or newer model families.

`TEST_llama-optimus` should be read as:

- an explicit continuation of that idea
- a practical rework of the old tuning flow
- a broader orchestration layer for modern local inference rather than a small throughput-only tuner

The original project deserves clear credit for the initial shape of the idea. This repo deliberately pushes it further.

## What It Does

- Auto-detects model architecture from GGUF metadata when possible
- Tunes `llama.cpp` with Optuna using multiple objectives, not just raw `tok/s`
- Profiles hardware for GPU/CPU topology and storage tiers
- Adapts search spaces for `transformer`, `diffused`, `lfm`, `bitnet`, and `mtp`
- Handles newer inference ideas such as speculative decoding and KV-cache policies when the local `llama.cpp` binary supports them
- Prints a clean final launch command for `llama_cpp.server`

## Why This Exists

Most local inference tuning still focuses on one number: throughput.

That is too narrow.

A good inference configuration is a tradeoff between:

- time to first token
- inter-token latency
- energy efficiency
- VRAM efficiency
- actual hardware constraints
- model architecture constraints

This project is meant to find useful real-world settings, not just flattering benchmark screenshots.

## Workflow

```text
model.gguf + llama.cpp/bin
          |
          v
┌──────────────────────────────────────────────┐
│ Inspect                                      │
│ - GGUF metadata                              │
│ - binary capabilities                        │
└──────────────────────┬───────────────────────┘
                       |
                       v
┌──────────────────────────────────────────────┐
│ Profile Host                                 │
│ - CPU topology                               │
│ - GPU features                               │
│ - MTDS latency path                          │
└──────────────────────┬───────────────────────┘
                       |
                       v
┌──────────────────────────────────────────────┐
│ Resolve Strategy                             │
│ - transformer                                │
│ - diffused / MoE                             │
│ - lfm                                        │
│ - bitnet                                     │
│ - mtp                                        │
└──────────────────────┬───────────────────────┘
                       |
                       v
┌──────────────────────────────────────────────┐
│ Stabilize                                    │
│ - warmup                                     │
│ - remove cold-start bias                     │
└──────────────────────┬───────────────────────┘
                       |
                       v
┌──────────────────────────────────────────────┐
│ Search                                       │
│ - P99 TTFT                                   │
│ - P99 ITL                                    │
│ - Tokens / Joule                             │
│ - VRAM efficiency                            │
└──────────────────────┬───────────────────────┘
                       |
                       v
┌──────────────────────────────────────────────┐
│ Emit                                         │
│ - final launch command                       │
│ - practical local inference config           │
└──────────────────────────────────────────────┘
```

## Mental Model

Think of the app as four layers:

```text
┌──────────────────────────────────────────────┐
│ Layer 1: Observe                             │
│ read GGUF metadata and local llama.cpp       │
├──────────────────────────────────────────────┤
│ Layer 2: Understand                          │
│ identify hardware and architecture           │
├──────────────────────────────────────────────┤
│ Layer 3: Search                              │
│ benchmark real configurations with Optuna    │
├──────────────────────────────────────────────┤
│ Layer 4: Ship                                │
│ print one command you can actually run       │
└──────────────────────────────────────────────┘
```

## Architecture-Aware Routing

The tool does not treat all GGUF models as if they were the same.

| Architecture | Strategy |
| --- | --- |
| `transformer` | Standard dense path, KV tuning enabled, speculative decoding allowed |
| `diffused` | MoE-style path, stronger focus on memory bandwidth and `ubatch` behavior |
| `lfm` | Liquid-style path, no normal KV-cache tuning, speculative decoding disabled |
| `bitnet` | CPU-first path, GPU layers forced to `0`, thread policy can be restricted |
| `mtp` | Similar to dense, but speculative decoding is bypassed |

Architecture resolution order:

1. GGUF metadata via `llama-gguf`
2. tensor / key signatures from the model file
3. filename heuristics
4. fallback to `transformer`

## Hardware Awareness

This project is intentionally not fully hardware-agnostic.

It contains targeted behavior for:

- NVIDIA Blackwell detection for `NVFP4` and `PDL`-style feature gating
- Ryzen X3D detection to avoid bad thread placement
- MTDS profiling across VRAM, system RAM, and NVMe-backed storage
- capability detection for the local `llama.cpp` binary so unsupported flags are skipped cleanly

The goal is not abstract elegance. The goal is good local inference settings on real machines.

## What Makes This Repo Different

Compared to the original throughput-only tuner, this repo adds:

- architecture-aware routing
- GGUF-based architecture auto-detection
- hardware profiling
- multi-objective selection instead of a single scalar score
- capability-aware flag emission for different `llama.cpp` builds
- a cleaner deployment-oriented final output

That is the main shift: from "search a few flags" to "orchestrate inference for the machine and model that actually exist."

## Requirements

- Python `>= 3.8`
- a working local `llama.cpp` build
- at least one `.gguf` model
- `llama-bench` available in your `llama.cpp/build/bin`

For best results, use a recent `llama.cpp` build. This repo already degrades gracefully if some newer flags are missing, but newer binaries expose more tuning surface.

## Install

### PyPI

```bash
pip install llama-optimus
```

### Local dev

```bash
git clone https://github.com/maacx2022/TEST_llama-optimus.git
cd TEST_llama-optimus
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quick Start

### With explicit paths

```bash
llama-optimus \
  --llama-bin /path/to/llama.cpp/build/bin \
  --model /path/to/model.gguf
```

### With environment variables

```bash
export LLAMA_BIN=/path/to/llama.cpp/build/bin
export MODEL_PATH=/path/to/model.gguf
llama-optimus
```

Expected flow:

```text
1. point the tool at llama.cpp/build/bin
2. point the tool at a GGUF model
3. let it resolve the architecture
4. let it warm up the system
5. let it search
6. copy the final launch command
```

### Fast smoke run

```bash
llama-optimus \
  --llama-bin /path/to/llama.cpp/build/bin \
  --model /path/to/model.gguf \
  --trials 3 \
  --repeat 1 \
  --n-tokens 32 \
  --n-warmup-runs 4
```

## Common Examples

### Let the app auto-detect the architecture

```bash
llama-optimus \
  --llama-bin /path/to/llama.cpp/build/bin \
  --model /path/to/model.gguf \
  --arch auto
```

### Force LFM mode

```bash
llama-optimus \
  --llama-bin /path/to/llama.cpp/build/bin \
  --model /path/to/lfm-model.gguf \
  --arch lfm \
  --trials 15 \
  --repeat 2
```

### Explore a diffused / MoE model

```bash
llama-optimus \
  --llama-bin /path/to/llama.cpp/build/bin \
  --model /path/to/lada2.gguf \
  --arch diffused \
  --trials 20 \
  --repeat 2
```

### Real GPU sweep

```bash
llama-optimus \
  --llama-bin /home/macko/llm/llama.cpp/build/bin \
  --model /home/macko/models/your-model.gguf \
  --trials 15 \
  --repeat 2 \
  --n-tokens 64 \
  --n-warmup-runs 6 \
  --n-warmup-tokens 64
```

## Final Output

The final block is meant to be copy-paste friendly:

```text
==================================================
[TEST_llama-optimus] OPTIMAL ORCHESTRATION FOUND:
Target: <model_name> | Draft: <draft_model_name>
Launch Command: python -m llama_cpp.server [OPTIMIZED_FLAGS]
==================================================
```

Example:

```text
==================================================
[TEST_llama-optimus] OPTIMAL ORCHESTRATION FOUND:
Target: LFM2.5-1.2B...gguf | Draft: disabled
Launch Command: python -m llama_cpp.server --model /path/to/model.gguf --threads 3 --batch-size 4296 --ubatch-size 1400 --n-gpu-layers 999 --override-tensor 'blk\.\d+\.ffn_down_exps\.=CPU' --nvfp4 --pdl
==================================================
```

## CLI Highlights

| Flag | Meaning |
| --- | --- |
| `--llama-bin` | Path to `llama.cpp/build/bin` |
| `--model` | Path to the `.gguf` model |
| `--arch` | `auto`, `transformer`, `diffused`, `lfm`, `bitnet`, `mtp` |
| `--trials` | Number of Optuna trials |
| `--repeat` | Benchmark repetitions per trial |
| `--n-tokens` | Tokens used in benchmarking |
| `--ngl-max` | Skip GPU layer estimation and set max directly |
| `--no-warmup` | Disable the warm-up stage |
| `--override-mode` | Control tensor override scanning |

See full CLI help:

```bash
llama-optimus --help
```

## Current Design Notes

- The optimizer is multi-objective, but still produces one practical final command by ranking Pareto candidates
- Capability detection matters because local `llama.cpp` builds do not all expose the same flags
- Some architecture-specific behavior is intentionally opinionated
- The project currently targets Linux best; Windows support is lighter and macOS profiling is not yet first-class

## Failure Modes To Expect

- Some trial combinations will fail. That is normal.
- Older `llama.cpp` binaries expose fewer flags. Unsupported flags should be skipped.
- Aggressive `batch`, `ubatch`, or `flash-attn` combinations may be unstable for a specific model.
- Cold runs can look falsely better than warm runs. Trust stabilized measurements.

## Tests

```bash
python -m pytest -q
python -m py_compile optimus.py src/llama_optimus/*.py test/*.py
```

## Project Layout

```text
TEST_llama-optimus/
├── src/llama_optimus/
│   ├── cli.py
│   ├── core.py
│   ├── hw_profiler.py
│   ├── inference_engines.py
│   ├── model_arch.py
│   ├── override_patterns.py
│   └── search_space.py
├── test/
├── assets/
├── optimus.py
└── README.md
```

## Roadmap Direction

- stronger GGUF metadata parsing
- broader non-NVIDIA hardware support
- more robust capability negotiation for future `llama.cpp` flags
- better report export for benchmark history

## Credits

- Original inspiration: [`BrunoArsioli/llama-optimus`](https://github.com/BrunoArsioli/llama-optimus)
- Core benchmark target: [`ggerganov/llama.cpp`](https://github.com/ggerganov/llama.cpp)
- Search engine: [`Optuna`](https://optuna.org/)

## License

MIT. See [LICENSE](/home/macko/Desktop/TEST_llama-optimus/LICENSE).
