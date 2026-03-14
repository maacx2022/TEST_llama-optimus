# llama_optimus/cli.py

import argparse
import os
import platform
import sys
from pathlib import Path

from .core import estimate_max_ngl, run_optimization, warmup_until_stable
from .hw_profiler import build_hardware_profile
from .model_arch import AUTO_ARCH_CHOICES, detect_architecture
from .override_patterns import OVERRIDE_PATTERNS
from .search_space import SEARCH_SPACE, max_threads
from llama_optimus import __version__


def main():
    parser = argparse.ArgumentParser(
        description="llama-optimus: benchmark and orchestrate llama.cpp inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--trials", type=int, default=45, help="Number of Optuna trials.")
    parser.add_argument("--model", type=str, help="Path to target model.")
    parser.add_argument("--llama-bin", type=str, help="Path to llama.cpp build/bin folder.")
    parser.add_argument("--metric", type=str, default="mean", choices=["tg", "pp", "mean"], help="Retained for compatibility; orchestration is always multi-metric.")
    parser.add_argument("--arch", type=str, default="auto", choices=AUTO_ARCH_CHOICES, help="Model architecture family or auto-detect from GGUF metadata.")
    parser.add_argument("--ngl-max", type=int, help="Maximum number of model layers for -ngl.")
    parser.add_argument("--repeat", "-r", type=int, default=3, help="Number of llama-bench runs per configuration.")
    parser.add_argument("--n-tokens", type=int, default=192, help="Number of generated tokens used during benchmarking.")
    parser.add_argument("--n-warmup-tokens", "-nwt", type=int, default=128, help="Warm-up token count.")
    parser.add_argument("--n-warmup-runs", type=int, default=12, help="Maximum warm-up iterations.")
    parser.add_argument("--no-warmup", action="store_true", help="Skip warm-up phase.")
    parser.add_argument("--override-mode", type=str, default="scan", choices=["none", "scan", "custom"], help=f"Override tensor scan mode. Presets: {OVERRIDE_PATTERNS.keys()}")
    parser.add_argument("--version", "-v", action="version", version=f"llama-optimus v{__version__}")
    args = parser.parse_args()

    llama_bin_path = args.llama_bin or os.environ.get("LLAMA_BIN")
    model_path = args.model or os.environ.get("MODEL_PATH")

    if not llama_bin_path or not model_path:
        sys.exit("ERROR: pass --llama-bin and --model, or set LLAMA_BIN and MODEL_PATH.")

    if platform.system() == "Windows":
        llama_bench_path = f"{llama_bin_path}/Release/llama-bench.exe"
    else:
        llama_bench_path = f"{llama_bin_path}/llama-bench"

    if not Path(llama_bench_path).is_file():
        sys.exit(f"ERROR: llama-bench not found at {llama_bench_path}")
    if not Path(model_path).is_file():
        sys.exit(f"ERROR: model not found at {model_path}")

    hardware_profile = build_hardware_profile()
    resolved_arch = detect_architecture(model_path, llama_bin_path) if args.arch == "auto" else args.arch
    print("")
    print("#################")
    print("# LLAMA-OPTIMUS #")
    print("#################")
    print("")
    print(f"Logical CPUs: {max_threads}")
    print(f"Profiler threads recommendation: {hardware_profile.recommended_threads}")
    print(f"CPU: {hardware_profile.cpu_name}")
    print(f"GPU: {hardware_profile.gpu_name}")
    print(f"Resolved architecture: {resolved_arch}")
    print(f"Blackwell features: NVFP4={hardware_profile.enable_nvfp4} PDL={hardware_profile.enable_pdl}")
    print(f"MTDS latency map (ms): {hardware_profile.mtds_latency_ms}")
    print(f"Path to 'llama-bench': {llama_bench_path}")
    print(f"Path to 'model.gguf': {model_path}")
    print("")

    if args.ngl_max is not None:
        SEARCH_SPACE["gpu_layers"]["high"] = args.ngl_max
    elif resolved_arch != "bitnet":
        SEARCH_SPACE["gpu_layers"]["high"] = estimate_max_ngl(
            llama_bench_path=llama_bench_path,
            model_path=model_path,
            min_ngl=0,
            max_ngl=SEARCH_SPACE["gpu_layers"]["high"],
        )

    if not args.no_warmup:
        warmup_until_stable(
            llama_bench_path=llama_bench_path,
            model_path=model_path,
            metric=args.metric,
            ngl=SEARCH_SPACE["gpu_layers"]["high"],
            min_runs=4,
            n_warmup_runs=args.n_warmup_runs,
            n_warmup_tokens=args.n_warmup_tokens,
            max_threads=hardware_profile.recommended_threads,
        )

    run_optimization(
        n_trials=args.trials,
        n_tokens=args.n_tokens,
        metric=args.metric,
        repeat=args.repeat,
        llama_bench_path=llama_bench_path,
        model_path=model_path,
        llama_bin_path=llama_bin_path,
        override_mode=args.override_mode,
        arch=resolved_arch,
        hardware_profile=hardware_profile,
    )


if __name__ == "__main__":
    main()
