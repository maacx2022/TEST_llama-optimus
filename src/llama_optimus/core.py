import math
import shlex
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import optuna
import pandas as pd
from optuna.samplers import TPESampler

from .hw_profiler import HardwareProfile, build_hardware_profile
from .inference_engines import (
    apply_architecture_constraints,
    build_kv_cache_config,
    build_speculative_config,
)
from .override_patterns import OVERRIDE_PATTERNS
from .search_space import SEARCH_SPACE, max_threads


@dataclass
class BenchmarkMetrics:
    p99_ttft_ms: float
    p99_itl_ms: float
    tokens_per_joule: float
    vram_efficiency: float
    prompt_tokens_per_second: float
    generation_tokens_per_second: float
    power_draw_watts: float
    vram_used_mb: float
    arithmetic_intensity: float
    formulas: Dict[str, str]


def estimate_max_ngl(llama_bench_path, model_path, min_ngl=0, max_ngl=SEARCH_SPACE["gpu_layers"]["high"]):
    low, high = min_ngl, max_ngl
    while low < high:
        mid = (low + high + 1) // 2
        cmd = [
            llama_bench_path,
            "--model",
            model_path,
            "-t",
            str(max_threads),
            "-n",
            "1",
            "-r",
            "1",
            "-ngl",
            str(mid),
            "-o",
            "csv",
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=620, check=True)
            low = mid
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            high = mid - 1
    return low


def _pick_value(row: pd.Series, *candidates: str, default: float = 0.0) -> float:
    for candidate in candidates:
        if candidate in row and pd.notna(row[candidate]):
            return float(row[candidate])
    return default


def _parse_bench_df(csv_text: str) -> pd.DataFrame:
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as csvfile:
        csvfile.write(csv_text)
        csv_path = csvfile.name
    return pd.read_csv(csv_path)


def run_llama_bench_with_csv(cmd: Sequence[str], metric: str = "mean") -> float:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=820)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or "llama-bench failed")

    df = _parse_bench_df(result.stdout)
    tg_rows = df[df.get("n_gen", 0) > 0] if "n_gen" in df else pd.DataFrame()
    pp_rows = df[df.get("n_prompt", 0) > 0] if "n_prompt" in df else pd.DataFrame()

    if metric == "tg" and not tg_rows.empty:
        return float(tg_rows["avg_ts"].iloc[0])
    if metric == "pp" and not pp_rows.empty:
        return float(pp_rows["avg_ts"].iloc[0])
    if not tg_rows.empty and not pp_rows.empty:
        return float(tg_rows["avg_ts"].iloc[0] + pp_rows["avg_ts"].iloc[0]) / 2.0
    return 0.0


def build_warmup_command(
    llama_bench_path: str,
    model_path: str,
    ngl: int,
    n_warmup_tokens: int,
    threads: int,
) -> List[str]:
    return [
        llama_bench_path,
        "--model",
        model_path,
        "-t",
        str(threads),
        "-ngl",
        str(ngl),
        "-r",
        "2",
        "-n",
        str(n_warmup_tokens),
        "-p",
        str(n_warmup_tokens),
        "-o",
        "csv",
    ]


def warmup_until_stable(
    llama_bench_path,
    model_path,
    metric,
    ngl,
    min_runs,
    n_warmup_runs,
    n_warmup_tokens,
    max_threads,
    stability_window: int = 3,
    tolerance: float = 0.03,
):
    history: List[float] = []
    target_runs = max(min_runs, n_warmup_runs)
    cmd = build_warmup_command(
        llama_bench_path=llama_bench_path,
        model_path=model_path,
        ngl=ngl,
        n_warmup_tokens=n_warmup_tokens,
        threads=max_threads,
    )

    for _ in range(target_runs):
        performance = run_llama_bench_with_csv(cmd, metric)
        history.append(performance)
        if len(history) >= max(min_runs, stability_window):
            recent = history[-stability_window:]
            mean_recent = sum(recent) / len(recent)
            if mean_recent and max(abs(item - mean_recent) / mean_recent for item in recent) <= tolerance:
                break
    return history


def _gpu_power_hint_watts(profile: HardwareProfile) -> float:
    if profile.is_blackwell:
        return 320.0
    if profile.gpu_name != "unknown":
        return 250.0
    return 95.0


def _approximate_vram_used_mb(gpu_layers: int, batch: int, ubatch: int, cpu_offload_ratio: float) -> float:
    layer_cost = gpu_layers * 96.0
    batch_cost = batch * 0.35
    ubatch_cost = ubatch * 0.2
    return max(256.0, (layer_cost + batch_cost + ubatch_cost) * (1.0 - cpu_offload_ratio * 0.45))


def _compute_multi_metrics(
    df: pd.DataFrame,
    params: Dict[str, object],
    hardware_profile: HardwareProfile,
    n_tokens: int,
) -> BenchmarkMetrics:
    tg_rows = df[df.get("n_gen", 0) > 0] if "n_gen" in df else pd.DataFrame()
    pp_rows = df[df.get("n_prompt", 0) > 0] if "n_prompt" in df else pd.DataFrame()
    tg_row = tg_rows.iloc[0] if not tg_rows.empty else pd.Series(dtype=float)
    pp_row = pp_rows.iloc[0] if not pp_rows.empty else pd.Series(dtype=float)

    tg_tps = _pick_value(tg_row, "avg_ts", default=0.0)
    pp_tps = _pick_value(pp_row, "avg_ts", default=tg_tps)
    p99_ttft_ms = _pick_value(pp_row, "p99_latency_ms", "p99_ms", default=(1000.0 / max(pp_tps, 0.001)))
    p99_itl_ms = _pick_value(tg_row, "p99_latency_ms", "p99_ms", default=(1000.0 / max(tg_tps, 0.001)))

    cpu_offload_ratio = float(params.get("cpu_offload_ratio", 0.0))
    power_draw_watts = _gpu_power_hint_watts(hardware_profile) + (float(params["threads"]) * 2.5)
    tokens_per_joule = max(tg_tps, 0.0) / max(power_draw_watts, 1.0)
    vram_used_mb = _approximate_vram_used_mb(
        gpu_layers=int(params["gpu_layers"]),
        batch=int(params["batch"]),
        ubatch=int(params["ubatch"]),
        cpu_offload_ratio=cpu_offload_ratio,
    )
    vram_efficiency = n_tokens / max(vram_used_mb, 1.0)

    flops = (int(params["batch"]) + int(params["ubatch"])) * max(n_tokens, 1) * 2.0
    bytes_touched = vram_used_mb * 1024.0 * 1024.0
    arithmetic_intensity = flops / max(bytes_touched, 1.0)

    formulas = {
        "P99_TTFT_ms": "P99_TTFT = p99(prompt_latency_ms) or 1000 / prompt_tokens_per_second",
        "P99_ITL_ms": "P99_ITL = p99(generation_latency_ms) or 1000 / generation_tokens_per_second",
        "Tokens_Per_Joule": "Tokens_Per_Joule = generation_tokens_per_second / approx_power_draw_watts",
        "VRAM_Efficiency": "VRAM_Efficiency = generated_tokens / vram_used_mb",
        "Arithmetic_Intensity": "I = FLOPs / Bytes",
    }

    return BenchmarkMetrics(
        p99_ttft_ms=p99_ttft_ms,
        p99_itl_ms=p99_itl_ms,
        tokens_per_joule=tokens_per_joule,
        vram_efficiency=vram_efficiency,
        prompt_tokens_per_second=pp_tps,
        generation_tokens_per_second=tg_tps,
        power_draw_watts=power_draw_watts,
        vram_used_mb=vram_used_mb,
        arithmetic_intensity=arithmetic_intensity,
        formulas=formulas,
    )


def _run_multi_metric_bench(cmd: Sequence[str], params: Dict[str, object], hardware_profile: HardwareProfile, n_tokens: int) -> BenchmarkMetrics:
    started = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or "llama-bench failed")
    df = _parse_bench_df(result.stdout)
    metrics = _compute_multi_metrics(df, params, hardware_profile, n_tokens)
    elapsed = time.perf_counter() - started
    metrics.formulas["Wall_Time_s"] = f"Wall_Time = {elapsed:.3f}"
    return metrics


def _base_trial_params(trial, hardware_profile: HardwareProfile) -> Dict[str, object]:
    batch = trial.suggest_int("batch", SEARCH_SPACE["batch_size"]["low"], SEARCH_SPACE["batch_size"]["high"])
    ubatch = trial.suggest_int("ubatch", SEARCH_SPACE["ubatch_size"]["low"], min(batch, SEARCH_SPACE["ubatch_size"]["high"]))
    threads = trial.suggest_int("threads", SEARCH_SPACE["threads"]["low"], min(SEARCH_SPACE["threads"]["high"], hardware_profile.recommended_threads))
    gpu_layers = trial.suggest_int("gpu_layers", SEARCH_SPACE["gpu_layers"]["low"], SEARCH_SPACE["gpu_layers"]["high"])
    cpu_offload_ratio = trial.suggest_float(
        "cpu_offload_ratio",
        SEARCH_SPACE["cpu_offload_ratio"]["low"],
        SEARCH_SPACE["cpu_offload_ratio"]["high"],
    )
    micro_batch_ratio = trial.suggest_float(
        "micro_batch_ratio",
        SEARCH_SPACE["micro_batch_ratio"]["low"],
        SEARCH_SPACE["micro_batch_ratio"]["high"],
    )
    params = {
        "batch": batch,
        "ubatch": ubatch,
        "threads": threads,
        "gpu_layers": gpu_layers,
        "flash_attn": trial.suggest_categorical("flash_attn", SEARCH_SPACE["flash_attn"]),
        "override_tensor": trial.suggest_categorical("override_tensor", SEARCH_SPACE["override_spc"]),
        "cpu_offload_ratio": cpu_offload_ratio,
        "micro_batch_ratio": micro_batch_ratio,
    }
    return params


def _build_benchmark_command(
    llama_bench_path: str,
    model_path: str,
    repeat: int,
    n_tokens: int,
    tuned_params: Dict[str, object],
    hardware_profile: HardwareProfile,
) -> List[str]:
    cmd = [
        llama_bench_path,
        "--model",
        model_path,
        "--batch-size",
        str(tuned_params["batch"]),
        "--ubatch-size",
        str(tuned_params["ubatch"]),
        "--threads",
        str(tuned_params["threads"]),
        "-ngl",
        str(tuned_params["gpu_layers"]),
        "-r",
        str(repeat),
        "-n",
        str(n_tokens),
        "-p",
        str(max(2 * n_tokens, n_tokens)),
        "-o",
        "csv",
        "--no-warmup",
    ]

    if tuned_params.get("flash_attn") == 1:
        cmd.append("--flash-attn")
    override_key = tuned_params.get("override_tensor", "none")
    if override_key != "none":
        cmd.extend(["--override-tensor", OVERRIDE_PATTERNS[override_key]])

    draft_model = tuned_params.get("draft_model")
    if draft_model:
        cmd.extend(
            [
                "-md",
                str(draft_model),
                "--draft-min",
                str(tuned_params.get("draft_min", 0)),
                "--draft-max",
                str(tuned_params.get("draft_max", 0)),
                "--ngld",
                str(tuned_params.get("draft_gpu_layers", 0)),
            ]
        )

    if tuned_params.get("kv_eviction_policy") in {"h2o", "chunkkv"}:
        cmd.extend(
            [
                "--kv-evict",
                str(tuned_params["kv_eviction_policy"]),
                "--h2o-heavy-hitters",
                str(tuned_params.get("h2o_heavy_hitters", 0)),
                "--h2o-window",
                str(tuned_params.get("h2o_window", 0)),
            ]
        )

    if tuned_params.get("arch") != "lfm":
        cmd.extend(
            [
                "--cache-type-k",
                str(tuned_params.get("cache_type_k", "f16")),
                "--cache-type-v",
                str(tuned_params.get("cache_type_v", "f16")),
            ]
        )

    if hardware_profile.enable_nvfp4:
        cmd.append("--nvfp4")
    if hardware_profile.enable_pdl:
        cmd.append("--pdl")
    return cmd


def _objective(
    trial: optuna.trial.Trial,
    n_tokens: int,
    repeat: int,
    llama_bench_path: str,
    model_path: str,
    arch: str,
    hardware_profile: HardwareProfile,
):
    params = _base_trial_params(trial, hardware_profile)
    kv_config = build_kv_cache_config(trial)
    params.update(asdict(kv_config))

    if arch != "mtp":
        params.update(asdict(build_speculative_config(trial, model_path)))
    else:
        params.update({"draft_model": None, "draft_min": 0, "draft_max": 0, "draft_gpu_layers": 0})

    tuned_params = apply_architecture_constraints(
        arch=arch,
        params=params,
        recommended_threads=hardware_profile.recommended_threads,
        optimal_offload_ratio=hardware_profile.optimal_offload_ratio,
    )
    cmd = _build_benchmark_command(
        llama_bench_path=llama_bench_path,
        model_path=model_path,
        repeat=repeat,
        n_tokens=n_tokens,
        tuned_params=tuned_params,
        hardware_profile=hardware_profile,
    )

    try:
        metrics = _run_multi_metric_bench(cmd, tuned_params, hardware_profile, n_tokens)
    except Exception as exc:
        trial.set_user_attr("failure", str(exc))
        return math.inf, math.inf, 0.0, 0.0

    trial.set_user_attr("command", shlex.join(cmd))
    trial.set_user_attr("hardware_profile", asdict(hardware_profile))
    trial.set_user_attr("metrics", asdict(metrics))
    trial.set_user_attr("tuned_params", tuned_params)
    trial.set_user_attr("debug_formulas", metrics.formulas)
    trial.set_user_attr("arithmetic_intensity", metrics.arithmetic_intensity)
    return (
        metrics.p99_ttft_ms,
        metrics.p99_itl_ms,
        metrics.tokens_per_joule,
        metrics.vram_efficiency,
    )


def _select_best_trial(study: optuna.study.Study) -> optuna.trial.FrozenTrial:
    def score(trial: optuna.trial.FrozenTrial) -> float:
        ttft, itl, tokens_per_joule, vram_eff = trial.values
        return (-ttft) + (-itl) + (tokens_per_joule * 100.0) + (vram_eff * 1000.0)

    complete_trials = [trial for trial in study.best_trials if trial.values]
    if not complete_trials:
        raise RuntimeError("No successful trials completed.")
    return max(complete_trials, key=score)


def _format_server_command(
    llama_bin_path: str,
    model_path: str,
    best_trial: optuna.trial.FrozenTrial,
    arch: str,
) -> str:
    attrs = best_trial.user_attrs
    hardware = attrs.get("hardware_profile", {})
    tuned = dict(attrs.get("tuned_params", {}))
    if not tuned:
        tuned = apply_architecture_constraints(
            arch=arch,
            params=dict(best_trial.params),
            recommended_threads=int(hardware.get("recommended_threads", max_threads)),
            optimal_offload_ratio=float(hardware.get("optimal_offload_ratio", 1.0)),
        )
    command = [
        "python",
        "-m",
        "llama_cpp.server",
        "--model",
        model_path,
        "--threads",
        str(tuned["threads"]),
        "--batch-size",
        str(tuned["batch"]),
        "--ubatch-size",
        str(tuned["ubatch"]),
        "--n-gpu-layers",
        str(tuned["gpu_layers"]),
    ]
    if tuned.get("flash_attn") == 1:
        command.append("--flash-attn")
    if tuned.get("override_tensor", "none") != "none":
        command.extend(["--override-tensor", OVERRIDE_PATTERNS[tuned["override_tensor"]]])
    if tuned.get("draft_model"):
        command.extend(
            [
                "--draft-model",
                str(tuned["draft_model"]),
                "--draft-min",
                str(tuned["draft_min"]),
                "--draft-max",
                str(tuned["draft_max"]),
                "--ngld",
                str(tuned["draft_gpu_layers"]),
            ]
        )
    if tuned.get("arch") != "lfm":
        command.extend(["--cache-type-k", tuned.get("cache_type_k", "f16"), "--cache-type-v", tuned.get("cache_type_v", "f16")])
    if tuned.get("kv_eviction_policy") in {"h2o", "chunkkv"}:
        command.extend(
            [
                "--kv-evict",
                str(tuned["kv_eviction_policy"]),
                "--h2o-heavy-hitters",
                str(tuned.get("h2o_heavy_hitters", 0)),
                "--h2o-window",
                str(tuned.get("h2o_window", 0)),
            ]
        )
    if hardware.get("enable_nvfp4"):
        command.append("--nvfp4")
    if hardware.get("enable_pdl"):
        command.append("--pdl")
    return shlex.join(command)


def format_optimal_output_block(model_path: str, draft_model_name: str, launch_command: str) -> str:
    return "\n".join(
        [
            "=" * 50,
            "[TEST_llama-optimus] OPTIMAL ORCHESTRATION FOUND:",
            f"Target: {Path(model_path).name} | Draft: {draft_model_name}",
            f"Launch Command: {launch_command}",
            "=" * 50,
        ]
    )


def run_optimization(
    n_trials,
    n_tokens,
    metric,
    repeat,
    llama_bench_path,
    model_path,
    llama_bin_path,
    override_mode,
    arch,
    hardware_profile: Optional[HardwareProfile] = None,
):
    del metric
    del override_mode
    profile = hardware_profile or build_hardware_profile()
    sampler = TPESampler(multivariate=True)
    study = optuna.create_study(
        directions=["minimize", "minimize", "maximize", "maximize"],
        sampler=sampler,
    )
    study.optimize(
        lambda trial: _objective(
            trial,
            n_tokens=n_tokens,
            repeat=repeat,
            llama_bench_path=llama_bench_path,
            model_path=model_path,
            arch=arch,
            hardware_profile=profile,
        ),
        n_trials=n_trials,
    )

    best_trial = _select_best_trial(study)
    tuned_params = best_trial.user_attrs.get("tuned_params", {})
    draft_model_name = tuned_params.get("draft_model") or "disabled"
    launch_command = _format_server_command(
        llama_bin_path=llama_bin_path,
        model_path=model_path,
        best_trial=best_trial,
        arch=arch,
    )

    print(format_optimal_output_block(model_path, draft_model_name, launch_command))

    return {
        "study": study,
        "best_trial": best_trial,
        "hardware_profile": profile,
        "launch_command": launch_command,
    }
