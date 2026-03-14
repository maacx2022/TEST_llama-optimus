import math
from types import SimpleNamespace

import pandas as pd

from llama_optimus.core import (
    _compute_multi_metrics,
    _format_server_command,
    _base_trial_params_for_arch,
    _select_best_trial,
    build_trial_summary,
    format_optimal_output_block,
)
from llama_optimus.hw_profiler import HardwareProfile
from llama_optimus.search_space import SEARCH_SPACE


def test_search_space_shape():
    assert isinstance(SEARCH_SPACE, dict)
    assert "batch_size" in SEARCH_SPACE
    assert "draft_max" in SEARCH_SPACE


def test_compute_multi_metrics_contains_debug_formulas():
    df = pd.DataFrame(
        [
            {"n_gen": 32, "n_prompt": 0, "avg_ts": 80.0},
            {"n_gen": 0, "n_prompt": 64, "avg_ts": 160.0},
        ]
    )
    profile = HardwareProfile(
        cpu_name="AMD Ryzen 9 9950X3D",
        logical_cores=32,
        physical_cores=16,
        recommended_threads=8,
        gpu_name="RTX PRO 6000 Blackwell",
        is_blackwell=True,
        enable_nvfp4=True,
        enable_pdl=True,
        is_x3d=True,
    )
    params = {"batch": 256, "ubatch": 128, "gpu_layers": 80, "threads": 8, "cpu_offload_ratio": 0.2}

    metrics = _compute_multi_metrics(df, params, profile, n_tokens=128)

    assert metrics.p99_ttft_ms > 0
    assert metrics.p99_itl_ms > 0
    assert metrics.tokens_per_joule > 0
    assert metrics.vram_efficiency > 0
    assert "Arithmetic_Intensity" in metrics.formulas


def test_format_server_command_uses_tuned_params():
    trial = SimpleNamespace(
        params={"batch": 1},
        user_attrs={
            "hardware_profile": {"recommended_threads": 8, "optimal_offload_ratio": 0.7},
            "tuned_params": {
                "arch": "transformer",
                "batch": 256,
                "ubatch": 128,
                "threads": 8,
                "gpu_layers": 40,
                "flash_attn": 1,
                "override_tensor": "none",
                "draft_model": "/models/qwen3-8b.gguf",
                "draft_min": 2,
                "draft_max": 8,
                "draft_gpu_layers": 12,
                "cache_type_k": "q8_0",
                "cache_type_v": "q4_0",
                "kv_eviction_policy": "h2o",
                "h2o_heavy_hitters": 16,
                "h2o_window": 512,
            },
        },
    )

    command = _format_server_command("/llama/bin", "/models/qwen3-32b.gguf", trial, "transformer")

    assert "--draft-model" in command
    assert "--cache-type-k" in command
    assert "--kv-evict" in command


def test_optimal_output_block_matches_requested_format():
    summary = {
        "successful_trials": 12,
        "total_trials": 15,
        "best_prompt_tps": 42000.0,
        "worst_prompt_tps": 39000.0,
        "median_prompt_tps": 40123.0,
        "best_generation_tps": 768.06,
        "worst_generation_tps": 742.10,
        "median_generation_tps": 755.43,
    }
    block = format_optimal_output_block(
        "/models/qwen3-32b.gguf",
        "/models/qwen3-8b.gguf",
        "python -m llama_cpp.server --model /models/qwen3-32b.gguf",
        "/llama/bin",
        summary,
    )

    assert block.splitlines()[0] == "=" * 50
    assert "[TEST_llama-optimus] OPTIMAL ORCHESTRATION FOUND:" in block
    assert "Target: qwen3-32b.gguf | Draft: /models/qwen3-8b.gguf" in block
    assert "[FORMULA] COPY/PASTE SERVER COMMAND:" in block
    assert "BEST TG tok/sec   : 768.06" in block
    assert "MEDIAN TG tok/sec : 755.43" in block
    assert "BEST PP tok/sec   : 42000.00" in block
    assert "LLAMA_BIN=/llama/bin" in block
    assert "MODEL=/models/qwen3-32b.gguf" in block
    assert "DRAFT_MODEL=/models/qwen3-8b.gguf" in block
    assert "```bash" in block
    assert "Launch Command: python -m llama_cpp.server --model /models/qwen3-32b.gguf" in block


def test_select_best_trial_rejects_all_failed_trials():
    failed_trial = SimpleNamespace(
        number=0,
        values=(math.inf, math.inf, 0.0, 0.0),
        user_attrs={"failure": "boom", "failed_command": "llama-bench ..."},
    )
    study = SimpleNamespace(best_trials=[failed_trial], trials=[failed_trial])

    try:
        _select_best_trial(study)
    except RuntimeError as exc:
        assert "No successful trials completed" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for all-failed trials")


def test_build_trial_summary_extracts_prompt_and_generation_stats():
    successful_trial_a = SimpleNamespace(
        values=(0.1, 1.3, 2.1, 0.001),
        user_attrs={
            "metrics": {
                "prompt_tokens_per_second": 40000.0,
                "generation_tokens_per_second": 760.0,
            }
        },
    )
    successful_trial_b = SimpleNamespace(
        values=(0.11, 1.4, 2.0, 0.001),
        user_attrs={
            "metrics": {
                "prompt_tokens_per_second": 42000.0,
                "generation_tokens_per_second": 780.0,
            }
        },
    )
    failed_trial = SimpleNamespace(values=(math.inf, math.inf, 0.0, 0.0), user_attrs={})
    study = SimpleNamespace(trials=[successful_trial_a, successful_trial_b, failed_trial])

    summary = build_trial_summary(study)

    assert summary["successful_trials"] == 2
    assert summary["total_trials"] == 3
    assert summary["best_generation_tps"] == 780.0
    assert summary["worst_generation_tps"] == 760.0
    assert summary["median_generation_tps"] == 770.0
    assert summary["best_prompt_tps"] == 42000.0


def test_throughput_profile_prefers_highest_generation_tokens_per_second():
    slower_but_prettier = SimpleNamespace(
        number=0,
        values=(0.09, 1.2, 2.5, 0.002),
        user_attrs={
            "metrics": {
                "prompt_tokens_per_second": 42000.0,
                "generation_tokens_per_second": 760.0,
            },
            "tuned_params": {"gpu_layers": 80, "cpu_offload_ratio": 0.1},
        },
    )
    faster_generation = SimpleNamespace(
        number=1,
        values=(0.12, 1.4, 2.0, 0.001),
        user_attrs={
            "metrics": {
                "prompt_tokens_per_second": 39000.0,
                "generation_tokens_per_second": 780.0,
            },
            "tuned_params": {"gpu_layers": 40, "cpu_offload_ratio": 0.4},
        },
    )
    study = SimpleNamespace(best_trials=[slower_but_prettier, faster_generation], trials=[slower_but_prettier, faster_generation])

    selected = _select_best_trial(study)  # balanced still has separate behavior
    assert selected is slower_but_prettier

    from llama_optimus.core import _select_best_trial_for_profile

    selected_throughput = _select_best_trial_for_profile(study, "throughput")
    assert selected_throughput is faster_generation


class RecordingTrial:
    def __init__(self):
        self.choices = {}

    def suggest_int(self, name, low, high):
        self.choices[name] = (low, high)
        return high

    def suggest_float(self, name, low, high):
        self.choices[name] = (low, high)
        return low

    def suggest_categorical(self, name, choices):
        self.choices[name] = list(choices)
        return choices[-1]


def test_lfm_throughput_uses_high_batch_and_ubatch_candidates(tmp_path):
    model_path = tmp_path / "lfm.gguf"
    model_path.write_bytes(b"x" * 1024)
    profile = HardwareProfile(
        cpu_name="AMD Ryzen 9 9950X3D",
        logical_cores=32,
        physical_cores=16,
        recommended_threads=8,
        gpu_name="RTX PRO 6000 Blackwell",
        is_blackwell=True,
        enable_nvfp4=True,
        enable_pdl=True,
        is_x3d=True,
    )
    trial = RecordingTrial()

    params = _base_trial_params_for_arch(
        trial,
        profile,
        arch="lfm",
        model_path=str(model_path),
        profile="throughput",
    )

    assert 9084 in trial.choices["batch"]
    assert 6338 in trial.choices["ubatch"]
    assert params["batch"] >= 9084
    assert params["ubatch"] >= 6338
