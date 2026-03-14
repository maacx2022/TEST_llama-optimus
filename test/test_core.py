from types import SimpleNamespace

import pandas as pd

from llama_optimus.core import (
    _compute_multi_metrics,
    _format_server_command,
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
    block = format_optimal_output_block(
        "/models/qwen3-32b.gguf",
        "/models/qwen3-8b.gguf",
        "python -m llama_cpp.server --model /models/qwen3-32b.gguf",
    )

    assert block.splitlines()[0] == "=" * 50
    assert "[TEST_llama-optimus] OPTIMAL ORCHESTRATION FOUND:" in block
    assert "Target: qwen3-32b.gguf | Draft: /models/qwen3-8b.gguf" in block
    assert "Launch Command: python -m llama_cpp.server --model /models/qwen3-32b.gguf" in block
