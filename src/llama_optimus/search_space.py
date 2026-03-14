import os

from .override_patterns import OVERRIDE_PATTERNS

max_threads = os.cpu_count() or 1

ARCH_CHOICES = ["transformer", "lfm", "bitnet", "diffused", "mtp"]
CACHE_TYPES = ["f16", "q8_0", "q4_0", "mixed"]
KV_EVICTION_POLICIES = ["disabled", "h2o", "chunkkv"]

SEARCH_SPACE = {
    "batch_size": {"low": 8, "high": 16384},
    "ubatch_size": {"low": 4, "high": 8192},
    "threads": {"low": 1, "high": max_threads},
    "gpu_layers": {"low": 0, "high": 149},
    "flash_attn": [0, 1],
    "override_spc": list(OVERRIDE_PATTERNS.keys()),
    "draft_max": {"low": 4, "high": 32},
    "draft_min": {"low": 1, "high": 8},
    "draft_gpu_layers": {"low": 0, "high": 64},
    "h2o_heavy_hitters": {"low": 8, "high": 128},
    "h2o_window": {"low": 128, "high": 4096},
    "cpu_offload_ratio": {"low": 0.0, "high": 1.0},
    "micro_batch_ratio": {"low": 0.1, "high": 1.0},
}

