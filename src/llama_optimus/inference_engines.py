import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from .search_space import CACHE_TYPES, KV_EVICTION_POLICIES, SEARCH_SPACE


@dataclass
class SpeculativeDecodingConfig:
    draft_model: Optional[str]
    draft_min: int
    draft_max: int
    draft_gpu_layers: int


@dataclass
class KVCacheConfig:
    eviction_policy: str
    heavy_hitter_count: int
    window: int
    cache_type_k: str
    cache_type_v: str


def suggest_draft_model(target_model: str) -> Optional[str]:
    model_path = Path(target_model)
    stem = model_path.stem
    lowered = stem.lower()
    hints = [
        (r"(\d+)b", lambda m: max(1, int(m.group(1)) // 4)),
        (r"(\d+)m", lambda m: max(64, int(m.group(1)) // 4)),
    ]
    for pattern, reducer in hints:
        match = re.search(pattern, lowered)
        if match:
            smaller = reducer(match)
            replacement = f"{smaller}b" if "b" in match.group(0) else f"{smaller}m"
            candidate = re.sub(pattern, replacement, stem, count=1, flags=re.IGNORECASE)
            return str(model_path.with_name(f"{candidate}{model_path.suffix}"))
    return None


def build_speculative_config(trial, target_model: str) -> SpeculativeDecodingConfig:
    draft_min = trial.suggest_int(
        "draft_min",
        SEARCH_SPACE["draft_min"]["low"],
        SEARCH_SPACE["draft_min"]["high"],
    )
    draft_max = trial.suggest_int(
        "draft_max",
        max(draft_min, SEARCH_SPACE["draft_max"]["low"]),
        SEARCH_SPACE["draft_max"]["high"],
    )
    draft_gpu_layers = trial.suggest_int(
        "draft_gpu_layers",
        SEARCH_SPACE["draft_gpu_layers"]["low"],
        SEARCH_SPACE["draft_gpu_layers"]["high"],
    )
    return SpeculativeDecodingConfig(
        draft_model=suggest_draft_model(target_model),
        draft_min=draft_min,
        draft_max=draft_max,
        draft_gpu_layers=draft_gpu_layers,
    )


def build_kv_cache_config(trial) -> KVCacheConfig:
    policy = trial.suggest_categorical("kv_eviction_policy", KV_EVICTION_POLICIES)
    heavy_hitters = trial.suggest_int(
        "h2o_heavy_hitters",
        SEARCH_SPACE["h2o_heavy_hitters"]["low"],
        SEARCH_SPACE["h2o_heavy_hitters"]["high"],
    )
    window = trial.suggest_int(
        "h2o_window",
        SEARCH_SPACE["h2o_window"]["low"],
        SEARCH_SPACE["h2o_window"]["high"],
    )
    cache_type_k = trial.suggest_categorical("cache_type_k", CACHE_TYPES)
    cache_type_v = trial.suggest_categorical("cache_type_v", CACHE_TYPES)
    return KVCacheConfig(
        eviction_policy=policy,
        heavy_hitter_count=heavy_hitters,
        window=window,
        cache_type_k=cache_type_k,
        cache_type_v=cache_type_v,
    )


def apply_architecture_constraints(
    arch: str,
    params: Dict[str, object],
    recommended_threads: int,
    optimal_offload_ratio: float,
) -> Dict[str, object]:
    tuned = dict(params)
    tuned["arch"] = arch

    if arch == "transformer":
        tuned["flash_attn"] = 1
        tuned["cpu_offload_ratio"] = min(float(tuned.get("cpu_offload_ratio", 0.0)), 0.35)
    elif arch == "diffused":
        tuned["flash_attn"] = 1
        tuned["ubatch"] = min(int(tuned["batch"]), int(tuned["ubatch"]))
        tuned["micro_batch_ratio"] = max(float(tuned.get("micro_batch_ratio", 0.5)), 0.35)
    elif arch == "lfm":
        tuned["gpu_layers"] = 999
        tuned["kv_eviction_policy"] = "disabled"
        tuned["cache_type_k"] = "f16"
        tuned["cache_type_v"] = "f16"
        tuned["cpu_offload_ratio"] = 0.0
    elif arch == "bitnet":
        tuned["gpu_layers"] = 0
        tuned["threads"] = recommended_threads
        tuned["cpu_offload_ratio"] = 1.0
        tuned["draft_model"] = None
    elif arch == "mtp":
        tuned["draft_model"] = None
        tuned["draft_min"] = 0
        tuned["draft_max"] = 0
        tuned["draft_gpu_layers"] = 0
    tuned["dynamic_offload_ratio"] = optimal_offload_ratio
    return tuned

