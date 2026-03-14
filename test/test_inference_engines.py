from llama_optimus.inference_engines import (
    apply_architecture_constraints,
    build_kv_cache_config,
    build_speculative_config,
    suggest_draft_model,
)


class FakeTrial:
    def suggest_int(self, name, low, high):
        return low if "min" in name else high

    def suggest_categorical(self, name, choices):
        return choices[-1]


class RecordingSpeculativeTrial:
    def __init__(self):
        self.ranges = {}

    def suggest_int(self, name, low, high):
        self.ranges[name] = (low, high)
        if name == "draft_min":
            return high
        if name == "draft_max":
            return low
        return high

    def suggest_categorical(self, name, choices):
        return choices[-1]


def test_suggest_draft_model_reduces_model_size_hint():
    draft = suggest_draft_model("/models/qwen3-32b-q4.gguf")
    assert draft is not None
    assert "8b" in draft.lower()


def test_architecture_constraints_for_bitnet_and_lfm():
    bitnet = apply_architecture_constraints(
        arch="bitnet",
        params={"batch": 64, "ubatch": 32, "threads": 24, "gpu_layers": 40},
        recommended_threads=8,
        optimal_offload_ratio=0.75,
        profile="balanced",
        gpu_available=False,
    )
    lfm = apply_architecture_constraints(
        arch="lfm",
        params={"batch": 64, "ubatch": 32, "threads": 24, "gpu_layers": 40},
        recommended_threads=8,
        optimal_offload_ratio=0.75,
        profile="balanced",
        gpu_available=True,
    )

    assert bitnet["gpu_layers"] == 0
    assert bitnet["threads"] == 8
    assert lfm["gpu_layers"] == 999
    assert lfm["kv_eviction_policy"] == "disabled"


def test_build_speculative_and_kv_configs_cover_new_knobs():
    trial = FakeTrial()

    speculative = build_speculative_config(trial, "/models/qwen3-32b-q4.gguf")
    kv_config = build_kv_cache_config(trial)

    assert speculative.draft_model is not None
    assert speculative.draft_max >= speculative.draft_min
    assert kv_config.eviction_policy in {"disabled", "h2o", "chunkkv"}
    assert kv_config.cache_type_k in {"f16", "q8_0", "q4_0"}


def test_build_speculative_config_uses_static_range_and_clamps_draft_max():
    trial = RecordingSpeculativeTrial()

    speculative = build_speculative_config(trial, "/models/qwen3-32b-q4.gguf")

    assert trial.ranges["draft_max"] == (4, 32)
    assert speculative.draft_min == 8
    assert speculative.draft_max == 8


def test_architecture_constraints_for_diffused_and_mtp():
    diffused = apply_architecture_constraints(
        arch="diffused",
        params={"batch": 64, "ubatch": 80, "threads": 24, "gpu_layers": 40, "micro_batch_ratio": 0.2},
        recommended_threads=8,
        optimal_offload_ratio=0.75,
        profile="balanced",
        gpu_available=True,
    )
    mtp = apply_architecture_constraints(
        arch="mtp",
        params={"batch": 64, "ubatch": 32, "threads": 24, "gpu_layers": 40, "draft_model": "/x.gguf"},
        recommended_threads=8,
        optimal_offload_ratio=0.75,
        profile="balanced",
        gpu_available=True,
    )

    assert diffused["flash_attn"] == 1
    assert diffused["ubatch"] <= diffused["batch"]
    assert mtp["draft_model"] is None
    assert mtp["draft_max"] == 0


def test_lfm_falls_back_to_cpu_when_gpu_unavailable():
    lfm = apply_architecture_constraints(
        arch="lfm",
        params={"batch": 64, "ubatch": 32, "threads": 24, "gpu_layers": 40},
        recommended_threads=8,
        optimal_offload_ratio=0.75,
        profile="balanced",
        gpu_available=False,
    )

    assert lfm["gpu_layers"] == 0
    assert lfm["cpu_offload_ratio"] == 1.0
    assert lfm["draft_model"] is None


def test_gpu_first_profile_keeps_gpu_layers_high_and_disables_override():
    tuned = apply_architecture_constraints(
        arch="diffused",
        params={
            "batch": 128,
            "ubatch": 64,
            "threads": 8,
            "gpu_layers": 4,
            "cpu_offload_ratio": 0.8,
            "override_tensor": "ffn_cpu_all",
            "cache_type_k": "q4_0",
            "cache_type_v": "q8_0",
        },
        recommended_threads=8,
        optimal_offload_ratio=0.75,
        profile="gpu-first",
        gpu_available=True,
    )

    assert tuned["gpu_layers"] >= 32
    assert tuned["override_tensor"] == "none"
    assert tuned["cache_type_k"] == "f16"
    assert tuned["cache_type_v"] == "f16"
