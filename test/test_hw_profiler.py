from llama_optimus.hw_profiler import is_blackwell_gpu, recommend_dynamic_offload_ratio


def test_blackwell_detection_handles_rtx_pro_6000():
    assert is_blackwell_gpu("NVIDIA RTX PRO 6000 Blackwell")


def test_dynamic_offload_ratio_stays_bounded():
    ratio = recommend_dynamic_offload_ratio({"vram": 0.2, "system_ram": 1.0, "nvme": 6.0})
    assert 0.2 <= ratio <= 1.0
