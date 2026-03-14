from pathlib import Path

from llama_optimus.model_arch import detect_architecture, detect_max_context


def test_detect_architecture_from_lfm_metadata_signature(tmp_path: Path):
    model = tmp_path / "model.gguf"
    model.write_text("general.architecture\nlfm2\nlfm2.shortconv.l_cache\nLiquidAI\n")

    assert detect_architecture(str(model)) == "lfm"


def test_detect_architecture_from_diffused_name_hint(tmp_path: Path):
    model = tmp_path / "LADA2-8B-diffused.gguf"
    model.write_text("placeholder")

    assert detect_architecture(str(model)) == "diffused"


def test_detect_architecture_defaults_to_transformer(tmp_path: Path):
    model = tmp_path / "Qwen3-8B.gguf"
    model.write_text("general.architecture\nllama\n")

    assert detect_architecture(str(model)) == "transformer"


def test_detect_architecture_does_not_map_moa_name_to_diffused(tmp_path: Path):
    model = tmp_path / "Qwen-MoA-14B.gguf"
    model.write_text("general.architecture\nllama\n")

    assert detect_architecture(str(model)) == "transformer"


def test_detect_max_context_from_metadata_signature(tmp_path: Path):
    model = tmp_path / "LFM2.5.gguf"
    model.write_text(
        "general.architecture = lfm2\n"
        "lfm2.context_length = 262144\n"
        "lfm2.rope.scaling.original_context_length = 131072\n"
    )

    assert detect_max_context(str(model)) == 262144
