from pathlib import Path

from llama_optimus.model_arch import detect_architecture


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
