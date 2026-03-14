import os
import re
import subprocess
from pathlib import Path
from typing import Optional

from .search_space import ARCH_CHOICES


AUTO_ARCH_CHOICES = ["auto", *ARCH_CHOICES]


def _safe_run(command):
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
    except Exception:
        return ""
    return (completed.stdout or "") + "\n" + (completed.stderr or "")


def _llama_gguf_path(llama_bin_path: Optional[str]) -> Optional[str]:
    if not llama_bin_path:
        return None
    candidate = Path(llama_bin_path) / "llama-gguf"
    return str(candidate) if candidate.is_file() else None


def inspect_model_signature(model_path: str, llama_bin_path: Optional[str] = None) -> str:
    gguf_bin = _llama_gguf_path(llama_bin_path)
    if gguf_bin:
        output = _safe_run([gguf_bin, model_path, "r", "n"])
        if output.strip():
            return output

    try:
        with open(model_path, "rb") as handle:
            blob = handle.read(2_000_000)
    except OSError:
        return ""

    ascii_chunks = re.findall(rb"[ -~]{4,}", blob)
    return "\n".join(chunk.decode("ascii", errors="ignore") for chunk in ascii_chunks)


def detect_max_context(model_path: str, llama_bin_path: Optional[str] = None) -> Optional[int]:
    gguf_root = Path("/home/macko/llm/llama.cpp/gguf-py")
    if gguf_root.is_dir():
        import sys

        sys.path.insert(0, str(gguf_root))
        try:
            from gguf.gguf_reader import GGUFReader

            reader = GGUFReader(model_path)
            for key in (
                "lfm2.context_length",
                "qwen2.context_length",
                "qwen3.context_length",
                "llama.context_length",
                "general.context_length",
            ):
                field = reader.fields.get(key)
                if field is not None:
                    value = field.parts[field.data[0]]
                    if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
                        return int(value[0])
                    return int(value)
        except Exception:
            pass
        finally:
            try:
                sys.path.remove(str(gguf_root))
            except ValueError:
                pass

    signature = inspect_model_signature(model_path, llama_bin_path)
    if not signature:
        return None

    lowered = signature.lower()
    patterns = [
        r"context_length[^=\n]*=\s*(\d+)",
        r"n_ctx_train[^=\n]*=\s*(\d+)",
        r"max_position_embeddings[^=\n]*=\s*(\d+)",
        r"original_context_length[^=\n]*=\s*(\d+)",
    ]
    candidates = []
    for pattern in patterns:
        candidates.extend(int(match) for match in re.findall(pattern, lowered))
    if not candidates:
        return None
    return max(candidates)


def detect_architecture(model_path: str, llama_bin_path: Optional[str] = None) -> str:
    signature = inspect_model_signature(model_path, llama_bin_path)
    model_name = Path(model_path).name.lower()
    lowered = f"{model_name}\n{signature}".lower()

    strong_patterns = [
        ("lfm", [r"\bgeneral\.architecture\b[\s\S]{0,80}\blfm", r"\blfm2\.", r"\bshortconv\b", r"\bliquidai\b", r"\blfm2\b"]),
        ("diffused", [r"\bgeneral\.architecture\b[\s\S]{0,80}\b.*moe", r"\bffn_.*_exps\b", r"\bexpert[s]?\b", r"\bqwen.*moe\b"]),
        ("bitnet", [r"\bgeneral\.architecture\b[\s\S]{0,80}\bbitnet", r"\bbitnet\b", r"\bternary\b"]),
        ("mtp", [r"\bgeneral\.architecture\b[\s\S]{0,80}\bmtp", r"\bmulti-token prediction\b", r"\bself-speculat"]),
    ]

    for arch, patterns in strong_patterns:
        if any(re.search(pattern, lowered) for pattern in patterns):
            return arch

    if any(token in model_name for token in ("moa", "mixture-of-agents", "mixture_of_agents")):
        return "transformer"

    filename_patterns = [
        ("lfm", [r"\blfm\b", r"\bliquid\b"]),
        ("diffused", [r"\bdiffused\b", r"\blada\b"]),
        ("bitnet", [r"\bbitnet\b", r"\b1\.58b\b"]),
        ("mtp", [r"\bmtp\b"]),
    ]

    for arch, patterns in filename_patterns:
        if any(re.search(pattern, model_name) for pattern in patterns):
            return arch

    architecture_aliases = {
        "transformer": ("transformer", "llama", "qwen", "gemma", "mistral", "glm"),
    }
    for arch, aliases in architecture_aliases.items():
        if any(alias in model_name for alias in aliases):
            return arch

    fallback_patterns = [
        ("lfm", [r"\blfm2\b"]),
        ("diffused", [r"\bmoe\b"]),
        ("bitnet", [r"\bbitnet\b"]),
        ("mtp", [r"\bmtp\b"]),
    ]

    for arch, patterns in fallback_patterns:
        if any(re.search(pattern, lowered) for pattern in patterns):
            return arch

    return "transformer"
