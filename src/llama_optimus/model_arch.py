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


def detect_architecture(model_path: str, llama_bin_path: Optional[str] = None) -> str:
    signature = inspect_model_signature(model_path, llama_bin_path)
    lowered = f"{Path(model_path).name}\n{signature}".lower()

    metadata_patterns = [
        ("lfm", [r"\bgeneral\.architecture\b[\s\S]{0,80}\blfm", r"\blfm2\.", r"\bshortconv\b", r"\bliquidai\b", r"\blfm2\b"]),
        ("diffused", [r"\bdiffused\b", r"\blada\b", r"\bmoe\b", r"\bexpert[s]?\b", r"\bffn_.*_exps\b"]),
        ("bitnet", [r"\bbitnet\b", r"\b1\.58b\b", r"\bternary\b"]),
        ("mtp", [r"\bmtp\b", r"\bmulti-token prediction\b", r"\bself-speculat"]),
    ]

    for arch, patterns in metadata_patterns:
        if any(re.search(pattern, lowered) for pattern in patterns):
            return arch

    architecture_aliases = {
        "transformer": ("transformer", "llama", "qwen", "gemma", "mistral", "glm"),
    }
    for arch, aliases in architecture_aliases.items():
        if any(alias in lowered for alias in aliases):
            return arch

    return "transformer"

