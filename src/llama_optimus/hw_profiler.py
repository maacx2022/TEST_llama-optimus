import os
import platform
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


BLACKWELL_MARKERS = (
    "blackwell",
    "rtx 50",
    "rtx pro 6000 blackwell",
    "gb200",
    "b200",
)

RYZEN_X3D_MARKERS = ("9950x3d", "9900x3d", "9800x3d", "7950x3d", "7900x3d")


@dataclass
class HardwareProfile:
    cpu_name: str
    logical_cores: int
    physical_cores: int
    recommended_threads: int
    gpu_name: str = "unknown"
    is_blackwell: bool = False
    enable_nvfp4: bool = False
    enable_pdl: bool = False
    is_x3d: bool = False
    mtds_latency_ms: Dict[str, float] = field(default_factory=dict)
    optimal_offload_ratio: float = 1.0


def _safe_run(command: List[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return ""
    return completed.stdout.strip()


def detect_cpu_name() -> str:
    cpu_name = platform.processor().strip()
    if cpu_name:
        return cpu_name

    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        match = re.search(r"model name\s+:\s+(.+)", cpuinfo.read_text())
        if match:
            return match.group(1).strip()

    return "unknown-cpu"


def detect_physical_cores() -> int:
    output = _safe_run(["lscpu", "-p=CORE"])
    if output:
        cores = {
            line.strip()
            for line in output.splitlines()
            if line and not line.startswith("#")
        }
        if cores:
            return len(cores)
    return os.cpu_count() or 1


def detect_gpu_name() -> str:
    query = _safe_run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
    )
    if query:
        return query.splitlines()[0].strip()

    lspci = _safe_run(["lspci"])
    for line in lspci.splitlines():
        lowered = line.lower()
        if "nvidia" in lowered and "vga" in lowered:
            return line.split(": ", 1)[-1].strip()
    return "unknown"


def is_blackwell_gpu(gpu_name: str) -> bool:
    lowered = gpu_name.lower()
    return any(marker in lowered for marker in BLACKWELL_MARKERS)


def _detect_x3d_recommended_threads(cpu_name: str, physical_cores: int) -> int:
    lowered = cpu_name.lower()
    if any(marker in lowered for marker in RYZEN_X3D_MARKERS):
        return min(8, physical_cores)
    return physical_cores


def _measure_filesystem_latency_ms(base_path: Path, sample_mb: int = 8) -> float:
    if not base_path.exists():
        return 0.0

    base_path.mkdir(parents=True, exist_ok=True)
    payload = b"0" * 1024 * 1024
    start = time.perf_counter()
    with tempfile.NamedTemporaryFile(dir=base_path, delete=True) as handle:
        for _ in range(sample_mb):
            handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        handle.seek(0)
        handle.read()
    return (time.perf_counter() - start) * 1000


def profile_mtds(storage_roots: Optional[Dict[str, Path]] = None) -> Dict[str, float]:
    roots = storage_roots or {
        "system_ram": Path("/tmp"),
        "nvme": Path.home(),
    }
    latencies = {
        tier: _measure_filesystem_latency_ms(path)
        for tier, path in roots.items()
    }
    latencies["vram"] = max(0.05, latencies.get("system_ram", 1.0) * 0.35)
    return latencies


def recommend_dynamic_offload_ratio(latencies_ms: Dict[str, float]) -> float:
    vram = latencies_ms.get("vram", 0.1)
    ram = latencies_ms.get("system_ram", 1.0)
    nvme = latencies_ms.get("nvme", 5.0)
    weighted = (1.5 * vram) + ram + (0.5 * nvme)
    ratio = 1.0 - min(0.75, (weighted / max(nvme, 0.1)) * 0.2)
    return round(max(0.2, min(1.0, ratio)), 3)


def build_hardware_profile() -> HardwareProfile:
    cpu_name = detect_cpu_name()
    physical_cores = detect_physical_cores()
    logical_cores = os.cpu_count() or physical_cores
    gpu_name = detect_gpu_name()
    latencies = profile_mtds()
    recommended_threads = _detect_x3d_recommended_threads(cpu_name, physical_cores)
    blackwell = is_blackwell_gpu(gpu_name)

    return HardwareProfile(
        cpu_name=cpu_name,
        logical_cores=logical_cores,
        physical_cores=physical_cores,
        recommended_threads=recommended_threads,
        gpu_name=gpu_name,
        is_blackwell=blackwell,
        enable_nvfp4=blackwell,
        enable_pdl=blackwell,
        is_x3d=recommended_threads < physical_cores or "x3d" in cpu_name.lower(),
        mtds_latency_ms=latencies,
        optimal_offload_ratio=recommend_dynamic_offload_ratio(latencies),
    )
