# __init__.py

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "0.1.9"

from .core import estimate_max_ngl, run_llama_bench_with_csv, run_optimization
from .hw_profiler import HardwareProfile, build_hardware_profile
from .model_arch import detect_architecture
from .search_space import SEARCH_SPACE
