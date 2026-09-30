"""Hardware detection for eWPU."""
from dataclasses import dataclass
import os

@dataclass(frozen=True)
class Hardware:
    cpu_cores: int
    gpu_available: bool
    gpu_name: str | None = None

def detect_hardware() -> Hardware:
    cores = os.cpu_count() or 1
    gpu_available = False
    gpu_name = None
    try:
        import cupy as cp
        count = cp.cuda.runtime.getDeviceCount()
        if count:
            gpu_available = True
            props = cp.cuda.runtime.getDeviceProperties(0)
            name = props.get("name", b"Unknown GPU")
            gpu_name = name.decode(errors="replace") if isinstance(name, bytes) else str(name)
    except Exception:
        pass
    return Hardware(cores, gpu_available, gpu_name)
