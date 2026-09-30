"""Public eWPU execution API."""
from dataclasses import dataclass
import time
from .detect import detect_hardware
from .scheduler import Scheduler

@dataclass
class EWPUResult:
    value: object
    devices_used: list[str]
    execution_time: float
    cpu_time: float
    gpu_time: float
    transfer_time: float
    split_ratio: float
    fallbacks: int
    speedup: float | None = None

def run(task, data, mode="auto", scheduler=None) -> EWPUResult:
    if mode not in {"auto", "cpu"}:
        raise ValueError("mode must be 'auto' or 'cpu'")
    scheduler = scheduler or Scheduler()
    hardware = detect_hardware()
    started = time.perf_counter()

    if mode == "cpu":
        value = scheduler.cpu.run(task, data)
        devices = ["CPU"]
        fallbacks = 0
        split = 0.0
    else:
        value, plan, fallbacks = scheduler.execute(task, data, hardware.gpu_available)
        devices = ["CPU"] if plan.gpu_fraction == 0 else ["CPU", "GPU"]
        split = plan.gpu_fraction

    elapsed = time.perf_counter() - started
    return EWPUResult(
        value=value,
        devices_used=devices,
        execution_time=elapsed,
        cpu_time=elapsed if devices == ["CPU"] else 0.0,
        gpu_time=0.0,
        transfer_time=0.0,
        split_ratio=split,
        fallbacks=fallbacks,
        speedup=None,
    )
