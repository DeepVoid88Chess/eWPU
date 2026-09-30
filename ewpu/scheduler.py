"""Initial conservative eWPU scheduler."""
from dataclasses import dataclass
import numpy as np
from .devices import CPUDevice, GPUDevice, DeviceError

@dataclass(frozen=True)
class Plan:
    cpu_fraction: float
    gpu_fraction: float
    reason: str

class Scheduler:
    def __init__(self, small_input_elements: int = 100_000, gpu_fraction: float = 0.70):
        if not 0.0 < gpu_fraction < 1.0:
            raise ValueError("gpu_fraction must be between 0 and 1")
        self.small_input_elements = small_input_elements
        self.gpu_fraction = gpu_fraction
        self.cpu = CPUDevice()

    def plan(self, task, data, gpu_available: bool) -> Plan:
        size = _size_of(data)
        if not gpu_available:
            return Plan(1.0, 0.0, "GPU unavailable")
        if size < self.small_input_elements:
            return Plan(1.0, 0.0, "workload below CPU/GPU split threshold")
        return Plan(1.0 - self.gpu_fraction, self.gpu_fraction, "large data-parallel workload")

    def execute(self, task, data, gpu_available: bool):
        plan = self.plan(task, data, gpu_available)
        if plan.gpu_fraction == 0:
            return self.cpu.run(task, data), plan, 0

        if task == "elementwise":
            array = np.asarray(data)
            cut = int(len(array) * plan.cpu_fraction)
            parts = [self.cpu.run(task, array[:cut])]
            fallbacks = 0
            try:
                parts.append(_to_numpy(GPUDevice().run(task, array[cut:])))
            except DeviceError:
                fallbacks += 1
                parts.append(self.cpu.run(task, array[cut:]))
            return np.concatenate(parts), plan, fallbacks

        if task == "matmul":
            a, b = data
            cut = int(a.shape[0] * plan.cpu_fraction)
            cpu_part = self.cpu.run(task, (a[:cut], b))
            fallbacks = 0
            try:
                gpu_part = _to_numpy(GPUDevice().run(task, (a[cut:], b)))
            except DeviceError:
                fallbacks += 1
                gpu_part = self.cpu.run(task, (a[cut:], b))
            return np.vstack([cpu_part, gpu_part]), plan, fallbacks

        raise ValueError(f"Unsupported task: {task}")

def _size_of(data) -> int:
    if isinstance(data, tuple):
        return sum(np.asarray(x).size for x in data)
    return np.asarray(data).size

def _to_numpy(value):
    return value.get() if hasattr(value, "get") else np.asarray(value)
