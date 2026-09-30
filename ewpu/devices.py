"""Common device interfaces used by the scheduler."""
from abc import ABC, abstractmethod
import numpy as np

class DeviceError(RuntimeError):
    pass

class Device(ABC):
    name: str
    @abstractmethod
    def run(self, task: str, data):
        raise NotImplementedError

class CPUDevice(Device):
    name = "CPU"
    def run(self, task: str, data):
        if task == "matmul":
            a, b = data
            return np.matmul(a, b)
        if task == "elementwise":
            return np.sin(data) * np.cos(data)
        raise ValueError(f"Unsupported task: {task}")

class GPUDevice(Device):
    name = "GPU"
    def __init__(self):
        try:
            import cupy as cp
            self.cp = cp
            if cp.cuda.runtime.getDeviceCount() < 1:
                raise DeviceError("No CUDA device")
        except Exception as exc:
            raise DeviceError("CuPy/CUDA GPU unavailable") from exc

    def run(self, task: str, data):
        try:
            cp = self.cp
            if task == "matmul":
                a, b = data
                return cp.matmul(cp.asarray(a), cp.asarray(b))
            if task == "elementwise":
                return cp.sin(cp.asarray(data)) * cp.cos(cp.asarray(data))
            raise ValueError(f"Unsupported task: {task}")
        except Exception as exc:
            raise DeviceError(f"GPU task failed: {exc}") from exc
