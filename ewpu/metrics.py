"""Timing and benchmark records."""
from dataclasses import dataclass, field

@dataclass
class Metrics:
    execution_time: float = 0.0
    cpu_time: float = 0.0
    gpu_time: float = 0.0
    transfer_time: float = 0.0
    samples: list[dict] = field(default_factory=list)
