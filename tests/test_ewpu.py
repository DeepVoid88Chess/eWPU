import numpy as np
import pytest
from ewpu import run
from ewpu.scheduler import Scheduler

def test_cpu_elementwise_matches_reference():
    x = np.linspace(-2, 2, 1000)
    np.testing.assert_allclose(run("elementwise", x, mode="cpu").value, np.sin(x) * np.cos(x))

def test_small_workload_stays_on_cpu():
    x = np.arange(10.0)
    assert run("elementwise", x, scheduler=Scheduler(small_input_elements=100)).devices_used == ["CPU"]

def test_split_plan_for_large_workload():
    x = np.arange(200.0)
    plan = Scheduler(small_input_elements=100, gpu_fraction=0.7).plan("elementwise", x, True)
    assert plan.cpu_fraction == pytest.approx(0.3)
    assert plan.gpu_fraction == pytest.approx(0.7)

def test_matmul_matches_reference():
    rng = np.random.default_rng(42)
    a, b = rng.random((8, 4)), rng.random((4, 3))
    np.testing.assert_allclose(run("matmul", (a, b), mode="cpu").value, a @ b)

def test_empty_elementwise():
    assert run("elementwise", np.array([]), mode="cpu").value.size == 0

def test_non_divisible_split():
    x = np.arange(101.0)
    plan = Scheduler(small_input_elements=10, gpu_fraction=0.7).plan("elementwise", x, True)
    assert 0 <= int(len(x) * plan.cpu_fraction) <= len(x)

def test_speedup_is_never_invented():
    result = run("elementwise", np.arange(1000.0), mode="cpu")
    assert result.speedup is None
