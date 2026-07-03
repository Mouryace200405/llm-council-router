"""
Energy and Power Efficiency Tracking Utilities
-----------------------------------------------
Provides lightweight estimation of energy consumption per inference call
so the pipeline can be evaluated on energy efficiency — a primary metric.
"""

import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class EnergySample:
    timestamp: float
    power_w: Optional[float]
    gpu_util_pct: Optional[float]
    gpu_mem_mb: Optional[float]


@dataclass
class EnergyReport:
    total_energy_j: float
    avg_power_w: float
    peak_power_w: float
    duration_s: float
    gpu_util_avg: Optional[float] = None
    samples: List[EnergySample] = field(default_factory=list)


class EnergyMonitor:
    """Samples GPU power metrics during inference via nvidia-smi."""

    def __init__(self, sample_interval_s: float = 0.1):
        self._interval = sample_interval_s
        self._samples: List[EnergySample] = []
        self._running = False
        self._start = 0.0
        self._gpu_available = self._check_gpu()

    def start(self):
        self._samples.clear()
        self._running = True
        self._start = time.time()

    def sample(self):
        if not self._running:
            return
        if self._gpu_available:
            power, util, mem = self._read_nvidia_smi()
        else:
            power, util, mem = None, None, None
        self._samples.append(EnergySample(
            timestamp=time.time(),
            power_w=power,
            gpu_util_pct=util,
            gpu_mem_mb=mem,
        ))

    def stop(self) -> EnergyReport:
        self._running = False
        elapsed = time.time() - self._start

        powers = [s.power_w for s in self._samples if s.power_w is not None]
        utils = [s.gpu_util_pct for s in self._samples if s.gpu_util_pct is not None]

        if not powers:
            return EnergyReport(
                total_energy_j=0.0,
                avg_power_w=0.0,
                peak_power_w=0.0,
                duration_s=elapsed,
            )

        avg_power = sum(powers) / len(powers)
        peak_power = max(powers)
        total_energy = avg_power * elapsed

        return EnergyReport(
            total_energy_j=total_energy,
            avg_power_w=avg_power,
            peak_power_w=peak_power,
            duration_s=elapsed,
            gpu_util_avg=sum(utils) / len(utils) if utils else None,
            samples=self._samples,
        )

    def _check_gpu(self) -> bool:
        try:
            subprocess.run(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            return True
        except Exception:
            logger.warning("nvidia-smi not available — energy monitoring disabled")
            return False

    def _read_nvidia_smi(self):
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=power.draw,utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True, text=True, timeout=2,
            )
            line = result.stdout.strip().split("\n")[0]
            parts = [p.strip() for p in line.split(",")]
            power = float(parts[0]) if parts[0] != "[N/A]" else None
            util = float(parts[1]) if parts[1] != "[N/A]" else None
            mem = float(parts[2]) if parts[2] != "[N/A]" else None
            return power, util, mem
        except Exception:
            return None, None, None
