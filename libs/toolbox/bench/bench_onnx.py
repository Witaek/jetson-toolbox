from __future__ import annotations

from pydantic import Field, BaseModel
from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import numpy as np 

import time
import onnx
import onnxruntime as ort 

@dataclass
class BenchmarkConfig(BaseModel):
    warmup_iter: int = Field(1, gt=0)
    bench_iter: int = Field(1, gt=0)
    provider: Literal["cuda", "cpu"] = "cpu"
    batch_size: int = Field(1, gt=0)
    input_shape: Optional[Sequence[int]] = None

@dataclass
class BenchmarkResults(BaseModel):
    onnx_path: str | Path
    warmup_iter: int = Field(1, gt=0)
    bench_iter: int = Field(1, gt=0)
    provider: Literal["cuda", "cpu"] = "cpu"

    avg_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    throughput_fps: float

def benchmark_onnx_speed(onnx_path: str | Path, 
                        config: BenchmarkConfig) -> BenchmarkResults:
    
    session = ort.Infer
