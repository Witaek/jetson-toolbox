from __future__ import annotations

import time
from pathlib import Path
from typing import List, Literal, Optional, Sequence

import numpy as np
import onnxruntime as ort
from pydantic import BaseModel, Field

PROVIDER_MAPPING = {
    "cuda":"CUDAExecutionProvider",
    "cpu" : "CPUExecutionProvider"
}

def map_provider(provider_key: str) ->  List[str] :
    if provider_key not in PROVIDER_MAPPING.keys():
        raise ValueError
    return [PROVIDER_MAPPING[provider_key]]

class BenchmarkConfig(BaseModel):
    warmup_iters: int = Field(1, gt=0)
    bench_iters: int = Field(1, gt=0)
    provider: Literal["cuda", "cpu"] = "cpu"
    batch_size: int = Field(1, gt=0)
    input_shape: Optional[Sequence[int]] = None

class BenchmarkResults(BaseModel):
    onnx_path: str | Path
    warmup_iters: int = Field(1, gt=0)
    bench_iters: int = Field(1, gt=0)
    provider: str = "cpu"
    batch_size: int = Field(1, gt=0)

    avg_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    throughput_fps: float

def benchmark_onnx_speed(onnx_path: str | Path, 
                        cfg: BenchmarkConfig) -> BenchmarkResults:
    
    provider = map_provider(cfg.provider)
    session = ort.InferenceSession(onnx_path, providers = provider)

    input_name = session.get_inputs()[0].name
    input_shape = cfg.input_shape or session.get_inputs()[0].shape

    shape = []
    for dim in input_shape:
        if isinstance(dim, str) or dim is None or dim == -1:
            shape.append(cfg.batch_size)
        else:
            shape.append(int(dim))

    dummy_input = np.random.randn(*shape).astype(np.float32)

    #Warmup
    for _ in range(cfg.warmup_iters):
        session.run(None, {input_name: dummy_input})

    #Timed inference
    latencies_ms: List[float] = []
    for _ in range(cfg.bench_iters):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        t1 = time.perf_counter()

        delta_t = t1 - t0
        latencies_ms.append(delta_t * 1000)

    latencies_ms = np.array(latencies_ms)
    avg_ms = float(latencies_ms.mean())
    p50_ms = float(np.percentile(latencies_ms, 50))
    p90_ms = float(np.percentile(latencies_ms, 90))
    p99_ms = float(np.percentile(latencies_ms, 99))

    throughput_fps = cfg.batch_size / (avg_ms / 1000.0)

    return BenchmarkResults(
        onnx_path = onnx_path,
        warmup_iters = cfg.warmup_iters,
        bench_iters = cfg.bench_iters,
        provider = provider[0],
        avg_ms=avg_ms,
        p50_ms=p50_ms,
        p90_ms=p90_ms,
        p99_ms=p99_ms,
        throughput_fps=throughput_fps,
        batch_size=cfg.batch_size
    )

def _format_result(r: BenchmarkResults) -> str:
  return (
    f"Model: {r.onnx_path}\n"
    f"Provider: {r.provider}\n"
    f"Batch size: {r.batch_size}\n"
    f"Warmup iters: {r.warmup_iters}, Bench iters: {r.bench_iters}\n"
    f"Latency (ms): avg={r.avg_ms:.3f}, p50={r.p50_ms:.3f}, "
    f"p90={r.p90_ms:.3f}, p99={r.p99_ms:.3f}\n"
    f"Throughput: {r.throughput_fps:.2f} fps\n"
  )


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(description="ONNX benchmark (MVP).")
  parser.add_argument("model", type=str, help="Path to ONNX model")
  parser.add_argument("--provider", type=str, default="cpu", choices=["cpu", "cuda"])
  parser.add_argument("--batch-size", type=int, default=1)
  parser.add_argument("--warmup-iters", type=int, default=10)
  parser.add_argument("--bench-iters", type=int, default=50)

  args = parser.parse_args()

  cfg = BenchmarkConfig(
    warmup_iters=args.warmup_iters,
    bench_iters=args.bench_iters,
    batch_size=args.batch_size,
    provider=args.provider,
  )
  result = benchmark_onnx_speed(args.model, cfg)
  print(_format_result(result))
