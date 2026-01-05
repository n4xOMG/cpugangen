"""Quantization utilities for HQPD models."""

from .dynamic_int8 import (
    quantize_linear_layers,
    save_quantized_model,
    load_quantized_model,
)

from .cpu_optimize import (
    configure_cpu_inference,
    get_optimal_thread_count,
)

__all__ = [
    "quantize_linear_layers",
    "save_quantized_model",
    "load_quantized_model",
    "configure_cpu_inference",
    "get_optimal_thread_count",
]


