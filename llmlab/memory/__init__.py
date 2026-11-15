"""GPU memory hierarchy, KV-cache accounting, Roofline modeling, and MoE interconnect bounds."""
from __future__ import annotations

from llmlab.memory.hardware import (
    ACCELERATORS,
    B200_SXM,
    H100_SXM5,
    H200_SXM,
    AcceleratorSpec,
    PrecisionFormat,
    UnsupportedPrecisionError,
)
from llmlab.memory.kv_cache import (
    DEEPSEEK_V3_MLA_MOE,
    LLAMA_70B_GQA,
    LLAMA_70B_HYPOTHETICAL_MHA,
    LAYOUT_BF16,
    LAYOUT_FP8,
    LAYOUT_FP16,
    LAYOUT_INT4_G128_ASYM,
    LAYOUT_INT4_G128_SYM,
    LAYOUT_INT8,
    LAYOUT_NVFP4,
    AttentionVariant,
    QuantizationLayout,
    TransformerArchitecture,
)
from llmlab.memory.roofline import (
    MoEAllToAllResult,
    RooflineAnalyzer,
    StepPhase,
    StepRooflineResult,
)

__all__ = [
    "ACCELERATORS",
    "B200_SXM",
    "DEEPSEEK_V3_MLA_MOE",
    "H100_SXM5",
    "H200_SXM",
    "LAYOUT_BF16",
    "LAYOUT_FP16",
    "LAYOUT_FP8",
    "LAYOUT_INT4_G128_ASYM",
    "LAYOUT_INT4_G128_SYM",
    "LAYOUT_INT8",
    "LAYOUT_NVFP4",
    "LLAMA_70B_GQA",
    "LLAMA_70B_HYPOTHETICAL_MHA",
    "AcceleratorSpec",
    "AttentionVariant",
    "MoEAllToAllResult",
    "PrecisionFormat",
    "QuantizationLayout",
    "RooflineAnalyzer",
    "StepPhase",
    "StepRooflineResult",
    "TransformerArchitecture",
    "UnsupportedPrecisionError",
]
