"""Hardware accelerator profiles and Roofline compute/bandwidth ridge models."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final


class UnsupportedPrecisionError(ValueError):
    """Raised when a hardware target lacks native Tensor Core support for a precision format."""


class PrecisionFormat(str, Enum):
    """Weight, activation, and KV-cache numerical storage formats."""

    BF16 = "bf16"
    FP16 = "fp16"
    FP8 = "fp8"
    INT8 = "int8"
    INT4 = "int4"
    NVFP4 = "nvfp4"

    @property
    def raw_bits_per_element(self) -> int:
        """Unscaled bit width per scalar element."""
        if self in (PrecisionFormat.BF16, PrecisionFormat.FP16):
            return 16
        if self in (PrecisionFormat.FP8, PrecisionFormat.INT8):
            return 8
        return 4

    @property
    def raw_bytes_per_element(self) -> float:
        """Unscaled byte width per scalar element."""
        return self.raw_bits_per_element / 8.0


@dataclass(frozen=True, slots=True)
class AcceleratorSpec:
    """Immutable physical compute, memory-hierarchy, and interconnect profile of a GPU."""

    name: str
    architecture: str
    hbm_generation: str
    sm_count: int
    datasheet_hbm_gb: int
    usable_hbm_bytes: int
    peak_hbm_bandwidth_gbps: float
    l2_cache_bytes: int
    smem_capacity_per_sm_bytes: int
    max_shared_mem_per_sm_bytes: int
    register_file_per_sm_bytes: int
    tmem_per_sm_bytes: int
    peak_bf16_tflops: float
    peak_fp8_tflops: float
    peak_nvfp4_tflops: float | None
    nvlink_unidir_gbps: float
    nic_unidir_gbps: float

    def __post_init__(self) -> None:
        if self.sm_count <= 0 or self.usable_hbm_bytes <= 0 or self.peak_hbm_bandwidth_gbps <= 0.0:
            raise ValueError("Accelerator SM count, HBM capacity, and HBM bandwidth must be positive.")

    @property
    def nominal_hbm_gib_bytes(self) -> int:
        """Nominal binary capacity (datasheet_hbm_gb * 2^30 bytes) before ECC/page-retire reserve."""
        return self.datasheet_hbm_gb * (1024**3)

    @property
    def total_register_file_bytes(self) -> int:
        """Aggregate on-chip SM register file storage across all Streaming Multiprocessors."""
        return self.sm_count * self.register_file_per_sm_bytes

    @property
    def total_smem_bytes(self) -> int:
        """Aggregate on-chip SM L1/shared memory pool across all Streaming Multiprocessors."""
        return self.sm_count * self.smem_capacity_per_sm_bytes

    @property
    def total_tmem_bytes(self) -> int:
        """Aggregate Blackwell 2D Tensor Memory (TMEM) across all SMs (0 on Hopper)."""
        return self.sm_count * self.tmem_per_sm_bytes

    @property
    def nvlink_bidir_gbps(self) -> float:
        """Aggregate bidirectional NVLink bandwidth marketed on hardware datasheets."""
        return self.nvlink_unidir_gbps * 2.0

    def peak_compute_tflops(self, fmt: PrecisionFormat, *, sparse_2_4: bool = False) -> float:
        """Peak Tensor Core throughput (TFLOP/s or TOP/s) for the selected execution format.

        Note: W4A16 ``INT4`` weights dequantize in registers/SMEM and execute MACs on
        BF16/FP16 Tensor Cores, so their compute ceiling matches ``peak_bf16_tflops``.
        ``NVFP4`` executes natively only on Blackwell Tensor Cores.
        """
        multiplier = 2.0 if sparse_2_4 else 1.0
        if fmt in (PrecisionFormat.BF16, PrecisionFormat.FP16, PrecisionFormat.INT4):
            return self.peak_bf16_tflops * multiplier
        if fmt in (PrecisionFormat.FP8, PrecisionFormat.INT8):
            return self.peak_fp8_tflops * multiplier
        if fmt == PrecisionFormat.NVFP4:
            if self.peak_nvfp4_tflops is None:
                raise UnsupportedPrecisionError(
                    f"Accelerator '{self.name}' ({self.architecture}) lacks native NVFP4 Tensor Cores."
                )
            return self.peak_nvfp4_tflops * multiplier
        raise UnsupportedPrecisionError(f"Unsupported format: {fmt}")

    def ridge_point(
        self,
        fmt: PrecisionFormat,
        *,
        mfu: float = 1.0,
        mbu: float = 1.0,
        sparse_2_4: bool = False,
    ) -> float:
        """Critical arithmetic intensity I* = (MFU * P_peak) / (MBU * B_peak) in FLOPs/byte."""
        if not (0.0 < mfu <= 1.0) or not (0.0 < mbu <= 1.0):
            raise ValueError("MFU and MBU utilization factors must lie in (0.0, 1.0].")
        p_peak_flops = self.peak_compute_tflops(fmt, sparse_2_4=sparse_2_4) * 1e12 * mfu
        b_peak_bps = self.peak_hbm_bandwidth_gbps * 1e9 * mbu
        return p_peak_flops / b_peak_bps


H100_SXM5: Final[AcceleratorSpec] = AcceleratorSpec(
    name="NVIDIA H100 SXM5",
    architecture="Hopper (GH100)",
    hbm_generation="HBM3",
    sm_count=132,
    datasheet_hbm_gb=80,
    usable_hbm_bytes=81_559 * (1024**2),
    peak_hbm_bandwidth_gbps=3350.0,
    l2_cache_bytes=50 * (1024**2),
    smem_capacity_per_sm_bytes=256 * 1024,
    max_shared_mem_per_sm_bytes=228 * 1024,
    register_file_per_sm_bytes=256 * 1024,
    tmem_per_sm_bytes=0,
    peak_bf16_tflops=989.4,
    peak_fp8_tflops=1978.8,
    peak_nvfp4_tflops=None,
    nvlink_unidir_gbps=450.0,
    nic_unidir_gbps=50.0,
)

H200_SXM: Final[AcceleratorSpec] = AcceleratorSpec(
    name="NVIDIA H200 SXM",
    architecture="Hopper (GH100 Refresh)",
    hbm_generation="HBM3e",
    sm_count=132,
    datasheet_hbm_gb=141,
    usable_hbm_bytes=143_771 * (1024**2),
    peak_hbm_bandwidth_gbps=4800.0,
    l2_cache_bytes=50 * (1024**2),
    smem_capacity_per_sm_bytes=256 * 1024,
    max_shared_mem_per_sm_bytes=228 * 1024,
    register_file_per_sm_bytes=256 * 1024,
    tmem_per_sm_bytes=0,
    peak_bf16_tflops=989.4,
    peak_fp8_tflops=1978.8,
    peak_nvfp4_tflops=None,
    nvlink_unidir_gbps=450.0,
    nic_unidir_gbps=50.0,
)

B200_SXM: Final[AcceleratorSpec] = AcceleratorSpec(
    name="NVIDIA B200 SXM",
    architecture="Blackwell (GB100 Dual-Die)",
    hbm_generation="HBM3e",
    sm_count=148,
    datasheet_hbm_gb=180,
    usable_hbm_bytes=183_359 * (1024**2),
    peak_hbm_bandwidth_gbps=8000.0,
    l2_cache_bytes=126 * (1024**2),
    smem_capacity_per_sm_bytes=256 * 1024,
    max_shared_mem_per_sm_bytes=228 * 1024,
    register_file_per_sm_bytes=256 * 1024,
    tmem_per_sm_bytes=256 * 1024,
    peak_bf16_tflops=2250.0,
    peak_fp8_tflops=4500.0,
    peak_nvfp4_tflops=9000.0,
    nvlink_unidir_gbps=900.0,
    nic_unidir_gbps=100.0,
)

ACCELERATORS: Final[dict[str, AcceleratorSpec]] = {
    "H100_SXM5": H100_SXM5,
    "H200_SXM": H200_SXM,
    "B200_SXM": B200_SXM,
}
