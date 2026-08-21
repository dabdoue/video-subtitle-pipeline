"""Pure runtime-selection policy for the optional Nemotron ASR server."""

from __future__ import annotations

from dataclasses import asdict, dataclass


MIB = 1024**2
GIB = 1024**3
RUNTIME_MODES = {"auto", "throughput", "streaming"}


@dataclass(frozen=True)
class RuntimeDecision:
    requested: str
    selected: str
    reason: str
    estimated_offline_bytes: int
    allowed_offline_bytes: int

    def metadata(self) -> dict:
        payload = asdict(self)
        payload["estimated_offline_mib"] = round(self.estimated_offline_bytes / MIB, 1)
        payload["allowed_offline_mib"] = round(self.allowed_offline_bytes / MIB, 1)
        del payload["estimated_offline_bytes"]
        del payload["allowed_offline_bytes"]
        return payload


def choose_runtime(
    requested: str,
    *,
    duration_seconds: float,
    free_gpu_bytes: int,
    memory_limit_gb: float | None,
    max_offline_seconds: float | None,
    auto_max_offline_seconds: float,
    reserve_gb: float,
    offline_fixed_mib: float,
    offline_mib_per_second: float,
) -> RuntimeDecision:
    requested = requested.casefold().strip()
    if requested not in RUNTIME_MODES:
        raise ValueError(
            f"Unknown runtime {requested!r}; expected one of {sorted(RUNTIME_MODES)}"
        )
    numeric = {
        "duration_seconds": duration_seconds,
        "free_gpu_bytes": free_gpu_bytes,
        "auto_max_offline_seconds": auto_max_offline_seconds,
        "reserve_gb": reserve_gb,
        "offline_fixed_mib": offline_fixed_mib,
        "offline_mib_per_second": offline_mib_per_second,
    }
    if any(value < 0 for value in numeric.values()):
        raise ValueError("Runtime duration and memory policy values cannot be negative")
    if memory_limit_gb is not None and memory_limit_gb <= 0:
        raise ValueError("memory_limit_gb must be greater than zero when supplied")
    if max_offline_seconds is not None and max_offline_seconds <= 0:
        raise ValueError("max_offline_seconds must be greater than zero when supplied")

    estimated = round(
        (offline_fixed_mib + offline_mib_per_second * duration_seconds) * MIB
    )
    available = max(0, free_gpu_bytes - round(reserve_gb * GIB))
    if memory_limit_gb is not None:
        available = min(available, round(memory_limit_gb * GIB))

    if requested == "streaming":
        return RuntimeDecision(
            requested, "streaming", "explicit_streaming", estimated, available
        )

    duration_limit = max_offline_seconds
    if duration_limit is None and requested == "auto" and auto_max_offline_seconds > 0:
        duration_limit = auto_max_offline_seconds
    if duration_limit is not None and duration_seconds > duration_limit:
        return RuntimeDecision(
            requested, "streaming", "offline_duration_limit", estimated, available
        )
    if estimated > available:
        return RuntimeDecision(
            requested, "streaming", "offline_memory_budget", estimated, available
        )
    return RuntimeDecision(requested, "offline", "offline_fits_budget", estimated, available)
