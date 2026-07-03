"""
Obfuscation policy grid for adaptive-adversary experiments.

Eight settings: clean baseline + seven Phase 3 test policies.
Keys are used in artifact paths and the adaptive registry.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

PaddingType = Literal["none", "linear128", "mtu"]
JitterKey = Literal["none", "low", "medium", "high"]


@dataclass(frozen=True)
class AdaptivePolicy:
    key: str
    padding_type: PaddingType
    jitter_key: JitterKey
    description: str

    @property
    def is_clean(self) -> bool:
        return self.key == "clean"

    @property
    def test_stem(self) -> str:
        """Stem of Phase 3 test artifact (obfuscated_*.pt)."""
        if self.is_clean:
            return "baseline"
        return f"obfuscated_{self.key}"


POLICIES: tuple[AdaptivePolicy, ...] = (
    AdaptivePolicy("clean", "none", "none", "Unobfuscated train/val (Phase 1 tensors)"),
    AdaptivePolicy("jitter_low", "none", "low", "Laplace jitter 1 ms"),
    AdaptivePolicy("jitter_medium", "none", "medium", "Laplace jitter 5 ms"),
    AdaptivePolicy("jitter_high", "none", "high", "Laplace jitter 20 ms"),
    AdaptivePolicy("linear128", "linear128", "none", "128-byte size padding"),
    AdaptivePolicy("mtu", "mtu", "none", "MTU (1500 B) padding"),
    AdaptivePolicy(
        "linear128_jitter_medium",
        "linear128",
        "medium",
        "128-byte padding + 5 ms jitter",
    ),
    AdaptivePolicy(
        "mtu_jitter_medium",
        "mtu",
        "medium",
        "MTU padding + 5 ms jitter",
    ),
)

POLICY_BY_KEY = {p.key: p for p in POLICIES}


def get_policy(key: str) -> AdaptivePolicy:
    if key not in POLICY_BY_KEY:
        known = ", ".join(POLICY_BY_KEY)
        raise KeyError(f"Unknown policy '{key}'. Known: {known}")
    return POLICY_BY_KEY[key]
