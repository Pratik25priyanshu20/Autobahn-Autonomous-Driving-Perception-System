"""Blind Spot Detection & Rear Cross-Traffic Alert (Phase 3.2).

Checks lateral tracks for objects in blind-spot zones and approaching
from the side.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class BSDWarning:
    side: str  # "left" | "right"
    track_id: int
    distance_m: float
    ttc_s: float | None = None


class BlindSpotDetector:
    """Detects objects in blind-spot zones."""

    def __init__(
        self,
        blind_spot_x_min: float = 1.5,
        blind_spot_x_max: float = 4.0,
        blind_spot_y_min: float = -3.0,
        blind_spot_y_max: float = 3.0,
        ttc_warn_s: float = 3.0,
    ):
        self.x_min = blind_spot_x_min
        self.x_max = blind_spot_x_max
        self.y_min = blind_spot_y_min
        self.y_max = blind_spot_y_max
        self.ttc_warn_s = ttc_warn_s

    def evaluate(self, tracks: list[Any]) -> list[BSDWarning]:
        """Evaluate all tracks and return BSD warnings."""
        warnings: list[BSDWarning] = []

        for trk in tracks:
            x_m = getattr(trk, "x", None)
            y_m = getattr(trk, "y", None)
            vx = getattr(trk, "vx", None)
            tid = getattr(trk, "track_id", None)
            if x_m is None or y_m is None or tid is None:
                continue

            # Check left blind spot
            if -self.x_max <= x_m <= -self.x_min and self.y_min <= y_m <= self.y_max:
                dist = abs(x_m)
                ttc = self._lateral_ttc(x_m, vx)
                if ttc is None or ttc <= self.ttc_warn_s:
                    warnings.append(BSDWarning(side="left", track_id=tid, distance_m=dist, ttc_s=ttc))

            # Check right blind spot
            elif self.x_min <= x_m <= self.x_max and self.y_min <= y_m <= self.y_max:
                dist = abs(x_m)
                ttc = self._lateral_ttc(x_m, vx)
                if ttc is None or ttc <= self.ttc_warn_s:
                    warnings.append(BSDWarning(side="right", track_id=tid, distance_m=dist, ttc_s=ttc))

        return warnings

    def _lateral_ttc(self, x_m: float, vx: float | None) -> float | None:
        if vx is None or abs(vx) < 0.1:
            return None
        # TTC = distance / closing rate toward ego center
        if x_m > 0 and vx < 0 or x_m < 0 and vx > 0:  # approaching from right
            return abs(x_m / vx)
        return None
