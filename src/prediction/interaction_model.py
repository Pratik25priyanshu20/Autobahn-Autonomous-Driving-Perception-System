"""Rule-based interaction model for behavioral prediction (Task 7).

Implements German traffic rule heuristics: gap acceptance, yield (Rechts-vor-Links),
following distance (2-second rule), and cut-in prediction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class InteractionEvent:
    """Detected traffic interaction event."""

    type: str  # "gap_acceptance", "yield", "following_distance", "cut_in"
    involved_track_ids: list[int]
    risk_level: str  # "low", "medium", "high"
    description: str
    time_to_event_s: float | None = None


class InteractionModel:
    """Evaluates rule-based driving interaction heuristics.

    All methods assume ego vehicle state and a list of tracked objects
    with world-frame positions (x, y in meters) and velocities (vx, vy).
    """

    def __init__(
        self,
        min_gap_s: float = 3.0,
        safe_following_s: float = 2.0,
        lane_width_m: float = 3.5,
        cut_in_lateral_threshold: float = 0.5,
    ):
        self.min_gap_s = min_gap_s
        self.safe_following_s = safe_following_s
        self.lane_width_m = lane_width_m
        self.cut_in_lateral_threshold = cut_in_lateral_threshold

    def gap_acceptance(
        self,
        ego: dict[str, float],
        adjacent_tracks: list[Any],
    ) -> InteractionEvent | None:
        """Check if gap in adjacent lane is large enough to merge.

        Ego dict: {"x": m, "y": m, "vx": m/s, "vy": m/s}
        Adjacent tracks: objects in the target lane.
        """
        if not adjacent_tracks:
            return None

        ego_y = ego.get("y", 0.0)
        ego_vy = ego.get("vy", 0.0)

        # Find closest ahead and behind in target lane
        ahead_gap_s = float("inf")
        behind_gap_s = float("inf")

        for trk in adjacent_tracks:
            ty = getattr(trk, "y", None)
            tvy = getattr(trk, "vy", None)
            if ty is None:
                continue
            dy = ty - ego_y
            rel_vy = (tvy or 0.0) - ego_vy

            if dy > 0:  # ahead
                time_gap = dy / max(abs(rel_vy), 0.1) if abs(rel_vy) > 0.01 else dy / 1.0
                ahead_gap_s = min(ahead_gap_s, abs(time_gap))
            else:  # behind
                time_gap = abs(dy) / max(abs(rel_vy), 0.1) if abs(rel_vy) > 0.01 else abs(dy) / 1.0
                behind_gap_s = min(behind_gap_s, abs(time_gap))

        min_gap = min(ahead_gap_s, behind_gap_s)
        if min_gap < self.min_gap_s:
            risk = "high" if min_gap < self.min_gap_s * 0.5 else "medium"
            involved = [getattr(t, "track_id", 0) for t in adjacent_tracks[:3]]
            return InteractionEvent(
                type="gap_acceptance",
                involved_track_ids=involved,
                risk_level=risk,
                description=f"Insufficient gap for merge: {min_gap:.1f}s < {self.min_gap_s:.1f}s",
                time_to_event_s=min_gap,
            )
        return None

    def yield_heuristic(
        self,
        ego: dict[str, float],
        tracks: list[Any],
    ) -> InteractionEvent | None:
        """German Rechts-vor-Links (right-before-left) yield rule.

        At intersections, vehicles approaching from the right have priority.
        Detects objects approaching from the right side with closing velocity.
        """
        ego_x = ego.get("x", 0.0)
        ego_y = ego.get("y", 0.0)

        for trk in tracks:
            tx = getattr(trk, "x", None)
            ty = getattr(trk, "y", None)
            tvx = getattr(trk, "vx", None)
            if tx is None or ty is None:
                continue

            # Object is to the right and at similar longitudinal position
            dx = tx - ego_x
            dy = abs(ty - ego_y)

            if dx > self.lane_width_m * 0.5 and dy < self.lane_width_m * 2 and tvx is not None and tvx < -0.5:
                    time_to_cross = abs(dx) / max(abs(tvx), 0.1)
                    risk = "high" if time_to_cross < 2.0 else "medium"
                    return InteractionEvent(
                        type="yield",
                        involved_track_ids=[getattr(trk, "track_id", 0)],
                        risk_level=risk,
                        description=f"Rechts-vor-Links: vehicle from right, TTC={time_to_cross:.1f}s",
                        time_to_event_s=time_to_cross,
                    )
        return None

    def following_distance(
        self,
        ego: dict[str, float],
        lead_track: Any,
    ) -> InteractionEvent | None:
        """Check 2-second rule for following distance.

        The safe following distance in seconds = distance / ego_speed.
        """
        if lead_track is None:
            return None

        ego_y = ego.get("y", 0.0)
        ego_vy = ego.get("vy", 0.0)
        lead_y = getattr(lead_track, "y", None)
        lead_x = getattr(lead_track, "x", None)

        if lead_y is None:
            return None

        # Only check vehicles in same lane corridor
        if lead_x is not None and abs(lead_x) > self.lane_width_m / 2:
            return None

        distance_m = lead_y - ego_y
        if distance_m <= 0:
            return None

        ego_speed = abs(ego_vy) if ego_vy else 1.0
        following_time_s = distance_m / max(ego_speed, 0.1)

        if following_time_s < self.safe_following_s:
            risk = "high" if following_time_s < self.safe_following_s * 0.5 else "medium"
            return InteractionEvent(
                type="following_distance",
                involved_track_ids=[getattr(lead_track, "track_id", 0)],
                risk_level=risk,
                description=f"Following too close: {following_time_s:.1f}s < {self.safe_following_s:.1f}s",
                time_to_event_s=following_time_s,
            )
        return None

    def cut_in_prediction(
        self,
        ego: dict[str, float],
        tracks: list[Any],
    ) -> InteractionEvent | None:
        """Detect vehicles with lateral velocity indicating cut-in to ego lane.

        A cut-in is predicted when an adjacent-lane vehicle has significant
        lateral velocity toward the ego lane center.
        """
        half_lane = self.lane_width_m / 2

        for trk in tracks:
            tx = getattr(trk, "x", None)
            tvx = getattr(trk, "vx", None)
            ty = getattr(trk, "y", None)
            if tx is None or tvx is None or ty is None:
                continue

            # Must be in adjacent lane (outside ego corridor)
            if abs(tx) <= half_lane:
                continue

            # Must have lateral velocity toward ego lane
            moving_toward_ego = (tx > 0 and tvx < -self.cut_in_lateral_threshold) or (
                tx < 0 and tvx > self.cut_in_lateral_threshold
            )

            if moving_toward_ego:
                dist_to_lane_edge = abs(tx) - half_lane
                time_to_cut_in = dist_to_lane_edge / max(abs(tvx), 0.01)
                risk = "high" if time_to_cut_in < 1.5 else "medium"
                return InteractionEvent(
                    type="cut_in",
                    involved_track_ids=[getattr(trk, "track_id", 0)],
                    risk_level=risk,
                    description=f"Cut-in predicted: lateral_v={tvx:.2f} m/s, time={time_to_cut_in:.1f}s",
                    time_to_event_s=time_to_cut_in,
                )
        return None

    def evaluate(
        self,
        ego_state: dict[str, float],
        tracks: list[Any],
    ) -> list[InteractionEvent]:
        """Run all interaction heuristics and return detected events."""
        events: list[InteractionEvent] = []

        # Find lead vehicle (in-corridor, closest ahead)
        half_lane = self.lane_width_m / 2
        ego_y = ego_state.get("y", 0.0)
        lead_track = None
        min_dy = float("inf")
        adjacent_tracks = []

        for trk in tracks:
            tx = getattr(trk, "x", None)
            ty = getattr(trk, "y", None)
            if tx is None or ty is None:
                continue
            if abs(tx) <= half_lane and (ty - ego_y) > 0:
                dy = ty - ego_y
                if dy < min_dy:
                    min_dy = dy
                    lead_track = trk
            if abs(tx) > half_lane:
                adjacent_tracks.append(trk)

        # Run heuristics
        ev = self.gap_acceptance(ego_state, adjacent_tracks)
        if ev:
            events.append(ev)

        ev = self.yield_heuristic(ego_state, tracks)
        if ev:
            events.append(ev)

        ev = self.following_distance(ego_state, lead_track)
        if ev:
            events.append(ev)

        ev = self.cut_in_prediction(ego_state, tracks)
        if ev:
            events.append(ev)

        if events:
            logger.debug("Interaction model: %d events detected", len(events))

        return events
