from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def draw_overlays(frame: Any, world_model: Any) -> Any:
    if cv2 is None:
        return frame
    annotated = frame.copy()
    cv2.putText(annotated, "Perception OK", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    return annotated


def draw_hud(frame: Any, fps: float, stages_ms: Dict[str, float], warnings: Optional[List[str]] = None):
    """Minimal HUD overlay with FPS and stage timings."""
    if cv2 is None:
        return frame

    render = frame.copy()
    y = 25
    cv2.putText(render, f"APS++ | FPS: {fps:5.1f}", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y += 28

    for name, ms in list(stages_ms.items())[:6]:
        cv2.putText(render, f"{name}: {ms:5.1f} ms", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)
        y += 22

    if warnings:
        y += 8
        for w in warnings[:3]:
            cv2.putText(render, w, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            y += 24

    return render


def draw_detections(frame: Any, detections: List[Any]) -> Any:
    if cv2 is None:
        return frame
    render = frame.copy()
    for det in detections:
        color = (0, 255, 0)
        cv2.rectangle(render, (det.x1, det.y1), (det.x2, det.y2), color, 2)
        label = f"{det.class_name} {det.conf:.2f}"
        cv2.putText(render, label, (det.x1, det.y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return render


def draw_tracks(frame: Any, tracks: List[Any], trajectories: Optional[Dict[int, List[tuple]]] = None) -> Any:
    if cv2 is None:
        return frame
    render = frame.copy()
    trajectories = trajectories or {}

    for tr in tracks:
        x1, y1, x2, y2 = tr.bbox_xyxy
        color = (255, 200, 0)
        cv2.rectangle(render, (x1, y1), (x2, y2), color, 2)
        label = f"ID {tr.track_id} | {tr.class_name}"
        cv2.putText(render, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        if tr.velocity_px_per_frame is not None:
            vx, vy = tr.velocity_px_per_frame
            cv2.putText(render, f"v=({vx:.1f},{vy:.1f})", (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    for tid, pts in trajectories.items():
        if len(pts) < 2:
            continue
        for i in range(1, len(pts)):
            cv2.line(render, pts[i - 1], pts[i], (255, 200, 0), 2)

    return render


def draw_fcw_pre(frame: Any, fcw_pre: Dict[str, Any]) -> Any:
    if not fcw_pre or fcw_pre.get("state") != "PRE" or cv2 is None:
        return frame
    ttc = fcw_pre.get("ttc_s", None)
    dist_px = fcw_pre.get("distance_px", None)
    if ttc is not None or dist_px is not None:
        dist_txt = f"d={dist_px:.0f}px" if dist_px is not None else ""
        ttc_txt = f"TTC={ttc:.2f}s" if ttc is not None else ""
        sep = " | " if dist_txt and ttc_txt else ""
        txt = f"FCW-PRE | {dist_txt}{sep}{ttc_txt}"
        cv2.putText(frame, txt, (15, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return frame


def draw_fcw(frame: Any, fcw: Dict[str, Any]) -> Any:
    if not fcw or cv2 is None:
        return frame

    state = fcw.get("state", "NORMAL")
    ttc = fcw.get("ttc_s", None)
    bbox = fcw.get("lead_bbox", None)

    if bbox is not None:
        x1, y1, x2, y2 = map(int, bbox)
        if state in ("WARNING", "CRITICAL"):
            color = (0, 0, 255)
        elif state == "CAUTION":
            color = (0, 165, 255)
        else:
            color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    if ttc is not None:
        txt = f"FCW: {state} | TTC={ttc:.2f}s | d={fcw.get('distance_m',0):.1f}m | v={fcw.get('rel_speed_mps',0):.1f}m/s"
        cv2.putText(frame, txt, (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return frame


def draw_drivable(frame: Any, drivable: Dict[str, Any]) -> Any:
    if cv2 is None or drivable is None:
        return frame
    mask = None
    # Support both dict-based and dataclass DrivableArea
    if hasattr(drivable, "mask"):
        mask = drivable.mask
    elif isinstance(drivable, dict):
        mask = drivable.get("mask")
    if mask is None:
        return frame
    overlay = frame.copy()
    overlay[mask == 1] = overlay[mask == 1] * 0.6 + np.array([0, 255, 0]) * 0.4
    return overlay


def draw_safety_banner(frame: Any, safety: Dict[str, Any]) -> Any:
    if not safety or cv2 is None:
        return frame
    state = safety.get("state", "NORMAL")
    msg = safety.get("message", "")
    color = safety.get("color", (0, 255, 0))
    x, y = 10, 10
    w = 620
    h = 38
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    alpha = 0.25
    frame[:] = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    cv2.putText(frame, f"SAFETY: {state} | {msg}", (x + 10, y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame


def draw_lanes(frame: Any, lanes: Dict[str, Any]) -> Any:
    if cv2 is None:
        return frame
    render = frame.copy()
    departure = lanes.get("lane_departure")
    lane_color = (0, 0, 255) if departure is not None else (0, 255, 0)
    left = lanes.get("left_line")
    right = lanes.get("right_line")
    if left:
        cv2.line(render, left[0], left[1], lane_color, 4)
    if right:
        cv2.line(render, right[0], right[1], lane_color, 4)
    offset = lanes.get("ego_offset_px")
    if offset is not None:
        cv2.putText(render, f"ego offset: {offset:.1f}px", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, lane_color, 2)
    if departure is not None:
        cv2.putText(render, f"LANE DEPARTURE {departure}", (15, render.shape[0] - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
    conf = lanes.get("lane_confidence")
    stable = lanes.get("lane_stable")
    allowed = lanes.get("ldw_allowed")
    jitter = lanes.get("lane_center_jitter_px")
    if conf is not None:
        txt = f"lane_conf={conf:.2f} stable={stable} ldw_allowed={allowed}"
        cv2.putText(render, txt, (15, render.shape[0] - 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if jitter is not None:
        cv2.putText(render, f"lane_jitter={jitter:.1f}px", (15, render.shape[0] - 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return render


def draw_world_text(frame: Any, world: Any) -> Any:
    if cv2 is None or world is None:
        return frame
    cv2.putText(
        frame,
        f"World: objs={len(getattr(world, 'tracks', []))} lane={'yes' if world.lanes else 'no'}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 0),
        1,
    )
    return frame
