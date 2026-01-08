from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from rich.console import Console
from tqdm import tqdm

from src.inputs.video_input import VideoInput
from src.runtime.orchestrator import Orchestrator
from src.utils.config import get, load_yaml
from src.utils.logger import setup_logger
from src.visualization.overlay import (
    draw_detections,
    draw_fcw,
    draw_fcw_pre,
    draw_hud,
    draw_lanes,
    draw_drivable,
    draw_safety_banner,
    draw_world_text,
    draw_tracks,
)
from src.bev import BEVRenderer


def make_run_dir(base_dir: str | Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description="AutobahnPerceptionStack (APS++) - Phase 0")
    parser.add_argument("--config", default="configs/system.yaml", help="Path to YAML config")
    parser.add_argument("--input", required=True, help="Path to input video")
    args = parser.parse_args()

    cfg: Dict[str, Any] = load_yaml(args.config)

    output_base = get(cfg, "runtime.output_dir", "results")
    run_dir = make_run_dir(output_base)
    logger = setup_logger(log_dir=run_dir, level=cfg.get("runtime", {}).get("log_level", "INFO"))
    safety_logger = None

    console = Console()
    console.print(f"[bold]APS++[/bold] run dir: {run_dir}")

    vin = VideoInput(args.input)
    logger.info("Input video: %s", args.input)
    if vin.meta:
        logger.info("Meta: fps=%.2f size=%dx%d frames=%d", vin.meta.fps, vin.meta.width, vin.meta.height, vin.meta.frame_count)
    from src.safety.safety_logger import SafetyLogger
    safety_logger = SafetyLogger(run_dir)

    out_video_path = run_dir / "output.mp4"
    save_video = bool(get(cfg, "runtime.save_video", True))
    save_metrics = bool(get(cfg, "runtime.save_metrics", True))
    overlay_enabled = bool(get(cfg, "runtime.overlay.enabled", True))

    writer = None
    if save_video:
        if cv2 is None:
            raise ImportError("opencv-python is required to save video output")
        width = vin.meta.width if vin.meta else 0
        height = vin.meta.height if vin.meta else 0
        fps = vin.meta.fps if vin.meta else 30
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError("Could not open VideoWriter (mp4v). Try a different codec/container.")

    orchestrator = Orchestrator(cfg, logger)
    bev_renderer = BEVRenderer()

    metrics = {
        "project": cfg.get("project", {}),
        "input": {"path": args.input, "meta": vin.meta.__dict__ if vin.meta else {}},
        "frames": [],
        "bev": {"max_objects": 0, "avg_objects": 0.0, "max_range_m": bev_renderer.size / bev_renderer.ppm if cv2 else 0},
    }
    bev_writer = None
    if save_video and cv2 is not None:
        bev_path = run_dir / "bev.mp4"
        bev_fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        bev_writer = cv2.VideoWriter(str(bev_path), bev_fourcc, vin.meta.fps if vin.meta else 30, (bev_renderer.size, bev_renderer.size))

    total = vin.meta.frame_count if vin.meta and vin.meta.frame_count > 0 else None
    bev_count = 0
    for frame_id, packet in tqdm(vin.frames(), total=total, desc="Processing"):
        wm = orchestrator.process_frame(frame_id, packet.frame)
        render = wm.frame

        if overlay_enabled:
            if wm.tracks:
                render = draw_tracks(render, wm.tracks, wm.trajectories)
            else:
                render = draw_detections(render, wm.detections)
            if wm.lanes:
                render = draw_lanes(render, wm.lanes)
            if wm.drivable_area:
                render = draw_drivable(render, wm.drivable_area)
            if wm.fcw_pre:
                render = draw_fcw_pre(render, wm.fcw_pre)
            if wm.fcw:
                render = draw_fcw(render, wm.fcw)
            if wm.safety:
                render = draw_safety_banner(render, wm.safety)
            render = draw_world_text(render, wm)
            render = draw_hud(render, wm.runtime.fps, wm.runtime.stages_ms, wm.warnings)

        if writer is not None and cv2 is not None:
            writer.write(render)

        bev_img = bev_renderer.render(wm, wm.fcw if wm.fcw else {})
        if bev_writer is not None and cv2 is not None:
            bev_writer.write(bev_img)
        # Save one BEV frame every 60 frames for docs
        if frame_id % 60 == 0:
            bev_snap = run_dir / f"bev_{frame_id:05d}.png"
            if cv2 is not None:
                cv2.imwrite(str(bev_snap), bev_img)

        if save_metrics:
            # update bev metrics
            obj_count = len(wm.tracks)
            bev_count += 1
            metrics["bev"]["max_objects"] = max(metrics["bev"]["max_objects"], obj_count)
            metrics["bev"]["avg_objects"] = ((metrics["bev"]["avg_objects"] * (bev_count - 1)) + obj_count) / bev_count
            metrics["frames"].append(
                {
                    "frame_id": frame_id,
                    "fps": wm.runtime.fps,
                    "stages_ms": wm.runtime.stages_ms,
                    "warnings": wm.warnings,
                    "detection_count": len(wm.detections),
                    "track_count": len(wm.tracks),
                    "lane": {"ego_offset_px": wm.lanes.get("ego_offset_px") if wm.lanes else None},
                    "drivable_confidence": wm.drivable_area.confidence if wm.drivable_area else None,
                    "fcw": wm.fcw,
                    "fcw_pre": wm.fcw_pre,
                    "safety": wm.safety,
                }
            )
        if safety_logger and wm.safety:
            safety_logger.log(
                frame_idx=frame_id,
                timestamp_s=frame_id / (vin.meta.fps or 30),
                safety_state=wm.safety.get("state", ""),
                message=wm.safety.get("message", ""),
                details=wm.safety.get("details", {}),
            )
        if safety_logger and wm.safety and wm.safety.get("fcw_event"):
            ev = wm.safety["fcw_event"]
            safety_logger.log(
                frame_idx=frame_id,
                timestamp_s=frame_id / (vin.meta.fps or 30),
                safety_state=f"FCW_{ev.get('state', '')}",
                message="FCW state change",
                details=ev,
            )

    vin.stop()
    if writer is not None:
        writer.release()
    if bev_writer is not None:
        bev_writer.release()

    if save_metrics:
        metrics_path = run_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        logger.info("Saved metrics: %s", metrics_path)

    if save_video:
        logger.info("Saved video: %s", out_video_path)
        if bev_writer is not None:
            logger.info("Saved BEV video: %s", run_dir / "bev.mp4")

    logger.info("Done.")


if __name__ == "__main__":
    main()
