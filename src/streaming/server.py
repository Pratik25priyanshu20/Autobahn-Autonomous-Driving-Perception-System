"""FastAPI + WebSocket streaming server (Phase 6.3).

Streams JPEG-encoded frames and world model JSON via WebSocket.
"""
from __future__ import annotations

import json
import threading
from typing import Any, Optional

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    import uvicorn
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse

    _HAS_FASTAPI = True
except ImportError:  # pragma: no cover
    _HAS_FASTAPI = False


class StreamingServer:
    """WebSocket server that streams frames + world model data."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        if not _HAS_FASTAPI:
            raise ImportError("fastapi and uvicorn are required for StreamingServer")
        self.host = host
        self.port = port
        self._app = FastAPI(title="APS++ Stream")
        self._latest_frame: Optional[bytes] = None
        self._latest_wm: Optional[dict] = None
        self._thread: Optional[threading.Thread] = None
        self._setup_routes()

    def _setup_routes(self):
        @self._app.get("/")
        async def index():
            return HTMLResponse(
                "<html><body><h1>APS++ Live Stream</h1>"
                "<p>Connect via WebSocket at /ws</p></body></html>"
            )

        @self._app.websocket("/ws")
        async def ws_endpoint(websocket: WebSocket):
            await websocket.accept()
            try:
                while True:
                    import asyncio
                    await asyncio.sleep(0.033)  # ~30fps
                    if self._latest_frame is not None:
                        import base64
                        payload = {
                            "frame": base64.b64encode(self._latest_frame).decode("ascii"),
                            "world_model": self._latest_wm or {},
                        }
                        await websocket.send_text(json.dumps(payload))
            except WebSocketDisconnect:
                pass

    def push_frame(self, frame: Any, world_model: Any = None) -> None:
        if cv2 is not None and frame is not None:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            self._latest_frame = buf.tobytes()
        if world_model is not None:
            self._latest_wm = {
                "frame_id": getattr(world_model, "frame_id", 0),
                "track_count": len(getattr(world_model, "tracks", [])),
                "detection_count": len(getattr(world_model, "detections", [])),
                "safety": getattr(world_model, "safety", {}),
                "fcw": getattr(world_model, "fcw", {}),
                "warnings": getattr(world_model, "warnings", []),
            }

    def start(self) -> None:
        self._thread = threading.Thread(
            target=uvicorn.run,
            args=(self._app,),
            kwargs={"host": self.host, "port": self.port, "log_level": "warning"},
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        pass  # Daemon thread will exit with main process
