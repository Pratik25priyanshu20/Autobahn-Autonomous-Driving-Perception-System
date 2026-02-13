"""CARLA simulator input (Phase 6.1).

Connects to CARLA, spawns ego vehicle with autopilot, attaches RGB camera,
and yields frames as FramePackets.
"""
from __future__ import annotations

import contextlib
import queue
import time
from collections.abc import Generator

import numpy as np

from src.inputs.base_input import BaseInput
from src.types.perception import FramePacket
from src.utils.logger import get_logger

try:
    import carla
except ImportError:  # pragma: no cover
    carla = None


class CarlaInput(BaseInput):
    """Input source from CARLA simulator."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 2000,
        town: str = "Town01",
        width: int = 1280,
        height: int = 720,
        fps: float = 20.0,
    ):
        self.host = host
        self.port = port
        self.town = town
        self.width = width
        self.height = height
        self.fps = fps
        self.logger = get_logger(__name__)
        self._client = None
        self._world = None
        self._vehicle = None
        self._camera = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=10)
        self._running = False

    def start(self) -> None:
        if carla is None:
            self.logger.warning("carla package not installed — running in stub mode")
            return

        self._client = carla.Client(self.host, self.port)
        self._client.set_timeout(10.0)
        self._world = self._client.load_world(self.town)
        self.logger.info("Connected to CARLA: %s:%d town=%s", self.host, self.port, self.town)

        # Spawn ego vehicle
        bp_lib = self._world.get_blueprint_library()
        vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]
        spawn_points = self._world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points available in CARLA world")
        self._vehicle = self._world.spawn_actor(vehicle_bp, spawn_points[0])
        self._vehicle.set_autopilot(True)
        self.logger.info("Spawned ego vehicle with autopilot")

        # Attach RGB camera
        camera_bp = bp_lib.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(self.width))
        camera_bp.set_attribute("image_size_y", str(self.height))
        camera_bp.set_attribute("fov", "90")
        camera_bp.set_attribute("sensor_tick", str(1.0 / self.fps))
        transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self._camera = self._world.spawn_actor(camera_bp, transform, attach_to=self._vehicle)
        self._camera.listen(self._camera_callback)
        self._running = True
        self.logger.info("Camera attached: %dx%d @ %.1f fps", self.width, self.height, self.fps)

    def _camera_callback(self, image) -> None:
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        bgr = array[:, :, :3]  # Drop alpha
        with contextlib.suppress(queue.Full):
            self._frame_queue.put_nowait(bgr.copy())

    def frames(self) -> Generator[tuple[int, FramePacket], None, None]:
        if carla is None:
            return

        idx = 0
        while self._running:
            try:
                frame = self._frame_queue.get(timeout=2.0)
            except queue.Empty:
                continue
            idx += 1
            yield idx, FramePacket(frame=frame, timestamp=time.time(), sensor_id="carla_rgb")

    def stop(self) -> None:
        self._running = False
        if self._camera is not None:
            self._camera.stop()
            self._camera.destroy()
        if self._vehicle is not None:
            self._vehicle.destroy()
        self.logger.info("CARLA actors destroyed")
