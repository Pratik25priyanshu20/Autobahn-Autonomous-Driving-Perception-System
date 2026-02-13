# Deployment Architecture

Target deployment topology for APS++ on automotive-grade ECUs with multi-sensor input support.

## Sensor Suite

| Sensor | Interface | Data Rate | Role |
|--------|-----------|-----------|------|
| Front camera | GMSL2 / MIPI CSI-2 | 4-6 Gbps | Primary perception (detection, tracking, lanes, depth) |
| LIDAR | Ethernet (PTP synced) | 100 Mbps | 3D point cloud for ground removal, clustering, fusion |
| Front radar | CAN FD / Ethernet | 5 Mbps | Doppler velocity, range, ghost-filtered detections |

All sensors feed into the orchestrator via `FramePacket` which carries camera frame, `PointCloud`, and `RadarFrame`.

## Target ECUs

### NVIDIA Orin (Primary Compute)

| Parameter | Value |
|-----------|-------|
| SoC | NVIDIA Orin (DRIVE AGX / Jetson AGX Orin) |
| GPU | Ampere, 2048 CUDA cores, 64 Tensor Cores |
| DLA | 2x Deep Learning Accelerators |
| CPU | 12-core Arm Cortex-A78AE |
| Memory | 32 GB LPDDR5 (204 GB/s) |
| TDP | 15-60 W (configurable power mode) |
| OS | Linux (DRIVE OS / JetPack) |

**Role**: Runs all neural inference (detection, depth, segmentation) via TensorRT engines. Hosts the full fusion pipeline (camera + LIDAR + radar), world model, safety evaluator, and sensor health monitor.

### TI TDA4VM (Safety Co-processor)

| Parameter | Value |
|-----------|-------|
| SoC | Texas Instruments TDA4VM (Jacinto 7) |
| DSP | 2x C7x + MMA (8 TOPS INT8) |
| CPU | 2x Arm Cortex-A72 + 6x Cortex-R5F |
| ISP | Integrated vision pre-processing |
| Memory | 4 GB LPDDR4 |
| TDP | 5-20 W |
| Safety | ASIL-D capable MCU island |

**Role**: Runs a lightweight safety-check replica. Receives fused world-model snapshots from Orin over Ethernet. Executes independent FCW/LDW plausibility checks. Can trigger autonomous emergency braking (AEB) if the Orin pipeline fails or disagrees.

## Deployment Topology

```
  Camera(s)       LIDAR           Radar
     |  GMSL2     |  Ethernet     |  CAN FD
     v             v               v
+----------------------------------------------------+
|                  NVIDIA Orin                         |
|                                                      |
| - YOLOv8 (TRT)    - LIDAR processor (voxel+RANSAC)  |
| - MiDAS (TRT)     - LIDAR-camera fusion              |
| - DeepLabV3 (TRT) - Radar processor (ghost filter)   |
| - Tracking         - Radar-camera fusion              |
| - Sensor Health    - Saliency / Grad-CAM              |
| - Kalman + Pred    - Interaction model                |
| - Safety Manager   - Config validator                 |
| - World Model      - Data recorder                    |
| - Control output                                      |
+----------------------------+-------------------------+
                             |
                  Ethernet (1 Gbps)
                  WorldModel JSON @ 20 Hz
                             |
                             v
                  +-----------------+
                  |   TI TDA4VM     |
                  |                 |
                  | - Safety replica|
                  | - FCW/LDW check |
                  | - Plausibility  |
                  | - AEB trigger   |
                  | - Watchdog      |
                  +-----------------+
                             |
                  CAN FD / Ethernet
                             |
                             v
                  Vehicle actuators
                  (steering, braking, throttle)
```

## Pipeline Stage Mapping

| Stage | ECU | Accelerator | Format | Latency Budget |
|-------|-----|-------------|--------|----------------|
| Image capture + ISP | Orin | VI/ISP | Raw -> YUV | 2 ms |
| YOLOv8n detection | Orin | GPU (TRT FP16) | TensorRT engine | 8 ms |
| MiDAS depth | Orin | GPU (TRT FP16) | TensorRT engine | 10 ms |
| DeepLabV3 segmentation | Orin | DLA (INT8) | TensorRT engine | 6 ms |
| Sensor health assessment | Orin | CPU | Python | 2 ms |
| ByteTrack tracking | Orin | CPU | NumPy | 3 ms |
| Kalman + prediction | Orin | CPU | NumPy | 2 ms |
| LIDAR processing | Orin | CPU | NumPy | 8 ms |
| LIDAR-camera fusion | Orin | CPU | NumPy | 3 ms |
| Radar processing | Orin | CPU | NumPy | 3 ms |
| Radar-camera fusion | Orin | CPU | NumPy | 2 ms |
| Interaction model | Orin | CPU | Python | 1 ms |
| Fusion + world model | Orin | CPU | Python dataclass | 2 ms |
| Safety evaluation | Orin | CPU | Rule-based | 3 ms |
| Safety plausibility | TDA4VM | DSP | Lightweight C impl | 5 ms |
| Control output | Orin | CPU | PID/MPC | 2 ms |
| **Total (parallel)** | | | | **~25 ms** |

Parallel execution of detection + depth + segmentation reduces the perception block from the sum of individual stages (~24 ms) to the slowest single stage (~10 ms).

End-to-end target: camera-to-actuator in under 40 ms at 25+ FPS.

## Latency Budget Breakdown

```
|<------ Orin GPU (parallel) ------>|<-- Orin CPU (serial) --->|
|                                   |                          |
| Detection    8 ms  \              |                          |
| Depth       10 ms   > max = 10ms | Track      3 ms          |
| Segmentation 6 ms  /             | Kalman     2 ms          |
|                                   | LIDAR      8 ms *        |
|                                   | Radar      3 ms          |
|                                   | Fusion     2 ms          |
|                                   | Interaction 1 ms          |
|                                   | Safety     3 ms          |
|                                   | Control    2 ms          |
|                                   |                          |
|<------------ 10 ms ------------->|<-------- 24 ms ---------->|
|                                                              |
| * LIDAR can be parallelized with tracking in future          |
|                                                              |
|<------------------ 34 ms (Orin total) --------------------->|
|                                                              |
| + ISP 2ms + network 1ms -> ~37 ms end-to-end                |
```

## Model Optimization Path

```
PyTorch (.pt)
  |
  | scripts/export_onnx.py --model all
  v
ONNX FP32 (.onnx)
  |
  +-- scripts/quantize_int8.py --all      --> ONNX INT8 (CPU deployment)
  |
  +-- scripts/build_tensorrt.py --fp16    --> TRT FP16 (Orin GPU)
  |
  +-- trtexec --int8 --calib=...          --> TRT INT8 (Orin DLA)
```

| Format | Use Case | Typical Speedup vs PyTorch |
|--------|----------|---------------------------|
| ONNX FP32 | CPU inference, portability | 1.5-2x |
| ONNX INT8 | CPU edge deployment | 2-3x |
| TRT FP16 | Orin GPU | 3-5x |
| TRT INT8 | Orin DLA | 5-8x |

## Failover Strategy

### Level 1: Graceful Degradation (Orin-internal)

The health monitor (`src/runtime/health_monitor.py`) and sensor health monitor (`src/safety/sensor_health.py`) track per-frame latency and sensor quality. When latency exceeds the watchdog threshold for N consecutive frames, or sensor health drops below threshold:

1. Depth and segmentation are disabled (non-safety-critical).
2. Saliency and recording are disabled.
3. Tracking interval doubles (process every other frame).
4. Detection + tracking + safety continue at reduced load.
5. DTC logged via `dtc_logger.py`.
6. System recovers automatically when conditions improve.

### Level 2: Sensor Dropout (Camera / LIDAR / Radar)

The sensor health monitor provides continuous per-sensor scoring:

| Sensor | Healthy | Degraded | Failed |
|--------|---------|----------|--------|
| Camera | > 0.7 | 0.3-0.7 | < 0.3 |
| LIDAR | > 0.7 | 0.3-0.7 | < 0.3 |
| Radar | > 0.7 | 0.3-0.7 | < 0.3 |

When a sensor fails:
1. Pipeline continues with remaining sensors (camera-only, camera+radar, etc.).
2. Sensor-specific fusion stages are skipped.
3. Safety margins are extended for reduced confidence.
4. DTC logged with sensor ID and failure mode.

### Level 3: ECU Disagree (Orin <-> TDA4VM)

The TDA4VM runs an independent safety check on each world-model snapshot. If results diverge beyond a configurable threshold:

1. TDA4VM flags a disagree condition.
2. Orin re-evaluates with a fallback detector (smaller model or frame skip).
3. If disagree persists for >200 ms, TDA4VM asserts AEB authority.

### Level 4: Orin Failure (Watchdog Timeout)

The TDA4VM sends a heartbeat request to Orin every 50 ms. If Orin fails to respond:

1. After 100 ms: TDA4VM switches to autonomous safety mode.
2. TDA4VM runs a minimal perception pipeline (camera via ISP, basic object detection on DSP).
3. TDA4VM commands controlled stop via CAN (progressive braking, hazard lights).
4. Driver takeover request issued on HMI.

### Level 5: Full ECU Failure

If both Orin and TDA4VM are unresponsive:

1. Hardware watchdog on CAN gateway triggers safe-stop relay.
2. Vehicle enters mechanical failsafe (hazard lights, engine cut if stationary).
3. Event logged to non-volatile storage for post-incident analysis.

## Communication Protocols

| Link | Protocol | Bandwidth | Latency |
|------|----------|-----------|---------|
| Camera -> Orin | GMSL2 / MIPI CSI-2 | 4-6 Gbps per lane | <1 ms |
| LIDAR -> Orin | Ethernet (UDP/PTP) | 100 Mbps | <1 ms |
| Radar -> Orin | CAN FD / Ethernet | 5 Mbps | <1 ms |
| Orin -> TDA4VM | Ethernet (UDP) | 1 Gbps | <1 ms |
| Orin -> Actuators | CAN FD | 5 Mbps | <1 ms |
| TDA4VM -> Actuators | CAN FD (backup) | 5 Mbps | <1 ms |
| Orin -> HMI | Ethernet / SOME/IP | 100 Mbps | <5 ms |

## Power Modes

| Mode | Orin Power | TDA4VM Power | Total | Use Case |
|------|-----------|-------------|-------|----------|
| Full | 40 W | 15 W | 55 W | Highway ADAS, all models + sensors active |
| Balanced | 25 W | 10 W | 35 W | Urban driving, INT8 models, radar-only fusion |
| Low Power | 15 W | 5 W | 20 W | Parking, detection-only, no LIDAR/radar |

## Deployment Checklist

1. Export all models to ONNX: `python scripts/export_onnx.py --model all`
2. Quantize to INT8: `python scripts/quantize_int8.py --all`
3. Build TensorRT engines on target: `python scripts/build_tensorrt.py`
4. Benchmark on target: `python scripts/benchmark_all.py --json results/bench.json`
5. Run latency budget: `python scripts/latency_budget.py`
6. Validate config: startup validation runs automatically
7. Validate safety properties: `pytest tests/test_safety_properties.py tests/test_iso26262.py`
8. Run failure scenarios: `python scripts/failure_scenarios.py`
9. Flash TDA4VM safety image with matching world-model schema version.
10. Run end-to-end integration test on vehicle harness.
