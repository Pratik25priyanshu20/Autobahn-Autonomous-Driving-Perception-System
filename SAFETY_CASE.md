# Safety Case

This document captures the safety reasoning for the APS++ perception stack, structured around goals, hazards, mitigations, and verification. The safety layer is designed with ISO 26262 alignment, including ASIL classification, plausibility checking, and diagnostic trouble code logging.

## Top-Level Goals

- **G1**: Provide timely and correct perception outputs within latency budget (≤ 50 ms per frame).
- **G2**: Detect and handle degraded conditions (sensor drop, model failure, overload, adverse weather) gracefully.
- **G3**: Surface actionable warnings to the vehicle controller and/or operator.
- **G4**: Prevent safety-critical state transitions from being missed or suppressed.
- **G5**: Validate config correctness at startup to prevent misconfigured deployments.
- **G6**: Provide cross-module plausibility checks to catch inconsistent perception outputs.

## Safety Systems

### Forward Collision Warning (FCW)

TTC-based state machine with four levels:

| State | Condition | Response |
|-------|-----------|----------|
| NORMAL | TTC > 4.0 s or no lead vehicle | No action |
| PRE | 2.5 s < TTC ≤ 4.0 s | Visual indicator, logging |
| WARNING | 1.5 s < TTC ≤ 2.5 s | Audible alert, control speed reduction |
| CRITICAL | TTC ≤ 1.5 s | Emergency alert, control applies braking |

TTC smoothing via `TTCFilter` (`src/adas/ttc_filter.py`) reduces flicker from noisy distance estimates. Thresholds are configurable in `configs/safety.yaml`.

### Lane Departure Warning (LDW)

Triggers when:
- Lane confidence ≥ 0.7 (both lines detected).
- Lane is stable (low jitter over recent frames via EMA smoothing).
- Ego offset exceeds the configured threshold.

LDW is gated by lane stability to avoid false warnings on unclear road markings. The `ldw_allowed` flag prevents nuisance alerts.

### Blind Spot Detection (BSD) / Rear Cross-Traffic Alert (RCTA)

`src/safety/bsd_rcta.py` monitors tracks in the lateral blind-spot zones:

- **Blind spot X range**: configurable lateral extent (default: 1.5-4.0 m from ego center).
- **Blind spot Y range**: configurable longitudinal extent (default: -3.0 to 3.0 m).
- **TTC threshold**: lateral TTC warning threshold (default: 2.0 s).
- **Warning levels**: WARNING (object present in zone) or CRITICAL (closing laterally with low TTC).

BSD warnings are integrated into the unified safety manager and visualized on the overlay.

### Occupancy Grid

`src/safety/occupancy_grid.py` projects non-drivable pixels from depth + segmentation into a BEV occupancy grid:

- Resolution: configurable (default: 0.5 m per cell).
- Max range: configurable (default: 20.0 m).
- Drivable pixels are suppressed (not marked as occupied).
- The grid is available in the world model for downstream reasoning.

### Weather / Visibility Detection

`src/perception/weather/visibility_detector.py` classifies frame visibility:

| Condition | Trigger | Action |
|-----------|---------|--------|
| clear | Normal brightness + contrast | Normal operation |
| fog | Low contrast + medium brightness | Warning, extend safety margins |
| dark | Low brightness | Warning, may trigger degraded mode |
| glare | High brightness | Warning, may trigger degraded mode |

When `degraded=True`, the orchestrator extends safety margins and logs the condition.

### Sensor Health Monitoring

`src/safety/sensor_health.py` provides real-time per-sensor quality scoring:

| Sensor | Assessment Method | Degradation Indicators |
|--------|-------------------|----------------------|
| Camera | Brightness (mean pixel value), blur (Laplacian variance), occlusion (Canny edge density) | Dark frame → score 0.06; normal → score 0.95 |
| LIDAR | Point count ratio vs expected, intensity distribution check | 50 points vs 10K expected → score 0.30 |
| Radar | Detection consistency via exponential moving average | 0 detections after consistent 10 → consistency 0.0 |

Overall health is a weighted average. When below threshold (default 0.5), the system triggers degradation warnings. Sensor health scores are visualized on the overlay HUD and recorded in the world model.

### ISO 26262 Safety Components

#### ASIL Classification (`src/safety/asil_classifier.py`)

Classifies hazard scenarios into ASIL levels (A through D) based on:
- Severity of potential harm
- Probability of exposure
- Controllability by the driver

Each detected hazard is assigned an ASIL level that determines the rigor of required validation.

#### Plausibility Checker (`src/safety/plausibility_checker.py`)

Cross-module sanity validation that runs after the safety manager:
- Detection count vs track count coherence (detect orphaned tracks or missing detections)
- TTC bounds validation (TTC should be non-negative, finite, and consistent with track velocity)
- Safety state consistency (state transitions follow valid FSM paths)

#### Redundant Detector (`src/safety/redundant_detector.py`)

Lightweight fallback detector used for cross-validation:
- Runs independently of the primary detector (simpler model or heuristic)
- Flags disagreements above a configurable threshold
- Provides defense-in-depth against primary detector failures

#### DTC Logger (`src/safety/dtc_logger.py`)

ISO 14229-style diagnostic trouble code logging:
- Logs structured DTCs to `dtc_log.jsonl` with timestamps and severity
- Covers sensor failures, model timeouts, state machine violations, and config errors
- Supports automated post-incident analysis

### Config Validation (`src/utils/config_validator.py`)

Startup validation catches misconfigurations before the pipeline runs:

| Check Type | Examples |
|------------|---------|
| Enum validation | `perception.runtime` must be in `{pytorch, onnx, tensorrt}` |
| Range validation | `perception.conf_thres` must be in [0.0, 1.0] |
| Dependency validation | `radar_fusion.enabled=true` requires `radar.enabled=true` |
| Safety config | `sensor_health.brightness_range` min must be < max |

Invalid configs raise `ConfigValidationError` with clear error messages listing all violations.

### Interaction Model (`src/prediction/interaction_model.py`)

Rule-based behavioral prediction for safety-relevant traffic scenarios:

| Heuristic | Description | Safety Relevance |
|-----------|-------------|-----------------|
| Gap acceptance | Checks if gap in adjacent lane exceeds minimum | Merge/lane-change safety |
| Yield heuristic | German right-before-left rule | Intersection safety |
| Following distance | 2-second rule validation | Rear-end collision prevention |
| Cut-in prediction | Lateral velocity toward ego lane | Cut-in collision avoidance |

Each heuristic produces an `InteractionEvent` with risk level, involved track IDs, and estimated time-to-event.

## Hazards and Mitigations

| ID | Hazard | Mitigation | Implementation |
|----|--------|------------|----------------|
| H1 | Missed or late detections | Watchdog deadlines in health monitor; adaptive frame skip preserves core detection | `health_monitor.py`, adaptive skip in `orchestrator.py` |
| H2 | False positives causing unnecessary braking | Confidence calibration (temperature scaling), track consistency via Kalman filter, ByteTrack high/low score matching | `calibration.py`, `kalman_tracker.py` |
| H3 | Poor perception in adverse weather | Weather/visibility detector triggers degraded mode; extend safety margins; log condition | `visibility_detector.py`, `orchestrator.py` |
| H4 | TTC misestimation | TTC bounds with conservative clamping; cross-check against Kalman-filtered velocity; EMA smoothing; plausibility validation | `ttc_filter.py`, `kalman_tracker.py`, `plausibility_checker.py` |
| H5 | Configuration drift | Versioned configs; startup validation catches invalid values; checksums logged; git hash in metrics | `config_validator.py`, `app.py` |
| H6 | Blind spot collisions | BSD/RCTA monitoring lateral tracks; configurable zones and TTC thresholds | `bsd_rcta.py`, `safety_manager.py` |
| H7 | Sensor failure | Sensor health monitor with per-sensor scoring; degraded mode with graceful feature shedding | `sensor_health.py`, `health_monitor.py` |
| H8 | Occupancy grid misses obstacles | Conservative projection (all non-drivable pixels are occupied); configurable resolution | `occupancy_grid.py` |
| H9 | Primary detector failure | Redundant detector cross-validates; disagreements trigger DTC and fallback mode | `redundant_detector.py`, `dtc_logger.py` |
| H10 | Cut-in collision | Interaction model detects lateral velocity toward ego lane; time-to-cut-in warning | `interaction_model.py` |
| H11 | Inconsistent perception outputs | Plausibility checker validates detection/track/TTC coherence post-safety | `plausibility_checker.py` |
| H12 | LIDAR/radar sensor dropout | Sensor health scoring detects point density and detection consistency drops; pipeline continues with camera-only fallback | `sensor_health.py`, `orchestrator.py` |

## Safety Monitoring

- **Latency budgets**: Per-module watchdog; trips publish `degraded_mode=True` in `SafetyStatus`.
- **Confidence thresholds**: Detection confidence calibration prevents overconfident false positives.
- **Track consistency**: Kalman filter smooths noisy position/velocity estimates; temporal predictor flags divergent trajectories.
- **Weather awareness**: Visibility detector runs per-frame; degraded conditions trigger extended safety margins.
- **Heartbeats**: Health monitor tracks consecutive misses; prolonged degradation triggers feature shedding.
- **Sensor health**: Per-sensor quality scores updated every frame; degradation triggers warnings and DTC logging.
- **Cross-validation**: Redundant detector + plausibility checker provide defense-in-depth.
- **Config safety**: Startup validation prevents misconfigured deployments from running.

## Degraded Mode Strategy

When the health monitor detects sustained overload, the weather detector reports degraded visibility, or sensor health drops below threshold:

1. **Shed non-essential modules** — disable depth estimation, segmentation, saliency, and recording first.
2. **Double tracking interval** — process tracking every other frame while maintaining detection.
3. **Extend safety margins** — increase TTC warning thresholds.
4. **Maintain core safety** — FCW and LDW continue with latest available data.
5. **Log and alert** — degraded events are logged to `safety_events.jsonl` and DTCs to `dtc_log.jsonl`; surfaced on the HUD.
6. **Auto-recover** — when latency drops below threshold, visibility improves, or sensor health recovers, restore full pipeline.

## Control Safety Gating

When control modules are enabled (`control.enabled: true`):

- **Pure Pursuit**: Throttle is reduced under WARNING state; braking is applied under CRITICAL.
- **MPC**: Safety constraint is injected into the optimization; trajectory is limited under elevated risk.
- **Kill switch**: If safety state is CRITICAL for sustained duration, control commands are zeroed.

## Verification Approach

### Unit Tests (27 modules, 179 tests)

| Test File | Coverage | Tests |
|-----------|----------|-------|
| `test_safety.py` | TTC/risk rule evaluation | Core safety logic |
| `test_safety_properties.py` | Property-based: TTC non-negative, risk finite, FCW state valid, SafetyManager never crashes | Hypothesis-based |
| `test_iso26262.py` | ASIL classifier, plausibility checker, DTC logger, redundant detector | 26 tests |
| `test_kalman_properties.py` | Kalman covariance PSD, update returns finite values | Property-based |
| `test_occupancy.py` | Empty grid, depth population, drivable mask suppression | 3 tests |
| `test_visibility.py` | Dark/bright/normal/fog frame classification | 4 tests |
| `test_perception_contracts.py` | Perception output schema validation | Contract tests |
| `test_sensor_health.py` | Normal/dark/blurred camera, low point cloud, empty radar | 14 tests |
| `test_interaction.py` | Safe/unsafe gap, yield detection, following violation, cut-in | 11 tests |
| `test_config_validator.py` | Enum/range/dependency/safety config validation | 16 tests |
| `test_orchestrator_unit.py` | Extracted stage methods (preprocess, detect, track, safety, etc.) | 10 tests |
| `test_e2e_smoke.py` | End-to-end pipeline: single frame, multi-frame, lane integration, safety output | 5 tests |
| `test_radar.py` | Ghost filter, cartesian math, clustering, projection, fusion, CSV parsing | 17 tests |
| `test_lidar.py` | RANSAC, voxelization, clustering, BEV encoding, KITTI loading, fusion | 16 tests |
| `test_recording.py` | Record/replay round-trip, interval skipping, compression | 7 tests |
| `test_saliency.py` | Output shape, normalization, overlay blending, no-detections case | 8 tests |

### Property-Based Testing (Hypothesis)

Safety-critical properties verified with random inputs:

- `compute_ttc()` returns non-negative values for positive distance and speed.
- `compute_ttc()` returns `None` for diverging vehicles.
- `risk_score()` always returns a finite float.
- `fcw_state()` always returns a valid state string.
- `SafetyManager.evaluate()` never raises an exception regardless of input combination.
- Kalman filter covariance remains positive semi-definite after arbitrary predict/update cycles.
- Kalman `update()` always returns finite state vectors.

### Integration Tests

- `test_end_to_end.py`: Verifies the full import chain loads without errors.
- `test_fusion.py`: Verifies perception-to-world-model fusion preserves detection data.
- `test_e2e_smoke.py`: Runs the full orchestrator pipeline on synthetic frames, validating world model output structure.

### CI Safety Gate

`.github/workflows/ci.yml` includes a dedicated safety test job:

```yaml
safety-tests:
  runs-on: ubuntu-latest
  steps:
    - run: pytest tests/test_safety*.py tests/test_iso26262.py -v
```

`.github/workflows/model-regression.yml` prevents model quality regressions:

```yaml
regression-check:
  # Fails if mAP drops >5% or latency increases >20%
  - run: python scripts/benchmark_all.py --compare baselines/benchmark_baseline.json
```

### Scenario Testing

- Replay datasets with ground truth for latency, miss rate, and false positive measurement.
- Fault injection: simulate dropped frames, delayed models, and noisy detections to confirm degraded behavior.
- Weather simulation: synthetic fog/dark/glare frames to validate visibility detector and degraded mode triggering.
- Sensor failure: `scripts/failure_scenarios.py` demonstrates camera blackout, LIDAR dropout, and radar inconsistency with measured degradation scores.
- CARLA scenarios: `scripts/run_scenarios.py` runs controlled scenarios in CARLA simulator with metrics collection.

### Controlled Failure Demonstrations

Run `python scripts/failure_scenarios.py` to verify graceful degradation:

| Scenario | Normal Score | Degraded Score | System Response |
|----------|-------------|----------------|-----------------|
| Camera blackout | 1.00 | 0.06 | Degradation warning, feature shedding |
| LIDAR dropout (50 pts) | 1.00 | 0.30 | Warning logged, camera-only fallback |
| Radar inconsistency | 1.00 | 0.00 | Consistency drop flagged, DTC logged |

## Evaluation Framework

`src/evaluation/` provides offline safety response evaluation:

- **Safety response latency** (`safety_metrics.py`): Measures time from ground-truth event to system response.
- **MOT metrics** (`mot_metrics.py`): MOTA, MOTP, IDF1, HOTA, ID-switches for tracking reliability.
- **Lane metrics** (`lane_metrics.py`): Precision/recall/F1 for lane detection accuracy.
- **MOT formatter** (`mot_formatter.py`): Parsers for MOT17 and KITTI tracking formats.

Run evaluation:

```bash
python scripts/evaluate.py --mode safety --predictions results/ --ground-truth data/gt/
python scripts/evaluate_mot.py --gt data/gt/ --pred results/ --format mot17
python scripts/generate_metrics_report.py   # Generates demo/METRICS.md
```
