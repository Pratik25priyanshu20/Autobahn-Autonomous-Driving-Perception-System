# Safety Case

This document captures the safety reasoning for the APS++ perception stack. It is structured around goals, hazards, mitigations, and verification.

## Top-Level Goals

- **G1**: Provide timely and correct perception outputs within latency budget (50-100 ms per frame).
- **G2**: Detect and handle degraded conditions (sensor drop, model failure, overload, adverse weather) gracefully.
- **G3**: Surface actionable warnings to the vehicle controller and/or operator.
- **G4**: Prevent safety-critical state transitions from being missed or suppressed.

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
- Lane is stable (low jitter over recent frames).
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

## Hazards and Mitigations

| ID | Hazard | Mitigation | Implementation |
|----|--------|------------|----------------|
| H1 | Missed or late detections | Watchdog deadlines in health monitor; adaptive frame skip preserves core detection | `health_monitor.py`, adaptive skip in `orchestrator.py` |
| H2 | False positives causing unnecessary braking | Confidence calibration (temperature scaling), track consistency via Kalman filter, ByteTrack high/low score matching | `calibration.py`, `kalman_tracker.py` |
| H3 | Poor perception in adverse weather | Weather/visibility detector triggers degraded mode; extend safety margins; log condition | `visibility_detector.py`, `orchestrator.py` |
| H4 | TTC misestimation | TTC bounds with conservative clamping; cross-check against Kalman-filtered velocity; EMA smoothing | `ttc_filter.py`, `kalman_tracker.py` |
| H5 | Configuration drift | Versioned configs; checksums logged at startup; required schema validation; git hash in metrics | `app.py`, `system.yaml` |
| H6 | Blind spot collisions | BSD/RCTA monitoring lateral tracks; configurable zones and TTC thresholds | `bsd_rcta.py`, `safety_manager.py` |
| H7 | Sensor failure | Health monitor consecutive-miss detection; degraded mode with graceful feature shedding | `health_monitor.py` |
| H8 | Occupancy grid misses obstacles | Conservative projection (all non-drivable pixels are occupied); configurable resolution | `occupancy_grid.py` |

## Safety Monitoring

- **Latency budgets**: Per-module watchdog; trips publish `degraded_mode=True` in `SafetyStatus`.
- **Confidence thresholds**: Detection confidence calibration prevents overconfident false positives.
- **Track consistency**: Kalman filter smooths noisy position/velocity estimates; temporal predictor flags divergent trajectories.
- **Weather awareness**: Visibility detector runs per-frame; degraded conditions trigger extended safety margins.
- **Heartbeats**: Health monitor tracks consecutive misses; prolonged degradation triggers feature shedding.

## Degraded Mode Strategy

When the health monitor detects sustained overload or the weather detector reports degraded visibility:

1. **Shed non-essential modules** — disable depth estimation and segmentation first.
2. **Double tracking interval** — process tracking every other frame while maintaining detection.
3. **Extend safety margins** — increase TTC warning thresholds.
4. **Maintain core safety** — FCW and LDW continue with latest available data.
5. **Log and alert** — degraded events are logged to `safety_events.jsonl` and surfaced on the HUD.
6. **Auto-recover** — when latency drops below threshold or visibility improves, restore full pipeline.

## Control Safety Gating

When control modules are enabled (`control.enabled: true`):

- **Pure Pursuit**: Throttle is reduced under WARNING state; braking is applied under CRITICAL.
- **MPC**: Safety constraint is injected into the optimization; trajectory is limited under elevated risk.
- **Kill switch**: If safety state is CRITICAL for sustained duration, control commands are zeroed.

## Verification Approach

### Unit Tests

| Test File | Coverage |
|-----------|----------|
| `test_safety.py` | TTC/risk rule evaluation |
| `test_safety_properties.py` | Property-based: TTC non-negative, risk finite, FCW state valid, SafetyManager never crashes |
| `test_kalman_properties.py` | Kalman covariance PSD, update returns finite values |
| `test_occupancy.py` | Empty grid, depth population, drivable mask suppression |
| `test_visibility.py` | Dark/bright/normal/fog frame classification |
| `test_perception_contracts.py` | Perception output schema validation |

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

### CI Safety Gate

`.github/workflows/ci.yml` includes a dedicated safety test job:

```yaml
safety-tests:
  runs-on: ubuntu-latest
  steps:
    - run: pytest tests/test_safety*.py -v
```

### Scenario Testing

- Replay datasets with ground truth for latency, miss rate, and false positive measurement.
- Fault injection: simulate dropped frames, delayed models, and noisy detections to confirm degrade behavior.
- Weather simulation: synthetic fog/dark/glare frames to validate visibility detector and degraded mode triggering.

## Evaluation Framework

`src/evaluation/` provides offline safety response evaluation:

- **Safety response latency** (`safety_metrics.py`): Measures time from ground-truth event to system response.
- **MOT metrics** (`mot_metrics.py`): MOTA/MOTP/ID-switches for tracking reliability.
- **Lane metrics** (`lane_metrics.py`): Precision/recall/F1 for lane detection accuracy.

Run evaluation:

```bash
python scripts/evaluate.py --mode safety --predictions results/ --ground-truth data/gt/
```
