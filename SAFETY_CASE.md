# Safety Case

This document captures the initial safety reasoning for the perception stack. It is structured around goals, hazards, mitigations, and verification.

## Top-level goals
- G1: Provide timely and correct perception outputs within latency budget (e.g., 50–100 ms per frame).
- G2: Detect and handle degraded conditions (sensor drop, model failure, overload) gracefully.
- G3: Surface actionable warnings to the vehicle controller and/or operator.

## Hazards and mitigations
- H1: Missed or late detections → **M1:** watchdog deadlines in `health_monitor`, fallback to lighter model or lower resolution.
- H2: False positives leading to unnecessary braking → **M2:** fusion smoothing and track consistency checks.
- H3: Poor lane/segmentation in adverse weather → **M3:** confidence gating and degrade-to-minimal safe mode (e.g., slow/stop).
- H4: Time-to-collision misestimation → **M4:** TTC bounds with conservative clamping and cross-check against relative velocity.
- H5: Configuration drift → **M5:** versioned configs, checksums logged at startup, and required schema validation.

## Safety monitoring
- Latency budgets per module; watchdog trips publish `degraded_mode=True` in `SafetyStatus`.
- Confidence thresholds and disagreement counters trigger warning escalation (visual + log).
- Heartbeats for inputs and inference workers; missing beats cause re-init or stop.

## Degraded mode strategy
- Drop non-essential perception heads first (depth/segmentation) to maintain object detection + tracking.
- Lower input resolution or frame rate while keeping TTC computation active.
- If fusion confidence falls below threshold, command `minimal risk` behavior to controller.

## Verification approach
- Unit tests for TTC/risk formulas (see `tests/test_safety.py`).
- Contract tests for perception outputs to ensure schemas and confidence ranges.
- Scenario tests on replay datasets to measure latency, miss rate, and false positives.
- Fault injection: simulate dropped frames, delayed models, and noisy detections to confirm degrade behavior.
