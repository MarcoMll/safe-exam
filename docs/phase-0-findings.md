# ExamGuard — Phase 0 Findings Report

> **Issue #12** — Consolidation of all Phase 0 calibration results.
> Go/no-go gate for Phase 1.
> Written: 2026-07-27
>
> ⚠️ **WIP — awaiting teammate experiment results.** Current data is from one hardware setup (desktop PC). Sections will be updated as teammates add their runs.

---

## 1. Detection Pipeline Summary

Phase 0 built and validated a unified frame processor combining two detection stacks:

**Object detection (YOLO)**
Model: `yolo26s.pt` (COCO pretrained). Single inference call per frame, results split by class — class 67 (`cell phone`) for phone detection, class 0 (`person`) for person count. Phone detection outputs a confidence score per frame; person detection identifies the primary subject (largest bounding box) and flags any additional detected persons as potential intruders.

**Gaze estimation (MediaPipe FaceMesh)**
MediaPipe FaceMesh with refined landmarks. Outputs multiple raw signals per frame: `head_yaw`, `head_pitch`, `eye_offset`, `gaze_yaw` (combined head + eye), and `iris_offset`. The runtime attention policy selects one signal and threshold to use as the off-center indicator. All signals are logged regardless of which is active — this allows backtesting without re-recording.

**Person intrusion policy**
A spatial policy layered on top of raw person count. Rather than flagging any `person_count > 1`, the policy requires a secondary person to satisfy a minimum area fraction, proximity overlap, or center-ROI conditions (2 of 3 rules must match). This distinguishes a background classmate from someone leaning into the camera's close zone.

**Capture loop**
Configurable target FPS (default 5 for production, 10 for debug/calibration). 15-second warmup discarded before profiling. Headless mode available — no debug overlay — for clean CPU measurement.

---

## 2. Recommended Thresholds

### Phone detection

**Threshold: `0.50` confidence**

| Threshold | False positive rate (non-phone scenarios) | True positive rate (phone scenarios) |
|-----------|------------------------------------------|--------------------------------------|
| 0.25 | 14.1% | 44.8% |
| 0.35 | 11.0% | 43.3% |
| 0.45 | 10.0% | 38.1% |
| **0.50** | **9.0%** | **36.7%** |
| 0.55 | 8.5% | 34.1% |
| 0.60 | 8.2% | 27.4% |

0.50 is the crossover point where the FP rate drops under the Issue #7 / #12 target of <10% without meaningfully sacrificing TP rate versus 0.55 or 0.60. Excluding the known dark water bottle outlier, FP drops to ~1.4%. Normal exam behavior (empty frame, fidgeting, looking around, hand-to-face, pen, notebook) is clean at 0.50.

### Gaze off-screen

**Profile B: `gaze_yaw`, `yaw_only`, 5° yaw threshold, 4s duration, 0.4s gap tolerance**

Two profiles emerged from backtesting experiment 1:

| | Profile A — Conservative | Profile B — Sensitive |
|---|---|---|
| Signal | `head_yaw` | `gaze_yaw` (head + eye) |
| Mode | yaw_only | yaw_only |
| Yaw threshold | 5° | 5° |
| Pitch threshold | 99° (disabled) | 99° (disabled) |
| Duration target | 6s | 4s |
| Gap tolerance | 0.4s | 0.4s |
| Suspicious scenarios detected | 4 / 6 | 5 / 6 |
| Natural FP scenarios | 2 / 6 | 3 / 6 |
| Writing / reading paper | Clean | Clean |
| Backtest score | 54.19 | 49.65 |

**Current default: Profile B.** It fires on more suspicious behaviors and keeps writing/reading clean — better for debugging and Phase 1 test development. The extra natural FPs (stretch, drink water) are expected to be filtered by Phase 1 duration + pattern logic. Profile A is the conservative alternative for contexts where raw signal noise must be minimized.

The yaw-only rule is the key finding from this experiment: pitch+yaw symmetric thresholds fail because reading and writing both involve pitch-down, which would flag normal exam behavior at any reasonable threshold.

### Person intrusion

**Default policy unchanged — not calibrated on real data.**

| Parameter | Default value |
|-----------|--------------|
| `roi_center_fraction` | 0.60 |
| `min_secondary_area_pct` | 0.05 |
| `primary_overlap_iou` | 0.10 |
| `min_rules_to_match` | 2 |

Experiment 1 was a pilot with a held laptop and a simulated second person on a monitor — not suitable for threshold tuning. The default policy produced zero false positives across all scenarios (directionally good), but the simulated intruder never triggered because its bbox area (~1.6% of frame) fell below the `min_secondary_area_pct = 0.05` threshold. This means the test never produced a true positive, so no Profile A/B comparison could be made. **The default values have not been validated. A real-setup experiment (helper, fixed-desk, classroom or similar) is required before these numbers can be trusted.**

---

## 3. CPU Profile Results

**Hardware:** Intel64 Family 6 Model 158 Stepping 13, 8 logical cores, Windows 11 (10.0.22631)
**Model stack:** `yolo26s.pt` + MediaPipe FaceMesh (refine landmarks)
**Protocol:** Headless, 15s warmup discarded

### Drill-down @ 5fps (30 seconds)

| Mode | Actual FPS | Avg machine CPU % | Peak machine CPU % | Avg RAM MB | Avg inference ms |
|------|------------|-------------------|--------------------|------------|------------------|
| `object` (YOLO only) | 5.03 | 27.1 | 30.5 | 429 | 82.3 |
| `face_gaze` (MediaPipe only) | 5.03 | 0.05 | 0.4 | 472 | 4.7 |
| `both` (full pipeline) | 5.03 | 27.1 | 37.7 | 475 | 84.3 |

**Key insight:** YOLO is ~95%+ of cost. MediaPipe is effectively free. Optimizing gaze will not improve CPU headroom — YOLO is the target.

### Sustained pipeline (`both` mode, 10 minutes)

| Target FPS | Actual FPS | Hit target? | Avg machine CPU % | Peak machine CPU % | Avg RAM MB | Avg inference ms |
|------------|------------|-------------|-------------------|--------------------|------------|------------------|
| **5** | **4.97** | Yes | **31.5** | 54.3 | 458 | 98.8 |
| **10** | **9.63** | Almost | **38.3** | 65.0 | 458 | 81.8 |
| **12** | **10.45** | **No** | **40.7** | 64.8 | 461 | 88.8 |
| **30** | **10.39** | **No** | **39.1** | 56.6 | 461 | 90.2 |

Hard ceiling on this hardware is approximately **10.4 fps** — targets of 12 or 30 produce the same frame rate. Trust 10-minute numbers over 30-second runs; sustained CPU (31.5% at 5fps) is higher than short-run (27.1%).

### Recommended FPS

| Use case | FPS | Rationale |
|----------|-----|-----------|
| **Production (exam companion)** | **5** | Sustains target; closest to <30% machine-CPU goal (31.5% sustained — marginal). ~20 frames over a 4s gaze streak is sufficient for duration-based detection. |
| **Calibration / debug / demo** | **10** | ~9.6fps sustained; denser signal. Use when machine is dedicated. |
| **Do not use** | ≥12 | Cannot sustain; same ~10.4fps ceiling as 30fps. |
| **Only if SEB forces it** | 2–3 | Not measured; use only if browser + SEB leave too little headroom at 5fps. |

The <30% machine-CPU target from Issue #10 is **marginally missed** at 5fps (31.5% avg, 54.3% peak). This is documented, not ignored. The margin is small and the 30-second drill-down looks cleaner (27.1%) — the 10-minute number is the one to trust.

---

## 4. Scenario Test Results

### Phone detection @ 0.50 threshold

**Non-phone scenarios (false positives) — target: <10%**

| Scenario | Detection rate @ 0.50 | Result |
|----------|-----------------------|--------|
| Empty frame | 0% | Pass |
| Empty frame + fidgeting | 0% | Pass |
| Empty frame + looking around | 0% | Pass |
| Hand to face | 0% | Pass |
| Drinking water | 0% | Pass |
| Pen by face | 0% | Pass |
| Pen moving | 0% | Pass |
| Notebook | 0% | Pass |
| Headset | 3% | Pass |
| Dark water bottle (partial / half) | 3% | Pass |
| Dark water bottle (drinking) | 0% | Pass |
| **Dark water bottle fully in frame** | **100%** | **FAIL — known outlier** |

Aggregate non-phone FP rate: 9.0% (within target). Excluding the dark water bottle outlier: ~1.4%.

**Phone scenarios (true positives)**

| Scenario | Detection rate @ 0.50 | Result |
|----------|-----------------------|--------|
| Phone in hand near face | 97% | Strong |
| Phone lower frame | 93% | Strong |
| Phone hidden by hand | 67% | Partial |
| Phone hidden by hand + half in frame | 33% | Weak |
| Phone flashed briefly | 13% | Weak |
| Phone sideways | 13% | Weak |
| Phone at edge of frame | 7% | Weak |
| Phone moved quickly lower frame | 7% | Weak |
| Phone moved quickly | 0% | Miss |

Obvious, sustained phone poses are reliably detected. Brief, edge, and partially-concealed phones are frequently missed at the threshold level alone — this is expected and scoped to Phase 1 temporal logic.

### Gaze (Profile B: gaze_yaw, yaw_only, 5°, 4s)

| Scenario | Longest streak | Fires @4s? | Assessment |
|----------|---------------|------------|------------|
| Natural writing | 1.34s | No | Pass |
| Natural reading paper | 3.77s | No | Pass |
| Natural stretch | 8.87s | Yes | FP — expected noise |
| Natural drink water | 11.64s | Yes | FP — expected noise |
| Natural fidgeting | 4.61s | Yes | FP — extra noise vs Profile A |
| Suspicious phone side (medium) | 8.79s | Yes | Detected |
| Suspicious phone side (long) | 5.27s | Yes | Detected |
| Suspicious eyes-only side | 14.73s | Yes | Detected |
| Suspicious phone under desk (medium) | 4.69s | Yes | Detected |
| Suspicious phone under desk (long) | 1.34s | No | Miss |

Writing and reading paper are clean. Stretch and drink water produce false raw signals — expected; Phase 1 pattern logic should filter these. Under-desk long pose is missed even at 4s Profile B.

### Person intrusion (pilot only — not validated)

| Scenario | Intrusion triggered | Notes |
|----------|--------------------|----|
| Solo normal exam | 0% | Correct — valid control |
| Background simulation (monitor) | 0% | Correct — but setup invalid (YOLO never saw a second person) |
| Intrusion (person on screen, leaning) | 0% | Incorrect — bbox ~1.6% of frame, below min_secondary_area_pct |
| Intrusion (person peeking) | 0% | Incorrect — secondary rarely detected |

Pipeline and CSV logging work correctly. Zero FP in controlled conditions is directionally good, but the test never produced a valid positive because the simulated second person was too small to satisfy policy thresholds. **Results are insufficient for go/no-go on intrusion detection — real experiment required.**

---

## 5. Known Limitations

### Phone detection
- **Dark water bottle fully in frame** is a persistent false positive (mean confidence ~0.85). Partial and drinking poses of the same bottle behave normally.
- **Brief, edge, and sideways phone appearances** are largely missed by threshold-based detection alone. Temporal logic (flag if phone seen in N of M frames) is the fix — scoped to Phase 1.
- **Single hardware profile.** All experiments used one desktop PC webcam with head/shoulders framing. Other cameras, laptops, or crops may need separate calibration runs.

### Gaze estimation
- **Stretch and drink water** produce yaw streaks of 8–12s — longer than the 4s threshold. Phase 1 duration + context logic should allow brief excursions or use secondary signals.
- **Eye-only cheating** was not reliably captured by the `eye` signal in experiment 1; the head moved enough to register on `head_yaw` first. More `suspicious_eyes_only` captures with deliberate eye isolation are needed.
- **Burst patterns** (multiple short glances in the same direction that never individually reach the duration threshold) are not evaluated. Duration-only is a Phase 0 baseline — burst/repeated-direction pattern logic is a Phase 1 direction.
- **Under-desk phone** (long duration) is missed at Profile B 4s. The under-desk long capture only produced a 1.34s streak, suggesting the gaze angle is too small when looking down vs. the yaw-only rule.

### Person intrusion
- Policy is not validated on real data. Experiment 1 used a held laptop and simulated intruder — both invalidate threshold tuning.
- Adjacent-seat glances (eyes only, no entry into camera frame) are out of scope and will not be detected by spatial intrusion logic.
- Screen viewing by a neighbor who never appears as a large/overlapping bbox will be missed.
- Requires a real helper or classroom setting to calibrate properly.

### CPU
- Only one machine tested (desktop PC, 8 cores). Weaker laptops should be profiled before locking a global FPS default.
- 5fps at 31.5% sustained average is marginal vs. the <30% target. Peaks reach 54.3%.
- If SEB + browser on weaker hardware leaves insufficient headroom, the fallback is 2–3fps (not measured) or a smaller YOLO model.

---

## 6. Go / No-Go Recommendation

**GO — proceed to Phase 1.**

The core detection pipeline is functional and calibrated. Phone and gaze thresholds are validated with real data and recommended values are documented. CPU is viable at 5fps on the test hardware. The main gap — person intrusion — is a known limitation with a clear path forward (real-setup experiment), not a blocker for Phase 1 architecture decisions.

**What Phase 1 should address immediately:**

- **Duration-based streak logic** for all signals (phone, gaze, intrusion). Phase 0 emits raw per-frame signals; Phase 1 needs the layer that converts N consecutive frames to a flag.
- **Pattern detection for burst gaze events** — multiple short glances in a direction that never individually cross the 4s threshold.
- **Stretch / drink water filtering** — Phase 1 flag logic should allow brief off-gaze excursions or require secondary context before promoting a raw gaze signal to a professor-facing flag.
- **Real person intrusion experiment** — fixed laptop on desk, real helper or classroom, run `--summarize` and `--backtest`, then update `DEFAULT_INTRUSION_POLICY` if defaults change.
- **CPU validation on weaker hardware** — re-run `--mode both --duration 600` on a laptop before locking 5fps as a global default.

**What is NOT blocking Phase 1:**
- Dark water bottle FP — known, documented, not typical exam behavior.
- Brief phone detections — temporal logic is the fix, and that's a Phase 1 build.
- Under-desk long gaze miss — the scenario is narrow; other signals (phone detection, shorter streaks) will fire first in most real cases.
