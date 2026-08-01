# ExamGuard — Phase 0 Findings Report

> **Issue #12** — Consolidation of all Phase 0 calibration results.
> Go/no-go gate for Phase 1.
> Written: 2026-07-27 · Updated: 2026-08-01
>
> Incorporates experiment 1 (desktop PC) and experiment 2 (laptop webcam). Runtime policy defaults are **not** changed in this update — recommended directions and open next steps are documented below.

---

## 1. Detection Pipeline Summary

Phase 0 built and validated a unified frame processor combining two detection stacks:

**Object detection (YOLO)**
Model: `yolo26s.pt` (COCO pretrained). Single inference call per frame, results split by class — class 67 (`cell phone`) for phone detection, class 0 (`person`) for person count. Phone detection outputs a confidence score per frame; person detection identifies the primary subject (largest bounding box) and flags any additional detected persons as potential intruders.

**Gaze estimation (MediaPipe FaceMesh)**
MediaPipe FaceMesh with refined landmarks. Outputs multiple raw signals per frame: `head_yaw`, `head_pitch`, `eye_offset`, `gaze_yaw` (combined head + eye), and `iris_offset`. The runtime attention policy selects one signal and threshold to use as the off-center indicator. All signals are logged regardless of which is active — this allows backtesting without re-recording.

**Person intrusion policy (current implementation — under reconsideration)**
A spatial policy layered on top of raw person count. Rather than flagging any `person_count > 1`, the policy requires a secondary person to satisfy a minimum area fraction, proximity overlap, or center-ROI conditions (2 of 3 rules must match). This distinguishes a background classmate from someone leaning into the camera's close zone. **Team direction after Phase 0 review: retire spatial person-intrusion as the multi-person signal and replace it with multi-face gaze analysis** (see §2 Person / multi-person and §6).

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

Gaze thresholds are **camera / setup dependent**. Experiment 1 (desktop PC) and experiment 2 (laptop webcam) produced different workable numbers on the same signal family. The team treats the laptop run as the better orientation for typical exam use and intends to move the policy toward the laptop-sensitive candidate once a labeled repeat confirms it. **Runtime defaults are unchanged in this report.**

#### Experiment 1 — desktop PC (`experiment_1_desktop_pc_camera`)

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

**Current runtime default (unchanged): Profile B** — `gaze_yaw`, yaw_only, 5°, 4s duration, 0.4s gap. It fires on more suspicious behaviors and keeps writing/reading clean on the desktop dataset.

The yaw-only rule remains the key shared finding: pitch+yaw symmetric thresholds fail because reading and writing both involve pitch-down, which would flag normal exam behavior at any reasonable threshold.

#### Experiment 2 — laptop webcam (`experiment_2_laptop`)

On the laptop setup, Profile B (yaw 5°, 4s) is **not safe**: every natural and suspicious scenario fired, including reading, writing, and stretch/drink.

| Scenario | Longest streak @ Profile B (5° / 4s) | Fires? |
|----------|--------------------------------------|--------|
| `natural_reading_paper_2_laptop` | 44.94s | Yes — FP |
| `natural_writting` | 8.28s | Yes — FP |
| `natural_stretch_and_drink_2_laptop` | 17.49s | Yes — FP |
| `suspicious_phone_side_2_laptop` | 10.14s | Yes |
| `suspicious_other_screen_2_laptop` | 23.26s | Yes |
| `suspicious_phone_under_desk_2_laptop` | 24.29s | Yes |

Laptop candidates that separate natural vs suspicious:

| | Profile C — gaze yaw (preferred direction) | Profile D — eye exploratory |
|---|---|---|
| Signal | `gaze_yaw` | `eye` |
| Mode | yaw_only | both |
| Yaw | **15°** | 3° |
| Pitch | 99° (disabled) | 10° |
| Duration | **4s** (sensitive) or 6s (stricter) | 4s |
| Gap tolerance | 0.4s | 0.4s |
| natural_fp | 0 / 3 | 0 / 3 |
| suspicious_tp | 2 / 3 | 3 / 3 |
| Writing / reading | Clean | Clean |
| Under-desk phone | Miss (yaw-only) | Hit (close to threshold) |

**Intended policy direction (not applied yet): Profile C at 4s** — `gaze_yaw`, yaw_only, **15°**, duration **4s**, gap 0.4s. Chosen because:

1. Laptop framing is closer to expected exam orientation than the desktop PC camera.
2. 4s keeps the raw layer sensitive for Phase 1 debugging and flag logic, while 15° clears natural reading/writing/stretch on the laptop data.
3. Profile D (`eye`, both axes) remains exploratory — it caught all three suspicious scenarios in experiment 2, but experiment 1 found the eye signal weaker, so it should not become the default from one laptop run.

**Caveat:** experiment 2 used non-canonical scenario labels (`natural_writting`, `natural_reading_paper_2_laptop`), so backtest writing/reading columns are unreliable; manual snapshots and `natural_fp_count` are the safer read. A labeled laptop repeat is required before changing runtime defaults.

#### Profile C snapshot (laptop — intended direction)

`gaze_yaw`, `yaw_only`, yaw `15°`, duration `4s`, gap `0.4s`

| Scenario | Longest streak | Fires @4s? | Assessment |
|----------|---------------|------------|------------|
| `natural_reading_paper_2_laptop` | 2.51s | No | Pass |
| `natural_writting` | 1.30s | No | Pass |
| `natural_stretch_and_drink_2_laptop` | 2.05s | No | Pass |
| `suspicious_phone_side_2_laptop` | 8.10s | Yes | Detected |
| `suspicious_other_screen_2_laptop` | 18.24s | Yes | Detected |
| `suspicious_phone_under_desk_2_laptop` | 0.00s | No | Miss — yaw-only; needs eye/iris/phone context |

### Person / multi-person (direction change)

**Spatial person-intrusion defaults were never validated on real data** (pilot only). The team has decided **not** to continue calibrating YOLO spatial intrusion as the multi-person product signal.

| Parameter | Default value (current code — not retuned) |
|-----------|--------------|
| `roi_center_fraction` | 0.60 |
| `min_secondary_area_pct` | 0.05 |
| `primary_overlap_iou` | 0.10 |
| `min_rules_to_match` | 2 |

Experiment 1 (person intrusion) used a held laptop and a simulated second person on a monitor — unsuitable for threshold tuning. Zero false positives in controls were directionally fine, but the simulated intruder never produced a true positive (bbox area ~1.6% of frame, below `min_secondary_area_pct`).

**New direction:** remove reliance on person-count / spatial intrusion for “second person in scene,” and instead extend **gaze estimation to multiple faces**. Goals:

1. Detect when a second (or background) person is looking toward the examinee’s screen.
2. Distinguish flags attributed to the **primary student** vs. **third-party / behind-the-student** screen looking in the same frame.
3. Support richer professor-facing outcomes (e.g. this student cheating vs. cheating visible in frame but not performed by the seated student).

This is a Phase 1 / follow-on design and experiment track — not a calibrated threshold in Phase 0.

---

## 3. CPU Profile Results

**Hardware (experiment 1 / desktop profiling):** Intel64 Family 6 Model 158 Stepping 13, 8 logical cores, Windows 11 (10.0.22631)
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

Hard ceiling on the desktop hardware is approximately **10.4 fps** — targets of 12 or 30 produce the same frame rate. Trust 10-minute numbers over 30-second runs; sustained CPU (31.5% at 5fps) is higher than short-run (27.1%).

### Cross-machine FPS observation (open issue)

Desktop sustained profiling reaches ~10 fps and cannot sustain higher targets. Informal laptop runs during teammate calibration observed roughly **~4 fps** sustained — well below the production target of 5 and far from debug/calibration targets of 10. This is a **blocking investigation item**: the team needs to determine why the pipeline cannot reach (or consistently hold) the intended FPS, and why laptop throughput is so much worse than desktop.

Open questions for the FPS investigation:

- Where is time spent (capture, YOLO, MediaPipe, post-process, overlay, I/O)?
- Is the laptop bottleneck CPU, thermal throttling, camera backend, or resolution/crop?
- Can a smaller YOLO model, lower input resolution, or frame skipping close the gap while keeping phone recall acceptable?
- What is a safe global default once laptop numbers are measured with the same headless 10-minute protocol as desktop?

### Recommended FPS (pending FPS investigation)

| Use case | FPS | Rationale |
|----------|-----|-----------|
| **Production (exam companion)** | **5** (provisional) | Sustains on the profiled desktop; closest to <30% machine-CPU goal (31.5% sustained — marginal). Laptop ~4 fps means this default is **not yet globally validated**. |
| **Calibration / debug / demo** | **10** (desktop only for now) | ~9.6fps sustained on desktop; denser signal. Not attainable on the observed laptop run. |
| **Do not use** | ≥12 | Cannot sustain on desktop; same ~10.4fps ceiling as 30fps. |
| **Only if SEB forces it** | 2–3 | Not measured; use only if browser + SEB leave too little headroom at 5fps. |

The <30% machine-CPU target from Issue #10 is **marginally missed** at 5fps on desktop (31.5% avg, 54.3% peak). Laptop CPU/FPS under the same protocol is not yet formally documented and must be completed before locking a global default.

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

### Gaze — desktop Profile B (current runtime) vs laptop Profile C (intended direction)

**Desktop experiment 1 @ Profile B** (`gaze_yaw`, yaw_only, 5°, 4s)

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

**Laptop experiment 2 @ Profile C** (`gaze_yaw`, yaw_only, 15°, 4s) — see §2 for full table. Natural reading/writing/stretch clean; side phone and other screen detected; under-desk missed by yaw-only.

Writing and reading paper are clean on both workable profiles when the yaw threshold matches the camera setup. Stretch/drink false positives appear on desktop Profile B and clear on laptop Profile C at 15°.

### Person intrusion (pilot only — superseded as product direction)

| Scenario | Intrusion triggered | Notes |
|----------|--------------------|----|
| Solo normal exam | 0% | Correct — valid control |
| Background simulation (monitor) | 0% | Correct — but setup invalid (YOLO never saw a second person) |
| Intrusion (person on screen, leaning) | 0% | Incorrect — bbox ~1.6% of frame, below min_secondary_area_pct |
| Intrusion (person peeking) | 0% | Incorrect — secondary rarely detected |

Pipeline and CSV logging work correctly, but results are insufficient for go/no-go on spatial intrusion. **The team is pivoting away from calibrating this path** toward multi-face gaze (see §2 and §6).

---

## 5. Known Limitations

### Phone detection
- **Dark water bottle fully in frame** is a persistent false positive (mean confidence ~0.85). Partial and drinking poses of the same bottle behave normally.
- **Brief, edge, and sideways phone appearances** are largely missed by threshold-based detection alone. Temporal logic (flag if phone seen in N of M frames) is the fix — scoped to Phase 1.
- **Single hardware profile for phone calibration.** Phone experiments used one desktop PC webcam with head/shoulders framing. Other cameras, laptops, or crops may need separate calibration runs.

### Gaze estimation
- **Thresholds do not transfer across setups.** Desktop Profile B (5°) fires on all natural laptop scenarios; laptop Profile C (15°) is the intended direction but needs a labeled repeat before runtime change.
- **Stretch and drink water** produce long yaw streaks on desktop Profile B. Laptop Profile C cleared these in experiment 2 — confirm on the repeat.
- **Eye-only cheating** was not reliably captured by the `eye` signal in experiment 1; Profile D looked stronger on laptop experiment 2 and remains exploratory.
- **Burst patterns** (multiple short glances in the same direction that never individually reach the duration threshold) are not evaluated. Duration-only is a Phase 0 baseline — burst/repeated-direction pattern logic is a Phase 1 direction.
- **Under-desk phone** is often missed by yaw-only gaze (desktop long pose; laptop Profile C). Eye/iris and phone-object context are needed.
- **Experiment 2 label hygiene** — non-canonical scenario names weaken automated backtest writing/reading columns. Future runs must use exact `natural_writing` / `natural_reading_paper` labels.

### Person / multi-person
- Spatial intrusion policy is not validated on real data and is **no longer the planned product approach**.
- Adjacent-seat glances and neighbor screen-looking without a large overlapping bbox are exactly the failure mode of spatial intrusion — motivating the multi-face gaze pivot.
- Multi-face gaze (detect secondary faces looking at the examinee’s screen; attribute flags to student vs third party) is **not yet implemented or calibrated**.

### CPU / FPS
- Desktop hard ceiling ~10.4 fps; production 5 fps is only marginally within the <30% CPU target.
- Laptop informal sustained rate ~4 fps — below the 5 fps production target. Formal headless profiling on laptop hardware is required.
- Root cause of the FPS ceiling (desktop and especially laptop) is **not yet diagnosed**.
- If SEB + browser on weaker hardware leaves insufficient headroom, fallback options include 2–3 fps (not measured) or a smaller YOLO model.

---

## 6. Go / No-Go Recommendation

**GO — proceed to Phase 1**, with explicit follow-ups before locking gaze policy and FPS defaults.

The core detection pipeline is functional. Phone thresholds are validated on the desktop dataset. Gaze has a clear laptop-driven direction (Profile C / 15° / 4s) that still needs a confirming experiment before runtime defaults change. Spatial person intrusion is being retired as a product path in favor of multi-face gaze. CPU is viable at 5 fps on the profiled desktop but **not yet proven on laptop**, where observed rates are ~4 fps.

**What Phase 1 / next Phase 0 follow-ups should address immediately:**

1. **Gaze policy confirmation (before changing runtime defaults)**  
   Repeat the laptop experiment with exact scenario labels (`natural_writing`, `natural_reading_paper`, sustained ~15s suspicious anchors, plus repeated short side glances). Re-summarize Profile C (`gaze_yaw`, 15°, 4s/6s) and keep Profile D (`eye`) as exploratory. Only then update the runtime attention policy toward Profile C sensitive (4s).

2. **Duration-based streak logic** for all signals (phone, gaze, and future multi-face gaze). Phase 0 emits raw per-frame signals; Phase 1 needs the layer that converts N consecutive frames to a flag.

3. **Pattern detection for burst gaze events** — multiple short glances in a direction that never individually cross the duration threshold.

4. **Stretch / drink water filtering** — Phase 1 flag logic should allow brief off-gaze excursions or require secondary context before promoting a raw gaze signal to a professor-facing flag (especially if desktop-like 5° settings remain in any profile).

5. **Retire spatial person intrusion; design multi-face gaze** — stop investing in YOLO spatial intrusion calibration. Instead, detect multiple faces and estimate whether secondary faces are looking at the examinee’s screen, with flag attribution for the seated student vs. third-party/behind-camera cheating visible in frame.

6. **FPS / performance investigation** — diagnose why the pipeline cannot reach higher targets on desktop (~10 fps ceiling) and why laptop sustained rates are ~4 fps. Re-run formal `--mode both --duration 600` profiling on laptop hardware; identify bottlenecks and evaluate YOLO size / resolution / skip strategies before locking a global FPS default.

**What is NOT blocking Phase 1:**
- Dark water bottle FP — known, documented, not typical exam behavior.
- Brief phone detections — temporal logic is the fix, and that's a Phase 1 build.
- Under-desk long gaze miss on yaw-only — narrow scenario; phone detection and eye/iris context are the complementary signals.
- Spatial intrusion not calibrated — superseded by the multi-face gaze direction rather than treated as a Phase 0 blocker.
