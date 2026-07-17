# Person intrusion policy calibration

Issue [#14](https://github.com/MarcoMll/safe-exam/issues/14). Calibrates the **spatial intrusion policy** — not raw person detection.

## What we are testing

YOLO already reports `person_count`. In an exam hall, that count will often be greater than one because classmates appear in the background.

The calibration question is:

> When multiple people are visible, does the policy flag **intrusion** (someone close enough / overlapping enough to view this screen) while ignoring normal background classmates?

| Scenario prefix | Person count may be | Intrusion should be |
|-----------------|---------------------|---------------------|
| `solo_*` | 1 | **False** |
| `background_*` | 2+ | **False** |
| `intrusion_*` | 2+ | **True** |

**Do not** score this experiment on `person_count > 1`. A `background_*` scenario with two visible people is a **pass** when intrusion stays false.

## Current default policy (pre-calibration)

From [`intrusion_policy.py`](../../../src/safe_exam/processor/intrusion_policy.py):

| Parameter | Default |
|-----------|---------|
| `roi_center_fraction` | 0.60 |
| `min_secondary_area_pct` | 0.05 |
| `primary_overlap_iou` | 0.10 |
| `min_rules_to_match` | 2 |

These remain **unchanged** after experiment 1 pilot — see findings below.

## How to run

Tooling lives in [`scripts/experiments/person_intrusion/`](../../../scripts/experiments/person_intrusion/).

```bash
cd scripts
python -m experiments.person_intrusion --experiment <experiment_name>
python -m experiments.person_intrusion --summarize
python -m experiments.person_intrusion --backtest
```

| Control | Action |
|---------|--------|
| Type scenario name in the terminal | Labels the next capture |
| `SPACE` (preview focused) | Record ~30 frames (~2.5s at 12 FPS) |
| `N` | Rename / start a new scenario |
| `Q` | Quit and print the session table |

Results append to:

```
docs/experiments/person-intrusion/results/<experiment>/person_intrusion.csv
```

### Recording requirements (important)

For results to be comparable across sessions:

- **Laptop fixed on a desk** — normal exam posture. Do not hold the laptop while recording; that changes framing, bbox sizes, and invalidates comparison with a seated exam.
- **Same camera** — built-in webcam, same resolution if possible.
- **Stable scene** — hold each scenario steady for the full capture window.

Simulated background (person on a monitor/photo) is useful for a **pilot**, but is not a substitute for a classroom session with real seated people.

### Recommended scenarios

**Control (`solo_*`)**
- `solo_normal_exam`

**Should NOT flag (`background_*`)**
- `background_classmate_behind_1_5m`
- `background_multiple_students`
- `background_person_on_monitor` (pilot only)

**SHOULD flag (`intrusion_*`)**
- `intrusion_lean_from_left` — neighbor seated, leans toward screen (needs a real person)
- `intrusion_lean_from_right`
- `intrusion_person_on_monitor` (pilot only — usually too small to trigger; see experiment 1)

## Acceptance targets

| Metric | Target |
|--------|--------|
| FP on `solo_*` + `background_*` | &lt; 5% intrusion rate |
| TP on `intrusion_*` | &gt; 80% intrusion rate |

## Experiment 1 — pilot (desktop PC camera)

| Field | Value |
|-------|-------|
| Experiment id | `experiment_1_desktop_pc_camera` |
| Date | 2026-07-17 |
| Setup | **Pilot — laptop held by hand**, not fixed on desk. Simulated second person via another laptop screen. |
| Resolution | 640×480 |
| Frames per scenario | 30 |
| Raw data | [results/experiment_1_desktop_pc_camera/person_intrusion.csv](results/experiment_1_desktop_pc_camera/person_intrusion.csv) |
| Policy decision | **No change** — data is not reliable enough to tune thresholds |

### Scenarios recorded

| Scenario | Mean persons | Intrusion @ default | Notes |
|----------|--------------|---------------------|-------|
| `solo_normal_exam` | 1.00 | 0% | Valid control. Primary bbox ~47% of frame. |
| `background_one_person_normal` | 1.00 | 0% | YOLO never saw a second person — background simulation did not register. |
| `intrusion_person_looking_over_laptop` | 2.00 | 0% | Second person on screen **was** detected every frame, but bbox too small for policy. |
| `intrusion_person_peaking` | 1.07 | 0% | Second person rarely detected; tiny edge bbox when present. |

### What we can still take from this pilot

1. **Pipeline works** — CSV logging, bbox JSON, feature columns, and summarize tooling all behaved as expected.

2. **Zero false positives** on every scenario — with default policy, nothing flagged. That is directionally good for solo, but **not proof** the policy handles real hall background (we never got `person_count > 1` in a background scenario).

3. **Why `intrusion_person_looking_over_laptop` did not hit policy** — secondary bbox was only ~**1.6%** of frame area (`max_secondary_area_pct ≈ 0.016`), below the `min_secondary_area_pct = 0.05` threshold. IoU with primary was ~**0.04**, below `0.10`. Only the center-ROI rule passed — policy requires **2 of 3** rules. A person on a screen reads as “small + distant,” which is exactly what the policy is designed to ignore.

4. **Monitor simulation ≠ spatial lean-in** — even when labeled `intrusion_*`, a face on another laptop is not the same geometry as a neighbor leaning into frame. Do not use this scenario to tune TP thresholds.

5. **Holding the laptop invalidated the session** — exam calibration must be re-recorded with the machine on a desk.

### What experiment 1 did *not* establish

- Whether real background classmates at 1.5m+ stay below intrusion thresholds
- Whether a seated lean-in from an adjacent desk triggers intrusion
- Optimal Profile A / Profile B policy values (`--backtest` skipped — no trustworthy positive/negative mix)

### Next recording session (when a helper or hall is available)

1. Fix laptop on desk; re-record `solo_normal_exam`.
2. **Background:** real person seated behind/at angle, or multiple people in frame — goal is `person_count ≥ 2` with intrusion **false**.
3. **Intrusion:** helper seated next desk, lean shoulder/head toward your screen without standing — goal is intrusion **true** on most frames.
4. Run `--summarize` and `--backtest`, document Profile A/B here, then update `DEFAULT_INTRUSION_POLICY` if defaults change.

```bash
cd scripts
python -m experiments.person_intrusion --experiment experiment_2_classroom_or_helper
python -m experiments.person_intrusion --summarize
python -m experiments.person_intrusion --backtest
```

## Known limitations (Phase 0)

Spatial intrusion from the victim's webcam only fires when a second person enters **your camera's close/overlap zone**. It does **not** detect:

- Adjacent-seat glances (eyes only, no entry into your frame)
- Someone leaning back in their chair to see your screen
- Screen-viewing when the neighbor never appears as a large/overlapping bbox

See issue #14 for scope. Secondary-face gaze and cross-laptop fusion are future work.

## Out of scope (Phase 0)

- Secondary-face gaze fusion ("looking at my screen")
- Duration streaks → professor-facing flags (Phase 1)
- Cross-laptop identity matching
