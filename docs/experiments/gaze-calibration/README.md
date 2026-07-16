# Gaze off-screen threshold calibration

Issue [#13](https://github.com/MarcoMll/safe-exam/issues/13). Calibrates **angle + duration** rules for sustained off-gaze. The goal is to choose a good raw attention signal first, then decide later how strict the professor-facing flag policy should be.

Current repo default recommendation: **Profile B** as the raw debugging/testing signal:

- combined `gaze_yaw`
- `yaw_only`
- yaw threshold **5 deg**
- pitch effectively disabled (`99 deg`)
- downstream duration target **4s** once Phase 1 streak logic is added

## How to run

Tooling lives in [`scripts/experiments/gaze_calibration/`](../../../scripts/experiments/gaze_calibration/) (durable experiment package). Findings and CSVs stay in this docs folder.

| Module | Role |
|--------|------|
| `__main__.py` | CLI entrypoint |
| `record.py` | Live capture session (camera) |
| `analyze.py` | `--summarize` and `--backtest` (no camera) |

### Recommended workflow

1. Record one experiment folder with `natural_*` and `suspicious_*` scenarios.
2. Run `--backtest` on the CSV to scan candidate configs.
3. Read the printed **Profile A** and **Profile B** suggestions.
4. Confirm finalists with `--summarize` on the same CSV.
5. Only then update the runtime attention policy defaults in the processor.

### Record behaviors (~45s each)

```bash
cd scripts
python -m experiments.gaze_calibration --experiment <experiment_name>
```


| Control                            | Action                                           |
| ---------------------------------- | ------------------------------------------------ |
| Type scenario name in the terminal | Labels the next capture                          |
| `SPACE` (preview focused)          | Record for `--duration` seconds (default **45**) |
| `N` / `Q`                          | Rename scenario / quit                           |


Results: `docs/experiments/gaze-calibration/results/<experiment>/gaze_calibration.csv`

### Backtest thresholds (no camera — use existing CSV)

Sweep hundreds of angle/duration configs on recorded data:

```bash
cd scripts
python -m experiments.gaze_calibration --backtest
python -m experiments.gaze_calibration --backtest path/to/gaze_calibration.csv
```

Writes `backtest_grid.csv` next to the source CSV and prints **Profile A** (conservative) and **Profile B** (sensitive) at the end.

Use this first. It is the fastest way to compare configs without re-recording.

### Summarize one config

```bash
cd scripts
python -m experiments.gaze_calibration --summarize \
  --mode yaw_only --pitch-threshold 99 --yaw-threshold 8 --gap-tolerance 0.4
```


| Flag                                    | Purpose                                                         |
| --------------------------------------- | --------------------------------------------------------------- |
| `--mode yaw_only`                       | Ignore pitch (writing/reading look down) — side glances are yaw |
| `--pitch-threshold` / `--yaw-threshold` | Angle cutoffs (degrees)                                         |
| `--iris-threshold`                      | For `iris` signal: normalized offset cutoff                     |
| `--gap-tolerance`                       | Tolerate brief face loss (blinks) inside a streak               |




### Naming scenarios


| Prefix         | Meaning                                                                                      |
| -------------- | -------------------------------------------------------------------------------------------- |
| `natural_*`    | Normal exam behavior — should not produce long **suspicious** streaks at the chosen duration |
| `suspicious_*` | Should produce a long streak when the behavior is sustained                                  |


Raw threshold crossings are expected during writing (pitch down). **Flag logic** (Phase 1) will combine signal, duration, and context — calibration documents the numbers.

---



## Experiment 1 — desktop PC camera


| Field         | Value                                                                                                                      |
| ------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Experiment id | `experiment_1_desktop_pc_camera`                                                                                           |
| Scenarios     | 12 × ~45s                                                                                                                  |
| Raw CSV       | [results/experiment_1_desktop_pc_camera/gaze_calibration.csv](results/experiment_1_desktop_pc_camera/gaze_calibration.csv) |
| Backtest grid | [results/experiment_1_desktop_pc_camera/backtest_grid.csv](results/experiment_1_desktop_pc_camera/backtest_grid.csv)       |




### Key finding: symmetric pitch+yaw @ 10° does not work

At pitch=10, yaw=10 (both axes), **reading and writing produce long off-center streaks** because looking at paper is mostly pitch-down. That would flag normal exam behavior.

**Use a yaw-only off-screen rule** for suspicious side-looking (ignore pitch for this policy).

### Backtest scoring (heuristic — not ground truth)

The grid ranks configs with a simple score so we can compare hundreds of combinations. It is a **search aid**, not a measure of real-world accuracy.

For each `(signal, angles, gap_tolerance) × duration`:

```
score = suspicious_tp × 10 + best_suspicious_streak_s
        − natural_fp × 5
        − 8  (if natural_writing fires at this duration)
        − 6  (if natural_reading_paper fires)
```


| Term                       | Meaning                                                                   |
| -------------------------- | ------------------------------------------------------------------------- |
| `suspicious_tp`            | Count of `suspicious_*` scenarios whose longest off-yaw streak ≥ duration |
| `natural_fp`               | Count of `natural_*` scenarios whose longest off-yaw streak ≥ duration    |
| `best_suspicious_streak_s` | Longest streak among suspicious scenarios (tie-break / bonus)             |


**Limitations:** all suspicious scenarios are weighted equally (phone-side and eyes-only count the same). A high score can come from one scenario (`suspicious_eyes_only_side` dominated experiment 1). 

### Two recommended configs (experiment 1)

Phase 0 has two layers: **(1) raw sustained off-gaze signal** and **(2) professor-facing flag policy** (Phase 1). These profiles pick numbers for layer 1. The experiment conclusion is which trade-off you want:


|                       | Profile A — Conservative          | Profile B — Sensitive                                   |
| --------------------- | --------------------------------- | ------------------------------------------------------- |
| **Goal**              | Fewer raw signals; writing-safe   | Catch more suspicious behavior; accept more natural FPs |
| **Phase 1**           | Simpler duration rule may suffice | Needs stricter pattern/context logic to avoid noise     |
| **Signal**            | `head_yaw`                        | `gaze_yaw` (head + eye combined)                        |
| **Mode**              | yaw_only                          | yaw_only                                                |
| **Yaw**               | **5°**                            | **5°**                                                  |
| **Duration**          | **6s**                            | **4s**                                                  |
| **Gap tolerance**     | 0.4s                              | 0.4s                                                    |
| **Backtest score**    | 54.19                             | 49.65                                                   |
| **suspicious_tp**     | 4 / 6                             | 5 / 6                                                   |
| **natural_fp**        | 2 / 6                             | 3 / 6                                                   |
| **writing / reading** | ok / ok                           | ok / ok                                                 |


**Profile A** wins the backtest score while keeping `natural_writing` and `natural_reading_paper` clean. It misses one suspicious scenario in this dataset (same as most yaw-only configs at 6s).

**Profile B** adds one more suspicious hit by lowering duration to 4s and using combined gaze yaw. Expect more false raw signals on stretch/drink/look-around — acceptable if Phase 1 filters them.

**Not chosen:** iris offset @ 0.06 / 4s hits 6/6 suspicious but **fires on writing** — too noisy for a default raw layer.

### Scenario snapshots for the two finalists

These are the important scenario outcomes from experiment 1

#### Profile A snapshot

`head_yaw`, `yaw_only`, yaw `5 deg`, duration `6s`, gap `0.4s`


| Scenario                             | Longest streak | Fires @6s? | Takeaway                      |
| ------------------------------------ | -------------- | ---------- | ----------------------------- |
| `natural_writing`                    | 1.76s          | n          | Good                          |
| `natural_reading_paper`              | 5.94s          | n          | Right below threshold         |
| `natural_stretch`                    | 10.30s         | Y          | False positive                |
| `natural_drink_water`                | 11.89s         | Y          | False positive                |
| `suspicious_phone_side_medium`       | 19.59s         | Y          | Good                          |
| `suspicious_phone_side_long`         | 8.46s          | Y          | Good                          |
| `suspicious_eyes_only_side`          | 24.28s         | Y          | Strongest suspicious scenario |
| `suspicious_phone_under_desk_medium` | 4.94s          | n          | Miss                          |
| `suspicious_phone_under_desk_long`   | 1.09s          | n          | Miss                          |




#### Profile B snapshot

`gaze_yaw`, `yaw_only`, yaw `5 deg`, duration `4s`, gap `0.4s`


| Scenario                             | Longest streak | Fires @4s? | Takeaway                     |
| ------------------------------------ | -------------- | ---------- | ---------------------------- |
| `natural_writing`                    | 1.34s          | n          | Good                         |
| `natural_reading_paper`              | 3.77s          | n          | Good                         |
| `natural_stretch`                    | 8.87s          | Y          | False positive               |
| `natural_drink_water`                | 11.64s         | Y          | False positive               |
| `natural_fidgiting`                  | 4.61s          | Y          | Extra noise vs Profile A     |
| `suspicious_phone_side_medium`       | 8.79s          | Y          | Good                         |
| `suspicious_phone_side_long`         | 5.27s          | Y          | Good                         |
| `suspicious_eyes_only_side`          | 14.73s         | Y          | Good                         |
| `suspicious_phone_under_desk_medium` | 4.69s          | Y          | Better recall than Profile A |
| `suspicious_phone_under_desk_long`   | 1.34s          | n          | Still missed                 |


**Why Profile B is the current default:** it is easier to debug because it fires more often on suspicious behaviors while still keeping `natural_writing` and `natural_reading_paper` clean. That makes it a better raw-signal layer for future flag logic and testing.

### Reproduce the summaries yourself

```bash
cd scripts

# Profile A — conservative (read head row in output)
python -m experiments.gaze_calibration --summarize \
  --mode yaw_only --pitch-threshold 99 --yaw-threshold 5 \
  --gap-tolerance 0.4 --duration-thresholds 4,6,8,12

# Profile B — sensitive (read gaze row in output)
python -m experiments.gaze_calibration --summarize \
  --mode yaw_only --pitch-threshold 99 --yaw-threshold 5 \
  --gap-tolerance 0.4 --duration-thresholds 4,6,8,12
```

`--summarize` prints tables for gaze, eye, head, and iris; use the row that matches each profile.

### Repository default after this experiment

The runtime now keeps two layers separate:

1. `FaceGazeDetector` computes raw outputs: `head_*`, `eye_*`, `gaze_*`, and `iris_offset_*`
2. The processor chooses one **runtime attention policy** that decides which signal currently counts as off-center

Current runtime policy default:

| Runtime policy field | Value | Why |
| -------------------- | ----- | --- |
| `signal` | `gaze` | Combined head + eye is the most useful raw debugging signal |
| `mode` | `yaw_only` | Ignores pitch so writing/reading/typing do not dominate |
| `yaw_threshold_deg` | `5.0` | Profile B sensitivity |
| `pitch_threshold_deg` | `99.0` | Effectively disables pitch for the active policy |

Processor session logs now include `attention_off_center_frames` for the active policy, plus the raw per-signal counters `head_off_center_frames`, `eye_off_center_frames`, and `gaze_off_center_frames`.

### Optional: minimal-noise manual pick (yaw = 8°)

If you want even fewer natural false positives and can accept missing some phone-side cases at 12s, **yaw = 8°** @ 12s (from manual `--summarize`, not top backtest score) keeps writing/reading clean and only flags obvious sustained head turns. Use when professor-facing noise matters more than catching every side glance.

This is also useful as a future **pattern-building primitive**: one long sustained off-screen event can be treated differently from multiple shorter glances in the same direction.

### What fires at yaw=8°, gap=0.4s (manual summarize)


| Scenario                       | gaze @12s | head @12s             | Notes                                             |
| ------------------------------ | --------- | --------------------- | ------------------------------------------------- |
| `natural_writing`              | n         | n                     | Good — writing not flagged                        |
| `natural_reading_paper`        | n         | n                     | Good                                              |
| `natural_thinking_up`          | n         | n                     | Good                                              |
| `natural_stretch`              | n         | n (gaze fires @8s)    | Stretch can look sideways — acceptable raw signal |
| `natural_drink_water`          | n         | n (gaze fires @8–11s) | Drinking involves head movement                   |
| `suspicious_phone_side_medium` | n         | **Y** (19.5s streak)  | Head yaw captures sustained side look             |
| `suspicious_phone_side_long`   | n         | Y @8s                 |                                                   |
| `suspicious_eyes_only_side`    | **Y**     | **Y**                 | Head moved more than eyes in this capture         |




### Acceptance criteria vs experiment 1


| Issue #13 criterion                                 | Status                                                                                                    |
| --------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| Writing should NOT trigger at recommended threshold | **Met** for Profile A and B (`natural_writing` clean)                                                     |
| Phone side ~15s should trigger                      | **Partial** — phone scenarios fire on head/gaze at 4–12s depending on config; no clean 15s anchor capture |
| Document duration + angle thresholds                | **This README** (Profiles A & B + scoring math)                                                           |




### Experiment conclusion

Pick **Profile A** if Phase 0 should emit fewer raw events and we are willing to miss occasional side glances until Phase 1 adds richer logic.

Pick **Profile B** if Phase 0 should be a **sensitive raw sensor** and Phase 1 will apply stricter flag rules (pattern, context, cooldown) to suppress stretch/drink noise.

Recommendation for this repository: **Profile B for now**. It is better for debugging, demos, and future test development because suspicious scenarios are easier to trigger on demand.

### Known limitations

1. **Eye-only cheating** did not show strongly on the `eye` signal in experiment 1 — likely performance (head moved) and/or eye angle scaling. More `suspicious_eyes` related captures needed.
2. **Stretch / drink / look-around** can produce yaw streaks — future flag logic should allow brief excursions or use secondary context.
3. **Blink dropouts** — use gap tolerance in duration logic (Phase 1).
4. **Burst patterns** (3 quick glances) not evaluated — duration-only rule is a Phase 0 baseline. A future direction is time-series / repeated-direction pattern logic, e.g. multiple short rightward glances that never individually cross the long-duration threshold.

---



## Adding another experiment

Same setup, new folder:

```bash
cd scripts
python -m experiments.gaze_calibration --experiment experiment_2_laptop_webcam
python -m experiments.gaze_calibration --backtest ../docs/experiments/gaze-calibration/results/experiment_2_laptop_webcam/gaze_calibration.csv
```

Then:

1. Add a short experiment block with camera/device context.
2. Paste the final Profile A and Profile B comparison table.
3. State which profile you recommend and why.
4. Only update the runtime attention policy after backtest and summarize agree.

