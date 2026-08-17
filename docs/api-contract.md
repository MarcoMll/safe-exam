# Agent ↔ server API contract

This is the HTTP contract between the ExamGuard **client agent** (`src/safe_exam/agent/`) and the **dev server** (`server/`).

Base URL comes from agent config `server_url` (local default: `http://127.0.0.1:8000`).

**Implementation status**

| Endpoint | Server (`server/main.py`) | Agent |
|----------|---------------------------|--------|
| `GET /health` | Done (Phase A) | Used by #39 checklist |
| `GET /auth/check` | Done (Phase A) | Used by #39 checklist |
| `POST /clip/upload` | Done (Phase A) | Done (#38) |
| `POST /session/start` | After #39 (Phase B) | #39 |
| `POST /session/end` | After #39 (Phase B) | #39 |
| `POST /metadata/ingest` | After #39 (Phase B) | #37 (buffer exists; POST not wired) |

Run the server: `uvicorn server.main:app --reload --port 8000` (see README).

---

## Auth

Every endpoint except `GET /health` requires:

```http
Authorization: Bearer {auth_token}
```

`auth_token` is the value from `config/ex.config.yml`. `Bearer` is the HTTP auth **scheme** (not a project nickname). Missing, malformed, or unknown tokens → **401**.

---

## Endpoints

### `GET /health`

Pre-exam checklist: is the server up?

**Auth:** none.

**Response `200`:**

```json
{ "status": "ok" }
```

---

### `GET /auth/check`

Pre-exam checklist: is the Bearer token accepted? Same JSON body as `/health`, but **requires** auth.

**Auth:** Bearer.

**Response `200`:**

```json
{ "status": "ok" }
```

**Response `401`:** missing header, not `Bearer …`, or unknown token.

---

### `POST /session/start`

Register one exam run. Returns a new **session id** (UUID v4).

**Auth:** Bearer.

**Request JSON:**

```json
{
  "exam_id": "EXAM_2026_FINAL",
  "student_id": "S12345"
}
```

**Response `200`:**

```json
{
  "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
}
```

`session_id` is a unique key for this run. Sort/filter sessions by `exam_id`, `student_id`, and `started_at` on the server — not by the UUID itself.

---

### `POST /session/end`

Close a session. Sent during agent shutdown (#39).

**Auth:** Bearer.

**Request JSON:**

```json
{
  "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "duration_seconds": 3600,
  "flag_count": 3,
  "clips_uploaded": 2,
  "clips_pending": 1
}
```

**Response `200`:**

```json
{ "status": "closed" }
```

---

### `POST /metadata/ingest`

Lightweight per-frame heartbeat. Posted about every 5 seconds (#37).

**Auth:** Bearer.

**Request JSON:**

```json
{
  "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "timestamp": 1720000000.0,
  "signals": [
    {
      "timestamp": 1719999995.2,
      "phone_detected": false,
      "phone_confidence": 0.0,
      "person_count": 1,
      "person_boxes": [],
      "frame_width": 640,
      "frame_height": 480,
      "face_detected": true,
      "head_pitch": -5.2,
      "head_yaw": 3.1,
      "eye_pitch": -2.1,
      "eye_yaw": 1.3,
      "gaze_pitch": -4.2,
      "gaze_yaw": 2.5,
      "iris_offset_x": 0.0,
      "iris_offset_y": 0.0,
      "head_direction": "Forward",
      "extra_person_detected": false,
      "gaze_off_seconds": 0.0,
      "fused_score": 0.0
    }
  ]
}
```

Field names:

| Field | Meaning |
|-------|---------|
| Outer `timestamp` | When this **batch** was sent |
| Inner `timestamp` | Unix time of that **frame** (`FrameResult.timestamp` via `as_dict()`) |

Do **not** use `t`. The GitHub #37 example used `t` as shorthand; the agent and this contract use `timestamp`.

Each signal is `ProcessFrameOutput.as_dict()` plus:

- `extra_person_detected` (`person_count > 1`)
- `gaze_off_seconds`
- `fused_score`

**Response `200`:**

```json
{ "received": 12 }
```

`received` is the number of signal objects in the batch. The agent can drop the in-memory buffer after 200.

---

### `POST /clip/upload`

Upload one flagged clip. Implemented on the agent (#38).

**Auth:** Bearer.

**Body:** `multipart/form-data` with exactly these field names:

| Field | Content-Type | File |
|-------|----------------|------|
| `clip` | `video/mp4` | H.264 MP4 |
| `sidecar` | `application/json` | Metadata JSON |

**Sidecar JSON:**

```json
{
  "exam_id": "EXAM_2026_FINAL",
  "student_id": "S12345",
  "timestamp": 1720000000.0,
  "phone_confidence": 0.71,
  "gaze_off_seconds": 4.0,
  "extra_person_detected": false,
  "fused_score": 0.82,
  "reasons": ["phone"]
}
```

`timestamp` here is the **flag** time (same field name as metadata, different event).

**Response `200`:**

```json
{ "status": "stored" }
```

The agent only checks HTTP status `200`. Any other status (or timeout) is a failure: retry with backoff, keep local files.

**Dev storage layout:**

```
server/storage/clips/{exam_id}/{student_id}/{timestamp}.mp4
server/storage/clips/{exam_id}/{student_id}/{timestamp}.json
```

---

## Status codes (agent behavior)

| Code | Meaning | Agent |
|------|---------|--------|
| `200` | Success | Clip: delete files + dequeue. Metadata: clear batch. |
| `401` | Bad or missing Bearer token | Treat as failure / retry |
| `4xx` / `5xx` | Rejected or server error | Treat as failure / retry |
| Timeout / network error | Unreachable | Treat as failure / retry |

---

## Locked decisions

| Topic | Choice |
|-------|--------|
| Auth | `Authorization: Bearer {auth_token}` |
| Clip multipart fields | `clip` + `sidecar` |
| Per-frame / flag time field | `timestamp` (never `t`) |
| Session id | UUID v4 from `POST /session/start` |
| Sort / filter | `exam_id`, `student_id`, `started_at` — not the UUID |
| Dev storage | Files under `server/storage/` |
| Database | After session wiring (#39) |
| Default port | `8000` (`server_url` must match) |
| Server layout | Repo-root `server/`, not `src/safe_exam/` |
