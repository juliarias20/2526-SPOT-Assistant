# 2526-SPOT-Assistant

A context-aware natural language command interpretation framework for autonomous robotic task execution on Boston Dynamics SPOT. Interprets free-form voice or text commands, decomposes them into structured task plans, grounds objects via affordance reasoning, and executes them on the physical robot.

**Thesis:** *A Context-Aware Natural Language Command Interpretation Framework for Autonomous Robotic Task Execution* — Jonathan Francisco, Cal Poly Pomona, CS 6640, Spring 2026.

---

## Project Structure

```
2526-SPOT-Assistant/
├── nlp_integration/              # Main framework (all commands run from here)
│   ├── run.py                    # ← Interactive task runner (start here)
│   ├── live_trials.py            # 20-trial live evaluation runner
│   ├── record_map.py             # GraphNav map recording utility
│   ├── cache_models.py           # One-time model download script (run before going offline)
│   ├── interpret.py              # Phase I: NLP interpretation
│   ├── executor.py               # Phase IV: task execution
│   ├── perception.py             # Phase III: perceptual grounding
│   ├── spot_skills.py            # SPOT SDK skill primitives
│   ├── evaluate_phase1.py        # Phase I evaluator
│   ├── evaluate_phase2.py        # Phase II evaluator
│   ├── evaluate_phase3.py        # Phase III evaluator
│   ├── evaluate_phase4.py        # Phase IV evaluator
│   ├── models/                   # Local model files (not tracked by git)
│   │   ├── bert-spot-intent/     # Fine-tuned BERT intent classifier (from Anvil)
│   │   └── sentence_transformers/# Cached SentenceTransformer weights (from cache_models.py)
│   ├── data/                     # Gold datasets + trial logs
│   └── maps/                     # GraphNav map files (after record_map.py)
└── object_detection/             # External YOLO detection server (optional)
    ├── network_compute_server.py # NCB server — serves custom YOLO models to SPOT
    ├── fetch.py                  # Reference grasp script (Boston Dynamics tutorial)
    ├── live_cam_detection.py     # Live camera feed with bounding boxes
    ├── YOLO_SPOT.ipynb           # Google Colab notebook to train custom models
    ├── models/                   # Custom trained .pt model files
    └── requirements.txt
```

---

## Requirements

- Python 3.10.11
- Windows 11 (tested environment)
- Boston Dynamics SPOT SDK `bosdyn-client==5.1.4`

### Install dependencies

```powershell
cd nlp_integration
pip install -r requirements.txt
```

For GPU / CUDA support with PyTorch, select your version at https://pytorch.org/get-started/locally/ before installing.

---

## First-Time Setup

Complete these steps once before your first session.

### 1. Download and cache all model weights

Run this once while connected to the internet. After this the system runs fully offline.

```powershell
cd nlp_integration
python cache_models.py
```

This downloads and caches:
- `sentence-transformers/all-MiniLM-L6-v2` → `models/sentence_transformers/`
- `spaCy en_core_web_sm` (if not already installed)
- Verifies `models/bert-spot-intent/` weights are present
- Verifies `yolov8n.pt` is present (downloads if missing)

If the BERT model is missing, run `finetune_bert.py` on Purdue Anvil first (see training section below).

### 2. Record the GraphNav map (live SPOT only)

Power on SPOT, stand it at your starting position, then run:

```powershell
cd nlp_integration

$env:SPOT_IP   = "192.168.80.3"
$env:SPOT_USER = "user"
$env:SPOT_PASS = "yourpassword"

python -X utf8 record_map.py --output maps/trial_space
```

Walk SPOT to each location and type a name when prompted:

```
desk        ← where objects are staged
table       ← second object staging area
kitchen     ← navigation target
user        ← IMPORTANT: stamp this at your standing position (delivery target)
done        ← when finished
```

The script prints a `WAYPOINT_MAP` dict and saves it to `maps/trial_space/waypoint_map.txt`.

### 3. Update WAYPOINT_MAP in spot_skills.py

Open `spot_skills.py` and replace the `WAYPOINT_MAP` placeholder with your recorded UUIDs:

```python
WAYPOINT_MAP: dict = {
    "desk":    "paste-uuid-here",
    "table":   "paste-uuid-here",
    "kitchen": "paste-uuid-here",
    "user":    "paste-uuid-here",
}
```

**Note:** `LOCATION_NOUNS` in `executor.py` is automatically derived from `WAYPOINT_MAP` keys, so you do not need to update it separately. Waypoint name matching also handles partial matches — `"kit"` will resolve to `"kitchen"`, `"work"` will resolve to `"workspace"`, etc.

### 4. Set SPOT_START_WAYPOINT

Note the UUID printed for the `user` waypoint. You will use it as `SPOT_START_WAYPOINT` when running live. This is the fallback localization position when no AprilTag is visible.

### 5. Stage objects for trials

Place objects in view from the appropriate waypoints:

| Location | Objects |
|---|---|
| Visible from start | pen, scissors, laptop, bottle, backpack |
| Desk waypoint | notebook, pen, stapler |
| Table waypoint | cup, charger |

---

## Running in Mock Mode (No Robot Required)

```powershell
cd nlp_integration
python run.py
```

All skills execute in dry-run mode and print what they would do. No SPOT connection needed. Use this for development and testing.

```
SPOT> bring me the pen
SPOT> go to the desk and bring me the notebook
SPOT> bring me something to write with
SPOT> hand me something sharp
SPOT> scan the room
SPOT> find my backpack
SPOT> exit
```

### Single command

```powershell
python run.py "go to the desk and bring me the notebook"
```

---

## Running Live on SPOT

### Standard (local YOLO detection)

```powershell
cd nlp_integration

$env:USE_SPOT            = "true"
$env:SPOT_IP             = "192.168.80.3"
$env:SPOT_USER           = "user"
$env:SPOT_PASS           = "yourpassword"
$env:SPOT_MAP_PATH       = "maps/trial_space"
$env:SPOT_START_WAYPOINT = "paste-user-waypoint-uuid-here"

python run.py --offline
```

On connect, the framework will:
1. Power on and stand SPOT
2. Clear any behavior faults from a previous session
3. Upload the GraphNav map and set localization
4. Save a debug camera snapshot to `debug_camera.jpg` — open this to verify YOLO is detecting objects before issuing commands

### With live camera feed window

```powershell
python run.py --offline --live-feed
```

Opens a cv2 window showing the front camera with YOLO bounding boxes (green). Press `q` in the window to close it — the REPL continues running.

```
SPOT> feed on       # start feed mid-session
SPOT> feed off      # stop feed
SPOT> status        # show connection and feed status
```

### With Network Compute Server (custom YOLO model)

Use this to run a custom-trained model from `object_detection/models/` for detection.

**Terminal 1 — start the detection server:**

```powershell
cd nlp_integration

python ..\object_detection\network_compute_server.py `
    -m ..\object_detection\models\<your_model>\<your_model>.pt `
    --username user --password yourpassword `
    192.168.80.3
```

Leave this terminal running.

**Terminal 2 — run with NCB enabled:**

```powershell
cd nlp_integration

$env:USE_SPOT            = "true"
$env:SPOT_IP             = "192.168.80.3"
$env:SPOT_USER           = "user"
$env:SPOT_PASS           = "yourpassword"
$env:SPOT_MAP_PATH       = "maps/trial_space"
$env:SPOT_START_WAYPOINT = "paste-user-waypoint-uuid-here"
$env:USE_COMPUTE_SERVER  = "true"
$env:NCB_MODEL_NAME      = "your_model_name"

python run.py --offline --live-feed --use-compute-server
```

When `USE_COMPUTE_SERVER=true`:
- `locate()` and `pick_up()` query the NCB server across all five fisheye cameras
- If the server is unreachable, the framework automatically falls back to local YOLO
- If the object is not immediately visible, SPOT performs up to 4 scan rotations (one full 360°) before reporting failure
- The live feed shows **orange boxes** for NCB detections and **green boxes** for local YOLO

---

## Running the 20-Trial Live Evaluation

Runs all 20 thesis evaluation commands with operator prompts between each trial.

```powershell
# Set all env vars as above, then:

# No feed
python live_trials.py --offline

# With local YOLO feed
python live_trials.py --offline --live-feed

# With NCB server feed (start network_compute_server.py first)
python live_trials.py --offline --live-feed --use-compute-server
```

Results are saved to `data/live_trials.jsonl` and `data/live_trials_<run_id>.json`.

At each prompt, type `y` to proceed or `n` to save and exit. After each trial you can type an operator note or press Enter to skip.

### live_trials.py flags

| Flag | Description |
|---|---|
| `--offline` | Run fully offline using cached models |
| `--live-feed` | Show live camera feed during trials |
| `--use-compute-server` | Route feed detections through NCB server |
| `--server <name>` | NCB server name (default: `fetch-server`) |
| `--model <name>` | NCB model name (default: `yolov8n`) |
| `--camera <source>` | Camera source for feed (default: `frontleft_fisheye_image`) |

---

## Running the Phase Evaluators (Dry-Run)

No SPOT required for any of these.

```powershell
cd nlp_integration

python evaluate_phase1.py    # intent accuracy + clarification F1
python evaluate_phase2.py    # clause splitting + task graph
python evaluate_phase3.py    # affordance grounding accuracy
python evaluate_phase4.py    # end-to-end task execution (mock)
```

Expected results:

| Phase | Metric | Score |
|---|---|---|
| I | Intent Accuracy | 0.840 |
| I | Clarification F1 | 0.824 |
| II | Clause Count Accuracy | 100% |
| II | Per-Clause Intent Accuracy | 96.4% |
| II | Step-Sequence F1 | 0.863 |
| II | Edge-Type Accuracy | 100% |
| III | Top-1 Grounding Accuracy | 85.0% |
| III | Mean Reciprocal Rank | 0.889 |
| IV | Task Completion Rate | 95.0% |
| IV | Plan / Object / Waypoint Accuracy | 100% |

---

## Retraining the BERT Classifier

If you need to add new intent classes or retrain from scratch:

```powershell
# Generate training data
python generate_training_data.py

# Submit to Purdue Anvil (requires ACCESS-CI allocation)
# Edit submit_finetune.sh with your Anvil username first
bash submit_finetune.sh

# Or run locally on a CUDA-capable machine
python finetune_bert.py
```

After training, copy the output directory to `models/bert-spot-intent/`. Then re-run `cache_models.py` to verify the new weights load correctly.

---

## Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `USE_SPOT` | `false` | Set `true` to connect to physical SPOT |
| `SPOT_IP` | `192.168.80.3` | SPOT's IP address |
| `SPOT_USER` | `user` | SPOT login username |
| `SPOT_PASS` | *(none)* | SPOT login password |
| `SPOT_MAP_PATH` | `maps/trial_space` | Path to recorded GraphNav map directory |
| `SPOT_START_WAYPOINT` | *(empty)* | UUID of start waypoint for no-fiducial localization |
| `OFFLINE_MODE` | `false` | Set `true` to force offline model loading (same as `--offline` flag) |
| `CAMERA_SOURCE` | `frontleft_fisheye_image` | Camera used for local YOLO detection |
| `USE_COMPUTE_SERVER` | `false` | Route locate/pick_up through NCB server |
| `NCB_SERVER_NAME` | `fetch-server` | NCB server service name |
| `NCB_MODEL_NAME` | `yolov8n` | Model name registered on the NCB server |
| `NCB_CONFIDENCE` | `0.5` | Minimum detection confidence for NCB |
| `YOLO_MODEL` | `yolov8n.pt` | Local YOLO model file path |

---

## run.py Commands Reference

| Command | Description |
|---|---|
| Any natural language | Interpreted and executed |
| `help` | Show example commands |
| `status` | Robot connection state and feed status |
| `feed on` | Start live camera feed (live mode only) |
| `feed off` | Stop live camera feed |
| `exit` / `quit` / `q` | Disconnect and exit |

### run.py flags

| Flag | Description |
|---|---|
| `--offline` | Run fully offline using cached models — recommended for all lab sessions |
| `--live-feed` | Open camera feed window on connect |
| `--use-compute-server` | Use NCB server for feed and task detections |
| `--server <name>` | NCB server name (default: `fetch-server`) |
| `--model <name>` | NCB model name (default: `yolov8n`) |
| `--camera <source>` | Camera source for feed (default: `frontleft_fisheye_image`) |

---

## Troubleshooting

**Localization failed (STATUS_NO_MATCHING_FIDUCIAL)**
Place an AprilTag in SPOT's view before connecting, or set `SPOT_START_WAYPOINT` to the UUID of the waypoint where SPOT is physically standing and make sure SPOT is at that position before connecting.

**Navigate command doesn't resolve a location**
Ensure the location name is in `WAYPOINT_MAP` in `spot_skills.py`. The resolver handles partial matches (`"kit"` → `"kitchen"`) but the base name must exist as a key. Check the `[spot] WARNING:` line in the output to see exactly what name was passed.

**Gray box in live feed / no detections**
The NCB server is not returning image data (this is expected — it only returns bounding boxes). The feed fetches the frame directly from SPOT's ImageClient. If the window is gray, verify the `--use-compute-server` flag is set correctly (note: `--use-commpute-server` with a double `m` is silently ignored).

**Object not found during fetch task**
SPOT will automatically rotate up to 4 times (one full 360°) scanning for the object before reporting failure. If it still fails, verify the object is within SPOT's camera range and the model confidence threshold (`NCB_CONFIDENCE`) is not set too high.

**HuggingFace authentication warnings at startup**
These are cosmetic — the system is loading from local cache and not downloading anything. Run with `--offline` to suppress them entirely.

**Behavior faults on connect**
SPOT clears behavior faults automatically on connect. If SPOT falls or has a fault between sessions, the `connect()` call handles it. If it persists, power cycle SPOT and reconnect.