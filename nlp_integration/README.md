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
│   ├── interpret.py              # Phase I: NLP interpretation
│   ├── executor.py               # Phase IV: task execution
│   ├── perception.py             # Phase III: perceptual grounding
│   ├── spot_skills.py            # SPOT SDK skill primitives
│   ├── evaluate_phase1.py        # Phase I evaluator
│   ├── evaluate_phase2.py        # Phase II evaluator
│   ├── evaluate_phase3.py        # Phase III evaluator
│   ├── evaluate_phase4.py        # Phase IV evaluator
│   ├── models/                   # Local model files (not tracked by git)
│   │   └── bert-spot-intent/     # Fine-tuned BERT intent classifier (from Anvil)
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

### Intent classifier model

The Phase I intent classifier uses a fine-tuned `bert-base-uncased` model trained on 1,863 domain-specific robot command examples. The model is stored locally and loaded at runtime with no internet connection required.

**The model files must be present at:** `nlp_integration/models/bert-spot-intent/`

The directory should contain: `config.json`, `model.safetensors`, `tokenizer.json`, `vocab.txt`, `tokenizer_config.json`, `special_tokens_map.json`, `label_map.json`.

If you need to retrain the model (e.g. adding new intent classes), see `generate_training_data.py` and `finetune_bert.py` — the training pipeline targets Purdue Anvil GPU nodes via Slurm but can run on any CUDA-capable machine.

---

## Quick Start — run.py

`run.py` is the main entry point. Run it from the `nlp_integration/` directory.

### Mock mode (no robot required)

```powershell
cd nlp_integration
python run.py
```

SPOT is not required. All skills execute in mock/dry-run mode and print what they would do. Use this for development and testing.

```
SPOT> bring me the pen
SPOT> go to the desk and bring me the notebook
SPOT> bring me something to write with
SPOT> hand me something sharp
SPOT> scan the room
SPOT> find my backpack
SPOT> exit
```

### Single command mode

```powershell
python run.py "go to the desk and bring me the notebook"
```

Executes one command and exits. Works in both mock and live mode.

---

## Live SPOT — Setup Checklist

Complete these steps **once** before your first live session.

### 1. Record the GraphNav map

Power on SPOT, stand it at your starting position, then run:

```powershell
cd nlp_integration

$env:SPOT_IP   = "192.168.80.3"
$env:SPOT_USER = "user"
$env:SPOT_PASS = "password"

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

The script prints a `WAYPOINT_MAP` dict and saves it to `maps/trial_space/waypoint_map.txt`. Copy the dict into `spot_skills.py` and note the UUID for the `user` waypoint — you will need it as `SPOT_START_WAYPOINT`.

### 2. Update spot_skills.py

Open `spot_skills.py` and replace the `WAYPOINT_MAP` placeholder:

```python
WAYPOINT_MAP: dict = {
    "desk":    "paste-uuid-here",
    "table":   "paste-uuid-here",
    "kitchen": "paste-uuid-here",
    "user":    "paste-uuid-here",
}
```

### 3. Stage objects for trials

Place objects in view from the appropriate waypoints:

| Location | Objects |
|---|---|
| Visible from start | pen, scissors, laptop, bottle, backpack |
| Desk waypoint | notebook, pen, stapler |
| Table waypoint | cup, charger |

---

## Live SPOT — Running run.py

Set environment variables, then launch:

```powershell
cd nlp_integration

$env:USE_SPOT            = "true"
$env:SPOT_IP             = "192.168.80.3"
$env:SPOT_USER           = "user"
$env:SPOT_PASS           = "password"
$env:SPOT_MAP_PATH       = "maps/trial_space"
$env:SPOT_START_WAYPOINT = "paste-user-waypoint-uuid-here"

python run.py
```

On connect, the framework will:
1. Power on and stand SPOT
2. Clear any behavior faults from previous sessions
3. Upload the GraphNav map and set localization
4. Capture a debug camera snapshot (`debug_camera.jpg`) — open this to verify YOLO is detecting objects before issuing commands

### With live camera feed window

```powershell
python run.py --live-feed
```

Opens a cv2 window showing the front camera with YOLO bounding boxes (green). Press `q` in the window to close the feed. The REPL continues running while the feed is open.

```
SPOT> feed on       # start feed mid-session
SPOT> feed off      # stop feed
SPOT> status        # show connection status and feed state
```

---

## Live SPOT — With Network Compute Server (Custom Models)

Use this when you want to run a custom-trained YOLO model from `object_detection/models/` for detection instead of the built-in `yolov8n.pt`.

### Step 1 — Start the detection server (Terminal 1)

```powershell
cd nlp_integration

python ..\object_detection\network_compute_server.py `
    -m ..\object_detection\models\<your_model>\<your_model>.pt `
    192.168.80.3
```

The server registers itself with SPOT's directory service. Leave this terminal running.

### Step 2 — Run with NCB enabled (Terminal 2)

```powershell
cd nlp_integration

$env:USE_SPOT            = "true"
$env:SPOT_IP             = "192.168.80.3"
$env:SPOT_USER           = "user"
$env:SPOT_PASS           = "password"
$env:SPOT_MAP_PATH       = "maps/trial_space"
$env:SPOT_START_WAYPOINT = "paste-user-waypoint-uuid-here"
$env:USE_COMPUTE_SERVER  = "true"
$env:NCB_SERVER_NAME     = "fetch-server"
$env:NCB_MODEL_NAME      = "<your_model_name>"
$env:NCB_CONFIDENCE      = "0.5"

python run.py --live-feed --use-compute-server
```

When `USE_COMPUTE_SERVER=true`, `locate()` and `pick_up()` route through the NCB server across all five fisheye cameras. If the server is unreachable, the framework falls back to local YOLO automatically.

The live feed window uses **orange boxes** for NCB detections and **green boxes** for local YOLO.

---

## Running the 20-Trial Live Evaluation

Used for the thesis Phase IV evaluation. Runs all 20 commands with operator prompts between each trial.

```powershell
# Set all env vars as above, then:

# No feed
python live_trials.py

# With local YOLO feed window
python live_trials.py --live-feed

# With NCB server feed (start network_compute_server.py first)
python live_trials.py --live-feed --use-compute-server
```

Results are saved to `data/live_trials.jsonl` and `data/live_trials_<run_id>.json`.

At each prompt, type `y` to proceed or `n` to save and exit. After each trial you can type an operator note (or press Enter to skip).

The live feed runs in the background — operator prompts and trial execution are unaffected while the feed window is open. Press `q` in the feed window to close just the window; trials continue. Ctrl+C and `n` at any prompt both stop the feed cleanly before saving.

### live_trials.py flags

| Flag | Description |
|---|---|
| `--live-feed` | Show live camera feed during trials |
| `--use-compute-server` | Route feed detections through NCB server |
| `--server <n>` | NCB server name (default: `fetch-server`) |
| `--model <n>` | NCB model name (default: `yolov8n`) |
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
| II | Per-Clause Intent Acc. | 96.4% |
| II | Step-Sequence F1 | 0.863 |
| II | Edge-Type Accuracy | 100% |
| III | Top-1 Grounding Acc. | 85.0% |
| III | Mean Reciprocal Rank | 0.889 |
| IV | Task Completion Rate | 95.0% |
| IV | Plan / Object / Waypoint Acc. | 100% |

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
| `--live-feed` | Open camera feed window on connect |
| `--use-compute-server` | Use NCB server for feed detections |
| `--server <name>` | NCB server name (default: `fetch-server`) |
| `--model <name>` | NCB model name (default: `yolov8n`) |
| `--camera <source>` | Camera source for feed (default: `frontleft_fisheye_image`) |

---

## Notes

- Run all commands from the `nlp_integration/` directory.
- `debug_camera.jpg` is saved to `nlp_integration/` on every live connect. Open it to verify camera and YOLO are working before starting trials.
- If SPOT has behavior faults from a previous session (e.g. it fell), `connect()` clears them automatically.
- If localization fails (no AprilTag visible), set `SPOT_START_WAYPOINT` to the UUID of the waypoint where SPOT is physically standing.
- The `user` waypoint must be stamped during map recording — it is the delivery target for all retrieve commands. Always stamp it at your standing position.