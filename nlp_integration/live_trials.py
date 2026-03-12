"""
live_trials.py
--------------
Phase IV: Live SPOT Trial Runner

Runs the 20-command live trial set against a physical SPOT robot.
Before each trial, prompts the operator to confirm readiness (objects staged,
robot in position). Typing 'n' or 'no' at any prompt exits immediately and
saves all completed trials to disk.

Extra fields logged vs. dry-run (evaluate_phase4.py):
    mode                -- always "live"
    detection_source    -- "spot" | "mock" (from perception module)
    detection_conf      -- YOLO confidence of the resolved object (if available)
    grasp_attempts      -- number of pick_up attempts before success or failure
    localization_ok     -- whether GraphNav reported localization at trial start
    operator_notes      -- free-text note entered by operator after each trial

Usage:
    set USE_SPOT=true
    set SPOT_IP=192.168.80.3
    set SPOT_USER=user
    set SPOT_PASS=password
    set SPOT_MAP_PATH=maps/trial_space
    python live_trials.py

Output:
    data/live_trials.jsonl      -- per-trial log (appends on re-run)
    data/live_trials_<run>.json -- full run summary with all records + metrics
"""

import json
import os
import signal
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from interpret import Phase1Interpreter
from executor import TaskExecutor, ExecutionResult
from spot_skills import SpotRobot

# ── Paths ──────────────────────────────────────────────────────────────────────
LOG_FILE = Path("data/live_trials.jsonl")

# ── Trial set ─────────────────────────────────────────────────────────────────
# 20 trials across 7 categories, mirroring the dry-run set for direct comparison.
# object_at: which waypoint the object should be staged at for multi-step trials.
LIVE_TRIALS = [
    # ── retrieve_object (5) ───────────────────────────────────────────────────
    {
        "id": "L01", "category": "retrieve_object",
        "command": "Bring me the pen",
        "expected_plan": ["locate", "pick_up", "deliver", "release"],
        "expected_object": "pen", "expected_waypoint": None,
        "expected_success": True,
        "setup_note": "Pen visible from start position.",
    },
    {
        "id": "L02", "category": "retrieve_object",
        "command": "Hand me the scissors",
        "expected_plan": ["locate", "pick_up", "deliver", "release"],
        "expected_object": "scissors", "expected_waypoint": None,
        "expected_success": True,
        "setup_note": "Scissors visible from start position.",
    },
    {
        "id": "L03", "category": "retrieve_object",
        "command": "Get me the laptop",
        "expected_plan": ["locate", "pick_up", "deliver", "release"],
        "expected_object": "laptop", "expected_waypoint": None,
        "expected_success": True,
        "setup_note": "Laptop open/flat on surface, visible from start position.",
    },
    {
        "id": "L04", "category": "retrieve_object",
        "command": "Give me the notebook",
        "expected_plan": ["locate", "pick_up", "deliver", "release"],
        "expected_object": "notebook", "expected_waypoint": None,
        "expected_success": True,
        "setup_note": "Notebook flat on surface, visible from start position.",
    },
    {
        "id": "L05", "category": "retrieve_object",
        "command": "Bring me the bottle",
        "expected_plan": ["locate", "pick_up", "deliver", "release"],
        "expected_object": "bottle", "expected_waypoint": None,
        "expected_success": True,
        "setup_note": "Bottle upright, visible from start position.",
    },
    # ── multi_step_retrieve (4) ───────────────────────────────────────────────
    {
        "id": "L06", "category": "multi_step_retrieve",
        "command": "Go to the desk and bring me the notebook",
        "expected_plan": ["navigate", "locate", "pick_up", "deliver", "release"],
        "expected_object": "notebook", "expected_waypoint": "desk",
        "expected_success": True,
        "setup_note": "Notebook staged at desk waypoint.",
    },
    {
        "id": "L07", "category": "multi_step_retrieve",
        "command": "Head to the table and grab me a cup",
        "expected_plan": ["navigate", "locate", "pick_up", "deliver", "release"],
        "expected_object": "cup", "expected_waypoint": "table",
        "expected_success": True,
        "setup_note": "Cup staged at table waypoint.",
    },
    {
        "id": "L08", "category": "multi_step_retrieve",
        "command": "Go to the table and fetch my charger",
        "expected_plan": ["navigate", "locate", "pick_up", "deliver", "release"],
        "expected_object": "charger", "expected_waypoint": "table",
        "expected_success": True,
        "setup_note": "Charger staged at table waypoint alongside cup.",
    },
    {
        "id": "L09", "category": "multi_step_retrieve",
        "command": "Go to the desk and bring me the stapler",
        "expected_plan": ["navigate", "locate", "pick_up", "deliver", "release"],
        "expected_object": "stapler", "expected_waypoint": "desk",
        "expected_success": True,
        "setup_note": "Stapler staged at desk waypoint.",
    },
    # ── vague_retrieve (4) ───────────────────────────────────────────────────
    {
        "id": "L10", "category": "vague_retrieve",
        "command": "Bring me something to write with",
        "expected_plan": ["locate", "pick_up", "deliver", "release"],
        "expected_object": "pen", "expected_waypoint": None,
        "expected_success": True,
        "setup_note": "Pen visible. Affordance: write → pen.",
    },
    {
        "id": "L11", "category": "vague_retrieve",
        "command": "Get me something to drink",
        "expected_plan": ["locate", "pick_up", "deliver", "release"],
        "expected_object": "bottle", "expected_waypoint": None,
        "expected_success": True,
        "setup_note": "Bottle visible. Affordance: drink → bottle.",
    },
    {
        "id": "L12", "category": "vague_retrieve",
        "command": "Hand me something sharp",
        "expected_plan": ["locate", "pick_up", "deliver", "release"],
        "expected_object": "scissors", "expected_waypoint": None,
        "expected_success": True,
        "setup_note": "Scissors visible. Affordance: sharp → scissors.",
    },
    {
        "id": "L13", "category": "vague_retrieve",
        "command": "Bring me something to carry my things in",
        "expected_plan": ["locate", "pick_up", "deliver", "release"],
        "expected_object": "backpack", "expected_waypoint": None,
        "expected_success": True,
        "setup_note": "Backpack visible. Affordance: carry → backpack. Harder phrase.",
    },
    # ── locate_object (2) ────────────────────────────────────────────────────
    {
        "id": "L14", "category": "locate_object",
        "command": "Find my backpack",
        "expected_plan": ["locate"],
        "expected_object": "backpack", "expected_waypoint": None,
        "expected_success": True,
        "setup_note": "Backpack visible in scene.",
    },
    {
        "id": "L15", "category": "locate_object",
        "command": "Where is the charger",
        "expected_plan": ["locate"],
        "expected_object": "charger", "expected_waypoint": None,
        "expected_success": True,
        "setup_note": "Charger visible in scene.",
    },
    # ── navigate (2) ─────────────────────────────────────────────────────────
    {
        "id": "L16", "category": "navigate",
        "command": "Go to the kitchen",
        "expected_plan": ["navigate"],
        "expected_object": None, "expected_waypoint": "kitchen",
        "expected_success": True,
        "setup_note": "Kitchen waypoint recorded in GraphNav map.",
    },
    {
        "id": "L17", "category": "navigate",
        "command": "Head to the desk",
        "expected_plan": ["navigate"],
        "expected_object": None, "expected_waypoint": "desk",
        "expected_success": True,
        "setup_note": "Desk waypoint recorded in GraphNav map.",
    },
    # ── scan_environment (2) ─────────────────────────────────────────────────
    {
        "id": "L18", "category": "scan_environment",
        "command": "Scan the room",
        "expected_plan": ["scan"],
        "expected_object": None, "expected_waypoint": None,
        "expected_success": True,
        "setup_note": "No special staging needed.",
    },
    {
        "id": "L19", "category": "scan_environment",
        "command": "Look around and tell me what you see",
        "expected_plan": ["scan", "scan"],
        "expected_object": None, "expected_waypoint": None,
        "expected_success": True,
        "setup_note": "No special staging needed.",
    },
    # ── edge_case (1) ─────────────────────────────────────────────────────────
    {
        "id": "L20", "category": "edge_case",
        "command": "Bring me something",
        "expected_plan": [],
        "expected_object": None, "expected_waypoint": None,
        "expected_success": False,
        "setup_note": "Should fail + request clarification. No staging needed.",
    },
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def append_jsonl(path: Path, record: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")

def save_run_summary(run_id: str, records: List[Dict]) -> Path:
    """Save full run summary as a single JSON file."""
    summary_path = Path(f"data/live_trials_{run_id}.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"run_id": run_id, "trials": records}, f, indent=2)
    return summary_path

def plan_from_result(result: ExecutionResult) -> List[str]:
    return [s.skill for s in result.steps]

def safe_pct(num: int, den: int) -> str:
    return "N/A" if den == 0 else f"{num / den:.1%}"

def bar(ratio: float, width: int = 30) -> str:
    filled = round(ratio * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"

def prompt_ready(trial: Dict, index: int, total: int) -> bool:
    """
    Prompt operator before each trial.
    Returns True to proceed, False to exit and save.
    """
    print("\n" + "─" * 62)
    print(f"  Trial {index}/{total}  [{trial['id']}]  {trial['category']}")
    print(f"  Command : \"{trial['command']}\"")
    print(f"  Setup   : {trial['setup_note']}")
    print("─" * 62)

    while True:
        ans = input("  Ready? [y]es / [n]o (exit and save): ").strip().lower()
        if ans in ("y", "yes", ""):
            return True
        if ans in ("n", "no", "q", "quit", "exit"):
            return False
        print("  Please enter 'y' to continue or 'n' to exit.")

def prompt_notes() -> str:
    """Optional operator note after each trial."""
    note = input("  Notes (press Enter to skip): ").strip()
    return note

# ── Live-specific data extraction ─────────────────────────────────────────────

def extract_live_fields(result: ExecutionResult, parsed: Dict) -> Dict[str, Any]:
    """
    Pull live-hardware-specific fields out of the result and parsed command.
    These fields are not present in dry-run logs.
    """
    grounding = parsed.get("grounding") or {}

    # Detection source: "spot" if live camera was used, "mock" otherwise
    detection_source = grounding.get("source", "unknown")

    # Best match detection confidence (from top candidate if available)
    detection_conf = None
    top_candidates = grounding.get("top_candidates", [])
    if top_candidates:
        detection_conf = top_candidates[0].get("detection_conf")

    # Count pick_up attempts (retries + 1 if it was attempted)
    grasp_attempts = 0
    for step in result.steps:
        if step.skill == "pick_up":
            grasp_attempts = step.retries + 1
            break

    return {
        "detection_source": detection_source,
        "detection_conf":   detection_conf,
        "grasp_attempts":   grasp_attempts,
    }

# ── Trial runner ──────────────────────────────────────────────────────────────

def run_trial(
    executor: TaskExecutor,
    trial: Dict,
    run_id: str,
    operator_notes: str = "",
) -> Dict[str, Any]:
    """Execute one trial and return the full log record."""

    t_start = time.time()
    result = executor.execute(trial["command"])
    elapsed = time.time() - t_start

    # Re-interpret to extract params (same as evaluate_phase4.py)
    parsed = executor.inter.interpret(trial["command"])
    params = executor._extract_params(parsed["intent"]["label"], parsed)

    pred_plan     = plan_from_result(result)
    pred_object   = params.get("object_label")
    pred_waypoint = params.get("waypoint_id")

    gold_plan     = trial.get("expected_plan", [])
    gold_object   = trial.get("expected_object")
    gold_waypoint = trial.get("expected_waypoint")

    plan_ok     = (pred_plan == gold_plan) if gold_plan is not None else None
    object_ok   = (pred_object == gold_object) if gold_object else None
    waypoint_ok = (pred_waypoint == gold_waypoint) if gold_waypoint else None

    expected_success = trial.get("expected_success", True)
    success_correct  = (result.success == expected_success)

    failed_skills = [s.skill for s in result.steps if not s.success]
    n_retries     = sum(s.retries for s in result.steps)

    live_fields = extract_live_fields(result, parsed)

    return {
        "run_id":             run_id,
        "trial_id":           trial["id"],
        "mode":               "live",
        "timestamp":          datetime.now(timezone.utc).isoformat(),
        "command":            trial["command"],
        "category":           trial["category"],
        "success":            result.success,
        "expected_success":   expected_success,
        "success_correct":    success_correct,
        "predicted_plan":     pred_plan,
        "expected_plan":      gold_plan,
        "plan_correct":       plan_ok,
        "predicted_object":   pred_object,
        "expected_object":    gold_object,
        "object_correct":     object_ok,
        "predicted_waypoint": pred_waypoint,
        "expected_waypoint":  gold_waypoint,
        "waypoint_correct":   waypoint_ok,
        "n_steps":            len(result.steps),
        "n_recovered":        result.recovered,
        "n_retries_total":    n_retries,
        "failed_skills":      failed_skills,
        "clarifications":     result.clarifications_needed,
        "error":              result.error,
        "elapsed_sec":        round(elapsed, 3),
        # Live-only fields
        "detection_source":   live_fields["detection_source"],
        "detection_conf":     live_fields["detection_conf"],
        "grasp_attempts":     live_fields["grasp_attempts"],
        "operator_notes":     operator_notes,
    }

# ── Metrics + report ──────────────────────────────────────────────────────────

def aggregate(records: List[Dict]) -> Dict[str, Any]:
    total           = len(records)
    actual_success  = sum(1 for r in records if r["success"])
    correct_outcome = sum(1 for r in records if r["success_correct"])

    plan_trials  = [r for r in records if r["plan_correct"] is not None]
    plan_correct = sum(1 for r in plan_trials if r["plan_correct"])

    obj_trials  = [r for r in records if r["object_correct"] is not None]
    obj_correct = sum(1 for r in obj_trials if r["object_correct"])

    wp_trials  = [r for r in records if r["waypoint_correct"] is not None]
    wp_correct = sum(1 for r in wp_trials if r["waypoint_correct"])

    total_steps     = sum(r["n_steps"] for r in records)
    total_recovered = sum(r["n_recovered"] for r in records)
    total_retries   = sum(r["n_retries_total"] for r in records)
    total_grasps    = sum(r.get("grasp_attempts", 0) for r in records)
    grasp_trials    = [r for r in records if r.get("grasp_attempts", 0) > 0]
    grasp_success   = sum(1 for r in grasp_trials if r["success"])

    skill_failures: Dict[str, int] = defaultdict(int)
    by_category: Dict[str, Dict]   = defaultdict(lambda: {"success": 0, "total": 0})
    for r in records:
        for sk in r["failed_skills"]:
            skill_failures[sk] += 1
        cat = r["category"]
        by_category[cat]["total"] += 1
        if r["success"]:
            by_category[cat]["success"] += 1

    return {
        "total":            total,
        "actual_success":   actual_success,
        "correct_outcome":  correct_outcome,
        "plan_correct":     plan_correct,
        "plan_trials":      len(plan_trials),
        "obj_correct":      obj_correct,
        "obj_trials":       len(obj_trials),
        "wp_correct":       wp_correct,
        "wp_trials":        len(wp_trials),
        "total_steps":      total_steps,
        "total_recovered":  total_recovered,
        "total_retries":    total_retries,
        "total_grasps":     total_grasps,
        "grasp_trials":     len(grasp_trials),
        "grasp_success":    grasp_success,
        "skill_failures":   dict(sorted(skill_failures.items(),
                                        key=lambda x: x[1], reverse=True)),
        "by_category":      dict(by_category),
    }

def print_report(records: List[Dict], m: Dict, run_id: str) -> None:
    W = 62
    is_live = os.environ.get("USE_SPOT", "false").lower() == "true"
    mode    = "LIVE" if is_live else "DRY-RUN (mock)"

    print("\n" + "═" * W)
    print("  Phase IV Live Trial Report")
    print(f"  Run ID   : {run_id}")
    print(f"  Trials   : {m['total']}    Mode: {mode}")
    print("═" * W)

    # 1. Task completion
    comp = m["actual_success"] / m["total"] if m["total"] else 0
    print(f"\n── 1. Task Completion Rate  (thesis target: 65–75%)")
    print(f"   {m['actual_success']}/{m['total']}  {comp:.1%}  {bar(comp)}")
    print(f"   {'O' if comp >= 0.65 else 'X'} {'Meets' if comp >= 0.65 else 'Below'} thesis threshold (≥65%)")
    oa = m["correct_outcome"] / m["total"] if m["total"] else 0
    print(f"\n   Outcome Accuracy: {m['correct_outcome']}/{m['total']}  {oa:.1%}")

    # 2. Plan accuracy
    print(f"\n── 2. Plan Accuracy")
    if m["plan_trials"]:
        pa = m["plan_correct"] / m["plan_trials"]
        print(f"   {m['plan_correct']}/{m['plan_trials']}  {pa:.1%}  {bar(pa)}")
    else:
        print("   No annotated plans -- skipped.")

    # 3. Object extraction
    print(f"\n── 3. Object Extraction Accuracy")
    if m["obj_trials"]:
        oe = m["obj_correct"] / m["obj_trials"]
        print(f"   {m['obj_correct']}/{m['obj_trials']}  {oe:.1%}  {bar(oe)}")

    # 4. Waypoint extraction
    print(f"\n── 4. Waypoint Extraction Accuracy")
    if m["wp_trials"]:
        we = m["wp_correct"] / m["wp_trials"]
        print(f"   {m['wp_correct']}/{m['wp_trials']}  {we:.1%}  {bar(we)}")

    # 5. Execution stats
    print(f"\n── 5. Execution Stats")
    print(f"   Total steps     : {m['total_steps']}")
    print(f"   Steps recovered : {m['total_recovered']}  ({safe_pct(m['total_recovered'], m['total_steps'])})")
    print(f"   Total retries   : {m['total_retries']}")
    if m["grasp_trials"]:
        gs = m["grasp_success"] / m["grasp_trials"]
        print(f"   Grasp success   : {m['grasp_success']}/{m['grasp_trials']}  {gs:.1%}  (live hardware)")

    # 6. Skill failures
    print(f"\n── 6. Failure Breakdown by Skill")
    if m["skill_failures"]:
        for sk, cnt in m["skill_failures"].items():
            print(f"   {sk:<18}  {cnt} failure{'s' if cnt != 1 else ''}")
    else:
        print("   No skill failures.  O")

    # 7. Category breakdown
    print(f"\n── 7. Per-Category Breakdown")
    cw = max((len(c) for c in m["by_category"]), default=8)
    for cat, counts in sorted(m["by_category"].items()):
        s, t = counts["success"], counts["total"]
        rate = s / t if t else 0
        print(f"   {cat:<{cw}}  {s}/{t}  {safe_pct(s,t)}  {bar(rate, 20)}")

    # 8. Trial detail (failures/mismatches only)
    mismatches = [
        r for r in records
        if not r["success_correct"]
        or r["plan_correct"] is False
        or r["object_correct"] is False
        or r["waypoint_correct"] is False
    ]
    print(f"\n── 8. Trial Detail  ({len(mismatches)} mismatch(es))")
    if not mismatches:
        print("   All trials matched expectations.  O")
    else:
        for r in mismatches:
            tag = "O" if r["success"] else "X"
            print(f"\n   [{tag}] {r['trial_id']}  \"{r['command']}\"")
            if r["plan_correct"] is False:
                print(f"        plan      pred → {r['predicted_plan']}")
                print(f"                  gold → {r['expected_plan']}")
            if r["object_correct"] is False:
                print(f"        object    pred → {r['predicted_object']!r}")
                print(f"                  gold → {r['expected_object']!r}")
            if r["waypoint_correct"] is False:
                print(f"        waypoint  pred → {r['predicted_waypoint']!r}")
                print(f"                  gold → {r['expected_waypoint']!r}")
            if not r["success_correct"]:
                print(f"        outcome   pred → success={r['success']}")
                print(f"                  gold → success={r['expected_success']}")
            if r.get("error"):
                print(f"        error  → {r['error']}")
            if r.get("operator_notes"):
                print(f"        notes  → {r['operator_notes']}")

    print(f"\n── Trial log : {LOG_FILE}")
    print("═" * W + "\n")

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    run_id  = datetime.now(timezone.utc).strftime("live_%Y%m%d_%H%M%S")
    records: List[Dict] = []

    print("\n" + "═" * 62)
    print("  SPOT Live Trial Runner")
    print(f"  Run ID : {run_id}")
    print(f"  Trials : {len(LIVE_TRIALS)}")
    print(f"  Log    : {LOG_FILE}")
    print("═" * 62)
    print("\n  Type 'n' at any ready prompt to exit and save completed trials.")
    print("  Ctrl+C also exits safely and saves.\n")

    # ── Graceful Ctrl+C handler ───────────────────────────────────────────────
    def _handle_interrupt(sig, frame):
        print("\n\n  [!] Interrupted. Saving completed trials...")
        _finalize(records, run_id)
        sys.exit(0)
    signal.signal(signal.SIGINT, _handle_interrupt)

    # ── Connect to SPOT ───────────────────────────────────────────────────────
    robot       = SpotRobot()
    interpreter = Phase1Interpreter()
    executor    = TaskExecutor(robot, interpreter=interpreter)
    robot.connect()

    try:
        for i, trial in enumerate(LIVE_TRIALS, 1):

            # ── Ready prompt ─────────────────────────────────────────────────
            if not prompt_ready(trial, i, len(LIVE_TRIALS)):
                print(f"\n  Exiting after {len(records)} completed trial(s).")
                break

            # ── Run trial ────────────────────────────────────────────────────
            record = run_trial(executor, trial, run_id)

            # ── Operator notes ───────────────────────────────────────────────
            tag    = "O " if record["success_correct"] else "X "
            p_tag  = "plan O" if record["plan_correct"] else ("plan X" if record["plan_correct"] is False else "plan -")
            o_tag  = f"obj = {record['predicted_object']!r}" if record["predicted_object"] else "obj = ?"
            status = "SUCCESS" if record["success"] else "FAILED "
            print(f"\n  {tag} {status}  {p_tag}  {o_tag}  ({record['elapsed_sec']:.2f}s)")

            notes = prompt_notes()
            record["operator_notes"] = notes

            # ── Save immediately ─────────────────────────────────────────────
            records.append(record)
            append_jsonl(LOG_FILE, record)
            print(f"  [saved] {trial['id']} logged to {LOG_FILE}")

    finally:
        robot.disconnect()
        _finalize(records, run_id)

def _finalize(records: List[Dict], run_id: str) -> None:
    """Print report and save summary JSON for however many trials completed."""
    if not records:
        print("  No trials completed — nothing to save.")
        return

    print(f"\n  {len(records)}/{len(LIVE_TRIALS)} trials completed.")
    m = aggregate(records)
    print_report(records, m, run_id)

    summary_path = save_run_summary(run_id, records)
    print(f"  Run summary saved: {summary_path}\n")


if __name__ == "__main__":
    main()