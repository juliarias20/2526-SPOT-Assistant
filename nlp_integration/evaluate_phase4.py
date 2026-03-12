"""
evaluate_phase4.py
------------------
Phase IV Evaluation: Autonomous Task Execution

Runs a batch of natural language commands through the full executor pipeline
(interpret → decompose → ground → execute), logs every trial to JSONL, and
prints a human-readable metrics report.

Metrics:
    1. Task Completion Rate      -- success / total          (thesis target: 65-75%)
    2. Plan Accuracy             -- predicted skill sequence == expected (ordered)
    3. Object Extraction Acc     -- object_label == expected_object (where annotated)
    4. Waypoint Extraction Acc   -- waypoint_id == expected_waypoint (where annotated)
    5. Recovery Rate             -- recovered steps / total steps executed
    6. Failure Breakdown by Skill -- which skills failed most often
    7. Per-Category Breakdown    -- completion rate per command category

Gold data:
    data/execution_trials_gold.jsonl
    Fields: id, command, category, expected_plan, expected_object (opt),
            expected_waypoint (opt), expected_success

Trial log output:
    data/execution_trials.jsonl
    One record per trial; appends on repeated runs (timestamped).

Run (dry-run, no SPOT required):
    python evaluate_phase4.py

Run with live SPOT:
    USE_SPOT=true SPOT_
"""

import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from interpret import Phase1Interpreter
from executor import TaskExecutor, ExecutionResult
from spot_skills import SpotRobot

# ── Paths ─────────────────────────────────────────────────────────────────────
GOLD_FILE  = Path("data/execution_gold.jsonl")
LOG_FILE   = Path("data/execution_trials.jsonl")


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    for line in path.read_text(encoding = "utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows

def append_jsonl(path: Path, record: Dict) -> None:
    path.parent.mkdir(parents = True, exist_ok = True)
    with path.open("a", encoding = "utf-8") as fh:
        fh.write(json.dumps(record) + "\n")

def plan_from_result(result: ExecutionResult) -> List[str]:
    """
    Extract the ordered skill sequence from an ExecutionResult.
    """
    return [s.skill for s in result.steps]

def plan_matches(pred: List[str], gold: List[str]) -> bool:
    """
    exact ordered match of skill sequences.
    """
    return pred == gold

def safe_pct(num: int, den: int) -> str:
    if den == 0:
        return "N/A"
    return f"{num / den:.1%}"

def bar(ratio: float, width: int = 30) -> str:
    """
    ASCII progress bar for a ratio in [0, 1]. 
    """
    filled = round(ratio * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"

# ── Trial runner ──────────────────────────────────────────────────────────────

def run_trial(
    executor: TaskExecutor,
    row: Dict,
    run_id: str,
) -> Dict[str, Any]:
    """
    Execute one gold command and return a trial record dict.

    Trial record fields:
        run_id, trial_id, timestamp, command, category,
        success, expected_success, success_correct,
        predicted_plan, expected_plan, plan_correct,
        predicted_object, expected_object, object_correct,
        predicted_waypoint, expected_waypoint, waypoint_correct,
        n_steps, n_recovered, n_retries_total,
        failed_skills, clarifications, error
    """

    t_start = time.time()
    result = executor.execute(row["command"])
    elapsed = time.time() - t_start

    pred_plan = plan_from_result(result)
    gold_plan = row.get("expected_plan", [])
    plan_ok = plan_matches(pred_plan, gold_plan) if gold_plan else None

    # Object / waypoint extraction: pull from executor's last params.
    # We re-run _extract_params here to inspect what the executor resolved.
    parsed = executor.inter.interpret(row["command"])
    params = executor._extract_params(parsed["intent"]["label"], parsed)
    pred_object = params.get("object_label")
    pred_waypoint = params.get("waypoint_id")

    gold_object = row.get("expected_object")
    gold_waypoint = row.get("expected_waypoint")

    object_ok = (pred_object == gold_object) if gold_object else None
    waypoint_ok = (pred_waypoint == gold_waypoint) if gold_waypoint else None

    failed_skills = [s.skill for s in result.steps if not s.success]
    n_retries = sum(s.retries for s in result.steps)

    expected_success = row.get("expected_success", True)
    success_correct = (result.success == expected_success)

    record = {
        "run_id":             run_id,
        "trial_id":           row["id"],
        "timestamp":          datetime.now(timezone.utc).isoformat(),
        "command":            row["command"],
        "category":           row.get("category", "unknown"),
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
    }

    return record

# ── Metrics aggregation ───────────────────────────────────────────────────────

def aggregate(records: List[Dict]) -> Dict[str, Any]:
    """Compute all aggregate metrics from a list of trial records."""

    total = len(records)

    # ── 1. Task completion rate ────────────────────────────────────────────────
    # "Correct" means success matched the expected outcome.
    # Actual successes (for the thesis completion-rate metric):
    actual_successes = sum(1 for r in records if r["success"])
    correct_outcomes = sum(1 for r in records if r["success_correct"])

    # ── 2. Plan accuracy ────────────────────────────────────────────────────────
    plan_trials  = [r for r in records if r["plan_correct"] is not None]
    plan_correct = sum(1 for r in plan_trials if r["plan_correct"])

    # ── 3. Object extraction accuracy ──────────────────────────────────────────
    obj_trials  = [r for r in records if r["object_correct"] is not None]
    obj_correct = sum(1 for r in obj_trials if r["object_correct"])

    # ── 4. Waypoint extraction accuracy ────────────────────────────────────────
    wp_trials  = [r for r in records if r["waypoint_correct"] is not None]
    wp_correct = sum(1 for r in wp_trials if r["waypoint_correct"])

    # ── 5. Recovery rate ────────────────────────────────────────────────────────
    total_steps     = sum(r["n_steps"] for r in records)
    total_recovered = sum(r["n_recovered"] for r in records)
    total_retries   = sum(r["n_retries_total"] for r in records)

    # ── 6. Failure breakdown by skill ──────────────────────────────────────────
    skill_failures: Dict[str, int] = defaultdict(int)
    for r in records:
        for skill in r["failed_skills"]:
            skill_failures[skill] += 1

    # ── 7. Per-category breakdown ───────────────────────────────────────────────
    by_category: Dict[str, Dict] = defaultdict(lambda: {"success": 0, "total": 0})
    for r in records:
        cat = r["category"]
        by_category[cat]["total"] += 1
        if r["success"]:
            by_category[cat]["success"] += 1

    return {
        "total":            total,
        "actual_successes": actual_successes,
        "correct_outcomes": correct_outcomes,
        "plan_correct":     plan_correct,
        "plan_trials":      len(plan_trials),
        "obj_correct":      obj_correct,
        "obj_trials":       len(obj_trials),
        "wp_correct":       wp_correct,
        "wp_trials":        len(wp_trials),
        "total_steps":      total_steps,
        "total_recovered":  total_recovered,
        "total_retries":    total_retries,
        "skill_failures":   dict(sorted(skill_failures.items(),
                                        key=lambda x: x[1], reverse=True)),
        "by_category":      dict(by_category),
    }


# ── Report printer ────────────────────────────────────────────────────────────

def print_report(records: List[Dict], m: Dict, run_id: str) -> None:
    """Print a formatted Phase IV evaluation report to stdout."""

    WIDTH = 62

    def hline(char="─"):
        print(char * WIDTH)

    def section(title: str):
        print(f"\n── {title}")

    print("\n" + "═" * WIDTH)
    print(f"  Phase IV Evaluation Report")
    print(f"  Run ID  : {run_id}")
    print(f"  Commands: {m['total']}    Mode: {'LIVE' if __import__('os').environ.get('USE_SPOT','false').lower()=='true' else 'DRY-RUN (mock)'}")
    print("═" * WIDTH)

    # ── 1. Task Completion Rate ────────────────────────────────────────────────
    section("1. Task Completion Rate  (thesis target: 65–75%)")
    comp_rate = m["actual_successes"] / m["total"] if m["total"] else 0
    print(f"   {m['actual_successes']}/{m['total']}  {comp_rate:.1%}  {bar(comp_rate)}")
    if comp_rate >= 0.65:
        print(f"   O Meets thesis threshold (≥65%)")
    else:
        print(f"   X Below thesis threshold (≥65%)  -- investigate failures")

    # Outcome accuracy (success matched expected)
    oa = m["correct_outcomes"] / m["total"] if m["total"] else 0
    print(f"\n   Outcome Accuracy (success == expected):  "
          f"{m['correct_outcomes']}/{m['total']}  {oa:.1%}")

    # ── 2. Plan Accuracy ───────────────────────────────────────────────────────
    section("2. Plan Accuracy  (predicted skill sequence == expected)")
    if m["plan_trials"]:
        pa = m["plan_correct"] / m["plan_trials"]
        print(f"   {m['plan_correct']}/{m['plan_trials']}  {pa:.1%}  {bar(pa)}")
    else:
        print("   No annotated expected_plan entries found — skipped.")

    # ── 3. Object Extraction Accuracy ─────────────────────────────────────────
    section("3. Object Extraction Accuracy")
    if m["obj_trials"]:
        oa2 = m["obj_correct"] / m["obj_trials"]
        print(f"   {m['obj_correct']}/{m['obj_trials']}  {oa2:.1%}  {bar(oa2)}")
    else:
        print("   No annotated expected_object entries found — skipped.")

    # ── 4. Waypoint Extraction Accuracy ───────────────────────────────────────
    section("4. Waypoint Extraction Accuracy")
    if m["wp_trials"]:
        wa = m["wp_correct"] / m["wp_trials"]
        print(f"   {m['wp_correct']}/{m['wp_trials']}  {wa:.1%}  {bar(wa)}")
    else:
        print("   No annotated expected_waypoint entries found — skipped.")

    # ── 5. Recovery Rate ──────────────────────────────────────────────────────
    section("5. Execution Stats")
    print(f"   Total steps executed : {m['total_steps']}")
    print(f"   Steps recovered      : {m['total_recovered']}  "
          f"(via retry)  {safe_pct(m['total_recovered'], m['total_steps'])}")
    print(f"   Total retries        : {m['total_retries']}")

    # ── 6. Failure Breakdown by Skill ─────────────────────────────────────────
    section("6. Failure Breakdown by Skill")
    if m["skill_failures"]:
        for skill, count in m["skill_failures"].items():
            print(f"   {skill:<18}  {count} failure{'s' if count != 1 else ''}")
    else:
        print("   No skill failures recorded.  O")

    # ── 7. Per-Category Breakdown ─────────────────────────────────────────────
    section("7. Per-Category Breakdown")
    cat_width = max((len(c) for c in m["by_category"]), default=8)
    for cat, counts in sorted(m["by_category"].items()):
        s, t = counts["success"], counts["total"]
        rate = s / t if t else 0
        print(f"   {cat:<{cat_width}}  {s}/{t}  {safe_pct(s,t)}  {bar(rate, 20)}")

    # ── Per-trial detail for failures / mismatches ─────────────────────────────
    failures = [
        r for r in records
        if not r["success_correct"]
        or r["plan_correct"] is False
        or r["object_correct"] is False
        or r["waypoint_correct"] is False
    ]

    section(f"8. Trial Detail  ({len(failures)} mismatch(es))")
    if not failures:
        print("   All trials matched expectations.  O")
    else:
        for r in failures:
            tag = "O" if r["success"] else "X"
            print(f"\n   [{tag}] {r['trial_id']}  \"{r['command']}\"")
            # Plan mismatch
            if r["plan_correct"] is False:
                print(f"        plan      pred → {r['predicted_plan']}")
                print(f"                  gold → {r['expected_plan']}")
            # Object mismatch
            if r["object_correct"] is False:
                print(f"        object    pred → {r['predicted_object']!r}")
                print(f"                  gold → {r['expected_object']!r}")
            # Waypoint mismatch
            if r["waypoint_correct"] is False:
                print(f"        waypoint  pred → {r['predicted_waypoint']!r}")
                print(f"                  gold → {r['expected_waypoint']!r}")
            # Success mismatch
            if not r["success_correct"]:
                print(f"        outcome   pred → success={r['success']}")
                print(f"                  gold → success={r['expected_success']}")
                if r["error"]:
                    print(f"        error  → {r['error']}")
                if r["clarifications"]:
                    print(f"        clarifs → {r['clarifications']}")

    # ── Trial log path ─────────────────────────────────────────────────────────
    print(f"\n── Trial log written to: {LOG_FILE}")
    print("═" * WIDTH + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not GOLD_FILE.exists():
        print(f"\n[!] Gold file not found: {GOLD_FILE}")
        print("    Expected location: data/execution_trials_gold.jsonl")
        print("    Create it or copy from the project repository.\n")
        return

    gold = load_jsonl(GOLD_FILE)
    print(f"\n[phase4] Loaded {len(gold)} gold commands from {GOLD_FILE}")

    # One shared interpreter and robot for the whole run (avoids reloading models)
    robot       = SpotRobot()   # USE_SPOT=false -> mock mode
    interpreter = Phase1Interpreter()
    executor    = TaskExecutor(robot, interpreter=interpreter)

    run_id  = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    records: List[Dict] = []

    print(f"[phase4] Starting evaluation run: {run_id}\n")
    hline = "─" * 62

    for i, row in enumerate(gold, 1):
        print(f"  [{i:02d}/{len(gold)}] {row['id']}  \"{row['command']}\"")
        record = run_trial(executor, row, run_id)
        records.append(record)
        append_jsonl(LOG_FILE, record)

        # One-line result summary per trial
        tag    = "O " if record["success_correct"] else "X "
        p_tag  = "plan O" if record["plan_correct"] else ("plan X" if record["plan_correct"] is False else "plan - ")
        o_tag  = f"obj = {record['predicted_object']!r}" if record["predicted_object"] else "obj = ?"
        status = "SUCCESS" if record["success"] else "FAILED "
        print(f"         {tag} {status}  {p_tag}  {o_tag}  ({record['elapsed_sec']:.2f}s)")

    print(f"\n[phase4] All {len(records)} trials complete.")

    m = aggregate(records)
    print_report(records, m, run_id)


if __name__ == "__main__":
    main()