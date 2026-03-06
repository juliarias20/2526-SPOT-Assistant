"""
evaluate_phase2.py
------------------
Evaluates Phase II outputs: clause splitting and per-clause intent classification.

Gold data format (commands_gold.jsonl) -- existing fields used:
    gold_intent         : str           overall intent label
    gold_steps          : List[str]     expected action sequence
    gold_clause_count   : int           (optional) expected number of clauses
    gold_clause_intents : List[str]     (optional) per-clause intent labels, in order

Run:
    python evaluate_phase2.py
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from interpret import Phase1Interpreter

DATA = Path("data/commands_gold.jsonl")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows

def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1

def steps_from_phase2(plan: Dict) -> List[str]:
    """
    Map clause-level intent labels back to the action-step vocabulary used in
    gold_steps so we can do a rough sequence comparison.

    Intent -> representative steps (simplified mapping)
    """
    INTENT_TO_STEPS = {
        "navigate":               ["navigate"],
        "scan_environment":       ["scan"],
        "locate_object":          ["locate"],
        "retrieve_object":        ["pick_up", "deliver"],
        "multi_step_retrieve":    ["navigate", "pick_up", "deliver"],
        "multi_step_manipulation": ["pick_up", "deliver"],
    }
    out: List[str] = []
    for step in plan.get("steps", []):
        label = step["intent"]["label"]
        out.extend(INTENT_TO_STEPS.get(label, ["unknown"]))
    # Deduplicate consecutive duplicates (keeps oredring but removes noise)
    deduped: List[str] = []
    for s in out:
        if not deduped or s != deduped[-1]:
            deduped.append(s)
    return deduped

def clause_intent_accuracy(
        pred_steps: List[Dict],
        gold_clause_intents: List[str],
) -> Tuple[int, int]:
    """
    Returns (correct, total) for per-clause intent matching.
    Aligns by positon; extra clasues on either side count as wrong.
    """
    total = max(len(pred_steps), len(gold_clause_intents))
    correct = 0
    for i in range(min(len(pred_steps), len(gold_clause_intents))):
        if pred_steps[i]["intent"]["label"] == gold_clause_intents[i]:
            correct += 1
    return correct, total

# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def main() -> None:
    rows = load_jsonl(DATA)
    engine = Phase1Interpreter()

    # ── 1. Clause-count accuracy ─────────────────────────────────────────────
    # For every example that carries gold_clause_count, check whether
    # the splitter produced the expected number of clauses.
    cc_correct = cc_total = 0

    # ── 2. Per-clause intent accuracy (requires gold_clause_intents) ─────────
    ci_correct = ci_total = 0

    # ── 3. Step-sequence overlap (set F1 over step labels) ───────────────────
    # Treats predicted and gold step lists as bags of labels and computes F1.
    seq_tp = seq_fp = seq_fn = 0

    # ── 4. Edge-type accuracy ────────────────────────────────────────────────
    # Compare predicted edge types against a heuristic gold derived from
    # the clause relations present in the gold (available without extra annotation).
    edge_correct = edge_total = 0

    # ── Per-example detail for failed cases ──────────────────────────────────
    failures: List[Dict] = []

    for r in rows:
        out = engine.interpret(r["command"])
        plan = out["phase2"]["plan"]
        steps = plan["steps"]
        edges = plan["edges"]
        clauses = out["clauses"]

        # ── 1. Clause count ──────────────────────────────────────────────────
        if "gold_clause_count" in r:
            cc_total += 1
            if len(steps) == r["gold_clause_count"]:
                cc_correct += 1
            else:
                failures.append({
                    "id":      r["id"],
                    "command": r["command"],
                    "check":   "clause_count",
                    "pred":    len(steps),
                    "gold":    r["gold_clause_count"],
                    "clauses": [s["text"] for s in steps],
                })

        # ── 2. Per-clause intent ─────────────────────────────────────────────
        if "gold_clause_intents" in r:
            c, t = clause_intent_accuracy(steps, r["gold_clause_intents"])
            ci_correct += c
            ci_total += t
            if c < t:
                failures.append({
                    "id":      r["id"],
                    "command": r["command"],
                    "check":   "clause_intent",
                    "pred":    [s["intent"]["label"] for s in steps],
                    "gold":    r["gold_clause_intents"],
                })

        # ── 3. Step-sequence overlap ─────────────────────────────────────────
        gold_steps_set = set(r.get("gold_steps", []))
        pred_steps_set = set(steps_from_phase2(plan))
        tp = len(pred_steps_set & gold_steps_set)
        fp = len(pred_steps_set - gold_steps_set)
        fn = len(gold_steps_set - pred_steps_set)
        seq_tp += tp
        seq_fp += fp
        seq_fn += fn
        
        if fp or fn:
            failures.append({
                "id":      r["id"],
                "command": r["command"],
                "check":   "step_sequence",
                "pred":    sorted(pred_steps_set),
                "gold":    sorted(gold_steps_set),
            })

        # ── 4. Edge-type accuracy ────────────────────────────────────────────
        # Build expected edge types from gold clause relations (heuristic).
        expected_edge_types: List[str] = []
        for cl in clauses[1:]: # all but last clause drives an edge
            rel = cl.get("relation", "root")
            if rel == "condition":
                expected_edge_types.append("condition_of")
            elif rel in ("sequence", "coordination", "temporal", "purpose"):
                expected_edge_types.append("sequence")
            # root: no edge expected

        pred_edge_types = [e["type"] for e in edges]
        for pe, ge in zip(pred_edge_types, expected_edge_types):
            edge_total += 1
            if pe == ge:
                edge_correct += 1

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Phase II Evaluation")
    print("=" * 60)

    # Clause count
    if cc_total:
        print(f"\n── Clause Count Accuracy (n={cc_total})")
        print(f"   {cc_correct}/{cc_total}  ({cc_correct/cc_total:.1%})")
    else:
        print("\n── Clause Count Accuracy")
        print("   No gold_clause_count annotations found — skipped.")
        print("   Add  \"gold_clause_count\": <int>  to commands_gold.jsonl to enable.")

    # Per-clause intent
    if ci_total:
        print(f"\n── Per-Clause Intent Accuracy (n={ci_total} clauses)")
        print(f"   {ci_correct}/{ci_total}  ({ci_correct/ci_total:.1%})")
    else:
        print("\n── Per-Clause Intent Accuracy")
        print("   No gold_clause_intents annotations found — skipped.")
        print("   Add  \"gold_clause_intents\": [\"intent\", ...]  to commands_gold.jsonl.")

    # Step-sequence F1
    prec, rec, f1 = prf(seq_tp, seq_fp, seq_fn)
    print(f"\n── Step-Sequence Set F1  (n={len(rows)} commands)")
    print(f"   Precision: {prec:.3f}   Recall: {rec:.3f}   F1: {f1:.3f}")
    print(f"   (tp={seq_tp}, fp={seq_fp}, fn={seq_fn})")

    # Edge types
    if edge_total:
        print(f"\n── Edge-Type Accuracy  (n={edge_total} edges)")
        print(f"   {edge_correct}/{edge_total}  ({edge_correct/edge_total:.1%})")
    else:
        print("\n── Edge-Type Accuracy:  no multi-clause examples with edges found.")

    # Failures
    if failures:
        print(f"\n── Failures / Mismatches  ({len(failures)} total)")
        for f in failures:
            print(f"\n  [{f['check']}]  {f['id']}: \"{f['command']}\"")
            print(f"    pred → {f['pred']}")
            print(f"    gold → {f['gold']}")
    else:
        print("\n── No failures recorded.")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()