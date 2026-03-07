"""
evaluate_phase3.py
------------------
Evaluates Phase III: affordance-based object grounding.

Metrics:
    1. Top-1 Grounding Accuracy     -- correct object is the best_match
    2. Top-3 Grounding Accuracy     -- correct object appears in top_candidates
    3. Mean Affordance Score        -- average similarity of correct object
    4. Mean Reciprocal Rank         -- how high the correct object ranks

Gold data format (grounding_gold.jsonl):
    {"id": "g001", "verb": "write",  "correct_object": "pen",     "scene": ["pen","cup","bottle","scissors","notebook"]}
    {"id": "g002", "verb": "drink",  "correct_object": "bottle",  "scene": ["pen","cup","bottle","scissors","notebook"]}

Run:
    python evaluate_phase3.py
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from perception import DetectedObject, PerceptionModule

DATA = Path("data/grounding_gold.jsonl")

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

def scene_from_labels(labels: List[str]) -> List[DetectedObject]:
    """
    Convert a list of label strings into mock DetectedObjects.
    """
    return [DetectedObject(label = l, confidence = 0.85, bbox = []) for l in labels]

def reciprocal_rank(ranked_labels: List[str], correct: str) -> float:
    for i, label in enumerate(ranked_labels, start = 1):
        if label == correct:
            return 1.0 / i
    return 0.0

# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def main() -> None:
    if not DATA.exists():
        print(f"\n[!] Gold file not found: {DATA}")
        print("     Create data / grounding_gold.jsonl to run Phase III evaluation.")
        print("    Format: {\"id\": \"g001\", \"verb\": \"write\", \"correct_object\": \"pen\", \"scene\": [\"pen\",\"cup\",...]}")
        return
    
    rows = load_jsonl(DATA)
    module = PerceptionModule()

    top1_correct = 0
    top3_correct = 0
    mrr_total = 0.0
    aff_scores = []
    failures: List[Dict] = []

    for r in rows:
        verb = r["verb"]
        correct = r["correct_object"]
        scene = scene_from_labels(r["scene"])

        result = module.ground(verb = verb, candidates = scene, top_k = 3)
        ranked = [c["label"] for c in result["top_candidates"]]
        all_ranked = [c.label for c in 
                      module._score_affordance(verb, scene)]
        
        # Top - 1
        if result["best_match"] == correct:
            top1_correct += 1

        # Top - 3
        if correct in ranked:
            top3_correct += 1

        # MRR
        mrr_total += reciprocal_rank(all_ranked, correct)

        # Affordance score of the correct object
        found_score = False
        for c in result["top_candidates"]:
            if c["label"] == correct:
                aff_scores.append(c["affordance_score"])
                found_score = True
                break
        if not found_score:
            # Corrrect object not in top-3, find its score manually
            for g in module._score_affordance(verb, scene):
                if g.label == correct:
                    aff_scores.append(g.affordance_score)
                    break

        # Log failures
        if result["best_match"] != correct:
            failures.append({
                "id":      r["id"],
                "verb":    verb,
                "correct": correct,
                "pred":    result["best_match"],
                "top3":    ranked,
                "scene":   r["scene"],
            })

    n = len(rows)
    print("\n" + "=" * 60)
    print("  Phase III Evaluation — Affordance Grounding")
    print("=" * 60)
    print(f"\n── Top-1 Grounding Accuracy  (n={n})")
    print(f"   {top1_correct}/{n}  ({top1_correct/n:.1%})")
    print(f"\n── Top-3 Grounding Accuracy  (n={n})")
    print(f"   {top3_correct}/{n}  ({top3_correct/n:.1%})")
    print(f"\n── Mean Reciprocal Rank")
    print(f"   {mrr_total/n:.3f}")
    if aff_scores:
        print(f"\n── Mean Affordance Score (correct object)")
        print(f"   {sum(aff_scores)/len(aff_scores):.3f}")

    if failures:
        print(f"\n── Failures  ({len(failures)} total)")
        for f in failures:
            print(f"\n  {f['id']}: verb='{f['verb']}' correct='{f['correct']}' pred='{f['pred']}'")
            print(f"    top3  → {f['top3']}")
            print(f"    scene → {f['scene']}")
    else:
        print("\n── No failures.")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()