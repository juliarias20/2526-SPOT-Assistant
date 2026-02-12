import json
from pathlib import Path
from typing import List, Dict

from interpret import Phase1Interpreter
from baselines.baseline_keyword import keyword_baseline
from baselines.baseline_rules_spacy import rules_spacy_baseline


DATA = Path("data/commands_gold.jsonl")


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def accuracy(preds: List[str], golds: List[str]) -> float:
    correct = sum(p == g for p, g in zip(preds, golds))
    return correct / max(len(golds), 1)


def prf(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def main():
    rows = load_jsonl(DATA)
    engine = Phase1Interpreter()

    gold_intents = [r["gold_intent"] for r in rows]

    # Our system
    sys_intents = []
    sys_amb = []
    gold_amb = []
    for r in rows:
        out = engine.interpret(r["command"])
        sys_intents.append(out["intent"]["label"])
        sys_amb.append(out["clarification"]["required"])
        gold_amb.append(bool(r.get("needs_clarification", False)))

    print("\n=== False Positives (system asked to clarify, gold says no) ===")
    for r in rows:
        out = engine.interpret(r["command"])
        sa = out["clarification"]["required"]
        ga = bool(r.get("needs_clarification", False))
        if sa and not ga:
            intent = out["intent"]
            print(f'{r["id"]}: "{r["command"]}"')
            print(f'  pred_intent={intent["label"]} conf={intent["confidence"]} '
                f'second={intent.get("second_confidence")} delta={intent.get("delta")} '
                f'amb={intent["is_ambiguous"]} reason={intent["ambiguity_reason"]}')
            print(f'  questions={out["clarification"]["questions"]}\n')



    # Baselines
    kw_intents = [keyword_baseline(r["command"])["intent"] for r in rows]
    sp_intents = [rules_spacy_baseline(r["command"])["intent"] for r in rows]

    print("=== Phase I Intent Accuracy ===")
    print(f"System:    {accuracy(sys_intents, gold_intents): .3f}")
    print(f"Keyword:   {accuracy(kw_intents, gold_intents): .3f}")
    print(f"spaCyRule: {accuracy(sp_intents, gold_intents): .3f}")


    # Ambiguity / clarification evaluation (binary)
    tp = sum(sa and ga for sa, ga in zip(sys_amb, gold_amb))
    fp = sum(sa and (not ga) for sa, ga in zip(sys_amb, gold_amb))
    fn = sum((not sa) and ga for sa, ga in zip(sys_amb, gold_amb))
    prec, rec, f1 = prf(tp, fp, fn)


    print("\n=== Clarification Detection (binary) ===")
    print(f"Precision: {prec: .3f}  Recall: {rec:.3f}  F1: {f1: .3f}")
    print(f"(tp = {tp}, fp = {fp}, fn = {fn}, n = {len(rows)})")



if __name__ == "__main__":
    main()