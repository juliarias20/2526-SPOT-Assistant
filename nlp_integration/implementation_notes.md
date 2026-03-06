# Implementation Notes - Week of 2026-01-26

What I built

Phase I interpreter: spaCy extraction + embedding-based intent classifier
Task schema v1
Baselines to be evaluated against (keyword, spaCy rules)
Command template
Evaluation script for Phase I

What worked

Basic single-step commands: "Go to the table and scan the room" / "Go to the table and bring the cup"

What failed / edge cases

Intent label mismatch for dual-step / multi-step commands (whole-command classifier can't handle multiple intents in one sentence — motivates Phase II clause-level classification)

Decisions made (and why)

Ambiguity thresholds tuned empirically (ambiguity_threshold=0.12, delta_threshold=0.015)
SentenceTransformers instead of fine-tuning: faster iteration, no labeled training data required
spaCy for extraction: pretrained, handles unstructured text, no extra training needed

Finished

Phase I interpreter + task schema v1
Baselines (keyword + spaCy rules)
evaluate_phase1.py
Expand commands_gold.jsonl to 50 commands
Tune ambiguity thresholds


Week of 2026-03-06
What I built

Phase II clause splitter (split_clauses) using PhraseMatcher + verb heuristics
build_phase2_plan(): clause-level task graph with per-clause intent + edges
evaluate_phase2.py: four metrics (clause count, per-clause intent, step-sequence F1, edge-type)
Annotated commands_gold.jsonl with gold_clause_count + gold_clause_intents for 14 multi-step examples
Hybrid intent classifier: verb-override for short clauses + embedding fallback for longer/ambiguous text
PLACE_VERBS set for placement-verb classification (put, place, set, drop, lay, leave)

What worked

Clause splitting handles: "and" joins, "if/when" conditionals, comma+then, comma+verb sequences
Hybrid classifier outperforms both baselines on intent accuracy
Edge logic correctly uses destination clause relation (b, not a) for edge type assignment

What failed / edge cases (resolved)

"next" in PhraseMatcher caused false split on "next to" → removed from connector phrases
intent_labels.json descriptions too overlapping → rewrote with hard negative phrases per label
Edge-type accuracy not firing → bug: was iterating clauses[:-1] instead of clauses[1:]
"pick up" not caught by verb override → added "pick", "take" to FETCH_VERBS
Placement verbs (put, place, set) predicted as locate_object → added PLACE_VERBS + override
Token threshold too short (6) for commands like "Hand me the charger next to the laptop" → raised to 8
ex034 three-clause split failing → added comma + verb heuristic to split_clauses

Decisions made (and why)

Verb-override at ≤8 tokens: embeddings are unreliable on short clauses; verb is the strongest signal
Hard negative phrases in intent_labels.json: prevents multi_step_retrieve from dominating all embeddings
Edge type assigned from destination clause (b.relation): connector belongs to the arriving clause, not departing

Final Phase I + II Results
MetricSystemKeyword BaselinespaCy BaselinePhase I Intent Accuracy0.7400.6800.680Clarification Precision1.000——Clarification Recall0.700——Clarification F10.824——Clause Count Accuracy100%——Per-Clause Intent Acc.96.4%——Step-Sequence F10.863——Edge-Type Accuracy100%——
Classifier progression (story for presentation)
VersionIntent AccPer-Clause IntentStep-Seq F1Baseline (keyword/spaCy)0.680——Embedding only0.62058.6%0.756Hybrid (verb + embedding)0.74096.4%0.863
Known remaining failures (intentional — Phase III motivation)

select_object missing on vague commands ("bring me something to write with") → requires affordance reasoning
locate missing on implicit search commands ("bring me my keys") → requires perception context
ex047 "Move the backpack" predicts navigate → "move" not in PLACE_VERBS (minor, add if desired)

Finished

Phase II clause splitter
build_phase2_plan() + edge logic fix
evaluate_phase2.py (4 metrics)
gold_clause_count + gold_clause_intents annotations (14 examples)
Hybrid classifier (verb override + embedding fallback)
PLACE_VERBS set
intent_labels.json rewrite with hard negatives
Punctuation stripping in clause cleaner
Phase I + II evaluation locked with final numbers

Next steps

 Stub SPOT interface (skill primitives: navigate, scan, pick_up, deliver)
 Phase III: integrate RGB/depth perception for object grounding
 Phase III: affordance-based object selection (resolve select_object failures)
 Implement task graph v2 (enrich nodes with extracted verb/object entities)
 Create system architecture diagrams
 Build presentation slides (results slide ready — use classifier progression table)
 Draft presentation outline
 Identify experimental plots
 Dry run presentation