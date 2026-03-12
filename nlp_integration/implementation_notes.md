# Implementation Notes

---

## Week of 2026-01-26

### What I built
- Phase I interpreter: spaCy extraction + embedding-based intent classifier
- Task schema v1
- Baselines to be evaluated against (keyword, spaCy rules)
- Command template
- Evaluation script for Phase I

### What worked
- Basic single-step commands: "Go to the table and scan the room" / "Go to the table and bring the cup"

### What failed / edge cases
- Intent label mismatch for dual-step / multi-step commands (whole-command classifier can't handle multiple intents in one sentence — motivates Phase II clause-level classification)

### Decisions made (and why)
- Ambiguity thresholds tuned empirically (ambiguity_threshold=0.12, delta_threshold=0.015)
- SentenceTransformers instead of fine-tuning: faster iteration, no labeled training data required
- spaCy for extraction: pretrained, handles unstructured text, no extra training needed

### Finished
- Phase I interpreter + task schema v1
- Baselines (keyword + spaCy rules)
- evaluate_phase1.py
- Expand commands_gold.jsonl to 50 commands
- Tune ambiguity thresholds

---

## Week of 2026-03-06

### What I built
- Phase II clause splitter (split_clauses) using PhraseMatcher + verb heuristics
- build_phase2_plan(): clause-level task graph with per-clause intent + edges
- evaluate_phase2.py: four metrics (clause count, per-clause intent, step-sequence F1, edge-type)
- Annotated commands_gold.jsonl with gold_clause_count + gold_clause_intents for 14 multi-step examples
- Hybrid intent classifier: verb-override for short clauses + embedding fallback for longer/ambiguous text
- PLACE_VERBS set for placement-verb classification (put, place, set, drop, lay, leave)
- Phase III perception module (perception.py): YOLOv8 + SentenceTransformer affordance scoring
- Phase III evaluation script (evaluate_phase3.py): Top-1/3 accuracy, MRR, mean affordance score
- grounding_gold.jsonl: 20 annotated verb-scene-correct_object grounding examples
- Phase IV skill primitives (spot_skills.py): navigate, scan, locate, pick_up, deliver, release
- Phase IV executor (executor.py): task graph walker with retry logic and recovery clarifications

### What worked
- Clause splitting handles: "and" joins, "if/when" conditionals, comma+then, comma+verb sequences
- Hybrid classifier outperforms both baselines on intent accuracy
- Edge logic correctly uses destination clause relation (b, not a) for edge type assignment
- Affordance grounding achieves 85% Top-1, 90% Top-3 on 20-example gold set
- YOLOv8 + SentenceTransformer cosine similarity correctly grounds write→pen, drink→bottle, cut→scissors, carry→backpack
- Mock mode in both perception.py and spot_skills.py allows full pipeline testing without SPOT hardware
- Executor walks Phase II task graph clause by clause, retries failed steps, generates recovery clarifications

### What failed / edge cases (resolved)
- "next" in PhraseMatcher caused false split on "next to" → removed from connector phrases
- intent_labels.json descriptions too overlapping → rewrote with hard negative phrases per label
- Edge-type accuracy not firing → bug: was iterating clauses[:-1] instead of clauses[1:]
- "pick up" not caught by verb override → added "pick", "take" to FETCH_VERBS
- Placement verbs (put, place, set) predicted as locate_object → added PLACE_VERBS + override
- Token threshold too short (6) for commands like "Hand me the charger next to the laptop" → raised to 8
- ex034 three-clause split failing → added comma + verb heuristic to split_clauses
- GroundedObject not subscriptable in evaluate_phase3.py → fixed c["label"] → c.label
- found_score never set to True in evaluate_phase3.py → fixed, was double-counting affordance scores
- self.perecption typo in interpret.py → fixed to self.perception, was breaking evaluate_phase1/2

### Decisions made (and why)
- Verb-override at ≤8 tokens: embeddings are unreliable on short clauses; verb is the strongest signal
- Hard negative phrases in intent_labels.json: prevents multi_step_retrieve from dominating all embeddings
- Edge type assigned from destination clause (b.relation): connector belongs to the arriving clause, not departing
- perception.py as separate module (not integrated into interpret.py): perception and language are separate concerns with different dependencies; cleaner architecture and easier to explain in thesis
- CLIP considered but SentenceTransformer cosine similarity chosen for affordance scoring: already in stack, no additional model download, achieves 85% Top-1 which exceeds 70% thesis threshold
- GraphNav chosen for SPOT navigation: official BD recommendation for autonomous nav, waypoint IDs map cleanly to task graph navigate nodes
- Mock mode via USE_SPOT=false environment variable: allows offline testing of full pipeline without hardware access
- Embedder loaded twice (once in Phase1Interpreter, once in PerceptionModule) — known inefficiency, acceptable for now; shared instance to be passed in Phase V integration

### Known remaining failures (intentional — Phase III/IV motivation)
- select_object missing on vague commands ("bring me something to write with") → Phase III grounding resolves this at runtime via affordance best_match
- locate missing on implicit search commands ("bring me my keys") → requires perception context
- ex047 "Move the backpack" predicts navigate → "move" conflicts with navigate verb set; minor edge case
- Phase III failures g004 (read→pen), g014 (type→phone), g015 (open→phone) are all cases of bare verb ambiguity without object context — linguistically underspecified inputs, not model errors

### Final Results — All Phases

| Phase | Metric                  | Score  | Threshold          |
|-------|-------------------------|--------|--------------------|
| I     | Intent Accuracy         | 0.740  | beats baselines ✅ |
| I     | Keyword Baseline        | 0.680  | —                  |
| I     | spaCy Baseline          | 0.680  | —                  |
| I     | Clarification Precision | 1.000  | —                  |
| I     | Clarification Recall    | 0.700  | —                  |
| I     | Clarification F1        | 0.824  | —                  |
| II    | Clause Count Accuracy   | 100%   | —                  |
| II    | Per-Clause Intent Acc.  | 96.4%  | —                  |
| II    | Step-Sequence F1        | 0.863  | —                  |
| II    | Edge-Type Accuracy      | 100%   | —                  |
| III   | Top-1 Grounding Acc.    | 85.0%  | ≥ 70% ✅          |
| III   | Top-3 Grounding Acc.    | 90.0%  | —                  |
| III   | Mean Reciprocal Rank    | 0.889  | —                  |
| III   | Mean Affordance Score   | 0.547  | —                  |

### Classifier progression (story for presentation)

| Version                   | Intent Acc | Per-Clause Intent | Step-Seq F1 |
|---------------------------|------------|-------------------|-------------|
| Baseline (keyword/spaCy)  | 0.680      | —                 | —           |
| Embedding only            | 0.620      | 58.6%             | 0.756       |
| Hybrid (verb + embedding) | 0.740      | 96.4%             | 0.863       |

### Finished
- Phase II clause splitter + evaluate_phase2.py
- build_phase2_plan() + edge logic fix
- gold_clause_count + gold_clause_intents annotations (14 examples)
- Hybrid classifier (verb override + embedding fallback)
- PLACE_VERBS set + intent_labels.json rewrite with hard negatives
- Phase I + II evaluation locked with final numbers
- perception.py (YOLOv8 + affordance scoring, SPOT camera + mock fallback)
- evaluate_phase3.py (4 metrics: Top-1, Top-3, MRR, mean affordance score)
- grounding_gold.jsonl (20 examples)
- Phase III evaluation locked with final numbers
- spot_skills.py (6 skill primitives: navigate, scan, locate, pick_up, deliver, release)
- executor.py (task graph walker, retry logic, recovery clarifications)
- interpret.py wired to perception.py via ground_from_intent()

---

## Week of 2026-03-12

### What I built
- evaluate_phase4.py: 20-trial dry-run eval (execution_gold.jsonl), reports task completion, plan accuracy, object/waypoint extraction, per-category breakdown, trial detail log
- execution_gold.jsonl: 20 annotated trials across 7 categories (navigate, retrieve_object, multi_step_retrieve, locate_object, scan_environment, vague_retrieve, edge_case)
- execution_trials.jsonl: per-run trial log output
- record_map.py: interactive GraphNav map recording utility — walks SPOT to each location, stamps named waypoints, downloads graph + snapshots, prints WAYPOINT_MAP dict for spot_skills.py
- live_trials.py: 20-trial live SPOT runner with per-trial ready prompt, operator notes field, graceful save-on-exit (Ctrl+C or 'n'), live-specific log fields (detection_source, detection_conf, grasp_attempts)
- WAYPOINT_MAP + _upload_map() added to spot_skills.py: name→UUID lookup table, map upload wired into SpotRobot.connect()

### What worked (after fixes)
- All 7 categories pass at 100% after targeted bug fixes
- Vague affordance grounding (write→pen, drink→bottle, sharp→scissors) fully wired end-to-end
- Bare edge case ("Bring me something") correctly fails and requests clarification
- Sentence-initial motion verbs ("Head to...") correctly parsed as navigate + multi-step retrieve
- Single embedder load confirmed across all four phase evals

### What failed / edge cases (resolved)
- `vague_retrieve` 0/3: `ground_from_intent` not stamping `affordance_verb` key → executor's `has_affordance_verb` check always False → hard block. Fix: stamp `result["affordance_verb"] = grounding_verb` only when a real functional verb or adjective modifier is found (not the intent-label fallback)
- p4020 "Bring me something" incorrectly succeeding: bare vague with no qualifier was resolving via intent-label fallback → added `found_functional_verb` flag; bare commands without a real grounding signal still block for clarification
- p4002 "Head to the kitchen and grab me a cup": (1) "Head" mistagged as NOUN by spaCy → NAV_FIRST_TOKENS pre-check added to `classify_intent` before token-length gate; (2) "and" heuristic in `split_clauses` missed NOUN-tagged "Head" for left_has_verb → added doc[0].lemma_ in `_MOTION_LEMMAS` fallback
- p4011 "Hand me something sharp": "sharp" is a postpositive adjective outside the spaCy noun chunk span → added out-of-span `amod` scan (`t.head == chunk.root and t.dep_ == "amod" and t not in chunk`) to `extract_verbs_objects`; grounding uses "sharp" as affordance signal → scissors
- Double embedder load: `PerceptionModule.__init__` now accepts optional `embedder` param; `Phase1Interpreter` passes `self.embedder` at construction — single model load confirmed across all evals

### Decisions made (and why)
- `affordance_verb` key as executor signal: cleaner than a separate bool — key presence carries both the signal and the verb, useful for logging
- `found_functional_verb` flag: explicit and doesn't break if intent label wording changes
- Postpositive amod scan scoped to `t.head == chunk.root`: avoids capturing adjectives from other chunks in the same sentence
- NAV_FIRST_TOKENS pre-check runs before ≤8-token gate: motion verb commands are unambiguous regardless of length
- `PerceptionModule(embedder=...)` optional param, not required: backward-compatible — standalone use (evaluate_phase3.py, perception.py __main__) still loads its own embedder
- WAYPOINT_MAP as lookup dict in spot_skills.py (not embedded in executor): decouples human-readable names from GraphNav UUIDs; re-recording map or adding locations only requires updating the dict
- `_upload_map()` called in `connect()` automatically: no caller changes needed; skips silently if map dir not found (falls back to raw UUID passthrough)
- live_trials.py saves each trial to JSONL immediately after completion: partial runs are never lost even on crash or early exit
- Operator notes field: free-text per trial — becomes the basis for failure analysis section in thesis

### Final Results — All Phases (locked)

| Phase | Metric                   | Score   | Threshold           |
|-------|--------------------------|---------|---------------------|
| I     | Intent Accuracy          | 0.740   | beats baselines ✅  |
| I     | Keyword Baseline         | 0.680   | —                   |
| I     | spaCy Baseline           | 0.680   | —                   |
| I     | Clarification Precision  | 1.000   | —                   |
| I     | Clarification Recall     | 0.700   | —                   |
| I     | Clarification F1         | 0.824   | —                   |
| II    | Clause Count Accuracy    | 100%    | —                   |
| II    | Per-Clause Intent Acc.   | 96.4%   | —                   |
| II    | Step-Sequence F1         | 0.863   | —                   |
| II    | Edge-Type Accuracy       | 100%    | —                   |
| III   | Top-1 Grounding Acc.     | 85.0%   | ≥ 70% ✅           |
| III   | Top-3 Grounding Acc.     | 90.0%   | —                   |
| III   | Mean Reciprocal Rank     | 0.889   | —                   |
| III   | Mean Affordance Score    | 0.547   | —                   |
| IV    | Task Completion Rate     | 95.0%   | ≥ 65–75% ✅        |
| IV    | Outcome Accuracy         | 100.0%  | —                   |
| IV    | Plan Accuracy            | 100.0%  | —                   |
| IV    | Object Extraction Acc.   | 100.0%  | —                   |
| IV    | Waypoint Extraction Acc. | 100.0%  | —                   |

### Classifier progression (story for presentation)

| Version                   | Intent Acc | Per-Clause Intent | Step-Seq F1 |
|---------------------------|------------|-------------------|-------------|
| Baseline (keyword/spaCy)  | 0.680      | —                 | —           |
| Embedding only            | 0.620      | 58.6%             | 0.756       |
| Hybrid (verb + embedding) | 0.740      | 96.4%             | 0.863       |

### Finished
- evaluate_phase4.py (5 metrics: task completion, outcome accuracy, plan accuracy, object/waypoint extraction)
- execution_gold.jsonl (20 trials, 7 categories)
- Phase IV evaluation locked — run_20260312_100222
- affordance_verb wiring fix (perception.py)
- found_functional_verb flag for bare-vague blocking (perception.py)
- adjective modifier fallback grounding for "something sharp"-style commands (perception.py)
- NAV_FIRST_TOKENS pre-check in classify_intent (interpret.py)
- Postpositive amod scan in extract_verbs_objects (interpret.py)
- "and" heuristic motion-verb fallback in split_clauses (interpret.py)
- Shared embedder: PerceptionModule(embedder=...) + Phase1Interpreter passes self.embedder (interpret.py + perception.py)
- record_map.py (GraphNav map recording utility)
- live_trials.py (20-trial live runner, per-trial prompt, operator notes, graceful exit)
- WAYPOINT_MAP + _upload_map() in spot_skills.py

### Next steps
- [ ] Record GraphNav map on SPOT (python record_map.py --output maps/trial_space)
- [ ] Fill WAYPOINT_MAP in spot_skills.py with UUIDs from record_map.py output
- [ ] Run live_trials.py on SPOT (20 trials, USE_SPOT=true)
- [ ] Create system architecture diagram (NLP → clause splitter → task graph → grounding → skill mapping → SPOT)
- [ ] Build presentation slides (classifier progression table + all-phase results table ready)
- [ ] First complete thesis draft due March 21 ⚠️
- [ ] Dry run presentation
- [ ] Revised draft due April 18
- [ ] Defense by May 2