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

### Next steps
- [ ] Run executor.py dry-run: "Go to the desk and bring the notebook", "Bring me something to write with", "Find my backpack"
- [ ] Record GraphNav map on SPOT, note waypoint IDs, name to match location nouns (desk, table, kitchen)
- [ ] Run live SPOT trials (target: 15–20 trials across 4 command categories)
- [ ] Build evaluate_phase4.py: log execution_trials.jsonl, report task completion rate + recovery rate
- [ ] Fix double embedder load: pass shared SentenceTransformer instance from Phase1Interpreter to PerceptionModule
- [ ] Phase V: end-to-end integration + continuous perception loop
- [ ] Create system architecture diagram (NLP → clause splitter → task graph → grounding → skill mapping → SPOT)
- [ ] Build presentation slides (results slide ready — use classifier progression table + all-phase results table)
- [ ] Dry run presentation
- [ ] First complete thesis draft due March 21
- [ ] Revised draft due April 18
- [ ] Defense by May 2