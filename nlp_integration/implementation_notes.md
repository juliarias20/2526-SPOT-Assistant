# Implementation Notes - Week of 2026-01-26

## What I built
- Phase I interpreter: spaCy extraction + embedding-based intetn classifier
- Task schema v1
- baselines to be evaluated against
- command template
- evaluation step for phase 1

## What worked
- basic level of commands with single steps "Go to the table and scan the room" / "Go to the table and bring the cup"

## What failed / edge cases 
- intent label mismatch for dual steps / multi step commands

## Decisions made (and why)
- Thrsholds for ambiguity
- SentenceTransformers instead of fine-tuning for ease of use
- Spacy comes with labeling of unstructured words and phrases and already trained

## Net steps
- Tune ambiguity thresholds (ambiguity_threshold, delta_threshold) so clarification detection improves
- Start Phase II: update split_clauses() to handle more connector types and evaluate decompoisiton accuracy (build evaluate_phase2.py next)
- Phase II evaluation harness (ordering accuracy / completeness vs your gold_steps)
- Task graph builder v2 (turn multi-step clauses into explicit nodes and edges)
- Clarification policy that matches your thesis metrics (when to ask, what to ask)
- SPOT skill primitive interface stub (so Phase IV is plug-and-play)

## Finished
- Completed Expand commands_gold.jsonl to 50 commands 