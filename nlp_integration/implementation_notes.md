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

## Next steps
- Add punctuation stripping (tiny quality win)
- Build evaluate_phase2.py
- Implement task graph v2
- Expand commands_gold.jsonl with gold_clause_intents for multi-step commands
- Stub SPOT interface
- Draft presentation outline
- Polish Phase II metrics
- Create system architecture diagrams
- Build slides
- Dry run presentation
- Identify experimental plots to show

## Finished
- Implement Phase II clause splitter
- Tune ambiguity thresholds (ambiguity_threshold, delta_threshold) so clarification detection improves
- Completed Expand commands_gold.jsonl to 50 commands 
