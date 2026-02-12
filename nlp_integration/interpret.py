import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import spacy
from sentence_transformers import SentenceTransformer, util

CONNECTORS = {"and", "then", "after", "before"}

FETCH_VERBS = {"get", "bring", "grab", "hand", "fetch", "give"}
FIND_VERBS = {"find", "locate"}
SCAN_VERBS = {"scan", "inspect", "check", "look"}

@dataclass
class IntentResult:
    label: str
    confidence: float
    second_confidence: float
    delta: float
    is_ambiguous: bool
    ambiguity_reason: Optional[str]

class Phase1Interpreter:
    """
    Phase I goal:
    - Extract verb(s), object candidates, and classify intent (embedding_based).
    - Create a minimal task graph (1-4 nodes) + clarification suggestion if ambiguous.
    """

    def __init__(
        self,
        intent_file: str = "models/intent_labels.json",
        embedding_model: str = "all-MiniLM-L6-v2",
        ambiguity_threshold: float = 0.12,
        delta_threshold: float = 0.015,
    ):
        self.nlp = spacy.load("en_core_web_sm")
        self.embedder = SentenceTransformer(embedding_model)
        self.ambiguity_threshold = ambiguity_threshold
        self.delta_threshold = delta_threshold

        with open(intent_file, "r", encoding="utf-8") as f:
            self.intent_labels: Dict[str, str] = json.load(f)

        # Precompute embeddings for intent descriptions
        self.intent_embs = {
            k: self.embedder.encode(v, convert_to_tensor=True)
            for k, v in self.intent_labels.items()
        }

    def split_clauses(self, text: str) -> List[str]:
        """
        Very lightweight splitter to support Phase II later; 
        fo Phase I we keep it simple
        """
        doc = self.nlp(text)
        clauses: List[List[str]] = []
        current: List[str] = []
        for token in doc:
            if token.text.lower() in CONNECTORS:
                if current:
                    clauses.append(current)
                    current = []
            else:
                current.append(token.text)
        if current:
            clauses.append(current)
        return [" ".join(c).strip() for c in clauses if "".join(c).strip()]
    
    def extract_verbs_objects(self, text: str) -> Tuple[List[str], List[Dict]]:
        doc = self.nlp(text)
        verbs: List[str] = []
        objects: List[Dict] = []

        # Collect Verbs (lemmatized)
        for t in doc:
            if t.pos_ == "VERB":
                verbs.append(t.lemma_)

        # Collect object-like noun chunks as candidates
        for chunk in doc.noun_chunks:
            head = chunk.root.lemma_
            modifiers = [t.text.lower() for t in chunk if t.dep_ in ("amod", "compound", "nummod")]
            objects.append({
                "text": chunk.text,
                "head": head,
                "modifiers": modifiers,
                "role": "unknown"
            })

        return verbs, objects
        
    def classify_intent(self, text: str) -> IntentResult:
        text_emb = self.embedder.encode(text, convert_to_tensor=True)
        sims = {k: float(util.cos_sim(text_emb, v)) for k, v in self.intent_embs.items()}
        ranked = sorted(sims.items(), key=lambda x: x[1], reverse=True)
        best_label, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        delta = best_score - second_score

        is_ambiguous = False
        reason = None

        if best_score < self.ambiguity_threshold:
            is_ambiguous = True
            reason = f"Low confidence intent (score = {best_score: .2f})."
        elif best_score < 0.25 and delta < self.delta_threshold:
            is_ambiguous = True
            reason = f"Top intents too close (Δ = {delta: .2f})."

        return IntentResult(best_label, best_score, second_score, delta, is_ambiguous, reason)
    
    def build_min_task_graph(self, intent: IntentResult, text: str) -> Dict:
        """
        Minimal Phase I mapping: intent -> skeleton plan.
        Phase II/III will expand. Here we keep it consistent and predictable.
        """
        nodes = []
        edges = []

        def add_node(action: str, conf: float, params: Optional[dict] = None) -> str:
            node_id = f"t{len(nodes) + 1}"
            nodes.append({"id": node_id, "action": action, "params": params or {}, "confidence": round(conf, 3)})
            return node_id
        
        # Skeletons
        if intent.label in ("navigate",):
            n1 = add_node("navigate", intent.confidence, {})
            # no edges needed
        elif intent.label in ("scan_environment",):
            add_node("scan", intent.confidence, {})
        elif intent.label in ("locate_object",):
            add_node("locate", intent.confidence, {})
        elif intent.label in ("retrieve_object", "multi_step_retrieve", "multi_step_manipulation"):
            n1 = add_node("navigate", max(intent.confidence - 0.1, 0.0), {})
            n2 = add_node("select_object", max(intent.confidence - 0.1, 0.0), {})
            n3 = add_node("pick_up", max(intent.confidence - 0.15, 0.0), {})
            n4 = add_node("deliver", max(intent.confidence - 0.15, 0.0), {})
            edges.extend([
                {"from": n1, "to": n2, "type": "sequence"},
                {"from": n2, "to": n3, "type": "sequence"},
                {"from": n3, "to": n4, "type": "sequence"},
            ])
        else:
            add_node("unknown", intent.confidence, {"note": "Intent not recognized"})

        return {"nodes": nodes, "edges": edges}

    def is_obvious_intent(self, verbs: List[str]) -> bool:
        v = set(verbs)
        return bool(v & (FETCH_VERBS | FIND_VERBS | SCAN_VERBS))

    def clarification_questions(self, text: str, intent: IntentResult, verbs: List[str], objects: List[Dict]) -> List[str]:
        questions = []

        clean = re.sub(r"[^\w\s]", "", text.lower())
        deictic = bool(re.search(r"\b(over )?there\b|\bthis\b|\bthat\b|\bthese\b|\bthose\b", clean))

        vague = any(o["head"] in ("something", "thing") for o in objects)
        possessive = any(re.search(r"\bmy\b", o["text"].lower()) for o in objects)
        explicit_object = any(o["head"] not in ("something", "thing") for o in objects)

        if deictic:
            questions.append("Can you specify what you are referring to when you say 'this/that/there'?")

        obvious = self.is_obvious_intent(verbs)
        
        hard_intent_amb = (intent.confidence < 0.10) or (intent.confidence < 0.18 and intent.delta < 0.01)
        if hard_intent_amb and (not obvious) and (not explicit_object or possessive):
            questions.append("Can you clarify what you want me to do (fetch, find, or scan)?")

        # Vague objects usually require clarification
        if vague:
            questions.append("What specific object should I look for (e.g., pen, notebook, bottle)?")

        # Possessive only matters if we likely need to SEARCH / LOCATE
        if possessive and intent.label in ("locate_object", "multi_step_retrieve"):
            questions.append("Can you describe what it looks like or where you last saw it?")
        
        return questions
    
    def interpret(self, text: str) -> Dict:
        verbs, objects = self.extract_verbs_objects(text)
        intent = self.classify_intent(text)
        task_graph = self.build_min_task_graph(intent, text)
        questions = self.clarification_questions(text, intent, verbs, objects)

        result = {
            "version": "1.0",
            "raw_command": text,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "language": "en",
            "intent": {
                "label": intent.label,
                "confidence": round(intent.confidence, 3),
                "second_confidence": round(intent.second_confidence, 3),
                "delta": round(intent.delta, 3),
                "is_ambiguous": intent.is_ambiguous,
                "ambiguity_reason": intent.ambiguity_reason,
            },
            "entities": {
                "verbs": list(dict.fromkeys(verbs)), # dedupe keep order
                "objects": objects,
                "locations": [], # reserved for later
            },
            "task_graph": task_graph,
            "clarification": {
                "required": len(questions) > 0,
                "questions": questions
            }
        }
        return result
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="*", help="Natural language command")
    args = parser.parse_args()

    cmd = " ".join(args.command).strip()
    if not cmd:
        cmd = input("Command: ").strip()

    engine = Phase1Interpreter()
    out = engine.interpret(cmd)
    print(json.dumps(out, indent=2))