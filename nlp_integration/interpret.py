import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from spacy.matcher import PhraseMatcher
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

    def _init_clause_matcher(self):
        """
        Initialize a PhraseMatcher to identify clause connectors for splitting.
        """
        if hasattr(self, "_clause_matcher") and self._clause_matcher is not None:
            return
        
        self._clause_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")

        #Multi-word connectors (order matters; longer phrases should be detected)
        phrases = [
            "and then",
            "after that",
            "then",
            "next",
            "finally",
            "followed by",
            "so that",
            "in order to",
            "before",
            "after",
            "once",
            "when",
            "while",
            "if",
            "unless",
        ]

        patterns = [self.nlp.make_doc(p) for p in phrases]
        self._clause_matcher.add("CONNECTOR_PHRASE", patterns)

        # Map Phrase -> relation label
        self._connector_relation = {
            "and then": "sequence",
            "after that": "sequence",
            "then": "sequence",
            "next": "sequence",
            "finally": "sequence",
            "followed by": "sequence",
            "before": "temporal",
            "after": "temporal",
            "once": "temporal",
            "when": "temporal",
            "while": "temporal:",
            "if": "condition",
            "unless": "condition",
            "so that": "purpose",
            "in order to": "purpose",
        }
        
    def split_clauses(self, text: str) -> List[str]:
        """
        Phase II clause splitter.

        Returns:
            Lis[{"order": int, "text": str, "connector": Optional[str], "relation": str}]

        Notes:
        - Uses PhraseMatcher for multiword connectors.
        - Also splits on ';' and strong punctuation boundaries.
        - Also attempts to split on 'and' when it likely joins two verbs/actions.
        """
        self._init_clause_matcher()
        doc = self.nlp(text)

        # 1) Gather connector spans from PhraseMatcher
        matches = self._clause_matcher(doc)
        spans = []
        for _, start, end in matches:
            span_text = doc[start:end].text.lower().strip()
            spans.append((start, end, span_text))
        spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))

        # Deduplicate overlapping spans: keep longest when overlaps
        filtered = []
        last_end = -1
        for start, end, stext in spans: 
            if start < last_end:
                continue
            filtered.append((start, end, stext))
            last_end = end

        # 2) Build split points (token indices) at connector starts
        split_points = []
        connector_at = {} # split_index -> connector phrase
        for start, end, stext in filtered:
            split_points.append(start)
            connector_at[start] = stext

        # 3) Add punctuation-based split points
        for i, tok in enumerate(doc):
            # split *after* this token
            if i + 1 < len(doc):
                split_points.append(i + 1)

        # 4) Add "and" heuristic splits when likely joining actions
        # Heuristic: token "and" where left and right both have verb nearby
        for i, tok in enumerate(doc):
            if tok.lower_ == "and":
                left_has_verb = any(t.pos_ == "VERB" for t in doc[max(0, i-4):i])
                right_has_verb = any(t.pos_ == "VERB" for t in doc[i+1:min(len(doc), i+5)])
                if left_has_verb and right_has_verb:
                    split_points.append(i) #split at 'and'
                    connector_at[i] = "and"

        # 5) Create sorted unique split points in bounds (ignore 0)
        split_points = sorted({p for p in split_points if 0 < p < len(doc)})

        # 6) Slice doc into segments
        segments = []
        start = 0
        prev_connector = None

        for sp in split_points:
            seg = doc[start:sp].text.strip()
            if seg:
                segments.append((seg, prev_connector))
            
    
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