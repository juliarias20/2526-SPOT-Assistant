import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from spacy.matcher import PhraseMatcher
import spacy
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
from perception import PerceptionModule

CONNECTORS = {"and", "then", "after", "before"}

FETCH_VERBS = {"get", "bring", "grab", "hand", "fetch", "give", "pick", "take"}
FIND_VERBS = {"find", "locate"}
SCAN_VERBS = {"scan", "inspect", "check", "look"}
PLACE_VERBS = {"put", "place", "set", "drop", "lay", "leave", "move"}

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
        bert_model_path: str = "models/bert-spot-intent",
        ambiguity_threshold: float = 0.70,
        delta_threshold: float = 0.10,
    ):
        self.nlp = spacy.load("en_core_web_sm")
        self.ambiguity_threshold = ambiguity_threshold
        self.delta_threshold = delta_threshold
        self.perception = PerceptionModule()

        # Load fine-tuned BERT classifier from local path (no internet needed)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = BertTokenizerFast.from_pretrained(bert_model_path, local_files_only=True)
        self._classifier = BertForSequenceClassification.from_pretrained(bert_model_path, local_files_only=True)
        self._classifier.to(self._device)
        self._classifier.eval()

        # Build id->label map from model config
        self._id2label = self._classifier.config.id2label  # {0: "retrieve_object", ...}

        with open(intent_file, "r", encoding="utf-8") as f:
            self.intent_labels: Dict[str, str] = json.load(f)

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
            "while": "temporal",
            "if": "condition",
            "unless": "condition",
            "so that": "purpose",
            "in order to": "purpose",
        }
        
    def split_clauses(self, text: str) -> List[Dict]:
        """
        Phase II clause splitter.

        Returns:
            List[{"order": int, "text": str, "connector": Optional[str], "relation": str}]
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

        # Deduplicate overlapping spans: keep earliest non-overlapping longest
        filtered = []
        last_end = -1
        for start, end, stext in spans:
            if start < last_end:
                continue
            filtered.append((start, end, stext))
            last_end = end

        split_points: List[int] = []
        connector_at: Dict[int, str] = {}

        # If a connector appears at the beginning (token 0), treat it as the connector
        # for the first clause instead of trying to split at 0.
        leading_connector = None
        leading_end = 0

        for start, end, stext in filtered:
            if start == 0:
                leading_connector = stext
                leading_end = end
                continue
            split_points.append(start)
            connector_at[start] = stext

        # 2) Strong punctuation splits (NOT every token)
        STRONG_PUNCT = {";", ":", ".", "!", "?"}
        for i, tok in enumerate(doc):
            if tok.text in STRONG_PUNCT and i + 1 < len(doc):
                split_points.append(i + 1)

        # 3) Comma split ONLY for introductory conditional/temporal clause: "If/When/Once/Unless/While ..., <main>"
        for i, tok in enumerate(doc):
            if tok.text == "," and i + 1 < len(doc):
                left = doc[:i].text.lower().strip()
                if left.startswith(("if ", "when ", "once ", "unless ", "while ")):
                    split_points.append(i + 1)
                # Also split "X, then Y" patterns
                right_tokens = [doc[j].lower_ for j in range(i + 1, min(i + 3, len(doc)))]
                if right_tokens and right_tokens[0] in ("then", "next", "finally"):
                    split_points.append(i + 1)
                
                # comma separating two verbal phrases (action list)
                left_has_verb = any(t.pos_ == "VERB" for t in doc[max(0, i - 4):i])
                right_has_verb = any(t.pos_ == "VERB" for t in doc[i + 1:min(len(doc), i + 5)])
                if left_has_verb and right_has_verb:
                    split_points.append(i + 1)

        # 4) "and" heuristic splits when likely joining actions
        # Note: also treat sentence-initial NOUN-tagged tokens whose lemma is a known
        # motion verb (e.g. "Head") as verbs for the purpose of this check.
        _MOTION_LEMMAS = {"head", "go", "walk", "navigate", "travel", "move", "proceed"}
        for i, tok in enumerate(doc):
            if tok.lower_ == "and":
                left_window = doc[max(0, i - 4):i]
                left_has_verb = any(t.pos_ == "VERB" for t in left_window) or (
                    i > 0 and doc[0].lemma_.lower() in _MOTION_LEMMAS
                )
                right_has_verb = any(t.pos_ == "VERB" for t in doc[i + 1:min(len(doc), i + 5)])
                if left_has_verb and right_has_verb:
                    split_points.append(i)
                    connector_at[i] = "and"

        # 5) Unique, sorted, in-bounds split points (ignore 0)
        split_points = sorted({p for p in split_points if 0 < p < len(doc)})

        # 6) Slice doc into segments
        segments = []
        start = 0
        prev_connector = leading_connector  # if "if/when/..." at start, first clause gets that label

        for sp in split_points:
            seg = doc[start:sp].text.strip()
            if seg:
                segments.append((seg, prev_connector))
            prev_connector = connector_at.get(sp, None)
            start = sp

        tail = doc[start:].text.strip()
        if tail:
            segments.append((tail, prev_connector))

        # 7) Clean leading connector text from segment
        def strip_leading_connector(seg_text: str, conn: Optional[str]) -> str:
            if not conn:
                return seg_text
            c = conn.lower().strip()
            s = seg_text.strip()
            if s.lower().startswith(c + " "):
                return s[len(c):].strip()
            return s

        out: List[Dict] = []
        for idx, (seg_text, conn) in enumerate(segments, start=1):
            cleaned = strip_leading_connector(seg_text, conn)
            cleaned = cleaned.strip()
            cleaned = re.sub(r"[,\.;:]+$", "", cleaned).strip()

            if conn is None:
                relation = "root"
            elif conn == "and":
                relation = "coordination"
            else:
                relation = self._connector_relation.get(conn, "unknown")

            out.append({
                "order": idx,
                "text": cleaned,
                "connector": conn,
                "relation": relation
            })

        return out
    
    def build_phase2_plan(self, clauses: List[Dict]) -> Dict:
        """
        Build a clause-level plan graph for Phase II. 
        - Each clause becomes a node (c1, c2, ...)
        - We classify intent per clause (same classifier as PHase I)
        - Add simple edges base do on clause relations
        """

        steps = []
        edges = []

        # Create nodes
        for c in clauses:
            c_intent = self.classify_intent(c["text"])
            step_type = "condition" if c["relation"] == "condition" else "action"
            
            steps.append({
                "id": f"c{c['order']}",
                "type": step_type,
                "text": c["text"],
                "connector": c["connector"],
                "relation": c["relation"],
                "intent": {
                    "label": c_intent.label,
                    "confidence": round(c_intent.confidence, 3),
                    "second_confidence": round(c_intent.second_confidence, 3),
                    "delta": round(c_intent.delta, 3),
                }
            })

        # Create edges between adjacent clauses
        for i in range (len(clauses) - 1):
            a = clauses[i]
            b = clauses[i + 1]
            from_id = f"c{a['order']}"
            to_id = f"c{b['order']}"

            if b ["relation"] == "condition":
                edges.append({"from": from_id, "to": to_id, "type": "condition_of"})
            elif b["relation"] in ("sequence", "coordination", "temporal", "purpose"):
                edges.append({"from": from_id, "to": to_id, "type": "sequence"})
            # root -> next: no edge required

        return {"steps": steps, "edges": edges}
    
    def extract_verbs_objects(self, text: str) -> Tuple[List[str], List[Dict]]:
        doc = self.nlp(text)
        verbs: List[str] = []
        objects: List[Dict] = []

        # Verbs that spaCy sometimes mistaggs as NOUN at sentence start
        MOTION_VERB_LEMMAS = {"head", "go", "walk", "navigate", "travel", "move", "proceed"}

        # Collect Verbs (lemmatized)
        for t in doc:
            if t.pos_ == "VERB":
                verbs.append(t.lemma_)
            # Also capture sentence-initial tokens whose lemma is a known motion verb
            # even if spaCy tagged them as NOUN (e.g. "Head" in "Head to the kitchen")
            elif t.i == 0 and t.lemma_.lower() in MOTION_VERB_LEMMAS:
                verbs.append(t.lemma_.lower())

        # Collect object-like noun chunks as candidates
        for chunk in doc.noun_chunks:
            head = chunk.root.lemma_
            # Skip noun chunks that are actually sentence-initial motion verbs
            if chunk.start == 0 and head.lower() in MOTION_VERB_LEMMAS:
                continue
            # Modifiers within the chunk span (e.g. "red cup" -> ["red"])
            modifiers = [t.text.lower() for t in chunk if t.dep_ in ("amod", "compound", "nummod")]
            # Also catch postpositive adjectives outside the chunk span
            # e.g. "something sharp" -- spaCy chunk = "something", "sharp" is amod on root
            modifiers += [
                t.text.lower() for t in doc
                if t.head == chunk.root
                and t.dep_ == "amod"
                and t not in chunk
            ]
            objects.append({
                "text": chunk.text,
                "head": head,
                "modifiers": modifiers,
                "role": "unknown"
            })

        return verbs, objects
        
    def classify_intent(self, text: str) -> IntentResult:
        # Pre-check: sentence-initial motion verbs that spaCy may tag as NOUN
        # (e.g. "Head to the kitchen..."). These are unambiguous regardless of length.
        NAV_FIRST_TOKENS = {"go", "head", "walk", "navigate", "travel", "move", "proceed"}
        first_token = text.strip().split()[0].lower().rstrip(".,!?") if text.strip() else ""
        if first_token in NAV_FIRST_TOKENS:
            return IntentResult("navigate", 0.90, 0.10, 0.80, False, None)

        # Verb - override for short clauses where embeddings are unreliable
        tokens = text.lower().split()
        if len(tokens) <= 8:
            first_verb = None
            doc = self.nlp(text)
            for t in doc:
                if t.pos_ == "VERB":
                    first_verb = t.lemma_
                    break
            if first_verb in FETCH_VERBS:
                return IntentResult("retrieve_object", 0.90, 0.10, 0.80, False, None)
            if first_verb in FIND_VERBS:
                return IntentResult("locate_object", 0.90, 0.10, 0.80, False, None)
            if first_verb in SCAN_VERBS:
                return IntentResult("scan_environment", 0.90, 0.10, 0.80, False, None)
            if first_verb in {"go", "move", "walk", "navigate", "travel", "head"}:
                return IntentResult("navigate", 0.90, 0.10, 0.80, False, None)
            if first_verb in PLACE_VERBS:
                return IntentResult("multi_step_manipulation", 0.90, 0.10, 0.80, False, None)

        # Fall through to fine-tuned BERT classifier for longer / ambiguous text
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self._classifier(**inputs).logits

        probs = torch.softmax(logits, dim=-1).squeeze()
        ranked = torch.argsort(probs, descending=True)

        best_id      = ranked[0].item()
        second_id    = ranked[1].item()
        best_score   = probs[best_id].item()
        second_score = probs[second_id].item()
        delta        = best_score - second_score

        best_label = self._id2label[best_id]

        is_ambiguous = False
        reason = None

        if best_score < self.ambiguity_threshold:
            is_ambiguous = True
            reason = f"Low confidence intent (prob = {best_score:.2f})."
        elif delta < self.delta_threshold:
            is_ambiguous = True
            reason = f"Top intents too close (Δ = {delta:.2f})."

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
            n3 = add_node("pick_up", max(intent.confidence - 0.15, 0.05), {})
            n4 = add_node("deliver", max(intent.confidence - 0.15, 0.05), {})
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
        if possessive and intent.label in ("locate_object", "multi_step_retrieve", "retrieve_object"):
            questions.append("Can you describe what it looks like or where you last saw it?")

        has_condition = any(c.get("relation") == "condition" for c in self.split_clauses(text))
        if has_condition and not explicit_object:
            questions.append("Should I search for object described before attempting to bring it?")
        
        return questions
    
    def interpret(self, text: str) -> Dict:
        verbs, objects = self.extract_verbs_objects(text)
        intent = self.classify_intent(text)
        task_graph = self.build_min_task_graph(intent, text)
        questions = self.clarification_questions(text, intent, verbs, objects)
        clauses = self.split_clauses(text)
        phase2_plan = self.build_phase2_plan(clauses)
        grounding = self.perception.ground_from_intent(
            intent_label = intent.label,
            verbs = verbs,
            objects = objects,
        )

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
            "clauses": clauses,
            "phase2": {
                "plan": phase2_plan
            },
            "task_graph": task_graph,
            "clarification": {
                "required": len(questions) > 0,
                "questions": questions
            },
            "grounding": grounding
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