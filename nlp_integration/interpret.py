import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import spacy
from sentence_transformers import SentenceTransformer, util

CONNECTORS = {"and", "then", "after", "before"}

@dataclass
class IntentResult:
    label: str
    confidence: float
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
        intent_file: str = "nlp_integration/models/intent_labels.json",
        embedding_model: str = "all-MiniLM-L6-v2",
        ambiguity_threshold: float = 0.55,
        delta_threshold: float = 0.08,
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
            modifiers = [t.text.lower() for t in chunk if t.dep_ in ("amod", "compund", "nummod")]
            role = "unkown"
            objects.append
            (
                {
                 "text": chunk.text,
                 "head": head,
                 "modifiers": modifiers,
                 "role": role
                }
            )

            # Simple location keywords
            locations = []
            for t in doc:
                if t.lemma_ in ("desk", "table", "kitchen", "room", "door", "counter"):
                    locations.append(t.lemma_)

            return verbs, objects
        
    # def classify_intent(self, text: str) -> IntentResult:
    #     text_emb = self.embedder.encode(text, convert_to_tensor=True)
    #     sims = {k: float(util.cos_sim(text_emb, v)) for k, v in self.intent_embs.items()}
    #     ranked = sorted(sims.items(), key=lambda x: x[1], reverse=True)
    #     best_label, best_score = ranked[0]
    #     second_score = ranked[1][1] if len(ranked) > 1 else 0.0

    #     is_ambiguous = False
    #     reason = None

    #     if best_score < self.ambiguity_threshold:
    #         is_ambiguous = True
    #         reason = f"Low confidence intent (score={best_score:.2f})."
    #     elif ()