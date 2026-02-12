import spacy
import re
from typing import Dict

nlp = spacy.load("en_core_web_sm")

def rules_spacy_baseline(text: str) -> Dict[str, str]:
    doc = nlp(text.lower())
    verb = None
    for t in doc:
        if t.pos_ == "VERB":
            verb = t.lemma_
            break

    if verb in ("get", "bring", "fetch", "grab", "take", "hand"):
        return {"intent": "retrieve_object"}
    if verb in ("go", "move", "walk"):
        return {"intent": "navigate"}
    if verb in ("look", "scan", "inspect"):
        return {"intent": "scan_environment"}
    if verb in ("find", "locate", "search"):
        return {"intent": "locate_object"}
    return {"intent": "unkown"}