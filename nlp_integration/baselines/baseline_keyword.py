import re
from typing import Dict

INTENT_KEYWORDS = {
    "retrieve_object": ["get", "bring", "fetch", "grab", "pick up"],
    "navigate": ["go", "walk", "move"],
    "scan_environment": ["look", "scan", "inspect"],
    "locate_object": ["find", "locate", "search"]
}

def keyword_baseline(text: str) -> Dict[str, str]:
    t = text.lower()
    for intent, kws in INTENT_KEYWORDS.items():
        for kw in kws:
            if re.search(rf"\b{re.escape(kw)}\b", t):
                return {"intent": intent}
    return {"intent": "unkown"}