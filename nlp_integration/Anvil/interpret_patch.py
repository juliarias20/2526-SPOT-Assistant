"""
interpret_patch.py
-------------------
Shows the changes needed in interpret.py to swap the fine-tuned
BertForSequenceClassification model in for the current
SentenceTransformer cosine-similarity approach.

This is NOT a standalone script — copy the relevant sections into
interpret.py after you have the trained model in:
    ./models/bert-spot-intent/

CHANGES SUMMARY:
  1. Replace SentenceTransformer import with transformers imports
  2. New __init__: load BertTokenizerFast + BertForSequenceClassification
     from local path instead of downloading
  3. Replace classify_intent() embedding path with classifier forward pass
  4. Keep the short-clause verb-override logic exactly as-is (it still works)
"""

# ─────────────────────────────────────────────────────────────────
# 1.  NEW IMPORTS  (replace the sentence_transformers import)
# ─────────────────────────────────────────────────────────────────

# REMOVE:
#   from sentence_transformers import SentenceTransformer, util

# ADD:
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification

# ─────────────────────────────────────────────────────────────────
# 2.  NEW __init__  (replace inside Phase1Interpreter)
# ─────────────────────────────────────────────────────────────────

# REMOVE:
#   self.embedder = SentenceTransformer(embedding_model)
#   self.intent_embs = {k: self.embedder.encode(v, ...) for k, v in ...}

# ADD (replace the old __init__ signature + body with this):

BERT_MODEL_PATH = "models/bert-spot-intent"   # relative to project root

def __init__(
    self,
    intent_file: str = "models/intent_labels.json",
    bert_model_path: str = BERT_MODEL_PATH,
    ambiguity_threshold: float = 0.70,  # now represents min softmax probability
    delta_threshold: float = 0.10,      # min gap between top-2 softmax probs
):
    self.nlp = spacy.load("en_core_web_sm")
    self.ambiguity_threshold = ambiguity_threshold
    self.delta_threshold = delta_threshold
    self.perception = PerceptionModule()

    # Load fine-tuned classifier from local path (no internet needed)
    self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self._tokenizer = BertTokenizerFast.from_pretrained(bert_model_path)
    self._classifier = BertForSequenceClassification.from_pretrained(bert_model_path)
    self._classifier.to(self._device)
    self._classifier.eval()

    # Build id->label map from model config
    self._id2label = self._classifier.config.id2label  # {0: "retrieve_object", ...}
    self._label2id = self._classifier.config.label2id

    with open(intent_file, "r", encoding="utf-8") as f:
        self.intent_labels: dict = json.load(f)


# ─────────────────────────────────────────────────────────────────
# 3.  NEW classify_intent()  (replace the embedding path only;
#     keep the short-clause verb-override block exactly as-is)
# ─────────────────────────────────────────────────────────────────

def classify_intent(self, text: str) -> "IntentResult":
    # ── Keep existing verb-override for short clauses ─────────────
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

    # ── Fine-tuned BERT classifier ────────────────────────────────
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

    probs = torch.softmax(logits, dim=-1).squeeze()  # shape: (num_labels,)
    ranked = torch.argsort(probs, descending=True)

    best_id     = ranked[0].item()
    second_id   = ranked[1].item()
    best_score  = probs[best_id].item()
    second_score = probs[second_id].item()
    delta = best_score - second_score

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


# ─────────────────────────────────────────────────────────────────
# NOTE ON THRESHOLD VALUES
# ─────────────────────────────────────────────────────────────────
#
# Old thresholds were cosine similarity scores (typically 0.10 - 0.30).
# New thresholds are softmax probabilities (0.0 - 1.0).
#
#   ambiguity_threshold = 0.70  →  flag ambiguous if top class < 70% confident
#   delta_threshold     = 0.10  →  flag ambiguous if top-2 gap < 10%
#
# These are reasonable starting values. Tune after running evaluate_phase1.py
# with the new model and checking which examples are newly flagged.
#
# ─────────────────────────────────────────────────────────────────
# NOTE ON EMBEDDER REMOVAL
# ─────────────────────────────────────────────────────────────────
#
# The old code passed self.embedder into PerceptionModule to avoid
# double-loading BERT. Since we're now using a different model class
# (BertForSequenceClassification vs SentenceTransformer), check whether
# perception.py still needs an embedder. If so, pass self._tokenizer
# and self._classifier, or keep a separate SentenceTransformer instance
# only in PerceptionModule.
#
# ─────────────────────────────────────────────────────────────────
