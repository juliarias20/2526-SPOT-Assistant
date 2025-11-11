import spacy
from sentence_transformers import SentenceTransformer, util

class SemanticTaskClassifier:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Task prompts to compare against
        self.task_descriptions = {
            "fetch": "Retrieve or pick up an object for the user",
            "go": "Move, walk, or go to a location",
            "heel": "Move, walk, or go to user",
            "search": "Look around or inspect the environment",
            "seek": "Find or identify the position of an object", 
        }

        # Precompute embeddings for efficiency
        self.task_embeddings = {
            task: self.model.encode(desc, convert_to_tensor=True)
            for task, desc in self.task_descriptions.items()
        }

    def extract_verb_and_object(self, text):
        doc = self.nlp(text)
        verb = None
        obj = None
        for token in doc:
            if token.pos_ == "VERB" and verb is None:
                verb = token.lemma_
            if token.dep_ in ("dobj", "pobj", "nsubjpass") and obj is None:
                obj = token.text
        return verb, obj
    
    def classify_task(self, verb):
        if not verb:
            return "unkown"
        verb_embedding = self.model.encode(verb, convert_to_tensor=True)
        similarities = {
            task: util.cos_sim(verb_embedding, task_emb).item()
            for task, task_emb in self.task_embeddings.items()
        }
        best_task = max(similarities, key=similarities.get)
        return best_task
    
    def parse_command(self, text):
        verb, obj = self.extract_verb_and_object(text)
        task = self.classify_task(verb)
        return {
            "raw_command": text,
            "verb": verb,
            "object": obj, 
            "task": task
        }