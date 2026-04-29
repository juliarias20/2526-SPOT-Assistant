"""
cache_models.py
---------------
Run this once while online to download and cache all model weights locally.
After this, the system runs fully offline via --offline flag or OFFLINE_MODE=true.

Usage:
    python cache_models.py
"""
import os
from pathlib import Path

CACHE_DIR = Path(__file__).parent / "models" / "sentence_transformers"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

print("Caching SentenceTransformer (all-MiniLM-L6-v2)...")
from sentence_transformers import SentenceTransformer
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", cache_folder=str(CACHE_DIR))
print(f"  Saved to: {CACHE_DIR}")

print("Verifying spaCy en_core_web_sm...")
import spacy
try:
    spacy.load("en_core_web_sm")
    print("  OK (already installed)")
except OSError:
    print("  Not found — running: python -m spacy download en_core_web_sm")
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)

print("Verifying BERT model weights...")
bert_path = Path(__file__).parent / "models" / "bert-spot-intent"
if not bert_path.exists():
    print(f"  WARNING: {bert_path} not found — run finetune_bert.py first")
else:
    from transformers import BertTokenizerFast, BertForSequenceClassification
    BertTokenizerFast.from_pretrained(str(bert_path))
    BertForSequenceClassification.from_pretrained(str(bert_path))
    print(f"  OK: {bert_path}")

print("Verifying YOLOv8 weights...")
yolo_path = os.environ.get("YOLO_MODEL", "yolov8n.pt")
if os.path.isfile(yolo_path):
    print(f"  OK: {yolo_path}")
else:
    print(f"  '{yolo_path}' not found locally — downloading via ultralytics...")
    from ultralytics import YOLO
    YOLO(yolo_path)
    print(f"  Cached by ultralytics to default location")

print("\nAll models cached. You can now run with --offline.")