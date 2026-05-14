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

print("\nVerifying Whisper model (faster-whisper base.en)...")
whisper_cache = Path(__file__).parent / "models" / "whisper"
whisper_cache.mkdir(parents=True, exist_ok=True)
try:
    from faster_whisper import WhisperModel
    WhisperModel(
        "base.en",
        device="cpu",
        compute_type="int8",
        download_root=str(whisper_cache),
    )
    print(f"  OK: {whisper_cache}")
except ImportError:
    print("  faster-whisper not installed — skipping (only needed for --voice).")
    print("  Install with: pip install faster-whisper sounddevice numpy")
except Exception as e:
    print(f"  WARNING: Whisper download failed: {e}")

# Also cache custom YOLO model if YOLO_MODEL env var points to one
custom_model = os.environ.get("YOLO_MODEL", "")
if custom_model and custom_model != "yolov8n.pt" and os.path.isfile(custom_model):
    print(f"\nCustom YOLO model found at {custom_model} — no download needed.")
elif custom_model and not os.path.isfile(custom_model):
    print(f"\nWARNING: Custom model '{custom_model}' not found at that path.")

print("\nAll models cached. You can now run with --offline.")