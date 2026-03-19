"""
generate_training_data.py
--------------------------
Generates a synthetic training dataset for fine-tuning BERT on SPOT robot
intent classification. Run this locally before uploading to Anvil.

Output: training_data.jsonl  (one JSON object per line)
        label_map.json        (label -> integer index)

Usage:
    python generate_training_data.py
    python generate_training_data.py --output my_data.jsonl --seed 42
"""

import json
import random
import argparse
from pathlib import Path

# ─────────────────────────────────────────────────────────────────
# RAW EXAMPLES  (command, intent_label)
# Target: ~300-500 per class for meaningful fine-tuning
# ─────────────────────────────────────────────────────────────────

EXAMPLES = {

    # ── retrieve_object ──────────────────────────────────────────
    "retrieve_object": [
        "Get me a pen.",
        "Bring me the red notebook.",
        "Hand me the charger next to the laptop.",
        "Get me the notebook.",
        "Bring me the bottle.",
        "Grab the stapler.",
        "Get the laptop charger.",
        "Bring the pen to me.",
        "Bring that over here.",
        "Pick up the one next to the chair.",
        "Bring me the usual thing.",
        "Fetch the water bottle.",
        "Can you grab that book on the shelf?",
        "Pass me the scissors.",
        "Hand me the folder on the desk.",
        "Could you bring me my coffee mug?",
        "Get that red marker for me.",
        "Bring over the tablet.",
        "Fetch me the keys from the counter.",
        "Hand me the remote control.",
        "Can you get my glasses?",
        "Retrieve the USB drive from the table.",
        "Bring me the nearest pen.",
        "Get me that thing right there.",
        "Pass me the stapler please.",
        "Bring the notebook here.",
        "I need the charger, can you get it?",
        "Go get me the blue folder.",
        "Bring me the package on the desk.",
        "Grab the bottle of water for me.",
        "Can you hand me that?",
        "Get me a drink.",
        "Bring something to write with.",
        "Get me something to eat.",
        "Bring me something soft.",
        "Get me something useful from the desk.",
        "Bring something healthy to eat.",
        "Get me something to cut with.",
        "Bring me something to clean with.",
        "Grab the item by the door.",
        "Hand over the clipboard.",
        "Can you pass me the binder?",
        "Bring me the bottle of water from the table.",
        "Fetch the document from the printer.",
        "Get me the file on the counter.",
        "Bring the white envelope here.",
        "Could you grab the phone charger?",
        "Can you get the notebook from there?",
        "Bring me the pencil case.",
        "Grab the ruler off the shelf.",
        "Hand me the tape.",
        "Pick up the glue and bring it over.",
        "Can you bring me that cup?",
        "Get the TV remote for me.",
        "Bring the package to me.",
        "Fetch the newspaper.",
        "Hand me the headphones.",
        "Grab me a snack.",
        "Get me the book on the table.",
        "Bring me that USB cable.",
        "Grab the controller and bring it here.",
        "Can you bring me a tissue?",
        "I need a pen, can you grab one?",
        "Fetch me the flashlight.",
        "Pass me the wrench.",
        "Bring me the screwdriver.",
        "Get me the notepad.",
        "Fetch the document from the shelf.",
        "Could you get me the tape measure?",
        "Bring me the safety glasses.",
        "Get me a bottle of water.",
        "Fetch the power cord.",
        "Hand me the folder labeled Q3.",
        "Bring me the thing I left on the couch.",
        "Can you retrieve the badge from the desk?",
        "Get me a pen from any drawer.",
        "Bring me whatever's on the table.",
    ],

    # ── locate_object ─────────────────────────────────────────────
    "locate_object": [
        "Find my backpack.",
        "Find the notebook.",
        "Locate the water bottle.",
        "Find my phone.",
        "Find my ID card.",
        "Find the thing I left on the table.",
        "Find my keys.",
        "Where is the charger?",
        "Can you find where I left my glasses?",
        "Look for my laptop.",
        "Search for the stapler.",
        "Do you see the TV remote anywhere?",
        "Can you find the scissors?",
        "Where did I put my wallet?",
        "Find the blue notebook.",
        "Search the room for my badge.",
        "Can you locate my headphones?",
        "Where's the folder I left here?",
        "Find the cable I left on the desk.",
        "Can you spot the sticky notes?",
        "Search for the flashlight.",
        "Where is my backpack?",
        "Find the USB drive.",
        "Can you locate the book I was reading?",
        "Look for my phone charger.",
        "Search for the white envelope.",
        "Find where the notebook is.",
        "Locate my badge.",
        "Can you find the box of staples?",
        "Find the screwdriver.",
        "Where's the tape?",
        "Search for my glasses.",
        "Find the thing I was looking for.",
        "Look for a pencil.",
        "Can you find where the mug is?",
        "Search for my keys.",
        "Find the magazine.",
        "Locate the device near the window.",
        "Find the item I lost earlier.",
        "Can you see my jacket anywhere?",
        "Where did the pen go?",
        "Find something I can use to write.",
        "Look for a container on the shelf.",
        "Can you find the charger for the tablet?",
        "Search for the book titled Design Patterns.",
        "Locate the nearest trash can.",
        "Find the emergency kit.",
        "Where is the first aid kit?",
        "Can you find where I left my coffee?",
        "Search the area for my lanyard.",
        "Find the object I was holding.",
        "Locate anything that looks like a pen.",
        "Where are my sunglasses?",
        "Find the clipboard.",
        "Search for the packet I brought.",
        "Where did I leave my phone?",
        "Find the notebook I need.",
        "Locate the tool I was using.",
    ],

    # ── scan_environment ──────────────────────────────────────────
    "scan_environment": [
        "Look around the room.",
        "Scan the room.",
        "Look around the desk area.",
        "Inspect the table.",
        "Check the shelf.",
        "Look around and tell me what you see.",
        "Survey the area.",
        "Scan the environment.",
        "Take a look around.",
        "Inspect the workspace.",
        "Check what's on the counter.",
        "Look at the area near the door.",
        "Scan for any objects nearby.",
        "Tell me what you see.",
        "Sweep the room.",
        "Do a visual scan of the area.",
        "Look around the lab.",
        "Inspect the hallway.",
        "Check the surroundings.",
        "Look around and describe what you see.",
        "Survey the workspace.",
        "Scan the kitchen area.",
        "Check the space near the window.",
        "Inspect the corner of the room.",
        "Look around the office.",
        "Scan the immediate area.",
        "Check everything on the table.",
        "Inspect what's around you.",
        "Look at the environment.",
        "Scan and report back.",
        "Look over there and tell me what's there.",
        "Do a quick scan.",
        "Inspect the area carefully.",
        "Check the perimeter.",
        "Look around for anything unusual.",
        "Scan the room and describe it.",
        "Check what's in this area.",
        "Inspect the items on the shelf.",
        "Look around and observe.",
        "Scan the vicinity.",
        "Take a look at the surroundings.",
        "Inspect your immediate vicinity.",
        "Look around for objects.",
        "Do a sweep of the room.",
        "Check the desk area.",
        "Survey the entire room.",
        "Scan the workspace for objects.",
        "Look at the table carefully.",
        "Inspect what you can see from here.",
        "Check the area around the couch.",
        "Scan the storage area.",
        "Look around the entrance.",
        "Inspect the layout of the room.",
        "Check the space in front of you.",
        "Survey the area near the bookshelf.",
    ],

    # ── navigate ──────────────────────────────────────────────────
    "navigate": [
        "Go to the door.",
        "Go to the desk.",
        "Go to the table.",
        "Navigate to the couch.",
        "Go to the bookshelf.",
        "Go over there.",
        "Head to the kitchen.",
        "Walk to the entrance.",
        "Go to the front of the room.",
        "Move to the hallway.",
        "Travel to the lab.",
        "Head over to the window.",
        "Go to that corner.",
        "Move towards the door.",
        "Walk over to the shelf.",
        "Navigate to the charging station.",
        "Go back to the starting point.",
        "Move to the other side of the room.",
        "Head to the storage area.",
        "Go to the workstation.",
        "Navigate to the couch area.",
        "Walk to the exit.",
        "Head to the table in the corner.",
        "Go to the counter.",
        "Move towards the window.",
        "Navigate back to base.",
        "Head over to the desk.",
        "Go to the conference room.",
        "Walk to the door on the left.",
        "Move to the right side of the room.",
        "Go over to the bookcase.",
        "Navigate to the white board.",
        "Head to the lab bench.",
        "Travel to station two.",
        "Go to the charging dock.",
        "Move forward.",
        "Go left.",
        "Turn right and walk forward.",
        "Head to the back of the room.",
        "Navigate to waypoint A.",
        "Go back to where you started.",
        "Move to the marked position.",
        "Walk to position three.",
        "Head to the reception area.",
        "Go to the nearest exit.",
        "Navigate to the starting location.",
        "Move to the designated area.",
        "Go to the front desk.",
        "Head to the center of the room.",
        "Walk toward the bookshelf.",
        "Navigate to the seating area.",
        "Go to the lab entrance.",
        "Head to station one.",
        "Move towards the couch.",
        "Go to the spot near the window.",
        "Navigate to the corner by the door.",
    ],

    # ── multi_step_retrieve ───────────────────────────────────────
    "multi_step_retrieve": [
        "Go to the table and bring the cup.",
        "Go to the desk and pick up the notebook.",
        "Go to the table, grab the bottle, then bring it to me.",
        "Find the charger and bring it to me.",
        "Go to the room and find the backpack.",
        "Go to the kitchen and bring me a drink.",
        "Go to the kitchen and find a snack.",
        "Go to the shelf and bring me a book.",
        "Head to the desk, pick up the pen, and bring it here.",
        "Go get the folder from the table and bring it to me.",
        "Go to the counter and bring me the keys.",
        "Head over to the desk and grab the charger for me.",
        "Go find the scissors and bring them here.",
        "Walk over to the shelf and get me the notebook.",
        "Go to the storage room and bring back the toolbox.",
        "Head to the kitchen and bring back a water bottle.",
        "Go to the closet and bring me my jacket.",
        "Navigate to the printer and bring the document.",
        "Go to the desk and retrieve the USB drive.",
        "Head to the bookshelf and bring me the manual.",
        "Go to the table and fetch the stapler.",
        "Walk to the cabinet and bring me the folder.",
        "Go grab the tablet from the desk and bring it here.",
        "Navigate to the counter and fetch me the pen.",
        "Go to the lab bench and bring the ruler.",
        "Head to the break room and get me a snack.",
        "Go retrieve the clipboard from the shelf.",
        "Walk to the back room and bring the box.",
        "Go to the window area and bring me what's there.",
        "Head over and pick up the nearest object for me.",
        "Go look for my phone and bring it back.",
        "Find the remote and bring it here.",
        "Go check the kitchen and bring something to drink.",
        "Navigate to the desk and pick up anything I left.",
        "Go to the table by the window and get the folder.",
        "Head to the storage area and get the power cord.",
        "Go find the item near the couch and bring it back.",
        "Head over to that corner and grab whatever is there.",
        "Go to the front and pick up the package for me.",
        "Navigate to the entrance and bring the badge.",
        "Walk to the charging station and bring the charger.",
        "Go look near the bookshelf and bring back a book.",
        "Go to the lab and bring back the safety glasses.",
        "Find the wrench and deliver it to me.",
        "Navigate to the break room and fetch a bottle of water.",
        "Head to the supply closet and bring back the scissors.",
        "Go get the notebook from the conference table.",
        "Walk to the kitchen and bring me the mug from the counter.",
        "Go to the locker and bring back my bag.",
        "Head to the equipment shelf and bring the flashlight.",
    ],

    # ── multi_step_manipulation ───────────────────────────────────
    "multi_step_manipulation": [
        "Pick up the bottle and put it on the desk.",
        "Pick up the bottle and place it near the door.",
        "Pick up the notebook and put it on the table.",
        "Grab the cup and set it on the desk.",
        "Pick up the charger and place it on the shelf.",
        "Move the backpack next to the chair.",
        "Pick up the pen and put it in the cup.",
        "Take the book and place it on the shelf.",
        "Move the bottle from the table to the counter.",
        "Grab the folder and put it in the drawer.",
        "Pick up the scissors and place them on the tray.",
        "Move the stapler to the other side of the desk.",
        "Take the notebook and put it away.",
        "Pick up that item and relocate it.",
        "Grab the cup and move it to the shelf.",
        "Take the phone and put it on the charger.",
        "Move the chair to the other corner.",
        "Pick up the box and place it by the door.",
        "Grab the backpack and set it on the table.",
        "Take the remote and put it on the couch.",
        "Move the toolbox to the workbench.",
        "Pick up the book and place it on the correct shelf.",
        "Grab the pen and put it in the holder.",
        "Move the laptop to the charging station.",
        "Take the folder and put it in the filing cabinet.",
        "Pick up the mug and place it in the kitchen.",
        "Grab the package and move it to the mail area.",
        "Take the clipboard and hang it on the wall.",
        "Move the item from the floor to the shelf.",
        "Pick it up and put it over there.",
        "Take the object and reposition it.",
        "Grab the tool and set it back in place.",
        "Move the notebook off the chair.",
        "Pick up the water bottle and put it in the fridge.",
        "Grab the USB drive and put it in the case.",
        "Take the tablet and place it on the stand.",
        "Move the binder to the bookshelf.",
        "Pick up the wrench and return it to the toolbox.",
        "Grab the cushion and put it back on the couch.",
        "Take the trash bag to the bin.",
        "Move the cables away from the walkway.",
        "Pick up the keyboard and put it in the storage bin.",
        "Grab the chair and move it to the other desk.",
        "Take the cup and place it in the dish rack.",
        "Move the box from the shelf to the floor.",
        "Pick up the book and put it on the cart.",
        "Grab the notebook and file it.",
        "Take the envelope and place it on the outbox.",
        "Move the plant to the window.",
        "Pick up the paper and put it in the recycling bin.",
    ],
}


# ─────────────────────────────────────────────────────────────────
# LABEL MAP
# ─────────────────────────────────────────────────────────────────

LABEL_MAP = {
    "retrieve_object":        0,
    "locate_object":          1,
    "scan_environment":       2,
    "navigate":               3,
    "multi_step_retrieve":    4,
    "multi_step_manipulation": 5,
}


def load_gold_examples(path: str = "commands_gold.jsonl") -> list:
    """Optionally seed with existing gold examples from the project."""
    records = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                intent = obj.get("gold_intent", "")
                if intent in LABEL_MAP:
                    records.append((obj["command"], intent))
        print(f"[INFO] Loaded {len(records)} examples from {path}")
    except FileNotFoundError:
        print(f"[WARN] {path} not found — skipping gold seed data.")
    return records


def augment(text: str) -> list:
    """
    Simple rule-based augmentation: lowercase, strip punctuation variant,
    add 'please' prefix/suffix, swap common synonyms.
    Returns a list of augmented variants (may be empty if nothing applies).
    """
    variants = []

    # lowercase / strip period
    variants.append(text.rstrip(".").lower())

    # add polite prefix
    variants.append("Please " + text[0].lower() + text[1:])
    variants.append("Can you " + text[0].lower() + text[1:].rstrip(".") + "?")

    # add urgency
    variants.append("Quickly " + text[0].lower() + text[1:])

    # swap common synonyms
    swaps = [
        ("Bring", "Carry"),
        ("Get", "Fetch"),
        ("Grab", "Pick up"),
        ("Find", "Search for"),
        ("Locate", "Look for"),
        ("Look around", "Scan"),
        ("Inspect", "Examine"),
        ("Navigate to", "Go to"),
        ("Head to", "Walk to"),
        ("Pick up", "Take"),
        ("Place", "Put"),
        ("Move", "Relocate"),
    ]
    for old, new in swaps:
        if text.startswith(old + " ") or text.startswith(old.lower() + " "):
            variants.append(text.replace(old, new, 1))
            variants.append(text.replace(old.lower(), new.lower(), 1))

    # deduplicate + remove identical to original
    seen = {text.strip()}
    unique = []
    for v in variants:
        v = v.strip()
        if v and v not in seen:
            seen.add(v)
            unique.append(v)
    return unique


def build_dataset(augment_data: bool = True, seed: int = 42) -> list:
    random.seed(seed)
    all_records = []

    # 1. Start with curated examples
    for intent, cmds in EXAMPLES.items():
        for cmd in cmds:
            all_records.append({"text": cmd, "label": LABEL_MAP[intent], "intent": intent})

    # 2. Merge gold JSONL if available
    for cmd, intent in load_gold_examples():
        all_records.append({"text": cmd, "label": LABEL_MAP[intent], "intent": intent})

    # 3. Augmentation pass
    if augment_data:
        augmented = []
        for rec in all_records:
            for variant in augment(rec["text"]):
                augmented.append({"text": variant, "label": rec["label"], "intent": rec["intent"]})
        all_records.extend(augmented)
        print(f"[INFO] After augmentation: {len(all_records)} total examples")

    # 4. Deduplicate by normalized text
    seen_texts = set()
    unique = []
    for rec in all_records:
        key = rec["text"].strip().lower()
        if key not in seen_texts:
            seen_texts.add(key)
            unique.append(rec)

    random.shuffle(unique)

    # Print class distribution
    from collections import Counter
    counts = Counter(r["intent"] for r in unique)
    print("\n[INFO] Class distribution:")
    for intent, count in sorted(counts.items()):
        print(f"  {intent:<30} {count:>4} examples")
    print(f"  {'TOTAL':<30} {len(unique):>4} examples\n")

    return unique


def main():
    parser = argparse.ArgumentParser(description="Generate SPOT intent training data")
    parser.add_argument("--output", default="training_data.jsonl", help="Output JSONL path")
    parser.add_argument("--label-map", default="label_map.json", help="Label map JSON path")
    parser.add_argument("--no-augment", action="store_true", help="Skip augmentation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = build_dataset(augment_data=not args.no_augment, seed=args.seed)

    # Write JSONL
    with open(args.output, "w", encoding="utf-8") as f:
        for rec in dataset:
            f.write(json.dumps(rec) + "\n")
    print(f"[OK] Wrote {len(dataset)} examples → {args.output}")

    # Write label map
    with open(args.label_map, "w", encoding="utf-8") as f:
        json.dump(LABEL_MAP, f, indent=2)
    print(f"[OK] Wrote label map → {args.label_map}")


if __name__ == "__main__":
    main()
