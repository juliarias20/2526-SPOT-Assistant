"""
perception.py
-------------
Phase III: Contectual Reaonsing and Perceptual Grounding

Responsibilities:
    1. Acquire an RGB frame -- from SPOT's front camera or a mock scene.
    2. Run YOLOv8 object detection to get candidate objects + bounding boxes.
    3. Score each candidate against the command's verb/intent using
        SentenceTransformer cosing similarity (affordance-based matching).
    3. Return a ranked list of grounded object candidates.

Integration with interpret.py:
    from perception import PerceptionModule
    perception = PerceptionModule()
    result = perception.ground(verb = "write", candidates = None)   # uses live camera
    result = perception.ground(verb = "write", candidates = mock_scene) # mock mode

SPOT SDK note:
    Set USE_SPOT = True and provide SPOT_IP to use live camera feed.
    Falls back to mock scene automatically if SDK is unavailable.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer, util

# ── YOLOv8 import (ultralytics) ──────────────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[perception] ultralytics not installed -- detection will use mock objects.")

# ── SPOT SDK import ───────────────────────────────────────────────────────────
try:
    import bosdyn.client
    import bosdyn.client.util
    from bosdyn.client.image import ImageClient
    SPOT_AVAILABLE = True
except ImportError:
    SPOT_AVAILABLE = False

# ── Configuration ─────────────────────────────────────────────────────────────
USE_SPOT: bool = os.environ.get("USE_SPOT", "false").lower() == "true"
SPOT_IP: str = os.environ.get("SPOT_IP", "192.168.80.3")
SPOT_USER: str = os.environ.get("SPOT_USER", "user")
SPOT_PASS: str = os.environ.get("SPOT_PASS", "password")

YOLO_MODEL_PATH: str = os.environ.get("YOLO_MODEL", "yolov8n.pt") # nano for speed
CAMERA_SOURCE: str = os.environ.get("CAMERA_SOURCE", "frontleft_fisheye_image")

# Minimum YOLO detection confidence to include a candidate
DETECTION_CONF_THRESHOLD: float = 0.30

# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class DetectedObject:
    """
    A single object detected by YOLOv8.
    bbox: [x1, y1, x2, y2] in pixels.
    """
    label:      str
    confidence: float
    bbox:       List[float] = field(default_factory = list)
    distance_m: Optional[float] = None  # populated if depth data available

@dataclass
class GroundedObject:
    """
    A DetectedObject that has been scored against a language query.
    affordance_score: cosine similarity between verb embedding and label embedding.
    combined_score: weighted combination of detection confidence + affordance score.
    """
    label:              str
    detection_conf:     float
    affordance_score:   float
    combined_score:     float
    bbox:               List[float] = field(default_factory = List)
    distance_m:         Optional[float] = None

# ── Mock scene (used when SPOT or YOLO unavailable) ───────────────────────────

DEFAULT_MOCK_SCENE: List[DetectedObject] = [
    DetectedObject("pen",        0.92, [100, 200, 130, 220]),
    DetectedObject("notebook",   0.88, [150, 200, 220, 260]),
    DetectedObject("cup",        0.85, [300, 180, 360, 240]),
    DetectedObject("bottle",     0.83, [400, 170, 440, 250]),
    DetectedObject("laptop",     0.90, [500, 150, 680, 300]),
    DetectedObject("stapler",    0.78, [200, 300, 240, 330]),
    DetectedObject("scissors",   0.81, [250, 310, 290, 340]),
    DetectedObject("phone",      0.87, [350, 290, 390, 330]),
    DetectedObject("backpack",   0.75, [600, 350, 720, 500]),
    DetectedObject("charger",    0.79, [460, 280, 510, 310]),
]

# ── Main module ───────────────────────────────────────────────────────────────

class PerceptionModule:
    """
    Phase III perception and affordance grounding.

    Usage (live SPOT):
        export USE_SPOT = true SPOT_IP = 192.168.80.3
        p = PerceptionModule()
        results = p.ground(verb = "write")

    Usage (mock / offline):
        p = PerceptionModule()
        results = p.ground(verb = "write", candidates = mock_scene)
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        detection_conf: float = DETECTION_CONF_THRESHOLD,
        affordance_weight: float = 0.6,
        detection_weight: float = 0.4,    
    ):
        """
        Args:
            embedding_model:    SentenceTransformer model name (reuse from interpret.py)
            detection_conf:     Minimum YOLO confidence to keep a detection.
            affordance_weight:  Weight of semantic similarity in combined score.
            detection_weight:   Weight of YOLO detection confidence in combined score.
        """
        self.detection_conf     = detection_conf
        self.affordance_weight  = affordance_weight
        self.detection_weight   = detection_weight

        # Share embedder with interpret.py if possible; otherwise load own copy
        self.embedder = SentenceTransformer(embedding_model)

        # YOLOv8
        self._yolo: Optional[object] = None
        if YOLO_AVAILABLE:
            try:
                self._yolo = YOLO(YOLO_MODEL_PATH)
                print(f"[perception] YOLOv8 loaded: {YOLO_MODEL_PATH}")
            except Exception as e:
                print(f"[perception] YOLOv8 load failed ({e}) -- using mock detection.")

        # SPOT image client
        self._spot_client: Optional[object] = None
        if USE_SPOT and SPOT_AVAILABLE:
            self._spot_client = self._init_spot_client()

    # ── SPOT initialisation ───────────────────────────────────────────────────

    def _init_spot_client(self) -> Optional[object]:
        try:
            sdk = bosdyn.client.create_standard_sdk("PerceptionModule")
            robot = sdk.create_robot(SPOT_IP)
            bosdyn.client.util.authenticate(robot)
            client = robot.ensure_client(ImageClient.default_service_name)
            print(f"[perception] SPOT image client connected: {SPOT_IP}")
            return client
        except Exception as e:
            print(f"[perception] SPOT connection failed ({e}) -- falling back to mock.")
            return None
        
    # ── Frame acquisition ─────────────────────────────────────────────────────

    def _get_frame_from_spot(self) -> Optional[np.ndarray]:
        """
        Grab one RGB frame from SPOT's front camera.
        """
        if self._spot_client is None:
            return None
        try:
            from bosdyn.api import image_pb2
            import cv2

            responses = self._spot_client.get_image_from_sources([CAMERA_SOURCE])
            if not responses:
                return None
            
            resp = responses[0]
            if resp.shot.image.pixel_form == image_pb2.Image.PIXEL_FORMAT_RGB_U8:
                dtype = np.uint8
                nchannels = 3
            else:
                dtype = np.uint8
                nchannels = 1

            img_array = np.frombuffer(resp.shot.image.data, dtype = dtype)
            if nchannels == 3:
                frame = img_array.reshape(
                    resp.shot.image.rows,
                    resp.shot.image.cols,
                    nchannels
                )
            else:
                frame = cv2. imdecode(img_array, cv2.IMREAD_COLOR)
            return frame
        except Exception as e:
            print(f"[perception] Frame grab failed ({e})")
            return None
        
    # ── YOLOv8 detection ──────────────────────────────────────────────────────

    def _detect_objects(self, frame: np.ndarray) -> List[DetectedObject]:
        """
        Run YOLOv8 on a frame and return filtered detections.
        """
        if self._yolo is None:
            return []
        
        results = self._yolo(frame, verbose = False)
        detected: List[DetectedObject] = []

        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < self.detection_conf:
                    continue
                cls_id = int(box.cls[0])
                label = self._yolo.model.names[cls_id]
                bbox = box.xyxy[0].tolist()
                detected.append(DetectedObject(label = label, confidence = conf, bbox = bbox))

        return detected
    
    # ── Affordance scoring ────────────────────────────────────────────────────

    def _score_affordance(
        self,
        verb: str,
        candidates: List[DetectedObject],
    ) -> List[GroundedObject]:
        """
        Score each candidate object against the action verb using cosine similarity.
        Higher score = object is more likely to satsify the verb's functional intent.

        Example: verb = "write" -> pen > marker > notebook > cup
        """
        if not candidates:
            return []
        
        verb_emb = self.embedder.encode(verb, convert_to_tensor = True)
        label_embs = self.embedder.encode(
            [c.label for c in candidates], convert_to_tensor = True
        )
        # cosin similarity: shape (n_candidates,)
        sims = util.cos_sim(verb_emb, label_embs)[0].tolist()

        grounded: List[GroundedObject] = []
        for obj, aff_score in zip(candidates, sims):
            combined = (
                self.affordance_weight * float(aff_score) +
                self.detection_weight * obj.confidence
            )
            grounded.append(GroundedObject(
                label               = obj.label,
                detection_conf      = obj.confidence,
                affordance_score    = round(float(aff_score), 4),
                combined_score      = round(combined, 4),
                bbox                = obj.bbox,
                distance_m          = obj.distance_m,
            ))

        return sorted(grounded, key = lambda x: x.combined_score, reverse = True)
    
    # ── Public interface ──────────────────────────────────────────────────────

    def get_scene_objects(
        self,
        candidates: Optional[List[DetectedObject]] = None,
    ) -> List[DetectedObject]:
        """
        Return current visible objects.
        Priority: explicit candidates -> SPOT camera -> mock scene.
        """
        if candidates is not None:
            return candidates
        
        if self._spot_client is not None:
            frame = self._get_frame_from_spot()
            if frame is not None and self._yolo is not None:
                detected = self._detect_objects(frame)
                if detected:
                    return detected
                print("[perception] No detections above threshold -- using mock scene.")

        return DEFAULT_MOCK_SCENE
    
    def ground(
        self,
        verb: str,
        candidates: Optional[List[DetectedObject]] = None,
        top_k: int = 3,
    ) -> Dict:
        """
        Main grounding entry point.

        Args:
            verb:       Action verb from the parsed command (e.g. "write", "drink", "cut").
            candidates: Optional explicit object list (mock or pre-detected).
                        If None, acquires from SPOT or falls back to mack scene.
            top_k:      Number of top candidates to return.

        Returns:
            {
                "verb": str,
                "scene_objects": [str, ...],
                "top_candidates": [
                    {
                        "label": str,
                        "affordance_score": float,
                        "detection_conf": float,
                        "combined_score": float,
                        "bbox": [...],
                    },
                    ...
                ],
                "best_match": str | None,
                "source": "spot" | "mock"
            }
        """
        scene = self.get_scene_objects(candidates)
        source = "mock" if (candidates is None and self._spot_client is None) else "spot"

        ranked = self._score_affordance(verb, scene)
        top = ranked[:top_k]

        return {
            "verb":             verb,
            "scene_objects":    [o.label for o in scene],
            "top_candidates": [
                {
                    "label":            g.label,
                    "affordance_score": g.affordance_score,
                    "detection_conf":   g.detection_conf,
                    "combined_score":   g.combined_score,
                    "bbox":             g.bbox,
                }
                for g in top
            ],
            "best_match":       top[0].label if top else None,
            "source":           source,
        }
    
    def ground_from_intent(
        self,
        intent_label: str,
        verbs: List[str],
        objects: List[Dict],
        candidates: Optional[List[DetectedObject]] = None,
    ) -> Optional[Dict]:
        """
        Called directly by interpret.py when select_object is needed.

        Determines the grounding verb from:
            1. The first action verb extracted from the command
            2. Falls back to the intent label if no useful verb found

        Returns grounding result dict, or None if grounding not required.
        """
        # Only ground when object selection is genuinely needed
        GROUNDING_INTENTS = {"retrieve_object", "multi_step_retrieve"}
        if intent_label not in GROUNDING_INTENTS:
            return None
        
        # Is the object vague? (head is "something" or "thing")
        has_vague = any(o.get("head") in ("something", "thing") for o in objects)
        if not has_vague:
            return None
        
        # Pick the best verb to ground against
        ACTION_VERBS = {"bring", "get", "fetch", "grab", "hand", "give", "pick", "take"}
        grounding_verb = None
        for v in verbs:
            if v.lower() not in ACTION_VERBS:
                grounding_verb = v.lower()
                break
        if grounding_verb is None:
            grounding_verb = intent_label.replace("_", " ")

        return self.ground(verb = grounding_verb, candidates = candidates)
    
# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    p = PerceptionModule()

    test_verbs = ["write", "drink", "cut", "clean", "scan", "carry"]
    for verb in test_verbs:
        result = p.ground(verb = verb)
        print(f"\nVerb: '{verb}'")
        print(f" Best Match : {result['best_match']}")
        for c in result["top_candidates"]:
            print(
                f" {c['label']:<15} "
                f"aff={c['affordance_score']:.3f} "
                f"det={c['detection_conf']:.3f} "
                f"combined={c['combined_score']:.3f}"
            )