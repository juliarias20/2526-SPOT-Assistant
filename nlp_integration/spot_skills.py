"""
spot_skills.py
--------------
Phase IV: Robot-Agnostic Skill Primitives for SPOT

Each function wraps a Boston Dynamis SDK call and returns a 
structured SkillResult. The executor (executor.py) calls these
functions by name, mapping task graph nodes to real robot actions.

Skill registry:
    navigate(waypoint_id, ...)      -> travel to a named GraphNav waypoint
    scan(...)                       -> rotate body and capture scene description
    locate(object_label, ...)       -> search visible scene for a named object
    pick_up(object_label, ...)      -> arm grasp using Spot Arm API
    deliver(...)                    -> extend arm to offer object to user
    release(...)                    -> open gripper to release held object

Environment variables (set before running):
    $env:CAMERA_SOURCE = "frontleft_fisheye_image"   # default
    $env:SPOT_START_WAYPOINT = "your-start-uuid-here"
    $env:USE_SPOT    = "true"
    $env:SPOT_IP     = "192.168.80.3"     # your SPOT's IP
    $env:SPOT_USER   = "user"
    $env:SPOT_PASS   = "yourpassword"
    $env:SPOT_MAP_PATH = "maps/trial_space"
"""

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ── SPOT SDK imports ──────────────────────────────────────────────────────────
try:
    import bosdyn.client
    import bosdyn.client.util
    import bosdyn.client.lease
    import bosdyn.client.robot_command
    import bosdyn.client.robot_state
    from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
    from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
    from bosdyn.client.robot_state import RobotStateClient
    from bosdyn.client import math_helpers
    from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
    from bosdyn.api import geometry_pb2, trajectory_pb2
    from bosdyn.client.frame_helpers import VISION_FRAME_NAME, BODY_FRAME_NAME

    # GraphNav
    from bosdyn.client.graph_nav import GraphNavClient
    from bosdyn.api.graph_nav import graph_nav_pb2, nav_pb2

    # Arm / manipulation
    from bosdyn.client.manipulation_api_client import ManipulationApiClient
    from bosdyn.api import manipulation_api_pb2

    SPOT_SDK_AVAILABLE = True
except ImportError:
    SPOT_SDK_AVAILABLE = False

# ── Configuration ─────────────────────────────────────────────────────────────
USE_SPOT: bool = os.environ.get("USE_SPOT", "false").lower() == "true"
SPOT_IP: str = os.environ.get("SPOT_IP", "192.168.80.3")
SPOT_USER: str = os.environ.get("SPOT_USER", "user")
SPOT_PASS: str = os.environ.get("SPOT_PASS", "password")

# Navigation
NAVIGATE_TIMEOUT_SEC: float = 30.0
NAVIGATE_POLL_SEC: float = 0.5

# Arm
ARM_READY_TIMEOUT_SEC: float = 5.0
GRASP_TIMEOUT_SEC: float = 15.0
DELIVER_TIMEOUT_SEC: float = 8.0

# ── GraphNav map ─────────────────────────────────────────────────────────────
# Path to the recorded GraphNav map directory (output of record_map.py).
# Override with the SPOT_MAP_PATH environment variable.
MAP_PATH: str = os.environ.get("SPOT_MAP_PATH", "maps/trial_space")

# Human-readable waypoint name -> GraphNav UUID.
# Populate this after running record_map.py — copy the printed WAYPOINT_MAP
# dict here. Navigate() resolves names through this table before calling SDK.
WAYPOINT_MAP: dict = {
    # "desk":    "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    # "table":   "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    # "kitchen": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    # "user":    "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
}

# Starting waypoint for no-fiducial localization fallback.
# Set this to the UUID of the waypoint SPOT starts at when no AprilTag
# is visible. Copy the UUID from record_map.py output for your start position.
# Override with the SPOT_START_WAYPOINT environment variable.
# Leave empty ("") to skip the fallback and rely on fiducial-only localization.
START_WAYPOINT: str = os.environ.get("SPOT_START_WAYPOINT", "")

# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class SkillResult:
    """
    Returned by every skill function.
    success:        True if skill completed without error.
    skill:          Name of the skill that produced this result.
    message:        Human-readable status or error decription.
    data:           Optional payload (e.g. detected objects, waypoint reached).
    mock:           True if this rsult came from dry-run mode.
    """
    success:        bool
    skill:          str
    message:        str
    data:           Dict[str, Any] = field(default_factory = dict)
    mock:           bool = False

# ── Perception singleton ─────────────────────────────────────────────────────
# PerceptionModule is expensive to construct (loads BertModel + YOLOv8).
# A single module-level instance is shared across all locate() calls for the
# lifetime of a live trial session. The embedder can optionally be pre-loaded
# by Phase1Interpreter and passed in on first use to avoid a redundant load.
_perception = None

def _get_perception(embedder=None):
    """Return the module-level PerceptionModule singleton, creating it if needed.

    Pass embedder on the first call to share the SentenceTransformer instance
    already loaded by Phase1Interpreter. Subsequent calls return the cached
    instance regardless of the embedder arg.
    """
    global _perception
    if _perception is None:
        from perception import PerceptionModule
        _perception = PerceptionModule(embedder=embedder)
    return _perception


# ── SPOT robot singleton ──────────────────────────────────────────────────────
class SpotRobot:
    """
    Manages a single authenticated SPOT connection and exposes
    the SK clients needed by each skill.

    Usage:
        robot = SpotRobot()
        robot.connect()
        ...
        robot.disconnect()

    or as a context manager:
        with SpotRobot() as robot:
            navigate(robot, "my_waypoint")
    """

    def __init__(self):
        self.robot              = None
        self.lease_client       = None
        self.lease              = None
        self.lease_ka           = None
        self.command_client     = None
        self.state_client       = None
        self.graph_nav_client   = None
        self.manip_client       = None
        self._connected         = False

    def connect(self) -> bool:
        if not SPOT_SDK_AVAILABLE:
            print("[spot] SDK not installed -- running in mock mode.")
            return False
        if not USE_SPOT:
            print("[spot] USE_SPOT = false - running in mock mode.")
            return False
        try:
            sdk = bosdyn.client.create_standard_sdk("SpotSkills")
            self.robot = sdk.create_robot(SPOT_IP)
            bosdyn.client.util.authenticate(self.robot)
            self.robot.time_sync.wait_for_sync()

            self.lease_client = self.robot.ensure_client(LeaseClient.default_service_name)
            self.command_client = self.robot.ensure_client(RobotCommandClient.default_service_name)
            self.state_client = self.robot.ensure_client(RobotStateClient.default_service_name)
            self.graph_nav_client = self.robot.ensure_client(GraphNavClient.default_service_name)
            self.manip_client = self.robot.ensure_client(ManipulationApiClient.default_service_name)

            self.lease = self.lease_client.acquire()
            self.lease_ka = LeaseKeepAlive(self.lease_client, must_acquire = True, return_at_exit = True)

            # Power on and stand
            self.robot.power_on(timeout_sec=20)
            assert self.robot.is_powered_on(), "SPOT failed to power on."
            cmd = RobotCommandBuilder.synchro_stand_command()
            self.command_client.robot_command(cmd)
            time.sleep(2.0)

            # Clear all behavior faults before doing anything else.
            # Faults from prior sessions block navigation and localization.
            self._clear_faults()

            # Upload GraphNav map and set localization
            if MAP_PATH and os.path.isdir(MAP_PATH):
                self._upload_map(MAP_PATH)
            else:
                print(f"[spot] WARNING: MAP_PATH '{MAP_PATH}' not found — "
                      "navigate() will fail. Run record_map.py first.")

            self._connected = True
            print(f"[spot] Connected and standing: {SPOT_IP}")
            return True
        except Exception as e:
            print(f"[spot] Connection failed: {e}")
            return False

    def _clear_faults(self):
        """Clear all active behavior faults.

        SDK 5.1.1: RobotCommandBuilder.behavior_fault_clear_command(fault_id)
        is the correct method. Falls back to direct proto construction if the
        helper is unavailable. Must run after power-on and lease acquisition.
        """
        try:
            state = self.state_client.get_robot_state()
            faults = state.behavior_fault_state.faults
            if not faults:
                print("[spot] No behavior faults to clear")
                return
            for fault in faults:
                fid = fault.behavior_fault_id
                # SDK 5.1.1: RobotCommandBuilder has behavior_fault_clear_command
                clear_fn = getattr(
                    RobotCommandBuilder, "behavior_fault_clear_command", None
                )
                if clear_fn is not None:
                    cmd = clear_fn(fid)
                else:
                    # Fallback: build proto directly
                    from bosdyn.api import robot_command_pb2
                    cmd = robot_command_pb2.RobotCommand()
                    cmd.full_body_command.clear_behavior_fault_request.behavior_fault_id = fid
                self.command_client.robot_command(cmd)
                print(f"[spot] Cleared fault id={fid} cause={fault.cause}")
            time.sleep(0.5)
            state2 = self.state_client.get_robot_state()
            remaining = state2.behavior_fault_state.faults
            if remaining:
                print(f"[spot] WARNING: {len(remaining)} fault(s) still active. "
                      "Clear manually from tablet if this persists.")
            else:
                print("[spot] All behavior faults cleared")
        except Exception as e:
            print(f"[spot] Fault clear error: {e}")

    def _debug_camera_snapshot(self):
        """Capture a front camera frame, run YOLO, save annotated image.

        Saved to debug_camera.jpg in the working directory.
        Open it to verify YOLO detections and bounding boxes are working
        before starting trials.
        """
        try:
            from bosdyn.client.image import ImageClient
            import numpy as np

            image_client = self.robot.ensure_client(
                ImageClient.default_service_name
            )
            sources = ["frontleft_fisheye_image"]
            responses = image_client.get_image_from_sources(sources)
            if not responses:
                print("[spot] Debug snapshot: no image returned")
                return

            resp = responses[0]
            img_bytes = resp.shot.image.data
            fmt = resp.shot.image.format

            # Decode to numpy
            import struct
            if fmt == 1:   # JPEG
                import io
                try:
                    from PIL import Image as PILImage
                    img = PILImage.open(io.BytesIO(img_bytes))
                    frame = np.array(img)
                except ImportError:
                    import cv2
                    arr = np.frombuffer(img_bytes, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            else:
                # Raw — width × height × channels
                w = resp.shot.image.cols
                h = resp.shot.image.rows
                frame = np.frombuffer(img_bytes, dtype=np.uint8).reshape(h, w, -1)

            # Run YOLO
            perc = _get_perception()
            results = perc._yolo(frame) if perc._yolo else None

            # Draw boxes and save
            try:
                import cv2
                annotated = frame.copy()
                if results:
                    for r in results:
                        for box in r.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            label = perc._yolo.names[int(box.cls[0])]
                            conf  = float(box.conf[0])
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(annotated, f"{label} {conf:.2f}",
                                        (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, (0, 255, 0), 1)
                cv2.imwrite("debug_camera.jpg", annotated)
                n_boxes = sum(len(r.boxes) for r in results) if results else 0
                print(f"[spot] Debug snapshot saved: debug_camera.jpg "
                      f"({n_boxes} detection(s))")
            except ImportError:
                # cv2 not available — save raw JPEG
                with open("debug_camera.jpg", "wb") as f:
                    f.write(img_bytes)
                print("[spot] Debug snapshot saved: debug_camera.jpg (no cv2 — raw JPEG)")

        except Exception as e:
            print(f"[spot] Debug snapshot failed: {e}")

    def _upload_map(self, map_dir: str) -> bool:
        """Upload a recorded GraphNav map and set localization via nearest fiducial.

        map_dir must contain:
            graph          — serialized GraphNav graph (from recording_command_client)
            waypoint_snapshots/   — one file per waypoint UUID
            edge_snapshots/       — one file per edge UUID

        Returns True if upload and localization succeeded.
        """
        import os
        from bosdyn.api.graph_nav import map_pb2
        from bosdyn.client.graph_nav import GraphNavClient

        try:
            graph_path = os.path.join(map_dir, "graph")
            if not os.path.exists(graph_path):
                print(f"[spot] No 'graph' file found in {map_dir}")
                return False

            # Load and upload graph
            with open(graph_path, "rb") as f:
                graph = map_pb2.Graph()
                graph.ParseFromString(f.read())
            self.graph_nav_client.upload_graph(graph=graph)
            print(f"[spot] Graph uploaded: {len(graph.waypoints)} waypoints, "
                  f"{len(graph.edges)} edges")

            # Upload waypoint snapshots
            wp_snap_dir = os.path.join(map_dir, "waypoint_snapshots")
            if os.path.isdir(wp_snap_dir):
                for fname in os.listdir(wp_snap_dir):
                    fpath = os.path.join(wp_snap_dir, fname)
                    with open(fpath, "rb") as f:
                        snapshot = map_pb2.WaypointSnapshot()
                        snapshot.ParseFromString(f.read())
                    self.graph_nav_client.upload_waypoint_snapshot(snapshot)
                print(f"[spot] Waypoint snapshots uploaded")

            # Upload edge snapshots
            edge_snap_dir = os.path.join(map_dir, "edge_snapshots")
            if os.path.isdir(edge_snap_dir):
                for fname in os.listdir(edge_snap_dir):
                    fpath = os.path.join(edge_snap_dir, fname)
                    with open(fpath, "rb") as f:
                        snapshot = map_pb2.EdgeSnapshot()
                        snapshot.ParseFromString(f.read())
                    self.graph_nav_client.upload_edge_snapshot(snapshot)
                print(f"[spot] Edge snapshots uploaded")

            # Set localization via nearest fiducial.
            # Localization message is in nav_pb2; SetLocalizationRequest
            # is in graph_nav_pb2. Both locations are probed for SDK compat.
            # SDK 5.1.1: probe both modules — location varies across 4.x/5.x
            from bosdyn.api.graph_nav import nav_pb2 as _nav_pb2
            from bosdyn.api.graph_nav import graph_nav_pb2 as _gn_pb2
            _Localization = (
                getattr(_nav_pb2, "Localization", None)
                or getattr(_gn_pb2, "Localization", None)
            )
            _SetLocReq = (
                getattr(_gn_pb2, "SetLocalizationRequest", None)
                or getattr(_nav_pb2, "SetLocalizationRequest", None)
            )
            if _Localization is None or _SetLocReq is None:
                print("[spot] WARNING: Could not find Localization types in "
                      "nav_pb2 or graph_nav_pb2 — skipping localization.")
            else:
                # Attempt 1: localize via nearest AprilTag fiducial.
                # SPOT must be able to see a fiducial from its start position.
                localized = False
                try:
                    localization = _Localization()
                    self.graph_nav_client.set_localization(
                        initial_guess_localization=localization,
                        ko_tform_body=None,
                        max_distance=None,
                        max_yaw=None,
                        fiducial_init=_SetLocReq.FIDUCIAL_INIT_NEAREST,
                    )
                    print("[spot] Localization set via nearest fiducial")
                    localized = True
                except Exception as fid_err:
                    print(f"[spot] Fiducial localization failed: {fid_err}")

                # Attempt 2: no-fiducial fallback using a known start waypoint.
                # Requires START_WAYPOINT to be set (UUID from record_map.py).
                # SPOT must be physically at that waypoint when connecting.
                if not localized:
                    if START_WAYPOINT:
                        try:
                            print(f"[spot] Trying no-fiducial localization "
                                  f"at waypoint '{START_WAYPOINT}'...")
                            localization = _Localization()
                            localization.waypoint_id = START_WAYPOINT
                            self.graph_nav_client.set_localization(
                                initial_guess_localization=localization,
                                ko_tform_body=None,
                                max_distance=None,
                                max_yaw=None,
                                fiducial_init=_SetLocReq.FIDUCIAL_INIT_NO_FIDUCIAL,
                            )
                            print(f"[spot] Localization set via waypoint "
                                  f"'{START_WAYPOINT}' (no fiducial)")
                            localized = True
                        except Exception as wpt_err:
                            print(f"[spot] No-fiducial localization failed: {wpt_err}")
                    else:
                        print("[spot] No START_WAYPOINT set — cannot attempt "
                              "no-fiducial fallback.")

                if not localized:
                    print("[spot] WARNING: Localization not set. "
                          "Navigate commands will fail.")
                    print("[spot] Fix: ensure SPOT can see an AprilTag, OR "
                          "set START_WAYPOINT to a known waypoint UUID and "
                          "place SPOT at that position before connecting.")
            return True

        except Exception as e:
            print(f"[spot] Map upload / localization failed: {e}")
            print("[spot] Navigate commands will fail until localization is set.")
            return False

    def disconnect(self):
        if self._connected and self.robot:
            try:
                cmd = RobotCommandBuilder.safe_power_off_command()
                self.command_client.robot_command(cmd)
            except Exception:
                pass
            if self.lease_ka:
                self.lease_ka.shutdown()
        self._connected = False
        print("[spot] Disconnected.")

    @property
    def connected(self) -> bool:
        return self._connected
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, *_):
        self.disconnect()

# ── Skill functions ───────────────────────────────────────────────────────────

def navigate(
    robot: SpotRobot,
    waypoint_id: str,
    timeout_sec: float = NAVIGATE_TIMEOUT_SEC,
) -> SkillResult:
    """
    Navigate to a GraphNav waypoint by ID.
    If not connected, returns a mock success for dry-run testing.
    """
    skill = "navigate"

    if not robot.connected:
        print(f"[mock] navigate -> waypoint = '{waypoint_id}'")
        return SkillResult(True, skill, f"Mock: navigated to '{waypoint_id}'",
                            {"waypoint_id": waypoint_id}, mock = True)
    try:
        nav_client = robot.graph_nav_client

        # Resolve human-readable name -> GraphNav UUID via WAYPOINT_MAP.
        # Falls back to the raw string if not found (allows UUID passthrough).
        resolved = WAYPOINT_MAP.get(waypoint_id.lower(), waypoint_id)
        if resolved != waypoint_id:
            print(f"[spot] navigate: '{waypoint_id}' -> '{resolved}'")
        waypoint_id = resolved

        # SDK 5.1.1: TravelParams moved to graph_nav_pb2
        # Probe both locations for safety across minor version differences.
        nav_kwargs = {}
        _TravelParams = (
            getattr(graph_nav_pb2, "TravelParams", None)
            or getattr(nav_pb2, "TravelParams", None)
        )
        if _TravelParams is not None:
            nav_kwargs["travel_params"] = _TravelParams(max_distance=0.5)

        # SDK 5.1.1: navigate_to(waypoint_id, cmd_duration, travel_params=None)
        # cmd_duration = how long (sec) the robot will attempt navigation
        cmd_id = nav_client.navigate_to(
            waypoint_id,
            cmd_duration=timeout_sec,
            **nav_kwargs
        )

        # Poll until complete or timeout
        start = time.time()
        while time.time() - start < timeout_sec:
            feedback = nav_client.navigation_feedback(cmd_id)
            status = feedback.status
            if status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_REACHED_GOAL:
                return SkillResult(True, skill, f"Reached waypoint '{waypoint_id}'",
                                   {"waypoint_id": waypoint_id})
            if status in (
                graph_nav_pb2.NavigationFeedbackResponse.STATUS_LOST,
                graph_nav_pb2.NavigationFeedbackResponse.STATUS_STUCK,
                graph_nav_pb2.NavigationFeedbackResponse.STATUS_COMMAND_OVERIDDEN,
            ):
                return SkillResult(False, skill,
                                   f"Navigation failed (status = {status})",
                                   {"waypoint_id": waypoint_id, "status": status})
            time.sleep(NAVIGATE_POLL_SEC)

        return SkillResult(False, skill, "Navigation timed out", {"waypoint_id": waypoint_id})

    except Exception as e:
        return SkillResult(False, skill, f"Navigation error: {e}")

def scan(
    robot: SpotRobot,
    n_rotations: int = 4,
) -> SkillResult:
    """
    Rotate SPOT in place to build a 360 degree scene description.
    Returns list of captured view angles (mock)or real rotation commands.
    """
    skill = "scan"

    if not robot.connected:
        print(f"[mock] can -> {n_rotations} rotations")
        views = [f"view_{i * (360 / n_rotations)}deg" for i in range (n_rotations)]
        return SkillResult(True, skill, "Mock: scan complete",
                           {"views": views, "object_count": 0}, mock = True)
    
    try:
        import math
        # Rotate at 0.5 rad/s. One full 360° = 2π / 0.5 = ~12.6 s.
        # For n_rotations partial sweeps, rotate for step_duration each,
        # pausing briefly between steps for image capture.
        rot_speed   = 0.5            # rad/s
        step_rad    = (2 * math.pi) / n_rotations
        step_dur    = step_rad / rot_speed   # seconds per step
        pause_dur   = 0.5            # pause at each position for camera

        for i in range(n_rotations):
            # LeaseKeepAlive manages the lease wallet — do NOT pass lease=
            # directly to robot_command; it expects the proto not the wrapper.
            end_t = time.time() + step_dur
            cmd = RobotCommandBuilder.synchro_velocity_command(
                v_x=0.0, v_y=0.0, v_rot=rot_speed
            )
            robot.command_client.robot_command(cmd, end_time_secs=end_t)
            time.sleep(step_dur + pause_dur)

        # Come to a stop
        stop = RobotCommandBuilder.synchro_stand_command()
        robot.command_client.robot_command(stop)
        time.sleep(0.5)

        return SkillResult(True, skill, "Scan complete — 360 degree rotation finished",
                           {"n_rotations": n_rotations})

    except Exception as e:
        return SkillResult(False, skill, f"Scan error: {e}")
    
def locate(
    robot: SpotRobot,
    object_label: str,
) -> SkillResult:
    """
    Search for a nambed object in the current scene using the perception module.
    Does not move -- reports whether object is visbil and its bounding box.
    """
    skill = "locate"

    if not robot.connected:
        print(f"[mock] locate -> object = '{object_label}'")
        return SkillResult(True, skill, f"Mock: '{object_label}' located",
                           {"object_label": object_label,
                            "found": True,
                            "bbox": [100, 150, 200, 250],
                            "confidence": 0.85}, mock = True)
    try:
        # Use module-level perception singleton — avoids reloading BertModel
        # and YOLOv8 on every locate() call during a live trial session.
        perc = _get_perception()
        scene = perc.get_scene_objects()

        match = next((o for o in scene if o.label.lower() == object_label.lower()), None)
        if match:
            return SkillResult(True, skill, f"Found '{object_label}' in scene",
                               {"object_label": object_label,
                               "found": True,
                               "bbox": match.bbox,
                               "confidence": match.confidence})
        else:
            return SkillResult(False, skill, f"'{object_label}' not visible in current scene",
                               {"object_label": object_label, "found": False})
        
    except Exception as e:
        return SkillResult(False, skill, f"Locate error: {e}")
    
def pick_up(
    robot: SpotRobot,
    object_label: str,
    bbox: Optional[List[float]] = None,
) -> SkillResult:
    """
    Grasp an object using the SPOT Arm manipulation API.
    bbox: [x1, y1, x2, y2] pixel coordinates from YOLO detection.
    If bbox is None, attemps grasp at scene center (fallback).
    """
    skill = "pick_up"

    if not robot.connected:
        print(f"[mock] pick -> object  '{object_label}' bbox = {bbox}")
        return SkillResult(True, skill, f"Mock: picked up '{object_label}'",
                           {"object_label": object_label, "bbox": bbox}, mock = True)
    try:
        # Build a grasp request from the image frame
        # Uses ManipulationApiRequest with pexel_xy grasp targeting
        if bbox:
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
        else:
            cx, cy = 320, 240 # image center fallback

        grasp = manipulation_api_pb2.PickObjectInImage(
            pixel_xy = geometry_pb2.Vec2(x = cx, y = cy),
            transforms_snapshot_for_camera = robot.state_client
                .get_robot_state().kinematic_state.transforms_snapshot,
            frame_name_image_sensor = BODY_FRAME_NAME,
            camera_model = None, # filled by SDK from robot state
        )

        request = manipulation_api_pb2.ManipulationApiRequest(
            pick_object_in_image = grasp
        )
        response = robot.manip_client.manipulation_api_command(
            manipulation_api_request = request
        )

        # Poll for completion
        start = time.time()
        while time.time() - start < GRASP_TIMEOUT_SEC:
            feedback = robot.manip_client.manipulation_api_feedback_command(
                manipulation_api_pb2.ManipulationApiFeedbackRequest(
                    manipulation_cmd_id = response.manipulation_cmd_id
                )
            )
            state = feedback.current_state
            if state == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED:
                return SkillResult(True, skill, f"Grasped '{object_label}'",
                                   {"object_label": object_label})
            if state == manipulation_api_pb2.MANIP_STATE_GRASP_FAILED:
                return SkillResult(False, skill, f"Grasp failed for '{object_label}'",
                                   {"object_label": object_label})
            time.sleep(0.5)

        return SkillResult(False, skill, "Grasp timed out", 
                           {"object_label": object_label})
    
    except Exception as e:
        return SkillResult(False, skill, f"Pick-up error: {e}")
    
def deliver(
        robot: SpotRobot,
        object_label: str = "",
) -> SkillResult:
    """
    Extend the arm forward to present the held object to the user.
    """
    skill = "deliver"

    if not robot.connected:
        print(f"[mock] deliver -> object = '{object_label}'")
        return SkillResult(True, skill, f"Mock: delivered '{object_label}'",
                           {"object_label": object_label}, mock = True)
    
    try:
        # Move arm to a forward-extended delivery pose
        from bosdyn.api import arm_command_pb2
        from bosdyn.util import seconds_to_duration

        hand_pose = math_helpers.SE3Pose(
            x = 0.75, y = 0.0, z = 0.25,    # 75cm forward, 25cm above base
            rot = math_helpers.Quat(w = 1, x = 0, y = 0, z = 0)
        )
        arm_cmd = RobotCommandBuilder.arm_pose_command(
            hand_pose.x, hand_pose.y, hand_pose.z,
            hand_pose.rot.w, hand_pose.rot.x,
            hand_pose.rot.y, hand_pose.rot.z,
            BODY_FRAME_NAME,
            seconds = DELIVER_TIMEOUT_SEC
        )
        cmd_id = robot.command_client.robot_command(arm_cmd)
        time.sleep(DELIVER_TIMEOUT_SEC)

        return SkillResult(True, skill, f"Delivered '{object_label}' -- arm extended",
                           {"object_label": object_label})
    
    except Exception as e:
        return SkillResult(False, skill, f"Deliver error: {e}")
    
def release(robot: SpotRobot) -> SkillResult:
    """
    Open the gripper to release the held object.
    Called after deliver() once the user has taken the object.
    """
    skill = "release"

    if not robot.connected:
        print("[mock] release -> gripper open")
        return SkillResult(True, skill, "Mock: gripper opened", mock = True)
    try:
        open_cmd = RobotCommandBuilder.claw_gripper_open_fraction_command(1.0)
        robot.command_client.robot_command(open_cmd)
        time.sleep(1.0)

        # Stow arm
        stow = RobotCommandBuilder.arm_stow_command()
        robot.command_client.robot_command(stow)
        time.sleep(2.0)
        return SkillResult(True, skill, "Gripper opened and arm stowed")
    except Exception as e:
        return SkillResult(False, skill, f"Release error: {e}")
    
# ── Skill registry (used by executor.py) ─────────────────────────────────────
SKILL_REGISTRY = {
    "navigate": navigate,
    "scan": scan,
    "locate": locate,
    "pick_up": pick_up,
    "deliver": deliver,
    "release": release,
}

# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Running skill dry-run (mock mode)...\n")
    robot = SpotRobot()     # USE_SPOT = false -> mock mode

    results = [
        navigate(robot, "waypoint_desk"),
        scan(robot),
        locate(robot, "pen"),
        pick_up(robot, "pen", bbox = [100, 150, 130, 180]),
        deliver(robot, "pen"),
        release(robot),
    ]

    for r in results:
        status = "SUCCESS" if r.success else "FAIL"
        mock_tag = " [mock]" if r.mock else ""
        print(f"    {status} {r.skill:<12} {r.message}{mock_tag}")