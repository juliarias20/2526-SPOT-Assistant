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
    SPOT_IP         192.168.80.3
    SPOT_USER       user
    SPOT_PASS       password
    USE_SPOT        true    (set to false to run in mock/dry-run mode)
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

# ── Perception singleton ──────────────────────────────────────────────────────
# PerceptionModule is expensive to construct (loads BertModel + YOLOv8).
# A single module-level instance is shared across all locate() calls for the
# lifetime of a live trial session. The embedder can optionally be pre-loaded
# by Phase1Interpreter and passed in on first use to avoid a third load.

_perception: Optional["PerceptionModule"] = None  # type: ignore[name-defined]

def _get_perception(embedder=None):
    """Return the module-level PerceptionModule singleton, creating it if needed.

    Pass embedder on the first call to share the SentenceTransformer instance
    that was already loaded by Phase1Interpreter, avoiding a redundant model load.
    Subsequent calls return the cached instance regardless of the embedder arg.
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
            self.robot.power_on(timeout_sec = 20)
            assert self.robot.is_powered_on(), "SPOT failed to power on."
            cmd = RobotCommandBuilder.synchro_stand_command()
            self.command_client.robot_command(cmd)
            time.sleep(1.5)

            self._connected = True
            print(f"[spot] Connected and standing: {SPOT_IP}")
            return True
        except Exception as e:
            print(f"[spot] Connection failed: {e}")
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

        # Resolve human-readable name to GraphNav UUID via WAYPOINT_MAP.
        # Falls back to the raw string if not found (allows UUID passthrough).
        resolved_id = WAYPOINT_MAP.get(waypoint_id.lower(), waypoint_id)
        if resolved_id != waypoint_id:
            print(f"[spot] Resolved '{waypoint_id}' -> '{resolved_id}'")

        cmd_id = nav_client.navigate_to(
            resolved_id,
            travel_params = nav_pb2.TravelParams(max_distance = 0.5)
        )

        # Poll until complete or timeout
        start = time.time()
        while time.time() - start < timeout_sec:
            feedback = nav_client.navigation_feedback(cmd_id)
            status = feedback.status
            if status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_REACHED_GOAL:
                return SkillResult(True, skill, f"Reached waypoint '{waypoint_id}'",
                                   {"waypoint_id": waypoint_id, "resolved_id": resolved_id})
            if status in (
                graph_nav_pb2.NavigationFeedbackResponse.STATUS_LOST,
                graph_nav_pb2.NavigationFeedbackResponse.STATUS_STUCK,
                graph_nav_pb2.NavigationFeedbackResponse.STATUS_COMMAND_OVERRIDDEN,
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
        step_rad = (2 * math.pi) / n_rotations
        for i in range(n_rotations):
            heading = i * step_rad
            cmd = RobotCommandBuilder.synchro_velocity_command(
                v_x = 0, v_y = 0, v_rot = 0.5   # slow rotation
            )
            robot.command_client.robot_command(cmd, end_time_secs = time.time() + 1.5)
            time.sleep(1.8)

        # Stop
        stop = RobotCommandBuilder.synchro_stand_command()
        robot.command_client.robot_command(stop)

        return SkillResult(True, skill, "Scan complete -- 360 degree rotation finished",
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
        # Use module-level perception singleton — avoids re-loading BertModel
        # on every locate() call during a live trial.
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