"""
run.py
------
Interactive command runner for the SPOT NLP framework.

Usage:
    # Interactive REPL (recommended):
    python run.py

    # Single command (exits after):
    python run.py "go to the desk and bring me the notebook"

    # Live SPOT with camera feed window (local YOLO):
    USE_SPOT=true SPOT_IP=192.168.80.3 python run.py --live-feed

    # Live SPOT with network compute server feed:
    #   1. Start the server first:
    #       python object_detection/network_compute_server.py -m yolov8n.pt 192.168.80.3
    #   2. Run with --use-compute-server:
    USE_SPOT=true python run.py --live-feed --use-compute-server --server fetch-server --model yolov8n

Commands:
    Any natural language task  ->  interpreted + executed
    exit / quit / q            ->  exit the runner
    help                       ->  show this message
    status                     ->  show robot connection status
    feed on / feed off         ->  toggle live feed window at runtime

Live feed modes:
    Local (default):        uses the YOLO model already loaded by perception.py.
                            No extra server needed. Green bounding boxes.
    Compute server:         routes through object_detection/network_compute_server.py
                            via SPOT's NetworkComputeBridge. Orange bounding boxes.
                            Allows custom-trained models from object_detection/models/.
"""

import sys
import os
import time
import threading
import argparse

from spot_skills import SpotRobot
from executor import TaskExecutor

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║          SPOT NLP Task Runner  —  Cal Poly Pomona            ║
║  Type a command to execute.  'exit' to quit.  'help' for info║
╚══════════════════════════════════════════════════════════════╝
"""

HELP = """
Examples:
  go to the desk
  bring me the pen
  head to the kitchen and grab me a cup
  find my backpack
  bring me something to write with
  hand me something sharp
  scan the room

Commands:
  exit / quit / q    Exit
  status             Show robot connection status
  feed on            Start live camera feed (if connected)
  feed off           Stop live camera feed
  help               Show this message
"""


# ── Live feed ─────────────────────────────────────────────────────────────────

class LiveFeedThread(threading.Thread):
    """
    Background thread that grabs frames from SPOT's front camera,
    runs YOLO on each frame, and shows annotated results in a cv2 window.

    Two modes controlled by use_compute_server:

    LOCAL (default)
        Calls SPOT's ImageClient directly, runs the PerceptionModule
        singleton's already-loaded YOLOv8 model on each frame.
        Nothing extra needs to be running.

    COMPUTE SERVER
        Routes requests through SPOT's NetworkComputeBridge to
        object_detection/network_compute_server.py, which can serve
        custom-trained models from object_detection/models/.
        Requires the server process to be started first:
            python object_detection/network_compute_server.py \\
                -m object_detection/models/<your_model>/<model>.pt 192.168.80.3
    """

    WINDOW = "SPOT Live Feed"

    def __init__(
        self,
        robot: SpotRobot,
        use_compute_server: bool = False,
        server_name: str = "fetch-server",
        model_name: str = "yolov8n",
        camera_source: str = "frontleft_fisheye_image",
        confidence: float = 0.30,
    ):
        super().__init__(daemon=True)
        self.robot = robot
        self.use_compute_server = use_compute_server
        self.server_name = server_name
        self.model_name = model_name
        self.camera_source = camera_source
        self.confidence = confidence
        self._stop_event = threading.Event()
        self._active = threading.Event()
        self._active.set()

    def stop(self):
        self._stop_event.set()

    def pause(self):
        self._active.clear()

    def resume(self):
        self._active.set()

    # ── Local YOLO mode ───────────────────────────────────────────────────────

    def _grab_local(self):
        """Grab frame via ImageClient, run local YOLO, return annotated frame."""
        try:
            import cv2
            import numpy as np
            from bosdyn.client.image import ImageClient
            from spot_skills import _get_perception

            image_client = self.robot.robot.ensure_client(
                ImageClient.default_service_name
            )
            responses = image_client.get_image_from_sources([self.camera_source])
            if not responses:
                return None

            resp = responses[0]
            img_bytes = resp.shot.image.data
            fmt = resp.shot.image.format

            if fmt == 1:  # JPEG
                arr = np.frombuffer(img_bytes, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            else:
                w = resp.shot.image.cols
                h = resp.shot.image.rows
                frame = np.frombuffer(img_bytes, dtype=np.uint8).reshape(h, w, -1)

            if frame is None:
                return None

            perc = _get_perception()
            if perc._yolo is None:
                return frame

            results = perc._yolo(frame, verbose=False)
            annotated = frame.copy()
            for r in results:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    if conf < self.confidence:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = perc._yolo.names[int(box.cls[0])]
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        annotated, f"{label} {conf:.2f}",
                        (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                    )
            return annotated

        except Exception as e:
            print(f"\n[feed] Local frame error: {e}")
            return None

    # ── Network compute server mode ───────────────────────────────────────────

    def _grab_server(self):
        """Route detection through object_detection/network_compute_server.py."""
        try:
            import cv2
            import numpy as np
            from google.protobuf import wrappers_pb2
            from bosdyn.api import network_compute_bridge_pb2, image_pb2
            from bosdyn.client.network_compute_bridge_client import NetworkComputeBridgeClient

            nc_client = self.robot.robot.ensure_client(
                NetworkComputeBridgeClient.default_service_name
            )

            image_source_and_service = network_compute_bridge_pb2.ImageSourceAndService(
                image_source=self.camera_source
            )
            input_data = network_compute_bridge_pb2.NetworkComputeInputData(
                image_source_and_service=image_source_and_service,
                model_name=self.model_name,
                min_confidence=self.confidence,
                rotate_image=network_compute_bridge_pb2.NetworkComputeInputData.ROTATE_IMAGE_ALIGN_HORIZONTAL,
            )
            server_data = network_compute_bridge_pb2.NetworkComputeServerConfiguration(
                service_name=self.server_name
            )
            req = network_compute_bridge_pb2.NetworkComputeRequest(
                input_data=input_data, server_config=server_data
            )
            resp = nc_client.network_compute_bridge_command(req)

            # Decode image from response
            img_data = resp.image_response.shot.image.data
            fmt = resp.image_response.shot.image.format
            if fmt == image_pb2.Image.FORMAT_RAW:
                rows = resp.image_response.shot.image.rows
                cols = resp.image_response.shot.image.cols
                frame = np.frombuffer(img_data, dtype=np.uint8).reshape(rows, cols, -1)
            else:
                arr = np.frombuffer(img_data, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if frame is None:
                return None

            # Draw bounding boxes from server response (orange — distinguishes from local)
            annotated = frame.copy()
            for obj in resp.object_in_image:
                conf_msg = wrappers_pb2.FloatValue()
                obj.additional_properties.Unpack(conf_msg)
                conf = conf_msg.value
                label = obj.name.split('_label_')[-1]

                pts = [[int(v.x), int(v.y)] for v in obj.image_properties.coordinates.vertexes]
                if len(pts) >= 2:
                    x1 = min(p[0] for p in pts)
                    y1 = min(p[1] for p in pts)
                    x2 = max(p[0] for p in pts)
                    y2 = max(p[1] for p in pts)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 165, 255), 2)
                    cv2.putText(
                        annotated, f"{label} {conf:.2f}",
                        (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1,
                    )
            return annotated

        except Exception as e:
            print(f"\n[feed] Compute server error: {e}")
            return None

    # ── Thread main loop ──────────────────────────────────────────────────────

    def run(self):
        try:
            import cv2
        except ImportError:
            print("[feed] cv2 not installed — pip install opencv-python")
            return

        mode_label = f"compute-server ({self.server_name}/{self.model_name})" \
            if self.use_compute_server else "local YOLO"
        print(f"\n[feed] Starting — {mode_label} | {self.camera_source}")
        print(f"[feed] Press 'q' in the feed window to close.\n")

        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW, 800, 600)

        while not self._stop_event.is_set():
            if not self._active.is_set():
                time.sleep(0.1)
                continue

            frame = self._grab_server() if self.use_compute_server else self._grab_local()

            if frame is not None:
                cv2.imshow(self.WINDOW, frame)

            key = cv2.waitKey(15) & 0xFF
            if key == ord('q'):
                print("\n[feed] Closed by user.")
                break

        cv2.destroyWindow(self.WINDOW)
        print("[feed] Feed stopped.")


# ── Result printing ───────────────────────────────────────────────────────────

def print_result(result, elapsed: float = 0.0):
    ok = "✓" if result.success else "✗"
    print(f"\n  {ok}  {result.command}")
    print(f"     Steps    : {len(result.steps)}")
    for step in result.steps:
        icon = "  ✓" if step.success else "  ✗"
        mock = " [mock]" if step.mock else ""
        retry = f" (retries={step.retries})" if step.retries else ""
        print(f"        {icon}  {step.skill}{mock}{retry}  — {step.message}")
    if result.clarifications_needed:
        print(f"\n  ⚠  Clarification needed:")
        for q in result.clarifications_needed:
            print(f"       → {q}")
    if result.error:
        print(f"\n  ✗  Error: {result.error}")
    elapsed_str = f"{elapsed:.2f}s" if elapsed else ""
    print(f"\n     {'SUCCESS' if result.success else 'FAILED'}  {elapsed_str}\n")


def run_single(command: str, executor: TaskExecutor):
    print(f"\n  Running: \"{command}\"")
    print("  " + "─" * 56)
    t0 = time.time()
    result = executor.execute(command)
    elapsed = time.time() - t0
    print_result(result, elapsed)
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("command", nargs="*", help="Single command to execute then exit")
    parser.add_argument("--live-feed", action="store_true",
                        help="Open live camera feed window when connected")
    parser.add_argument("--use-compute-server", action="store_true",
                        help="Use object_detection/network_compute_server.py for detections")
    parser.add_argument("--server", default="fetch-server",
                        help="Compute server service name (default: fetch-server)")
    parser.add_argument("--model", default="yolov8n",
                        help="Model name on compute server (default: yolov8n)")
    parser.add_argument("--camera", default="frontleft_fisheye_image",
                        help="Camera source (default: frontleft_fisheye_image)")
    args, _ = parser.parse_known_args()

    single_cmd = " ".join(args.command).strip() if args.command else None

    print(BANNER)

    # ── Connect ───────────────────────────────────────────────────────────────
    robot = SpotRobot()
    connected = robot.connect()
    if connected:
        mode = "LIVE"
    elif os.environ.get("USE_SPOT", "false").lower() == "true":
        mode = "DRY-RUN (mock) — SPOT unreachable"
        print("  ⚠  USE_SPOT=true but SPOT connection failed — running in mock mode.")
    else:
        mode = "DRY-RUN (mock)"

    print(f"  Mode     : {mode}")
    print(f"  USE_SPOT : {os.environ.get('USE_SPOT', 'false')}")
    if args.live_feed:
        if connected:
            feed_mode = f"compute-server ({args.server}/{args.model})" \
                if args.use_compute_server else "local YOLO"
            print(f"  Feed     : enabled ({feed_mode})")
        else:
            print(f"  Feed     : disabled (not connected)")
    print()

    # ── Build executor ────────────────────────────────────────────────────────
    from interpret import Phase1Interpreter
    from spot_skills import _get_perception

    interpreter = Phase1Interpreter()
    executor = TaskExecutor(robot, interpreter=interpreter)

    if connected:
        _get_perception()
        robot._debug_camera_snapshot()

    # ── Start live feed thread ────────────────────────────────────────────────
    feed_thread = None
    if args.live_feed and connected:
        feed_thread = LiveFeedThread(
            robot=robot,
            use_compute_server=args.use_compute_server,
            server_name=args.server,
            model_name=args.model,
            camera_source=args.camera,
        )
        feed_thread.start()

    # ── Single-command mode ───────────────────────────────────────────────────
    if single_cmd:
        run_single(single_cmd, executor)
        if feed_thread:
            feed_thread.stop()
            feed_thread.join(timeout=2)
        robot.disconnect()
        return

    # ── Interactive REPL ──────────────────────────────────────────────────────
    print("  Ready. Enter a command:\n")
    try:
        while True:
            try:
                command = input("  SPOT> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  Exiting.")
                break

            if not command:
                continue

            low = command.lower()

            if low in ("exit", "quit", "q"):
                print("  Goodbye.")
                break

            if low == "help":
                print(HELP)
                continue

            if low == "status":
                feed_status = "running" if (feed_thread and feed_thread.is_alive()) else "off"
                print(f"\n  Robot connected : {robot.connected}")
                print(f"  Mode            : {mode}")
                print(f"  Live feed       : {feed_status}\n")
                continue

            if low == "feed on":
                if not connected:
                    print("  ⚠  Not connected to SPOT.\n")
                elif feed_thread and feed_thread.is_alive():
                    feed_thread.resume()
                    print("  Feed resumed.\n")
                else:
                    feed_thread = LiveFeedThread(
                        robot=robot,
                        use_compute_server=args.use_compute_server,
                        server_name=args.server,
                        model_name=args.model,
                        camera_source=args.camera,
                    )
                    feed_thread.start()
                    print("  Feed started.\n")
                continue

            if low == "feed off":
                if feed_thread and feed_thread.is_alive():
                    feed_thread.stop()
                    feed_thread.join(timeout=2)
                    feed_thread = None
                    print("  Feed stopped.\n")
                else:
                    print("  Feed is not running.\n")
                continue

            run_single(command, executor)

    finally:
        if feed_thread and feed_thread.is_alive():
            feed_thread.stop()
            feed_thread.join(timeout=2)
        robot.disconnect()


if __name__ == "__main__":
    main()