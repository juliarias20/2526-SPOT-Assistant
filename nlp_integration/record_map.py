"""
record_map.py
-------------
GraphNav Map Recording Utility for SPOT Live Trials

Run this script ONCE to:
    1. Start a GraphNav recording session
    2. Walk SPOT to each location and stamp a named waypoint
    3. Stop recording and download the map to disk
    4. Print all waypoint name -> UUID mappings for WAYPOINT_MAP in spot_skills.py

Usage:
    python record_map.py --output maps/trial_space

Then follow the interactive prompts to walk the space and name each waypoint.

Requirements:
    - SPOT must be powered on and reachable at SPOT_IP
    - Set environment variables: SPOT_IP, SPOT_USER, SPOT_PASS
    - Run from the nlp_integration directory

After recording, copy the printed WAYPOINT_MAP dict into spot_skills.py.
"""

import os
import time
import argparse
from pathlib import Path

import bosdyn.client
import bosdyn.client.util
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.graph_nav import GraphNavClient
from bosdyn.client.map_processing import MapProcessingServiceClient
from bosdyn.client.recording import GraphNavRecordingServiceClient
from bosdyn.api.graph_nav import recording_pb2

# ── Configuration ──────────────────────────────────────────────────────────────
SPOT_IP   = os.environ.get("SPOT_IP",   "192.168.80.3")
SPOT_USER = os.environ.get("SPOT_USER", "user")
SPOT_PASS = os.environ.get("SPOT_PASS", "password")


def main(output_dir: str):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Connect ────────────────────────────────────────────────────────────────
    sdk = bosdyn.client.create_standard_sdk("RecordMap")
    robot = sdk.create_robot(SPOT_IP)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()
    print(f"[record] Connected to SPOT: {SPOT_IP}")

    lease_client    = robot.ensure_client(LeaseClient.default_service_name)
    graph_nav       = robot.ensure_client(GraphNavClient.default_service_name)
    recording       = robot.ensure_client(GraphNavRecordingServiceClient.default_service_name)
    map_processing  = robot.ensure_client(MapProcessingServiceClient.default_service_name)

    lease = lease_client.acquire()
    with LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):

        # ── Clear any existing map on robot ────────────────────────────────────
        graph_nav.clear_graph()
        print("[record] Cleared existing GraphNav graph.")

        # ── Start recording ────────────────────────────────────────────────────
        response = recording.start_recording()
        # STATUS_OK = 0 in the recording proto; compare numerically to avoid
        # SDK version differences in how the enum constant is exposed
        if response.status != 0:
            print(f"[record] Failed to start recording: {response.status}")
            return
        print("[record] Recording started. Walk SPOT to each location.")
        print("         Carry SPOT with the tablet or use keyboard teleoperation.\n")

        # ── Waypoint stamping loop ─────────────────────────────────────────────
        waypoint_map = {}   # name -> uuid

        while True:
            print("─" * 50)
            name = input(
                "At a location? Enter a name to stamp a waypoint\n"
                "(e.g. 'desk', 'table', 'kitchen'), or 'done' to stop: "
            ).strip().lower()

            if name == "done":
                break
            if not name:
                continue

            # Create a waypoint annotation with the human-readable name
            wp_response = recording.create_waypoint(waypoint_name=name)
            if wp_response.status == recording_pb2.CreateWaypointResponse.STATUS_OK:
                uuid = wp_response.created_waypoint.id
                waypoint_map[name] = uuid
                print(f"[record] ✓ Stamped '{name}' -> {uuid}")
            else:
                print(f"[record] ✗ Failed to create waypoint '{name}': {wp_response.status}")

        # ── Stop recording ─────────────────────────────────────────────────────
        recording.stop_recording()
        print("\n[record] Recording stopped.")

        # ── Optimize map ───────────────────────────────────────────────────────
        print("[record] Optimizing map (anchoring)...")
        try:
            from bosdyn.api.graph_nav import map_processing_pb2
            map_processing.process_anchoring(
                params=map_processing_pb2.ProcessAnchoringRequest.Params()
            )
            print("[record] Map anchoring complete.")
        except Exception as e:
            print(f"[record] Anchoring skipped (non-fatal): {e}")

        # ── Download and save map ──────────────────────────────────────────────
        print("[record] Downloading map...")
        graph = graph_nav.download_graph()

        graph_path = output_path / "graph"
        with open(graph_path, "wb") as f:
            f.write(graph.SerializeToString())
        print(f"[record] Graph saved: {graph_path}")

        # Download waypoint snapshots
        snapshot_dir = output_path / "waypoint_snapshots"
        snapshot_dir.mkdir(exist_ok=True)
        for wp in graph.waypoints:
            if wp.snapshot_id:
                snap = graph_nav.download_waypoint_snapshot(wp.snapshot_id)
                snap_path = snapshot_dir / wp.snapshot_id
                with open(snap_path, "wb") as f:
                    f.write(snap.SerializeToString())
        print(f"[record] Waypoint snapshots saved: {snapshot_dir}")

        # Download edge snapshots
        edge_dir = output_path / "edge_snapshots"
        edge_dir.mkdir(exist_ok=True)
        for edge in graph.edges:
            if edge.snapshot_id:
                snap = graph_nav.download_edge_snapshot(edge.snapshot_id)
                snap_path = edge_dir / edge.snapshot_id
                with open(snap_path, "wb") as f:
                    f.write(snap.SerializeToString())
        print(f"[record] Edge snapshots saved: {edge_dir}")

    # ── Print WAYPOINT_MAP for spot_skills.py ──────────────────────────────────
    print("\n" + "=" * 60)
    print("  Map recording complete!")
    print(f"  Map saved to: {output_path.resolve()}")
    print("=" * 60)
    print("\nCopy this WAYPOINT_MAP into spot_skills.py:\n")
    print("WAYPOINT_MAP: dict[str, str] = {")
    for name, uuid in waypoint_map.items():
        print(f'    "{name}": "{uuid}",')
    print("}")
    print()

    # Also save to a file for reference
    map_txt = output_path / "waypoint_map.txt"
    with open(map_txt, "w") as f:
        f.write("WAYPOINT_MAP: dict[str, str] = {\n")
        for name, uuid in waypoint_map.items():
            f.write(f'    "{name}": "{uuid}",\n')
        f.write("}\n")
    print(f"[record] Waypoint map also saved to: {map_txt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record a GraphNav map for SPOT live trials.")
    parser.add_argument(
        "--output", "-o",
        default="maps/trial_space",
        help="Directory to save the map files (default: maps/trial_space)"
    )
    args = parser.parse_args()
    main(args.output)