"""
run.py
------
Interactive command runner for the SPOT NLP framework.

Usage:
    # Interactive REPL (recommended):
    python run.py

    # Single command (exits after):
    python run.py "go to the desk and bring me the notebook"

    # Live SPOT (set env vars first):
    USE_SPOT=true SPOT_IP=192.168.80.3 python run.py

Commands:
    Any natural language task  ->  interpreted + executed
    exit / quit / q            ->  exit the runner
    help                       ->  show this message
    status                     ->  show robot connection status
"""

import sys
import os
import time

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
  help               Show this message
"""

def print_result(result, elapsed: float = 0.0):
    """Pretty-print an ExecutionResult."""
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
    """Execute one command and print result."""
    import time
    print(f"\n  Running: \"{command}\"")
    print("  " + "─" * 56)
    t0 = time.time()
    result = executor.execute(command)
    elapsed = time.time() - t0
    print_result(result, elapsed)
    return result


def main():
    single_cmd = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else None

    print(BANNER)

    # Connect robot — falls back to mock automatically if SPOT
    # is unreachable or USE_SPOT is not set.
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
    print()

    from interpret import Phase1Interpreter
    from spot_skills import _get_perception
    interpreter = Phase1Interpreter()
    executor = TaskExecutor(robot, interpreter=interpreter)

    # Warm perception singleton with shared embedder so locate() never
    # triggers a second model load during a live session.
    if connected:
        _get_perception(embedder=interpreter.embedder)
        robot._debug_camera_snapshot()

    # Single-command mode
    if single_cmd:
        run_single(single_cmd, executor)
        robot.disconnect()
        return

    # Interactive REPL
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
                print(f"\n  Robot connected : {robot.connected}")
                print(f"  Mode            : {mode}\n")
                continue

            run_single(command, executor)

    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()