"""
executor.py
-----------
Phase IV: Autonomous Task Execution

Walks a Phase II task graph (from interpret.py) step by step,
dispatches each node ot tthe matching SPOT skill, monitors success,
and triggers recovery or clarification on failure.

Usage:
    from interpret import Phas1Interpreter
    form executor import TaskExecutor
    from spt_skills import SpotRobot

    robot   = SpotRobot()
    robot.connect()
    executor = TaskExecutor(robot)

    result = executor. execute("Bring me something to write with")
    print(result)

    robot.disconnect()

Dry-run (no SPOT):
    USE_SPOT = false python executor.py "Go to the desk and bring the notebook"
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from interpret import Phase1Interpreter
from spot_skills import(
    SKILL_REGISTRY,
    SkillResult,
    SpotRobot,
    navigate,
    scan,
    locate,
    pick_up,
    deliver,
    release,
)

# ── Configuration ──────────────────────────────────────────────────────────────

MAX_RETRIES: int    = 2     # Retry a failed skill this many times before giving up
RETRY_DELAY: float  = 1.5   # Seconds between retries

# ── Intent → skill sequence mapping ──────────────────────────────────────────
# Maps a whole-command intent label to an ordered list of skill names.
# Used when the task graph is a single-clause command.
INTENT_SKILL_MAP: Dict[str, List[str]] = {
    "navigate":               ["navigate"],
    "scan_environment":       ["scan"],
    "locate_object":          ["locate"],
    "retrieve_object":        ["locate", "pick_up", "deliver", "release"],
    "multi_step_retrieve":    ["navigate", "locate", "pick_up", "deliver", "release"],
    "multi_step_manipulation":["locate", "pick_up", "navigate", "deliver", "release"],
}

# Nouns that identify a place (waypoint) rather than a retrieve target.
# Used by _extract_params to separate navigation destinations from objects.
LOCATION_NOUNS: frozenset = frozenset({
    "desk", "table", "kitchen", "room", "shelf",
    "door", "counter", "lab", "office", "hallway", "closet",
})

# Pronouns that spaCy may surface as object entities (e.g. "me" lemmatizes to "I").
# These are never valid retrieve targets and must be excluded from explicit objects.
PRONOUNS: frozenset = frozenset({
    "i", "me", "my", "mine", "myself",
    "you", "your", "yours", "yourself",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "it", "its", "itself",
    "we", "us", "our", "ours", "ourselves",
    "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "whose", "where", "when", "how", "why",
})

# ── Result types ──────────────────────────────────────────────────────────────
@dataclass
class StepLog:
    step_id:    str
    skill:      str
    success:    bool
    message:    str
    retries:    int = 0
    mock:       bool = False

@dataclass
class ExecutionResult:
    """
    Full result of executing a natural langauge command.
    success:    True if all required stesps completed successsfully.
    command:    Original natural language command.
    steps:      Log of eah skill execution.
    recovered:  Number of steps that succeeded after at least one retry.
    clarifications_needed:  Questions raised before  during execution.
    """
    success:                bool
    command:                str
    steps:                  List[StepLog] = field(default_factory=list)
    recovered:              int = 0
    clarifications_needed:  List[str] = field(default_factory=list)
    error:                  Optional[str] = None

    def summary(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        lines = [
            f"[{status}] \"{self.command}\"",
            f"  Steps: {len(self.steps)}    |   Recovered: {self.recovered}",
        ]
        for s in self.steps:
            icon = "o" if s.success else "x"
            retry_str = f"  (retry = {s.retries})" if s.retries else ""
            mock_str  = " [mock]" if s.mock else ""
            lines.append(f"     {icon} {s.step_id} {s.skill:<14} {s.message}{retry_str}{mock_str}")
        if self.clarifications_needed:
            lines.append("  Clarifications:")
            for q in self.clarifications_needed:
                lines.append(f"      {q}")
        if self.error:
            lines.append(f"  Error: {self.error}")
        return "\n".join(lines)
    
# ── Executor ──────────────────────────────────────────────────────────────────

class TaskExecutor:
    """
    Executes a natural language command on SPOT by:
        1. Parsing the command with Phase1Interpreter
        2. Checking for required clarifications before execution
        3. Walking the Phase II task graph clause by clause
        4. Dispatching each clause's intent to the SPOT skill registry
        5. Monitoring success and retrying or replanning on failure
    """

    def __init__(
            self,
            robot: SpotRobot,
            interpreter: Optional[Phase1Interpreter] = None,
            max_retries: int = MAX_RETRIES,
    ):
        self.robot = robot
        self.inter = interpreter or Phase1Interpreter()
        self.max_retries = max_retries

    # ── Parameter extraction ──────────────────────────────────────────────────

    def _extract_params(self, intent: str, parsed: Dict) -> Dict[str, Any]:
        """
        Pull execution parameters out of the interpreted command.
        Returns keyword arguments for the matching skill function.

        Key design decitions:
        - Location-type nouns (desk, table, kitchen...) are separated from retrieve-target nouns so that "Go to the desk and bring the notebook"
          correctly yields waypoint_id = 'desk' and object_label = 'notebook'.
        - The waypoint fallback no longer requires a specific intent label -- if a location noun is present in the object list, it is always used.
        - locations[] entries are handled as either strings or dicts (Phase I may return either form depending on entity extraction depth).
        """
        params: Dict[str, Any] = {}
        objects = parsed.get("entities", {}).get("objects", [])

        location_objects = [o for o in objects if o["head"] in LOCATION_NOUNS]
        explicit = [
            o for o in objects 
            if o["head"].lower() not in ("something", "thing")
            and o["head"].lower() not in LOCATION_NOUNS
            and o["head"].lower() not in PRONOUNS
         ]

        # Object label: grounding best_match > first lon-location explicit noun
        grounding = parsed.get("grounding")
        if grounding and grounding.get("best_match"):
            params["object_label"] = grounding["best_match"]
        elif explicit:
            params["object_label"] = explicit[0]["head"]

        # Bounding box from grounding
        if grounding and grounding.get("top_candidates"):
            top = grounding["top_candidates"][0]
            if top.get("bbox"):
                params["bbox"] = top["bbox"]

        # Waypoint: use first location entity if available
        locations = parsed.get("entities", {}).get("locations", [])
        if locations:
            loc = locations[0]
            # locations[] entries may be plain strings or {"head": ... ...} dicts
            params["waypoint_id"] = loc["head"] if isinstance(loc, dict) else loc
        elif location_objects:
            params["waypoint_id"] = location_objects[0]["head"]


        return params
    
    # ── Single skill dispatch with retry ─────────────────────────────────────

    def _run_skill(
        self,
        step_id: str,
        skill_name: str,
        params: Dict[str, Any],
    ) -> StepLog:
        """
        Call a skill by name with retry logic.
        Returns a StepLog recording the outcome.
        """
        skill_fn = SKILL_REGISTRY.get(skill_name)
        if skill_fn is None:
            return StepLog(step_id, skill_name, False,
                           f"Skill '{skill_name}' not found in registry.")
        
        last_result: Optional[SkillResult] = None
        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                print(f"    [executor] Retrying {skill_name} (attempt {attempt + 1})...")
                time.sleep(RETRY_DELAY)

            # Call skill - pass only params it accepts
            try:
                if skill_name == "navigate":
                    waypoint = params.get("waypoint_id", "unknown_waypoint")
                    last_result = skill_fn(self.robot, waypoint)
                elif skill_name in ("locate", "pick_up"):
                    label = params.get("object_label", "unknown_object")
                    bbox = params.get("bbox")
                    if skill_name == "locate":
                        last_result = skill_fn(self.robot, label)
                    else:
                        last_result = skill_fn(self.robot, label, bbox)
                elif skill_name in ("deliver",):
                    label = params.get("object_label", "")
                    last_result = skill_fn(self.robot, label)
                else:
                    last_result = skill_fn(self.robot)
            except Exception as e:
                last_result = SkillResult(False, skill_name, f"Exception: {e}")

            if last_result.success:
                break
        
        return StepLog(
            step_id = step_id,
            skill = skill_name,
            success = last_result.success,
            message = last_result.message,
            retries = attempt,
            mock = last_result.mock,
        )

    # ── Map intent label to skill sequence ────────────────────────────────────

    def _skills_for_intent(self, intent_label: str) -> List[str]:
        return INTENT_SKILL_MAP.get(intent_label, ["scan"])

    # ── Main execution entry point ────────────────────────────────────────────

    def execute(self, command: str) -> ExecutionResult:
        """
        Parse and execute a natural lanuage command on SPOT.

        Args:
            command: Free-form natural langauge instruction.

        Returns:
            ExecutionResult with step logs and overall success status.
        """
        print(f"\n[executor] Command: \"{command}\"")

        # 1. Interpret
        parsed = self.inter.interpret(command)
        clarifications = parsed.get("clarification", {}).get("questions", [])

        # 2. Block on hard clarifications (vague object with no grounding result)
        grounding = parsed.get("grounding")
        has_vague = any(
            o["head"] in ("something", "thing")
            for o in parsed.get("entities", {}).get("objects", [])
        )
        # affordance_verb is set by the grounding layer when a functional verb is found
        has_affordance_verb = bool(
            grounding and grounding.get("affordance_verb")
        )
        grounding_resolved = (
            grounding is not None
            and grounding.get("best_match") is not None
            and (has_affordance_verb or not has_vague)
        )
        if has_vague and not grounding_resolved:
            return ExecutionResult(
                success               = False,
                command               = command,
                clarifications_needed = clarifications,
                error = "Cannot execute: object is unspecified and no affordance verb found. "
                        "Please describe what you need the object for (e.g. 'something to write with').",
            )
        
        # 3. Build execution plan from Phase II task graph
        steps_plan: List[Dict] = parsed.get("phase2", {}).get("plan", {}).get("steps", [])

        # If multi-clause, execute each clause's skill sequence in order
        execution_steps: List[Dict[str, Any]] = []
        if len(steps_plan) > 1:
            for step in steps_plan:
                intent_label = step["intent"]["label"]
                skills = self._skills_for_intent(intent_label)
                for skill in skills:
                    execution_steps.append({
                        "step_id": step["id"],
                        "skill": skill,
                        "intent": intent_label,
                    })
        else:
            # Single clause -- use whole-command intent
            overall_intent = parsed["intent"]["label"]
            for skill in self._skills_for_intent(overall_intent):
                execution_steps.append({
                    "step_id": "c1",
                    "skill": skill,
                    "intent": overall_intent,
                })

        print(f"[executor] Plan: {[s['skill'] for s in execution_steps]}")

        # 4. Execute steps
        params = self._extract_params(parsed["intent"]["label"], parsed)

        _identity_keywords = ("specific object", "look for", "describe what it",
                              "last saw", "what should I")
        if params.get("object_label"):
            clarifications = [
                q for q in clarifications
                if not any(kw in q.lower() for kw in _identity_keywords)
            ]
        logs:       List[StepLog] = []
        recovered = 0
        overall_success = True

        for s in execution_steps:
            log = self._run_skill(s["step_id"], s["skill"], params)
            logs.append(log)

            if log.retries > 0 and log.success:
                recovered += 1

            if not log.success:
                print(f"    [executor] Step {s['step_id']}:{s['skill']} FAILED -- {log.message}")
                overall_success = False

                if s["skill"] == "navigate":
                    clarifications.append(
                        "I could not navigate to the destination. "
                        "Can you specify a waypoint or guid me to the location?"
                    )
                elif s["skill"] == "pick_up":
                    clarifications.append(
                        f"I could not pick up '{params.get('object_label', 'the object')}'."
                        "Can you reposition it or confirm it is reachable?"
                    )
                break   # stop on first failure; let caller decide to replan
        
        return ExecutionResult(
            success = overall_success,
            command = command,
            steps = logs,
            recovered = recovered,
            clarifications_needed = clarifications,
        )
        

# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    cmd = " ".join(sys.argv[1:]).strip()
    if not cmd:
        cmd = input("Command: ").strip()

    robot = SpotRobot() # USE_SPOT = false -> mock mode
    executor = TaskExecutor(robot)
    result = executor.execute(cmd)

    print("\n" + "=" * 60)
    print(result.summary())
    print("=" * 60 + "\n")