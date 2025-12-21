"""Reflector Agent for structured execution validation using checklist approach.

This module provides reflection capabilities including:
1. Checklist-based analysis with subtask completion detection
2. Subtask revision generation when needed
"""

from typing import Any, Dict, List, Optional
import numpy as np

from browser_env import Action, Trajectory
from browser_env.utils import Observation
from llms import lm_config

from .reflector.checklist_analyzer import ChecklistAnalyzer
from .reflector.subtask_reviser import SubtaskReviser


class ReflectorAgent:
    """Unified reflector agent using checklist-based execution analysis.

    Performs structured validation through 5 key checks:
    1. Pattern Check: Detect repetitive or erroneous patterns in recent intents
    2. Execution Check: Verify if the latest action executed successfully
    3. Subtask Completion Check: Determine if current subtask is completed (NEW)
    4. Subtask Revision Check: Determine if current subtask needs revision
    5. Task Completion Check: Check if the overall task is completed
    """

    def __init__(self, lm_config: lm_config.LMConfig) -> None:
        self.lm_config = lm_config
        self.checklist_analyzer = ChecklistAnalyzer(lm_config)
        self.subtask_reviser = SubtaskReviser(lm_config)

        # Reflection history
        self.reflection_history: List[Dict[str, Any]] = []

    def reflect_execution(
        self,
        trajectory: Trajectory,
        intentions: List[str],
        actions: List[Action],
        current_intention: str,
        latest_action: Action,
        current_observation: Observation,
        context_summary: Dict[str, Any],
        current_subtask: Optional[str] = None,
        high_level_task: Optional[str] = None,
        all_subtasks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Reflect on execution using structured checklist approach.

        Args:
            trajectory: Current execution trajectory
            intentions: List of all intentions so far
            actions: List of all actions executed so far
            current_intention: The intention that was being fulfilled
            latest_action: The most recently executed action
            current_observation: The observation after action execution
            context_summary: Current context from Context Agent
            current_subtask: Current subtask being worked on (from SubtaskManager)
            high_level_task: Original high-level task goal
            all_subtasks: List of all subtasks (for revision)

        Returns:
            Dictionary containing checklist results and metadata
        """
        try:
            # Prepare inputs for checklist
            recent_intents = intentions[-5:] if intentions else []
            image_before, image_after = self._extract_before_after_images(
                trajectory, current_observation
            )

            # Run unified checklist analysis
            checklist_result = self.checklist_analyzer.analyze(
                recent_intents=recent_intents,
                image_before=image_before,
                image_after=image_after,
                latest_action=latest_action,
                current_subtask=current_subtask or current_intention,
                high_level_task=high_level_task or context_summary.get("summary", ""),
            )

            # Build reflection result
            reflection = {
                "checklist": checklist_result,
                "has_pattern_issue": checklist_result.get("has_pattern_issue", False),
                "execution_successful": checklist_result.get("execution_successful", True),
                "subtask_completed": checklist_result.get("subtask_completed", False),  # NEW
                "subtask_needs_revision": checklist_result.get("subtask_needs_revision", False),
                "task_completed": checklist_result.get("task_completed", False),
                "current_intention": current_intention,
                "latest_action": latest_action,
                "reflection_number": len(self.reflection_history) + 1,
            }

            # Store in reflection history
            self.reflection_history.append(reflection)

            return reflection

        except Exception as e:
            # Create error reflection with safe defaults
            error_reflection = {
                "checklist": {
                    "has_pattern_issue": False,
                    "execution_successful": False,
                    "subtask_completed": False,  # NEW
                    "subtask_needs_revision": False,
                    "task_completed": False,
                },
                "has_pattern_issue": False,
                "execution_successful": False,
                "subtask_completed": False,  # NEW
                "subtask_needs_revision": False,
                "task_completed": False,
                "current_intention": current_intention,
                "latest_action": latest_action,
                "reflection_number": len(self.reflection_history) + 1,
                "error": str(e),
            }

            self.reflection_history.append(error_reflection)
            return error_reflection

    def generate_revised_subtask(
        self,
        high_level_task: str,
        current_subtask: str,
        all_subtasks: List[str],
        intentions: List[str],
        current_observation: Observation,
    ) -> Dict[str, Any]:
        """Generate a revised subtask when current subtask needs revision.

        This method should be called when subtask_needs_revision=True.

        Args:
            high_level_task: Original high-level task goal
            current_subtask: Current subtask that needs revision
            all_subtasks: List of all subtasks
            intentions: List of all intentions so far
            current_observation: Current page observation

        Returns:
            Dictionary containing:
                - revised_subtask: The new revised subtask string
                - reasoning: Explanation for the revision
                - success: Whether revision generation succeeded
        """
        print(f"🔄 Reflector Agent: Generating revised subtask...")
        
        result = self.subtask_reviser.generate_revised_subtask(
            high_level_task=high_level_task,
            current_subtask=current_subtask,
            all_subtasks=all_subtasks,
            recent_intents=intentions[-5:] if intentions else [],
            current_observation=current_observation,
        )

        if result.get("success"):
            print(f"🔄 Revised subtask: {result['revised_subtask'][:80]}...")
        else:
            print(f"⚠️  Subtask revision failed, keeping original: {current_subtask[:80]}...")

        return result

    def _extract_before_after_images(
        self, trajectory: Trajectory, current_observation: Observation
    ) -> tuple:
        """Extract before and after images from trajectory.

        Returns:
            tuple: (image_before, image_after) - numpy arrays or None
        """
        image_before = None
        image_after = None

        # Get current observation image (after)
        image_after = current_observation.get("image_raw")
        if image_after is None:
            image_after = current_observation.get("image")

        # Get previous observation image (before) from trajectory
        # Trajectory structure: [StateInfo, Action, StateInfo, Action, ...]
        state_infos = [item for item in trajectory if isinstance(item, dict) and 'observation' in item]

        if len(state_infos) >= 2:
            # Get second-to-last state (before the action)
            obs_before = state_infos[-2].get("observation", {})
            image_before = obs_before.get("image_raw")
            if image_before is None:
                image_before = obs_before.get("image")
        elif len(state_infos) == 1:
            # First step - use the same image for before
            obs_before = state_infos[0].get("observation", {})
            image_before = obs_before.get("image_raw")
            if image_before is None:
                image_before = obs_before.get("image")

        return image_before, image_after

    def reset_reflection_history(self) -> None:
        """Reset reflection history for a new task."""
        self.reflection_history.clear()

    def get_latest_checklist(self) -> Optional[Dict[str, bool]]:
        """Get the most recent checklist result.

        Returns:
            Latest checklist dictionary or None if no reflections exist
        """
        if self.reflection_history:
            return self.reflection_history[-1].get("checklist")
        return None
