"""Planner Agent for next action generation.

This module generates next actions for task execution.
"""

from typing import Any, Dict, List, Optional

from browser_env.utils import Observation
from llms import lm_config

from .planner.current_state_analyzer import CurrentStateAnalyzer


class PlannerAgent:
    """Generates next actions for task execution.

    Responsibilities:
    1. Generate next atomic action based on current task state
    """

    def __init__(self, lm_config: lm_config.LMConfig) -> None:
        self.lm_config = lm_config
        self.state_analyzer = CurrentStateAnalyzer(lm_config)

    def decompose_task(
        self,
        user_goal: str,
        context_summary: Dict[str, Any],
        current_observation: Optional[Observation] = None,
    ) -> Dict[str, Any]:
        """Initialize task for planning (simplified - no actual decomposition).

        Args:
            user_goal: Original user goal/task description
            context_summary: Current context from Context Agent
            current_observation: Current page observation

        Returns:
            Dictionary containing task info
        """
        # Simplified: no decomposition, just return task info
        return {
            "task_goal": user_goal,
            "reasoning": f"Task initialized: {user_goal}",
        }

    def generate_next_action(
        self,
        user_goal: str,
        context_summary: Dict[str, Any],
        current_observation: Optional[Observation] = None,
    ) -> Dict[str, Any]:
        """Generate the next atomic action based on current task state.

        Args:
            user_goal: Original user goal/task description
            context_summary: Current context from Context Agent
            current_observation: Current page observation

        Returns:
            Dictionary containing next action and metadata
        """
        # Analyze current state to determine next action
        state_analysis = self.state_analyzer.analyze_current_state(
            user_goal=user_goal,
            current_observation=current_observation or {"text": ""},
            context_summary=context_summary,
        )

        next_atomic_action = state_analysis.get("next_atomic_action", "")
        reasoning = state_analysis.get("reasoning", "")
        response = state_analysis.get("response", "")

        # Use atomic action as intention
        if next_atomic_action:
            selected_intention = next_atomic_action
        else:
            selected_intention = f"Continue working on: {user_goal}"

        return {
            "intention": selected_intention,
            "next_atomic_action": next_atomic_action,
            "reasoning": reasoning,
            "state_analysis": state_analysis,
            "user_goal": user_goal,
            "response": response,
        }

    def generate_intention(
        self,
        user_goal: str,
        context_summary: Dict[str, Any],
        current_observation: Optional[Observation] = None,
        previous_intentions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Generate the next execution intention based on current state.

        This is the main entry point called by Coordinator.

        Args:
            user_goal: Original user goal/task description
            context_summary: Current context from Context Agent
            current_observation: Current page observation
            previous_intentions: List of intentions already completed (not used)

        Returns:
            Dictionary containing selected intention and metadata
        """
        # Generate next action directly
        return self.generate_next_action(user_goal, context_summary, current_observation)

    def reset_planning_state(self) -> None:
        """Reset planning state for a new task."""
        # No state to reset in simplified version
        pass
