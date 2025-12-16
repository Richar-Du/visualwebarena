"""Context manager for agent execution history and progress tracking."""

from typing import Any, Dict, List

from browser_env import Action
from browser_env.utils import Observation


class StateManager:
    """Manages execution history for context awareness."""

    def __init__(self) -> None:
        self.observations: List[Observation] = []
        self.actions: List[Action] = []
        self.reflections: List[Dict[str, Any]] = []
        self.intentions: List[str] = []
        self.user_goal: str = ""

    def add_observation(self, observation: Observation) -> None:
        """Add a new observation to the history."""
        self.observations.append(observation)

    def add_action(self, action: Action) -> None:
        """Add a new action to the history."""
        self.actions.append(action)

    def add_reflection(self, reflection: Dict[str, Any]) -> None:
        """Add a new reflection to the history."""
        self.reflections.append(reflection)

    def add_intention(self, intention: str) -> None:
        """Add a new intention to the history."""
        self.intentions.append(intention)

    def get_all_observations(self) -> List[Observation]:
        """Get all observations."""
        return self.observations

    def get_all_actions(self) -> List[Action]:
        """Get all actions."""
        return self.actions

    def get_all_reflections(self) -> List[Dict[str, Any]]:
        """Get all reflections."""
        return self.reflections

    def get_all_intentions(self) -> List[str]:
        """Get all intentions."""
        return self.intentions

    def get_latest_observation(self) -> Observation:
        """Get the most recent observation."""
        return self.observations[-1] if self.observations else None

    def get_latest_action(self) -> Action:
        """Get the most recent action."""
        return self.actions[-1] if self.actions else None

    def get_latest_reflection(self) -> Dict[str, Any]:
        """Get the most recent reflection."""
        return self.reflections[-1] if self.reflections else None

    def get_latest_intention(self) -> str:
        """Get the most recent intention."""
        return self.intentions[-1] if self.intentions else None

    def get_history(self) -> Dict[str, Any]:
        """Get complete execution history."""
        return {
            "observations": self.observations,
            "actions": self.actions,
            "reflections": self.reflections,
            "intentions": self.intentions,
            "total_steps": len(self.actions),
            "total_observations": len(self.observations),
            "total_reflections": len(self.reflections),
            "total_intentions": len(self.intentions),
        }

    def set_user_goal(self, user_goal: str) -> None:
        """Set the user goal for this task."""
        self.user_goal = user_goal

    def get_user_goal(self) -> str:
        """Get the user goal for this task."""
        return self.user_goal

    def clear(self) -> None:
        """Clear all history."""
        self.observations.clear()
        self.actions.clear()
        self.reflections.clear()
        self.intentions.clear()
        self.user_goal = ""