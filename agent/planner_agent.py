"""Planner Agent for task decomposition and next action generation.

This module provides planning capabilities with explicit subtask state tracking.
The current subtask is tracked by SubtaskManager, not inferred by LLM.
"""

from typing import Any, Dict, List, Optional

from browser_env.utils import Observation
from llms import lm_config

from .planner.task_decomposer import TaskDecomposer
from .planner.current_state_analyzer import CurrentStateAnalyzer
from .planner.subtask_manager import SubtaskManager


class PlannerAgent:
    """Decomposes complex tasks and generates next actions for execution.

    Responsibilities:
    1. Initial task decomposition into manageable subtasks
    2. Generate next atomic action based on current subtask (from SubtaskManager)
    
    Note: The PlannerAgent no longer infers which subtask is current.
    Instead, it receives the current subtask from SubtaskManager (managed by Coordinator).
    """

    def __init__(self, lm_config: lm_config.LMConfig) -> None:
        self.lm_config = lm_config
        self.task_decomposer = TaskDecomposer(lm_config)
        self.state_analyzer = CurrentStateAnalyzer(lm_config)
        
        # SubtaskManager for explicit state tracking
        self.subtask_manager = SubtaskManager()
        
        # Task decomposition flag
        self.task_decomposed: bool = False

    def decompose_task(
        self,
        user_goal: str,
        context_summary: Dict[str, Any],
        current_observation: Optional[Observation] = None,
    ) -> Dict[str, Any]:
        """Decompose the task into subtasks (called once at the beginning).
        
        Args:
            user_goal: Original user goal/task description
            context_summary: Current context from Context Agent
            current_observation: Current page observation
            
        Returns:
            Dictionary containing decomposition results and subtask info
        """
        if self.task_decomposed:
            # Already decomposed, return current state
            return {
                "subtasks": self.subtask_manager.get_all_subtasks(),
                "current_subtask": self.subtask_manager.get_current_subtask(),
                "current_index": self.subtask_manager.get_current_index(),
                "total_subtasks": len(self.subtask_manager.get_all_subtasks()),
                "task_decomposed": True,
            }
        
        print("🎯 Planner Agent: Decomposing task...")
        decomposition_result = self.task_decomposer.decompose_task(
            user_goal=user_goal,
            current_observation=current_observation or {"text": ""},
            context_summary=context_summary,
        )
        
        subtasks = decomposition_result.get("subtasks", [])
        
        # Initialize subtask manager
        self.subtask_manager.initialize(subtasks)
        self.task_decomposed = True
        
        print(f"🎯 Task decomposed into {len(subtasks)} subtasks:")
        for i, subtask in enumerate(subtasks, 1):
            print(f"   {i}. {subtask}")
        
        return {
            "subtasks": subtasks,
            "current_subtask": self.subtask_manager.get_current_subtask(),
            "current_index": self.subtask_manager.get_current_index(),
            "total_subtasks": len(subtasks),
            "task_decomposed": True,
            "reasoning": decomposition_result.get("reasoning", ""),
        }

    def generate_next_action(
        self,
        user_goal: str,
        context_summary: Dict[str, Any],
        current_observation: Optional[Observation] = None,
    ) -> Dict[str, Any]:
        """Generate the next atomic action based on current subtask.
        
        The current subtask is obtained from SubtaskManager (explicit tracking),
        NOT inferred by LLM from observation.
        
        Args:
            user_goal: Original user goal/task description
            context_summary: Current context from Context Agent
            current_observation: Current page observation
            
        Returns:
            Dictionary containing next action and metadata
        """
        # Get current subtask from manager (explicit, not inferred)
        current_subtask = self.subtask_manager.get_current_subtask()
        
        if not current_subtask:
            # No more subtasks, task might be complete
            return {
                "intention": f"Complete the task: {user_goal}",
                "next_atomic_action": f"Verify task completion for: {user_goal}",
                "current_subtask": "",
                "reasoning": "All subtasks completed, verifying final state",
                "all_subtasks": self.subtask_manager.get_all_subtasks(),
                "current_step_index": self.subtask_manager.get_current_index(),
                "total_subtasks": len(self.subtask_manager.get_all_subtasks()),
            }
        
        print(f"🎯 Planner Agent: Generating action for subtask: {current_subtask[:80]}...")
        
        # Analyze current state to determine next action
        # Note: We now pass current_subtask explicitly, not asking LLM to infer it
        state_analysis = self.state_analyzer.analyze_current_state(
            user_goal=user_goal,
            current_subtask=current_subtask,  # Explicit, not inferred
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
            selected_intention = f"Continue working on: {current_subtask}"
        
        return {
            "intention": selected_intention,
            "next_atomic_action": next_atomic_action,
            "current_subtask": current_subtask,
            "reasoning": reasoning,
            "all_subtasks": self.subtask_manager.get_all_subtasks(),
            "current_step_index": self.subtask_manager.get_current_index(),
            "total_subtasks": len(self.subtask_manager.get_all_subtasks()),
            "task_decomposed": self.task_decomposed,
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
        Handles both initial decomposition and subsequent action generation.

        Args:
            user_goal: Original user goal/task description
            context_summary: Current context from Context Agent
            current_observation: Current page observation
            previous_intentions: List of intentions already completed (not used)

        Returns:
            Dictionary containing selected intention and metadata
        """
        # Step 1: Perform task decomposition if not done
        if not self.task_decomposed:
            self.decompose_task(user_goal, context_summary, current_observation)
        
        # Step 2: Generate next action based on current subtask
        return self.generate_next_action(user_goal, context_summary, current_observation)

    def get_subtask_manager(self) -> SubtaskManager:
        """Get the subtask manager for external access.
        
        Returns:
            SubtaskManager instance
        """
        return self.subtask_manager

    def mark_current_subtask_completed(self) -> bool:
        """Mark the current subtask as completed and advance.
        
        Called by Coordinator when Reflector confirms subtask completion.
        
        Returns:
            True if advanced to next subtask, False if all completed
        """
        return self.subtask_manager.mark_current_completed()

    def revise_current_subtask(self, revised_subtask: str) -> None:
        """Revise the current subtask with a new version.
        
        Called by Coordinator when Reflector provides a revised subtask.
        
        Args:
            revised_subtask: The revised subtask
        """
        self.subtask_manager.revise_current_subtask(revised_subtask)

    def get_all_subtasks(self) -> List[str]:
        """Get all subtasks.
        
        Returns:
            List of all subtasks
        """
        return self.subtask_manager.get_all_subtasks()

    def get_current_subtask(self) -> str:
        """Get current subtask.
        
        Returns:
            Current subtask string
        """
        return self.subtask_manager.get_current_subtask()

    def all_subtasks_completed(self) -> bool:
        """Check if all subtasks are completed.
        
        Returns:
            True if all subtasks completed
        """
        return self.subtask_manager.all_completed()

    def reset_planning_state(self) -> None:
        """Reset planning state for a new task."""
        self.subtask_manager.reset()
        self.task_decomposed = False
