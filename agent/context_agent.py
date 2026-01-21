"""Context Agent for managing global state and progress tracking."""

from typing import Any, Dict, List, Optional

import torch

from browser_env import Action, Trajectory
from browser_env.utils import Observation
from llms import lm_config

from .context.summary_generator import SummaryGenerator
from .context import StateManager
from .prompts.prompt_loader import generate_llm_prompt_from_template
from .memory import MemoryBank, MemoryGenerator


class ContextAgent:
    """Manages global state and context summarization.

    Responsible for maintaining task execution history and generating
    comprehensive context summaries for other agents.
    """

    def __init__(self, lm_config: lm_config.LMConfig, memory_config: Dict[str, Any]) -> None:
        self.lm_config = lm_config
        self.state_manager = StateManager()
        self.summary_generator = SummaryGenerator(lm_config)


        # Initialize memory system
        self.enable_memory = memory_config.get("enable_memory", False)
        self.enable_memory_store = memory_config.get("enable_memory_store", False)
        self.memory_content = ""
        self.current_summary = None
        if self.enable_memory or self.enable_memory_store:
            device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
            self.memory_bank = MemoryBank(
                memory_dir=memory_config.get('memory_dir', 'agent_memories'),
                embedding_model=memory_config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2'),
                top_k=memory_config.get('top_k', 3),
                device=device
            )
            self.memory_generator = MemoryGenerator(lm_config)
            self.window_size = memory_config.get('window_size', 3)
        
        self.url_history: List[str] = []
        self.last_url: str = ""
        self.page_changed_since_action: bool = False

    def reset(self) -> None:
        """Reset context agent state for a new task."""
        self.state_manager = StateManager()
        self.current_summary = None
        self.memory_content = ""
        self.url_history: List[str] = []
        self.last_url: str = ""
        self.page_changed_since_action: bool = False

    def update_state(self,
        current_observation: Optional[Observation] = None,
        latest_intention: Optional[str] = None,
        latest_action: Optional[Action] = None,
        latest_reflection: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update context state."""
        if current_observation:
            self.state_manager.add_observation(current_observation)

        if latest_intention:
            self.state_manager.add_intention(latest_intention)

        if latest_action:
            self.state_manager.add_action(latest_action)

        if latest_reflection:
            self.state_manager.add_reflection(latest_reflection)

    def update_context(
        self,
        trajectory: Trajectory,
        user_goal: str,
        current_observation: Optional[Observation] = None,
        latest_intention: Optional[str] = None,
        latest_action: Optional[Action] = None,
        latest_reflection: Optional[Dict[str, Any]] = None,
        current_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update context state and generate comprehensive summary.

        Args:
            trajectory: Current execution trajectory
            user_goal: Original user goal/task
            current_observation: Latest page observation
            latest_intention: Most recent intention
            latest_action: Most recent action taken
            latest_reflection: Most recent reflection from Reflector Agent
            current_url: Current page URL for state tracking

        Returns:
            Dictionary containing updated context information
        """
        # Use data from Coordinator if provided, otherwise extract from trajectory
        self.update_state(current_observation, latest_intention, latest_action, latest_reflection)

        # URL change detection
        url_changed = False
        url_change_info = ""
        if current_url:
            if self.last_url and current_url != self.last_url:
                url_changed = True
                self.page_changed_since_action = True
                url_change_info = f"[PAGE CHANGED] URL changed from '{self._truncate_url(self.last_url)}' to '{self._truncate_url(current_url)}'"
                print(f"🔄 {url_change_info}")
            self.url_history.append(current_url)
            self.last_url = current_url

        observations = self.state_manager.get_all_observations()
        actions = self.state_manager.get_all_actions()
        reflections = self.state_manager.get_all_reflections()
        intentions = self.state_manager.get_all_intentions()

        # Generate context summary using data from Coordinator
        summary, all_history_intentions, all_history_actions = self.summary_generator.generate_summary(
            user_goal=user_goal,
            current_summary=self.current_summary,
            observations=observations,
            actions=actions,
            intentions=intentions,
        )
        
        # Prepend URL change info to summary if page changed
        if url_changed and url_change_info:
            summary = f"{url_change_info}\n{summary}"
        
        self.current_summary = summary


        # Return comprehensive context information
        return {
            "summary": summary,
            "memory_content": self.memory_content,
            "history_intentions": all_history_intentions,
            "history_actions": all_history_actions,
            "state_history": {
                "observations": observations,
                "actions": actions,
                "reflections": reflections,
                "intentions": intentions,
                "total_steps": len(actions),
            },
            "latest_observation": current_observation,
            "latest_intention": latest_intention,
            "latest_action": latest_action,
            # URL tracking info
            "url_changed": url_changed,
            "current_url": current_url,
            "url_history": self.url_history[-5:],  # Last 5 URLs
        }
    
    def _truncate_url(self, url: str, max_length: int = 60) -> str:
        """Truncate URL for display."""
        if len(url) <= max_length:
            return url
        return url[:max_length] + "..."

    
    def _extract_observations_from_trajectory(self, trajectory: Trajectory) -> List[Observation]:
        """Extract observations from trajectory (fallback method)."""
        observations = []
        for item in trajectory:
            if isinstance(item, dict) and "observation" in item:
                observations.append(item["observation"])
        return observations
    
    def _extract_actions_from_trajectory(self, trajectory: Trajectory) -> List[Action]:
        """Extract actions from trajectory (fallback method)."""
        actions = []
        for item in trajectory:
            if isinstance(item, dict) and "action_type" in item:
                actions.append(item)
        return actions

    def reset(self) -> None:
        """Reset all context state for a new task."""
        self.state_manager.clear()

    def get_current_state(self) -> Dict[str, Any]:
        """Get current context state without updating."""
        history = self.state_manager.get_history()
        return {
            "state_history": history,
            "latest_observation": self.state_manager.get_latest_observation(),
            "latest_action": self.state_manager.get_latest_action(),
            "latest_intention": self.state_manager.get_latest_intention(),
            "total_steps": history.get("total_steps", 0),
        }

    def check_task_completion(self, user_goal: str, actions: Optional[List[Action]] = None) -> bool:
        """Check if the task is considered complete based on current state.

        This is a simplified check - task completion is primarily determined
        by reaching max_steps or explicit STOP action in the coordinator.

        Args:
            user_goal: Original user goal
            actions: List of actions (optional, uses state_manager if not provided)

        Returns:
            True if task appears complete, False otherwise
        """
        # Task completion is determined by the coordinator based on:
        # 1. Max steps reached
        # 2. Actor generating a STOP action
        # This method provides a basic fallback check
        if actions is None:
            history = self.state_manager.get_history()
            actions = history.get("actions", [])
        
        # Check if the last action was a STOP action
        if actions and actions[-1].get("action_type") == "STOP":
            return True
        
        return False

    def initialize_task_memory(self, user_goal: str) -> Dict[str, Any]:
        """Initialize memory tracking for a new task.

        Args:
            user_goal: The user's goal/task description

        Returns:
            Relevant memories from past similar tasks
        """
        if not self.enable_memory:
            self.memory_content = ""
            return {"memory_content": "", "relevant_memories": []}

        # Store user goal in state manager for memory generation
        self.state_manager.set_user_goal(user_goal)

        # Retrieve relevant memories for this task
        try:
            relevant_memories = self.memory_bank.search_memories(query=user_goal)

            self.memory_content = self._get_mem_str(relevant_memories)
            return {
                "memory_content": self.memory_content,
                "relevant_memories": relevant_memories,
            }
        except Exception as e:
            print(f"Error retrieving memories: {e}")
            self.memory_content = ""
            return {
                "memory_content": "",
                "relevant_memories": [],
            }

    def _get_mem_str(self, memories: List[Dict[str, Any]]) -> str:
        """Convert list of memories to a formatted string.

        Args:
            memories: List of memory dictionaries from memory bank

        Returns:
            Formatted string of memories
        """
        if not memories:
            return ""

        mem_str = ""
        for i, mem in enumerate(memories):
            mem_str += f"Memory {i+1}:\n"
            mem_str += f"Title: {mem['title']}\n"
            mem_str += f"Description: {mem['description']}\n"
            mem_str += f"Content: {mem['content']}\n\n"

        return mem_str.strip()

    def generate_and_store_memory(self, task_completed: bool) -> Optional[str]:
        """Generate and store memory from completed task.

        Args:
            task_completed: Whether the task was completed successfully

        Returns:
            Memory ID if successful, None otherwise
        """
        if not self.enable_memory_store:
            return None

        try:
            # Get user goal from state manager
            user_goal = self.state_manager.get_user_goal()

            # Generate memory from trajectory using state manager data
            memory_data = self.memory_generator.generate_memory_from_trajectory(
                user_goal=user_goal,
                observations=self.state_manager.get_all_observations(),
                intentions=self.state_manager.get_all_intentions(),
                actions=self.state_manager.get_all_actions(),
                task_completed=task_completed,
                window_size=self.window_size,
            )

            # Store memory in bank
            memory_id = self.memory_bank.add_memory(
                title=memory_data["title"],
                description=memory_data["description"],
                content=memory_data["content"],
                success_rate=memory_data.get("success_rate", 0.0),
            )

            print(f"💾 Memory stored: {memory_data['title'][:50]}... (ID: {memory_id})")
            return memory_id

        except Exception as e:
            print(f"Error generating memory: {e}")
            return None

