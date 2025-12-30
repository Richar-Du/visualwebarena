"""Actor Agent for executing high-level intentions with specific browser actions."""

from typing import Any, Dict, List, Optional

from PIL import Image

from browser_env import Trajectory
from browser_env.utils import Observation
from llms import lm_config

from .actor.action_executor import ActionExecutor
from .actor.browser_action_executor import BrowserActionExecutor


class ActorAgent:
    """Executes high-level intentions using specific browser actions.

    Uses the new prompt system from multi_agent_prompts_fixed.json to generate
    browser actions from high-level intentions.
    """

    def __init__(
        self,
        action_set_tag: str,
        lm_config: lm_config.LMConfig,
        prompt_constructor=None,  # Kept for compatibility but not used
        captioning_fn=None,  # Kept for compatibility but not used
    ) -> None:
        """Initialize Actor Agent with the new prompt system."""
        self.action_set_tag = action_set_tag
        self.lm_config = lm_config

        # Initialize new browser action executor with the new prompt system
        self.browser_action_executor = BrowserActionExecutor(lm_config, action_set_tag)

        # Initialize action executor for validation and tracking
        self.action_executor = ActionExecutor(action_set_tag)

        # Track intention execution history
        self.intention_history: List[Dict[str, Any]] = []

    def execute_intention(
        self,
        intention: str,
        current_observation: Observation,
        trajectory: Trajectory,
        meta_data: Optional[Dict[str, Any]] = None,
        images: Optional[List[Image.Image]] = None,
    ) -> Dict[str, Any]:
        """Execute a high-level intention and generate specific actions.

        Args:
            intention: High-level intention from Planner Agent
            current_observation: Current page observation
            trajectory: Current execution trajectory
            meta_data: Additional metadata for execution
            images: Optional input images (not used, kept for compatibility)

        Returns:
            Dictionary containing execution results
        """
        # Record intention execution attempt
        execution_record = {
            "intention": intention,
            "timestamp": None,  # Would be set in actual implementation
            "observation_before": current_observation,
        }

        try:
            # Use the new BrowserActionExecutor to generate action
            result = self.browser_action_executor.execute_action(
                intention=intention,
                trajectory=trajectory,
                meta_data=meta_data or {},
                images=images,
            )

            # Extract action, LLM response, and extracted intention
            action = result["action"]
            llm_response = result["llm_response"]
            extracted_intention = result.get("extracted_intention", intention)

            # Validate the generated action (execution will be handled externally)
            validation_result = self.action_executor.validate_action(action)

            # Record validation results
            execution_record.update({
                "generated_action": action,
                "validation_result": validation_result,
                "llm_response": llm_response,
                "extracted_intention": extracted_intention,
                # intention_fulfilled will be determined after actual execution
            })

            # Store in intention history
            self.intention_history.append(execution_record)

            return {
                "action": action,
                "validation_result": validation_result,
                "intention": intention,  # Original high-level intention (user_goal)
                "extracted_intention": extracted_intention,  # LLM's reasoning from <think> tags
                # intention_fulfilled will be determined by actual browser execution
                "intention_fulfilled": False,  # Default to False, will be updated after execution
                "execution_history_length": len(self.intention_history),
                "llm_response": llm_response,
                "response": f"LLM Response: {llm_response[:200]}{'...' if len(llm_response) > 200 else ''}",
            }

        except Exception as e:
            # Provide more detailed error information
            error_details = str(e)
            if "prompt" in error_details.lower():
                error_details += " (Prompt system issue)"
            elif "action" in error_details.lower():
                error_details += " (Action generation failure)"
            elif "traject" in error_details.lower():
                error_details += " (Trajectory processing issue)"

            # Record failed execution
            execution_record.update({
                "error": error_details,
                "exception_type": type(e).__name__,
            })
            self.intention_history.append(execution_record)

            return {
                "error": error_details,
                "intention": intention,
                "intention_fulfilled": False,
                "exception_type": type(e).__name__,
                "response": f"Execution failed: {error_details}",
            }

    def reset_intention_history(self) -> None:
        """Reset intention execution history for a new task."""
        self.intention_history.clear()
        self.action_executor.reset_execution_history()

    def get_recent_intentions(self, count: int = 5) -> List[Dict[str, Any]]:
        """Get the most recent intention executions.

        Args:
            count: Number of recent intentions to return

        Returns:
            List of recent intention execution records
        """
        return self.intention_history[-count:] if self.intention_history else []