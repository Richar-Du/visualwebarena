"""Actor Agent for executing high-level intentions with specific browser actions."""

from typing import Any, Dict, List, Optional

from PIL import Image

from browser_env import Trajectory
from browser_env.utils import Observation
from llms import lm_config

from agent import PromptAgent  # Import existing PromptAgent
from .actor.action_executor import ActionExecutor


class ActorAgent(PromptAgent):
    """Executes high-level intentions using specific browser actions.

    Extends the existing PromptAgent to work with high-level intentions from
    the Planner Agent while maintaining compatibility with the existing codebase.
    """

    def __init__(
        self,
        action_set_tag: str,
        lm_config: lm_config.LMConfig,
        prompt_constructor,
        captioning_fn=None,
    ) -> None:
        """Initialize Actor Agent with enhanced capabilities."""
        # Initialize parent PromptAgent with existing parameters
        super().__init__(
            action_set_tag=action_set_tag,
            lm_config=lm_config,
            prompt_constructor=prompt_constructor,
            captioning_fn=captioning_fn,
        )

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
            images: Optional input images

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
            # Create a simple intention message that works with the existing prompt system
            intention_message = f"Execute browser actions to fulfill this intention: {intention}"
            
            # Use existing PromptAgent's next_action method with the intention message
            try:
                action = self.next_action(
                    trajectory=trajectory,
                    intent=intention_message,
                    meta_data=meta_data or {},
                    images=images,
                    output_response=True,
                )
            except Exception as next_action_error:
                import traceback
                traceback.print_exc()
                print(f"🎬 Actor Error: {str(next_action_error)[:200]}")
                print(f"🎬 Error Type: {type(next_action_error).__name__}")
                raise next_action_error

            # Extract LLM raw response from action
            llm_response = action.get("raw_prediction", "No LLM response available")

            # Validate the generated action (execution will be handled externally)
            validation_result = self.action_executor.validate_action(action)

            # Record validation results
            execution_record.update({
                "generated_action": action,
                "validation_result": validation_result,
                "llm_response": llm_response,
                # intention_fulfilled will be determined after actual execution
            })

            # Store in intention history
            self.intention_history.append(execution_record)

            return {
                "action": action,
                "validation_result": validation_result,
                "intention": intention,
                # intention_fulfilled will be determined by actual browser execution
                "intention_fulfilled": False,  # Default to False, will be updated after execution
                "execution_history_length": len(self.intention_history),
                "llm_response": llm_response,
                "response": f"LLM Response: {llm_response[:200]}{'...' if len(llm_response) > 200 else ''}",
            }

        except Exception as e:
            # Provide more detailed error information
            error_details = str(e)
            if "prompt_constructor" in error_details.lower():
                error_details += " (Prompt constructor issue)"
            elif "next_action" in error_details.lower():
                error_details += " (next_action method failure)"
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