"""Task decomposition for planning Agent with multimodal support."""

from typing import Any, Dict, List
import numpy as np

from PIL import Image

from browser_env.utils import Observation, pil_to_b64
from llms import lm_config, call_llm
from ..prompts.prompt_loader import load_prompt_template
from ..utils import is_multimodal_model


class TaskDecomposer:
    """Decomposes complex tasks into manageable subtasks with multimodal support."""

    def __init__(self, lm_config: lm_config.LMConfig) -> None:
        self.lm_config = lm_config
        # Check if the model supports multimodal inputs
        self.is_multimodal = is_multimodal_model(lm_config.model)

    def decompose_task(
        self,
        user_goal: str,
        current_observation: Observation,
        context_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Decompose user goal into 3-5 manageable subtasks.

        Args:
            user_goal: Original user goal
            current_observation: Current page observation (includes image_raw for visual analysis)
            context_summary: Context summary from Context Agent

        Returns:
            Dictionary containing task decomposition results
        """
        # Analyze current page content
        # Handle both StateInfo format and direct observation format
        if isinstance(current_observation, dict) and "observation" in current_observation:
            # StateInfo format: {"observation": obs, "info": info}
            obs_data = current_observation["observation"]
        else:
            # Direct observation format
            obs_data = current_observation

        current_page_text = obs_data.get("text", "") if isinstance(obs_data, dict) else str(obs_data)
        
        # Extract image for multimodal analysis
        current_image = None
        if isinstance(obs_data, dict):
            current_image = obs_data.get("image_raw")
            if current_image is None:
                current_image = obs_data.get("image")

        memory = context_summary.get("memory_content", "")

        # Try multimodal approach if supported and image available
        if self.is_multimodal and current_image is not None:
            try:
                decomposition = self._decompose_multimodal(
                    user_goal=user_goal,
                    current_image=current_image,
                    memory=memory
                )
                return decomposition
            except Exception as e:
                print(f"Multimodal task decomposition failed: {e}, falling back to text-only")

        # Fallback to text-only decomposition
        return self._decompose_text_only(
            user_goal=user_goal,
            current_page_text=current_page_text,
            memory=memory
        )

    def _decompose_multimodal(
        self,
        user_goal: str,
        current_image: np.ndarray,
        memory: str
    ) -> Dict[str, Any]:
        """Decompose task using multimodal (image + text) input."""
        # Convert numpy array to PIL Image
        if isinstance(current_image, np.ndarray):
            pil_image = Image.fromarray(current_image)
        else:
            pil_image = current_image

        # Load prompt template (multimodal version without current_page_text)
        if memory != "":
            prompt_text = load_prompt_template(
                "planner_agent",
                "task_decomposition_w_mem",
                memory=memory,
                user_goal=user_goal
            )
        else:
            prompt_text = load_prompt_template(
                "planner_agent",
                "task_decomposition",
                user_goal=user_goal
            )

        # Build multimodal message
        content = [
            {"type": "text", "text": "Current page screenshot:"},
            {
                "type": "image_url",
                "image_url": {"url": pil_to_b64(pil_image)}
            },
            {"type": "text", "text": prompt_text}
        ]

        messages = [{"role": "user", "content": content}]

        response = call_llm(self.lm_config, messages).strip()
        return self._parse_decomposition_response(response)

    def _decompose_text_only(
        self,
        user_goal: str,
        current_page_text: str,
        memory: str
    ) -> Dict[str, Any]:
        """Fallback text-only task decomposition."""
        # Build decomposition prompt using text template
        if memory != "":
            prompt = load_prompt_template(
                "planner_agent",
                "task_decomposition_w_mem_text",
                memory=memory,
                user_goal=user_goal,
                current_page_text=current_page_text
            )
        else:
            prompt = load_prompt_template(
                "planner_agent",
                "task_decomposition_text",
                user_goal=user_goal,
                current_page_text=current_page_text
            )

        try:
            response = call_llm(
                self.lm_config, [{"role": "user", "content": prompt}]
            ).strip()

            # Parse the LLM response into structured format
            decomposition = self._parse_decomposition_response(response)

        except Exception as e:
            # Fallback decomposition
            decomposition = self._generate_fallback_decomposition(user_goal, str(e))

        return decomposition

    def _parse_decomposition_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM task decomposition response into structured format."""
        # Simple parsing for subtasks
        decomposition = {
            "subtasks": [],
            "reasoning": response,
        }

        # Try to extract subtasks
        lines = response.split('\n')
        for line in lines:
            line = line.strip()

            # Skip empty lines and headers
            if not line or line.lower().startswith(('here are', 'subtasks:', 'steps:', 'breakdown:')):
                continue

            # Remove numbering and bullet points
            cleaned = line.lstrip('0123456789.-* ')
            cleaned = cleaned.lstrip('- ')
            cleaned = cleaned.lstrip('• ')

            # Clean up common prefixes
            for prefix in ['Subtask:', 'Step:', 'Task:']:
                if cleaned.startswith(prefix):
                    cleaned = cleaned[len(prefix):].strip()
                    break

            if cleaned and len(cleaned) > 10:  # Reasonable minimum length
                decomposition["subtasks"].append(cleaned)

        # If no subtasks were parsed, use the full response
        if not decomposition["subtasks"] and len(response.strip()) > 10:
            decomposition["subtasks"] = [response.strip()]

        # Limit to 3-5 subtasks
        decomposition["subtasks"] = decomposition["subtasks"][:5]

        return decomposition

    def _generate_fallback_decomposition(self, user_goal: str, error: str) -> Dict[str, Any]:
        """Generate fallback task decomposition when LLM fails."""
        # Generate generic subtasks based on common web task patterns
        subtasks = []

        # Common web task patterns
        if any(keyword in user_goal.lower() for keyword in ['search', 'find', 'look for']):
            subtasks.extend([
                "Navigate to search functionality",
                "Enter search query",
                "Review search results",
                "Select relevant option"
            ])

        elif any(keyword in user_goal.lower() for keyword in ['buy', 'purchase', 'order', 'cart']):
            subtasks.extend([
                "Locate product or service to purchase",
                "Add item to shopping cart",
                "Proceed to checkout process",
                "Complete purchase"
            ])

        elif any(keyword in user_goal.lower() for keyword in ['information', 'details', 'about']):
            subtasks.extend([
                "Look for information sections or links",
                "Navigate to relevant information pages",
                "Extract and review requested information"
            ])

        else:
            # Generic subtasks
            subtasks = [
                f"Get started with the task: {user_goal}",
                f"Make progress on: {user_goal}",
                f"Complete the task: {user_goal}"
            ]

        return {
            "subtasks": subtasks[:5],  # Limit to 5 subtasks
            "reasoning": f"Fallback decomposition due to error: {error}",
        }
