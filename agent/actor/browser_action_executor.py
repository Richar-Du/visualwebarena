"""Browser action execution for Actor Agent with multimodal support.

This module generates specific browser actions based on high-level intentions.
"""

from typing import Any, Dict, Optional
import re
import numpy as np

from PIL import Image

from browser_env import Trajectory
from browser_env.utils import Observation, StateInfo, pil_to_b64
from browser_env.actions import create_id_based_action, ActionParsingError, create_none_action
from llms import lm_config, call_llm
from ..prompts.prompt_loader import load_prompt_template, get_prompt_loader
from ..utils import is_multimodal_model


class BrowserActionExecutor:
    """Generates specific browser actions using the new prompt system."""

    def __init__(self, lm_config: lm_config.LMConfig, action_set_tag: str = "id_accessibility_tree") -> None:
        self.lm_config = lm_config
        self.action_set_tag = action_set_tag
        # Check if the model supports multimodal inputs
        self.is_multimodal = is_multimodal_model(lm_config.model)

    def execute_action(
        self,
        intention: str,
        trajectory: Trajectory,
        meta_data: Dict[str, Any],
        images: Optional[list[Image.Image]] = None,
    ) -> Dict[str, Any]:
        """Generate browser action for the given intention.

        Args:
            intention: High-level intention from Planner Agent
            trajectory: Current execution trajectory
            meta_data: Additional metadata including action history
            images: Optional input images for the task (for multimodal support)

        Returns:
            Dictionary containing generated action and metadata
        """
        # Get current state from trajectory
        state_info: StateInfo = trajectory[-1]  # type: ignore[assignment]
        
        # Extract observation
        obs = state_info["observation"].get("accessibility_tree", state_info["observation"].get("text", ""))
        
        # Truncate observation if needed
        max_obs_length = self.lm_config.gen_config.get("max_obs_length")
        if max_obs_length:
            if self.lm_config.provider == "google":
                obs = obs[:max_obs_length]
            else:
                # Note: We don't have tokenizer here, so just truncate by chars for safety
                obs = obs[:max_obs_length]
        
        # Get page info
        page = state_info["info"]["page"]
        url = page.url
        
        # Get context summary from context_agent instead of previous action
        context_summary = meta_data.get("context_summary", "No context available")
        
        # Extract image for multimodal analysis
        current_image = None
        obs_data = state_info["observation"]
        if isinstance(obs_data, dict):
            current_image = obs_data.get("image_raw")
            if current_image is None:
                current_image = obs_data.get("image")

        # Try multimodal approach if supported and (current image or input images) available
        has_input_images = images is not None and len(images) > 0
        if self.is_multimodal and (current_image is not None or has_input_images):
            try:
                response = self._execute_multimodal(
                    intention=intention,
                    observation=obs,
                    url=url,
                    context=context_summary,
                    current_image=current_image,
                    input_images=images,
                )
            except Exception as e:
                print(f"Multimodal action execution failed: {e}, falling back to text-only")
                response = self._execute_text_only(
                    intention=intention,
                    observation=obs,
                    url=url,
                    context=context_summary,
                )
        else:
            # Fallback to text-only
            response = self._execute_text_only(
                intention=intention,
                observation=obs,
                url=url,
                context=context_summary,
            )

        # Parse the action from the response
        try:
            parsed_action = self._extract_action(response)
            action = create_id_based_action(parsed_action)
            action["raw_prediction"] = response
        except ActionParsingError as e:
            print(f"Action parsing error: {e}")
            action = create_none_action()
            action["raw_prediction"] = response

        # Extract the actual intention from the LLM's thinking process
        extracted_intention = self._extract_intention(response)

        return {
            "action": action,
            "llm_response": response,
            "intention": intention,  # Original high-level intention (user_goal)
            "extracted_intention": extracted_intention,  # LLM's reasoning from <think> tags
        }

    def _execute_multimodal(
        self,
        intention: str,
        observation: str,
        url: str,
        context: str,
        current_image: np.ndarray,
        input_images: Optional[list[Image.Image]] = None,
    ) -> str:
        """Execute action generation using multimodal (image + text) input."""
        # Convert numpy array to PIL Image
        if isinstance(current_image, np.ndarray):
            pil_image = Image.fromarray(current_image)
        else:
            pil_image = current_image

        # Load prompt template from the new system
        prompt_text = load_prompt_template(
            "actor_agent",
            "action_execution",
            observation=observation,
            url=url,
            objective=intention,
            context=context,
        )

        # Build multimodal message
        content = [
            {"type": "text", "text": "Current page screenshot:"},
            {
                "type": "image_url",
                "image_url": {"url": pil_to_b64(pil_image)}
            },
        ]

        # Add input images if provided
        if input_images is not None and len(input_images) > 0:
            for image_i, input_image in enumerate(input_images):
                content.extend([
                    {"type": "text", "text": f"Input image {image_i+1}:"},
                    {
                        "type": "image_url",
                        "image_url": {"url": pil_to_b64(input_image)}
                    },
                ])

        # Add the text prompt
        content.append({"type": "text", "text": prompt_text})

        messages = [{"role": "user", "content": content}]

        response = call_llm(self.lm_config, messages).strip()
        return response

    def _execute_text_only(
        self,
        intention: str,
        observation: str,
        url: str,
        context: str,
    ) -> str:
        """Fallback text-only action generation."""
        # Load text-only prompt template
        prompt = load_prompt_template(
            "actor_agent",
            "action_execution_text",
            observation=observation,
            url=url,
            objective=intention,
            context=context,
        )

        response = call_llm(
            self.lm_config, [{"role": "user", "content": prompt}]
        ).strip()

        return response

    def _extract_action(self, response: str) -> str:
        """Extract action from LLM response.

        The response should contain the action in the format:
        <action>
        In summary, the next action I will perform is ```action```
        </action>
        """
        # First try to extract from <action> tags
        if "<action>" in response and "</action>" in response:
            # Extract content between <action> and </action> tags
            action_start = response.find("<action>")
            action_end = response.find("</action>")
            if action_start != -1 and action_end != -1:
                action_content = response[action_start + 8:action_end].strip()

                # Within the action content, look for the answer phrase and action splitter
                answer_phrase = "In summary, the next action I will perform is"
                action_splitter = "```"

                if answer_phrase in action_content:
                    # Split by answer phrase and get the part after it
                    after_phrase = action_content.split(answer_phrase, 1)[1]
                    # Find action between splitters
                    action_splits = after_phrase.split(action_splitter)
                    if len(action_splits) >= 2:
                        return action_splits[1].strip()

        # Fallback: try the original method (for backward compatibility)
        answer_phrase = "In summary, the next action I will perform is"
        action_splitter = "```"

        if answer_phrase in response:
            # Split by answer phrase and get the part after it
            after_phrase = response.split(answer_phrase, 1)[1]
            # Find action between splitters
            action_splits = after_phrase.split(action_splitter)
            if len(action_splits) >= 2:
                return action_splits[1].strip()

        # Fallback: try to find any text between backticks
        action_pattern = r'```(.+?)```'
        matches = re.findall(action_pattern, response, re.DOTALL)
        if matches:
            return matches[0].strip()

        # If no pattern found, raise error
        raise ActionParsingError(
            f"Cannot parse action from response {response}"
        )

    def _extract_intention(self, response: str) -> str:
        """Extract intention from LLM response.

        Extract the <think> portion which contains the reasoning and intention.

        Args:
            response: LLM response containing <think> and <action> tags

        Returns:
            Extracted intention text from <think> tags, or fallback text
        """
        # Extract content between <think> and </think> tags
        if "<think>" in response and "</think>" in response:
            think_start = response.find("<think>")
            think_end = response.find("</think>")
            if think_start != -1 and think_end != -1:
                think_content = response[think_start + 7:think_end].strip()
                # Truncate if too long (keep most relevant reasoning)
                if len(think_content) > 500:
                    think_content = think_content[:500] + "..."
                return think_content

        # Fallback: if no <think> tags, return a generic intention
        return f"Execute next step toward goal based on current page state"

