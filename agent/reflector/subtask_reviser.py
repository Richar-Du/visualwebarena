"""Subtask Reviser for generating revised subtasks when needed.

This module generates a revised subtask when the Reflector determines
that the current subtask needs revision (subtask_needs_revision=True).
"""

import re
from typing import Any, Dict, List, Optional
import numpy as np

from PIL import Image

from browser_env.utils import Observation, pil_to_b64
from llms import lm_config, call_llm
from ..prompts.prompt_loader import load_prompt_template
from ..utils import is_multimodal_model


class SubtaskReviser:
    """Generates revised subtasks when the current subtask cannot be completed as planned.
    
    This is called when:
    - subtask_needs_revision=True from the Reflector checklist
    - The current subtask is blocked or cannot make progress
    """

    def __init__(self, lm_config: lm_config.LMConfig) -> None:
        self.lm_config = lm_config
        self.is_multimodal = is_multimodal_model(lm_config.model)

    def generate_revised_subtask(
        self,
        high_level_task: str,
        current_subtask: str,
        all_subtasks: List[str],
        recent_intents: List[str],
        current_observation: Observation,
    ) -> Dict[str, Any]:
        """Generate a revised subtask based on current state.

        Args:
            high_level_task: Original high-level task goal
            current_subtask: Current subtask that needs revision
            all_subtasks: List of all subtasks
            recent_intents: Recent action intents
            current_observation: Current page observation

        Returns:
            Dictionary containing:
                - revised_subtask: The new revised subtask string
                - reasoning: Explanation for the revision
                - success: Whether revision generation succeeded
        """
        # Handle observation format
        if isinstance(current_observation, dict) and "observation" in current_observation:
            obs_data = current_observation["observation"]
        else:
            obs_data = current_observation

        current_page_text = obs_data.get("text", "") if isinstance(obs_data, dict) else str(obs_data)

        # Extract image for multimodal analysis
        current_image = None
        if isinstance(obs_data, dict):
            current_image = obs_data.get("image_raw")
            if current_image is None:
                current_image = obs_data.get("image")

        # Format subtasks and intents
        all_subtasks_str = "\n".join([f"{i+1}. {subtask}" for i, subtask in enumerate(all_subtasks)])
        recent_intents_str = "\n".join([f"{i+1}. {intent}" for i, intent in enumerate(recent_intents[-5:])])
        if not recent_intents_str:
            recent_intents_str = "No recent intents"

        # Try multimodal approach if supported and image available
        if self.is_multimodal and current_image is not None:
            try:
                return self._revise_multimodal(
                    high_level_task=high_level_task,
                    current_subtask=current_subtask,
                    all_subtasks_str=all_subtasks_str,
                    recent_intents_str=recent_intents_str,
                    current_image=current_image,
                )
            except Exception as e:
                print(f"Multimodal subtask revision failed: {e}, falling back to text-only")

        # Fallback to text-only revision
        return self._revise_text_only(
            high_level_task=high_level_task,
            current_subtask=current_subtask,
            all_subtasks_str=all_subtasks_str,
            recent_intents_str=recent_intents_str,
            current_page_text=current_page_text,
        )

    def _revise_multimodal(
        self,
        high_level_task: str,
        current_subtask: str,
        all_subtasks_str: str,
        recent_intents_str: str,
        current_image: np.ndarray,
    ) -> Dict[str, Any]:
        """Generate revised subtask using multimodal (image + text) input."""
        # Convert numpy array to PIL Image
        if isinstance(current_image, np.ndarray):
            pil_image = Image.fromarray(current_image)
        else:
            pil_image = current_image

        # Load prompt template
        prompt_text = load_prompt_template(
            "reflector_agent",
            "subtask_revision",
            high_level_task=high_level_task,
            current_subtask=current_subtask,
            all_subtasks=all_subtasks_str,
            recent_intents=recent_intents_str,
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
        return self._parse_revision_response(response, current_subtask)

    def _revise_text_only(
        self,
        high_level_task: str,
        current_subtask: str,
        all_subtasks_str: str,
        recent_intents_str: str,
        current_page_text: str,
    ) -> Dict[str, Any]:
        """Fallback text-only subtask revision."""
        # Load text-only prompt template
        prompt = load_prompt_template(
            "reflector_agent",
            "subtask_revision_text",
            high_level_task=high_level_task,
            current_subtask=current_subtask,
            all_subtasks=all_subtasks_str,
            recent_intents=recent_intents_str,
            current_page_text=current_page_text,
        )

        try:
            response = call_llm(
                self.lm_config, [{"role": "user", "content": prompt}]
            ).strip()
            return self._parse_revision_response(response, current_subtask)
        except Exception as e:
            print(f"Text-only subtask revision failed: {e}")
            return {
                "revised_subtask": current_subtask,  # Keep original if revision fails
                "reasoning": f"Revision failed: {e}",
                "success": False,
                "response": "",
            }

    def _parse_revision_response(self, response: str, current_subtask: str) -> Dict[str, Any]:
        """Parse LLM response to extract revised subtask.

        Args:
            response: Raw LLM response
            current_subtask: Original subtask (fallback)

        Returns:
            Dictionary with revised_subtask, reasoning, and success flag
        """
        result = {
            "revised_subtask": current_subtask,  # Default to original
            "reasoning": "",
            "success": False,
            "response": response,
        }

        try:
            # Extract content from XML tags
            think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL | re.IGNORECASE)
            if think_match:
                result["reasoning"] = think_match.group(1).strip()

            revised_match = re.search(r'<revised_subtask>(.*?)</revised_subtask>', response, re.DOTALL | re.IGNORECASE)
            if revised_match:
                revised_subtask = revised_match.group(1).strip()
                # Clean up the revised subtask (remove any numbering, extra whitespace)
                revised_subtask = re.sub(r'^\d+\.\s*', '', revised_subtask)  # Remove leading numbers
                revised_subtask = revised_subtask.strip()
                
                if revised_subtask and len(revised_subtask) > 5:  # Reasonable minimum length
                    result["revised_subtask"] = revised_subtask
                    result["success"] = True

        except Exception as e:
            print(f"Error parsing revision response: {e}")
            result["reasoning"] = f"Parse error: {e}"

        # If no revised subtask was extracted, try simple fallback
        if not result["success"]:
            # Try to find any line that looks like a subtask
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                # Skip empty lines, XML tags, and very short lines
                if not line or line.startswith('<') or len(line) < 10:
                    continue
                # Skip reasoning/explanation lines
                if any(word in line.lower() for word in ['because', 'since', 'however', 'therefore', 'the reason']):
                    continue
                # This might be the revised subtask
                result["revised_subtask"] = line
                result["success"] = True
                break

        return result

