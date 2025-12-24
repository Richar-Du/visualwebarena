"""Checklist-based execution analyzer for Reflector Agent with multimodal support.

This module performs unified checklist analysis including:
1. Pattern Check: Detect repetitive or erroneous patterns
2. Execution Check: Verify if action executed successfully
3. Task Completion Check: Check if overall task is completed
"""

import json
import re
from typing import Any, Dict, List, Optional
import numpy as np

from PIL import Image

from browser_env import Action
from browser_env.utils import pil_to_b64
from llms import lm_config, call_llm
from ..prompts.prompt_loader import load_prompt_template
from ..utils import is_multimodal_model


class ChecklistAnalyzer:
    """Performs unified checklist analysis on execution state.

    Evaluates key checks:
    1. Pattern Check: Detect repetitive or erroneous patterns in recent intents
    2. Execution Check: Verify if the latest action executed successfully
    3. Task Completion Check: Check if the overall task is completed
    """

    def __init__(self, lm_config: lm_config.LMConfig) -> None:
        self.lm_config = lm_config
        self.is_multimodal = is_multimodal_model(lm_config.model)

    def analyze(
        self,
        recent_intents: List[str],
        image_before: Optional[np.ndarray],
        image_after: Optional[np.ndarray],
        latest_action: Action,
        high_level_task: str,
    ) -> Dict[str, Any]:
        """Run unified checklist analysis.

        Args:
            recent_intents: Last 5 action intents
            image_before: Screenshot before action
            image_after: Screenshot after action
            latest_action: The executed action
            high_level_task: Original task goal

        Returns:
            Dictionary with checklist results
        """
        # Get action details
        action_text = self._decode_action_text(latest_action)
        action_type = latest_action.get("action_type", "UNKNOWN")
        element_id = latest_action.get("element_id", "N/A")

        # Try multimodal approach if supported and images available
        if self.is_multimodal and image_before is not None and image_after is not None:
            try:
                messages = self._build_multimodal_prompt(
                    recent_intents=recent_intents,
                    image_before=image_before,
                    image_after=image_after,
                    action_type=action_type,
                    element_id=element_id,
                    action_text=action_text,
                    high_level_task=high_level_task,
                )
                response = call_llm(self.lm_config, messages).strip()
                result = self._parse_response(response)
                result["raw_response"] = response
                return result
            except Exception as e:
                print(f"Multimodal checklist analysis failed: {e}, falling back to text-only")

        # Fallback to text-only analysis
        return self._analyze_text_only(
            recent_intents=recent_intents,
            action_type=action_type,
            element_id=element_id,
            action_text=action_text,
            high_level_task=high_level_task,
        )

    def _build_multimodal_prompt(
        self,
        recent_intents: List[str],
        image_before: np.ndarray,
        image_after: np.ndarray,
        action_type: str,
        element_id: str,
        action_text: str,
        high_level_task: str,
    ) -> List[Dict[str, Any]]:
        """Build multimodal prompt for checklist analysis."""
        # Convert numpy arrays to PIL Images
        if isinstance(image_before, np.ndarray):
            pil_image_before = Image.fromarray(image_before)
        else:
            pil_image_before = image_before

        if isinstance(image_after, np.ndarray):
            pil_image_after = Image.fromarray(image_after)
        else:
            pil_image_after = image_after

        # Format recent intents
        intents_str = "\n".join([f"{i+1}. {intent}" for i, intent in enumerate(recent_intents)])
        if not intents_str:
            intents_str = "No previous intents"

        # Load prompt template
        prompt_text = load_prompt_template(
            "reflector_agent",
            "reflection_checklist",
            recent_intents=intents_str,
            action_type=action_type,
            element_id=element_id,
            action_text=action_text,
            high_level_task=high_level_task,
        )

        # Build OpenAI Vision API format message with before/after images
        content = [
            {"type": "text", "text": "BEFORE ACTION - Screenshot before executing the action:"},
            {
                "type": "image_url",
                "image_url": {"url": pil_to_b64(pil_image_before)}
            },
            {"type": "text", "text": "AFTER ACTION - Screenshot after executing the action:"},
            {
                "type": "image_url",
                "image_url": {"url": pil_to_b64(pil_image_after)}
            },
            {"type": "text", "text": prompt_text}
        ]

        return [{"role": "user", "content": content}]

    def _analyze_text_only(
        self,
        recent_intents: List[str],
        action_type: str,
        element_id: str,
        action_text: str,
        high_level_task: str,
    ) -> Dict[str, Any]:
        """Fallback text-only checklist analysis."""
        # Format recent intents
        intents_str = "\n".join([f"{i+1}. {intent}" for i, intent in enumerate(recent_intents)])
        if not intents_str:
            intents_str = "No previous intents"

        # Load text-only prompt template
        prompt_text = load_prompt_template(
            "reflector_agent",
            "reflection_checklist_text",
            recent_intents=intents_str,
            action_type=action_type,
            element_id=element_id,
            action_text=action_text,
            high_level_task=high_level_task,
        )

        try:
            response = call_llm(
                self.lm_config, [{"role": "user", "content": prompt_text}]
            ).strip()
            result = self._parse_response(response)
            result["raw_response"] = response
            return result
        except Exception as e:
            print(f"Text-only checklist failed: {e}")
            result = self._get_default_result()
            result["raw_response"] = f"Error: {e}"
            return result

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response to extract 5 boolean checklist values.

        Args:
            response: Raw LLM response containing JSON

        Returns:
            Dictionary with 5 boolean values
        """
        result = self._get_default_result()

        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                parsed = json.loads(json_str)

                # Extract boolean values with type conversion
                if "has_pattern_issue" in parsed:
                    result["has_pattern_issue"] = self._to_bool(parsed["has_pattern_issue"])
                if "execution_successful" in parsed:
                    result["execution_successful"] = self._to_bool(parsed["execution_successful"])
                if "task_completed" in parsed:
                    result["task_completed"] = self._to_bool(parsed["task_completed"])

                return result

            # Fallback: try to parse keywords from response
            result = self._parse_keywords(response)

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Error parsing checklist response: {e}")

        return result

    def _parse_keywords(self, response: str) -> Dict[str, Any]:
        """Parse keywords from response as fallback."""
        result = self._get_default_result()
        response_lower = response.lower()

        # Pattern issue detection
        if "pattern issue: true" in response_lower or "has_pattern_issue: true" in response_lower:
            result["has_pattern_issue"] = True
        elif "repetitive" in response_lower and "detected" in response_lower:
            result["has_pattern_issue"] = True

        # Execution success detection
        if "execution_successful: false" in response_lower or "execution failed" in response_lower:
            result["execution_successful"] = False
        elif "successfully" in response_lower or "execution_successful: true" in response_lower:
            result["execution_successful"] = True

        # Task completion detection
        if "task_completed: true" in response_lower or "task is complete" in response_lower:
            result["task_completed"] = True

        return result

    def _get_default_result(self) -> Dict[str, Any]:
        """Get default checklist result."""
        return {
            "has_pattern_issue": False,
            "execution_successful": True,
            "task_completed": False,
        }

    def _to_bool(self, value: Any) -> bool:
        """Convert various value types to boolean."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "yes", "1")
        if isinstance(value, (int, float)):
            return bool(value)
        return False

    def _decode_action_text(self, action: Action) -> str:
        """Decode action text from ID list to readable string."""
        text_ids = action.get("text", [])
        if isinstance(text_ids, list) and text_ids:
            try:
                from browser_env.actions import _id2key
                return ''.join(_id2key[id_num] if 0 <= id_num < len(_id2key) else '?' for id_num in text_ids)
            except (ImportError, IndexError):
                return ''.join(chr(id_num) if 32 <= id_num <= 126 else '?' for id_num in text_ids)
        return "N/A"
