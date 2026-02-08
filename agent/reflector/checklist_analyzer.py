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
    2. Task Completion Check: Check if the overall task is completed
    """

    def __init__(self, lm_config: lm_config.LMConfig) -> None:
        self.lm_config = lm_config
        self.is_multimodal = is_multimodal_model(lm_config.model)

    def analyze(
        self,
        recent_intents: List[str],
        recent_actions: List[Action],
        image_before: Optional[np.ndarray],
        image_after: Optional[np.ndarray],
        latest_action: Action,
        high_level_task: str,
        context_summary: str = "",
    ) -> Dict[str, Any]:
        """Run unified checklist analysis.

        Args:
            recent_intents: Last 5 action intents
            recent_actions: Last 5 executed actions
            image_before: Screenshot before action
            image_after: Screenshot after action
            latest_action: The executed action
            high_level_task: Original task goal
            context_summary: Current context summary from Context Agent

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
                    recent_actions=recent_actions,
                    image_before=image_before,
                    image_after=image_after,
                    action_type=action_type,
                    element_id=element_id,
                    action_text=action_text,
                    high_level_task=high_level_task,
                    context_summary=context_summary,
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
            recent_actions=recent_actions,
            action_type=action_type,
            element_id=element_id,
            action_text=action_text,
            high_level_task=high_level_task,
            context_summary=context_summary,
        )

    def _build_multimodal_prompt(
        self,
        recent_intents: List[str],
        recent_actions: List[Action],
        image_before: np.ndarray,
        image_after: np.ndarray,
        action_type: str,
        element_id: str,
        action_text: str,
        high_level_task: str,
        context_summary: str = "",
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

        # Format recent actions
        actions_str = "\n".join([f"{i+1}. {self._format_action(action)}" for i, action in enumerate(recent_actions)])
        if not actions_str:
            actions_str = "No previous actions"

        # Load prompt template
        prompt_text = load_prompt_template(
            "reflector_agent",
            "reflection_checklist",
            recent_intents=intents_str,
            recent_actions=actions_str,
            action_type=action_type,
            element_id=element_id,
            action_text=action_text,
            high_level_task=high_level_task,
            context_summary=context_summary if context_summary else "No context summary available",
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
        recent_actions: List[Action],
        action_type: str,
        element_id: str,
        action_text: str,
        high_level_task: str,
        context_summary: str = "",
    ) -> Dict[str, Any]:
        """Fallback text-only checklist analysis."""
        # Format recent intents
        intents_str = "\n".join([f"{i+1}. {intent}" for i, intent in enumerate(recent_intents)])
        if not intents_str:
            intents_str = "No previous intents"

        # Format recent actions
        actions_str = "\n".join([f"{i+1}. {self._format_action(action)}" for i, action in enumerate(recent_actions)])
        if not actions_str:
            actions_str = "No previous actions"

        # Load text-only prompt template
        prompt_text = load_prompt_template(
            "reflector_agent",
            "reflection_checklist_text",
            recent_intents=intents_str,
            recent_actions=actions_str,
            action_type=action_type,
            element_id=element_id,
            action_text=action_text,
            high_level_task=high_level_task,
            context_summary=context_summary if context_summary else "No context summary available",
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
        """Parse LLM response to extract checklist values including task_completion_reason.

        Args:
            response: Raw LLM response containing JSON

        Returns:
            Dictionary with checklist values including task_completion_reason
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
                if "has_goal_deviation" in parsed:
                    result["has_goal_deviation"] = self._to_bool(parsed["has_goal_deviation"])
                if "task_completed" in parsed:
                    result["task_completed"] = self._to_bool(parsed["task_completed"])
                # Extract pattern_issue_reason (new field)
                if "pattern_issue_reason" in parsed:
                    result["pattern_issue_reason"] = str(parsed["pattern_issue_reason"])
                # Extract goal_deviation_reason
                if "goal_deviation_reason" in parsed:
                    result["goal_deviation_reason"] = str(parsed["goal_deviation_reason"])
                # Extract task_completion_reason
                if "task_completion_reason" in parsed:
                    result["task_completion_reason"] = str(parsed["task_completion_reason"])

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
            result["pattern_issue_reason"] = "Pattern issue detected based on keyword analysis."
        elif "repetitive" in response_lower and "detected" in response_lower:
            result["has_pattern_issue"] = True
            result["pattern_issue_reason"] = "Repetitive action pattern detected."
        elif "same element" in response_lower and "click" in response_lower:
            result["has_pattern_issue"] = True
            result["pattern_issue_reason"] = "repeated_clicks: Same element clicked multiple times."

        # Goal deviation detection
        if "goal deviation: true" in response_lower or "has_goal_deviation: true" in response_lower:
            result["has_goal_deviation"] = True
            result["goal_deviation_reason"] = "Goal deviation detected based on keyword analysis."
        elif "drifting" in response_lower or "off track" in response_lower:
            result["has_goal_deviation"] = True
            result["goal_deviation_reason"] = "Agent appears to be drifting off track."
        elif "wrong color" in response_lower or "wrong price" in response_lower:
            result["has_goal_deviation"] = True
            result["goal_deviation_reason"] = "Product attribute mismatch detected."

        # Task completion detection
        if "task_completed: true" in response_lower or "task is complete" in response_lower:
            result["task_completed"] = True
            result["task_completion_reason"] = "Task appears complete based on keyword detection."
        else:
            result["task_completion_reason"] = "Task not yet complete based on keyword detection."

        return result

    def _get_default_result(self) -> Dict[str, Any]:
        """Get default checklist result."""
        return {
            "has_pattern_issue": False,
            "pattern_issue_reason": "",  # Field for pattern issue explanation
            "has_goal_deviation": False,
            "goal_deviation_reason": "",  # Field for goal deviation explanation
            "task_completed": False,
            "task_completion_reason": "",  # Field for task completion explanation
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

    def _format_action(self, action: Action) -> str:
        """Format an action object into a readable string."""
        if not action or not isinstance(action, dict):
            return "Invalid action"

        action_type = action.get("action_type", "UNKNOWN")
        element_id = action.get("element_id", "N/A")
        action_text = self._decode_action_text(action)

        if element_id != "N/A" and action_text != "N/A":
            return f"{action_text}[{element_id}]"
        else:
            return f"Type: {action_type}"
