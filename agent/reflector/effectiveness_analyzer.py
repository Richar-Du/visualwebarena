"""Simplified Effectiveness analysis for Reflector Agent with multimodal support."""

from typing import Any, Dict, List
import numpy as np

from PIL import Image

from browser_env import Action, Trajectory
from browser_env.utils import pil_to_b64
from llms import lm_config, call_llm
from ..prompts.prompt_loader import load_prompt_template
from ..utils import is_multimodal_model


class EffectivenessAnalyzer:
    """Analyzes the effectiveness of actions in progressing toward the goal.
    
    Supports multimodal analysis using page screenshots for better visual understanding.
    """

    def __init__(self, lm_config: lm_config.LMConfig) -> None:
        self.lm_config = lm_config
        # Check if the model supports multimodal inputs
        self.is_multimodal = is_multimodal_model(lm_config.model)

    def analyze(
        self,
        trajectory: Trajectory,
        current_intention: str,
        latest_action: Action,
        context_summary: Dict[str, Any],
    ) -> str:
        """Analyze the effectiveness of the latest action using visual comparison.

        Args:
            trajectory: Current execution trajectory
            current_intention: The intention that was being fulfilled
            latest_action: The most recently executed action
            context_summary: Current context from Context Agent

        Returns:
            Natural language response about action effectiveness
        """
        # Extract images from trajectory for visual comparison
        image_before, image_after = self._extract_before_after_images(trajectory)

        # Convert text IDs back to readable string
        text_ids = latest_action.get("text", [])
        if isinstance(text_ids, list) and text_ids:
            try:
                # Import the ID to key mapping from browser_env
                from browser_env.actions import _id2key
                action_text = ''.join(_id2key[id_num] if 0 <= id_num < len(_id2key) else '?' for id_num in text_ids)
            except (ImportError, IndexError):
                # Fallback: try to convert IDs to characters directly
                action_text = ''.join(chr(id_num) if 32 <= id_num <= 126 else '?' for id_num in text_ids)
        else:
            action_text = "N/A"

        # Get context summary text
        summary_text = context_summary.get("summary", "No context summary available")

        # Format action details
        action_details = f"Type: {latest_action.get('action_type', 'UNKNOWN')}, Element: {latest_action.get('element_id', 'N/A')}, Details: {action_text}"

        # Try multimodal approach if supported and images available
        if self.is_multimodal and image_before is not None and image_after is not None:
            try:
                messages = self._build_multimodal_effectiveness_prompt(
                    current_intention=current_intention,
                    action_details=action_details,
                    context_summary=summary_text[:500] if len(summary_text) > 500 else summary_text,
                    image_before=image_before,
                    image_after=image_after
                )
                response = call_llm(self.lm_config, messages).strip()
                return response
            except Exception as e:
                print(f"Multimodal effectiveness analysis failed: {e}, falling back to text-only")

        # Fallback to text-only analysis
        return self._analyze_text_only(
            trajectory, current_intention, latest_action, context_summary, action_text
        )

    def _extract_before_after_images(self, trajectory: Trajectory) -> tuple:
        """Extract before and after images from trajectory.
        
        Returns:
            tuple: (image_before, image_after) - numpy arrays or None
        """
        image_before = None
        image_after = None

        # Trajectory structure: [StateInfo, Action, StateInfo, Action, ...]
        # We need the second-to-last StateInfo (before) and the last StateInfo (after)
        
        state_infos = [item for item in trajectory if isinstance(item, dict) and 'observation' in item]
        
        if len(state_infos) >= 2:
            # Get before image (second-to-last state)
            obs_before = state_infos[-2].get("observation", {})
            image_before = obs_before.get("image_raw")
            if image_before is None:
                image_before = obs_before.get("image")
            
            # Get after image (last state)
            obs_after = state_infos[-1].get("observation", {})
            image_after = obs_after.get("image_raw")
            if image_after is None:
                image_after = obs_after.get("image")
        elif len(state_infos) == 1:
            # Only one state, use it for both (first step)
            obs = state_infos[-1].get("observation", {})
            image_after = obs.get("image_raw")
            if image_after is None:
                image_after = obs.get("image")

        return image_before, image_after

    def _build_multimodal_effectiveness_prompt(
        self,
        current_intention: str,
        action_details: str,
        context_summary: str,
        image_before: np.ndarray,
        image_after: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Build multimodal prompt for visual effectiveness analysis.
        
        Args:
            current_intention: The intention being fulfilled
            action_details: Details of the executed action
            context_summary: Current context summary
            image_before: Screenshot before action execution
            image_after: Screenshot after action execution
            
        Returns:
            List of message dictionaries for the multimodal LLM API
        """
        # Convert numpy arrays to PIL Images
        if isinstance(image_before, np.ndarray):
            pil_image_before = Image.fromarray(image_before)
        else:
            pil_image_before = image_before
            
        if isinstance(image_after, np.ndarray):
            pil_image_after = Image.fromarray(image_after)
        else:
            pil_image_after = image_after

        # Load prompt template
        prompt_text = load_prompt_template(
            "reflector_agent",
            "effectiveness_analysis",
            current_intention=current_intention,
            latest_action=action_details,
            context_summary=context_summary
        )

        # Build OpenAI Vision API format message with before/after images
        content = [
            {"type": "text", "text": "BEFORE ACTION - Page screenshot before executing the action:"},
            {
                "type": "image_url",
                "image_url": {"url": pil_to_b64(pil_image_before)}
            },
            {"type": "text", "text": "AFTER ACTION - Page screenshot after executing the action:"},
            {
                "type": "image_url",
                "image_url": {"url": pil_to_b64(pil_image_after)}
            },
            {"type": "text", "text": prompt_text}
        ]

        return [{"role": "user", "content": content}]

    def _analyze_text_only(
        self,
        trajectory: Trajectory,
        current_intention: str,
        latest_action: Action,
        context_summary: Dict[str, Any],
        action_text: str
    ) -> str:
        """Fallback text-only analysis when multimodal is not available."""
        # Get recent trajectory context
        recent_context = self._extract_recent_context(trajectory, latest_action)

        # Get context summary text
        summary_text = context_summary.get("summary", "No context summary available")

        # Build analysis prompt using text fallback template
        prompt = load_prompt_template(
            "reflector_agent",
            "effectiveness_analysis_text",
            current_intention=current_intention,
            latest_action=f"Type: {latest_action.get('action_type', 'UNKNOWN')}, Element: {latest_action.get('element_id', 'N/A')}, Details: {action_text}",
            context_summary=summary_text[:500] if len(summary_text) > 500 else summary_text,
            trajectory_summary=recent_context
        )

        try:
            response = call_llm(
                self.lm_config, [{"role": "user", "content": prompt}]
            ).strip()
            return response
        except Exception as e:
            # Fallback effectiveness analysis
            action_type = latest_action.get("action_type", "UNKNOWN")
            return f"Fallback analysis: Action '{action_type}' executed for intention: {current_intention[:100]}."

    def _extract_recent_context(self, trajectory: Trajectory, latest_action: Action) -> str:
        """Extract relevant context from the recent trajectory (text-only fallback)."""
        context_parts = []

        # Get last 3 action-observation pairs
        recent_steps = trajectory[-6:]  # Last 3 pairs (obs-action-obs-action-obs-action-obs)

        page_count = 1
        action_count = 1

        for step in recent_steps:
            if isinstance(step, dict):
                # Check if it's an observation (StateInfo) by looking for 'observation' key
                if 'observation' in step:
                    obs_text = step.get("observation", {}).get("text", "")[:200]
                    context_parts.append(f"Page {page_count}: {obs_text}...")
                    page_count += 1
                # Check if it's an action by looking for 'action_type' key
                elif 'action_type' in step:
                    action_type = step.get("action_type", "UNKNOWN")
                    element_id = step.get("element_id", "N/A")
                    context_parts.append(f"Action {action_count}: {action_type} on {element_id}")
                    action_count += 1

        return " | ".join(context_parts)
