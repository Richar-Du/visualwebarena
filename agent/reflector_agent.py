"""Reflector Agent for execution validation, analysis, and recovery suggestions."""

from typing import Any, Dict, List, Optional
import numpy as np

from PIL import Image

from browser_env import Action, Trajectory
from browser_env.utils import Observation, pil_to_b64
from llms import lm_config, call_llm

from .reflector.effectiveness_analyzer import EffectivenessAnalyzer
from .reflector.pattern_detector import PatternDetector
from .prompts.prompt_loader import load_prompt_template
from .utils import is_multimodal_model


class ReflectorAgent:
    """Simplified reflector agent for execution analysis with multimodal support.

    Responsible for analyzing action effectiveness and detecting execution patterns
    to provide insights for better decision making.
    """

    def __init__(self, lm_config: lm_config.LMConfig) -> None:
        self.lm_config = lm_config
        self.effectiveness_analyzer = EffectivenessAnalyzer(lm_config)
        self.pattern_detector = PatternDetector()
        # Check if the model supports multimodal inputs
        self.is_multimodal = is_multimodal_model(lm_config.model)

        # Reflection history
        self.reflection_history: List[Dict[str, Any]] = []

    def reflect_execution(
        self,
        trajectory: Trajectory,
        intentions: List[str],
        actions: List[Action],
        current_intention: str,
        latest_action: Action,
        current_observation: Observation,
        context_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Reflect on the execution of the latest action.

        Args:
            trajectory: Current execution trajectory
            intentions: List of all intentions so far
            actions: List of all actions executed so far
            current_intention: The intention that was being fulfilled
            latest_action: The most recently executed action
            current_observation: The observation after action execution
            context_summary: Current context from Context Agent

        Returns:
            Dictionary containing simplified reflection results
        """
        try:
            # 1. Analyze action effectiveness (natural language only)
            effectiveness_result = self.effectiveness_analyzer.analyze(
                trajectory=trajectory,
                current_intention=current_intention,
                latest_action=latest_action,
                context_summary=context_summary,
            )

            # 2. Detect execution patterns
            pattern_result = self.pattern_detector.detect_patterns(actions, intentions)

            # 3. Generate triple summary with visual comparison
            triple_summary = self._generate_enhanced_triple_summary(
                trajectory, current_intention, latest_action, current_observation
            )

            # Create simplified reflection
            reflection = {
                "effectiveness_analyzer": effectiveness_result,
                "pattern_detector": pattern_result,
                "triple_summary": triple_summary,
                "current_intention": current_intention,
                "latest_action": latest_action,
                "reflection_number": len(self.reflection_history) + 1,
            }

            # Store in reflection history
            self.reflection_history.append(reflection)

            return reflection

        except Exception as e:
            # Create error reflection
            error_reflection = {
                "effectiveness_analyzer": f"Error during effectiveness analysis: {str(e)}",
                "pattern_detector": f"Error during pattern detection: {str(e)}",
                "triple_summary": f"Error during triple summary generation: {str(e)}",
                "current_intention": current_intention,
                "latest_action": latest_action,
                "reflection_number": len(self.reflection_history) + 1,
            }

            self.reflection_history.append(error_reflection)
            return error_reflection

    def _generate_enhanced_triple_summary(
        self,
        trajectory: Trajectory,
        current_intention: str,
        latest_action: Action,
        current_observation: Observation,
    ) -> str:
        """Generate an enhanced (O_{t-1}, I_t, A_t, O_t) triple summary using VLM with visual comparison."""

        # Extract before and after images
        image_before, image_after = self._extract_before_after_images(trajectory, current_observation)

        # Get action details
        action_type = latest_action.get("action_type", "UNKNOWN")
        element_id = latest_action.get("element_id", "N/A")

        # Correctly convert text IDs back to string
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

        # Try multimodal approach if supported and images available
        if self.is_multimodal and image_before is not None and image_after is not None:
            try:
                messages = self._build_multimodal_triple_summary_prompt(
                    current_intention=current_intention,
                    action_type=action_type,
                    element_id=element_id,
                    action_text=action_text,
                    image_before=image_before,
                    image_after=image_after
                )
                response = call_llm(self.lm_config, messages).strip()
                return response
            except Exception as e:
                print(f"Multimodal triple summary failed: {e}, falling back to text-only")

        # Fallback to text-only analysis
        return self._generate_text_only_triple_summary(
            trajectory, current_intention, latest_action, current_observation, action_type, element_id, action_text
        )

    def _extract_before_after_images(self, trajectory: Trajectory, current_observation: Observation) -> tuple:
        """Extract before and after images from trajectory.
        
        Returns:
            tuple: (image_before, image_after) - numpy arrays or None
        """
        image_before = None
        image_after = None

        # Get current observation image (after)
        image_after = current_observation.get("image_raw")
        if image_after is None:
            image_after = current_observation.get("image")

        # Get previous observation image (before) from trajectory
        # Trajectory structure: [StateInfo, Action, StateInfo, Action, ...]
        state_infos = [item for item in trajectory if isinstance(item, dict) and 'observation' in item]
        
        if len(state_infos) >= 2:
            # Get second-to-last state (before the action)
            obs_before = state_infos[-2].get("observation", {})
            image_before = obs_before.get("image_raw")
            if image_before is None:
                image_before = obs_before.get("image")
        elif len(state_infos) == 1:
            # First step - use the same image for before
            obs_before = state_infos[0].get("observation", {})
            image_before = obs_before.get("image_raw")
            if image_before is None:
                image_before = obs_before.get("image")

        return image_before, image_after

    def _build_multimodal_triple_summary_prompt(
        self,
        current_intention: str,
        action_type: str,
        element_id: str,
        action_text: str,
        image_before: np.ndarray,
        image_after: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Build multimodal prompt for visual triple summary.
        
        Args:
            current_intention: The intention being fulfilled
            action_type: Type of action executed
            element_id: Element ID that was interacted with
            action_text: Text content of the action (if applicable)
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
            "triple_summary",
            current_intention=current_intention,
            action_type=action_type,
            element_id=element_id,
            action_text=action_text
        )

        # Build OpenAI Vision API format message with before/after images
        content = [
            {"type": "text", "text": "PREVIOUS STATE (O_[t-1]) - Page screenshot BEFORE the action:"},
            {
                "type": "image_url",
                "image_url": {"url": pil_to_b64(pil_image_before)}
            },
            {"type": "text", "text": "CURRENT STATE (O_t) - Page screenshot AFTER the action:"},
            {
                "type": "image_url",
                "image_url": {"url": pil_to_b64(pil_image_after)}
            },
            {"type": "text", "text": prompt_text}
        ]

        return [{"role": "user", "content": content}]

    def _generate_text_only_triple_summary(
        self,
        trajectory: Trajectory,
        current_intention: str,
        latest_action: Action,
        current_observation: Observation,
        action_type: str,
        element_id: str,
        action_text: str
    ) -> str:
        """Fallback text-only triple summary generation."""
        # Get previous observation from trajectory
        obs_before = "No previous observation available"
        if len(trajectory) >= 3:
            prev_state = trajectory[-3]
            if isinstance(prev_state, dict) and 'observation' in prev_state:
                obs_text = prev_state.get("observation", {}).get("text", "")[:200]
                if len(obs_text) > 200:
                    obs_before = obs_text + "..."
                else:
                    obs_before = obs_text

        obs_after_text = current_observation.get("text", "")[:200]
        if len(current_observation.get("text", "")) > 200:
            obs_after_text += "..."

        # Build enhanced triple summary prompt using text fallback template
        prompt = load_prompt_template(
            "reflector_agent",
            "triple_summary_text",
            **{"t-1": obs_before},
            current_intention=current_intention,
            action_type=action_type,
            element_id=element_id,
            action_text=action_text,
            current_observation=obs_after_text
        )

        try:
            response = call_llm(
                self.lm_config, [{"role": "user", "content": prompt}]
            ).strip()
            return response
        except Exception as e:
            # Fallback simple triple summary
            success_indicator = "Success" if action_type not in ["NONE", "STOP"] else "Failed"
            return f"Intent: {current_intention[:50]}{'...' if len(current_intention) > 50 else ''} | Action: {action_type} on {element_id} | Result: {success_indicator} (Enhanced summary unavailable: {str(e)})"

    def reset_reflection_history(self) -> None:
        """Reset reflection history for a new task."""
        self.reflection_history.clear()
