"""Current state analysis for planning Agent with multimodal support.

This module generates the next atomic action based on a GIVEN current subtask.
The current subtask is explicitly provided by SubtaskManager (not inferred by LLM).
"""

from typing import Any, Dict, Optional
import numpy as np

from PIL import Image

from browser_env.utils import Observation, pil_to_b64
from llms import lm_config, call_llm
from ..prompts.prompt_loader import load_prompt_template
from ..utils import is_multimodal_model


class CurrentStateAnalyzer:
    """Generates next atomic action for a given subtask with multimodal support.
    
    Key change from previous version:
    - NO LONGER infers which subtask is current (that's done by SubtaskManager)
    - Receives current_subtask as INPUT parameter
    - Only outputs next_atomic_action
    """

    def __init__(self, lm_config: lm_config.LMConfig) -> None:
        self.lm_config = lm_config
        # Check if the model supports multimodal inputs
        self.is_multimodal = is_multimodal_model(lm_config.model)

    def analyze_current_state(
        self,
        user_goal: str,
        current_subtask: str,  # Now explicit input, not inferred
        current_observation: Observation,
        context_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate next atomic action for the given subtask.

        Args:
            user_goal: Original user goal
            current_subtask: Current subtask being worked on (from SubtaskManager)
            current_observation: Current page observation (includes image_raw for visual analysis)
            context_summary: Current context from Context Agent

        Returns:
            Dictionary containing next action analysis results
        """
        # Handle both StateInfo format and direct observation format
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

        # Get context information
        cur_summary = context_summary.get("summary", "")
        memory = context_summary.get("memory_content", "")

        # Try multimodal approach if supported and image available
        if self.is_multimodal and current_image is not None:
            try:
                analysis = self._analyze_multimodal(
                    user_goal=user_goal,
                    current_subtask=current_subtask,
                    current_image=current_image,
                    context_summary=cur_summary,
                    memory=memory,
                )
                return analysis
            except Exception as e:
                print(f"Multimodal state analysis failed: {e}, falling back to text-only")

        # Fallback to text-only analysis
        return self._analyze_text_only(
            user_goal=user_goal,
            current_subtask=current_subtask,
            current_page_text=current_page_text,
            context_summary=cur_summary,
            memory=memory,
        )

    def _analyze_multimodal(
        self,
        user_goal: str,
        current_subtask: str,
        current_image: np.ndarray,
        context_summary: str,
        memory: str,
    ) -> Dict[str, Any]:
        """Analyze current state using multimodal (image + text) input."""
        # Convert numpy array to PIL Image
        if isinstance(current_image, np.ndarray):
            pil_image = Image.fromarray(current_image)
        else:
            pil_image = current_image

        # Load prompt template - now only generates next_action
        if memory != "":
            prompt_text = load_prompt_template(
                "planner_agent",
                "next_action_generation_w_mem",
                memory=memory,
                user_goal=user_goal,
                current_subtask=current_subtask,
                context_summary=context_summary
            )
        else:
            prompt_text = load_prompt_template(
                "planner_agent",
                "next_action_generation",
                user_goal=user_goal,
                current_subtask=current_subtask,
                context_summary=context_summary
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
        return self._parse_analysis_response(response, current_subtask)

    def _analyze_text_only(
        self,
        user_goal: str,
        current_subtask: str,
        current_page_text: str,
        context_summary: str,
        memory: str,
    ) -> Dict[str, Any]:
        """Fallback text-only state analysis."""
        # Build state analysis prompt using text template
        if memory != "":
            prompt = load_prompt_template(
                "planner_agent",
                "next_action_generation_w_mem_text",
                memory=memory,
                user_goal=user_goal,
                current_subtask=current_subtask,
                current_page_text=current_page_text,
                context_summary=context_summary,
            )
        else:
            prompt = load_prompt_template(
                "planner_agent",
                "next_action_generation_text",
                user_goal=user_goal,
                current_subtask=current_subtask,
                current_page_text=current_page_text,
                context_summary=context_summary,
            )

        try:
            response = call_llm(
                self.lm_config, [{"role": "user", "content": prompt}]
            ).strip()
            # Parse the LLM response into structured format
            analysis = self._parse_analysis_response(response, current_subtask)

        except Exception as e:
            # Fallback analysis
            print(f"🎯 Current State Analyzer: Error in analysis: {str(e)}")
            analysis = self._generate_fallback_analysis(user_goal, current_subtask, str(e))

        return analysis

    def _parse_analysis_response(self, response: str, current_subtask: str) -> Dict[str, Any]:
        """Parse LLM response into structured format using XML tags.
        
        Now only extracts next_action (current_subtask is provided as input).
        """
        import re

        # Default analysis structure
        analysis = {
            "current_subtask": current_subtask,  # Provided as input, not inferred
            "next_atomic_action": "",
            "reasoning": "",
            "response": response
        }

        try:
            # Extract content from XML tags using regex
            think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL | re.IGNORECASE)
            if think_match:
                analysis["reasoning"] = think_match.group(1).strip()

            next_action_match = re.search(r'<next_action>(.*?)</next_action>', response, re.DOTALL | re.IGNORECASE)
            if next_action_match:
                analysis["next_atomic_action"] = next_action_match.group(1).strip()

        except Exception as e:
            # If parsing fails, fall back to using the entire response
            analysis["reasoning"] = response
            analysis["next_atomic_action"] = response

        # If no next action extracted, use a simple fallback
        if not analysis["next_atomic_action"]:
            print(f"🎯 Current State Analyzer: No next atomic action extracted, using current subtask: {current_subtask}")
            analysis["next_atomic_action"] = f"Continue working on: {current_subtask}"

        return analysis

    def _generate_fallback_analysis(self, user_goal: str, current_subtask: str, error: str) -> Dict[str, Any]:
        """Generate fallback state analysis when LLM fails."""
        # Generate simple fallback action
        next_action = f"Continue working on: {current_subtask}"

        return {
            "current_subtask": current_subtask,
            "next_atomic_action": next_action,
            "reasoning": f"Fallback analysis due to error: {error}"
        }
