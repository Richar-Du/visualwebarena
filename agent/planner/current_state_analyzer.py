"""Current state analysis for planning Agent."""

from typing import Any, Dict, List

from browser_env.utils import Observation
from llms import lm_config, call_llm
from ..prompts.prompt_loader import load_prompt_template


class CurrentStateAnalyzer:
    """Analyzes current state to determine current subtask and next atomic action."""

    def __init__(self, lm_config: lm_config.LMConfig) -> None:
        self.lm_config = lm_config

    def analyze_current_state(
        self,
        user_goal: str,
        subtasks: List[str],
        current_observation: Observation,
        context_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Analyze current state to determine current subtask and next atomic action.

        Args:
            user_goal: Original user goal
            subtasks: List of decomposed subtasks
            current_observation: Current page observation
            context_summary: Current context from Context Agent

        Returns:
            Dictionary containing state analysis results
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

        # Get context information
        cur_summary = context_summary.get("summary", "")

        # Build state analysis prompt using template
        subtasks_str = "\n".join([f"{i+1}. {subtask}" for i, subtask in enumerate(subtasks)])

        memory = context_summary.get("memory_content", "")
        # Build decomposition prompt using template
        if memory != "":
            prompt = load_prompt_template(
                "planner_agent",
                "current_state_analysis_w_mem",
                memory=memory,
                user_goal=user_goal,
                subtasks=subtasks_str,
                current_page_text=current_page_text,
                context_summary=cur_summary,
            )
        else:
            prompt = load_prompt_template(
                "planner_agent",
                "current_state_analysis",
                user_goal=user_goal,
                subtasks=subtasks_str,
                current_page_text=current_page_text,
                context_summary=cur_summary,
            )

        try:
            response = call_llm(
                self.lm_config, [{"role": "user", "content": prompt}]
            ).strip()
            # Parse the LLM response into structured format
            analysis = self._parse_analysis_response(response, subtasks)

        except Exception as e:
            # Fallback analysis
            print(f"🎯 Current State Analyzer: Error in analysis: {str(e)}")
            analysis = self._generate_fallback_analysis(user_goal, subtasks, str(e))

        return analysis

    def _parse_analysis_response(self, response: str, subtasks: List[str]) -> Dict[str, Any]:
        """Parse LLM state analysis response into structured format using XML tags."""
        import re

        # Default analysis structure
        analysis = {
            "current_subtask": "",
            "next_atomic_action": "",
            "reasoning": "",
            "response": response
        }

        try:
            # Extract content from XML tags using regex
            think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL | re.IGNORECASE)
            if think_match:
                analysis["reasoning"] = think_match.group(1).strip()

            current_subtask_match = re.search(r'<current_subtask>(.*?)</current_subtask>', response, re.DOTALL | re.IGNORECASE)
            if current_subtask_match:
                analysis["current_subtask"] = current_subtask_match.group(1).strip()

            next_action_match = re.search(r'<next_action>(.*?)</next_action>', response, re.DOTALL | re.IGNORECASE)
            if next_action_match:
                analysis["next_atomic_action"] = next_action_match.group(1).strip()

        except Exception as e:
            # If parsing fails, fall back to using the entire response
            analysis["reasoning"] = response
            analysis["next_atomic_action"] = response

        # If no current subtask extracted, try to match from subtasks list
        if not analysis["current_subtask"] and subtasks:
            # Simple keyword matching with the reasoning
            reasoning_lower = analysis["reasoning"].lower()
            for subtask in subtasks:
                subtask_lower = subtask.lower()
                # Match if any significant words from subtask appear in reasoning
                subtask_words = [word for word in subtask_lower.split() if len(word) > 3]
                if any(word in reasoning_lower for word in subtask_words):
                    analysis["current_subtask"] = subtask
                    break
            else:
                # Default to first subtask if no match found
                analysis["current_subtask"] = subtasks[0]

        # If no next action extracted, use a simple fallback
        if not analysis["next_atomic_action"]:
            print(f"🎯 Current State Analyzer: No next atomic action extracted, using current subtask: {analysis['current_subtask']}")
            analysis["next_atomic_action"] = f"Continue working on: {analysis['current_subtask']}"

        return analysis

    def _generate_fallback_analysis(self, user_goal: str, subtasks: List[str], error: str) -> Dict[str, Any]:
        """Generate fallback state analysis when LLM fails."""
        # Simple fallback logic
        if subtasks:
            current_subtask = subtasks[0]  # Default to first subtask
        else:
            current_subtask = f"Work on: {user_goal}"

        # Generate simple fallback action
        next_action = f"Continue working on: {current_subtask}"

        return {
            "current_subtask": current_subtask,
            "next_atomic_action": next_action,
            "reasoning": f"Fallback analysis due to error: {error}"
        }
