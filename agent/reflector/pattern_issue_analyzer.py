"""Pattern Issue Analyzer for detailed pattern error analysis and correction suggestions.

This module performs detailed analysis when pattern issues are detected, providing
specific correction guidance to prevent repetitive errors.
"""

from typing import Any, Dict, List, Optional
import json

from browser_env import Action
from llms import lm_config, call_llm
from ..prompts.prompt_loader import load_prompt_template


class PatternIssueAnalyzer:
    """Analyzes pattern issues in detail and provides correction suggestions.

    When a pattern issue is initially detected, this analyzer performs a deeper
    analysis to confirm the issue and provide specific guidance on what actions
    to avoid and what alternatives to try.
    """

    def __init__(self, lm_config: lm_config.LMConfig) -> None:
        self.lm_config = lm_config

    def analyze_pattern_issue(
        self,
        recent_intents: List[str],
        recent_actions: List[Action],
        current_intention: str,
        latest_action: Action,
        high_level_task: str,
        checklist_raw_response: str,
    ) -> Dict[str, Any]:
        """Analyze pattern issue in detail and provide correction suggestions.

        Args:
            recent_intents: Last few intentions that led to the pattern
            recent_actions: Last few actions that show the pattern
            current_intention: Current intention being executed
            latest_action: The most recent action executed
            high_level_task: Original high-level task goal
            checklist_raw_response: Raw response from initial checklist analysis

        Returns:
            Dictionary containing detailed pattern analysis and correction suggestions
        """
        # Format inputs for LLM analysis
        intents_str = "\n".join([f"{i+1}. {intent}" for i, intent in enumerate(recent_intents)])
        if not intents_str:
            intents_str = "No previous intents"

        actions_str = "\n".join([
            f"{i+1}. {self._format_action(action)}"
            for i, action in enumerate(recent_actions)
        ])
        if not actions_str:
            actions_str = "No previous actions"

        current_action_str = self._format_action(latest_action)

        # Load and format prompt
        prompt = load_prompt_template(
            "reflector_agent",
            "pattern_issue_analysis",
            recent_intents=intents_str,
            recent_actions=actions_str,
            current_intention=current_intention,
            latest_action=current_action_str,
            high_level_task=high_level_task,
            checklist_response=checklist_raw_response,
        )

        try:
            response = call_llm(
                self.lm_config, [{"role": "user", "content": prompt}]
            ).strip()

            # Parse the response
            analysis_result = self._parse_analysis_response(response)
            analysis_result["analysis_successful"] = True

        except Exception as e:
            print(f"Pattern issue analysis failed: {e}")
            analysis_result = self._get_default_analysis_result()
            analysis_result["analysis_successful"] = False

        return analysis_result

    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response for pattern issue analysis.

        Expected response format should contain:
        - pattern_confirmed: boolean
        - pattern_description: string description
        - prohibited_actions: list of actions to avoid
        - alternative_actions: list of suggested actions
        - correction_guidance: general guidance text

        Args:
            response: Raw LLM response

        Returns:
            Parsed analysis result
        """
        result = self._get_default_analysis_result()

        try:
            # Try to extract JSON from response
            json_match = response.find('{')
            json_end = response.rfind('}') + 1
            if json_match != -1 and json_end > json_match:
                json_str = response[json_match:json_end]
                parsed = json.loads(json_str)

                # Extract fields
                result["pattern_confirmed"] = parsed.get("pattern_confirmed", False)
                result["pattern_description"] = parsed.get("pattern_description", "")
                result["prohibited_actions"] = parsed.get("prohibited_actions", [])
                result["alternative_actions"] = parsed.get("alternative_actions", [])
                result["correction_guidance"] = parsed.get("correction_guidance", "")

                return result

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Error parsing pattern analysis JSON: {e}")

        # Fallback: extract from text patterns
        result = self._parse_text_patterns(response)

        return result

    def _parse_text_patterns(self, response: str) -> Dict[str, Any]:
        """Fallback parsing using text patterns when JSON parsing fails."""
        result = self._get_default_analysis_result()
        response_lower = response.lower()

        # Check if pattern is confirmed
        if "pattern_confirmed: true" in response_lower or "confirmed: true" in response_lower:
            result["pattern_confirmed"] = True
        elif "no pattern" in response_lower or "pattern_confirmed: false" in response_lower:
            result["pattern_confirmed"] = False
            return result  # Early return if no pattern

        # Extract pattern description
        desc_patterns = [
            r"pattern description:?\s*(.+?)(?:\n|$)",
            r"description:?\s*(.+?)(?:\n|$)",
        ]
        for pattern in desc_patterns:
            import re
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                result["pattern_description"] = match.group(1).strip()
                break

        # Extract prohibited actions (look for lists or bullet points)
        prohibited_section = self._extract_section(response, "prohibited", "alternative")
        if prohibited_section:
            result["prohibited_actions"] = self._extract_list_items(prohibited_section)

        # Extract alternative actions
        alternative_section = self._extract_section(response, "alternative", "")
        if alternative_section:
            result["alternative_actions"] = self._extract_list_items(alternative_section)

        # Extract guidance
        guidance_patterns = [
            r"guidance:?\s*(.+?)(?:\n|$)",
            r"correction guidance:?\s*(.+?)(?:\n|$)",
        ]
        for pattern in guidance_patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                result["correction_guidance"] = match.group(1).strip()
                break

        return result

    def _extract_section(self, text: str, start_keyword: str, end_keyword: str) -> str:
        """Extract text between two keywords."""
        import re
        start_pattern = rf"{start_keyword}.*?:?\s*"
        start_match = re.search(start_pattern, text, re.IGNORECASE)

        if not start_match:
            return ""

        start_pos = start_match.end()

        if end_keyword:
            end_pattern = rf"{end_keyword}.*?:?\s*"
            end_match = re.search(end_pattern, text[start_pos:], re.IGNORECASE)
            if end_match:
                end_pos = start_pos + end_match.start()
            else:
                end_pos = len(text)
        else:
            end_pos = len(text)

        return text[start_pos:end_pos].strip()

    def _extract_list_items(self, text: str) -> List[str]:
        """Extract list items from text (bullet points, numbered lists, etc.)."""
        import re
        items = []

        # Look for bullet points or numbered items
        patterns = [
            r"[•\-\*]\s*(.+?)(?=[•\-\*]|\d+\.|$)",  # Bullet points
            r"\d+\.\s*(.+?)(?=\d+\.|$)",  # Numbered lists
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            if matches:
                items.extend([match.strip() for match in matches if match.strip()])

        # If no structured lists found, try to split by newlines
        if not items:
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            items = [line for line in lines if len(line) > 10]  # Filter out very short lines

        return items

    def _get_default_analysis_result(self) -> Dict[str, Any]:
        """Get default analysis result structure."""
        return {
            "pattern_confirmed": False,
            "pattern_description": "",
            "prohibited_actions": [],
            "alternative_actions": [],
            "correction_guidance": "",
            "analysis_successful": False,
        }

    def _format_action(self, action: Action) -> str:
        """Format an action object into a readable string."""
        if not action or not isinstance(action, dict):
            return "Invalid action"

        action_type = action.get("action_type", "UNKNOWN")
        element_id = action.get("element_id", "N/A")
        action_text = self._decode_action_text(action)

        formatted_parts = [f"Type: {action_type}"]
        if element_id != "N/A":
            formatted_parts.append(f"Element: {element_id}")
        if action_text != "N/A":
            formatted_parts.append(f"Text: '{action_text}'")

        return ", ".join(formatted_parts)

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
