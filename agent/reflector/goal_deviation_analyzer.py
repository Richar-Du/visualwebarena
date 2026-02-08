"""Goal Deviation Analyzer for detecting product attribute mismatches and inefficient search patterns.

This module performs detailed analysis when goal deviation is detected, including:
1. Product Attribute Check: Verify product matches requirements (color, price, quantity, style, etc.)
2. Search Pattern Check: Detect inefficient scrolling/clicking patterns that indicate wrong search results
3. Generate actionable feedback for course correction
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


class GoalDeviationAnalyzer:
    """Analyzes goal deviation and provides correction guidance.
    
    Detects two main types of issues:
    1. Product Attribute Mismatch: Wrong color, price, quantity, style, etc.
    2. Inefficient Search Pattern: Repeated scroll-click loops without finding target
    """
    
    # Pattern for detecting inefficient search: type -> (scroll -> click)+ pattern
    SCROLL_CLICK_PATTERN_THRESHOLD = 3  # Number of scroll-click pairs to trigger warning
    MAX_PAGINATION_PAGE = 3  # Maximum page number before triggering warning
    
    def __init__(self, lm_config: lm_config.LMConfig) -> None:
        self.lm_config = lm_config
        self.is_multimodal = is_multimodal_model(lm_config.model)
    
    def analyze(
        self,
        recent_actions: List[Action],
        high_level_task: str,
        current_observation: Optional[np.ndarray] = None,
        context_summary: str = "",
        current_url: str = "",
        start_url: str = "",
    ) -> Dict[str, Any]:
        """Perform comprehensive goal deviation analysis.
        
        Args:
            recent_actions: Recent action history for pattern detection
            high_level_task: Original task goal with product requirements
            current_observation: Current screenshot for visual verification
            context_summary: Current context summary
            current_url: Current page URL for pagination detection
            start_url: Initial task URL for suggesting return action
            
        Returns:
            Dictionary containing:
                - has_deviation: bool - Whether deviation is detected
                - deviation_type: str - Type of deviation (attribute_mismatch, scroll_pattern, pagination, etc.)
                - should_intercept: bool - Whether to intercept current action
                - intercept_action: str - What action to take (go_back, change_query, etc.)
                - correction_guidance: str - Detailed guidance for Actor
                - prohibited_actions: List[str] - Actions to avoid
                - alternative_actions: List[str] - Suggested alternatives
        """
        result = self._get_default_result()
        
        # Step 0: Check URL for pagination (e.g., p=3, page=3)
        if current_url:
            pagination_result = self._detect_pagination_from_url(current_url, start_url)
            if pagination_result["detected"]:
                result.update({
                    "has_deviation": True,
                    "deviation_type": "pagination",
                    "should_intercept": True,
                    "intercept_action": "return_to_start",
                    "correction_guidance": pagination_result["guidance"],
                    "prohibited_actions": pagination_result["prohibited_actions"],
                    "alternative_actions": pagination_result["alternative_actions"],
                })
                return result
        
        # Step 1: Check for inefficient scroll-click pattern
        scroll_pattern_result = self._detect_scroll_click_pattern(recent_actions)
        if scroll_pattern_result["detected"]:
            result.update({
                "has_deviation": True,
                "deviation_type": "scroll_pattern",
                "should_intercept": True,
                "intercept_action": "change_search_strategy",
                "correction_guidance": scroll_pattern_result["guidance"],
                "prohibited_actions": scroll_pattern_result["prohibited_actions"],
                "alternative_actions": scroll_pattern_result["alternative_actions"],
            })
            return result
        
        # Step 2: Use LLM to check product attribute match (if multimodal and image available)
        if self.is_multimodal and current_observation is not None:
            attribute_result = self._check_product_attributes(
                high_level_task=high_level_task,
                current_observation=current_observation,
                context_summary=context_summary,
                recent_actions=recent_actions,
            )
            if attribute_result.get("has_mismatch", False):
                result.update({
                    "has_deviation": True,
                    "deviation_type": "attribute_mismatch",
                    "should_intercept": attribute_result.get("should_intercept", False),
                    "intercept_action": attribute_result.get("intercept_action", "go_back"),
                    "correction_guidance": attribute_result.get("correction_guidance", ""),
                    "prohibited_actions": attribute_result.get("prohibited_actions", []),
                    "alternative_actions": attribute_result.get("alternative_actions", []),
                })
                return result
        
        return result
    
    def _detect_pagination_from_url(self, current_url: str, start_url: str = "") -> Dict[str, Any]:
        """Detect if the current URL indicates excessive pagination.
        
        Looks for patterns like:
        - p=3, p=4, etc.
        - page=3, page=4, etc.
        - /page/3, /page/4, etc.
        
        If page number >= MAX_PAGINATION_PAGE, trigger warning.
        """
        result = {
            "detected": False,
            "page_number": 0,
            "guidance": "",
            "prohibited_actions": [],
            "alternative_actions": [],
        }
        
        if not current_url:
            return result
        
        # Try to extract page number from URL using various patterns
        page_number = 0
        
        # Pattern 1: p=N or page=N in query string
        page_match = re.search(r'[?&](p|page)=(\d+)', current_url, re.IGNORECASE)
        if page_match:
            page_number = int(page_match.group(2))
        
        # Pattern 2: /page/N in path
        if page_number == 0:
            path_match = re.search(r'/page/(\d+)', current_url, re.IGNORECASE)
            if path_match:
                page_number = int(path_match.group(1))
        
        # Pattern 3: /p/N in path
        if page_number == 0:
            path_match = re.search(r'/p/(\d+)', current_url, re.IGNORECASE)
            if path_match:
                page_number = int(path_match.group(1))
        
        # Pattern 4: -pageN or _pageN
        if page_number == 0:
            suffix_match = re.search(r'[-_]page(\d+)', current_url, re.IGNORECASE)
            if suffix_match:
                page_number = int(suffix_match.group(1))
        
        result["page_number"] = page_number
        
        # Trigger warning if page number >= threshold
        if page_number >= self.MAX_PAGINATION_PAGE:
            result["detected"] = True
            
            # Prepare guidance with start_url if available
            start_url_hint = f" Return to the start page ({start_url}) and" if start_url else ""
            
            result["guidance"] = (
                f"⚠️ EXCESSIVE PAGINATION DETECTED: You are currently on page {page_number} of search results. "
                f"Browsing through many pages of results is inefficient and unlikely to find the target product. "
                f"The item you're looking for is probably not in this search.{start_url_hint} "
                f"try using different search keywords or filters to find the correct product."
            )
            result["prohibited_actions"] = [
                "Do NOT continue to next page or scroll further",
                "Do NOT click on items on this page - they are unlikely to match",
                "Stop browsing through pagination"
            ]
            
            if start_url:
                result["alternative_actions"] = [
                    f"goto [{start_url}] - Return to start page and try a different approach",
                    "go_back - Go back to search page and modify search query",
                    "type [search_box] [more_specific_keywords] - Use more specific search terms"
                ]
            else:
                result["alternative_actions"] = [
                    "go_back - Return to search page",
                    "type [search_box] [more_specific_keywords] - Use more specific search terms",
                    "Use category filters to narrow down results"
                ]
        
        return result
    
    def _detect_scroll_click_pattern(self, recent_actions: List[Action]) -> Dict[str, Any]:
        """Detect inefficient scroll-click patterns in action history.
        
        Pattern: type [query] -> (scroll down -> click)+ repeated multiple times
        This indicates the agent is scrolling through results without finding the target.
        """
        result = {
            "detected": False,
            "guidance": "",
            "prohibited_actions": [],
            "alternative_actions": [],
        }
        
        if len(recent_actions) < 4:
            return result
        
        # Extract action types from recent actions
        action_sequence = []
        for action in recent_actions[-10:]:  # Look at last 10 actions
            action_type = str(action.get("action_type", "")).upper()
            if action_type in ["SCROLL", "CLICK", "TYPE", "STOP"]:
                # For scroll, also track direction
                if action_type == "SCROLL":
                    direction = action.get("direction", "down")
                    action_sequence.append(f"SCROLL_{direction.upper()}")
                else:
                    action_sequence.append(action_type)
        
        # Count scroll-click pairs
        scroll_click_count = 0
        i = 0
        while i < len(action_sequence) - 1:
            if action_sequence[i] == "SCROLL_DOWN" and action_sequence[i+1] == "CLICK":
                scroll_click_count += 1
                i += 2
            else:
                i += 1
        
        # Also check for consecutive scrolls without meaningful progress
        consecutive_scrolls = 0
        max_consecutive_scrolls = 0
        for action in action_sequence:
            if action == "SCROLL_DOWN":
                consecutive_scrolls += 1
                max_consecutive_scrolls = max(max_consecutive_scrolls, consecutive_scrolls)
            else:
                consecutive_scrolls = 0
        
        # Trigger if we see repeated scroll-click pattern or too many consecutive scrolls
        if scroll_click_count >= self.SCROLL_CLICK_PATTERN_THRESHOLD:
            result["detected"] = True
            result["guidance"] = (
                f"Detected inefficient search pattern: You have performed {scroll_click_count} "
                f"scroll-click cycles without finding the target product. This suggests the current "
                f"search results may not contain what you're looking for. "
                f"RECOMMENDED: Return to the search page and try a different search query, "
                f"or use more specific keywords to narrow down results."
            )
            result["prohibited_actions"] = [
                "Do not continue scrolling through the same results",
                "Do not click on more items in the current list"
            ]
            result["alternative_actions"] = [
                "go_back - Return to search/listing page",
                "type [new_query] - Try a different search query with more specific keywords",
                "Use filters to narrow down results (price range, category, etc.)"
            ]
        elif max_consecutive_scrolls >= 4:
            result["detected"] = True
            result["guidance"] = (
                f"Detected excessive scrolling: You have scrolled {max_consecutive_scrolls} times "
                f"consecutively. This may indicate difficulty finding the target item in the current view. "
                f"Consider refining your search or using filters."
            )
            result["prohibited_actions"] = [
                "Avoid further scrolling without interaction"
            ]
            result["alternative_actions"] = [
                "go_back - Return to refine search",
                "Use category or filter navigation"
            ]
        
        return result
    
    def _check_product_attributes(
        self,
        high_level_task: str,
        current_observation: np.ndarray,
        context_summary: str,
        recent_actions: List[Action],
    ) -> Dict[str, Any]:
        """Use LLM to verify product attributes match task requirements."""
        
        try:
            # Convert numpy array to PIL Image
            if isinstance(current_observation, np.ndarray):
                pil_image = Image.fromarray(current_observation)
            else:
                pil_image = current_observation
            
            # Format recent actions
            actions_str = "\n".join([
                f"{i+1}. {self._format_action(action)}" 
                for i, action in enumerate(recent_actions[-5:])
            ])
            
            # Load prompt template
            prompt_text = load_prompt_template(
                "reflector_agent",
                "goal_deviation_analysis",
                high_level_task=high_level_task,
                context_summary=context_summary,
                recent_actions=actions_str,
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
            
            response = call_llm(self.lm_config, [{"role": "user", "content": content}]).strip()
            return self._parse_attribute_response(response)
            
        except Exception as e:
            print(f"Product attribute check failed: {e}")
            return {"has_mismatch": False}
    
    def _parse_attribute_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response for product attribute analysis."""
        result = {
            "has_mismatch": False,
            "should_intercept": False,
            "intercept_action": "",
            "correction_guidance": "",
            "prohibited_actions": [],
            "alternative_actions": [],
        }
        
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                parsed = json.loads(json_str)
                
                result["has_mismatch"] = self._to_bool(parsed.get("has_mismatch", False))
                result["should_intercept"] = self._to_bool(parsed.get("should_intercept", False))
                result["intercept_action"] = parsed.get("intercept_action", "go_back")
                result["correction_guidance"] = parsed.get("correction_guidance", "")
                
                if "prohibited_actions" in parsed and isinstance(parsed["prohibited_actions"], list):
                    result["prohibited_actions"] = parsed["prohibited_actions"][:3]
                if "alternative_actions" in parsed and isinstance(parsed["alternative_actions"], list):
                    result["alternative_actions"] = parsed["alternative_actions"][:3]
                    
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Error parsing attribute response: {e}")
        
        return result
    
    def _format_action(self, action: Action) -> str:
        """Format an action object into a readable string."""
        if not action or not isinstance(action, dict):
            return "Invalid action"
        
        action_type = action.get("action_type", "UNKNOWN")
        element_id = action.get("element_id", "N/A")
        
        # Decode text if present
        text = self._decode_action_text(action)
        
        if action_type == "SCROLL":
            direction = action.get("direction", "unknown")
            return f"scroll [{direction}]"
        elif action_type == "CLICK":
            return f"click [{element_id}]"
        elif action_type == "TYPE":
            return f"type [{element_id}] [{text}]"
        elif action_type == "STOP":
            return f"stop [{text}]"
        else:
            return f"{action_type} [{element_id}]"
    
    def _decode_action_text(self, action: Action) -> str:
        """Decode action text from ID list to readable string."""
        text_ids = action.get("text", [])
        if isinstance(text_ids, list) and text_ids:
            try:
                from browser_env.actions import _id2key
                return ''.join(_id2key[id_num] if 0 <= id_num < len(_id2key) else '?' for id_num in text_ids)
            except (ImportError, IndexError):
                return ''.join(chr(id_num) if 32 <= id_num <= 126 else '?' for id_num in text_ids)
        return ""
    
    def _to_bool(self, value: Any) -> bool:
        """Convert various value types to boolean."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "yes", "1")
        if isinstance(value, (int, float)):
            return bool(value)
        return False
    
    def _get_default_result(self) -> Dict[str, Any]:
        """Get default result structure."""
        return {
            "has_deviation": False,
            "deviation_type": "",
            "should_intercept": False,
            "intercept_action": "",
            "correction_guidance": "",
            "prohibited_actions": [],
            "alternative_actions": [],
        }
