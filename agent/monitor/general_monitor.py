"""General Monitor for centralized trajectory monitoring and behavior correction.

Simplified Monitor that:
1. Calls Reflector every step to detect issues
2. When issues detected, calls PatternIssueAnalyzer for detailed guidance
3. Injects correction guidance (prohibited_actions, alternative_actions, correction_guidance) to Actor
4. Validates final results before STOP action
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json
from PIL import Image
import numpy as np

from browser_env import Action, Trajectory
from browser_env.utils import Observation, pil_to_b64
from llms import lm_config, call_llm

from ..context_agent import ContextAgent
from ..reflector_agent import ReflectorAgent
from ..reflector.pattern_issue_analyzer import PatternIssueAnalyzer
from ..reflector.goal_deviation_analyzer import GoalDeviationAnalyzer


@dataclass
class MonitorFeedback:
    """Structured feedback from Monitor to be injected into Actor's context."""
    has_issue: bool
    prohibited_actions: List[str]
    alternative_actions: List[str]
    correction_guidance: str
    context_summary: str
    is_completion_guidance: bool = False  # Indicates task completion guidance (not an issue)
    checklist: Dict[str, Any] = field(default_factory=dict)  # Checklist analysis results for logging
    
    def to_prompt_injection(self) -> str:
        """Generate the prompt injection string for Actor."""
        # Generate feedback for issues OR completion guidance
        if not self.has_issue and not self.is_completion_guidance:
            return ""
        
        lines = []
        lines.append("=" * 60)
        
        if self.is_completion_guidance:
            lines.append("✅ MONITOR FEEDBACK - TASK COMPLETION DETECTED ✅")
        else:
            lines.append("⚠️ MONITOR FEEDBACK - PLEASE READ CAREFULLY ⚠️")
        lines.append("")
        
        if self.correction_guidance:
            lines.append(f"📋 GUIDANCE: {self.correction_guidance}")
            lines.append("")
        
        if self.prohibited_actions:
            lines.append("🚫 PROHIBITED ACTIONS (DO NOT DO THESE):")
            for i, action in enumerate(self.prohibited_actions[:3], 1):
                lines.append(f"   {i}. {action}")
            lines.append("")
        
        if self.alternative_actions:
            if self.is_completion_guidance:
                lines.append("✅ RECOMMENDED ACTION:")
            else:
                lines.append("✅ SUGGESTED ALTERNATIVES (TRY THESE):")
            for i, action in enumerate(self.alternative_actions[:3], 1):
                lines.append(f"   {i}. {action}")
            lines.append("")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


class GeneralMonitor:
    """Simplified Monitor that detects issues and provides correction guidance.
    
    Key Features:
    - Calls Reflector every step
    - When issues detected, uses PatternIssueAnalyzer for detailed guidance
    - Injects prohibited_actions, alternative_actions, correction_guidance to Actor
    - Validates final results before STOP action
    """
    
    def __init__(
        self,
        lm_config: lm_config.LMConfig,
        memory_config: Dict[str, Any] = {},
        **kwargs  # Accept but ignore legacy params
    ) -> None:
        """Initialize the General Monitor."""
        self.lm_config = lm_config
        
        # Initialize capability agents
        self.context_agent = ContextAgent(lm_config, memory_config)
        self.reflector_agent = ReflectorAgent(lm_config)
        self.pattern_issue_analyzer = PatternIssueAnalyzer(lm_config)
        self.goal_deviation_analyzer = GoalDeviationAnalyzer(lm_config)
        
        # Internal state tracking
        self.step_count: int = 0
        self.action_history: List[Dict[str, Any]] = []
        self.feedback_history: List[MonitorFeedback] = []
        self.global_goal: str = ""
    
    def initialize(self, global_goal: str) -> None:
        """Initialize monitor for a new task."""
        self.global_goal = global_goal
        self.step_count = 0
        self.action_history.clear()
        self.feedback_history.clear()
        
        # Reset capability agents
        self.context_agent.reset()
        self.reflector_agent.reset_reflection_history()
    
    def step(
        self,
        action: Action,
        observation: Observation,
        trajectory: Trajectory,
        intention: Optional[str] = None,
    ) -> Tuple[MonitorFeedback, bool]:
        """Main monitoring step after Actor executes an action.
        
        Returns:
            Tuple of (MonitorFeedback, should_stop)
        """
        self.step_count += 1
        
        # Record action in history
        action_record = {
            "step": self.step_count,
            "action": action,
            "action_type": action.get("action_type", "UNKNOWN"),
            "element_id": action.get("element_id"),
            "intention": intention,
        }
        self.action_history.append(action_record)
        
        # === Step 1: Update Context ===
        context_result = self.context_agent.update_context(
            trajectory=trajectory,
            user_goal=self.global_goal,
            current_observation=observation,
            latest_intention=intention,
            latest_action=action,
            latest_reflection=None,
        )
        context_summary = context_result.get("summary", "")
        
        # Check if this is a STOP action
        is_stop_action = action.get("action_type") == "STOP"
        
        # === Step 2: Call Reflector every step (starting after step 2) ===
        # Also call Reflector when STOP action is issued (regardless of step count)
        if is_stop_action:
            print(f"🔍 Monitor (Step {self.step_count}): Running Reflector analysis...")
            reflection_result = self._check_action(
                trajectory=trajectory,
                action=action,
                observation=observation,
                intention=intention,
            )
            
            checklist = reflection_result.get("checklist", {})
            has_pattern_issue = checklist.get("has_pattern_issue", False)
            has_goal_deviation = checklist.get("has_goal_deviation", False)
            task_completed = checklist.get("task_completed", False)
            task_completion_reason = checklist.get("task_completion_reason", "")
            
            # === Handle STOP action based on checklist's task_completed ===
            if is_stop_action:
                if task_completed and not has_goal_deviation:
                    # Checklist confirms task is complete AND no goal deviation, allow STOP
                    print(f"✅ Monitor: STOP action validated by checklist - task is complete")
                    if task_completion_reason:
                        print(f"   Reason: {task_completion_reason[:200]}")
                    feedback = MonitorFeedback(
                        has_issue=False,
                        prohibited_actions=[],
                        alternative_actions=[],
                        correction_guidance="",
                        context_summary=context_summary,
                        checklist=checklist,
                    )
                    self.feedback_history.append(feedback)
                    return feedback, True  # Allow STOP
                else:
                    # STOP rejected - determine the specific reason
                    goal_deviation_reason = checklist.get("goal_deviation_reason", "")
                    pattern_issue_reason = checklist.get("pattern_issue_reason", "")
                    
                    # Build rejection reason based on what was detected
                    rejection_reasons = []
                    prohibited_actions = ["stop - The current state does not match the task goal"]
                    alternative_actions = []
                    
                    if has_goal_deviation and goal_deviation_reason:
                        print(f"❌ Monitor: STOP rejected - Goal deviation detected!")
                        print(f"   Deviation: {goal_deviation_reason[:200]}")
                        rejection_reasons.append(f"GOAL DEVIATION: {goal_deviation_reason}")
                        
                        # Add specific guidance based on deviation type
                        reason_lower = goal_deviation_reason.lower()
                        if "attribute" in reason_lower or "color" in reason_lower or "price" in reason_lower:
                            prohibited_actions.append("Do not complete with the wrong product")
                            alternative_actions.extend([
                                "go_back - Return to search results",
                                "Find a product matching the required attributes"
                            ])
                        elif "pagination" in reason_lower or "page" in reason_lower:
                            prohibited_actions.append("Do not continue browsing pages")
                            alternative_actions.extend([
                                "go_back - Return to search page",
                                "Change search keywords to find the correct product"
                            ])
                        elif "search_pattern" in reason_lower or "scroll" in reason_lower:
                            prohibited_actions.append("Stop the current inefficient pattern")
                            alternative_actions.extend([
                                "go_back - Return to refine search",
                                "Try different search terms or filters"
                            ])
                        else:
                            alternative_actions.append("Review the task and adjust approach")
                    
                    if has_pattern_issue and pattern_issue_reason:
                        print(f"❌ Monitor: STOP rejected - Pattern issue detected!")
                        print(f"   Pattern: {pattern_issue_reason[:200]}")
                        rejection_reasons.append(f"PATTERN ISSUE: {pattern_issue_reason}")
                        alternative_actions.append("Break the repetitive pattern before completing")
                    
                    if not rejection_reasons and task_completion_reason:
                        # No specific issue detected, use task_completion_reason
                        print(f"❌ Monitor: STOP rejected - Task is NOT complete")
                        print(f"   Reason: {task_completion_reason[:200]}")
                        rejection_reasons.append(task_completion_reason)
                        alternative_actions.append("Continue working towards the task goal")
                    
                    if not rejection_reasons:
                        rejection_reasons.append("Task is not yet complete. Please continue working towards the objective.")
                        alternative_actions.append("Continue with the task")
                    
                    # Combine all rejection reasons into guidance
                    rejection_guidance = " | ".join(rejection_reasons)
                    
                    rejection_feedback = MonitorFeedback(
                        has_issue=True,
                        prohibited_actions=prohibited_actions,
                        alternative_actions=alternative_actions,
                        correction_guidance=rejection_guidance,
                        context_summary=context_summary,
                        checklist=checklist,
                    )
                    self.feedback_history.append(rejection_feedback)
                    return rejection_feedback, False  # Reject STOP, continue execution
            
            # Get goal_deviation_reason from checklist
            goal_deviation_reason = checklist.get("goal_deviation_reason", "")
            
            # === Handle goal deviation with detailed feedback ===
            if has_goal_deviation:
                print(f"⚠️ Monitor: Goal deviation detected!")
                if goal_deviation_reason:
                    print(f"   Reason: {goal_deviation_reason[:200]}")
                
                # Determine suggested action based on deviation type
                suggested_actions = []
                prohibited_actions = []
                
                reason_lower = goal_deviation_reason.lower()
                if "attribute" in reason_lower or "color" in reason_lower or "price" in reason_lower or "mismatch" in reason_lower:
                    # Product attribute mismatch - suggest going back
                    prohibited_actions = [
                        "Do not add this item to cart or wishlist",
                        "Do not proceed with checkout for this item"
                    ]
                    suggested_actions = [
                        "go_back - Return to search results to find the correct item",
                        "Look for items matching the required attributes"
                    ]
                elif "pagination" in reason_lower or "page" in reason_lower:
                    # Excessive pagination - suggest returning to start or changing query
                    prohibited_actions = [
                        "Do NOT continue to next page or scroll further",
                        "Do NOT click on items on this page - they are unlikely to match",
                        "Stop browsing through pagination"
                    ]
                    suggested_actions = [
                        "go_back - Return to search page and try different search keywords",
                        "type [search_box] [more_specific_keywords] - Use more specific search terms",
                        "Use category filters to narrow down results"
                    ]
                elif "search_pattern" in reason_lower or "scroll" in reason_lower:
                    # Inefficient search pattern - suggest changing strategy
                    prohibited_actions = [
                        "Stop scrolling through the same results",
                        "Do not continue clicking items in current list"
                    ]
                    suggested_actions = [
                        "go_back - Return to search page",
                        "type [search_box_id] [new_query] - Try different search keywords"
                    ]
                else:
                    # Task type violation - general guidance
                    prohibited_actions = ["Avoid actions that deviate from the task goal"]
                    suggested_actions = ["Review the task objective and adjust your approach"]
                
                feedback = MonitorFeedback(
                    has_issue=True,
                    prohibited_actions=prohibited_actions,
                    alternative_actions=suggested_actions,
                    correction_guidance=goal_deviation_reason if goal_deviation_reason else "Goal deviation detected. Please review your actions.",
                    context_summary=context_summary,
                    checklist=checklist,
                )
                
                print(f"📋 Deviation Guidance: {feedback.correction_guidance[:100]}...")
                self.feedback_history.append(feedback)
                return feedback, False
            
            # === Handle pattern issue with PatternIssueAnalyzer ===
            if has_pattern_issue:
                pattern_issue_reason = checklist.get("pattern_issue_reason", "")
                print(f"⚠️ Monitor: Pattern issue detected!")
                if pattern_issue_reason:
                    print(f"   Reason: {pattern_issue_reason[:200]}")
                
                recent_intents = [r["intention"] for r in self.action_history[-5:] if r.get("intention")]
                recent_actions = [r["action"] for r in self.action_history[-5:]]
                
                issue_analysis = self.pattern_issue_analyzer.analyze_pattern_issue(
                    recent_intents=recent_intents,
                    recent_actions=recent_actions,
                    current_intention=intention or self.global_goal,
                    latest_action=action,
                    high_level_task=self.global_goal,
                    pattern_issue_reason=pattern_issue_reason,
                )
                
                feedback = MonitorFeedback(
                    has_issue=True,
                    prohibited_actions=issue_analysis.get("prohibited_actions", []),
                    alternative_actions=issue_analysis.get("alternative_actions", []),
                    correction_guidance=issue_analysis.get("correction_guidance", pattern_issue_reason),
                    context_summary=context_summary,
                    checklist=checklist,
                )
                
                print(f"📋 Pattern Guidance: {feedback.correction_guidance[:100]}..." if feedback.correction_guidance else "No guidance")
                self.feedback_history.append(feedback)
                return feedback, False
            
            # No issues detected - check for task completion
            feedback = MonitorFeedback(
                has_issue=False,
                prohibited_actions=[],
                alternative_actions=[],
                correction_guidance="",
                context_summary=context_summary,
                checklist=checklist,
            )
            
            # Check for task completion - generate feedback for Actor to issue STOP
            if task_completed:
                print("✅ Monitor: Task appears complete, guiding Actor to issue STOP...")
                # Create feedback to guide Actor to issue STOP action
                completion_guidance = task_completion_reason if task_completion_reason else \
                    "The task appears to be complete based on the current state."
                completion_feedback = MonitorFeedback(
                    has_issue=False,  # Not an issue, but guidance
                    prohibited_actions=[],
                    alternative_actions=["stop [your_answer]"],
                    correction_guidance=f"{completion_guidance} Please issue a STOP action with the appropriate answer.",
                    context_summary=context_summary,
                    is_completion_guidance=True,  # Enable prompt injection for completion guidance
                    checklist=checklist,
                )
                self.feedback_history.append(completion_feedback)
                # Return should_stop=False to let Actor decide, but provide completion guidance
                return completion_feedback, False
        else:
            print(f"🔍 Monitor (Step {self.step_count}): Skipping Reflector analysis for early steps.")
            feedback = MonitorFeedback(
                has_issue=False,
                prohibited_actions=[],
                alternative_actions=[],
                correction_guidance="",
                context_summary=context_summary,
            )
        
        self.feedback_history.append(feedback)
        
        return feedback, False  # Never auto-stop, let checklist decide
    
    def _check_action(
        self,
        trajectory: Trajectory,
        action: Action,
        observation: Observation,
        intention: Optional[str],
    ) -> Dict[str, Any]:
        """Check action validity using ReflectorAgent."""
        recent_intentions = [r["intention"] for r in self.action_history[-5:] if r.get("intention")]
        recent_actions = [r["action"] for r in self.action_history[-5:]]
        
        return self.reflector_agent.reflect_execution(
            trajectory=trajectory,
            intentions=recent_intentions,
            actions=recent_actions,
            current_intention=intention or self.global_goal,
            latest_action=action,
            current_observation=observation,
            context_summary={"summary": self.context_agent.current_summary or ""},
            high_level_task=self.global_goal,
        )

    def validate_result(self, question: str, proposed_answer: str, current_observation: Optional[Observation] = None) -> Dict[str, Any]:
        """
        Pre-Output Check: Validate the answer before stopping.
        Uses the last page screenshot if available to help validation.
        """
        if not self.lm_config:
            return {"is_correct": True, "reason": "No LLM for validation"}
            
        content = []
        
        # Add screenshot if available
        if current_observation:
            obs_image = current_observation.get("image_raw")
            if obs_image is None:
                obs_image = current_observation.get("image")
            
            if isinstance(obs_image, np.ndarray):
                obs_image = Image.fromarray(obs_image)
                
            if obs_image:
                content.append({
                    "type": "text", 
                    "text": "Screenshot of the final page where the answer was found:"
                })
                content.append({
                    "type": "image_url",
                    "image_url": {"url": pil_to_b64(obs_image)}
                })

        user_prompt_text = f"""
User Goal: {question}
Proposed Answer: {proposed_answer}

Is this answer satisfactory and complete based on the goal? 
If NO, provide:
1. Specific reason why it is incorrect or incomplete.
2. Constructive suggestions on what the agent should do next to find the correct answer.

Output JSON: {{ 
    "is_correct": boolean, 
    "reason": "string",
    "suggestion": "string" 
}}
"""
        content.append({"type": "text", "text": user_prompt_text})

        prompt = [
            {"role": "system", "content": "You are a quality assurance monitor. Validate if the proposed answer satisfies the user goal."},
            {"role": "user", "content": content}
        ]
        
        try:
            response = call_llm(self.lm_config, prompt).strip()
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "{" in response:
                json_str = response[response.find("{"):response.rfind("}")+1]
            else:
                json_str = response
            
            result = json.loads(json_str)
            
            # If rejected, record this validation failure into context state
            if not result.get("is_correct", True):
                self.context_agent.update_state(
                    latest_reflection={
                        "monitor_validation": {
                            "status": "rejected",
                            "reason": result.get("reason"),
                            "suggestion": result.get("suggestion")
                        }
                    }
                )
                
            return result
        except Exception as e:
            print(f"⚠️ Result validation failed: {e}")
            return {"is_correct": True, "reason": "Validation error, allowing", "suggestion": ""}
    
    def get_latest_feedback(self) -> Optional[MonitorFeedback]:
        """Get the most recent feedback."""
        return self.feedback_history[-1] if self.feedback_history else None
    
    def get_prompt_injection(self) -> str:
        """Get the prompt injection string for the next Actor call."""
        feedback = self.get_latest_feedback()
        if feedback:
            return feedback.to_prompt_injection()
        return ""
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        issue_count = sum(1 for f in self.feedback_history if f.has_issue)
        return {
            "total_steps": self.step_count,
            "issue_count": issue_count,
            "feedback_count": len(self.feedback_history),
        }
    
    def reset(self) -> None:
        """Reset the monitor for a new task."""
        self.initialize(self.global_goal)
