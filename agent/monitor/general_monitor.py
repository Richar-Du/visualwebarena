"""General Monitor for centralized trajectory monitoring and behavior correction.

Simplified Monitor that:
1. Calls Reflector every step to detect issues
2. When issues detected, calls PatternIssueAnalyzer for detailed guidance
3. Injects correction guidance (prohibited_actions, alternative_actions, correction_guidance) to Actor
4. Validates final results before STOP action
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
from PIL import Image
import numpy as np

from browser_env import Action, Trajectory
from browser_env.utils import Observation, pil_to_b64
from llms import lm_config, call_llm

from ..context_agent import ContextAgent
from ..reflector_agent import ReflectorAgent
from ..reflector.pattern_issue_analyzer import PatternIssueAnalyzer


@dataclass
class MonitorFeedback:
    """Structured feedback from Monitor to be injected into Actor's context."""
    has_issue: bool
    prohibited_actions: List[str]
    alternative_actions: List[str]
    correction_guidance: str
    context_summary: str
    
    def to_prompt_injection(self) -> str:
        """Generate the prompt injection string for Actor."""
        if not self.has_issue:
            return ""
        
        lines = []
        lines.append("=" * 60)
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
        
        # === Step 2: Call Reflector every step (starting after step 2) ===
        if self.step_count > 2:
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
            
            # If any issue detected, call PatternIssueAnalyzer for detailed guidance
            if has_pattern_issue or has_goal_deviation:
                print("⚠️ Monitor: Issue detected, getting detailed guidance...")
                
                recent_intents = [r["intention"] for r in self.action_history[-5:] if r.get("intention")]
                recent_actions = [r["action"] for r in self.action_history[-5:]]
                
                issue_analysis = self.pattern_issue_analyzer.analyze_pattern_issue(
                    recent_intents=recent_intents,
                    recent_actions=recent_actions,
                    current_intention=intention or self.global_goal,
                    latest_action=action,
                    high_level_task=self.global_goal,
                    checklist_raw_response=checklist.get("raw_response", ""),
                )
                
                feedback = MonitorFeedback(
                    has_issue=True,
                    prohibited_actions=issue_analysis.get("prohibited_actions", []),
                    alternative_actions=issue_analysis.get("alternative_actions", []),
                    correction_guidance=issue_analysis.get("correction_guidance", ""),
                    context_summary=context_summary,
                )
                
                print(f"📋 Guidance: {feedback.correction_guidance[:100]}..." if feedback.correction_guidance else "No guidance")
            else:
                feedback = MonitorFeedback(
                    has_issue=False,
                    prohibited_actions=[],
                    alternative_actions=[],
                    correction_guidance="",
                    context_summary=context_summary,
                )
            
            # Check for task completion
            if task_completed:
                self.feedback_history.append(feedback)
                return feedback, True
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
        
        # Check for STOP action
        should_stop = action.get("action_type") == "STOP"
        
        return feedback, should_stop
    
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
