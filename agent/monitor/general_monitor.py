"""General Monitor for centralized trajectory monitoring and behavior correction.

This module implements a unified monitoring system that:
1. Records trajectory history and generates concise context summaries
2. Detects anomalous behaviors (repetition, goal deviation, stuck states)
3. Provides corrective feedback to guide agent behavior

Design Pattern: "Monitor as Primary, Capability by Reference"
- Monitor takes over post-action processing
- Internally references Context and Reflector capabilities
- Provides unified decision: ALLOW, WARN, or ROLLBACK
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import copy

from browser_env import Action, Trajectory
from browser_env.utils import Observation
from llms import lm_config

from ..context_agent import ContextAgent
from ..reflector_agent import ReflectorAgent


class MonitorDecision(Enum):
    """Monitor's decision after observing an action."""
    ALLOW = "allow"           # Action is fine, continue normally
    WARN = "warn"             # Issue detected, inject warning feedback
    ROLLBACK = "rollback"     # Severe issue, suggest different approach
    STOP = "stop"             # Task should be terminated


@dataclass
class MonitorFeedback:
    """Structured feedback from the Monitor to be injected into Actor's context."""
    decision: MonitorDecision
    message: str                          # Human-readable feedback message
    severity: int                         # 1-10 severity level
    detected_issues: List[str]            # List of detected issues
    suggested_actions: List[str]          # Suggested alternative actions
    context_summary: str                  # Current context summary
    inject_to_prompt: bool                # Whether to inject into Actor's prompt
    
    def to_prompt_injection(self) -> str:
        """Generate the prompt injection string for Actor.
        
        Enhanced to be more emphatic and include explicit prohibitions.
        """
        if not self.inject_to_prompt:
            return ""
        
        if self.decision == MonitorDecision.ALLOW:
            return ""
        
        lines = []
        
        # Header with strong emphasis
        lines.append("=" * 60)
        
        if self.decision == MonitorDecision.WARN:
            lines.append("⚠️⚠️⚠️ CRITICAL MONITOR WARNING ⚠️⚠️⚠️")
            lines.append(f"Message: {self.message}")
            lines.append("")
            lines.append("🚫 YOU MUST NOT repeat the same action!")
            lines.append("🚫 DO NOT click the same element again!")
            lines.append("✅ You MUST try a DIFFERENT approach!")
            
        elif self.decision == MonitorDecision.ROLLBACK:
            lines.append("🚨🚨🚨 MONITOR ALERT - IMMEDIATE ACTION REQUIRED 🚨🚨🚨")
            lines.append(f"Message: {self.message}")
            lines.append("")
            lines.append("❌ YOUR PREVIOUS APPROACH HAS FAILED!")
            lines.append("❌ DO NOT continue with the same strategy!")
            lines.append("✅ You MUST try a COMPLETELY DIFFERENT approach!")
            lines.append("✅ Consider: scrolling, navigating to a different page, or using different elements!")
            
        elif self.decision == MonitorDecision.STOP:
            lines.append("🛑🛑🛑 MONITOR STOP COMMAND 🛑🛑🛑")
            lines.append(f"Message: {self.message}")
        
        lines.append("")
        
        if self.detected_issues:
            lines.append("📋 DETECTED ISSUES:")
            for i, issue in enumerate(self.detected_issues[:3], 1):
                lines.append(f"   {i}. {issue}")
            lines.append("")
        
        # Generate explicit prohibitions based on action history
        if self.decision in [MonitorDecision.WARN, MonitorDecision.ROLLBACK]:
            lines.append("🚫 PROHIBITED ACTIONS (DO NOT DO THESE):")
            lines.append("   - DO NOT click the same button/element again")
            lines.append("   - DO NOT type the same text again")
            lines.append("   - DO NOT repeat any recent action")
            lines.append("")
        
        if self.suggested_actions and self.decision in [MonitorDecision.WARN, MonitorDecision.ROLLBACK]:
            lines.append("✅ REQUIRED ALTERNATIVES (DO ONE OF THESE):")
            for i, action in enumerate(self.suggested_actions[:3], 1):
                lines.append(f"   {i}. {action}")
            lines.append("")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


class GeneralMonitor:
    """Centralized Monitor that supervises agent execution.
    
    The Monitor takes over post-action processing by:
    1. Calling Context capability to record and summarize trajectory
    2. Calling Reflector capability to check action validity
    3. Running Global Check logic to detect patterns requiring intervention
    
    Key Features:
    - Consecutive error detection (e.g., 3 Reflector errors -> force intervention)
    - Repetitive action detection
    - Goal deviation detection
    - Stuck state detection
    """
    
    def __init__(
        self,
        lm_config: lm_config.LMConfig,
        memory_config: Dict[str, Any] = {},
        # Thresholds for anomaly detection
        consecutive_error_threshold: int = 3,
        repetition_window: int = 5,
        max_same_action_count: int = 3,
    ) -> None:
        """Initialize the General Monitor.
        
        Args:
            lm_config: Language model configuration
            memory_config: Memory configuration passed to ContextAgent
            consecutive_error_threshold: Number of consecutive errors before intervention
            repetition_window: Window size for detecting repetitive actions
            max_same_action_count: Max times same action can repeat before warning
        """
        self.lm_config = lm_config
        
        # Initialize capability agents (by reference)
        self.context_agent = ContextAgent(lm_config, memory_config)
        self.reflector_agent = ReflectorAgent(lm_config)
        
        # Thresholds
        self.consecutive_error_threshold = consecutive_error_threshold
        self.repetition_window = repetition_window
        self.max_same_action_count = max_same_action_count
        
        # Internal state tracking
        self.step_count: int = 0
        self.consecutive_errors: int = 0
        self.action_history: List[Dict[str, Any]] = []
        self.feedback_history: List[MonitorFeedback] = []
        self.global_goal: str = ""
        
        # Issue counters for pattern detection
        self.pattern_issue_count: int = 0
        self.stuck_detection_count: int = 0
        self.deviation_warnings: int = 0
    
    def initialize(self, global_goal: str) -> None:
        """Initialize monitor for a new task.
        
        Args:
            global_goal: The high-level task goal
        """
        self.global_goal = global_goal
        self.step_count = 0
        self.consecutive_errors = 0
        self.action_history.clear()
        self.feedback_history.clear()
        self.pattern_issue_count = 0
        self.stuck_detection_count = 0
        self.deviation_warnings = 0
        
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
        
        This method:
        1. Records the action and updates context (via ContextAgent)
        2. Checks action validity (via ReflectorAgent)
        3. Runs global check logic for pattern detection
        4. Generates appropriate feedback
        
        Args:
            action: The action that was executed
            observation: The resulting observation
            trajectory: Full execution trajectory
            intention: The intention behind the action (optional)
        
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
        
        # === Step 1: Update Context (via ContextAgent capability) ===
        context_result = self._update_context(
            trajectory=trajectory,
            observation=observation,
            action=action,
            intention=intention,
        )
        context_summary = context_result.get("summary", "")
        
        # === Step 2: Check Action (via ReflectorAgent capability) ===
        reflection_result = self._check_action(
            trajectory=trajectory,
            action=action,
            observation=observation,
            intention=intention,
        )
        
        # === Step 3: Global Check Logic ===
        global_check_result = self._global_check(
            action=action,
            observation=observation,
            reflection_result=reflection_result,
        )
        
        # === Step 4: Generate Feedback ===
        feedback = self._generate_feedback(
            context_summary=context_summary,
            reflection_result=reflection_result,
            global_check_result=global_check_result,
        )
        
        self.feedback_history.append(feedback)
        
        # Determine if execution should stop
        should_stop = feedback.decision == MonitorDecision.STOP
        
        # Check for STOP action
        if action.get("action_type") == "STOP":
            should_stop = True
        
        return feedback, should_stop
    
    def _update_context(
        self,
        trajectory: Trajectory,
        observation: Observation,
        action: Action,
        intention: Optional[str],
    ) -> Dict[str, Any]:
        """Update context using ContextAgent capability.
        
        Returns context result including summary.
        """
        return self.context_agent.update_context(
            trajectory=trajectory,
            user_goal=self.global_goal,
            current_observation=observation,
            latest_intention=intention,
            latest_action=action,
            latest_reflection=None,  # Will be updated after reflection
        )
    
    def _check_action(
        self,
        trajectory: Trajectory,
        action: Action,
        observation: Observation,
        intention: Optional[str],
    ) -> Dict[str, Any]:
        """Check action validity using ReflectorAgent capability.
        
        Returns reflection result.
        """
        # Get recent intentions and actions for reflection
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
    
    def _global_check(
        self,
        action: Action,
        observation: Observation,
        reflection_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Perform global pattern checks.
        
        This is the key NEW logic that goes beyond individual reflection:
        - Detects consecutive errors
        - Detects repetitive actions
        - Detects stuck states
        - Detects goal deviation
        
        Returns:
            Dictionary with check results and detected issues
        """
        issues = []
        severity = 0
        suggested_actions = []
        
        # === Check 1: Consecutive Reflector Errors ===
        checklist = reflection_result.get("checklist", {})
        has_pattern_issue = checklist.get("has_pattern_issue", False)
        
        if has_pattern_issue:
            self.consecutive_errors += 1
            self.pattern_issue_count += 1
        else:
            self.consecutive_errors = 0  # Reset on success
        
        if self.consecutive_errors >= self.consecutive_error_threshold:
            issues.append(f"Consecutive errors detected ({self.consecutive_errors} times)")
            severity = max(severity, 8)
            suggested_actions.append("Try a completely different approach")
            suggested_actions.append("Navigate to a different page")
            suggested_actions.append("Use search or navigation elements")
        
        # === Check 2: Repetitive Actions ===
        repetition_result = self._detect_repetition()
        if repetition_result["is_repetitive"]:
            issues.append(f"Repetitive action detected: {repetition_result['description']}")
            severity = max(severity, 6)
            suggested_actions.extend(repetition_result.get("alternatives", []))
        
        # === Check 3: Stuck State Detection ===
        stuck_result = self._detect_stuck_state(observation)
        if stuck_result["is_stuck"]:
            self.stuck_detection_count += 1
            issues.append(f"Agent appears stuck: {stuck_result['reason']}")
            severity = max(severity, 7)
            suggested_actions.append("Try scrolling to find more elements")
            suggested_actions.append("Try a different navigation path")
        
        # === Check 4: Pattern Issue Analysis from Reflector ===
        pattern_analysis = checklist.get("pattern_issue_analysis", {})
        if pattern_analysis.get("pattern_confirmed", False):
            issues.append(f"Pattern issue: {pattern_analysis.get('pattern_description', 'Unknown')}")
            severity = max(severity, 7)
            # Use Reflector's suggested alternatives
            reflector_alternatives = pattern_analysis.get("alternative_actions", [])
            suggested_actions.extend(reflector_alternatives[:2])
        
        return {
            "has_issues": len(issues) > 0,
            "issues": issues,
            "severity": severity,
            "suggested_actions": suggested_actions,
            "consecutive_errors": self.consecutive_errors,
            "pattern_issue_count": self.pattern_issue_count,
            "stuck_count": self.stuck_detection_count,
        }
    
    def _detect_repetition(self) -> Dict[str, Any]:
        """Detect repetitive actions in recent history.
        
        Returns:
            Dictionary with repetition detection result
        """
        if len(self.action_history) < 2:
            return {"is_repetitive": False}
        
        recent = self.action_history[-self.repetition_window:]
        
        # Count occurrences of same action type + element
        action_signatures = []
        for record in recent:
            action = record["action"]
            sig = f"{action.get('action_type', 'UNK')}:{action.get('element_id', 'N/A')}"
            action_signatures.append(sig)
        
        # Find most common signature
        from collections import Counter
        signature_counts = Counter(action_signatures)
        most_common, count = signature_counts.most_common(1)[0]
        
        if count >= self.max_same_action_count:
            return {
                "is_repetitive": True,
                "description": f"Same action '{most_common}' repeated {count} times",
                "repeated_action": most_common,
                "count": count,
                "alternatives": [
                    "Try a different element",
                    "Try a different action type",
                    "Scroll to find new elements",
                ],
            }
        
        # Check for oscillation pattern (A -> B -> A -> B)
        if len(action_signatures) >= 4:
            if (action_signatures[-1] == action_signatures[-3] and 
                action_signatures[-2] == action_signatures[-4] and
                action_signatures[-1] != action_signatures[-2]):
                return {
                    "is_repetitive": True,
                    "description": "Oscillation pattern detected (switching between two actions)",
                    "alternatives": [
                        "Break the cycle with a third action",
                        "Try navigating elsewhere first",
                    ],
                }
        
        return {"is_repetitive": False}
    
    def _detect_stuck_state(self, observation: Observation) -> Dict[str, Any]:
        """Detect if the agent is stuck.
        
        A stuck state is when:
        - Multiple consecutive actions don't change the page
        - Error messages appear repeatedly
        - No progress is being made
        
        Returns:
            Dictionary with stuck detection result
        """
        if len(self.action_history) < 3:
            return {"is_stuck": False}
        
        # Simple heuristic: if last N actions all failed or are NONE
        recent = self.action_history[-3:]
        none_count = sum(1 for r in recent if r["action_type"] in ["NONE", "UNKNOWN"])
        
        if none_count >= 2:
            return {
                "is_stuck": True,
                "reason": f"Multiple failed actions ({none_count} NONE actions in last 3 steps)",
            }
        
        return {"is_stuck": False}
    
    def _generate_feedback(
        self,
        context_summary: str,
        reflection_result: Dict[str, Any],
        global_check_result: Dict[str, Any],
    ) -> MonitorFeedback:
        """Generate feedback based on all check results.
        
        Decision logic:
        - ALLOW: No issues detected
        - WARN: Minor issues, inject warning
        - ROLLBACK: Severe issues, force different approach
        - STOP: Task should terminate
        
        Returns:
            MonitorFeedback with decision and message
        """
        # Check for task completion
        checklist = reflection_result.get("checklist", {})
        if checklist.get("task_completed", False):
            return MonitorFeedback(
                decision=MonitorDecision.STOP,
                message="Task completed successfully",
                severity=0,
                detected_issues=[],
                suggested_actions=[],
                context_summary=context_summary,
                inject_to_prompt=False,
            )
        
        # If no issues detected
        if not global_check_result["has_issues"]:
            return MonitorFeedback(
                decision=MonitorDecision.ALLOW,
                message="Action executed normally",
                severity=0,
                detected_issues=[],
                suggested_actions=[],
                context_summary=context_summary,
                inject_to_prompt=False,
            )
        
        # Determine decision based on severity
        severity = global_check_result["severity"]
        issues = global_check_result["issues"]
        suggestions = global_check_result["suggested_actions"]
        
        if severity >= 8:
            # Severe issue - force rollback
            decision = MonitorDecision.ROLLBACK
            message = "Critical issue detected. You MUST try a different approach."
        elif severity >= 5:
            # Moderate issue - warn but allow to continue
            decision = MonitorDecision.WARN
            message = "Issue detected. Consider trying a different approach."
        else:
            # Minor issue - allow with note
            decision = MonitorDecision.ALLOW
            message = "Minor issue noted."
        
        return MonitorFeedback(
            decision=decision,
            message=message,
            severity=severity,
            detected_issues=issues,
            suggested_actions=suggestions,
            context_summary=context_summary,
            inject_to_prompt=(decision in [MonitorDecision.WARN, MonitorDecision.ROLLBACK]),
        )
    
    def get_latest_feedback(self) -> Optional[MonitorFeedback]:
        """Get the most recent feedback."""
        return self.feedback_history[-1] if self.feedback_history else None
    
    def get_prompt_injection(self) -> str:
        """Get the prompt injection string for the next Actor call.
        
        This should be injected into the Actor's prompt.
        """
        feedback = self.get_latest_feedback()
        if feedback:
            return feedback.to_prompt_injection()
        return ""
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        return {
            "total_steps": self.step_count,
            "consecutive_errors": self.consecutive_errors,
            "pattern_issue_count": self.pattern_issue_count,
            "stuck_detection_count": self.stuck_detection_count,
            "deviation_warnings": self.deviation_warnings,
            "feedback_count": len(self.feedback_history),
            "warn_count": sum(1 for f in self.feedback_history if f.decision == MonitorDecision.WARN),
            "rollback_count": sum(1 for f in self.feedback_history if f.decision == MonitorDecision.ROLLBACK),
        }
    
    def reset(self) -> None:
        """Reset the monitor for a new task."""
        self.initialize(self.global_goal)
