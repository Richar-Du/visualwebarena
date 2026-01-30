"""Trajectory Logger for generating HTML visualization of agent execution.

This module provides functionality to:
1. Record each step's observation and SOM-annotated screenshots
2. Log detailed LLM inputs and outputs for all agents (Actor/Context/Reflector/Monitor)
3. Generate a comprehensive HTML report for trajectory visualization
"""

import os
import json
import base64
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from io import BytesIO

import numpy as np
from PIL import Image


@dataclass
class LLMCall:
    """Record of a single LLM call."""
    agent_name: str           # actor, context, reflector, monitor
    step_number: int
    timestamp: str
    prompt: str               # Full prompt sent to LLM
    response: str             # Full response from LLM
    parsed_result: Dict[str, Any] = field(default_factory=dict)  # Parsed/structured result


@dataclass
class StepRecord:
    """Record of a single execution step."""
    step_number: int
    timestamp: str
    
    # Screenshots (base64 encoded)
    observation_screenshot: Optional[str] = None      # Raw screenshot
    som_screenshot: Optional[str] = None              # SOM-annotated screenshot
    
    # Observation text (accessibility tree)
    observation_text: str = ""
    
    # Current URL
    current_url: str = ""
    
    # Action taken
    action: Dict[str, Any] = field(default_factory=dict)
    action_str: str = ""
    
    # LLM calls in this step
    llm_calls: List[LLMCall] = field(default_factory=list)
    
    # Monitor feedback (if any)
    monitor_decision: str = ""
    monitor_feedback: str = ""


class TrajectoryLogger:
    """Logger for recording and visualizing agent execution trajectories."""
    
    def __init__(self, output_dir: str, task_name: str = "task"):
        """Initialize the trajectory logger.
        
        Args:
            output_dir: Directory to save the HTML report
            task_name: Name of the task for the report title
        """
        self.output_dir = output_dir
        self.task_name = task_name
        self.start_time = datetime.now()
        
        # Execution records
        self.user_goal: str = ""
        self.steps: List[StepRecord] = []
        self.current_step: Optional[StepRecord] = None
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
    
    def set_user_goal(self, goal: str) -> None:
        """Set the user's goal/task description."""
        self.user_goal = goal
    
    def start_step(self, step_number: int) -> None:
        """Start recording a new step."""
        self.current_step = StepRecord(
            step_number=step_number,
            timestamp=datetime.now().isoformat()
        )
    
    def log_observation(
        self,
        observation_text: str,
        observation_image: Optional[Any] = None,
        som_image: Optional[Any] = None,
        current_url: str = ""
    ) -> None:
        """Log the observation for the current step.
        
        Args:
            observation_text: The accessibility tree or page text
            observation_image: Raw screenshot (numpy array or PIL Image)
            som_image: SOM-annotated screenshot (numpy array or PIL Image)
            current_url: Current page URL
        """
        if self.current_step is None:
            return
        
        self.current_step.observation_text = observation_text[:10000]  # Limit size
        self.current_step.current_url = current_url
        
        # Convert and encode screenshots
        if observation_image is not None:
            self.current_step.observation_screenshot = self._image_to_base64(observation_image)
        
        if som_image is not None:
            self.current_step.som_screenshot = self._image_to_base64(som_image)
    
    def log_llm_call(
        self,
        agent_name: str,
        prompt: str,
        response: str,
        parsed_result: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log an LLM call.
        
        Args:
            agent_name: Name of the agent (actor, context, reflector, monitor)
            prompt: Full prompt sent to LLM
            response: Full response from LLM
            parsed_result: Optional parsed/structured result
        """
        if self.current_step is None:
            return
        
        llm_call = LLMCall(
            agent_name=agent_name,
            step_number=self.current_step.step_number,
            timestamp=datetime.now().isoformat(),
            prompt=prompt,
            response=response,
            parsed_result=parsed_result or {}
        )
        self.current_step.llm_calls.append(llm_call)
    
    def log_action(self, action: Dict[str, Any], action_str: str = "") -> None:
        """Log the action taken in this step."""
        if self.current_step is None:
            return
        
        self.current_step.action = action
        self.current_step.action_str = action_str
    
    def log_monitor_feedback(self, decision: str, feedback: str) -> None:
        """Log monitor feedback for this step."""
        if self.current_step is None:
            return
        
        self.current_step.monitor_decision = decision
        self.current_step.monitor_feedback = feedback
    
    def end_step(self) -> None:
        """End the current step and save it."""
        if self.current_step is not None:
            self.steps.append(self.current_step)
            self.current_step = None
    
    def generate_html_report(self, filename: str = "trajectory_report.html") -> str:
        """Generate the HTML trajectory report.
        
        Returns:
            Path to the generated HTML file
        """
        html_path = os.path.join(self.output_dir, filename)
        
        html_content = self._generate_html()
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"📄 Trajectory report saved to: {html_path}")
        return html_path
    
    def _image_to_base64(self, image: Any) -> str:
        """Convert image to base64 string."""
        try:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            if isinstance(image, Image.Image):
                # Resize if too large
                max_size = (1280, 1280)
                image.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                buffer = BytesIO()
                image.save(buffer, format='PNG')
                return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            print(f"Warning: Failed to encode image: {e}")
        
        return ""
    
    def _generate_html(self) -> str:
        """Generate the complete HTML content."""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        steps_html = "\n".join([self._generate_step_html(step) for step in self.steps])
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agent Trajectory Report - {self._escape_html(self.task_name)}</title>
    <style>
        :root {{
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-tertiary: #21262d;
            --text-primary: #c9d1d9;
            --text-secondary: #8b949e;
            --accent-blue: #58a6ff;
            --accent-green: #3fb950;
            --accent-yellow: #d29922;
            --accent-red: #f85149;
            --accent-purple: #a371f7;
            --border-color: #30363d;
        }}
        
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        /* Header */
        .header {{
            background: linear-gradient(135deg, var(--bg-secondary), var(--bg-tertiary));
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 30px;
            border: 1px solid var(--border-color);
        }}
        
        .header h1 {{
            font-size: 28px;
            margin-bottom: 15px;
            color: var(--accent-blue);
        }}
        
        .header .meta {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            color: var(--text-secondary);
            font-size: 14px;
        }}
        
        .header .meta span {{
            background: var(--bg-tertiary);
            padding: 6px 12px;
            border-radius: 6px;
        }}
        
        .goal-box {{
            background: var(--bg-tertiary);
            padding: 15px 20px;
            border-radius: 8px;
            margin-top: 20px;
            border-left: 4px solid var(--accent-blue);
        }}
        
        .goal-box h3 {{
            color: var(--accent-blue);
            font-size: 14px;
            margin-bottom: 8px;
        }}
        
        /* Step Card */
        .step-card {{
            background: var(--bg-secondary);
            border-radius: 12px;
            margin-bottom: 25px;
            border: 1px solid var(--border-color);
            overflow: hidden;
        }}
        
        .step-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 15px 20px;
            background: var(--bg-tertiary);
            border-bottom: 1px solid var(--border-color);
            cursor: pointer;
        }}
        
        .step-header:hover {{
            background: #2d333b;
        }}
        
        .step-number {{
            background: var(--accent-blue);
            color: white;
            padding: 5px 12px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 14px;
        }}
        
        .step-action {{
            font-size: 15px;
            color: var(--accent-green);
            font-family: monospace;
        }}
        
        .step-url {{
            font-size: 13px;
            color: var(--text-secondary);
            max-width: 400px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
        
        .step-content {{
            padding: 20px;
        }}
        
        /* Screenshots */
        .screenshots {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }}
        
        .screenshot-box {{
            background: var(--bg-tertiary);
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid var(--border-color);
        }}
        
        .screenshot-box h4 {{
            padding: 10px 15px;
            background: var(--bg-primary);
            font-size: 13px;
            color: var(--text-secondary);
        }}
        
        .screenshot-box img {{
            width: 100%;
            height: auto;
            display: block;
        }}
        
        /* LLM Calls */
        .llm-calls {{
            margin-top: 20px;
        }}
        
        .llm-call {{
            background: var(--bg-tertiary);
            border-radius: 8px;
            margin-bottom: 15px;
            border: 1px solid var(--border-color);
            overflow: hidden;
        }}
        
        .llm-call-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 12px 15px;
            background: var(--bg-primary);
            cursor: pointer;
        }}
        
        .llm-call-header:hover {{
            background: #1c2128;
        }}
        
        .agent-badge {{
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
            text-transform: uppercase;
        }}
        
        .agent-badge.actor {{ background: var(--accent-green); color: #000; }}
        .agent-badge.context {{ background: var(--accent-blue); color: #000; }}
        .agent-badge.reflector {{ background: var(--accent-yellow); color: #000; }}
        .agent-badge.monitor {{ background: var(--accent-purple); color: #fff; }}
        
        .llm-content {{
            padding: 15px;
            display: none;
        }}
        
        .llm-content.expanded {{
            display: block;
        }}
        
        .prompt-section, .response-section {{
            margin-bottom: 15px;
        }}
        
        .prompt-section h5, .response-section h5 {{
            color: var(--text-secondary);
            font-size: 12px;
            margin-bottom: 8px;
            text-transform: uppercase;
        }}
        
        .code-block {{
            background: var(--bg-primary);
            padding: 15px;
            border-radius: 6px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 13px;
            white-space: pre-wrap;
            word-break: break-word;
            max-height: 400px;
            overflow-y: auto;
            border: 1px solid var(--border-color);
        }}
        
        /* Monitor Feedback */
        .monitor-feedback {{
            background: var(--bg-tertiary);
            border-radius: 8px;
            padding: 15px;
            margin-top: 15px;
            border-left: 4px solid var(--accent-purple);
        }}
        
        .monitor-feedback.warn {{
            border-left-color: var(--accent-yellow);
        }}
        
        .monitor-feedback.rollback {{
            border-left-color: var(--accent-red);
        }}
        
        .monitor-feedback h4 {{
            font-size: 14px;
            margin-bottom: 10px;
        }}
        
        /* Observation Text */
        .observation-text {{
            background: var(--bg-tertiary);
            border-radius: 8px;
            padding: 15px;
            margin-top: 15px;
            border: 1px solid var(--border-color);
        }}
        
        .observation-text h4 {{
            font-size: 13px;
            color: var(--text-secondary);
            margin-bottom: 10px;
        }}
        
        .observation-text pre {{
            font-family: monospace;
            font-size: 12px;
            white-space: pre-wrap;
            max-height: 300px;
            overflow-y: auto;
        }}
        
        /* Toggle button */
        .toggle-btn {{
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 12px;
            cursor: pointer;
        }}
        
        .toggle-btn:hover {{
            background: var(--bg-primary);
        }}
        
        /* Summary footer */
        .summary {{
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid var(--border-color);
            margin-top: 30px;
        }}
        
        .summary h3 {{
            color: var(--accent-blue);
            margin-bottom: 15px;
        }}
        
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }}
        
        .summary-item {{
            background: var(--bg-tertiary);
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        
        .summary-item .value {{
            font-size: 24px;
            font-weight: bold;
            color: var(--accent-green);
        }}
        
        .summary-item .label {{
            font-size: 12px;
            color: var(--text-secondary);
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>🤖 Agent Trajectory Report</h1>
            <div class="meta">
                <span>📅 {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}</span>
                <span>⏱️ Duration: {duration:.1f}s</span>
                <span>📊 Total Steps: {len(self.steps)}</span>
            </div>
            <div class="goal-box">
                <h3>🎯 Task Goal</h3>
                <p>{self._escape_html(self.user_goal)}</p>
            </div>
            <div style="margin-top:15px;display:flex;gap:10px;">
                <button class="toggle-btn" onclick="expandAllLLMCalls()" style="padding:8px 16px;">📖 Expand All LLM Calls</button>
                <button class="toggle-btn" onclick="collapseAllLLMCalls()" style="padding:8px 16px;">📕 Collapse All LLM Calls</button>
            </div>
        </div>
        
        <!-- Steps -->
        {steps_html}
        
        <!-- Summary -->
        <div class="summary">
            <h3>📈 Execution Summary</h3>
            <div class="summary-grid">
                <div class="summary-item">
                    <div class="value">{len(self.steps)}</div>
                    <div class="label">Total Steps</div>
                </div>
                <div class="summary-item">
                    <div class="value">{sum(len(s.llm_calls) for s in self.steps)}</div>
                    <div class="label">LLM Calls</div>
                </div>
                <div class="summary-item">
                    <div class="value">{duration:.1f}s</div>
                    <div class="label">Duration</div>
                </div>
                <div class="summary-item">
                    <div class="value">{sum(1 for s in self.steps if s.monitor_decision)}</div>
                    <div class="label">Monitor Alerts</div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Toggle LLM call content
        document.querySelectorAll('.llm-call-header').forEach(header => {{
            header.addEventListener('click', () => {{
                const content = header.nextElementSibling;
                content.classList.toggle('expanded');
                const btn = header.querySelector('.toggle-btn');
                btn.textContent = content.classList.contains('expanded') ? '▼ Collapse' : '▶ Expand';
            }});
        }});
        
        // Toggle step content (observation text)
        document.querySelectorAll('.obs-toggle').forEach(btn => {{
            btn.addEventListener('click', (e) => {{
                e.stopPropagation();
                const obsText = btn.closest('.step-card').querySelector('.observation-text');
                obsText.style.display = obsText.style.display === 'none' ? 'block' : 'none';
                btn.textContent = obsText.style.display === 'none' ? 'Show Observation' : 'Hide Observation';
            }});
        }});
        
        // Auto-expand Actor LLM calls by default for easier viewing
        document.querySelectorAll('.agent-badge.actor').forEach(badge => {{
            const header = badge.closest('.llm-call-header');
            if (header) {{
                const content = header.nextElementSibling;
                content.classList.add('expanded');
                const btn = header.querySelector('.toggle-btn');
                if (btn) btn.textContent = '▼ Collapse';
            }}
        }});
        
        // Expand/Collapse All functionality
        function expandAllLLMCalls() {{
            document.querySelectorAll('.llm-content').forEach(content => {{
                content.classList.add('expanded');
            }});
            document.querySelectorAll('.llm-call-header .toggle-btn').forEach(btn => {{
                btn.textContent = '▼ Collapse';
            }});
        }}
        
        function collapseAllLLMCalls() {{
            document.querySelectorAll('.llm-content').forEach(content => {{
                content.classList.remove('expanded');
            }});
            document.querySelectorAll('.llm-call-header .toggle-btn').forEach(btn => {{
                btn.textContent = '▶ Expand';
            }});
        }}
    </script>
</body>
</html>"""
        
        return html
    
    def _generate_step_html(self, step: StepRecord) -> str:
        """Generate HTML for a single step."""
        # Screenshots
        screenshots_html = ""
        if step.observation_screenshot or step.som_screenshot:
            obs_img = f'<img src="data:image/png;base64,{step.observation_screenshot}" alt="Observation">' if step.observation_screenshot else '<p style="padding:20px;color:var(--text-secondary);">No screenshot</p>'
            som_img = f'<img src="data:image/png;base64,{step.som_screenshot}" alt="SOM">' if step.som_screenshot else '<p style="padding:20px;color:var(--text-secondary);">No screenshot</p>'
            
            screenshots_html = f"""
            <div class="screenshots">
                <div class="screenshot-box">
                    <h4>📷 Raw Screenshot</h4>
                    {obs_img}
                </div>
                <div class="screenshot-box">
                    <h4>🔢 SOM Annotated</h4>
                    {som_img}
                </div>
            </div>"""
        
        # LLM calls
        llm_calls_html = ""
        for call in step.llm_calls:
            agent_class = call.agent_name.lower()
            parsed_json = json.dumps(call.parsed_result, indent=2, ensure_ascii=False) if call.parsed_result else ""
            
            llm_calls_html += f"""
            <div class="llm-call">
                <div class="llm-call-header">
                    <span class="agent-badge {agent_class}">{call.agent_name}</span>
                    <span style="color:var(--text-secondary);font-size:12px;">{call.timestamp}</span>
                    <button class="toggle-btn">▶ Expand</button>
                </div>
                <div class="llm-content">
                    <div class="prompt-section">
                        <h5>📤 Prompt</h5>
                        <div class="code-block">{self._escape_html(call.prompt)}</div>
                    </div>
                    <div class="response-section">
                        <h5>📥 Response</h5>
                        <div class="code-block">{self._escape_html(call.response)}</div>
                    </div>
                    {f'<div class="response-section"><h5>📋 Parsed Result</h5><div class="code-block">{self._escape_html(parsed_json)}</div></div>' if parsed_json else ''}
                </div>
            </div>"""
        
        # Monitor feedback  
        monitor_html = ""
        if step.monitor_decision:
            decision_class = "warn" if step.monitor_decision == "warn" else "rollback" if step.monitor_decision == "rollback" else ""
            monitor_html = f"""
            <div class="monitor-feedback {decision_class}">
                <h4>🔍 Monitor Decision: {step.monitor_decision.upper()}</h4>
                <pre>{self._escape_html(step.monitor_feedback)}</pre>
            </div>"""
        
        # Observation text (collapsed by default)
        obs_text_html = f"""
        <div class="observation-text" style="display:none;">
            <h4>📝 Accessibility Tree</h4>
            <pre>{self._escape_html(step.observation_text[:5000])}{'...(truncated)' if len(step.observation_text) > 5000 else ''}</pre>
        </div>""" if step.observation_text else ""
        
        return f"""
        <div class="step-card">
            <div class="step-header">
                <span class="step-number">Step {step.step_number}</span>
                <span class="step-action">{self._escape_html(step.action_str or 'N/A')}</span>
                <span class="step-url">{self._escape_html(step.current_url[:80] + '...' if len(step.current_url) > 80 else step.current_url)}</span>
                <button class="obs-toggle toggle-btn">Show Observation</button>
            </div>
            <div class="step-content">
                {screenshots_html}
                
                <div class="llm-calls">
                    <h4 style="margin-bottom:15px;color:var(--text-secondary);">💬 LLM Calls ({len(step.llm_calls)})</h4>
                    {llm_calls_html}
                </div>
                
                {monitor_html}
                {obs_text_html}
            </div>
        </div>"""
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        if not text:
            return ""
        return (text
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
                .replace("'", "&#39;"))
