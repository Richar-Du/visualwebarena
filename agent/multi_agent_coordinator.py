"""Multi-Agent Coordinator for managing collaborative agent execution.

This module coordinates Context, Planner, Actor, and Reflector agents
with explicit subtask state management and task completion handling.
"""

from typing import Any, Dict, List, Optional, Union
import json
import re
import os
import traceback
from datetime import datetime

from PIL import Image

from browser_env import Action, ActionTypes, Trajectory
from browser_env.helper_functions import get_action_description
from llms import lm_config

# Try importing specific Observation types
try:
    from browser_env.utils import Observation as BrowserObservation
    ObservationType = Union[BrowserObservation, dict]
except ImportError:
    # Fallback types for when browser_env is not available
    BrowserObservation = None
    ObservationType = Union[dict, Dict[str, Any]]

# Type aliases for better type hints
ObservationTypeAlias = ObservationType

import copy

from .context_agent import ContextAgent
from .planner_agent import PlannerAgent
from .actor_agent import ActorAgent
from .reflector_agent import ReflectorAgent
from .coordinator.workflow_manager import WorkflowManager
from .coordinator.communication_hub import CommunicationHub


class MultiAgentCoordinator:
    """Coordinates multiple agents for collaborative task execution.

    Manages the interaction between Context, Planner, Actor, and Reflector agents
    to achieve complex web automation tasks through coordinated execution.
    
    Key features:
    - Explicit subtask state tracking via SubtaskManager
    - Subtask completion detection and progression
    - Subtask revision when needed
    - Task completion and stop action handling
    """

    def __init__(self, lm_config: lm_config.LMConfig,
                 existing_prompt_agent,
                 browser_env=None,
                 result_dir: str = "results",
                 memory_config: Dict[str, Any]= {},
                 webjudge_result_root: Optional[str] = None) -> None:
        self.lm_config = lm_config

        # Get action set tag from existing agent or use default
        action_set_tag = getattr(existing_prompt_agent, 'action_set_tag', 'som')

        self.enable_memory = memory_config.get("enable_memory", False)
        self.enable_memory_store = memory_config.get("enable_memory_store", False)

        # Initialize individual agents with memory enabled if specified
        self.context_agent = ContextAgent(lm_config, memory_config)
        self.planner_agent = PlannerAgent(lm_config)
        self.actor_agent = ActorAgent(
            action_set_tag=action_set_tag,
            lm_config=lm_config,
            prompt_constructor=getattr(existing_prompt_agent, 'prompt_constructor', None),
            captioning_fn=getattr(existing_prompt_agent, 'captioning_fn', None)
        )
        self.reflector_agent = ReflectorAgent(lm_config)

        # Initialize coordination components
        self.workflow_manager = WorkflowManager()
        self.communication_hub = CommunicationHub()

        # Browser environment for action execution
        self.browser_env = browser_env

        # Result directory and logging setup
        self.result_dir = result_dir
        # WebJudge 结果根目录（与现有结果分开保存）
        self.webjudge_result_root = webjudge_result_root or os.path.join(self.result_dir, "webjudge_results")  # WebJudge 输出根目录
        self.log_file_path = os.path.join(result_dir, "agent_responses.log")
        self.observation_log_path = os.path.join(result_dir, "observations.json")
        self.images_dir = os.path.join(result_dir, "images")
        self.images_som_dir = os.path.join(result_dir, "images_som")
        self._setup_logging()

        # Execution state
        self.trajectory: Trajectory = []
        self.intentions: List[str] = []
        self.actions: List[Action] = []
        self.reflections: List[Dict[str, Any]] = []
        self.webjudge_action_history: List[str] = []  # 存动作文本历史
        self.webjudge_thoughts: List[str] = []  # 存每步意图/思考
        self.task_metadata: Dict[str, Any] = {}  # 任务元信息缓存
        self.task_output_dir: Optional[str] = None  # 当前任务输出目录
        self.trajectory_dir: Optional[str] = None  # 截图存放目录
        self.result_json_path: Optional[str] = None  # result.json 路径
        self.current_task_id: str = ""  # 当前任务 ID

        # Meta data for action history tracking (required by DirectPromptConstructor)
        # Initialize with "None" as the first action, matching run.py implementation
        self.meta_data: Dict[str, Any] = {"action_history": ["None"]}

        # Task configuration
        self.user_goal: str = ""
        self.max_steps: int = 30
        self.current_observation: Optional["ObservationTypeAlias"] = None
        
        # Stop action data storage (for future evaluation use)
        self.stop_action_data: Optional[Dict[str, Any]] = None

    def _setup_logging(self) -> None:
        """Setup logging for agent responses."""
        try:
            # Ensure result directory exists
            os.makedirs(self.result_dir, exist_ok=True)

            # Create images directory
            os.makedirs(self.images_dir, exist_ok=True)

            # Create images_som directory for SOM-annotated screenshots
            os.makedirs(self.images_som_dir, exist_ok=True)

            # Create or clear the log file
            with open(self.log_file_path, 'w', encoding='utf-8') as f:
                f.write(f"Multi-Agent Execution Log - Started at {datetime.now().isoformat()}\n")
                f.write("=" * 80 + "\n\n")

            # Initialize observation log file with empty list
            with open(self.observation_log_path, 'w', encoding='utf-8') as f:
                json.dump([], f, indent=2)

        except Exception as e:
            print(f"Warning: Failed to setup logging: {e}")

    def log_agent_response(self, agent_name: str, step_number: int, response_data: Dict[str, Any]) -> None:
        """Log agent response summary to file."""
        try:
            timestamp = datetime.now().isoformat()

            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                f.write(f"[{timestamp}] Step {step_number} - {agent_name.upper()} Agent Response\n")
                f.write("-" * 60 + "\n")
                f.write(json.dumps(response_data, indent=2, ensure_ascii=False))
                f.write("\n\n")
        except Exception as e:
            print(f"Warning: Failed to log {agent_name} response: {e}")

    def log_observation(self, step_number: int, observation: Dict[str, Any]) -> None:
        """Log observation text and save screenshot image."""
        try:
            # Load existing observations
            with open(self.observation_log_path, 'r', encoding='utf-8') as f:
                observations = json.load(f)

            # Add new observation
            obs_entry = {
                "step": step_number,
                "timestamp": datetime.now().isoformat(),
                "text": observation.get("text", ""),
                "has_image": observation.get("image") is not None
            }
            observations.append(obs_entry)

            # Save updated observations
            with open(self.observation_log_path, 'w', encoding='utf-8') as f:
                json.dump(observations, f, indent=2, ensure_ascii=False)

            # Save screenshot image if available (prefer image_raw for cleaner logs)
            image_to_save = observation.get("image_raw")
            if image_to_save is None:
                image_to_save = observation.get("image")
            if image_to_save is not None:
                image_path = os.path.join(self.images_dir, f"step_{step_number:03d}.png")
                # Convert numpy array to PIL Image and save
                from PIL import Image
                import numpy as np

                if isinstance(image_to_save, np.ndarray):
                    img = Image.fromarray(image_to_save)
                    img.save(image_path)

            # Save SOM-annotated screenshot (image with bounding boxes and IDs)
            image_som = observation.get("image")
            if image_som is not None:
                image_som_path = os.path.join(self.images_som_dir, f"step_{step_number:03d}.png")
                from PIL import Image
                import numpy as np

                if isinstance(image_som, np.ndarray):
                    img_som = Image.fromarray(image_som)
                    img_som.save(image_som_path)

        except Exception as e:
            print(f"Warning: Failed to log observation for step {step_number}: {e}")


    def _prepare_webjudge_output(self, task_metadata: Optional[Dict[str, Any]], webjudge_root: Optional[str]) -> None:
        """Create per-task directories for WebJudge-compatible outputs."""
        self.task_metadata = task_metadata or {}  # 记录任务元信息
        self.current_task_id = self.task_metadata.get("task_id") or datetime.now().strftime("%Y%m%d_%H%M%S")  # 若无 task_id 用时间戳代替
        base_root = webjudge_root or self.webjudge_result_root  # 选择输出根目录
        self.task_output_dir = os.path.join(base_root, self.current_task_id)  # 当前任务输出目录
        self.trajectory_dir = os.path.join(self.task_output_dir, "trajectory")  # 截图子目录
        os.makedirs(self.trajectory_dir, exist_ok=True)
        self.result_json_path = os.path.join(self.task_output_dir, "result.json")  # 结果文件路径
        self.webjudge_action_history = []  # 重置动作记录
        self.webjudge_thoughts = []  # 重置思考记录

    def _capture_and_save_screenshot(self, step_number: int) -> Optional[str]:
        """Capture full-page screenshot directly from Playwright page (no SOM)."""
        if self.browser_env is None or not hasattr(self.browser_env, "page"):
            return None  # 无浏览器实例时跳过
        if not self.trajectory_dir:
            return None  # 未初始化目录时跳过
        screenshot_path = os.path.join(self.trajectory_dir, f"step_{step_number:03d}.png")
        try:
            self.browser_env.page.screenshot(path=screenshot_path, full_page=True)  # 直接截全页
            return screenshot_path
        except Exception as e:
            print(f"Warning: Failed to capture screenshot for step {step_number}: {e}")
            return None

    def _format_action_for_webjudge(self, executed_action: Action, info: Optional[Dict[str, Any]]) -> str:
        """Format action string for WebJudge action_history.
        输出格式：action [id] [tag] [text] -> action_type
        在 SOM 模式下确保 id/tag/text 与模型在观察中看到的一致：
        - 优先从上一状态的 text 观察解析 [id] [tag] [text]
        - 若不可用，再尝试从 raw_prediction 中解析
        - 非 SOM 模式或仍失败时再回退到可访问性树角色或启发式
        """
        action_type = executed_action.get("action_type", "UNKNOWN")
        raw_pred = executed_action.get("raw_prediction") or ""
        action_set_tag = getattr(self.actor_agent, "action_set_tag", "id_accessibility_tree")

        # 优先使用解析器得到的 element_id
        elem_id = str(executed_action.get("element_id") or "")
        tag = ""
        text = ""

        # 1) 从上一状态的 text 观察中解析 "[id] [tag] [text]"（优先使用调用方提供的 prev_text_observation）
        print(f"action_set_tag={action_set_tag}")
        if action_set_tag == "som":
            try:
                prev_text = ""
                if info and isinstance(info, dict):
                    prev_text = info.get("prev_text_observation") or ""
                if not prev_text:
                    # 回退：通过 trajectory 寻找上一个状态的 text 观察
                    found_first_dict = False
                    for item in reversed(self.trajectory):
                        if isinstance(item, dict):
                            if not found_first_dict:
                                found_first_dict = True  # 跳过最新 new_state_info
                                continue
                            prev_text = item.get("observation", {}).get("text", "")
                            break
                if prev_text:
                    id_tag_text = {}
                    for line in str(prev_text).splitlines():
                        m3 = re.search(r"\[\s*(\d+)\s*\]\s*\[\s*([^\]]+)\s*\]\s*\[\s*(.*?)\s*\]", line)
                        if m3:
                            _id = m3.group(1)
                            id_tag_text[_id] = (m3.group(2).strip(), m3.group(3).strip())
                    if elem_id and elem_id in id_tag_text:
                        tag, text = id_tag_text[elem_id]
                    elif not elem_id and id_tag_text:
                        # 如果未提供 element_id，取第一行作为默认
                        first_id = next(iter(id_tag_text.keys()))
                        elem_id = first_id
                        tag, text = id_tag_text[first_id]
            except Exception:
                pass

        # 2) 若未解析到，则尝试从 raw_prediction 中解析 "[id] [tag] [text]" 或 "[id] [tag]"
        if (not tag or not text):
            try:
                m3 = re.search(r"\[\s*(\d+)\s*\]\s*\[\s*([^\]]+)\s*\]\s*\[\s*(.*?)\s*\]", raw_pred)
                m2 = re.search(r"\[\s*(\d+)\s*\]\s*\[\s*([^\]]+)\s*\]", raw_pred) if not m3 else None
                if m3:
                    if not elem_id:
                        elem_id = m3.group(1)
                    tag = tag or m3.group(2).strip()
                    text = text or m3.group(3).strip()
                elif m2:
                    if not elem_id:
                        elem_id = m2.group(1)
                    tag = tag or m2.group(2).strip()
            except Exception:
                pass

        # 3) 非 SOM 或仍失败：从可访问性树元数据提取角色(tag)和名称(text)
        if (not tag or not text) and info and isinstance(info, dict):
            observation_metadata = info.get("observation_metadata") or {}
            text_meta = observation_metadata.get("text") or {}
            obs_nodes_info = text_meta.get("obs_nodes_info") or {}
            node_info = obs_nodes_info.get(elem_id)
            if node_info:
                node_text = node_info.get("text", "")
                # node_text 形如："[1234] button 'Add to Cart' ..."
                try:
                    after = node_text.split("]", 1)[1].strip()
                    role = after.split(" ", 1)[0]
                    name_match = re.search(r"'([^']*)'", after)
                    if role and not tag:
                        tag = role
                    if name_match and not text:
                        text = name_match.group(1)
                except Exception:
                    pass

        # 4) 最后回退：基于 action_type 的启发式
        if not tag:
            tag_map = {
                ActionTypes.TYPE: "textbox",
                ActionTypes.CLICK: "clickable",
                ActionTypes.HOVER: "element",
                ActionTypes.GOTO_URL: "url",
            }
            tag = tag_map.get(action_type, "element")

        if not elem_id:
            elem_id = "unknown"
        if text is None:
            text = ""

        return f"[{elem_id}] [{tag}] [{text}] -> {action_type}"

    def _format_action_for_webjudge_html(self, executed_action: Action, info: Optional[Dict[str, Any]]) -> str:
        """备用：尽量还原原始 HTML 的动作描述（如 <a href="...">文本</a> -> CLICK）。
        如需启用，将调用处的 _format_action_for_webjudge 替换为本函数即可。
        """
        action_type = executed_action.get("action_type", "UNKNOWN")
        observation_metadata = None
        if info and isinstance(info, dict):
            observation_metadata = info.get("observation_metadata")

        node = None
        if observation_metadata and "text" in observation_metadata:
            text_meta = observation_metadata["text"]
            node = text_meta.get("obs_nodes_info", {}).get(executed_action.get("element_id"))

        if node:
            tag = node.get("tag") or node.get("nodeName") or "div"
            attrs = []
            for k in ["id", "class", "href", "url", "name", "aria-label", "placeholder", "type", "value"]:
                v = node.get(k) or node.get(k.replace("-", "_"))
                if v:
                    attrs.append(f'{k}="{v}"')
            role = node.get("role")
            if role:
                attrs.append(f'role="{role}"')
            attr_str = (" " + " ".join(attrs)) if attrs else ""
            inner = node.get("text") or node.get("alt") or node.get("name") or node.get("value") or tag
            return f"<{tag}{attr_str}>{inner}</{tag}> -> {action_type}"

        # SOM 或缺元数据时退回简单描述
        elem_id = executed_action.get("element_id", "unknown")
        return f"<element id=\"{elem_id}\"> -> {action_type}"

    def _save_webjudge_result(self, final_result_response: str) -> None:
        """Persist WebJudge style result.json."""
        if not self.result_json_path:
            return
        payload = {
            "task_id": self.task_metadata.get("task_id", self.current_task_id),
            "task": self.task_metadata.get("task") or self.user_goal,
            "final_result_response": final_result_response,  # 最终回答
            "action_history": self.webjudge_action_history,  # 动作列表
            "thoughts": self.webjudge_thoughts,  # 思考列表
        }
        for key in ["website", "reference_length", "level", "confirmed_task"]:
            if key in self.task_metadata:
                payload[key] = self.task_metadata[key]  # 可选元数据透传

        os.makedirs(os.path.dirname(self.result_json_path), exist_ok=True)
        with open(self.result_json_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


    def execute_task(
        self,
        user_goal: str,
        start_observation: Optional["ObservationTypeAlias"] = None,
        max_steps: int = 30,
        images: Optional[List[Image.Image]] = None,
        task_metadata: Optional[Dict[str, Any]] = None,
        webjudge_root: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute a complete task using coordinated multi-agent approach.

        Args:
            user_goal: The user's goal/task description
            start_observation: Initial page observation
            max_steps: Maximum number of execution steps
            images: Optional input images for the task

        Returns:
            Dictionary containing complete execution results
        """
        # Reset all state before starting a new task
        # This ensures clean separation between consecutive task executions
        self.reset()
        
        # Initialize task
        self.user_goal = user_goal
        self.max_steps = max_steps
        self.current_observation = start_observation

        # Prepare WebJudge output directories
        self._prepare_webjudge_output(task_metadata, webjudge_root)  # 为本任务创建输出目录

        # Reset meta_data for new task execution (required by DirectPromptConstructor)
        # Initialize with "None" as the first action, matching run.py implementation
        self.meta_data = {"action_history": ["None"]}

        # Initialize workflow and monitoring
        workflow_init = self.workflow_manager.initialize_workflow(max_steps)

        # Register agents with communication hub
        self._register_agents()

        # Update shared context
        self.communication_hub.update_shared_context("user_goal", user_goal)
        self.communication_hub.update_shared_context("max_steps", max_steps)

        if self.enable_memory:
            # Initialize memory system for this task
            print("🧠 Initializing memory system...")
            memory_initialization = self.context_agent.initialize_task_memory(user_goal)
            if memory_initialization.get("relevant_memories"):
                print(f"📚 Found {len(memory_initialization['relevant_memories'])} relevant memories")
                if memory_initialization.get("memory_content"):
                    print(f"📝 Memory content: {memory_initialization['memory_content'][:100]}...")
            else:
                print("📚 No relevant memories found for this task")

        # Initialize trajectory with initial state
        # This is required because DirectPromptConstructor expects trajectory[-1] to exist
        if not self.trajectory:
            # Handle different formats of start_observation
            if start_observation is None:
                # No observation provided, create empty state
                initial_observation = {"text": "", "image": None}
                initial_info = {"page": type('SimplePage', (), {'url': ''})(), "observation_metadata": {}}
            elif isinstance(start_observation, dict) and "observation" in start_observation and "info" in start_observation:
                # start_observation is already in StateInfo format: {"observation": obs, "info": info}
                initial_observation = start_observation["observation"]
                # Deep copy observation_metadata to avoid reference sharing issues
                source_info = start_observation["info"]
                initial_info = {
                    "page": source_info.get("page"),
                    "fail_error": source_info.get("fail_error", ""),
                    "observation_metadata": copy.deepcopy(source_info.get("observation_metadata", {}))
                }
                # Ensure observation has both text and image fields
                if isinstance(initial_observation, dict):
                    if "text" not in initial_observation:
                        initial_observation["text"] = ""
                    if "image" not in initial_observation:
                        initial_observation["image"] = None
            else:
                # start_observation is the observation itself
                initial_observation = start_observation
                # Ensure observation is a dict with "text" and "image" keys
                if isinstance(initial_observation, dict):
                    if "text" not in initial_observation:
                        initial_observation["text"] = ""
                    if "image" not in initial_observation:
                        initial_observation["image"] = None
                else:
                    initial_observation = {"text": str(initial_observation) if initial_observation else "", "image": None}

                # Create a simple page-like object with url attribute
                class SimplePage:
                    def __init__(self, url: str = ""):
                        self.url = url
                initial_info = {
                    "page": SimplePage(url=""),  # Will be updated when browser is initialized
                    "observation_metadata": {}
                }

            initial_state_info = {
                "observation": copy.deepcopy(initial_observation),
                "info": copy.deepcopy(initial_info)
            }
            # Ensure observation has text, image, and image_raw fields
            if isinstance(initial_observation, dict):
                if "text" not in initial_observation:
                    initial_observation["text"] = ""
                if "image" not in initial_observation:
                    initial_observation["image"] = None
                if "image_raw" not in initial_observation:
                    # Fallback: use image as image_raw if not available
                    initial_observation["image_raw"] = initial_observation.get("image")
        else:
            # start_observation is the observation itself
            initial_observation = start_observation
            # Ensure observation is a dict with "text", "image", and "image_raw" keys
            if isinstance(initial_observation, dict):
                if "text" not in initial_observation:
                    initial_observation["text"] = ""
                if "image" not in initial_observation:
                    initial_observation["image"] = None
                if "image_raw" not in initial_observation:
                    # Fallback: use image as image_raw if not available
                    initial_observation["image_raw"] = initial_observation.get("image")
            else:
                initial_observation = {"text": str(initial_observation) if initial_observation else "", "image": None, "image_raw": None}

            # Create a simple page-like object with url attribute
            class SimplePage:
                def __init__(self, url: str = ""):
                    self.url = url
            initial_info = {
                "page": SimplePage(url=""),  # Will be updated when browser is initialized
                "observation_metadata": {}
            }
            # Save initial screenshot as step_000.png (playwright full page, no SOM)
            self._capture_and_save_screenshot(0)  # 初始页截图留档
            if initial_observation.get("image") is not None:
                initial_image_path = os.path.join(self.images_dir, "step_000.png")
                # Convert numpy array to PIL Image and save
                from PIL import Image
                import numpy as np

        initial_state_info = {
            "observation": initial_observation,
            "info": initial_info
        }
        self.trajectory.append(initial_state_info)
        self.current_observation = initial_observation

        # Save initial screenshot as step_000.png (prefer image_raw for cleaner logs)
        image_to_save = initial_observation.get("image_raw")
        if image_to_save is None:
            image_to_save = initial_observation.get("image")
        if image_to_save is not None:
            initial_image_path = os.path.join(self.images_dir, "step_000.png")
            # Convert numpy array to PIL Image and save
            from PIL import Image
            import numpy as np

            if isinstance(image_to_save, np.ndarray):
                img = Image.fromarray(image_to_save)
                img.save(initial_image_path)
                print(f"📸 Saved initial screenshot as {initial_image_path}")

        # Save initial SOM-annotated screenshot
        image_som = initial_observation.get("image")
        if image_som is not None:
            initial_som_path = os.path.join(self.images_som_dir, "step_000.png")
            from PIL import Image
            import numpy as np

            if isinstance(image_som, np.ndarray):
                img_som = Image.fromarray(image_som)
                img_som.save(initial_som_path)
                print(f"📸 Saved initial SOM screenshot as {initial_som_path}")
        
        # Track continuation decision for final summary
        continuation_decision = {"reason": "Execution started"}
        
        step_result = None
        # Main execution loop
        while True:
            try:
                # Check if we should continue
                context_summary = self._get_current_context_summary()
                continuation_decision = self.workflow_manager.should_continue_execution(context_summary)

                if not continuation_decision["should_continue"]:
                    self.context_agent.update_state(
                        current_observation=self.current_observation,
                        latest_intention=self.intentions[-1] if self.intentions else None,
                        latest_action=self.actions[-1] if self.actions else None,
                        latest_reflection=self.reflections[-1] if self.reflections else None,
                    )
                    break

                # Handle intervention requirements
                if continuation_decision.get("requires_intervention"):
                    pass

                # Execute one coordination cycle
                step_result = self._execute_coordination_cycle(images)

                # Check for early termination
                if step_result.get("should_terminate", False):
                    termination_reason = step_result.get("termination_reason", "Unknown")
                    print(f"🛑 Task terminated: {termination_reason}")
                    continuation_decision["reason"] = termination_reason
                    break

                # Update current observation
                if step_result.get("new_observation"):
                    self.current_observation = step_result["new_observation"]

                
            except Exception as e:
                # Record error and continue
                print(f"Error during agent execution: {e}")
                traceback.print_exc()
                break

        # Finalize execution
        # Ensure trajectory ends with an Action for compatibility with evaluators
        from browser_env.actions import create_stop_action, ActionTypes
        from beartype.door import is_bearable
        from browser_env import Action
        
        # Check if trajectory is empty or the last element is not an Action
        if not self.trajectory or not is_bearable(self.trajectory[-1], Action):
            # If we have a stored stop action from actor agent, use it
            if self.stop_action_data and "raw_action" in self.stop_action_data:
                # Use the original stop action from actor agent
                final_stop_action = self.stop_action_data["raw_action"]
                print("🔧 Using stored STOP action from actor agent for trajectory")
            else:
                # Create a new STOP action
                final_stop_action = create_stop_action("Task completed")
                print("🔧 Created new STOP action for trajectory")
            
            # Add the STOP action to ensure trajectory ends with an Action
            self.trajectory.append(final_stop_action)
        
        final_context_summary = self._get_current_context_summary()
        workflow_final = self.workflow_manager.finalize_workflow(
            final_state="completed",
            completion_reason=continuation_decision.get("reason", "Execution completed")
        )

        # Log final execution summary - use context agent completion check
        task_completed = self.context_agent.check_task_completion(self.user_goal, actions=self.actions)

        if self.enable_memory_store:
            self.context_agent.generate_and_store_memory(task_completed)

        final_summary = {
            "total_steps_executed": len(self.actions),
            "task_completed": task_completed,
            "completion_percentage": "Completed" if task_completed else "In Progress",
            "total_intentions": len(self.intentions),
            "total_actions": len(self.actions),
            "total_reflections": len(self.reflections),
            "stop_action_data": self.stop_action_data,  # Include stop action data
        }
        self.log_agent_response("execution_summary", len(self.actions), final_summary)

        # Save WebJudge style result
        final_result_response = step_result.get("execution_result", {}).get("action", {}).get("answer",{}) or ("Task completed" if task_completed else "Task incomplete")  # 生成最终回答
        self._save_webjudge_result(final_result_response)  # 写出 WebJudge 结果

        # Return comprehensive execution result
        return {
            "task_info": {
                "user_goal": self.user_goal,
                "max_steps": self.max_steps,
                "total_steps_executed": len(self.actions),
                "task_completed": task_completed,
                "completion_percentage": "Completed" if task_completed else "In Progress",
            },
            "execution_trajectory": {
                "intentions": self.intentions,
                "actions": self.actions,
                "reflections": self.reflections,
                "trajectory_length": len(self.trajectory),
            },
            "workflow_results": workflow_final,
            "final_context": final_context_summary,
            "stop_action_data": self.stop_action_data,  # Include stop action data
        }

    def _register_agents(self) -> None:
        """Register all agents with the communication hub."""
        self.communication_hub.register_agent(
            "context_agent", {"state": "ready", "capabilities": ["context_management", "summary_generation"]}
        )
        self.communication_hub.register_agent(
            "planner_agent", {"state": "ready", "capabilities": ["task_planning", "intention_generation"]}
        )
        self.communication_hub.register_agent(
            "actor_agent", {"state": "ready", "capabilities": ["action_execution", "browser_interaction"]}
        )
        self.communication_hub.register_agent(
            "reflector_agent", {"state": "ready", "capabilities": ["reflection", "validation", "recovery"]}
        )

    def _execute_coordination_cycle(self, images: Optional[List[Image.Image]] = None) -> Dict[str, Any]:
        """Execute one complete coordination cycle.

        Args:
            images: Optional input images

        Returns:
            Dictionary containing cycle results
        """
        step_number = self.workflow_manager.current_step + 1
        print(f"🔄 Executing step {step_number}")


        # 1. Context Agent updates context
        print("🧠 Context Agent: Updating context...")
        try:
            
            context_result = self.context_agent.update_context(
                trajectory=self.trajectory,
                user_goal=self.user_goal,
                current_observation=self.current_observation,
                latest_intention=self.intentions[-1] if self.intentions else None,
                latest_action=self.actions[-1] if self.actions else None,
                latest_reflection=self.reflections[-1] if self.reflections else None,
            )
            # Show only key context information
            summary = context_result.get("summary", "No summary")
            print(f"🧠 Context: {summary[:300]}{'...' if len(summary) > 300 else ''}")

            # Store response information for execution summary

            # Log context agent response summary with detailed breakdown
            context_response = {
                "summary": summary,
            }
            self.log_agent_response("context_agent", step_number, context_response)
            
        except Exception as e:
            print(f"🧠 Context Error: {str(e)[:100]}{'...' if len(str(e)) > 100 else ''}")
            context_result = {
                "summary": "Error generating context",
            }
            # Log context agent error summary
            error_response = {
                "error": str(e),
                "summary": "Error generating context",
            }
            self.log_agent_response("context_agent", step_number, error_response)


        # Check for task completion using context agent's completion check
        if self.context_agent.check_task_completion(self.user_goal, actions=self.actions):
            return {
                "should_terminate": True,
                "termination_reason": "Task completed",
                "context_result": context_result,
            }

        # 2. Planner Agent generates intention
        print("🎯 Planner Agent: Generating intention...")
        try:
            planning_result = self.planner_agent.generate_intention(
                user_goal=self.user_goal,
                context_summary=context_result,
                current_observation=self.current_observation,
                previous_intentions=self.intentions,
            )

            current_intention = planning_result["intention"]
            self.intentions.append(current_intention)

            # Show key planner information
            current_subtask = planning_result.get("current_subtask", "")
            next_atomic_action = planning_result.get("next_atomic_action", "")
            reasoning = planning_result.get("reasoning", "")
            all_subtasks = planning_result.get("all_subtasks", [])
            current_step_index = planning_result.get("current_step_index", 0)
            total_subtasks = planning_result.get("total_subtasks", 0)

            print(f"🎯 Current Subtask [{current_step_index + 1}/{total_subtasks}]: {current_subtask[:100]}{'...' if len(current_subtask) > 100 else ''}")
            if next_atomic_action != current_subtask:
                print(f"🎯 Next Atomic Action: {next_atomic_action[:100]}{'...' if len(next_atomic_action) > 100 else ''}")

            # Show all subtasks overview
            if all_subtasks and total_subtasks > 0:
                print(f"🎯 Task Overview ({total_subtasks} subtasks):")
                for i, subtask in enumerate(all_subtasks):
                    if i < current_step_index:
                        status = "✅"  # Completed
                    elif i == current_step_index:
                        status = "🔄"  # In progress
                    else:
                        status = "⏳"  # Pending
                    print(f"   {status} {i+1}. {subtask[:80]}{'...' if len(subtask) > 80 else ''}")

            # Show selected intention
            print(f"✅ Selected Intention: {current_intention[:100]}{'...' if len(current_intention) > 100 else ''}")


            # Log planner agent response summary
            planner_response = {
                "intention": current_intention,
                "current_subtask": current_subtask,
                "next_atomic_action": next_atomic_action,
                "reasoning": reasoning,
                "all_subtasks": all_subtasks,
                "current_step_index": current_step_index,
                "total_subtasks": total_subtasks,
                "task_decomposed": planning_result.get("task_decomposed", False),
                "response": planning_result.get("response", "")
            }
            self.log_agent_response("planner_agent", step_number, planner_response)
        except Exception as e:
            print(f"🎯 Planner Error: {str(e)[:100]}{'...' if len(str(e)) > 100 else ''}")
            # Create fallback planning result for error case
            planning_result = {
                "intention": f"Continue working on: {self.user_goal}",
                "current_subtask": f"Continue working on: {self.user_goal}",
                "next_atomic_action": f"Continue working on: {self.user_goal}",
                "reasoning": f"Fallback intention due to error: {str(e)}",
                "all_subtasks": [f"Complete the task: {self.user_goal}"],
                "current_step_index": 0,
                "total_subtasks": 1,
                "task_decomposed": False
            }
            current_intention = planning_result["intention"]
            self.intentions.append(current_intention)
            current_subtask = planning_result["current_subtask"]
            all_subtasks = planning_result["all_subtasks"]

            # Log planner agent error summary
            error_response = {
                "error": str(e),
                "intention": planning_result["intention"],
                "current_subtask": planning_result["current_subtask"],
                "next_atomic_action": planning_result["next_atomic_action"],
                "reasoning": planning_result["reasoning"],
                "task_decomposed": planning_result["task_decomposed"]
            }

        # 3. Actor Agent executes intention
        print("🎬 Actor Agent: Executing intention...")
        try:
            # Merge step_number into meta_data while preserving action_history
            # This matches the pattern used in run.py
            meta_data_for_action = self.meta_data.copy()
            meta_data_for_action["step_number"] = step_number
            
            # Ensure current_observation has text, image, and image_raw fields
            current_obs = self.current_observation or {"text": "", "image": None, "image_raw": None}
            if isinstance(current_obs, dict):
                if "text" not in current_obs:
                    current_obs["text"] = ""
                if "image" not in current_obs:
                    current_obs["image"] = None
                if "image_raw" not in current_obs:
                    current_obs["image_raw"] = current_obs.get("image")
            
            execution_result = self.actor_agent.execute_intention(
                intention=current_intention,
                current_observation=current_obs,
                trajectory=self.trajectory,
                meta_data=meta_data_for_action,
                images=images,
            )
            
            # Check if execution was successful and contains action
            if "action" in execution_result:
                executed_action = execution_result["action"]
                self.actions.append(copy.deepcopy(executed_action))

                # Check if this is a STOP action
                action_type = executed_action.get("action_type")
                if action_type == ActionTypes.STOP:
                    # Extract and store stop action data for future evaluation use
                    self.stop_action_data = self._extract_stop_action_data(executed_action)
                    print(f"🛑 STOP action detected. Answer: {self.stop_action_data.get('answer', 'N/A')[:100]}")

                # Execute action in browser environment if available
                if self.browser_env is not None:
                    try:
                        print(f"🔍 Executing action in browser: {action_type}")
                        obs, reward, terminated, truncated, info = self.browser_env.step(executed_action)

                        # Ensure observation has text, image, and image_raw fields
                        if isinstance(obs, dict):
                            if "text" not in obs:
                                obs["text"] = ""
                            if "image" not in obs:
                                obs["image"] = None
                            if "image_raw" not in obs:
                                # Fallback: use image as image_raw if not available
                                obs["image_raw"] = obs.get("image")
                        else:
                            obs = {"text": str(obs) if obs else "", "image": None, "image_raw": None}

                        # Update current observation with new browser state
                        self.current_observation = obs  # Keep as observation format
                        new_observation = obs  # Keep as observation format

                        # Log observation (text and image)
                        self.log_observation(step_number, obs)
                        # Capture playwright full-page screenshot for WebJudge
                        self._capture_and_save_screenshot(step_number)  # 保存当前步截图

                        # Determine if intention is fulfilled based on execution success
                        intention_fulfilled = reward == 1.0  # reward is 1.0 for success, 0.0 for failure

                        print(f"✅ Browser execution successful - URL: {info.get('page', {}).url if 'page' in info else 'Unknown'}")
                    except Exception as e:
                        print(f"❌ Browser execution failed: {str(e)}")
                        intention_fulfilled = False
                        info = None
                        new_observation = self.current_observation
                else:
                    # No browser environment - simulate success for compatibility
                    intention_fulfilled = False  # Will be determined by reflection
                    info = None
                    new_observation = self.current_observation

                # Update execution result with browser execution results
                execution_result["intention_fulfilled"] = intention_fulfilled

            else:
                # Create a fallback action for failed executions
                executed_action = {
                    "action_type": "NONE",
                    "error": execution_result.get("error", "Unknown execution error"),
                    "intention": current_intention
                }
                self.actions.append(executed_action)
                info = None
                new_observation = self.current_observation

            # Show key actor execution information
            action_type = executed_action.get("action_type", "UNKNOWN")
            intention_fulfilled = execution_result.get("intention_fulfilled", False)
            print(f"🎬 Actor: {action_type} - Fulfilled: {intention_fulfilled}")

            # Show LLM response if available (truncated)
            llm_response = execution_result.get("llm_response", "")
            if llm_response:
                print(f"   LLM Response: {llm_response[:500]}{'...' if len(llm_response) > 500 else ''}")


            # Log actor agent response summary
            actor_response = {
                'response': llm_response,
                "action_type": action_type,
                "fulfilled": intention_fulfilled
            }
            self.log_agent_response("actor_agent", step_number, actor_response)

        except Exception as e:
            executed_action = {
                "action_type": "NONE",
                "error": str(e),
                "intention": current_intention
            }
            self.actions.append(executed_action)
            info = None
            new_observation = self.current_observation
            import traceback
            traceback.print_exc()

            print(f"🎬 Actor Error: {str(e)[:100]}{'...' if len(str(e)) > 100 else ''}")


            # Store error response for execution summary, try to extract LLM response
            error_details = str(e)
            llm_response = "No LLM response available due to exception"

            # Try to get LLM response from execution_result if available
            if 'execution_result' in dir() and hasattr(execution_result, 'get'):
                if execution_result.get("llm_response"):
                    llm_response = execution_result.get("llm_response")
                elif execution_result.get("response"):
                    llm_response = execution_result.get("response")
                if execution_result.get("error"):
                    error_details = execution_result.get("error")

            # Log actor agent error summary
            self.log_agent_response("actor_agent", step_number, {"error": error_details})

        # 4. Action execution is now handled above with browser environment

        # Add to trajectory (following run.py pattern)
        # trajectory should contain: StateInfo, Action, StateInfo, Action, StateInfo...
        # trajectory[-1] should already be the current StateInfo, so we just add action and new state
        prev_state_info = None

        if self.trajectory and isinstance(self.trajectory[-1], dict):
            prev_state_info = self.trajectory[-1]
            
        else:
            prev_state_info = None
        

        # observation_metadata = self.trajectory[-1].get("info", {}).get("observation_metadata", {})

        # step_meta_path = os.path.join(
        #     self.trajectory_dir,
        #     f"observation_metadata_step_{self.workflow_manager.current_step:03d}_id_01.json"
        # )
        # try:
        #     with open(step_meta_path, "w", encoding="utf-8") as f:
        #         json.dump(observation_metadata, f, ensure_ascii=False, indent=2)
        # except Exception as e:
        #     print(f"Warning: Failed to save observation metadata for step {self.workflow_manager.current_step}: {e}")


        # 在trajectory更新前，保存所有偶数位置的observation_metadata
        # for pos in range(0, len(self.trajectory), 2):  # 偶数位置是StateInfo
        #     if pos < len(self.trajectory):
        #         state_info = self.trajectory[pos]
        #         if isinstance(state_info, dict):
        #             observation_metadata = state_info.get("info", {}).get("observation_metadata", {})

        #             step_meta_path = os.path.join(
        #                 self.trajectory_dir,
        #                 f"observation_metadata_step_{self.workflow_manager.current_step:03d}_trajectory_pos_{pos:02d}.json"
        #             )
        #             try:
        #                 with open(step_meta_path, "w", encoding="utf-8") as f:
        #                     json.dump(observation_metadata, f, ensure_ascii=False, indent=2)
        #             except Exception as e:
        #                 print(f"Warning: Failed to save trajectory observation metadata for step {self.workflow_manager.current_step}, pos {pos}: {e}")

        prev_info_for_desc = prev_state_info.get("info") if isinstance(prev_state_info, dict) else None
        prev_text_observation = prev_state_info.get("observation", {}).get("text", "") if isinstance(prev_state_info, dict) else ""
        formatter_info = copy.deepcopy(prev_info_for_desc) if isinstance(prev_info_for_desc, dict) else {}
        formatter_info["prev_text_observation"] = prev_text_observation

        self.trajectory.append(executed_action)
        if info is None:
            info = {
                "page": type('Page', (), {'url': ''})(),
                "observation_metadata": {}
            }

        # Create new state_info from new_observation
        # 深拷贝info以避免trajectory中所有StateInfo引用同一个对象
        new_state_info = {
            "observation": copy.deepcopy(new_observation),  # new_observation is now observation format
            "info": copy.deepcopy(info)
        }
        self.trajectory.append(new_state_info)

        # WebJudge logging
        self.webjudge_thoughts.append(current_intention)  # 记录本步意图
        use_html_format = False  # 如需 HTML 标签样式，改为 True
        formatter = self._format_action_for_webjudge_html if use_html_format else self._format_action_for_webjudge
        self.webjudge_action_history.append(formatter(executed_action, formatter_info))  # 记录本步动作（使用执行前的 info 避免 DOM 变动导致缺元素）
        self._capture_and_save_screenshot(step_number)  # 记录本步截图

        # Update current observation to stay in sync with trajectory
        self.current_observation = new_observation
        
        # Update action_history in meta_data (required by DirectPromptConstructor)
        # For multi-agent simulation, use simplified action descriptions to avoid dependency on complex observation_metadata
        action_type = executed_action.get("action_type", "UNKNOWN")

        if action_type == ActionTypes.NONE:
            action_str = f"Failed action: {executed_action.get('error', 'Unknown error')}"
        elif action_type == ActionTypes.GOTO_URL:
            url = executed_action.get("url", "unknown")
            action_str = f"Navigate to {url}"
        elif action_type == ActionTypes.CLICK:
            element_id = executed_action.get("element_id", "unknown")
            action_str = f"Click on element {element_id}"
        elif action_type == ActionTypes.TYPE:
            element_id = executed_action.get("element_id", "unknown")
            text = executed_action.get("text", [""])[0] if executed_action.get("text") else ""
            action_str = f"Type '{text}' into element {element_id}"
        elif action_type == ActionTypes.SCROLL:
            direction = executed_action.get("direction", "unknown")
            action_str = f"Scroll {direction}"
        elif action_type == ActionTypes.HOVER:
            element_id = executed_action.get("element_id", "unknown")
            action_str = f"Hover over element {element_id}"
        elif action_type == ActionTypes.STOP:
            action_str = f"Stop action"
        else:
            action_str = f"Action: {action_type}"

        self.meta_data["action_history"].append(action_str)

        # 5. Reflector Agent reflects on execution with checklist approach
        # Get current subtask from planning result
        current_subtask = planning_result.get("current_subtask", current_intention)
        all_subtasks = planning_result.get("all_subtasks", [])
        
        reflection_result = self.reflector_agent.reflect_execution(
            trajectory=self.trajectory,
            intentions=self.intentions,
            actions=self.actions,
            current_intention=current_intention,
            latest_action=executed_action,
            current_observation=new_observation,
            context_summary=context_result,
            current_subtask=current_subtask,
            high_level_task=self.user_goal,
            all_subtasks=all_subtasks,
        )

        self.reflections.append(reflection_result)

        # Show checklist results
        checklist = reflection_result.get("checklist", {})
        print(f"📋 Reflector Checklist:")
        print(f"   - Pattern Issue: {checklist.get('has_pattern_issue', False)}")
        print(f"   - Execution Success: {checklist.get('execution_successful', True)}")
        print(f"   - Subtask Completed: {checklist.get('subtask_completed', False)}")  # NEW
        print(f"   - Subtask Needs Revision: {checklist.get('subtask_needs_revision', False)}")
        print(f"   - Task Completed: {checklist.get('task_completed', False)}")

        # Log reflector agent response summary with checklist format
        reflector_response = {
            "has_pattern_issue": reflection_result.get("has_pattern_issue", False),
            "execution_successful": reflection_result.get("execution_successful", True),
            "subtask_completed": reflection_result.get("subtask_completed", False),  # NEW
            "subtask_needs_revision": reflection_result.get("subtask_needs_revision", False),
            "task_completed": reflection_result.get("task_completed", False),
            "raw_response": checklist.get("raw_response", ""),
        }
        self.log_agent_response("reflector_agent", step_number, reflector_response)

        # 6. Handle subtask progression based on reflection results
        
        # Check for task completion with STOP action
        task_completed = reflection_result.get("task_completed", False)
        is_stop_action = executed_action.get("action_type") == ActionTypes.STOP
        
        if task_completed and is_stop_action:
            print(f"🎉 Task completed with STOP action!")
            return {
                "should_terminate": True,
                "termination_reason": "Task completed with STOP action",
                "step_number": step_number,
                "context_result": context_result,
                "planning_result": planning_result,
                "execution_result": execution_result,
                "reflection_result": reflection_result,
                "new_observation": new_observation,
            }
        
        # Handle subtask completion - advance to next subtask
        subtask_completed = reflection_result.get("subtask_completed", False)
        if subtask_completed:
            print(f"✅ Subtask completed! Advancing to next subtask...")
            has_more_subtasks = self.planner_agent.mark_current_subtask_completed()
            if not has_more_subtasks:
                print(f"🎉 All subtasks completed!")
                # Don't terminate yet - let the reflector determine if overall task is done
        
        # Handle subtask revision if needed
        # NOTE: Subtask revision feature is temporarily disabled pending further refinement
        # subtask_needs_revision = reflection_result.get("subtask_needs_revision", False)
        # if subtask_needs_revision and not subtask_completed:
        #     print(f"🔄 Subtask needs revision, generating revised subtask...")
        #     revision_result = self.reflector_agent.generate_revised_subtask(
        #         high_level_task=self.user_goal,
        #         current_subtask=current_subtask,
        #         all_subtasks=all_subtasks,
        #         intentions=self.intentions,
        #         current_observation=new_observation,
        #     )
        #     
        #     if revision_result.get("success"):
        #         revised_subtask = revision_result["revised_subtask"]
        #         self.planner_agent.revise_current_subtask(revised_subtask)
        #         print(f"🔄 Subtask revised to: {revised_subtask[:80]}...")
        #         
        #         # Log revision
        #         revision_log = {
        #             "original_subtask": current_subtask,
        #             "revised_subtask": revised_subtask,
        #             "reasoning": revision_result.get("reasoning", ""),
        #         }
        #         self.log_agent_response("subtask_revision", step_number, revision_log)

        # 7. Record workflow step
        self.workflow_manager.record_execution_step(
            step_number=step_number,
            intention=current_intention,
            action=executed_action,
            observation=new_observation,
            reflection=reflection_result
        )

        return {
            "should_terminate": False,
            "step_number": step_number,
            "context_result": context_result,
            "planning_result": planning_result,
            "execution_result": execution_result,
            "reflection_result": reflection_result,
            "new_observation": new_observation,
        }

    def _extract_stop_action_data(self, action: Action) -> Dict[str, Any]:
        """Extract data from a STOP action for future evaluation use.
        
        Args:
            action: The STOP action containing answer data
            
        Returns:
            Dictionary containing extracted stop action data
        """
        stop_data = {
            "action_type": "STOP",
            "answer": "",
            "raw_action": action,
        }
        
        # Try to extract the answer from the action
        # The answer might be in different fields depending on action format
        if "answer" in action:
            stop_data["answer"] = action["answer"]
        elif "text" in action:
            text = action["text"]
            if isinstance(text, list):
                # Decode text from ID list if needed
                try:
                    from browser_env.actions import _id2key
                    stop_data["answer"] = ''.join(_id2key[id_num] if 0 <= id_num < len(_id2key) else '?' for id_num in text)
                except (ImportError, IndexError):
                    stop_data["answer"] = ''.join(chr(id_num) if 32 <= id_num <= 126 else '?' for id_num in text)
            else:
                stop_data["answer"] = str(text)
        elif "args" in action and action["args"]:
            # Some formats store answer in args
            stop_data["answer"] = str(action["args"][0]) if action["args"] else ""
        
        return stop_data

    def _get_current_context_summary(self) -> Dict[str, Any]:
        """Get current context summary without full update."""
        return self.context_agent.get_current_state()

    def reset(self) -> None:
        """Reset all execution state for a new task.
        
        This method should be called before starting a new task to ensure
        clean state separation between consecutive task executions.
        """
        # Reset coordinator execution state
        self.trajectory = []
        self.intentions = []
        self.actions = []
        self.reflections = []
        self.meta_data = {"action_history": ["None"]}
        self.user_goal = ""
        self.current_observation = None
        self.stop_action_data = None  # Reset stop action data
        
        # Reset all sub-agents
        self.context_agent.reset()
        self.planner_agent.reset_planning_state()
        self.actor_agent.reset_intention_history()
        self.reflector_agent.reset_reflection_history()
        
        # Reset coordination components
        self.workflow_manager.reset_workflow()
        
        # Reset logging for new task
        self._setup_logging()
