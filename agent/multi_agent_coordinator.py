"""Multi-Agent Coordinator for managing collaborative agent execution.

This module coordinates Context, Planner, Actor, and Reflector agents
with explicit subtask state management and task completion handling.
"""

from typing import Any, Dict, List, Optional, Union
import json
import os
import copy
from datetime import datetime

from PIL import Image
import numpy as np

from browser_env import Action, ActionTypes, Trajectory
from browser_env.helper_functions import get_action_description
from browser_env.actions import create_stop_action, _id2key
from beartype.door import is_bearable
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


from .context_agent import ContextAgent
from .actor_agent import ActorAgent
from .reflector_agent import ReflectorAgent
from .monitor import GeneralMonitor
from .trajectory_logger import TrajectoryLogger


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
                 monitor_config: Dict[str, Any] = {},
                 clear_result_dir: bool = False,
                 save_images: bool = True) -> None:
        self.lm_config = lm_config

        # Get action set tag from existing agent or use default
        action_set_tag = getattr(existing_prompt_agent, 'action_set_tag', 'som')

        self.enable_memory = memory_config.get("enable_memory", False)
        self.enable_memory_store = memory_config.get("enable_memory_store", False)

        # Monitor configuration
        self.enable_monitor = monitor_config.get("enable_monitor", False)
        
        # Output configuration
        self.clear_result_dir = clear_result_dir
        self.save_images = save_images

        # Initialize individual agents with memory enabled if specified
        self.context_agent = ContextAgent(lm_config, memory_config)
        self.actor_agent = ActorAgent(
            action_set_tag=action_set_tag,
            lm_config=lm_config,
            prompt_constructor=getattr(existing_prompt_agent, 'prompt_constructor', None),
            captioning_fn=getattr(existing_prompt_agent, 'captioning_fn', None)
        )
        self.reflector_agent = ReflectorAgent(lm_config)

        # Initialize GeneralMonitor if enabled (NEW)
        self.monitor: Optional[GeneralMonitor] = None
        if self.enable_monitor:
            self.monitor = GeneralMonitor(
                lm_config=lm_config,
                memory_config=memory_config,
            )
            print("🔍 GeneralMonitor enabled")
        else:
            print("📋 Baseline mode (GeneralMonitor disabled)")

        # Browser environment for action execution
        self.browser_env = browser_env

        # Result directory and logging setup
        self.result_dir = result_dir
        self.log_file_path = os.path.join(result_dir, "agent_responses.log")
        self.observation_log_path = os.path.join(result_dir, "observations.json")
        self.images_dir = os.path.join(result_dir, "images")
        self.images_som_dir = os.path.join(result_dir, "images_som")
        self._setup_logging()

        # Initialize TrajectoryLogger for HTML report generation
        self.trajectory_logger = TrajectoryLogger(
            output_dir=result_dir,
            task_name="Agent Execution"
        )

        # Execution state
        self.trajectory: Trajectory = []
        self.intentions: List[str] = []
        self.actions: List[Action] = []
        self.reflections: List[Dict[str, Any]] = []
        
        # Monitor feedback storage
        self.monitor_feedback: Optional[str] = None

        # Meta data for action history tracking (required by DirectPromptConstructor)
        # Initialize with "None" as the first action, matching run.py implementation
        self.meta_data: Dict[str, Any] = {"action_history": ["None"]}

        # Task configuration
        self.user_goal: str = ""
        self.max_steps: int = 30
        self.current_step: int = 0
        self.current_observation: Optional["ObservationTypeAlias"] = None
        
        # Stop action data storage (for future evaluation use)
        self.stop_action_data: Optional[Dict[str, Any]] = None

    def _setup_logging(self) -> None:
        """Setup logging for agent responses."""
        try:
            # Clear result directory if requested
            if self.clear_result_dir and os.path.exists(self.result_dir):
                import shutil
                shutil.rmtree(self.result_dir)
                print(f"Cleared result directory: {self.result_dir}")

            # Ensure result directory exists
            os.makedirs(self.result_dir, exist_ok=True)

            # Create images directories only if saving images is enabled
            if self.save_images:
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

            # Save screenshot image if available and saving is enabled (prefer image_raw for cleaner logs)
            if self.save_images:
                image_to_save = observation.get("image_raw")
                if image_to_save is None:
                    image_to_save = observation.get("image")
                if image_to_save is not None:
                    image_path = os.path.join(self.images_dir, f"step_{step_number:03d}.png")
                    # Convert numpy array to PIL Image and save
                    if isinstance(image_to_save, np.ndarray):
                        img = Image.fromarray(image_to_save)
                        img.save(image_path)

            # Save SOM-annotated screenshot (image with bounding boxes and IDs) if saving is enabled
            if self.save_images:
                image_som = observation.get("image")
                if image_som is not None:
                    image_som_path = os.path.join(self.images_som_dir, f"step_{step_number:03d}.png")
                    if isinstance(image_som, np.ndarray):
                        img_som = Image.fromarray(image_som)
                        img_som.save(image_som_path)

        except Exception as e:
            print(f"Warning: Failed to log observation for step {step_number}: {e}")


    def execute_task(
        self,
        user_goal: str,
        start_observation: Optional["ObservationTypeAlias"] = None,
        max_steps: int = 30,
        images: Optional[List[Image.Image]] = None,
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

        # Initialize workflow and monitoring
        self.current_step = 0
        
        # Initialize trajectory logger for HTML report
        self.trajectory_logger.set_user_goal(user_goal)

        # Initialize GeneralMonitor if enabled
        if self.enable_monitor and self.monitor:
            print("🔍 Initializing GeneralMonitor for task...")
            self.monitor.initialize(user_goal)
            self.monitor_feedback = None
        
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
        # Note: trajectory is always empty here since we call reset() at the start
        # Handle different formats of start_observation
        if start_observation is None:
            # No observation provided, create empty state
            initial_observation = {"text": "", "image": None, "image_raw": None}
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

        initial_state_info = {
            "observation": initial_observation,
            "info": initial_info
        }
        self.trajectory.append(initial_state_info)
        self.current_observation = initial_observation

        # Save initial screenshot as step_000.png if saving is enabled (prefer image_raw for cleaner logs)
        if self.save_images:
            image_to_save = initial_observation.get("image_raw")
            if image_to_save is None:
                image_to_save = initial_observation.get("image")
            if image_to_save is not None:
                initial_image_path = os.path.join(self.images_dir, "step_000.png")
                # Convert numpy array to PIL Image and save
                if isinstance(image_to_save, np.ndarray):
                    img = Image.fromarray(image_to_save)
                    img.save(initial_image_path)
                    print(f"📸 Saved initial screenshot as {initial_image_path}")

            # Save initial SOM-annotated screenshot
            image_som = initial_observation.get("image")
            if image_som is not None:
                initial_som_path = os.path.join(self.images_som_dir, "step_000.png")
                if isinstance(image_som, np.ndarray):
                    img_som = Image.fromarray(image_som)
                    img_som.save(initial_som_path)
                    print(f"📸 Saved initial SOM screenshot as {initial_som_path}")
        
        # Track continuation decision for final summary
        continuation_decision = {"reason": "Execution started"}
        
        # Main execution loop
        while True:
            try:
                # Check if we should continue
                context_summary = self._get_current_context_summary()
                # Check step limit - this is the primary stopping condition
                continuation_decision = {
                    "should_continue": self.current_step < self.max_steps,
                    "reason": "Maximum steps reached" if self.current_step >= self.max_steps else "Execution should continue",
                }

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
                print(f"❌ Execution error: {e}")
                break

        # Finalize execution
        # Ensure trajectory ends with an Action for compatibility with evaluators
        
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

        # Log final execution summary - use context agent completion check
        task_completed = self.context_agent.check_task_completion(self.user_goal, actions=self.actions)

        if self.enable_memory_store:
            self.context_agent.generate_and_store_memory(task_completed)
        
        # Generate HTML trajectory report
        try:
            self.trajectory_logger.generate_html_report("trajectory_report.html")
        except Exception as e:
            print(f"⚠️ Failed to generate trajectory report: {e}")

        # final_summary = {
        #     "total_steps_executed": len(self.actions),
        #     "task_completed": task_completed,
        #     "completion_percentage": "Completed" if task_completed else "In Progress",
        #     "total_intentions": len(self.intentions),
        #     "total_actions": len(self.actions),
        #     "total_reflections": len(self.reflections),
        #     "stop_action_data": self.stop_action_data,  # Include stop action data
        # }
        # self.log_agent_response("execution_summary", len(self.actions), final_summary)

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
            "final_context": final_context_summary,
            "stop_action_data": self.stop_action_data,  # Include stop action data
        }


    def _execute_coordination_cycle(self, images: Optional[List[Image.Image]] = None) -> Dict[str, Any]:
        """Execute one complete coordination cycle.

        Args:
            images: Optional input images

        Returns:
            Dictionary containing cycle results
        """
        step_number = self.current_step + 1
        print(f"🔄 Executing step {step_number}")
        
        # Start trajectory logging for this step
        self.trajectory_logger.start_step(step_number)

        # Initialize context_result for both modes
        context_result = {"summary": ""}
        
        # Extract current URL for logging
        current_url = ""
        if self.trajectory and len(self.trajectory) >= 1:
            last_state = self.trajectory[-1]
            if isinstance(last_state, dict) and "info" in last_state:
                info = last_state["info"]
                if info and hasattr(info.get("page"), "url"):
                    current_url = info["page"].url
        
        # Log observation to trajectory logger
        if self.current_observation:
            obs = self.current_observation
            if isinstance(obs, dict):
                self.trajectory_logger.log_observation(
                    observation_text=obs.get("text", "")[:10000],
                    observation_image=obs.get("image_raw"),
                    som_image=obs.get("image"),
                    current_url=current_url
                )

        # 1. Context Agent updates context (ONLY in Monitor mode)
        if self.enable_monitor and self.monitor:
            print("🧠 Context Agent: Updating context...")
            try:
                # Extract current URL from trajectory info (already done above)
                
                context_result = self.context_agent.update_context(
                    trajectory=self.trajectory,
                    user_goal=self.user_goal,
                    current_observation=self.current_observation,
                    latest_intention=self.intentions[-1] if self.intentions else None,
                    latest_action=self.actions[-1] if self.actions else None,
                    latest_reflection=self.reflections[-1] if self.reflections else None,
                    current_url=current_url,
                )
                # Show only key context information
                summary = context_result.get("summary", "No summary")
                print(f"🧠 Context: {summary[:300]}{'...' if len(summary) > 300 else ''}")

                # Log context agent response summary
                context_response = {
                    "summary": summary,
                }
                self.log_agent_response("context_agent", step_number, context_response)
                
                # Log to trajectory logger for HTML report
                self.trajectory_logger.log_llm_call(
                    agent_name="Context",
                    prompt=f"User Goal: {self.user_goal}",
                    response=summary,
                    parsed_result={"summary": summary[:500]}
                )
                
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
                self.trajectory_logger.end_step()
                return {
                    "should_terminate": True,
                    "termination_reason": "Task completed",
                    "context_result": context_result,
                }


        # 3. Actor Agent executes intention
        print("🎬 Actor Agent: Executing intention...")
        current_intention = self.user_goal
        execution_result = None
        info = None
        try:
            # Merge step_number into meta_data while preserving action_history
            # This matches the pattern used in run.py
            meta_data_for_action = self.meta_data.copy()
            meta_data_for_action["step_number"] = step_number
            # Add context information from context_agent instead of using action_history as previous_action
            meta_data_for_action["context_summary"] = context_result.get("summary", "No context available")

            # Add pattern issue analysis results from previous reflection to help actor agent avoid repetitive errors
            # Use the latest reflection if available
            latest_reflection = self.reflections[-1] if self.reflections else None
            if latest_reflection:
                checklist = latest_reflection.get("checklist", {})
                pattern_issue_analysis = checklist.get("pattern_issue_analysis")
                if pattern_issue_analysis and pattern_issue_analysis.get("pattern_confirmed", False):
                    meta_data_for_action["pattern_issue_analysis"] = pattern_issue_analysis
                    print(f"🎬 Actor: Pattern issue analysis from previous step provided to guide action selection")
            
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
                monitor_feedback=self.monitor_feedback,  # Inject Monitor feedback
            )
            
            # Check if execution was successful and contains action
            if "action" in execution_result:
                executed_action = execution_result["action"]
                self.actions.append(executed_action)

                # Check if this is a STOP action
                action_type = executed_action.get("action_type")
                if action_type == ActionTypes.STOP:
                    # Extract and store stop action data for future evaluation use
                    self.stop_action_data = self._extract_stop_action_data(executed_action)
                    proposed_answer = self.stop_action_data.get('answer', 'N/A')
                    print(f"🛑 STOP action detected. Answer: {proposed_answer[:100]}")
                    # Note: STOP validation is now handled by Monitor's checklist in step() method

                # Execute action in browser environment if available
                if self.browser_env is not None and action_type != "NONE":
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
            # if llm_response:
            #     print(f"   LLM Response: {llm_response[:500]}{'...' if len(llm_response) > 500 else ''}")


            # Update Context Agent with the extracted intention from Actor's reasoning
            # Only update Context Agent in monitor mode
            extracted_intention = execution_result.get("extracted_intention")
            if extracted_intention:
                # Update intentions history for pattern detection
                self.intentions.append(extracted_intention)
                # Only call Context Agent in monitor mode
                if self.enable_monitor and self.monitor:
                    print(f"🧠 Updating Context with Actor's intention: {extracted_intention[:100]}{'...' if len(extracted_intention) > 100 else ''}")
                    self.context_agent.update_state(
                        latest_intention=extracted_intention
                    )

            # Log actor agent response summary with full LLM prompt and response
            llm_prompt = execution_result.get("llm_prompt", "") if execution_result else ""
            actor_response = {
                "llm_prompt": llm_prompt,
                "llm_response": llm_response,  # Complete LLM output including <think> and <action>
                "action_type": action_type,
                "fulfilled": intention_fulfilled,
                "extracted_intention": extracted_intention
            }
            self.log_agent_response("actor_agent", step_number, actor_response)
            
            # NOTE: trajectory_logger.log_llm_call for Actor is done later after action_str is defined

        except Exception as e:
            executed_action = {
                "action_type": "NONE",
                "error": str(e),
                "intention": current_intention
            }
            self.actions.append(executed_action)
            info = None
            new_observation = self.current_observation

            print(f"🎬 Actor Error: {str(e)[:100]}{'...' if len(str(e)) > 100 else ''}")


            # Store error response for execution summary, try to extract LLM response
            error_details = str(e)
            llm_response = "No LLM response available due to exception"

            # Try to get LLM response from execution_result if available
            if execution_result is not None:
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
        self.trajectory.append(executed_action)
        if info is None:
            info = {
                "page": type('Page', (), {'url': ''})(),
                "observation_metadata": {}
            }

        # Create new state_info from new_observation
        # IMPORTANT: Deep copy observation_metadata to avoid reference sharing issue.
        # browser_env.step() returns info with observation_metadata that references
        # internal processor.meta_data which gets mutated on subsequent steps.
        # Without deep copy, all trajectory entries would share the same metadata.
        info_copy = {
            "page": info.get("page"),  # page is already a new DetachedPage per step
            "fail_error": info.get("fail_error", ""),
            "observation_metadata": copy.deepcopy(info.get("observation_metadata", {}))
        }
        new_state_info = {
            "observation": new_observation,  # new_observation is now observation format
            "info": info_copy
        }
        self.trajectory.append(new_state_info)

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
        
        # Log Actor LLM call to trajectory logger (must be after action_str is defined)
        llm_response = execution_result.get("llm_response", "") if execution_result else ""
        llm_prompt = execution_result.get("llm_prompt", "") if execution_result else ""
        intention_fulfilled = execution_result.get("intention_fulfilled", False) if execution_result else False
        self.trajectory_logger.log_llm_call(
            agent_name="Actor",
            prompt=llm_prompt or f"Goal: {current_intention}\nMonitor Feedback: {self.monitor_feedback or 'None'}",
            response=llm_response or "",
            parsed_result={
                "action_type": str(action_type),
                "fulfilled": intention_fulfilled,
                "action_str": action_str,
            }
        )
        self.trajectory_logger.log_action(executed_action, action_str)

        # 5. Post-action processing: Monitor Mode vs Baseline Mode
        # ============================================================
        
        if self.enable_monitor and self.monitor:
            # === MODE A: Monitor Mode  ===
            # Monitor internally handles Context update and Reflector check
            print("🔍 Monitor: Processing action...")
            
            monitor_feedback, should_stop = self.monitor.step(
                action=executed_action,
                observation=new_observation,
                trajectory=self.trajectory,
                intention=current_intention,
            )
            
            # Store monitor feedback for next Actor call
            self.monitor_feedback = monitor_feedback.to_prompt_injection()
            
            # Log monitor decision
            if monitor_feedback.has_issue:
                print(f"🔍 Monitor: Issue detected")
                if monitor_feedback.correction_guidance:
                    print(f"   - Guidance: {monitor_feedback.correction_guidance[:200]}...")
                if monitor_feedback.prohibited_actions:
                    print(f"   - Prohibited: {', '.join(monitor_feedback.prohibited_actions[:2])}")
                if monitor_feedback.alternative_actions:
                    print(f"   - Alternative Actions: {', '.join(monitor_feedback.alternative_actions[:2])}")
            else:
                print(f"🔍 Monitor: No issues detected")
            
            # Create reflection result from monitor feedback for compatibility
            # Use the actual checklist from monitor_feedback if available
            checklist_data = monitor_feedback.checklist if monitor_feedback.checklist else {
                "has_pattern_issue": monitor_feedback.has_issue,
                "pattern_issue_reason": "",
                "should_stop": should_stop,
                "stop_reason": "",
                "next_step_suggestion": "",
            }
            
            reflection_result = {
                "checklist": checklist_data,
                "monitor_feedback": {
                    "has_issue": monitor_feedback.has_issue,
                    "prohibited_actions": monitor_feedback.prohibited_actions,
                    "alternative_actions": monitor_feedback.alternative_actions,
                    "correction_guidance": monitor_feedback.correction_guidance,
                },
            }
            self.reflections.append(reflection_result)
            
            # Log monitor response with detailed checklist information
            self.log_agent_response("monitor", step_number, {
                "reflector_llm_prompt": checklist_data.get("llm_prompt", "")[:2000],
                "reflector_llm_response": checklist_data.get("raw_response", "")[:2000],
                "has_issue": monitor_feedback.has_issue,
                "prohibited_actions": monitor_feedback.prohibited_actions,
                "alternative_actions": monitor_feedback.alternative_actions,
                "correction_guidance": monitor_feedback.correction_guidance[:200] if monitor_feedback.correction_guidance else "",
                "prompt_injection": self.monitor_feedback[:200] if self.monitor_feedback else "",
                "checklist": {
                    "has_pattern_issue": checklist_data.get("has_pattern_issue", False),
                    "pattern_issue_reason": checklist_data.get("pattern_issue_reason", "")[:200],
                    "should_stop": checklist_data.get("should_stop", False),
                    "stop_reason": checklist_data.get("stop_reason", "")[:200],
                    "next_step_suggestion": checklist_data.get("next_step_suggestion", "")[:200],
                },
            })
            
            # Log checklist result to trajectory logger for HTML report
            self.trajectory_logger.log_checklist_result(checklist_data)
            
            # Log to trajectory logger for HTML report - use actual Reflector LLM prompt/response
            reflector_prompt = checklist_data.get("llm_prompt", f"Action: {action_str}\nIntention: {current_intention}")
            reflector_raw_response = checklist_data.get("raw_response", self.monitor_feedback or "No feedback")
            self.trajectory_logger.log_llm_call(
                agent_name="Reflector",
                prompt=reflector_prompt,
                response=reflector_raw_response,
                parsed_result={
                    "has_issue": monitor_feedback.has_issue,
                    "guidance": monitor_feedback.correction_guidance[:100] if monitor_feedback.correction_guidance else "",
                    "has_pattern_issue": checklist_data.get("has_pattern_issue", False),
                    "should_stop": checklist_data.get("should_stop", False),
                    "stop_reason": checklist_data.get("stop_reason", ""),
                }
            )
            self.trajectory_logger.log_monitor_feedback(
                decision="issue" if monitor_feedback.has_issue else "ok",
                feedback=self.monitor_feedback or ""
            )
            
            # Check for task completion - ONLY based on checklist's decision (should_stop)
            # Actor's STOP action alone doesn't terminate; checklist must confirm should_stop
            if should_stop:
                is_stop_action = executed_action.get("action_type") == ActionTypes.STOP
                completion_reason = "Task completed (validated by checklist)" if is_stop_action else \
                                  "Monitor detected task completion"
                print(f"🎉 {completion_reason}!")
                
                # Log monitor feedback and end step
                self.trajectory_logger.log_monitor_feedback(
                    decision="issue" if monitor_feedback.has_issue else "ok",
                    feedback=self.monitor_feedback or ""
                )
                self.trajectory_logger.end_step()
                
                return {
                    "should_terminate": True,
                    "termination_reason": completion_reason,
                    "step_number": step_number,
                    "context_result": context_result,
                    "execution_result": execution_result,
                    "reflection_result": reflection_result,
                    "new_observation": new_observation,
                    "monitor_feedback": self.monitor_feedback,
                }
        
        else:
            # === MODE B: Actor-Only Mode (No Context/Reflector) ===
            # Only Actor executes, no reflection or context updates
            # This is used when Monitor is disabled
            
            reflection_result = {}  # Empty reflection in actor-only mode
            self.monitor_feedback = None
            
            # Only check for STOP action to terminate
            is_stop_action = executed_action.get("action_type") == ActionTypes.STOP
            
            if is_stop_action:
                print(f"🎉 Task completed with STOP action!")
                # Log actor-only mode completion with full details
                llm_response = execution_result.get("llm_response", "") if execution_result else ""
                self.log_agent_response("actor_only", step_number, {
                    "mode": "actor_only",
                    "llm_response": llm_response,
                    "action_type": str(executed_action.get("action_type")),
                    "element_id": executed_action.get("element_id"),
                    "text": executed_action.get("text"),
                    "answer": executed_action.get("answer"),
                    "completed": True,
                })
                self.trajectory_logger.end_step()
                return {
                    "should_terminate": True,
                    "termination_reason": "Task completed with STOP action",
                    "step_number": step_number,
                    "context_result": context_result,
                    "execution_result": execution_result,
                    "reflection_result": reflection_result,
                    "new_observation": new_observation,
                }
            
            # Log actor action in actor-only mode with full details
            llm_response = execution_result.get("llm_response", "") if execution_result else ""
            extracted_intention = execution_result.get("extracted_intention", "") if execution_result else ""
            
            # Print detailed execution info to console in baseline mode
            print(f"📋 Baseline Mode - Step {step_number} Details:")
            print(f"   Action: {executed_action.get('action_type')} on element {executed_action.get('element_id', 'N/A')}")
            if executed_action.get("text"):
                text_preview = str(executed_action.get("text"))[:100]
                print(f"   Text: {text_preview}{'...' if len(str(executed_action.get('text', ''))) > 100 else ''}")
            if extracted_intention:
                print(f"   Intention: {extracted_intention[:150]}{'...' if len(extracted_intention) > 150 else ''}")
            if llm_response:
                response_preview = llm_response[:300].replace('\n', ' ')
                print(f"   LLM Response Preview: {response_preview}{'...' if len(llm_response) > 300 else ''}")
            
            self.log_agent_response("actor_only", step_number, {
                "mode": "actor_only",
                "llm_response": llm_response,
                "extracted_intention": extracted_intention,
                "action_type": str(executed_action.get("action_type")),
                "element_id": executed_action.get("element_id"),
                "text": executed_action.get("text"),
                "completed": False,
            })

        # 7. Update current step
        self.current_step = step_number
        
        # End trajectory logging for this step
        self.trajectory_logger.end_step()

        return {
            "should_terminate": False,
            "step_number": step_number,
            "context_result": context_result,
            "execution_result": execution_result,
            "reflection_result": reflection_result,
            "new_observation": new_observation,
            "monitor_feedback": self.monitor_feedback if self.enable_monitor else None,
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
        self.actor_agent.reset_intention_history()
        self.reflector_agent.reset_reflection_history()
        
        # Reset monitor feedback (important for baseline mode)
        self.monitor_feedback = None
        
        # Reset coordination components
        self.current_step = 0
        
        # Reset logging for new task
        self._setup_logging()
