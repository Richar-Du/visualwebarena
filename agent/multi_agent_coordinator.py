"""Multi-Agent Coordinator for managing collaborative agent execution."""

from typing import Any, Dict, List, Optional, Union
import json
import os
from datetime import datetime

from PIL import Image

from browser_env import Action, Trajectory, ActionTypes, create_stop_action
from browser_env.actions import is_equivalent
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
    """

    def __init__(self, lm_config: lm_config.LMConfig,
                 existing_prompt_agent,
                 browser_env=None,
                 result_dir: str = "results",
                 memory_config: Dict[str, Any]= {}) -> None:
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
        self.log_file_path = os.path.join(result_dir, "agent_responses.log")
        self._setup_logging()

        # Execution state
        self.trajectory: Trajectory = []
        self.intentions: List[str] = []
        self.actions: List[Action] = []
        self.reflections: List[Dict[str, Any]] = []

        # Early stop thresholds (can be provided via memory_config or use defaults)
        self.parsing_failure_th = memory_config.get("parsing_failure_th", 3)
        self.repeating_action_th = memory_config.get("repeating_action_th", 5)

        # Meta data for action history tracking (required by DirectPromptConstructor)
        # Initialize with "None" as the first action, matching run.py implementation
        self.meta_data: Dict[str, Any] = {"action_history": ["None"]}

        # Task configuration
        self.user_goal: str = ""
        self.max_steps: int = 30
        self.current_observation: Optional["ObservationTypeAlias"] = None

    def _setup_logging(self) -> None:
        """Setup logging for agent responses."""
        try:
            # Ensure result directory exists
            os.makedirs(self.result_dir, exist_ok=True)

            # Create or clear the log file
            with open(self.log_file_path, 'w', encoding='utf-8') as f:
                f.write(f"Multi-Agent Execution Log - Started at {datetime.now().isoformat()}\n")
                f.write("=" * 80 + "\n\n")
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
        # Initialize task
        self.user_goal = user_goal
        self.max_steps = max_steps
        self.current_observation = start_observation

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
                initial_info = start_observation["info"]
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
                "observation": initial_observation,
                "info": initial_info
            }
            self.trajectory.append(initial_state_info)
            self.current_observation = initial_observation
        
        # Main execution loop
        while True:
            try:
                # Early-stop check (mirrors run.py behavior)
                early_flag, early_info = self._early_stop(self.max_steps, {
                    "parsing_failure": self.parsing_failure_th,
                    "repeating_action": self.repeating_action_th,
                })
                if early_flag:
                    # Append a stop action to the trajectory and end execution
                    stop_action = create_stop_action(f"Early stop: {early_info}")
                    self.actions.append(stop_action)
                    self.trajectory.append(stop_action)
                    # Log and break
                    self.log_agent_response("coordinator", len(self.actions), {"early_stop": early_info})
                    break
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
                    break

                # Update current observation
                if step_result.get("new_observation"):
                    self.current_observation = step_result["new_observation"]

                
            except Exception as e:
                # Record error and continue
                break

        # Finalize execution
        final_context_summary = self._get_current_context_summary()
        workflow_final = self.workflow_manager.finalize_workflow(
            final_state="completed",
            completion_reason=continuation_decision.get("reason", "Execution completed")
        )

        # Log final execution summary - use context agent completion check
        task_completed = self.context_agent.check_task_completion(self.user_goal)

        if self.enable_memory_store:
            self.context_agent.generate_and_store_memory(task_completed)

        final_summary = {
            "total_steps_executed": len(self.actions),
            "task_completed": task_completed,
            "completion_percentage": "Completed" if task_completed else "In Progress",
            "total_intentions": len(self.intentions),
            "total_actions": len(self.actions),
            "total_reflections": len(self.reflections)
        }
        self.log_agent_response("execution_summary", len(self.actions), final_summary)

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
        }

    def _early_stop(self, max_steps: int, thresholds: Dict[str, int]) -> tuple[bool, str]:
        """Check whether need to stop early, similar to run.py's early_stop.

        Returns (flag, reason_str)
        """
        # reach the max step
        try:
            num_steps = (len(self.trajectory) - 1) / 2
        except Exception:
            num_steps = 0

        if num_steps >= max_steps:
            return True, f"Reach max steps {max_steps}"

        # Parsing failure check
        k = thresholds.get("parsing_failure", 3)
        if k > 0:
            last_k_actions = self.trajectory[1::2][-k:]
            if len(last_k_actions) >= k:
                if all([
                    (isinstance(action, dict) and action.get("action_type") == ActionTypes.NONE)
                    for action in last_k_actions
                ]):
                    return True, f"Failed to parse actions for {k} times"

        # Repeating action check
        k = thresholds.get("repeating_action", 5)
        action_seq = self.trajectory[1::2]
        if len(action_seq) == 0:
            return False, ""

        last_action = action_seq[-1]
        try:
            last_action_type = last_action.get("action_type")
        except Exception:
            last_action_type = None

        if last_action_type != ActionTypes.TYPE:
            if len(action_seq) >= k:
                last_k_actions = action_seq[-k:]
                if all([is_equivalent(action, last_action) for action in last_k_actions]):
                    return True, f"Same action for {k} times"
        else:
            # typing action: check frequency across full sequence
            try:
                count_same = sum([1 for action in action_seq if is_equivalent(action, last_action)])
                if count_same >= k:
                    return True, f"Same typing action for {k} times"
            except Exception:
                pass

        return False, ""

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
            print(f"🧠 Context: {summary[:100]}{'...' if len(summary) > 100 else ''}")

            # Store response information for execution summary

            # Log context agent response summary with detailed breakdown
            context_response = {
                "summary": summary,
                "observation_summary": context_result.get("observation_summary", ""),
                "action_summary": context_result.get("action_summary", ""),
                "reflection_summary": context_result.get("reflection_summary", "")
            }
            self.log_agent_response("context_agent", step_number, context_response)
            
        except Exception as e:
            print(f"🧠 Context Error: {str(e)[:100]}{'...' if len(str(e)) > 100 else ''}")
            context_result = {
                "summary": "Error generating context",
                "observation_summary": "",
                "action_summary": "",
                "reflection_summary": ""
            }

            # Log context agent error summary
            error_response = {
                "error": str(e),
                "summary": "Error generating context",
                "observation_summary": "",
                "action_summary": "",
                "reflection_summary": ""
            }
            self.log_agent_response("context_agent", step_number, error_response)


        # Check for task completion using context agent's completion check
        if self.context_agent.check_task_completion(self.user_goal):
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

            print(f"🎯 Current Subtask: {current_subtask[:100]}{'...' if len(current_subtask) > 100 else ''}")
            if next_atomic_action != current_subtask:
                print(f"🎯 Next Atomic Action: {next_atomic_action[:100]}{'...' if len(next_atomic_action) > 100 else ''}")
            # print(f"🎯 Progress: Step {current_step_index + 1}/{total_subtasks}")

            # Show all subtasks overview
            if all_subtasks and total_subtasks > 0:
                print(f"🎯 Task Overview ({total_subtasks} subtasks):")
                for i, subtask in enumerate(all_subtasks, 1):
                    status = "✅" if i <= current_step_index + 1 else "⏳"
                    print(f"   {status} {i}. {subtask[:80]}{'...' if len(subtask) > 80 else ''}")

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

            # Log planner agent error summary
            error_response = {
                "error": str(e),
                "intention": planning_result["intention"],
                "current_subtask": planning_result["current_subtask"],
                "next_atomic_action": planning_result["next_atomic_action"],
                "reasoning": planning_result["reasoning"],
                "task_decomposed": planning_result["task_decomposed"]
            }
            self.log_agent_response("planner_agent", step_number, error_response)

        # 3. Actor Agent executes intention
        print("🎬 Actor Agent: Executing intention...")
        try:
            # Merge step_number into meta_data while preserving action_history
            # This matches the pattern used in run.py
            meta_data_for_action = self.meta_data.copy()
            meta_data_for_action["step_number"] = step_number
            
            # Ensure current_observation has both text and image fields
            current_obs = self.current_observation or {"text": "", "image": None}
            if isinstance(current_obs, dict):
                if "text" not in current_obs:
                    current_obs["text"] = ""
                if "image" not in current_obs:
                    current_obs["image"] = None
            
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
                self.actions.append(executed_action)

                # Execute action in browser environment if available
                # If the executed action is a STOP action, mirror run.py behavior:
                # append the action to the trajectory and terminate execution cycle
                if executed_action.get("action_type") == ActionTypes.STOP:
                    print(f"🔍 Executing action in browser: {executed_action.get('action_type', 'UNKNOWN')}")
                    # update action history and trajectory so evaluators see Action as last element
                    action_str = f"STOP: {executed_action.get('answer', '')}"
                    try:
                        self.meta_data["action_history"].append(action_str)
                    except Exception:
                        self.meta_data["action_history"] = ["None", action_str]

                    # append the stop action to trajectory and return termination signal
                    self.trajectory.append(executed_action)
                    return {
                        "should_terminate": True,
                        "step_number": step_number,
                        "context_result": context_result,
                        "planning_result": planning_result,
                        "execution_result": execution_result,
                        "reflection_result": {},
                        "new_observation": None,
                    }

                if self.browser_env is not None:
                    try:
                        print(f"🔍 Executing action in browser: {executed_action.get('action_type', 'UNKNOWN')}")
                        obs, reward, terminated, truncated, info = self.browser_env.step(executed_action)

                        # Ensure observation has both text and image fields
                        if isinstance(obs, dict):
                            if "text" not in obs:
                                obs["text"] = ""
                            if "image" not in obs:
                                obs["image"] = None
                        else:
                            obs = {"text": str(obs) if obs else "", "image": None}

                        # Update current observation with new browser state
                        self.current_observation = obs  # Keep as observation format
                        new_observation = obs  # Keep as observation format

                        # Determine if intention is fulfilled based on execution success
                        intention_fulfilled = reward == 1.0  # reward is 1.0 for success, 0.0 for failure

                        # # Update trajectory with new state
                        # state_info = {"observation": obs, "info": info}
                        # self.trajectory.append(state_info)

                        print(f"✅ Browser execution successful - URL: {info.get('page', {}).url if 'page' in info else 'Unknown'}")

                        action_success = True
                    except Exception as e:
                        print(f"❌ Browser execution failed: {str(e)}")
                        intention_fulfilled = False
                        action_success = False
                        info = None
                        new_observation = self.current_observation
                else:
                    # No browser environment - simulate success for compatibility
                    intention_fulfilled = False  # Will be determined by reflection
                    action_success = True
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
                action_success = False
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
            action_success = False
            info = None
            new_observation = self.current_observation

            print(f"🎬 Actor Error: {str(e)[:100]}{'...' if len(str(e)) > 100 else ''}")


            # Store error response for execution summary, try to extract LLM response
            error_details = str(e)
            llm_response = "No LLM response available due to exception"

            # Try to get LLM response from execution_result if available
            if hasattr(execution_result, 'get') and execution_result.get("llm_response"):
                llm_response = execution_result.get("llm_response")
            elif hasattr(execution_result, 'get') and execution_result.get("response"):
                llm_response = execution_result.get("response")

            if hasattr(execution_result, 'get') and execution_result.get("error"):
                error_details = execution_result.get("error")
            elif hasattr(execution_result, 'get') and execution_result.get("exception_type"):
                error_details = f"{execution_result.get('exception_type', 'Exception')}: {error_details}"


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
        new_state_info = {
            "observation": new_observation,  # new_observation is now observation format
            "info": info
        }
        self.trajectory.append(new_state_info)

        # Update current observation to stay in sync with trajectory
        self.current_observation = new_observation
        
        # Update action_history in meta_data (required by DirectPromptConstructor)
        # For multi-agent simulation, use simplified action descriptions to avoid dependency on complex observation_metadata
        action_type = executed_action.get("action_type", "UNKNOWN")

        if action_type == "NONE":
            action_str = f"Failed action: {executed_action.get('error', 'Unknown error')}"
        elif action_type == "GOTO_URL":
            url = executed_action.get("url", "unknown")
            action_str = f"Navigate to {url}"
        elif action_type == "CLICK":
            element_id = executed_action.get("element_id", "unknown")
            action_str = f"Click on element {element_id}"
        elif action_type == "TYPE":
            element_id = executed_action.get("element_id", "unknown")
            text = executed_action.get("text", [""])[0] if executed_action.get("text") else ""
            action_str = f"Type '{text}' into element {element_id}"
        elif action_type == "SCROLL":
            direction = executed_action.get("direction", "unknown")
            action_str = f"Scroll {direction}"
        elif action_type == "HOVER":
            element_id = executed_action.get("element_id", "unknown")
            action_str = f"Hover over element {element_id}"
        else:
            action_str = f"Action: {action_type}"

        self.meta_data["action_history"].append(action_str)

        # 5. Reflector Agent reflects on execution
        reflection_result = self.reflector_agent.reflect_execution(
            trajectory=self.trajectory,
            intentions=self.intentions,
            actions=self.actions,
            current_intention=current_intention,
            latest_action=executed_action,
            current_observation=new_observation,
            context_summary=context_result,
        )

        self.reflections.append(reflection_result)

        # Show key reflection information
        success = reflection_result.get("success", False)
        helpful = reflection_result.get("helpful", False)
        stuck = reflection_result.get("stuck", False)
        effectiveness_result = reflection_result.get("effectiveness_analyzer", "")
        pattern_result = reflection_result.get("pattern_detector", "")
        triple_summary = reflection_result.get("triple_summary", "")


        # Log reflector agent response summary
        reflector_response = {
            "success": success,
            "helpful": helpful,
            "stuck": stuck,
            "effectiveness_result": effectiveness_result,
            "pattern_result": pattern_result,
            "triple_summary": triple_summary
        }
        self.log_agent_response("reflector_agent", step_number, reflector_response)

        # 6. Record workflow step
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


    def _get_current_context_summary(self) -> Dict[str, Any]:
        """Get current context summary without full update."""
        return self.context_agent.get_current_state()


  