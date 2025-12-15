"""Multi-Agent Web Arena Runner.

This module implements a multi-agent system for web automation tasks,
using Context, Planner, Actor, and Reflector agents.
"""

import argparse
import json
import os
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import openai
from browser_env import (
    ScriptBrowserEnv,
    Action,
    create_id_based_action,
    ActionTypes,
    StateInfo,
    Trajectory,
    create_stop_action,
)
from browser_env.utils import Observation
from agent import PromptAgent
from llms import lm_config, call_llm
from PIL import Image
from evaluation_harness import image_utils, evaluator_router
from browser_env.helper_functions import (
    RenderHelper,
    get_action_description,
)
from browser_env.auto_login import get_site_comb_from_filepath
# Import the full multi-agent coordinator
from agent.multi_agent_coordinator import MultiAgentCoordinator
from agent.prompts.prompt_constructor import PromptConstructor, DirectPromptConstructor, CoTPromptConstructor, MultimodalCoTPromptConstructor
from llms import lm_config

DATASET = os.environ["DATASET"]

LOG_FOLDER = "log_files"
Path(LOG_FOLDER).mkdir(parents=True, exist_ok=True)
LOG_FILE_NAME = f"{LOG_FOLDER}/log_{time.strftime('%Y%m%d%H%M%S', time.localtime())}.log"

logger = logging.getLogger("logger")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

file_handler = logging.FileHandler(LOG_FILE_NAME)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

# Set the log format
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

def generate_execution_summary(execution_result: Dict[str, Any]) -> str:
    """Generate a human-readable execution summary from execution result.

    Args:
        execution_result: The execution result from multi-agent coordinator

    Returns:
        Human-readable summary string
    """
    lines = []
    lines.append("Multi-Agent Web Automation Execution Summary")
    lines.append("=" * 50)
    lines.append("")

    # Task information
    if "goal" in execution_result:
        goal = execution_result["goal"]
        lines.append("Task Information:")
        lines.append(f"  Goal: {goal}")
        lines.append(f"  Completed: {'Yes' if execution_result.get('success_rate', 0) >= 0.8 else 'No'}")
        lines.append(f"  Completion: {execution_result.get('success_rate', 0) * 100:.1f}%")
        lines.append(f"  Steps Executed: {execution_result.get('total_steps', 0)}")
        if 'execution_time_formatted' in execution_result:
            lines.append(f"  Execution Time: {execution_result['execution_time_formatted']}")
        lines.append("")

    # Agent performance
    if 'reflections' in execution_result and execution_result['reflections']:
        lines.append("Agent Performance:")

        # Count successful actions
        actions = execution_result.get('actions', [])
        successful_actions = sum(1 for action in actions if action.get('action_type') != 'NONE')
        total_actions = len(actions)

        lines.append(f"  actor_agent:")
        lines.append(f"    total_intentions: {len(execution_result.get('intentions', []))}")
        lines.append(f"    successful_actions: {successful_actions}")
        lines.append(f"    failed_actions: {total_actions - successful_actions}")
        if total_actions > 0:
            fulfillment_rate = successful_actions / total_actions
            lines.append(f"    fulfillment_rate: {fulfillment_rate * 100:.1f}%")

        # Count reflection success
        reflections = execution_result['reflections']
        successful_reflections = sum(1 for reflection in reflections if reflection.get('success', False))
        helpful_reflections = sum(1 for reflection in reflections if reflection.get('helpful', False))
        stuck_reflections = sum(1 for reflection in reflections if reflection.get('stuck', False))

        lines.append(f"  reflector_agent:")
        lines.append(f"    total_reflections: {len(reflections)}")
        lines.append(f"    successful_reflections: {successful_reflections}")
        lines.append(f"    helpful_reflections: {helpful_reflections}")
        lines.append(f"    stuck_reflections: {stuck_reflections}")
        if len(reflections) > 0:
            success_rate = successful_reflections / len(reflections)
            helpful_rate = helpful_reflections / len(reflections)
            lines.append(f"    success_rate: {success_rate * 100:.1f}%")
            lines.append(f"    helpful_rate: {helpful_rate * 100:.1f}%")
            lines.append(f"    stuck_rate: {stuck_reflections * 100:.1f}%")

        lines.append("")

    # Overall assessment
    lines.append("Overall Assessment:")
    success_rate = execution_result.get('success_rate', 0)
    lines.append(f"  Success: {'Yes' if success_rate >= 0.8 else 'No'}")
    lines.append(f"  Success Rate: {success_rate * 100:.1f}%")
    lines.append(f"  Total Actions: {len(execution_result.get('actions', []))}")
    lines.append(f"  Total Steps: {execution_result.get('total_steps', 0)}")
    lines.append("")

    return "\n".join(lines)


def config():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Multi-Agent Web Arena Runner")

    # Required arguments
    parser.add_argument("--config_file", type=str, required=True,
                       help="Path to JSON configuration file")

    # Commonly overridden arguments (for convenience)
    parser.add_argument("--start_url", type=str,
                       help="Override starting URL (overrides config file)")
    parser.add_argument("--intent", type=str,
                       help="Override task intent (overrides config file)")
    parser.add_argument("--max_steps", type=int, default=30,
                       help="Override maximum steps (overrides config file)")
    parser.add_argument("--result_dir", type=str,
                       help="Override result directory (overrides config file)")

    # Debugging options
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output (overrides config file)")
    parser.add_argument("--dry_run", action="store_true",
                       help="Show configuration without executing")

    # eval
    parser.add_argument("--test_config_base_dir", type=str)
    parser.add_argument("--test_start_idx", type=int, default=0, help="Start index (inclusive) for test config files")
    parser.add_argument("--test_end_idx", type=int, default=1, help="End index (exclusive) for test config files")
    parser.add_argument("--eval_captioning_model_device", type=str, default="cpu")
    parser.add_argument("--eval_captioning_model", type=str, default="qwen3-vl-plus")
    parser.add_argument("--captioning_model", type=str, default="qwen3-vl-plus")
    
    return parser.parse_args()

def load_config_file(config_file: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as e:
        return {}


def merge_config_with_args(config: Dict[str, Any], args) -> Dict[str, Any]:
    """Merge loaded config with command line arguments."""
    merged = config.copy()

    # Only process a simplified set of command line arguments
    for key, value in vars(args).items():
        if value is not None and key not in ["config_file", "dry_run"]:
            # Handle overrides for commonly changed parameters
            if key in ["start_url", "intent", "max_steps"]:
                if "task" not in merged:
                    merged["task"] = {}
                merged["task"][key] = value
            elif key in ["result_dir", "verbose"]:
                if "output" not in merged:
                    merged["output"] = {}
                merged["output"][key] = value
            else:
                # Direct override for any other arguments
                merged[key] = value

    return merged


def test(args, test_file_list):
    """Run the multi-agent system."""
    config_file = args.config_file
    # Load and merge configuration
    file_config = load_config_file(config_file)
    config = merge_config_with_args(file_config, args)

    # Handle dry run
    if args.dry_run:
        return

    # Setup result directory
    result_dir = config.get('output', {}).get('result_dir', 'results')
    if not Path(result_dir).exists():
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        print(f"Created result directory: {result_dir}")

    # Add result_dir to config for coordinator
    if 'output' not in config:
        config['output'] = {}
    config['output']['result_dir'] = result_dir

    # Import the full multi-agent coordinator
    from agent.multi_agent_coordinator import MultiAgentCoordinator
    from agent import PromptAgent
    from agent.prompts.prompt_constructor import PromptConstructor
    from llms import lm_config

    # Create LM config
    try:
        # Extract model config from the config dictionary
        model_config = config.get('model', {})

        # Create LMConfig directly from the dictionary
        lm_cfg = lm_config.LMConfig(
            provider=model_config.get('provider', 'openai'),
            model=model_config.get('model', 'gpt-4'),
            mode=model_config.get('mode', 'chat')
        )

        # Add generation config if available
        if model_config:
            lm_cfg.gen_config.update({
                'temperature': model_config.get('temperature', 1.0),
                'top_p': model_config.get('top_p', 0.9),
                'max_tokens': model_config.get('max_tokens', 384),
                'context_length': model_config.get('context_length', 0),
                'stop_token': model_config.get('stop_token', None),
                'max_obs_length': model_config.get('max_obs_length', 0),
                'max_retry': model_config.get('max_retry', 3)
            })
    except (KeyError, AttributeError) as e:
        # Fallback to minimal config if required fields missing
        lm_cfg = lm_config.LMConfig(
            provider=config.get('model', {}).get('provider', 'openai'),
            model=config.get('model', {}).get('model', 'gpt-4'),
            mode=config.get('model', {}).get('mode', 'chat')
        )

    # Get browser environment configuration
    browser_config = config.get('browser', {})
    observation_type = browser_config.get('observation_type', 'accessibility_tree')
    
    # Load captioning model if needed (similar to run.py)
    caption_image_fn = None
    if observation_type in [
        "accessibility_tree_with_captioner",
        "image_som",
    ]:
        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        captioning_model = config.get('model', {}).get('captioning_model', 'Salesforce/blip2-flan-t5-xl')
        caption_image_fn = image_utils.get_captioning_fn(
            device, dtype, captioning_model
        )
        
    eval_caption_image_fn = None
    if not eval_caption_image_fn:
        eval_caption_image_fn = image_utils.get_captioning_fn(
            args.eval_captioning_model_device,
            torch.float16 if torch.cuda.is_available() and args.eval_captioning_model_device == "cuda" else torch.float32,
            args.eval_captioning_model,
        )

    # Build viewport_size from config
    viewport_size = {
        "width": browser_config.get('viewport_width', 1280),
        "height": browser_config.get('viewport_height', 720),
    }

    # Create browser environment
    env = ScriptBrowserEnv(
        headless=browser_config.get('headless', False),  # Set to False for debugging
        slow_mo=browser_config.get('slow_mo', 100),
        observation_type=observation_type,
        current_viewport_only=browser_config.get('current_viewport_only', True),
        viewport_size=viewport_size,
        save_trace_enabled=browser_config.get('save_trace_enabled', True),
        sleep_after_execution=browser_config.get('sleep_after_execution', 0.5),
        captioning_fn=caption_image_fn,
    )

    # Determine if model is multimodal and select appropriate prompt constructor
    from llms.tokenizers import Tokenizer
    
    model_name = lm_cfg.model.lower()
    is_multimodal_model = (
        "gemini" in model_name or 
        ("gpt-4" in model_name and "vision" in model_name)
    )
    is_image_observation = observation_type in ["image", "image_som"]
    
    # Get instruction path from config or use default
    instruction_path = config.get('instruction_path')
    if not instruction_path:
        # Select default instruction path based on observation type and model
        if is_multimodal_model and is_image_observation:
            instruction_path = 'agent/prompts/jsons/p_multimodal_cot_id_actree_3s.json'
        else:
            instruction_path = 'agent/prompts/jsons/p_cot_id_actree_3s.json'
    
    # Load instruction to check prompt_constructor type
    with open(instruction_path) as f:
        instruction_data = json.load(f)
        constructor_type = instruction_data.get("meta_data", {}).get("prompt_constructor", "DirectPromptConstructor")
    
    # Create appropriate prompt constructor
    tokenizer = Tokenizer(lm_cfg.provider, lm_cfg.model)
    if constructor_type == "MultimodalCoTPromptConstructor":
        prompt_constructor = MultimodalCoTPromptConstructor(
            instruction_path=instruction_path,
            lm_config=lm_cfg,
            tokenizer=tokenizer
        )
    elif constructor_type == "CoTPromptConstructor":
        prompt_constructor = CoTPromptConstructor(
            instruction_path=instruction_path,
            lm_config=lm_cfg,
            tokenizer=tokenizer
        )
    else:
        prompt_constructor = DirectPromptConstructor(
            instruction_path=instruction_path,
            lm_config=lm_cfg,
            tokenizer=tokenizer
        )

    # Create base prompt agent for multi-agent coordinator
    # Use action_set_tag from configuration instead of hardcoding
    action_set_tag = config.get('observation', {}).get('action_set_tag', 'id_accessibility_tree')
    base_agent = PromptAgent(
        action_set_tag=action_set_tag,
        lm_config=lm_cfg,
        prompt_constructor=prompt_constructor,
        captioning_fn=caption_image_fn if observation_type == "accessibility_tree_with_captioner" else None,
    )

    # Create multi-agent coordinator with browser environment
    coordinator = MultiAgentCoordinator(lm_cfg,
                                        base_agent,
                                        browser_env=env,
                                        result_dir=result_dir,
                                        memory_config=config.get('memory', {}))

    # Execute workflow with initial observation from browser
    # Use start_url from config if available
    import tempfile
    import subprocess
    import requests
    
    scores = []
    for cfg_file in test_file_list:
        try:
            render_helper = RenderHelper(cfg_file, result_dir, action_set_tag)
            
            with open(cfg_file) as f:
                _c = json.load(f)
                intent = _c["intent"]
                task_id = _c["task_id"]
                start_url = _c["start_url"]
                image_paths = _c.get("image", None)
                images = []

                # 处理自动登录 (Auto Login Cookie 替换)
                if _c.get("storage_state"): # 使用 .get 防止 None 报错
                    cookie_file_name = os.path.basename(_c["storage_state"])
                    comb = get_site_comb_from_filepath(cookie_file_name)
                    temp_dir = tempfile.mkdtemp()
                    # 调用子进程刷新 cookie
                    subprocess.run(
                        [
                            "python", "browser_env/auto_login.py",
                            "--auth_folder", temp_dir,
                            "--site_list", *comb,
                        ]
                    )
                    _c["storage_state"] = f"{temp_dir}/{cookie_file_name}"
                    assert os.path.exists(_c["storage_state"])
                    # update the config file
                    cfg_file = f"{temp_dir}/{os.path.basename(cfg_file)}"
                    with open(cfg_file, "w") as f:
                        json.dump(_c, f)
                        
                # Load input images for the task, if any.      
                if image_paths is not None:
                    if isinstance(image_paths, str):
                        image_paths = [image_paths]
                    for image_path in image_paths:
                        # Load image either from the web or from a local path.
                        if image_path.startswith("http"):
                            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
                            input_image = Image.open(requests.get(image_path, stream=True, headers = headers).raw)
                        else:
                            input_image = Image.open(image_path)

                        images.append(input_image)

            logger.info(f"[Config file]: {cfg_file}")
            logger.info(f"[Processing Task]: {task_id} - {intent}")
            
            base_agent.reset(cfg_file)
            trajectory: Trajectory = []
            obs, info = env.reset(options={"config_file": cfg_file})
            state_info: StateInfo = {"observation": obs, "info": info}
            trajectory.append(state_info)
            meta_data = {"action_history": ["None"]}
            
            res = coordinator.execute_task(
                user_goal=intent,
                start_observation={"observation": obs, "info": info},
                max_steps=args.max_steps
            )
            
            logger.info("Evaluating trajectory...")

            # --- Dump full trajectory for debugging (sanitized) ---
            try:
                traj = coordinator.trajectory
                logger.info(f"Raw trajectory length: {len(traj)}")

                def _serialize_element(el):
                    # StateInfo
                    if isinstance(el, dict) and 'observation' in el:
                        obs = el.get('observation')
                        info = el.get('info', {}) or {}
                        page = info.get('page') if isinstance(info, dict) else None
                        url = getattr(page, 'url', None) if page is not None else None
                        if isinstance(obs, dict):
                            text = obs.get('text', '')
                            has_image = obs.get('image') is not None
                        else:
                            text = str(obs) if obs is not None else ''
                            has_image = False
                        return {
                            'type': 'StateInfo',
                            'text_preview': (text[:500] + '...') if len(text) > 500 else text,
                            'has_image': bool(has_image),
                            'page_url': url,
                        }

                    # Action
                    if isinstance(el, dict) and 'action_type' in el:
                        try:
                            at_name = ActionTypes(el.get('action_type')).name
                        except Exception:
                            at_name = str(el.get('action_type'))
                        return {
                            'type': 'Action',
                            'action_type': at_name,
                            'element_id': el.get('element_id'),
                            'answer_preview': (el.get('answer')),
                            'raw_prediction_preview': (el.get('raw_prediction')),
                        }

                    # Fallback
                    try:
                        return {'type': str(type(el)), 'repr': str(el)[:200]}
                    except Exception:
                        return {'type': str(type(el)), 'repr': '<unserializable>'}

                simple_traj = [_serialize_element(x) for x in traj]
                trace_dir = Path(result_dir) / 'traces'
                trace_dir.mkdir(parents=True, exist_ok=True)
                traj_path = trace_dir / f"{task_id}_full_trajectory.json"
                with open(traj_path, 'w', encoding='utf-8') as _f:
                    json.dump(simple_traj, _f, ensure_ascii=False, indent=2)
                # logger.info(f"Saved sanitized full trajectory to {traj_path}")

                # # Print a concise preview to console (first 10 and last 5 entries)
                # preview_head = simple_traj[:10]
                # preview_tail = simple_traj[-5:] if len(simple_traj) > 5 else []
                # logger.info("Trajectory preview (head):\n" + json.dumps(preview_head, ensure_ascii=False, indent=2))
                # if preview_tail:
                #     logger.info("Trajectory preview (tail):\n" + json.dumps(preview_tail, ensure_ascii=False, indent=2))
            except Exception as e:
                logger.info(f"Failed to serialize/print trajectory: {e}")

            evaluator = evaluator_router(
                cfg_file, 
                captioning_fn=eval_caption_image_fn
            )
            
            # 执行评估
            # TODO：需要传入完整的轨迹 trajectory, 配置文件, 和当前的 page 对象(用于截图/获取最终DOM)
            score = evaluator(
                trajectory=coordinator.trajectory,
                config_file=cfg_file,
                page=env.page
            )
            
            scores.append(score)
            
            result_status = "PASS" if score == 1 else "FAIL"
            logger.info(f"[Result] ({result_status}) Task: {task_id}, Score: {score}")

            if args.save_trace_enabled:
                trace_path = Path(args.result_dir) / "traces" / f"{task_id}.zip"
                trace_path.parent.mkdir(parents=True, exist_ok=True)
                env.save_trace(trace_path)
                
        except openai.OpenAIError as e:
            logger.info(f"[OpenAI Error] {repr(e)}")
        except Exception as e:
            logger.info(f"[Unhandled Error] {repr(e)}]")
            import traceback

            # write to error file
            with open(Path(args.result_dir) / "error.txt", "a") as f:
                f.write(f"[Config file]: {cfg_file}\n")
                f.write(f"[Unhandled Error] {repr(e)}\n")
                f.write(traceback.format_exc())  # write stack trace to file

    env.close()
    if len(scores):
        logger.info(f"Average score: {sum(scores) / len(scores)}")

def prepare(args: argparse.Namespace) -> None:
    # convert prompt python files to json
    from agent.prompts import to_json

    to_json.run()

    # prepare result dir
    result_dir = args.result_dir
    if not result_dir:
        result_dir = (
            f"cache/results_{time.strftime('%Y%m%d%H%M%S', time.localtime())}"
        )
    if not Path(result_dir).exists():
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        args.result_dir = result_dir
        logger.info(f"Create result dir: {result_dir}")

    if not (Path(result_dir) / "traces").exists():
        (Path(result_dir) / "traces").mkdir(parents=True)

    # log the log file
    with open(os.path.join(result_dir, "log_files.txt"), "a+") as f:
        f.write(f"{LOG_FILE_NAME}\n")

import glob
def get_unfinished(config_files: list[str], result_dir: str) -> list[str]:
    result_files = glob.glob(f"{result_dir}/*.html")
    task_ids = [
        os.path.basename(f).split(".")[0].split("_")[1] for f in result_files
    ]
    unfinished_configs = []
    for config_file in config_files:
        task_id = os.path.basename(config_file).split(".")[0]
        if task_id not in task_ids:
            unfinished_configs.append(config_file)
    return unfinished_configs


def dump_config(args: argparse.Namespace) -> None:
    config_file = Path(args.result_dir) / "config.json"
    if not config_file.exists():
        with open(config_file, "w") as f:
            json.dump(vars(args), f, indent=4)
            logger.info(f"Dump config to {config_file}")

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    args = config()
    args.sleep_after_execution = 2.5
    prepare(args)

    test_config_base_dir = args.test_config_base_dir

    test_file_list = []
    st_idx = args.test_start_idx
    ed_idx = args.test_end_idx
    for i in range(st_idx, ed_idx):
        test_file_list.append(os.path.join(test_config_base_dir, f"{i}.json"))
    test_file_list = get_unfinished(test_file_list, args.result_dir)
    print(f"Total {len(test_file_list)} tasks left")
    args.render = False
    args.render_screenshot = True
    args.save_trace_enabled = True

    args.current_viewport_only = True
    dump_config(args)

    test(args, test_file_list)