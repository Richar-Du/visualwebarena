"""Multi-Agent Web Arena Runner.

This module implements a multi-agent system for web automation tasks,
using Context, Planner, Actor, and Reflector agents.
"""

import argparse
import json
import os
import requests
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

import traceback

from browser_env import (
    ScriptBrowserEnv,
    Action,
    create_id_based_action,
)
from browser_env.utils import Observation
from agent import PromptAgent
from llms import lm_config, call_llm
from PIL import Image
from evaluation_harness import image_utils

# Import the full multi-agent coordinator
from agent.multi_agent_coordinator import MultiAgentCoordinator
from agent.prompts.prompt_constructor import PromptConstructor, DirectPromptConstructor, CoTPromptConstructor, MultimodalCoTPromptConstructor
from llms import lm_config


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
    parser.add_argument("--task_list_file", type=str,
                       help="Path to JSON task list (e.g., Online_Mind2Web.json)")

    # Commonly overridden arguments (for convenience)
    parser.add_argument("--start_url", type=str,
                       help="Override starting URL (overrides config file)")
    parser.add_argument("--intent", type=str,
                       help="Override task intent (overrides config file)")
    parser.add_argument("--max_steps", type=int,
                       help="Override maximum steps (overrides config file)")
    parser.add_argument("--result_dir", type=str,
                       help="Override result directory (overrides config file)")

    # Debugging options
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output (overrides config file)")
    parser.add_argument("--dry_run", action="store_true",
                       help="Show configuration without executing")
    parser.add_argument("--start_id", type=int,
                       help="the start id of the task list")
    parser.add_argument("--end_id", type=int,
                       help="the end id of the task list") #设置起始di和终止id

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
            elif key == "task_list_file":
                merged["task_list_file"] = value
            else:
                # Direct override for any other arguments
                merged[key] = value

    return merged


def test(args, config_file):
    """Run the multi-agent system."""

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
    config['output']['result_dir'] = result_dir  #因为result_dir是从命令行中读取进来的

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
    observation_type = config.get('observation', {}).get('observation_type', 'accessibility_tree')
    
    # Load captioning model if needed (similar to run.py)
    caption_image_fn = None
    if observation_type in [
        "accessibility_tree_with_captioner",
    ]:
        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        captioning_model = config.get('model', {}).get('captioning_model', 'Salesforce/blip2-flan-t5-xl')
        caption_image_fn = image_utils.get_captioning_fn(
            device, dtype, captioning_model
        )

    # Build viewport_size from config
    viewport_size = {
        "width": browser_config.get('viewport_width', 1280),
        "height": browser_config.get('viewport_height', 720),
    }

    # Get observation config for browser parameters
    observation_config = config.get('observation', {})
    output_config = config.get('output', {})

    # Create browser environment
    env = ScriptBrowserEnv(
        headless=browser_config.get('headless', False),  # Set to False for debugging
        slow_mo=browser_config.get('slow_mo', 100),
        observation_type=observation_type,
        current_viewport_only=observation_config.get('current_viewport_only', True),
        viewport_size=viewport_size,
        save_trace_enabled=output_config.get('save_trace_enabled', True),
        sleep_after_execution=browser_config.get('sleep_after_execution', 0.5),
        captioning_fn=caption_image_fn,
    )

    # Determine if model is multimodal and select appropriate prompt constructor
    from llms.tokenizers import Tokenizer
    
    model_name = lm_cfg.model.lower()
    is_multimodal_model = (
        "gemini" in model_name or 
        ("gpt-4" in model_name and "vision" in model_name) or
        ("gpt-4o" in model_name) or (True)
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

    task_cfg = config.get('task', {})
    output_cfg = config.get('output', {})
    task_metadata_base = config.get('task_metadata') or {
        "task_id": task_cfg.get('task_id'),
        "task": task_cfg.get('intent'),
        "confirmed_task": task_cfg.get('confirmed_task'),
        "website": task_cfg.get('website') or task_cfg.get('start_url'),
        "reference_length": task_cfg.get('reference_length'),
        "level": task_cfg.get('level'),
    }
    task_metadata_base = {k: v for k, v in task_metadata_base.items() if v is not None}  # 只保留有值的元数据
    webjudge_root = output_cfg.get('webjudge_root')  # 可自定义 WebJudge 输出根目录

    # 任务列表模式：从 task_list_file 读取并依次执行
    task_list_path = config.get("task_list_file")
    task_list = []
    if task_list_path:
        with open(task_list_path, "r", encoding="utf-8") as f:
            task_list = json.load(f)
        print(f"Loaded {len(task_list)} tasks from {task_list_path}")

    def run_single_task(single_task_meta: Dict[str, Any]):
        # 将列表里的字段映射到运行所需的 task 配置与元数据
        per_task_cfg = config.get('task', {}).copy() if isinstance(config.get('task', {}), dict) else {}
        per_task_cfg['intent'] = single_task_meta.get('confirmed_task') or single_task_meta.get('task') or per_task_cfg.get('intent', 'Not specified')
        per_task_cfg['start_url'] = single_task_meta.get('website') or per_task_cfg.get('start_url')
        per_task_cfg['task_id'] = single_task_meta.get('task_id')
        per_task_cfg['reference_length'] = single_task_meta.get('reference_length')
        per_task_cfg['level'] = single_task_meta.get('level')
        # 如未指定 max_steps，则尝试用 reference_length 作为上限
        if per_task_cfg.get('max_steps') is None and single_task_meta.get('reference_length'):
            per_task_cfg['max_steps'] = single_task_meta['reference_length']

        # 合成任务元数据
        tm = task_metadata_base.copy()
        tm.update({k: v for k, v in single_task_meta.items() if v is not None})
        tm['task'] = per_task_cfg.get('intent')

        # 创建新的协调器以清空内部轨迹
        coordinator = MultiAgentCoordinator(lm_cfg,
                                            base_agent,
                                            browser_env=env,
                                            result_dir=result_dir,
                                            memory_config=config.get('memory', {}),
                                            webjudge_result_root=webjudge_root)

        # 加载输入图片（若有）
        image_paths = per_task_cfg.get('image')
        images = []
        if image_paths is not None:
            if isinstance(image_paths, str):
                image_paths = [image_paths]
            for image_path in image_paths:
                if image_path.startswith("http"):
                    input_image = Image.open(requests.get(image_path, stream=True).raw)
                else:
                    input_image = Image.open(image_path)
                images.append(input_image)

        # 重置浏览器到指定起始页
        start_url = per_task_cfg.get('start_url')
        reset_options = {"start_url": start_url} if start_url else None
        initial_obs, initial_info = env.reset(options=reset_options)
        initial_observation = {"observation": initial_obs, "info": initial_info}

        # 执行任务
        return coordinator.execute_task(
            user_goal=per_task_cfg.get('intent', 'Not specified'),
            start_observation=initial_observation,
            max_steps=per_task_cfg.get('max_steps', 3),
            images=images if images else None,
            task_metadata=tm,  # 传递任务元信息供落盘
            webjudge_root=webjudge_root  # 指定 WebJudge 输出根目录
        )

    # 若提供任务列表则顺序执行，否则执行单任务
    if task_list:
        results = []
        for idx, t in enumerate(task_list):
            if idx < args.start_id or idx >= args.end_id:
                continue
            print(f"Running task {idx+1}/{len(task_list)}: {t.get('task_id')}")
            try:
                results.append(run_single_task(t))
            except Exception as e:
                print(f"Error: {e}")
                traceback.print_exc()
                results.append(None)
        return results
    else:
        return run_single_task(task_metadata_base)


if __name__ == "__main__":
    args = config()
    try:
        test(args, args.config_file)
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()