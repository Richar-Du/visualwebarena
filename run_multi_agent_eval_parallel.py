"""Parallel Multi-Agent Web Arena Runner.

This module implements a parallel evaluation system for multi-agent web automation tasks,
using multiprocessing to run multiple evaluations concurrently.
"""

import argparse
import json
import os
import logging
import time
import multiprocessing as mp
from multiprocessing import Queue, Manager
from queue import Empty
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import traceback

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
from agent.multi_agent_coordinator import MultiAgentCoordinator
from agent.prompts.prompt_constructor import (
    PromptConstructor,
    DirectPromptConstructor,
    CoTPromptConstructor,
    MultimodalCoTPromptConstructor
)

DATASET = os.environ.get("DATASET", "vwa")

LOG_FOLDER = "log_files"
Path(LOG_FOLDER).mkdir(parents=True, exist_ok=True)
LOG_FILE_NAME = f"{LOG_FOLDER}/log_{time.strftime('%Y%m%d%H%M%S', time.localtime())}_parallel.log"


def setup_worker_logger(worker_id: int, result_dir: str):
    """Setup logger for each worker process."""
    worker_logger = logging.getLogger(f"worker_{worker_id}")
    worker_logger.setLevel(logging.INFO)
    
    # Create worker-specific log file
    worker_log_file = Path(result_dir) / f"worker_{worker_id}.log"
    file_handler = logging.FileHandler(worker_log_file)
    file_handler.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter("%(asctime)s - [Worker %(name)s] - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    
    worker_logger.addHandler(file_handler)
    
    return worker_logger


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
        if value is not None and key not in ["config_file", "dry_run", "num_workers"]:
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


def evaluate_single_task(
    task_info: Tuple[str, str],
    args_dict: Dict[str, Any],
    config: Dict[str, Any],
    worker_id: int,
    result_dir: str
) -> Dict[str, Any]:
    """Evaluate a single task in a worker process.
    
    Args:
        task_info: Tuple of (config_file_path, task_name)
        args_dict: Dictionary of command line arguments
        config: Configuration dictionary
        worker_id: Worker process ID
        result_dir: Result directory path
        
    Returns:
        Dictionary containing task result information
    """
    # Ensure no asyncio event loop exists before each task evaluation
    import asyncio
    try:
        loop = asyncio.get_running_loop()
        # If we can get a running loop, that's the problem
        raise RuntimeError("Detected running asyncio loop in task evaluation")
    except RuntimeError:
        # Good - no running loop
        pass
    
    cfg_file, task_name = task_info
    
    # Setup worker logger
    worker_logger = setup_worker_logger(worker_id, result_dir)
    
    try:
        # Load config to get task_id
        with open(cfg_file) as f:
            _c = json.load(f)
            task_id = str(_c["task_id"])
            intent = _c["intent"]
            start_url = _c["start_url"]

        worker_logger.info(f"Starting task {task_name}_{task_id}: {intent}")

        # Create task-specific result directory
        result_folder_name = f"{task_name}_{task_id}"
        task_result_dir = os.path.join(result_dir, result_folder_name)
        if not Path(task_result_dir).exists():
            Path(task_result_dir).mkdir(parents=True, exist_ok=True)

        # Create LM config
        model_config = config.get('model', {})
        lm_cfg = lm_config.LMConfig(
            provider=model_config.get('provider', 'openai'),
            model=model_config.get('model', 'gpt-4'),
            mode=model_config.get('mode', 'chat')
        )

        # Add generation config
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

        # Get browser environment configuration
        browser_config = config.get('browser', {})
        observation_type = config.get('observation', {}).get('observation_type', 'accessibility_tree')
        
        # Load captioning model if needed
        caption_image_fn = None
        if observation_type in ["accessibility_tree_with_captioner"]:
            device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            captioning_model = config.get('model', {}).get('captioning_model', 'Salesforce/blip2-flan-t5-xl')
            caption_image_fn = image_utils.get_captioning_fn(device, dtype, captioning_model)

        # Build viewport_size from config
        viewport_size = {
            "width": browser_config.get('viewport_width', 1280),
            "height": browser_config.get('viewport_height', 720),
        }

        # Create browser environment (each worker gets its own)
        env = ScriptBrowserEnv(
            headless=browser_config.get('headless', True),  # Use headless for parallel execution
            slow_mo=browser_config.get('slow_mo', 100),
            observation_type=observation_type,
            current_viewport_only=config.get('observation', {}).get('current_viewport_only', True),
            viewport_size=viewport_size,
            save_trace_enabled=browser_config.get('save_trace_enabled', True),
            sleep_after_execution=browser_config.get('sleep_after_execution', 0.5),
            captioning_fn=caption_image_fn,
        )

        # Determine if model is multimodal and select appropriate prompt constructor
        from llms.tokenizers import Tokenizer
        from agent.utils import is_multimodal_model

        model_name = lm_cfg.model.lower()
        is_multimodal = is_multimodal_model(lm_cfg.model)
        is_image_observation = observation_type in ["image", "image_som"]

        # Validate multimodal requirements
        if observation_type == "image_som" and not is_multimodal:
            raise ValueError(f"Model '{lm_cfg.model}' does not support multimodal inputs")

        # Get instruction path from config or use default
        instruction_path = config.get('instruction_path')
        if not instruction_path:
            if is_multimodal and is_image_observation:
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

        # Create base prompt agent
        action_set_tag = config.get('observation', {}).get('action_set_tag', 'id_accessibility_tree')
        base_agent = PromptAgent(
            action_set_tag=action_set_tag,
            lm_config=lm_cfg,
            prompt_constructor=prompt_constructor,
            captioning_fn=caption_image_fn if observation_type == "accessibility_tree_with_captioner" else None,
        )

        # Create render helper
        render_helper = RenderHelper(cfg_file, task_result_dir, action_set_tag)

        # Create multi-agent coordinator
        coordinator = MultiAgentCoordinator(
            lm_cfg,
            base_agent,
            browser_env=env,
            result_dir=task_result_dir,
            memory_config=config.get('memory', {}),
            clear_result_dir=config.get('output', {}).get('clear_result_dir', False),
            save_images=config.get('output', {}).get('save_images', True)
        )

        # Handle auto-login and load task config
        import tempfile
        import subprocess
        import requests

        with open(cfg_file) as f:
            _c = json.load(f)
            intent = _c["intent"]
            start_url = _c["start_url"]
            image_paths = _c.get("image", None)
            images = []

            # Handle auto-login
            if _c.get("storage_state"):
                cookie_file_name = os.path.basename(_c["storage_state"])
                comb = get_site_comb_from_filepath(cookie_file_name)
                temp_dir = tempfile.mkdtemp()
                subprocess.run(
                    [
                        "python", "browser_env/auto_login.py",
                        "--auth_folder", temp_dir,
                        "--site_list", *comb,
                    ]
                )
                _c["storage_state"] = f"{temp_dir}/{cookie_file_name}"
                assert os.path.exists(_c["storage_state"])
                cfg_file = f"{temp_dir}/{os.path.basename(cfg_file)}"
                with open(cfg_file, "w") as f:
                    json.dump(_c, f)

            # Load input images
            if image_paths is not None:
                if isinstance(image_paths, str):
                    image_paths = [image_paths]
                for image_path in image_paths:
                    if image_path.startswith("http"):
                        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                        input_image = Image.open(requests.get(image_path, stream=True, headers=headers).raw)
                    else:
                        input_image = Image.open(image_path)
                    images.append(input_image)

        # Reset agent and environment
        base_agent.reset(cfg_file)
        obs, info = env.reset(options={"config_file": cfg_file})

        # Execute task
        worker_logger.info(f"Executing task {task_name}_{task_id}")
        res = coordinator.execute_task(
            user_goal=intent,
            start_observation={"observation": obs, "info": info},
            max_steps=args_dict.get('max_steps', 30)
        )

        worker_logger.info(f"Evaluating task {task_name}_{task_id}")

        # Save trajectory
        try:
            traj = coordinator.trajectory
            
            def _serialize_element(el):
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
                
                if isinstance(el, dict) and 'action_type' in el:
                    try:
                        at_name = ActionTypes(el.get('action_type')).name
                    except Exception:
                        at_name = str(el.get('action_type'))
                    return {
                        'type': 'Action',
                        'action_type': at_name,
                        'element_id': el.get('element_id'),
                        'answer': el.get('answer'),
                        'raw_prediction_preview': el.get('raw_prediction'),
                    }
                
                try:
                    return {'type': str(type(el)), 'repr': str(el)[:200]}
                except Exception:
                    return {'type': str(type(el)), 'repr': '<unserializable>'}

            simple_traj = [_serialize_element(x) for x in traj]
            traj_path = Path(task_result_dir) / f"{task_id}_full_trajectory.json"
            with open(traj_path, 'w', encoding='utf-8') as _f:
                json.dump(simple_traj, _f, ensure_ascii=False, indent=2)
        except Exception as e:
            worker_logger.warning(f"Failed to serialize trajectory: {e}")

        # Evaluate
        eval_caption_image_fn = None
        evaluator = evaluator_router(cfg_file, captioning_fn=eval_caption_image_fn)
        score = evaluator(
            trajectory=coordinator.trajectory,
            config_file=cfg_file,
            page=env.page
        )

        # Save trace if enabled
        if args_dict.get('save_trace_enabled', True):
            trace_path = Path(task_result_dir) / f"{task_id}.zip"
            env.save_trace(trace_path)

        # Close environment
        env.close()

        result_status = "PASS" if score == 1 else "FAIL"
        worker_logger.info(f"Task {task_name}_{task_id} completed: {result_status}, Score: {score}")

        return {
            "success": True,
            "task_name": task_name,
            "task_id": task_id,
            "intent": intent,
            "start_url": start_url,
            "score": score,
            "status": result_status,
            "config_file": cfg_file,
            "worker_id": worker_id
        }

    except openai.OpenAIError as e:
        worker_logger.error(f"OpenAI Error for {task_name}_{task_id}: {repr(e)}")
        return {
            "success": False,
            "task_name": task_name,
            "task_id": task_id if 'task_id' in locals() else "unknown",
            "error": f"OpenAI Error: {repr(e)}",
            "worker_id": worker_id
        }
    except Exception as e:
        worker_logger.error(f"Unhandled Error for {task_name}_{task_id}: {repr(e)}")
        worker_logger.error(traceback.format_exc())
        
        # Write error to file
        error_file = Path(result_dir) / f"error_worker_{worker_id}.txt"
        with open(error_file, "a") as f:
            f.write(f"[Config file]: {cfg_file}\n")
            f.write(f"[Unhandled Error] {repr(e)}\n")
            f.write(traceback.format_exc())
        
        return {
            "success": False,
            "task_name": task_name,
            "task_id": task_id if 'task_id' in locals() else "unknown",
            "error": f"Unhandled Error: {repr(e)}",
            "worker_id": worker_id
        }


def worker_process(
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    args_dict: Dict[str, Any],
    config: Dict[str, Any],
    worker_id: int,
    result_dir: str
):
    """Worker process that evaluates tasks from the queue.
    
    Args:
        task_queue: Queue containing tasks to evaluate
        result_queue: Queue to put results into
        args_dict: Dictionary of command line arguments
        config: Configuration dictionary
        worker_id: Worker process ID
        result_dir: Result directory path
    """
    # Clean up any inherited asyncio event loop to prevent conflicts with Playwright sync API
    import asyncio
    
    # Method 1: Try to get and close any existing event loop
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.stop()
        if not loop.is_closed():
            loop.close()
    except RuntimeError:
        pass  # No event loop exists, which is fine
    
    # Method 2: Explicitly set event loop to None to ensure no loop exists
    try:
        asyncio.set_event_loop(None)
    except RuntimeError:
        pass
    
    # Method 3: Reset event loop policy
    asyncio.set_event_loop_policy(None)
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    
    while True:
        try:
            task_info = task_queue.get(timeout=1)
            if task_info is None:  # Poison pill
                break
            
            result = evaluate_single_task(task_info, args_dict, config, worker_id, result_dir)
            result_queue.put(result)
            
        except Exception as e:
            # Queue is empty or other error
            if isinstance(e, Empty):
                continue
            else:
                print(f"Worker {worker_id} error: {e}")
                break


def test_parallel(args, test_file_list, num_workers: int = 3):
    """Run the multi-agent system with parallel evaluation.
    
    Args:
        args: Command line arguments
        test_file_list: List of (config_file, task_name) tuples to evaluate
        num_workers: Number of parallel worker processes
    """
    config_file = args.config_file
    file_config = load_config_file(config_file)
    config = merge_config_with_args(file_config, args)

    if args.dry_run:
        return

    # Setup result directory
    result_dir = config.get('output', {}).get('result_dir', 'results')
    if not Path(result_dir).exists():
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        print(f"Created result directory: {result_dir}")

    if 'output' not in config:
        config['output'] = {}
    config['output']['result_dir'] = result_dir

    # Setup main logger
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    file_handler = logging.FileHandler(LOG_FILE_NAME)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"Starting parallel evaluation with {num_workers} workers")
    logger.info(f"Total tasks to evaluate: {len(test_file_list)}")

    # Create queues
    manager = Manager()
    task_queue = manager.Queue()
    result_queue = manager.Queue()

    # Populate task queue
    for task_info in test_file_list:
        task_queue.put(task_info)

    # Add poison pills
    for _ in range(num_workers):
        task_queue.put(None)

    # Convert args to dictionary for pickling
    args_dict = vars(args)

    # Start worker processes
    processes = []
    for worker_id in range(num_workers):
        p = mp.Process(
            target=worker_process,
            args=(task_queue, result_queue, args_dict, config, worker_id, result_dir)
        )
        p.start()
        processes.append(p)
        logger.info(f"Started worker {worker_id}")

    # Collect results
    results = []
    completed = 0
    total = len(test_file_list)
    
    while completed < total:
        try:
            result = result_queue.get(timeout=1)
            results.append(result)
            completed += 1
            
            if result["success"]:
                logger.info(f"Progress: {completed}/{total} - Task {result['task_name']}_{result['task_id']}: {result['status']}")
            else:
                logger.error(f"Progress: {completed}/{total} - Task {result['task_name']}_{result['task_id']}: FAILED - {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            if not isinstance(e, Empty):
                logger.error(f"Error collecting results: {e}")
            continue

    # Wait for all processes to finish
    for p in processes:
        p.join()

    logger.info("All workers finished")

    # Aggregate results
    scores = []
    task_scores = {}
    task_details = {}
    
    for result in results:
        if result["success"]:
            score = result["score"]
            scores.append(score)
            
            score_key = f"{result['task_name']}_{result['task_id']}"
            task_scores[score_key] = score
            task_details[score_key] = {
                "task_name": result["task_name"],
                "task_id": result["task_id"],
                "intent": result["intent"],
                "start_url": result["start_url"],
                "score": score,
                "status": result["status"],
                "config_file": result["config_file"],
                "worker_id": result["worker_id"]
            }

    if len(scores) > 0:
        logger.info(f"Average score: {sum(scores) / len(scores):.4f}")

        # Calculate per-task statistics
        task_statistics = {}
        for key, score in task_scores.items():
            task_name = key.rsplit('_', 1)[0]
            if task_name not in task_statistics:
                task_statistics[task_name] = {
                    "total": 0,
                    "passed": 0,
                    "failed": 0,
                    "scores": []
                }
            task_statistics[task_name]["total"] += 1
            task_statistics[task_name]["scores"].append(score)
            if score == 1:
                task_statistics[task_name]["passed"] += 1
            else:
                task_statistics[task_name]["failed"] += 1

        # Calculate average score for each task
        for task_name, stats in task_statistics.items():
            stats["average_score"] = sum(stats["scores"]) / len(stats["scores"])

        # Generate detailed results JSON file
        results_file = Path(result_dir) / "task_scores.json"
        results_data = {
            "task_scores": task_scores,
            "task_details": task_details,
            "task_statistics": task_statistics,
            "summary": {
                "total_tasks": len(test_file_list),
                "completed_tasks": len(scores),
                "failed_tasks": len(test_file_list) - len(scores),
                "average_score": sum(scores) / len(scores) if len(scores) > 0 else 0,
                "passed_tasks": sum(1 for score in scores if score == 1),
                "num_workers": num_workers
            }
        }

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Detailed task scores saved to: {results_file}")

        # Log per-task statistics
        logger.info("\n=== Per-Task Statistics ===")
        for task_name, stats in task_statistics.items():
            logger.info(f"{task_name}: {stats['passed']}/{stats['total']} passed, average score: {stats['average_score']:.2f}")
    else:
        logger.warning("No tasks completed successfully")


def prepare(args: argparse.Namespace) -> None:
    """Prepare environment for evaluation."""
    from agent.prompts import to_json
    to_json.run()

    result_dir = args.result_dir
    if not result_dir:
        result_dir = f"cache/results_{time.strftime('%Y%m%d%H%M%S', time.localtime())}_parallel"
    if not Path(result_dir).exists():
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        args.result_dir = result_dir
        print(f"Create result dir: {result_dir}")

    if not (Path(result_dir) / "traces").exists():
        (Path(result_dir) / "traces").mkdir(parents=True)

    with open(os.path.join(result_dir, "log_files.txt"), "a+") as f:
        f.write(f"{LOG_FILE_NAME}\n")


def get_unfinished(config_files: list, result_dir: str) -> list:
    """Get unfinished config files."""
    if not os.path.exists(result_dir):
        return config_files

    result_dirs = [d for d in os.listdir(result_dir) if os.path.isdir(os.path.join(result_dir, d))]
    completed_keys = set(result_dirs)

    unfinished_configs = []
    for config_file, task_name in config_files:
        task_id = os.path.basename(config_file).split(".")[0]
        result_key = f"{task_name}_{task_id}"
        if result_key not in completed_keys:
            unfinished_configs.append((config_file, task_name))
    return unfinished_configs


def dump_config(args: argparse.Namespace) -> None:
    """Dump configuration to file."""
    config_file = Path(args.result_dir) / "config.json"
    if not config_file.exists():
        with open(config_file, "w") as f:
            json.dump(vars(args), f, indent=4)
            print(f"Dump config to {config_file}")


def config():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Parallel Multi-Agent Web Arena Runner")

    # Required arguments
    parser.add_argument("--config_file", type=str, required=True,
                       help="Path to JSON configuration file")

    # Parallel execution
    parser.add_argument("--num_workers", type=int, default=3,
                       help="Number of parallel worker processes (default: 3)")

    # Commonly overridden arguments
    parser.add_argument("--start_url", type=str,
                       help="Override starting URL")
    parser.add_argument("--intent", type=str,
                       help="Override task intent")
    parser.add_argument("--max_steps", type=int, default=30,
                       help="Override maximum steps")
    parser.add_argument("--result_dir", type=str,
                       help="Override result directory")

    # Debugging options
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    parser.add_argument("--dry_run", action="store_true",
                       help="Show configuration without executing")

    # Eval arguments
    parser.add_argument("--test_config", type=str,
                       help="Path to test config JSON file (e.g., config_subset.json)")
    parser.add_argument("--eval_captioning_model_device", type=str, default="cpu")
    parser.add_argument("--eval_captioning_model", type=str, default="qwen3-vl-plus")
    parser.add_argument("--captioning_model", type=str, default="qwen3-vl-plus")

    return parser.parse_args()


if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn' for better compatibility
    mp.set_start_method('spawn', force=True)
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    args = config()
    args.sleep_after_execution = 2.5
    prepare(args)

    # Load test config file
    if not args.test_config:
        raise ValueError("--test_config argument is required. Please provide path to config_subset.json")

    with open(args.test_config, 'r') as f:
        test_config = json.load(f)

    # Base directory for test configs
    test_config_base_dir = "config_files/vwa/"

    # Build test file list from config_subset.json
    test_file_list = []
    for task_name, config_files in test_config.items():
        task_dir = os.path.join(test_config_base_dir, f"test_{task_name}")
        for config_file in config_files:
            config_path = os.path.join(task_dir, config_file)
            if os.path.exists(config_path):
                test_file_list.append((config_path, task_name))
            else:
                print(f"Warning: Config file not found: {config_path}")

    print(f"Total {len(test_file_list)} tasks to evaluate")

    # Filter out finished tasks
    test_file_list = get_unfinished(test_file_list, args.result_dir)
    print(f"Total {len(test_file_list)} tasks left after filtering finished ones")

    args.render = False
    args.render_screenshot = True
    args.save_trace_enabled = True
    args.current_viewport_only = True

    dump_config(args)

    # Run parallel evaluation
    test_parallel(args, test_file_list, num_workers=args.num_workers)

