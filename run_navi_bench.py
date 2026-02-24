"""Navi-Bench Evaluation Runner.

This script evaluates the multi-agent web automation system on the
Yutori Navi-Bench benchmark (https://huggingface.co/datasets/yutori-ai/navi-bench).

It bridges the existing agent architecture (sync Playwright via ScriptBrowserEnv)
with navi-bench's async evaluators by running evaluator calls in a dedicated
async event loop.

Usage:
    python run_navi_bench.py --config_file config_navi_bench.json
    python run_navi_bench.py --config_file config_navi_bench.json --task_ids "navi_bench/craigslist/craigslist_basic_filters/0,navi_bench/craigslist/craigslist_basic_filters/4"
    python run_navi_bench.py --config_file config_navi_bench.json --domains craigslist,apartments
    python run_navi_bench.py --config_file config_navi_bench.json --max_tasks 10
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

# ──────────────────────────────────────────────────────────────────────
# Ensure navi-bench package is importable
# ──────────────────────────────────────────────────────────────────────
NAVI_BENCH_DIR = os.path.join(os.path.dirname(__file__), "navi-bench")
if NAVI_BENCH_DIR not in sys.path:
    sys.path.insert(0, NAVI_BENCH_DIR)

# ──────────────────────────────────────────────────────────────────────
# Set VisualWebArena environment variables BEFORE importing browser_env.
#
# browser_env/env_config.py asserts that site URLs (REDDIT, SHOPPING, etc.)
# are set at IMPORT TIME.  Navi-Bench evaluates on real public websites
# (Craigslist, Apartments.com, Google Flights, OpenTable, Resy) and does
# NOT use the VisualWebArena self-hosted sites.  Setting placeholder URLs
# here satisfies the assertion without affecting Navi-Bench evaluation.
# ──────────────────────────────────────────────────────────────────────
_VWA_PLACEHOLDERS = {
    "DATASET": "visualwebarena",
    "REDDIT": "http://placeholder.reddit.example",
    "SHOPPING": "http://placeholder.shopping.example",
    "WIKIPEDIA": "http://placeholder.wiki.example",
    "HOMEPAGE": "http://placeholder.homepage.example",
    "CLASSIFIEDS": "http://placeholder.classifieds.example",
    "CLASSIFIEDS_RESET_TOKEN": "placeholder_token",
}
for _key, _val in _VWA_PLACEHOLDERS.items():
    os.environ.setdefault(_key, _val)

# ──────────────────────────────────────────────────────────────────────
# Imports from the existing agent framework
# ──────────────────────────────────────────────────────────────────────
from browser_env import (
    ScriptBrowserEnv,
    Action,
    ActionTypes,
    Trajectory,
    create_stop_action,
)
from agent import PromptAgent
from agent.multi_agent_coordinator import MultiAgentCoordinator
from agent.prompts.prompt_constructor import (
    PromptConstructor,
    DirectPromptConstructor,
    CoTPromptConstructor,
    MultimodalCoTPromptConstructor,
)
from llms import lm_config
from llms.tokenizers import Tokenizer
from agent.utils import is_multimodal_model
from evaluation_harness import image_utils

# ──────────────────────────────────────────────────────────────────────
# Imports from navi-bench
# ──────────────────────────────────────────────────────────────────────
from datasets import load_dataset
from navi_bench.base import DatasetItem, BaseMetric, instantiate

# ──────────────────────────────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────────────────────────────
LOG_FOLDER = "log_files"
Path(LOG_FOLDER).mkdir(parents=True, exist_ok=True)
LOG_FILE_NAME = f"{LOG_FOLDER}/navi_bench_{time.strftime('%Y%m%d%H%M%S', time.localtime())}.log"

logger = logging.getLogger("navi_bench_runner")
logger.setLevel(logging.INFO)
logger.propagate = False  # Prevent duplicate output via root logger
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
file_handler = logging.FileHandler(LOG_FILE_NAME, encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ──────────────────────────────────────────────────────────────────────
# Async helper – run async evaluator methods from sync code
# ──────────────────────────────────────────────────────────────────────
# navi-bench evaluators are async, but ScriptBrowserEnv uses sync
# Playwright which runs its own internal asyncio event loop.
#
# We CANNOT call loop.run_until_complete() in the main thread because
# Playwright's loop is already running there.  Instead, we spin up
# a dedicated background thread with its own fresh event loop.
# ──────────────────────────────────────────────────────────────────────
import threading
from concurrent.futures import Future


def run_async(coro):
    """Run an async coroutine from synchronous context.

    Executes the coroutine in a separate thread with its own event loop
    to avoid conflicts with Playwright's internal asyncio loop.
    """
    future: Future = Future()

    def _target():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(coro)
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)
        finally:
            loop.close()

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout=120)  # generous timeout for network-heavy evaluators

    if thread.is_alive():
        raise TimeoutError("run_async: evaluator call timed out after 120 seconds")

    return future.result()


# ══════════════════════════════════════════════════════════════════════
# SyncPageAsyncAdapter
# ══════════════════════════════════════════════════════════════════════
class SyncPageAsyncAdapter:
    """Wraps a **sync** Playwright Page to satisfy navi-bench's async
    ``page.evaluate(js)`` calls.

    navi-bench evaluators call ``await page.evaluate(script)`` inside their
    ``update()`` method. Since we use ``sync_playwright``, ``env.page`` is a
    *sync* ``Page`` object whose ``.evaluate()`` returns a plain value
    (not a coroutine).

    This adapter intercepts ``.evaluate()`` calls and wraps the sync result
    in an awaitable, so that ``await page.evaluate(...)`` works from async
    evaluator code.

    All other attribute accesses are transparently forwarded to the real page.
    """

    def __init__(self, sync_page):
        # Store with object.__setattr__ to avoid triggering __getattr__
        object.__setattr__(self, "_sync_page", sync_page)

    # ── async evaluate ────────────────────────────────────────────────
    async def evaluate(self, expression, arg=None, **kwargs):
        """Call sync page.evaluate and return the result as an awaitable."""
        sync_page = object.__getattribute__(self, "_sync_page")
        if arg is not None:
            return sync_page.evaluate(expression, arg, **kwargs)
        return sync_page.evaluate(expression, **kwargs)

    # ── transparent forwarding ────────────────────────────────────────
    def __getattr__(self, name):
        sync_page = object.__getattribute__(self, "_sync_page")
        return getattr(sync_page, name)

    @property
    def url(self):
        sync_page = object.__getattribute__(self, "_sync_page")
        return sync_page.url


# ══════════════════════════════════════════════════════════════════════
# Dataset Loading
# ══════════════════════════════════════════════════════════════════════

HF_DATASET = "yutori-ai/navi-bench"
HF_SPLIT = "validation"


def load_navi_bench_tasks(
    task_ids: Optional[List[str]] = None,
    domains: Optional[List[str]] = None,
    difficulties: Optional[List[str]] = None,
    max_tasks: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load and filter tasks from the Navi-Bench HuggingFace dataset.

    Args:
        task_ids: If provided, only load these specific task IDs.
        domains: If provided, filter to these domains (e.g. ["craigslist", "resy"]).
        difficulties: If provided, filter to these difficulties (e.g. ["easy", "medium"]).
        max_tasks: Maximum number of tasks to load.

    Returns:
        List of dataset row dicts.
    """
    logger.info(f"Loading Navi-Bench dataset from {HF_DATASET}/{HF_SPLIT} ...")
    dataset = load_dataset(HF_DATASET, split=HF_SPLIT)
    logger.info(f"Loaded {len(dataset)} tasks from Navi-Bench")

    tasks = []
    for row in dataset:
        # Filter by task_id
        if task_ids and row["task_id"] not in task_ids:
            continue
        # Filter by domain
        if domains and row["domain"] not in domains:
            continue
        # Filter by difficulty
        if difficulties and row.get("suggested_difficulty") not in difficulties:
            continue
        tasks.append(dict(row))

    # Sort by task_id for deterministic ordering
    tasks.sort(key=lambda x: x["task_id"])

    if max_tasks is not None and len(tasks) > max_tasks:
        tasks = tasks[:max_tasks]

    logger.info(f"Selected {len(tasks)} tasks after filtering")
    return tasks


# ══════════════════════════════════════════════════════════════════════
# Core Evaluation Logic
# ══════════════════════════════════════════════════════════════════════


def evaluate_single_task(
    task_row: Dict[str, Any],
    coordinator: MultiAgentCoordinator,
    env: ScriptBrowserEnv,
    max_steps: int,
    result_dir: str,
    save_trace: bool = True,
) -> Dict[str, Any]:
    """Evaluate a single Navi-Bench task using the multi-agent system.

    Steps:
        1. Generate task config from DatasetItem (resolves dynamic dates)
        2. Instantiate the navi-bench evaluator
        3. Reset browser env and navigate to the task URL
        4. Run the multi-agent coordinator to completion
        5. Call evaluator.update() for each URL the agent visited
        6. Call evaluator.compute() to get the final score

    Returns:
        Dict with task_id, score, reasoning, and metadata.
    """
    task_id = task_row["task_id"]
    domain = task_row["domain"]
    difficulty = task_row.get("suggested_difficulty", "unknown")

    logger.info(f"{'='*70}")
    logger.info(f"[Task] {task_id}  (domain={domain}, difficulty={difficulty})")
    logger.info(f"{'='*70}")

    # ── 1. Generate task config ──────────────────────────────────────
    dataset_item = DatasetItem.model_validate(task_row)
    task_config = dataset_item.generate_task_config()

    task_description = task_config.task
    start_url = task_config.url
    eval_config = task_config.eval_config

    logger.info(f"[Task URL]  {start_url}")
    logger.info(f"[Task Desc] {task_description}")

    # ── 2. Instantiate evaluator ─────────────────────────────────────
    evaluator: BaseMetric = instantiate(eval_config)
    logger.info(f"[Evaluator] {evaluator.__class__.__name__}")

    # ── 3. Create task-specific result directory ─────────────────────
    # Sanitize task_id for filesystem: navi_bench/craigslist/xxx/0 -> craigslist_xxx_0
    safe_task_id = task_id.replace("navi_bench/", "").replace("/", "_")
    task_result_dir = os.path.join(result_dir, safe_task_id)
    Path(task_result_dir).mkdir(parents=True, exist_ok=True)

    # ── 4. Reset browser env and navigate to start URL ───────────────
    # Build an instance config compatible with ScriptBrowserEnv.reset()
    instance_config = {"start_url": start_url}

    obs, info = env.reset(options=instance_config)

    # ── 5. Reset and initialize navi-bench evaluator ─────────────────
    run_async(evaluator.reset())

    # First evaluator update with the start URL
    async_page = SyncPageAsyncAdapter(env.page)
    run_async(evaluator.update(url=start_url, page=async_page))

    # ── 6. Hook evaluator into browser navigation ────────────────────
    # We will call evaluator.update() after each agent step.
    # Track all URLs visited for logging.
    visited_urls: List[str] = [start_url]

    # ── 7. Run multi-agent coordinator ───────────────────────────────
    # Use suggested max_steps from benchmark if available
    effective_max_steps = task_row.get("suggested_max_steps") or max_steps

    # Create a new coordinator instance for this task (with task-specific result dir)
    coordinator.result_dir = task_result_dir
    coordinator.log_file_path = os.path.join(task_result_dir, "agent_responses.log")
    coordinator.observation_log_path = os.path.join(task_result_dir, "observations.json")
    coordinator.images_dir = os.path.join(task_result_dir, "images")
    coordinator.images_som_dir = os.path.join(task_result_dir, "images_som")
    coordinator._setup_logging()
    
    # Patch the browser step to hook evaluator updates
    original_step = env.step

    def patched_step(action: Action):
        result = original_step(action)
        # After each browser action, update the navi-bench evaluator
        try:
            current_url = env.page.url
            if current_url and current_url not in visited_urls:
                visited_urls.append(current_url)
            async_page = SyncPageAsyncAdapter(env.page)
            run_async(evaluator.update(url=current_url, page=async_page))
            logger.debug(f"[Evaluator Update] url={current_url}")
        except Exception as e:
            logger.warning(f"[Evaluator Update Warning] {e}")
        return result

    env.step = patched_step

    try:
        execution_result = coordinator.execute_task(
            user_goal=task_description,
            start_observation={"observation": obs, "info": info},
            max_steps=effective_max_steps,
        )
    finally:
        # Restore original step function
        env.step = original_step

    # ── 8. Final evaluator update with current page state ────────────
    try:
        final_url = env.page.url
        if final_url and final_url not in visited_urls:
            visited_urls.append(final_url)
        async_page = SyncPageAsyncAdapter(env.page)
        run_async(evaluator.update(url=final_url, page=async_page))
    except Exception as e:
        logger.warning(f"[Final Evaluator Update Warning] {e}")

    # ── 9. Compute evaluation result ─────────────────────────────────
    eval_result = run_async(evaluator.compute())
    score = getattr(eval_result, "score", 0.0)
    reasoning = getattr(eval_result, "reasoning", "")

    # ── 10. Save trace if enabled ────────────────────────────────────
    if save_trace:
        try:
            trace_path = Path(task_result_dir) / f"{safe_task_id}.zip"
            env.save_trace(trace_path)
        except Exception as e:
            logger.warning(f"[Trace Save Warning] {e}")

    # ── 11. Save task result ─────────────────────────────────────────
    task_result = {
        "task_id": task_id,
        "domain": domain,
        "difficulty": difficulty,
        "task_description": task_description,
        "start_url": start_url,
        "score": score,
        "reasoning": reasoning,
        "status": "PASS" if score >= 1.0 else ("PARTIAL" if score > 0 else "FAIL"),
        "total_steps": len(coordinator.actions),
        "max_steps_allowed": effective_max_steps,
        "visited_urls": visited_urls,
        "evaluator_class": evaluator.__class__.__name__,
        "eval_result_details": {
            k: v for k, v in vars(eval_result).items()
            if isinstance(v, (str, int, float, bool, list, dict, type(None)))
        } if hasattr(eval_result, "__dict__") else {},
    }

    result_file = Path(task_result_dir) / "navi_bench_result.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(task_result, f, indent=2, ensure_ascii=False, default=str)

    status_emoji = "✅" if score >= 1.0 else ("🟡" if score > 0 else "❌")
    logger.info(f"{status_emoji} [Result] {task_id}: score={score:.2f} ({task_result['status']})")
    if reasoning:
        logger.info(f"   Reasoning: {reasoning}")

    return task_result


# ══════════════════════════════════════════════════════════════════════
# CLI & Main
# ══════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Navi-Bench Evaluation Runner for Multi-Agent System"
    )
    parser.add_argument(
        "--config_file", type=str, required=True,
        help="Path to the agent JSON configuration file (e.g. config_navi_bench.json)",
    )
    parser.add_argument(
        "--task_ids", type=str, default=None,
        help="Comma-separated list of specific task IDs to evaluate",
    )
    parser.add_argument(
        "--domains", type=str, default=None,
        help="Comma-separated list of domains to filter (craigslist,apartments,google_flights,opentable,resy)",
    )
    parser.add_argument(
        "--difficulties", type=str, default=None,
        help="Comma-separated list of difficulties to filter (easy,medium,hard)",
    )
    parser.add_argument(
        "--max_tasks", type=int, default=None,
        help="Maximum number of tasks to evaluate",
    )
    parser.add_argument(
        "--max_steps", type=int, default=30,
        help="Default maximum steps per task (overridden by benchmark suggested_max_steps if available)",
    )
    parser.add_argument(
        "--result_dir", type=str, default=None,
        help="Override result output directory",
    )
    parser.add_argument(
        "--no_trace", action="store_true",
        help="Disable saving Playwright traces",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="List tasks without executing them",
    )
    parser.add_argument(
        "--enable_monitor", action="store_true",
        help="Explicitly enable the monitor module",
    )
    parser.add_argument(
        "--disable_monitor", action="store_true",
        help="Explicitly disable the monitor module",
    )
    return parser.parse_args()


def load_config(config_file: str) -> Dict[str, Any]:
    with open(config_file, "r", encoding="utf-8") as f:
        return json.load(f)


def build_env_and_coordinator(config: Dict[str, Any], args: argparse.Namespace) -> Tuple[ScriptBrowserEnv, MultiAgentCoordinator, PromptAgent]:
    """Build the ScriptBrowserEnv, PromptAgent, and MultiAgentCoordinator from config."""

    # Apply monitor overrides from arguments
    if args.enable_monitor:
        if "monitor" not in config: config["monitor"] = {}
        config["monitor"]["enable_monitor"] = True
    elif args.disable_monitor:
        if "monitor" not in config: config["monitor"] = {}
        config["monitor"]["enable_monitor"] = False

    model_config = config.get("model", {})
    lm_cfg = lm_config.LMConfig(
        provider=model_config.get("provider", "openai"),
        model=model_config.get("model", "gpt-4"),
        mode=model_config.get("mode", "chat"),
    )
    if model_config:
        lm_cfg.gen_config.update({
            "temperature": model_config.get("temperature", 1.0),
            "top_p": model_config.get("top_p", 0.9),
            "max_tokens": model_config.get("max_tokens", 8192),
            "context_length": model_config.get("context_length", 0),
            "stop_token": model_config.get("stop_token", None),
            "max_obs_length": model_config.get("max_obs_length", 0),
            "max_retry": model_config.get("max_retry", 3),
        })

    # ── Browser Environment ──────────────────────────────────────────
    browser_config = config.get("browser", {})
    observation_type = config.get("observation", {}).get("observation_type", "accessibility_tree")

    caption_image_fn = None
    if observation_type == "accessibility_tree_with_captioner":
        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        captioning_model = model_config.get("captioning_model", "Salesforce/blip2-flan-t5-xl")
        caption_image_fn = image_utils.get_captioning_fn(device, dtype, captioning_model)

    viewport_size = {
        "width": browser_config.get("viewport_width", 1280),
        "height": browser_config.get("viewport_height", 720),
    }

    env = ScriptBrowserEnv(
        headless=browser_config.get("headless", True),
        slow_mo=browser_config.get("slow_mo", 0),
        observation_type=observation_type,
        current_viewport_only=config.get("observation", {}).get("current_viewport_only", True),
        viewport_size=viewport_size,
        save_trace_enabled=browser_config.get("save_trace_enabled", True),
        sleep_after_execution=browser_config.get("sleep_after_execution", 0.5),
        captioning_fn=caption_image_fn,
    )

    # ── Prompt Constructor ───────────────────────────────────────────
    is_mm = is_multimodal_model(lm_cfg.model)
    is_image_obs = observation_type in ["image", "image_som"]

    instruction_path = config.get("instruction_path")
    if not instruction_path:
        if is_mm and is_image_obs:
            instruction_path = "agent/prompts/jsons/p_multimodal_cot_id_actree_3s.json"
        else:
            instruction_path = "agent/prompts/jsons/p_cot_id_actree_3s.json"

    with open(instruction_path) as f:
        instruction_data = json.load(f)
        constructor_type = instruction_data.get("meta_data", {}).get(
            "prompt_constructor", "DirectPromptConstructor"
        )

    tokenizer = Tokenizer(lm_cfg.provider, lm_cfg.model)
    if constructor_type == "MultimodalCoTPromptConstructor":
        prompt_constructor = MultimodalCoTPromptConstructor(
            instruction_path=instruction_path, lm_config=lm_cfg, tokenizer=tokenizer
        )
    elif constructor_type == "CoTPromptConstructor":
        prompt_constructor = CoTPromptConstructor(
            instruction_path=instruction_path, lm_config=lm_cfg, tokenizer=tokenizer
        )
    else:
        prompt_constructor = DirectPromptConstructor(
            instruction_path=instruction_path, lm_config=lm_cfg, tokenizer=tokenizer
        )

    # ── Base Agent ───────────────────────────────────────────────────
    action_set_tag = config.get("observation", {}).get("action_set_tag", "id_accessibility_tree")
    base_agent = PromptAgent(
        action_set_tag=action_set_tag,
        lm_config=lm_cfg,
        prompt_constructor=prompt_constructor,
        captioning_fn=caption_image_fn if observation_type == "accessibility_tree_with_captioner" else None,
    )

    # ── Coordinator ──────────────────────────────────────────────────
    result_dir = config.get("output", {}).get("result_dir", "results/navi_bench")
    coordinator = MultiAgentCoordinator(
        lm_config=lm_cfg,
        existing_prompt_agent=base_agent,
        browser_env=env,
        result_dir=result_dir,
        memory_config=config.get("memory", {}),
        monitor_config=config.get("monitor", {}),
        clear_result_dir=False,
        save_images=config.get("output", {}).get("save_images", False),
    )

    return env, coordinator, base_agent


def generate_summary_report(
    all_results: List[Dict[str, Any]],
    result_dir: str,
    total_time: float,
) -> None:
    """Generate and save a summary report of all task results."""

    # ── Per-domain statistics ────────────────────────────────────────
    domain_stats: Dict[str, Dict[str, Any]] = {}
    difficulty_stats: Dict[str, Dict[str, Any]] = {}

    for r in all_results:
        domain = r["domain"]
        difficulty = r.get("difficulty", "unknown")
        score = r["score"]

        for key, stats in [(domain, domain_stats), (difficulty, difficulty_stats)]:
            if key not in stats:
                stats[key] = {"total": 0, "passed": 0, "partial": 0, "failed": 0, "scores": []}
            stats[key]["total"] += 1
            stats[key]["scores"].append(score)
            if score >= 1.0:
                stats[key]["passed"] += 1
            elif score > 0:
                stats[key]["partial"] += 1
            else:
                stats[key]["failed"] += 1

    for stats_dict in [domain_stats, difficulty_stats]:
        for key in stats_dict:
            scores = stats_dict[key]["scores"]
            stats_dict[key]["avg_score"] = sum(scores) / len(scores) if scores else 0

    # ── Overall summary ──────────────────────────────────────────────
    all_scores = [r["score"] for r in all_results]
    summary = {
        "benchmark": "navi-bench v1",
        "timestamp": datetime.now().isoformat(),
        "total_time_seconds": round(total_time, 1),
        "total_time_formatted": time.strftime("%H:%M:%S", time.gmtime(total_time)),
        "overall": {
            "total_tasks": len(all_results),
            "passed": sum(1 for s in all_scores if s >= 1.0),
            "partial": sum(1 for s in all_scores if 0 < s < 1.0),
            "failed": sum(1 for s in all_scores if s == 0),
            "avg_score": sum(all_scores) / len(all_scores) if all_scores else 0,
            "success_rate": sum(1 for s in all_scores if s >= 1.0) / len(all_scores) if all_scores else 0,
        },
        "by_domain": {
            k: {key: v for key, v in stats.items() if key != "scores"}
            for k, stats in sorted(domain_stats.items())
        },
        "by_difficulty": {
            k: {key: v for key, v in stats.items() if key != "scores"}
            for k, stats in sorted(difficulty_stats.items())
        },
        "task_results": [
            {
                "task_id": r["task_id"],
                "domain": r["domain"],
                "difficulty": r.get("difficulty", "unknown"),
                "score": r["score"],
                "status": r["status"],
                "total_steps": r["total_steps"],
                "reasoning": r.get("reasoning", ""),
            }
            for r in all_results
        ],
    }

    # Save JSON report
    report_path = Path(result_dir) / "navi_bench_summary.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Summary report saved to {report_path}")

    # ── Print console summary ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  NAVI-BENCH EVALUATION SUMMARY")
    print("=" * 70)
    
    overall = summary["overall"]
    print(f"\n  Total Tasks:   {overall['total_tasks']}")
    print(f"  Passed:        {overall['passed']}  ({overall['success_rate']*100:.1f}%)")
    print(f"  Partial:       {overall['partial']}")
    print(f"  Failed:        {overall['failed']}")
    print(f"  Avg Score:     {overall['avg_score']:.3f}")
    print(f"  Total Time:    {summary['total_time_formatted']}")

    print(f"\n  {'─'*66}")
    print(f"  {'Domain':<20} {'Tasks':>6} {'Pass':>6} {'Fail':>6} {'Avg Score':>10}")
    print(f"  {'─'*66}")
    for domain, stats in sorted(domain_stats.items()):
        print(
            f"  {domain:<20} {stats['total']:>6} {stats['passed']:>6} "
            f"{stats['failed']:>6} {stats['avg_score']:>10.3f}"
        )

    print(f"\n  {'─'*66}")
    print(f"  {'Difficulty':<20} {'Tasks':>6} {'Pass':>6} {'Fail':>6} {'Avg Score':>10}")
    print(f"  {'─'*66}")
    for diff, stats in sorted(difficulty_stats.items()):
        print(
            f"  {diff:<20} {stats['total']:>6} {stats['passed']:>6} "
            f"{stats['failed']:>6} {stats['avg_score']:>10.3f}"
        )

    print(f"\n  {'─'*66}")
    print(f"  Individual Results:")
    print(f"  {'─'*66}")
    for r in all_results:
        emoji = "✅" if r["score"] >= 1.0 else ("🟡" if r["score"] > 0 else "❌")
        print(f"  {emoji}  {r['task_id']:<50} {r['score']:.2f}  ({r['total_steps']} steps)")

    print("\n" + "=" * 70 + "\n")


# ══════════════════════════════════════════════════════════════════════
# Main Entry Point
# ══════════════════════════════════════════════════════════════════════


def main():
    os.environ.setdefault("DATASET", "visualwebarena")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    args = parse_args()

    # ── Load config ──────────────────────────────────────────────────
    config = load_config(args.config_file)
    logger.info(f"Loaded config from {args.config_file}")

    # Override result dir if specified
    if args.result_dir:
        if "output" not in config:
            config["output"] = {}
        config["output"]["result_dir"] = args.result_dir

    result_dir = config.get("output", {}).get("result_dir", "results/navi_bench")
    Path(result_dir).mkdir(parents=True, exist_ok=True)

    # ── Load tasks ───────────────────────────────────────────────────
    task_ids = args.task_ids.split(",") if args.task_ids else None
    domains = args.domains.split(",") if args.domains else None
    difficulties = args.difficulties.split(",") if args.difficulties else None

    tasks = load_navi_bench_tasks(
        task_ids=task_ids,
        domains=domains,
        difficulties=difficulties,
        max_tasks=args.max_tasks,
    )

    if not tasks:
        logger.error("No tasks found matching the specified filters!")
        return

    # ── Dry run ──────────────────────────────────────────────────────
    if args.dry_run:
        print("\n[DRY RUN] Tasks that would be evaluated:\n")
        for i, task in enumerate(tasks):
            item = DatasetItem.model_validate(task)
            tc = item.generate_task_config()
            print(f"  {i+1}. {task['task_id']}")
            print(f"     Domain: {task['domain']}  Difficulty: {task.get('suggested_difficulty', '?')}")
            print(f"     URL:  {tc.url}")
            print(f"     Task: {tc.task[:120]}{'...' if len(tc.task)>120 else ''}")
            print()
        print(f"Total: {len(tasks)} tasks")
        return

    # ── Build environment and coordinator ────────────────────────────
    logger.info("Building browser environment and agent coordinator...")
    env, coordinator, base_agent = build_env_and_coordinator(config, args)

    # ── Run evaluation ───────────────────────────────────────────────
    all_results: List[Dict[str, Any]] = []
    start_time = time.time()

    for task_idx, task_row in enumerate(tasks):
        task_id = task_row["task_id"]
        logger.info(f"\n[Progress] Task {task_idx+1}/{len(tasks)}: {task_id}")

        try:
            result = evaluate_single_task(
                task_row=task_row,
                coordinator=coordinator,
                env=env,
                max_steps=args.max_steps,
                result_dir=result_dir,
                save_trace=not args.no_trace,
            )
            all_results.append(result)

        except Exception as e:
            logger.error(f"[Error] Task {task_id} failed: {e}")
            logger.error(traceback.format_exc())

            # ── Clean up Playwright context to prevent cascading errors ──
            # If env.reset() → setup() fails after sync_playwright().__enter__()
            # but before reset_finished is set to True, the Playwright internal
            # asyncio loop is left dangling.  The next env.reset() will then fail
            # with "Playwright Sync API inside asyncio loop".  Force-close it here.
            try:
                if hasattr(env, 'context_manager') and env.context_manager is not None:
                    env.context_manager.__exit__(None, None, None)
                    env.reset_finished = False
                    logger.info("[Cleanup] Closed orphaned Playwright context")
            except Exception as cleanup_err:
                logger.debug(f"[Cleanup] Playwright context cleanup: {cleanup_err}")

            # Record failure
            error_result = {
                "task_id": task_id,
                "domain": task_row["domain"],
                "difficulty": task_row.get("suggested_difficulty", "unknown"),
                "task_description": "",
                "start_url": "",
                "score": 0.0,
                "reasoning": f"Error: {str(e)}",
                "status": "ERROR",
                "total_steps": 0,
                "max_steps_allowed": args.max_steps,
                "visited_urls": [],
                "evaluator_class": "N/A",
                "eval_result_details": {},
            }
            all_results.append(error_result)

            # Save error to file
            error_file = Path(result_dir) / "errors.log"
            with open(error_file, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"Task: {task_id}\n")
                f.write(f"Error: {e}\n")
                f.write(traceback.format_exc())

    total_time = time.time() - start_time

    # ── Cleanup ──────────────────────────────────────────────────────
    try:
        env.close()
    except Exception:
        pass

    # ── Generate summary ─────────────────────────────────────────────
    if all_results:
        generate_summary_report(all_results, result_dir, total_time)
    else:
        logger.warning("No results to summarize!")


if __name__ == "__main__":
    main()
