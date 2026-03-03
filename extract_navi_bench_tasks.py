import json
import os
from datetime import datetime
from datasets import load_dataset
import sys

# Ensure navi-bench package is in path
NAVI_BENCH_DIR = os.path.join(os.path.dirname(__file__), "navi-bench")
if NAVI_BENCH_DIR not in sys.path:
    sys.path.insert(0, NAVI_BENCH_DIR)

from navi_bench.base import DatasetItem

def extract_evaluator_name(eval_config: dict) -> str:
    """Extract just the class name from the evaluator _target_ string."""
    if not eval_config or "_target_" not in eval_config:
        return "Unknown"
    target = eval_config["_target_"]
    # Usually format is "navi_bench.domain.module.ClassName" or "...generate_task_config"
    # we want to get the class name or the module that handles it
    parts = target.split(".")
    
    # If it points to generate_task_config, the real evaluator is constructed inside that function.
    # We can peek at the module name instead.
    if parts[-1] == "generate_task_config":
        return parts[-2].replace("_", " ").title() + " (Custom Logic)"
        
    return parts[-1]

def main():
    print("Loading Navi-Bench dataset...")
    try:
        dataset = load_dataset("yutori-ai/navi-bench", split="validation")
        total_tasks = len(dataset)
        print(f"Loaded {total_tasks} tasks.")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    output_file = "navi_bench_tasks_summary.md"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Navi-Bench Tasks Summary\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Tasks: {total_tasks}\n\n")
        
        # Group by domain
        domains = {}
        for row in dataset:
            domain = row.get("domain", "unknown")
            if domain not in domains:
                domains[domain] = []
            domains[domain].append(row)
            
        # Write content grouped by domain
        for domain, tasks in sorted(domains.items()):
            f.write(f"## Domain: {domain.title()}\n")
            f.write(f"**Tasks in this domain**: {len(tasks)}\n\n")
            
            for index, row in enumerate(sorted(tasks, key=lambda x: x["task_id"])):
                task_id = row["task_id"]
                difficulty = row.get("suggested_difficulty", "?")
                
                # Parse config dynamically to get the fully rendered task description
                try:
                    item = DatasetItem.model_validate(row)
                    tc = item.generate_task_config()
                    task_desc = tc.task
                    
                    # Extract evaluator information
                    if hasattr(tc, 'eval_config'):
                        evaluator = extract_evaluator_name(tc.eval_config)
                    else:
                        eval_config = json.loads(row.get("task_generation_config_json", "{}"))
                        evaluator = extract_evaluator_name(eval_config)
                        
                    start_url = tc.url
                except Exception as e:
                    task_desc = f"[Error parsing task description: {e}]"
                    evaluator = "Unknown"
                    start_url = "Unknown"

                f.write(f"### {index + 1}. `{task_id}`\n")
                f.write(f"- **Difficulty**: {difficulty}\n")
                f.write(f"- **Start URL**: {start_url}\n")
                f.write(f"- **Task Instruction**: {task_desc}\n")
                f.write(f"- **Evaluator Method**: `{evaluator}`\n\n")
            
            f.write("---\n\n")

    print(f"Successfully wrote task details and evaluators to {output_file}")

if __name__ == "__main__":
    main()
