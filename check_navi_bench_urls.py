"""
Navi-Bench URL Accessibility Checker
=====================================
Checks whether all 100 Navi-Bench task URLs are reachable from the current machine.
This helps diagnose whether task failures are due to network/access issues or
browser environment problems.

Usage:
    python check_navi_bench_urls.py
    python check_navi_bench_urls.py --timeout 10
    python check_navi_bench_urls.py --domains craigslist,apartments
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

# Add navi-bench to path
NAVI_BENCH_DIR = os.path.join(os.path.dirname(__file__), "navi-bench")
if NAVI_BENCH_DIR not in sys.path:
    sys.path.insert(0, NAVI_BENCH_DIR)

try:
    import requests
except ImportError:
    print("[!] 'requests' not installed. Falling back to urllib.")
    requests = None

from datasets import load_dataset
from navi_bench.base import DatasetItem


# ── ANSI Colors ──────────────────────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
GRAY = "\033[90m"
BOLD = "\033[1m"
NC = "\033[0m"


def check_url(url: str, timeout: int = 15) -> dict:
    """Check if a URL is accessible. Returns a dict with status info."""
    result = {
        "url": url,
        "accessible": False,
        "status_code": None,
        "redirect_url": None,
        "error": None,
        "response_time_ms": None,
    }

    start = time.time()
    try:
        if requests:
            resp = requests.get(
                url,
                timeout=timeout,
                allow_redirects=True,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                },
            )
            result["status_code"] = resp.status_code
            result["accessible"] = resp.status_code < 400
            if resp.url != url:
                result["redirect_url"] = resp.url
        else:
            import urllib.request
            import urllib.error

            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36"
                },
            )
            resp = urllib.request.urlopen(req, timeout=timeout)
            result["status_code"] = resp.getcode()
            result["accessible"] = True
            if resp.url != url:
                result["redirect_url"] = resp.url

    except Exception as e:
        err_name = type(e).__name__
        if "Timeout" in err_name or "timeout" in str(e).lower():
            result["error"] = "TIMEOUT"
        elif "Connection" in err_name or "refused" in str(e).lower():
            result["error"] = "CONNECTION_REFUSED"
        elif "SSL" in err_name:
            result["error"] = "SSL_ERROR"
        else:
            result["error"] = f"{err_name}: {str(e)[:120]}"

    result["response_time_ms"] = int((time.time() - start) * 1000)
    return result


def main():
    parser = argparse.ArgumentParser(description="Check Navi-Bench task URL accessibility")
    parser.add_argument("--timeout", type=int, default=15, help="HTTP timeout in seconds (default: 15)")
    parser.add_argument("--domains", type=str, default=None, help="Comma-separated list of domains to check")
    parser.add_argument("--workers", type=int, default=10, help="Number of parallel workers (default: 10)")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")
    args = parser.parse_args()

    # ── Load tasks ───────────────────────────────────────────────────
    print(f"\n{BOLD}{'='*70}{NC}")
    print(f"{BOLD}  Navi-Bench URL Accessibility Checker{NC}")
    print(f"{BOLD}{'='*70}{NC}\n")

    print(f"{CYAN}[*] Loading Navi-Bench dataset...{NC}")
    dataset = load_dataset("yutori-ai/navi-bench", split="validation")
    print(f"{GREEN}[+] Loaded {len(dataset)} tasks{NC}")

    # ── Generate task configs (resolves dynamic URLs/dates) ──────────
    print(f"{CYAN}[*] Generating task configs (resolving dynamic dates)...{NC}")
    tasks = []
    for row in dataset:
        row_dict = dict(row)
        if args.domains:
            allowed = [d.strip() for d in args.domains.split(",")]
            if row_dict.get("domain") not in allowed:
                continue
        try:
            item = DatasetItem.model_validate(row_dict)
            config = item.generate_task_config()
            tasks.append({
                "task_id": row_dict["task_id"],
                "domain": row_dict.get("domain", "unknown"),
                "difficulty": row_dict.get("suggested_difficulty", "unknown"),
                "url": config.url,
                "task": config.task[:100],
            })
        except Exception as e:
            tasks.append({
                "task_id": row_dict["task_id"],
                "domain": row_dict.get("domain", "unknown"),
                "difficulty": row_dict.get("suggested_difficulty", "unknown"),
                "url": "CONFIG_ERROR",
                "task": f"Error: {e}",
            })

    print(f"{GREEN}[+] {len(tasks)} tasks to check{NC}")

    # ── Deduplicate URLs ─────────────────────────────────────────────
    # Multiple tasks may share the same start URL
    unique_urls = {}
    for t in tasks:
        url = t["url"]
        if url != "CONFIG_ERROR" and url not in unique_urls:
            unique_urls[url] = t["domain"]

    print(f"{CYAN}[*] {len(unique_urls)} unique URLs to test{NC}")

    # ── Domain breakdown ────────────────────────────────────────────
    domain_counts = defaultdict(int)
    for t in tasks:
        domain_counts[t["domain"]] += 1
    print(f"\n{GRAY}  Domain breakdown:{NC}")
    for domain, count in sorted(domain_counts.items()):
        print(f"{GRAY}    {domain}: {count} tasks{NC}")

    # ── Check URLs in parallel ───────────────────────────────────────
    print(f"\n{CYAN}[*] Checking URL accessibility (timeout={args.timeout}s, workers={args.workers})...{NC}\n")

    url_results = {}
    checked = 0
    total = len(unique_urls)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_url = {
            executor.submit(check_url, url, args.timeout): url
            for url in unique_urls
        }

        for future in as_completed(future_to_url):
            url = future_to_url[future]
            result = future.result()
            url_results[url] = result
            checked += 1

            # Print progress
            domain = unique_urls[url]
            status_icon = f"{GREEN}✓{NC}" if result["accessible"] else f"{RED}✗{NC}"
            status_detail = f"HTTP {result['status_code']}" if result["status_code"] else result["error"]
            time_str = f"{result['response_time_ms']}ms" if result["response_time_ms"] else "?"

            print(
                f"  [{checked:3d}/{total}] {status_icon} "
                f"{GRAY}[{domain:16s}]{NC} "
                f"{url[:65]:65s} "
                f"{status_detail:25s} "
                f"{GRAY}{time_str}{NC}"
            )

    # ── Map results back to tasks ────────────────────────────────────
    for t in tasks:
        url = t["url"]
        if url in url_results:
            t["check_result"] = url_results[url]
        else:
            t["check_result"] = {
                "accessible": False,
                "error": "CONFIG_ERROR",
                "status_code": None,
                "response_time_ms": None,
            }

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{BOLD}{'='*70}{NC}")
    print(f"{BOLD}  RESULTS SUMMARY{NC}")
    print(f"{BOLD}{'='*70}{NC}")

    # Overall
    accessible_urls = sum(1 for r in url_results.values() if r["accessible"])
    blocked_urls = total - accessible_urls
    print(f"\n  {BOLD}Unique URLs:{NC}  {total}")
    print(f"  {GREEN}Accessible:{NC}   {accessible_urls}")
    print(f"  {RED}Blocked:{NC}      {blocked_urls}")

    # Per domain
    print(f"\n  {BOLD}{'Domain':<20} {'Tasks':>6} {'URLs OK':>8} {'URLs Fail':>10} {'Status'}{NC}")
    print(f"  {'─'*65}")

    domain_summary = defaultdict(lambda: {"tasks": 0, "ok": 0, "fail": 0, "errors": []})
    for t in tasks:
        d = t["domain"]
        domain_summary[d]["tasks"] += 1
        if t["check_result"]["accessible"]:
            domain_summary[d]["ok"] += 1
        else:
            domain_summary[d]["fail"] += 1
            err = t["check_result"].get("error", "Unknown")
            if err and err not in domain_summary[d]["errors"]:
                domain_summary[d]["errors"].append(err)

    for domain, stats in sorted(domain_summary.items()):
        status = f"{GREEN}ALL OK{NC}" if stats["fail"] == 0 else f"{RED}BLOCKED{NC}"
        print(
            f"  {domain:<20} {stats['tasks']:>6} {stats['ok']:>8} {stats['fail']:>10}   {status}"
        )
        if stats["errors"]:
            for err in stats["errors"][:3]:
                print(f"  {GRAY}{'':20} → {err}{NC}")

    # ── List blocked URLs ────────────────────────────────────────────
    blocked = [r for r in url_results.values() if not r["accessible"]]
    if blocked:
        print(f"\n  {RED}{BOLD}Blocked URLs:{NC}")
        for r in sorted(blocked, key=lambda x: x["url"]):
            print(f"    {RED}✗{NC} {r['url']}")
            err_detail = r['error'] if r['error'] else "HTTP " + str(r['status_code'])
            print(f"      {GRAY}Error: {err_detail}{NC}")

    # ── Diagnosis ────────────────────────────────────────────────────
    print(f"\n  {BOLD}Diagnosis:{NC}")
    if blocked_urls == 0:
        print(f"  {GREEN}All URLs reachable! If tasks still fail, the issue is in the{NC}")
        print(f"  {GREEN}browser environment or agent execution, not network access.{NC}")
    elif blocked_urls == total:
        print(f"  {RED}ALL URLs blocked! Possible causes:{NC}")
        print(f"  {RED}  - Network proxy/firewall blocking outbound HTTP{NC}")
        print(f"  {RED}  - DNS resolution failure{NC}")
        print(f"  {RED}  - No internet connectivity{NC}")
    else:
        blocked_domains = [d for d, s in domain_summary.items() if s["fail"] > 0]
        print(f"  {YELLOW}Partial access. Blocked domains: {', '.join(blocked_domains)}{NC}")
        print(f"  {YELLOW}Some sites may be geo-restricted or rate-limited.{NC}")

    print(f"\n{BOLD}{'='*70}{NC}\n")

    # ── Save JSON report ─────────────────────────────────────────────
    output_file = args.output or "navi_bench_url_check.json"
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_tasks": len(tasks),
        "unique_urls": total,
        "accessible": accessible_urls,
        "blocked": blocked_urls,
        "domain_summary": {
            d: {k: v for k, v in s.items()}
            for d, s in sorted(domain_summary.items())
        },
        "tasks": [
            {
                "task_id": t["task_id"],
                "domain": t["domain"],
                "difficulty": t["difficulty"],
                "url": t["url"],
                "accessible": t["check_result"]["accessible"],
                "status_code": t["check_result"].get("status_code"),
                "error": t["check_result"].get("error"),
                "response_time_ms": t["check_result"].get("response_time_ms"),
            }
            for t in tasks
        ],
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"{GREEN}[+] Detailed report saved to: {output_file}{NC}\n")


if __name__ == "__main__":
    main()
