#!/usr/bin/env python3
"""
Job Posting Skill Demand Analyzer
=================================

Analyzes skill demand trends from public job posting APIs.
Can be published to GitHub for transparency.

Data Sources:
- Adzuna API (free tier: 250 calls/day)
- USAJobs API (government jobs, unlimited)

Usage:
    # Basic usage (requires ADZUNA_APP_ID and ADZUNA_API_KEY env vars)
    python analyze_job_postings.py

    # Search specific skills
    python analyze_job_postings.py --skills "python,react,prompt engineering,machine learning"

    # Compare to historical (if you have cached data)
    python analyze_job_postings.py --compare-to data/2023_job_data.json

    # Export results
    python analyze_job_postings.py --output results.json --csv results.csv

    # Use USAJobs instead of Adzuna
    python analyze_job_postings.py --source usajobs

    # Debug mode
    python analyze_job_postings.py --debug

Requirements:
    pip install requests python-dotenv

API Setup:
    Adzuna: https://developer.adzuna.com/ (free account gives 250 calls/day)
    USAJobs: https://developer.usajobs.gov/ (free, requires email)

Author: RezScore (https://ai.rezscore.com)
License: MIT
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

try:
    import requests
except ImportError:
    print("Error: requests library required. Run: pip install requests")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional


# =============================================================================
# CONFIGURATION
# =============================================================================

# Skills to analyze - add/remove as needed
DEFAULT_SKILLS = [
    # AI/ML Skills
    "prompt engineering",
    "ChatGPT",
    "GPT",
    "LLM",
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "generative AI",
    "LangChain",
    "OpenAI",
    "Claude",
    "Anthropic",
    "MLOps",
    "TensorFlow",
    "PyTorch",

    # Data Skills
    "data engineering",
    "data science",
    "Python",
    "SQL",
    "Pandas",
    "Databricks",
    "Snowflake",

    # Cloud/DevOps
    "AWS",
    "Azure",
    "GCP",
    "Kubernetes",
    "Docker",
    "Terraform",

    # Traditional (for comparison)
    "Excel",
    "WordPress",
    "PHP",
    "SEO",
    "copywriting",
]

# API Configuration
ADZUNA_BASE_URL = "https://api.adzuna.com/v1/api/jobs"
USAJOBS_BASE_URL = "https://data.usajobs.gov/api/search"


# =============================================================================
# API CLIENTS
# =============================================================================

class AdzunaClient:
    """Client for Adzuna Job Search API."""

    def __init__(self, app_id: str, api_key: str, country: str = "us"):
        self.app_id = app_id
        self.api_key = api_key
        self.country = country
        self.base_url = f"{ADZUNA_BASE_URL}/{country}/search/1"
        self.rate_limit_remaining = 250

    def search(self, skill: str, results_per_page: int = 50) -> Dict:
        """Search for jobs mentioning a skill."""
        params = {
            "app_id": self.app_id,
            "app_key": self.api_key,
            "results_per_page": results_per_page,
            "what": skill,
            "content-type": "application/json",
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            return {
                "skill": skill,
                "count": data.get("count", 0),
                "mean_salary": data.get("mean", None),
                "results": data.get("results", [])[:10],  # Sample of jobs
                "source": "adzuna",
                "timestamp": datetime.now().isoformat(),
            }
        except requests.exceptions.RequestException as e:
            print(f"  Error searching for '{skill}': {e}")
            return {"skill": skill, "count": 0, "error": str(e)}

    def get_salary_histogram(self, skill: str) -> Dict:
        """Get salary distribution for a skill."""
        url = f"{ADZUNA_BASE_URL}/{self.country}/histogram"
        params = {
            "app_id": self.app_id,
            "app_key": self.api_key,
            "what": skill,
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException:
            return {}


class USAJobsClient:
    """Client for USAJobs API (US Government jobs)."""

    def __init__(self, email: str, api_key: Optional[str] = None):
        self.email = email
        self.api_key = api_key
        self.base_url = USAJOBS_BASE_URL

    def search(self, skill: str, results_per_page: int = 50) -> Dict:
        """Search for government jobs mentioning a skill."""
        headers = {
            "Host": "data.usajobs.gov",
            "User-Agent": self.email,
        }
        if self.api_key:
            headers["Authorization-Key"] = self.api_key

        params = {
            "Keyword": skill,
            "ResultsPerPage": results_per_page,
        }

        try:
            response = requests.get(
                self.base_url, headers=headers, params=params, timeout=30
            )
            response.raise_for_status()
            data = response.json()

            search_result = data.get("SearchResult", {})
            count = search_result.get("SearchResultCountAll", 0)
            items = search_result.get("SearchResultItems", [])

            # Extract salary data from results
            salaries = []
            for item in items:
                job = item.get("MatchedObjectDescriptor", {})
                salary_min = job.get("PositionRemuneration", [{}])[0].get("MinimumRange")
                salary_max = job.get("PositionRemuneration", [{}])[0].get("MaximumRange")
                if salary_min and salary_max:
                    try:
                        salaries.append((float(salary_min) + float(salary_max)) / 2)
                    except (ValueError, TypeError):
                        pass

            mean_salary = sum(salaries) / len(salaries) if salaries else None

            return {
                "skill": skill,
                "count": count,
                "mean_salary": mean_salary,
                "results": items[:10],
                "source": "usajobs",
                "timestamp": datetime.now().isoformat(),
            }
        except requests.exceptions.RequestException as e:
            print(f"  Error searching for '{skill}': {e}")
            return {"skill": skill, "count": 0, "error": str(e)}


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_skills(
    client,
    skills: List[str],
    delay: float = 0.5,
    debug: bool = False
) -> List[Dict]:
    """Analyze demand for a list of skills."""
    results = []

    print(f"\nAnalyzing {len(skills)} skills...")
    print("-" * 50)

    for i, skill in enumerate(skills, 1):
        print(f"[{i}/{len(skills)}] Searching: {skill}...", end=" ", flush=True)

        result = client.search(skill)
        results.append(result)

        count = result.get("count", 0)
        salary = result.get("mean_salary")
        salary_str = f"${salary:,.0f}" if salary else "N/A"

        print(f"{count:,} jobs (avg salary: {salary_str})")

        if debug and result.get("results"):
            print(f"       Sample: {result['results'][0].get('title', 'N/A')[:50]}")

        # Rate limiting
        time.sleep(delay)

    return results


def calculate_demand_index(results: List[Dict], baseline_skill: str = "Python") -> List[Dict]:
    """Calculate relative demand index compared to a baseline skill."""
    # Find baseline count
    baseline_count = 1
    for r in results:
        if r["skill"].lower() == baseline_skill.lower():
            baseline_count = max(r.get("count", 1), 1)
            break

    # Calculate index
    for r in results:
        r["demand_index"] = (r.get("count", 0) / baseline_count) * 100

    return results


def compare_periods(
    current: List[Dict],
    historical: List[Dict]
) -> List[Dict]:
    """Compare current results to historical data."""
    historical_map = {r["skill"].lower(): r for r in historical}

    for r in current:
        skill_lower = r["skill"].lower()
        if skill_lower in historical_map:
            hist = historical_map[skill_lower]
            hist_count = hist.get("count", 0)
            curr_count = r.get("count", 0)

            if hist_count > 0:
                r["change_pct"] = ((curr_count - hist_count) / hist_count) * 100
            elif curr_count > 0:
                r["change_pct"] = float("inf")  # New skill
            else:
                r["change_pct"] = 0

            r["historical_count"] = hist_count
            r["historical_timestamp"] = hist.get("timestamp")

    return current


def print_results(results: List[Dict], title: str = "JOB POSTING ANALYSIS"):
    """Print formatted results."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

    # Sort by count
    sorted_results = sorted(results, key=lambda x: x.get("count", 0), reverse=True)

    # Header
    print(f"\n{'Skill':<25} {'Job Count':>12} {'Avg Salary':>12} {'Demand Idx':>12}")
    print("-" * 70)

    for r in sorted_results:
        skill = r["skill"][:24]
        count = f"{r.get('count', 0):,}"
        salary = f"${r.get('mean_salary', 0):,.0f}" if r.get("mean_salary") else "N/A"
        index = f"{r.get('demand_index', 0):.1f}" if "demand_index" in r else "-"

        # Add change indicator if available
        change = r.get("change_pct")
        if change is not None:
            if change == float("inf"):
                change_str = " 🆕"
            elif change > 20:
                change_str = f" ↑{change:.0f}%"
            elif change < -20:
                change_str = f" ↓{abs(change):.0f}%"
            else:
                change_str = ""
        else:
            change_str = ""

        print(f"{skill:<25} {count:>12} {salary:>12} {index:>12}{change_str}")

    # Summary statistics
    total_jobs = sum(r.get("count", 0) for r in results)
    avg_salary = [r.get("mean_salary") for r in results if r.get("mean_salary")]
    avg_salary = sum(avg_salary) / len(avg_salary) if avg_salary else 0

    print("-" * 70)
    print(f"{'TOTAL':<25} {total_jobs:>12,}")
    print(f"{'Average Salary':<25} {'':>12} ${avg_salary:>10,.0f}")

    # Highlight AI skills
    ai_skills = ["prompt engineering", "chatgpt", "gpt", "llm", "machine learning",
                 "generative ai", "langchain", "openai", "claude", "mlops"]
    ai_results = [r for r in results if r["skill"].lower() in ai_skills]

    if ai_results:
        ai_total = sum(r.get("count", 0) for r in ai_results)
        print(f"\n{'AI-Related Jobs':<25} {ai_total:>12,}")
        print(f"{'AI % of Total':<25} {(ai_total/max(total_jobs,1)*100):>11.1f}%")


def export_results(results: List[Dict], json_path: str = None, csv_path: str = None):
    """Export results to JSON and/or CSV."""
    if json_path:
        with open(json_path, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "results": results,
            }, f, indent=2, default=str)
        print(f"\nExported JSON to: {json_path}")

    if csv_path:
        import csv
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "skill", "count", "mean_salary", "demand_index",
                "change_pct", "source", "timestamp"
            ])
            writer.writeheader()
            for r in results:
                writer.writerow({
                    "skill": r.get("skill"),
                    "count": r.get("count"),
                    "mean_salary": r.get("mean_salary"),
                    "demand_index": r.get("demand_index"),
                    "change_pct": r.get("change_pct"),
                    "source": r.get("source"),
                    "timestamp": r.get("timestamp"),
                })
        print(f"Exported CSV to: {csv_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze job posting skill demand trends",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--skills",
        type=str,
        help="Comma-separated list of skills to analyze"
    )
    parser.add_argument(
        "--source",
        choices=["adzuna", "usajobs"],
        default="adzuna",
        help="Data source to use (default: adzuna)"
    )
    parser.add_argument(
        "--compare-to",
        type=str,
        help="Path to historical JSON data for comparison"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file path"
    )
    parser.add_argument(
        "--csv",
        type=str,
        help="Output CSV file path"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between API calls in seconds (default: 0.5)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output"
    )

    args = parser.parse_args()

    # Parse skills
    if args.skills:
        skills = [s.strip() for s in args.skills.split(",")]
    else:
        skills = DEFAULT_SKILLS

    # Initialize client
    if args.source == "adzuna":
        app_id = os.environ.get("ADZUNA_APP_ID")
        api_key = os.environ.get("ADZUNA_API_KEY")

        if not app_id or not api_key:
            print("Error: ADZUNA_APP_ID and ADZUNA_API_KEY environment variables required")
            print("\nGet free API keys at: https://developer.adzuna.com/")
            print("\nAlternatively, use USAJobs (no API key required):")
            print("  python analyze_job_postings.py --source usajobs")
            sys.exit(1)

        client = AdzunaClient(app_id, api_key)
        print(f"Using Adzuna API (US market)")

    else:  # usajobs
        email = os.environ.get("USAJOBS_EMAIL", "user@example.com")
        api_key = os.environ.get("USAJOBS_API_KEY")  # Optional
        client = USAJobsClient(email, api_key)
        print(f"Using USAJobs API (US Government jobs)")

    print(f"Analyzing {len(skills)} skills...")
    print(f"Timestamp: {datetime.now().isoformat()}")

    # Run analysis
    results = analyze_skills(client, skills, delay=args.delay, debug=args.debug)

    # Calculate demand index
    results = calculate_demand_index(results)

    # Compare to historical if provided
    if args.compare_to:
        try:
            with open(args.compare_to) as f:
                historical = json.load(f).get("results", [])
            results = compare_periods(results, historical)
            print(f"\nCompared to historical data from: {args.compare_to}")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load historical data: {e}")

    # Print results
    print_results(results)

    # Export if requested
    export_results(results, json_path=args.output, csv_path=args.csv)

    # Insights
    print("\n" + "=" * 70)
    print(" KEY INSIGHTS")
    print("=" * 70)

    # Top growing skills
    if any(r.get("change_pct") for r in results):
        growing = [r for r in results if r.get("change_pct", 0) > 20]
        if growing:
            print("\n📈 Fastest Growing Skills:")
            for r in sorted(growing, key=lambda x: x.get("change_pct", 0), reverse=True)[:5]:
                print(f"   • {r['skill']}: +{r['change_pct']:.0f}%")

        declining = [r for r in results if r.get("change_pct", 0) < -20]
        if declining:
            print("\n📉 Declining Skills:")
            for r in sorted(declining, key=lambda x: x.get("change_pct", 0))[:5]:
                print(f"   • {r['skill']}: {r['change_pct']:.0f}%")

    # Highest paying
    with_salary = [r for r in results if r.get("mean_salary")]
    if with_salary:
        print("\n💰 Highest Paying Skills:")
        for r in sorted(with_salary, key=lambda x: x.get("mean_salary", 0), reverse=True)[:5]:
            print(f"   • {r['skill']}: ${r['mean_salary']:,.0f}")

    print("\n" + "-" * 70)
    print("Data source: " + ("Adzuna" if args.source == "adzuna" else "USAJobs"))
    print("For methodology, see: https://github.com/rezscore/job-market-analysis")
    print("-" * 70)


if __name__ == "__main__":
    main()
