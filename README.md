# Job Market Analysis

**Analyze real job posting data to see what skills are actually in demand.**

This repository contains the methodology and scripts behind our analysis: [The 2026 Job Market: What the Data Actually Shows](https://blog.rezscore.com/the-2026-job-market-what-the-data-actually-shows-c074b96a8325)

## Key Findings

- **"Prompt Engineer" is not a real job** — only 7,359 postings (vs 140,068 for Software Engineer)
- **ChatGPT skills pay $55K less than Anthropic skills** ($129K vs $184K)
- **NFT and Metaverse are dead** — under 300 job postings each
- **Crypto infrastructure is alive** — Stablecoins, DeFi, Ethereum paying $170K+
- **Enterprise tech beats trendy frameworks** — Oracle has 2x the jobs of React
- **"No degree required" jobs outnumber CS degree jobs 5:1**

## Data Source

We use the [Adzuna API](https://developer.adzuna.com/) which aggregates job postings from thousands of sources. Free tier gives 250 API calls/day.

## Quick Start

```bash
# Clone the repo
git clone https://github.com/rezscore/job-market-analysis.git
cd job-market-analysis

# Install dependencies
pip install requests python-dotenv

# Set up API keys (get free keys at https://developer.adzuna.com/)
export ADZUNA_APP_ID=your_app_id
export ADZUNA_API_KEY=your_api_key

# Run the analysis
python analyze_job_postings.py
```

## Usage

### Basic Analysis
```bash
python analyze_job_postings.py
```

### Custom Skills
```bash
python analyze_job_postings.py --skills "Python,React,Rust,Go,Kubernetes"
```

### Export Results
```bash
python analyze_job_postings.py --output results.json --csv results.csv
```

### Compare to Historical Data
```bash
# Save today's data
python analyze_job_postings.py --output data/2026_jan.json

# Later, compare to it
python analyze_job_postings.py --compare-to data/2026_jan.json
```

### Use USAJobs (No API Key Required)
```bash
python analyze_job_postings.py --source usajobs
```

## Sample Output

```
======================================================================
 JOB POSTING ANALYSIS
======================================================================

Skill                        Job Count   Avg Salary   Demand Idx
----------------------------------------------------------------------
Python                         137,176     $150,232        100.0
Software Engineer              140,068     $157,233        102.1
Machine Learning                77,214     $157,104         56.3
AI Engineer                     32,665     $172,681         23.8
Prompt Engineering               7,359     $164,840          5.4
ChatGPT                          3,670     $129,036          2.7
----------------------------------------------------------------------

AI-Related Jobs                191,575
AI % of Total                     8.9%

💰 Highest Paying Skills:
   • Anthropic: $183,906
   • MLOps: $182,259
   • GPT: $179,161
```

## What We Analyze

### AI & Machine Learning
- Prompt Engineering, ChatGPT, GPT, LLM, Claude, Anthropic
- Machine Learning, Deep Learning, MLOps
- TensorFlow, PyTorch, LangChain, OpenAI

### Programming Languages & Frameworks
- Python, JavaScript, TypeScript, Rust, Go
- React, Node.js, Django, FastAPI

### Cloud & Infrastructure
- AWS, Azure, GCP, Kubernetes, Docker, Terraform

### Crypto & Web3
- Blockchain, DeFi, Ethereum, Stablecoin, Smart Contracts
- NFT, Metaverse (spoiler: they're dead)

### Traditional Tech
- Excel, WordPress, PHP, SEO
- Oracle, SAP, Salesforce, Workday

## Methodology

1. **Data Source**: Adzuna API (aggregates from Indeed, LinkedIn, Glassdoor, etc.)
2. **Geography**: US market (configurable)
3. **Metrics**:
   - Job count: Total postings mentioning the skill
   - Avg salary: Mean salary from postings with salary data
   - Demand index: Relative to Python (baseline = 100)
4. **Limitations**:
   - Point-in-time snapshot (job market changes daily)
   - Salary data only from postings that include it (~30%)
   - Some skills have false positives (e.g., "Go" matches non-tech jobs)

## Contributing

Found an interesting skill to analyze? Open a PR or issue.

## License

MIT

## About

Built by [RezScore](https://ai.rezscore.com) — AI-powered resume analysis with 13M+ resumes analyzed since 2014.

**See where your skills stand**: [ai.rezscore.com/skills](https://ai.rezscore.com/skills)
