# Lead Enrichment Agent

This is a Python CLI that takes a LinkedIn profile URL, searches the public web for likely matching profiles/websites, verifies identity overlap, and writes:

- a Markdown report
- a JSON payload
- a grounded outreach draft
- an optional agent trace

It does not use any paid APIs.
It should only be used on public data and with a manual review step before outreach.

## What it does

1. Accepts a LinkedIn profile URL as input.
2. Tries to extract seed facts from the LinkedIn URL and the public page metadata:
   - full name
   - headline / role
   - company
   - location
3. Runs free public-web searches using DuckDuckGo HTML results.
4. Plans search queries for profile discovery.
5. Fetches candidate pages from social networks and websites.
6. Filters out generic search results, directory pages, and weak content pages that are not real profile candidates.
7. Verifies each candidate with heuristic guardrails and, when available, a Groq-hosted model.
8. Produces a report with:
   - lead summary
   - accepted matches
   - rejected/manual-review candidates
   - grounded identity decision
   - grounded outreach message

## Execution Modes

- `auto`: use Groq if `GROQ_API_KEY` is configured and the API is reachable, otherwise fall back to heuristics
- `agent`: prefer the Groq-backed agent flow
- `fallback`: run the deterministic pipeline only

The agent mode uses a hosted free-tier model for:

- search planning
- structured evidence extraction
- conservative identity verification
- final lead summary and outreach drafting

The fallback mode keeps the project usable when Groq is not configured.

## Why the identity check is reasonable

The hard part of this assignment is not finding URLs. It is deciding whether the URLs belong to the same lead.

This project uses multiple signals:

- profile-like URL patterns on supported social platforms
- person/profile structured data on pages that expose it
- exact full-name match
- first-name + last-name match
- username overlap with the LinkedIn slug
- shared company words
- shared role/headline words
- shared location words
- penalties for partial/conflicting name evidence
- penalties for search pages, topic pages, and other generic results

That means:

- `strong` is useful evidence, not proof
- `probable` should usually be manually reviewed
- `weak` and `unlikely` should not be trusted for outreach without checking
- if no profile-like matches are found, the tool should say that instead of inventing a confident result

## Limitations

- LinkedIn often rate-limits or blocks scraping. The script handles this by falling back to the LinkedIn URL slug and public search results.
- If the LinkedIn URL is a placeholder or does not expose public metadata, the enrichment quality will be limited.
- Groq is optional, but agent mode is strongest when `GROQ_API_KEY` is configured and a supported model such as `llama-3.1-8b-instant` is available.
- Search engine HTML can change over time. This is a prototype for an interview assignment, not a production crawler.
- Some websites block automated requests.
- Even in agent mode, identity decisions are conservative and should be manually reviewed before outreach.

## Setup

Install Python 3.10+ and then:

```bash
pip install -r requirements.txt
```

Optional for agent mode:

1. Create a Groq API key at `https://console.groq.com/keys`
2. In PowerShell, set the environment variable:

```powershell
$env:GROQ_API_KEY="your_groq_api_key_here"
```

To persist it for future terminals on Windows:

```powershell
[System.Environment]::SetEnvironmentVariable("GROQ_API_KEY", "your_groq_api_key_here", "User")
```

Quick connection test:

```powershell
$headers = @{ Authorization = "Bearer $env:GROQ_API_KEY" }
Invoke-RestMethod -Uri "https://api.groq.com/openai/v1/models" -Headers $headers
```

## Usage

```bash
python lead_enricher.py
```

The script will prompt for:

- `Enter LinkedIn profile URL:`
- `Enter Markdown output filename [lead_report.md]:`
- `Enter JSON output filename [lead_report.json]:`

Press Enter to keep the default output filenames.

You can still run it directly with arguments:

```bash
python lead_enricher.py "https://www.linkedin.com/in/example-person-123456/"
python lead_enricher.py "https://www.linkedin.com/in/example-person-123456/" --markdown-out custom_report.md --json-out custom_report.json
python lead_enricher.py --markdown-out custom_report.md --json-out custom_report.json
python lead_enricher.py --mode agent --model llama-3.1-8b-instant --trace-out agent_trace.json
python lead_enricher.py --mode fallback
```

## Output

`lead_report.md` includes:

- the LinkedIn seed data
- execution mode and model info
- accepted matches
- manual-review and rejected candidates
- identity decision
- grounded outreach message

`lead_report.json` includes the same information in structured form.
If `--trace-out` is used, the agent trace is also written separately.

## How to explain this in the interview

If you need to present the approach, the clean framing is:

- I avoided paid APIs by using public web search and direct page fetches.
- I kept heuristic guardrails in place so the model cannot freely hallucinate identity matches.
- I treated identity resolution as a grounded decision with accepted, rejected, and manual-review outcomes.
- I made the workflow agentic by adding search planning, tool use, evidence extraction, verification, and grounded synthesis.
- I preserved a deterministic fallback path so the project still works without a model provider configured.

## Future improvements

- add async fetching and rate limiting
- add richer domain-specific extractors for GitHub, Medium, X, personal sites, etc.
- add caching to avoid repeated searches
- add an optional local LLM for better summaries if the environment allows it
