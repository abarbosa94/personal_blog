# Affiliation evidence search MCP server

This local MCP server exposes live, citation-bearing web search for the
affiliation reconciliation pipeline. It is intended to discover evidence, not
to infer a country from an organization's headquarters.

Tavily is the default provider. Brave remains available as an optional
alternative.

## Tools

- `search_web`: one general Brave Web Search request.
- `search_affiliation_evidence`: searches for an author, organization,
  optional paper title, and historical paper date using focused queries,
  deduplicates results, and ranks likely first-party sources first.

Every provider response is cached in SQLite with its retrieval timestamp.
Subsequent identical calls are reproducible until the configured cache expires.

## Install

From this experiment directory:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[mcp]"
```

Create a free Tavily API key and expose it to the MCP process:

```powershell
$env:TAVILY_API_KEY = "..."
```

The Tavily Researcher plan currently includes 1,000 credits per month without a
credit card. This server uses basic searches, which cost one credit each.

Optional settings:

```text
AFFILIATION_SEARCH_CACHE=artifacts/affiliation-search/search-cache.sqlite3
AFFILIATION_SEARCH_CACHE_DAYS=30
AFFILIATION_SEARCH_PROVIDER=tavily
```

Do not commit API keys or place them in MCP tool arguments.

## Run directly

```powershell
.\.venv\Scripts\affiliation-search-mcp.exe
```

The server uses stdio. It writes no logs or other text to stdout.

## Codex project configuration

The repository includes `.codex/config.toml`, which registers this stdio server
for trusted Codex sessions in this project. It forwards
`TAVILY_API_KEY` from the local environment and stores only non-secret
cache settings in the configuration.

Set the key before starting or restarting Codex:

```powershell
$env:TAVILY_API_KEY = "..."
```

Then restart the Codex client and inspect `/mcp`, or run `codex mcp list`.

To use Brave instead, set `AFFILIATION_SEARCH_PROVIDER=brave` and provide
`BRAVE_SEARCH_API_KEY`.

## Evidence policy for V9

An affiliation branch may be accepted only when the evidence:

1. identifies the author and organization;
2. explicitly states a branch location;
3. overlaps the paper date or clearly describes historical employment;
4. comes from a sufficiently authoritative source; and
5. has no unresolved conflict with equally strong evidence.

Otherwise the MCP result should be displayed as a suggestion in manual review.

The batch augmenter applies the same policy by default:

```powershell
.\.venv\Scripts\python.exe -m conference_pipeline.v2_pipeline augment `
  data/processed/input.jsonl `
  --output data/processed/output-v9.jsonl
```

Use `--no-search-evidence` for an explicitly offline run. Use the separate
`search-augment` command to apply V9 evidence to an already frozen enriched
file without repeating PDF, NER, or ROR processing.

Search escalation runs only for affiliations that still lack a country. A country is
assigned automatically only when two independent result pages explicitly link
the same author and organization to the same country, at least one result is
first-party/academic, and at least one contains period-relevant evidence.
As a narrower medium-confidence exception, an author's own LinkedIn `/in/`
profile may stand alone when it explicitly names the author, organization, and
country and was retrieved within one year of the paper. LinkedIn company pages,
third-party profiles, older undated profiles, and conflicting evidence do not
qualify.
Conflicts and weaker evidence are retained in `search_affiliation_evidence`
diagnostics with URLs, snippets, retrieval time, confidence, and rationale.
