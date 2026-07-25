# Scaling My Posts experiment

This directory contains the reproducible translation experiment supporting the
“Scaling My Posts with AI Agents” article. Generated, report-ready evidence remains
under `posts/data/` so the article and website can consume it directly.

Scholarly references used by the experiment and its post live in `references.bib`.
Quarto notebooks can reuse them by adding this to their YAML front matter:

```yaml
bibliography: ../experiments/scaling-my-posts/references.bib
```

Use Pandoc citation keys in Markdown, for example:

```markdown
LaBSE provides multilingual sentence embeddings [@feng2022labse].
```

Quarto generates the formatted citation and bibliography during rendering.

## Target CI workflow

Translation does not run during the source-content PR. The source PR is reviewed
and merged first. Its push to `main` then triggers the translation job, which:

1. extracts and translates the source post;
2. validates and renders the translated document;
3. publishes a temporary preview; and
4. opens a separate translation PR for human review.

Merging or closing the translation PR removes its temporary preview. Production
publication happens only after the translation PR is reviewed and merged.

## Layout

```text
experiments/scaling-my-posts/
├── docker/
│   ├── Dockerfile.ci
│   └── docker-compose.ci.yml
├── requirements/
│   ├── benchmark.txt
│   ├── review-ui-test.txt
│   └── xcomet.txt
├── src/
│   ├── judge_factory.py
│   ├── judge_interface.py
│   ├── judge_providers.py
│   ├── local_ci.py
│   ├── prompts.py
│   ├── translation_benchmark.py
│   ├── translation_eval.py
│   └── translation_review_server.py
├── tests/
└── ui/
    ├── alignment/
    └── human/
```

The judge boundary is intentionally split:

- `judge_interface.py` defines the provider-neutral adapter contract and result types.
- `judge_providers.py` implements Kimi and OpenAI integrations.
- `judge_factory.py` owns provider registration and construction.
- `prompts.py` is the single source of truth for the versioned MQM and pairwise prompts.

## Commands

Run commands from the repository root:

```powershell
$experiment = "experiments/scaling-my-posts"

uv venv --python 3.12 .venv-benchmark
uv pip install --python .venv-benchmark\Scripts\python.exe `
  -r "$experiment/requirements/benchmark.txt"

.venv-benchmark\Scripts\python.exe "$experiment/src/translation_eval.py" align
.venv-benchmark\Scripts\python.exe "$experiment/src/translation_review_server.py"
.venv-benchmark\Scripts\python.exe "$experiment/src/translation_eval.py" freeze
.venv-benchmark\Scripts\python.exe "$experiment/src/translation_eval.py" predict --threads 16
.venv-benchmark\Scripts\python.exe "$experiment/src/translation_eval.py" judge --dry-run
```

Run the experiment tests without invoking paid providers:

```powershell
.venv-benchmark\Scripts\python.exe -m pytest -q experiments/scaling-my-posts/tests
```

## NLLB screening criterion

NLLB was screened out of the paid LLM-judge stage after the sentence-level pilot.
This was an exploratory decision, not a preregistered exclusion. To make it auditable,
`src/screening.py` applies the retrospective rule used in the article:

- Flag a prediction when any normalized four-token n-gram occurs at least four times.
- Fail a model-direction when more than 5% of its segments are flagged.
- Require a production candidate to pass in both translation directions.

Reproduce the segment-level flags with:

```powershell
.venv-benchmark\Scripts\python.exe `
  experiments/scaling-my-posts/src/screening.py `
  posts/data/translation-eval-predictions.csv
```

The rule flags 0 of 36 NLLB English-to-Portuguese predictions and 4 of 36
Portuguese-to-English predictions (11.1%): `p05-a04`, `p06-a02`, `p08-a01`, and
`p08-a03`. NLLB therefore failed the Portuguese-to-English screen and was excluded
from the bidirectional paid-judge comparison.

The evaluated checkpoint was `facebook/nllb-200-distilled-600M` at revision
`f8d333a098d19b4fd9a8b18f94170487ad3f821d`. It ran on CPU with Transformers
4.53.2, forced the target-language BOS token, used four-beam generation, truncated
inputs at 512 tokens, and set `max_new_tokens` to
`min(512, max(32, 2 * longest_source_tokens + 32))`.

## Refactoring plan

This change establishes the experiment boundary and separates prompts, the judge
interface, provider implementations, and factory construction. The remaining
`translation_eval.py` pipeline should be decomposed in behavior-preserving slices:

1. Move shared constants and Pydantic schemas into `config.py` and `schemas.py`.
2. Move notebook parsing, segmentation, LaBSE similarity, and dynamic programming
   into `alignment.py`.
3. Move translator orchestration and checkpoint validation into `prediction.py`,
   reusing model definitions from `translation_benchmark.py`.
4. Move judge-job construction, resumable execution, and validation into `judging.py`.
5. Move MQM, pairwise, xCOMET, bootstrap, and human-agreement aggregation into
   `reporting.py`.
6. Leave argument parsing and command dispatch only in `cli.py`, with a minimal
   `run.py` entry point.

Each slice should move its focused tests at the same time. Preserve existing CLI
arguments and artifact schemas until the article is published so old experiment
results remain reproducible.

## Rehearse the runner locally with Docker

The local rehearsal runs the evidence-producing parts of the proposed CI workflow in
Ubuntu before any workflow receives repository write permissions. The Compose service
is limited to 4 CPUs and 16 GB of memory, matching the documented resource envelope
of a standard public `ubuntu-latest` GitHub-hosted runner. Disk usage is measured and
reported against a 14 GB budget because Docker Compose does not enforce that limit.

Run the fast rehearsal:

```powershell
docker compose `
  -f experiments/scaling-my-posts/docker/docker-compose.ci.yml `
  run --rm translation-ci
```

This runs the focused experiment tests, renders the post, and writes ignored evidence
to:

```text
experiments/scaling-my-posts/artifacts/local-ci/execution-evidence.json
```

Run the same checks plus two real Tower+ translations:

```powershell
docker compose `
  -f experiments/scaling-my-posts/docker/docker-compose.ci.yml `
  run --rm translation-ci --with-model-smoke
```

The Hugging Face cache is stored in a named Docker volume so the multi-gigabyte model
does not need to be downloaded for every rehearsal.

After the runner or article changes, refresh the sanitized evidence committed with the
post:

```powershell
docker compose `
  -f experiments/scaling-my-posts/docker/docker-compose.ci.yml `
  run --rm translation-ci `
  --with-model-smoke `
  --public-summary-output posts/data/translation-local-ci-evidence.json
```

The ignored raw evidence retains command tails and local diagnostic details. The
public summary contains the notebook hash, resource envelope, measurements, stage
statuses, model revision, smoke translations, and explicit limitations without local
paths or captured command output.

A passing local run demonstrates that the tested translation, validation, and render
commands work in a clean, resource-constrained Linux container. It does **not** verify
GitHub token permissions, branch creation, pull-request creation, GitHub Pages
previews, or preview cleanup. Those boundaries require a later GitHub Actions smoke
test with deliberately limited permissions.
