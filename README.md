[//]: # (This template replaces README.md when someone creates a new repo with the fastpages template.)

![](https://github.com/abarbosa94/personal_blog/workflows/Quarto%20Publish/badge.svg) 

https://abarbosa94.github.io/personal_blog/

# My Blog


_powered by [Quarto](https://quarto.org/)_

## Authoring posts

Write new posts as Jupyter notebooks (`.ipynb`) in `posts/`. Put Quarto YAML
front matter in the first raw cell, keep the narrative in Markdown cells, and
save any computational results in the notebook before publishing. The shared
post configuration freezes those outputs, and the Quarto workflow converts the
notebook into the website.

## Adding a bilingual post

Create one source document per language in `posts/`. Pair the documents with the
`translation` field and set the document language with `lang`:

```yaml
# English version
lang: en
translation: my-post-pt-br.qmd
```

```yaml
# Brazilian Portuguese version
lang: pt-BR
translation: my-post.qmd
language-version: translation
```

The language switcher is added automatically to both pages. Only the version
without `language-version: translation` appears in the home-page listing. Move
that field to the English document when Portuguese should be the listed version.

To generate a reviewable Portuguese notebook locally with the pinned Tower+ 2B
model, run:

```powershell
docker compose -f experiments/scaling-my-posts/docker/docker-compose.ci.yml run --rm `
  --entrypoint python translation-ci `
  experiments/scaling-my-posts/src/translate_notebook.py `
  posts/2026-07-23-Scaling-MyPost-WithAIAgents.ipynb `
  posts/2026-07-23-Scaling-MyPost-WithAIAgents-pt-BR.ipynb
```

The generated notebook remains a draft and includes a machine-translation
warning. Review its language and technical meaning before publishing it.

To test both pages and their language switcher locally, include drafts in the
preview:

```powershell
quarto preview -M draft:false
```

Open the English or Portuguese post and use the language link at the top of the
article. Quarto converts each notebook to HTML as part of the preview.

The intended automation runs only after the source-content PR has been reviewed
and merged. The resulting push to `main` triggers translation, validation, and a
temporary rendered artifact, then opens a separate translation PR. Human review
and merging of that translation PR remain distinct from the original source
review.

The workflow can also be started manually from **Actions → Create translation
PR → Run workflow**. Supply the English notebook path under `posts/`; this is
useful for a source post that reached `main` before the workflow was installed.
The generated pull request is a draft and includes the publication checklist.

## Reproducing the translation evaluation

The original passage-level experiment is retained as an overlap-metric baseline.
The sentence-level study in
`experiments/scaling-my-posts/src/translation_eval.py` uses this staged flow:

```text
historical notebooks -> LaBSE alignment proposals -> human-reviewed segments
                     -> model predictions -> MQM/pairwise judge + xCOMET
                     -> aggregate tables -> executed Jupyter post
```

Install the normal benchmark environment and propose ordered sentence matches:

```powershell
uv venv --python 3.12 .venv-benchmark
uv pip install --python .venv-benchmark\Scripts\python.exe `
  -r experiments/scaling-my-posts/requirements/benchmark.txt
.venv-benchmark\Scripts\python.exe experiments/scaling-my-posts/src/translation_eval.py align
```

Review the alignment proposals in the local browser interface rather than editing
the CSV directly:

```powershell
.venv-benchmark\Scripts\python.exe experiments/scaling-my-posts/src/translation_review_server.py
```

The command opens `http://127.0.0.1:8765/` and saves every decision, note, and
text override atomically to
`posts/data/translation-eval-alignment-review.csv`. The interface shows one
English/Portuguese proposal at a time, supports high-priority and status filters,
and includes keyboard shortcuts, jump-by-ID, previous/next navigation, and Undo.
Press `Ctrl+C` in the terminal to stop it.

Every row must eventually use one of these terminal statuses before the dataset
can be frozen:

- `accept`: include the pair in the benchmark;
- `localized`: related content that is not a direct translation;
- `exclude`: navigation, unmatched, or otherwise unsuitable material.

`defer` records uncertainty without forcing a guess and remains unresolved until
you revisit it. Exclusions and manual text overrides require a reviewer note.

Use `reviewed_english` or `reviewed_portuguese` only for an explicit, documented
correction or to isolate the translated portion of a many-to-many proposal. Keep
the historical text in the original columns and explain every edit in
`review_note`. The freeze command deliberately fails while any row still says
`needs_review`.

```powershell
.venv-benchmark\Scripts\python.exe experiments/scaling-my-posts/src/translation_eval.py freeze
.venv-benchmark\Scripts\python.exe experiments/scaling-my-posts/src/translation_eval.py predict --threads 16
```

Inference shows a `tqdm` progress bar for every model and direction. Completed
model/direction results are written atomically to the prediction and runtime CSVs;
rerunning the same command validates and reuses those checkpoints. Pass
`--restart` only when you intentionally want to discard the checkpoints and
recompute the requested models.

Inspect the paid job count without making an API request, then run the resumable
judge locally. Kimi K3 is the default provider. `MOONSHOT_API_KEY` is read only
from the process environment; `.env` files are ignored to reduce the chance of
committing a credential.

```powershell
.venv-benchmark\Scripts\python.exe experiments/scaling-my-posts/src/translation_eval.py judge --dry-run
$env:MOONSHOT_API_KEY = "..."
# First make four paid calls and inspect the JSONL artifact.
.venv-benchmark\Scripts\python.exe experiments/scaling-my-posts/src/translation_eval.py judge --limit 4
# Resume the remaining requests after the pilot looks correct.
.venv-benchmark\Scripts\python.exe experiments/scaling-my-posts/src/translation_eval.py judge
```

The default judge is `kimi-k3` at `max` reasoning effort with an 8,192-token
completion cap. It emits strict structured MQM errors and repeats every pairwise
comparison with the candidates reversed. The paid stage defaults to Marian and
Tower+ (288 requests); NLLB remains in the timing/overlap appendix but is screened
out of the judge stage after its Portuguese-to-English predictions failed the
repetition criterion documented in
`experiments/scaling-my-posts/README.md`. The reproducible detector is
`experiments/scaling-my-posts/src/screening.py`.

Each completed request is appended to JSONL with its provider, requested and
returned model names, inference settings, hashes, usage, final structured content,
and timestamp. K3's `reasoning_content` is deliberately omitted. An interrupted
run resumes without rebilling finished work. Because `kimi-k3` is currently an
unversioned alias, the returned model name and timestamp are important provenance.

Judge providers implement the contract in
`experiments/scaling-my-posts/src/judge_interface.py` and are registered by
`experiments/scaling-my-posts/src/judge_factory.py`. The shared MQM and pairwise
rubrics live in `experiments/scaling-my-posts/src/prompts.py`. Switching back to
the OpenAI adapter does not change the evaluation pipeline or rubric:

```powershell
$env:OPENAI_API_KEY = "..."
.venv-benchmark\Scripts\python.exe experiments/scaling-my-posts/src/translation_eval.py judge `
  --judge-provider openai
.venv-benchmark\Scripts\python.exe experiments/scaling-my-posts/src/translation_eval.py aggregate `
  --judge-provider openai
```

The default artifact name includes the provider. Use a different explicit output
path when comparing two configurations of the same provider; the resume guard
rejects attempts to mix them in one JSONL artifact. New providers implement the
same adapter interface and register with `JudgeAdapterFactory`.

xCOMET-XL is optional, gated, non-commercial research software with a large
checkpoint. Accept its Hugging Face terms first, create its separate environment,
and run it on CPU:

```powershell
uv venv --python 3.12 .venv-xcomet
uv pip install --python .venv-xcomet\Scripts\python.exe `
  -r experiments/scaling-my-posts/requirements/xcomet.txt
$env:HF_TOKEN = "..."
.venv-xcomet\Scripts\python.exe experiments/scaling-my-posts/src/translation_eval.py xcomet
```

Finally, aggregate the complete run and create the blinded 18-item bilingual
review sheet. Keep the generated key closed until the review sheet is complete.

```powershell
.venv-benchmark\Scripts\python.exe experiments/scaling-my-posts/src/translation_eval.py aggregate
.venv-benchmark\Scripts\python.exe experiments/scaling-my-posts/src/translation_eval.py human-sample
# Review the blinded sample in a keyboard-friendly local UI. Decisions auto-save.
.venv-benchmark\Scripts\python.exe experiments/scaling-my-posts/src/translation_review_server.py --mode human
# Keep translation-eval-human-key.csv closed until the review is complete.
.venv-benchmark\Scripts\python.exe experiments/scaling-my-posts/src/translation_eval.py human-agreement
```

The human reviewer shows one source/reference/candidate pair at a time and supports
previous/next navigation, defer and undo, confidence labels, structured failure
tags, notes, golden-set marking, filters, and keyboard shortcuts. Model identities
and the Kimi preference remain blinded until `human-agreement` joins the review to
the separate key.
It opens on `http://127.0.0.1:8766/`; the earlier alignment-review mode uses port
`8765`, so both workflows can remain open without being confused.

Automated tests never call a paid judge provider or download xCOMET:

```powershell
.venv-benchmark\Scripts\python.exe -m pytest -q experiments/scaling-my-posts/tests
```

The review server has no runtime dependencies. To run its optional Chrome/Edge
workflow test, install Playwright (it reuses the browser already installed on the
machine):

```powershell
uv pip install --python .venv-benchmark\Scripts\python.exe `
  -r experiments/scaling-my-posts/requirements/review-ui-test.txt
.venv-benchmark\Scripts\python.exe -m pytest -q `
  experiments\scaling-my-posts\tests\test_translation_review_server.py
```

