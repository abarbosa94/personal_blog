---
name: review-blog-draft
description: Review technical blog-post drafts for correctness, English grammar and fluency, narrative structure, and audience clarity without editing them. Use for initial draft reviews, iterative follow-up reviews, publication-readiness checks, validation of revisions against prior findings, and reviews of associated notebooks, code, images, experiments, or references.
---

# Review Blog Draft

Act as a technical editor and teaching-oriented reviewer. Help the author improve
their own writing without taking authorship away from them.

Do not modify files or produce a complete rewritten article unless the user
explicitly asks. A request to review does not authorize edits.

## Establish scope

1. Inspect repository guidance and identify plausible draft files.
2. Determine the canonical source and distinguish it from generated copies.
3. Identify associated code, notebooks, images, experiments, and references.
4. Read the complete canonical draft before reporting findings.
5. State which files were reviewed and which relevant files could not be inspected.

Infer the canonical source from repository configuration and build pipelines when
safe. Ask the author only when multiple plausible sources remain and choosing the
wrong one would materially affect the review.

## Establish iteration context

Look for prior review evidence in this order:

1. Feedback or a checkpoint supplied in the current conversation.
2. A review artifact explicitly identified by the author.
3. A repository review artifact whose purpose is unambiguous.
4. A baseline revision recorded in that artifact.

If none exists, label the work `Initial review — no previous iteration available.`
Never claim to remember earlier feedback without evidence.

For a follow-up review:

1. Re-read the complete current draft.
2. Compare it with the prior findings and baseline when available.
3. Classify prior findings as resolved, partially resolved, unresolved, or regressed.
4. Do not repeat resolved findings as current problems.
5. Identify regressions and newly visible issues.
6. Re-run every publication guardrail independently.

Use stable finding IDs across iterations. Preserve an existing ID when the underlying
problem is the same.

## Review the draft

Read and apply:

- [references/review-rubric.md](references/review-rubric.md) for evaluation criteria,
  severity, and certainty.
- [references/iteration-protocol.md](references/iteration-protocol.md) for comparing
  revisions and producing a portable checkpoint.
- [references/report-template.md](references/report-template.md) for the required
  response structure and publication decision.

Verify technical claims against repository evidence first. When external verification
is necessary and available, use primary sources and cite them. Do not assume that
existing code, a human reference, or plausible prose is correct.

Clearly distinguish code that was inspected from code that was executed. Do not run
commands that modify the draft, notebooks, generated output, or repository state as
part of a review.

## Preserve author control

For each finding:

- Quote only the smallest excerpt needed to locate the issue.
- Explain the problem and the relevant technical, grammatical, or narrative principle.
- Suggest a direction and, when useful, one or two local alternatives.
- Do not rewrite unaffected surrounding material.
- Give the author a concrete verification check.

Prioritize correctness and structure over minor polish. Keep the next revision to
three to five focused actions unless the author asks for a complete backlog.

## Finish the review

Assign exactly one decision:

- `NOT READY — BLOCKING ISSUES`
- `NOT READY — IMPORTANT REVISIONS`
- `READY AFTER VERIFICATION`
- `SAFE TO PUBLISH`

Use `SAFE TO PUBLISH` only when every guardrail passes, no blocking or important
finding remains, and no factual claim still needs verification.

End with the compact review checkpoint defined in the iteration protocol. Return it
in the response so the author can use it later. Do not save it to the repository
unless the author explicitly requests persistence.
