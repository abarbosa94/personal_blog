# Review Report Template

Follow this structure. Keep prose proportional to the number and severity of findings.

## Overall assessment

Briefly assess publication readiness and the most consequential strengths and
weaknesses. List the canonical draft and associated files reviewed.

## Progress since the previous iteration

For the first review, write:

`Initial review — no previous iteration available.`

Otherwise group prior findings into:

- Resolved
- Partially resolved
- Regressed
- Remaining

## Findings

Group findings by severity: Blocking, Important, Minor, and Optional. Write `None`
when a group is empty.

Use this shape for each finding:

```text
ID:
Status: Confirmed issue | Likely issue | Needs verification | Optional improvement
Location:
Original excerpt:
Problem:
Why it matters:
Suggested direction:
Verification check:
```

## Questions for the author

Ask only questions that materially affect technical correctness, audience assumptions,
or the narrative and cannot be inferred safely from available evidence. Write `None`
when there are no such questions.

## Recommended next revision

Give three to five prioritized actions, starting with blocking and important findings.

## Guardrail report

```text
| Guardrail | Result | Reason |
|---|---|---|
| Technical correctness | PASS/FAIL/NEEDS VERIFICATION | ... |
| Code correctness | PASS/FAIL/NEEDS VERIFICATION | ... |
| Claims and evidence | PASS/FAIL/NEEDS VERIFICATION | ... |
| Grammar and fluency | PASS/FAIL/NEEDS VERIFICATION | ... |
| Storytelling and structure | PASS/FAIL/NEEDS VERIFICATION | ... |
| Audience clarity | PASS/FAIL/NEEDS VERIFICATION | ... |
| Terminology consistency | PASS/FAIL/NEEDS VERIFICATION | ... |
```

## Publication decision

Write exactly one:

- `NOT READY — BLOCKING ISSUES`
- `NOT READY — IMPORTANT REVISIONS`
- `READY AFTER VERIFICATION`
- `SAFE TO PUBLISH`

Explain the decision in two or three sentences.

## Review checkpoint

Render the portable checkpoint from `iteration-protocol.md`.
