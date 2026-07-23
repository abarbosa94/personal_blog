# Iteration Protocol

Use this protocol to make reviews portable across conversations and points in time.

## Compare revisions

When prior evidence is available:

1. Match prior findings by stable ID.
2. Mark each as `resolved`, `partially resolved`, `unresolved`, or `regressed`.
3. Explain only changes that affect the disposition.
4. Assign new IDs only to genuinely new problems.
5. Re-read the whole draft so a local fix cannot hide a global regression.

Do not infer that an issue is resolved merely because its quoted sentence changed.
Check the underlying verification condition.

## Finding IDs

Use a category prefix and a sequence number:

- `TECH-001`: technical correctness
- `CODE-001`: executable example or implementation
- `EVID-001`: claim or evidence
- `LANG-001`: grammar or fluency
- `FLOW-001`: narrative or structure
- `CLAR-001`: audience clarity
- `TERM-001`: terminology consistency

Reuse IDs across iterations. Never renumber open findings for presentation.

## Baseline identity

Record the canonical file and, when available without modifying the repository:

- Git commit plus dirty-worktree status, or
- A content hash, or
- A clearly identified dated version supplied by the author.

Do not imply that a commit identifies uncommitted draft content.

## Portable checkpoint

End every review with a compact YAML checkpoint in a fenced block:

```yaml
review_checkpoint:
  canonical_file: "path/to/draft"
  baseline: "commit, content hash, or unknown"
  reviewed_at: "YYYY-MM-DD"
  audience: "stated, inferred, or unknown"
  intended_takeaway: "one sentence"
  publication_decision: "one allowed decision"
  open_findings:
    - id: "TECH-001"
      severity: "Important"
      status: "Confirmed issue"
      summary: "Short description of the underlying problem"
      verification: "Observable condition that resolves it"
  resolved_findings:
    - id: "FLOW-001"
      summary: "Short description"
  claims_needing_verification:
    - id: "EVID-001"
      claim: "Claim requiring evidence"
      required_evidence: "Evidence that would settle it"
```

Omit empty list entries or use `[]`. Keep the checkpoint concise; it is state for the
next review, not a duplicate report.

Do not write checkpoints or review reports to disk unless the author explicitly asks.
If persistence is requested, confirm the destination is intended to be public before
including private notes, unpublished claims, or sensitive material.
