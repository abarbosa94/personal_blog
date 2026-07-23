# Review Rubric

Apply all four perspectives on every review, even when the author asks to emphasize
only one or two.

## Technical correctness

Check:

- Factual claims, terminology, qualifications, and internal consistency.
- Code examples, APIs, algorithms, architectures, complexity, and performance claims.
- Whether simplifications could mislead the intended audience.
- Whether claims require experiments, evidence, references, or external verification.
- Contradictions among prose, code, figures, results, and conclusions.

Use one certainty status:

- `Confirmed issue`: supported directly by the draft, repository, execution evidence,
  or an authoritative source.
- `Likely issue`: evidence strongly suggests a problem but does not establish it.
- `Needs verification`: correctness depends on unavailable or unchecked evidence.
- `Optional improvement`: a defensible enhancement rather than an error.

Do not convert uncertainty into a factual accusation. State what evidence would settle
the question.

## Grammar and fluency

Identify grammar, spelling, tense, reference, repetition, sentence-length,
capitalization, punctuation, terminology, and idiomatic-English problems.

For each material issue, explain the relevant principle and offer local suggestions.
Do not silently correct the source or rewrite the whole passage.

## Storytelling and logical flow

Evaluate:

- Whether the introduction establishes motivation, audience, and a clear promise.
- Whether the problem is understood before the proposed solution appears.
- Whether concepts are introduced before use.
- Whether sections, examples, and transitions advance the central argument.
- Whether detail is proportionate to the intended audience.
- Whether detours, repetition, or abrupt jumps interrupt the narrative.
- Whether the conclusion fulfills the introduction's promise.

Identify the exact point where a reader may become confused, lose interest, or infer
the wrong takeaway.

## Audience clarity

Check:

- Whether the main takeaway is identifiable.
- Whether acronyms and specialized terms are introduced.
- Whether abstractions have sufficient concrete examples.
- Whether examples reconnect to the main argument.
- Whether sentences generally communicate one principal idea at a time.
- Whether assumed knowledge matches the stated or inferred audience.

## Severity

- `Blocking`: A factual failure, broken central argument, unsafe instruction, unusable
  code, or other defect that must be fixed before publication.
- `Important`: Materially affects correctness, clarity, credibility, or narrative
  comprehension.
- `Minor`: Improves polish but does not prevent publication.
- `Optional`: A subjective enhancement, alternative, or expansion rather than a defect.

Severity and certainty are independent. For example, an unsupported central claim may
be `Important` and `Needs verification`.

## Publication guardrails

Mark each guardrail `PASS`, `FAIL`, or `NEEDS VERIFICATION`:

1. Technical correctness
2. Code correctness
3. Claims and evidence
4. Grammar and fluency
5. Storytelling and structure
6. Audience clarity
7. Terminology consistency

Use `PASS` only for the current complete draft, not merely because it improved.

## Review discipline

- Do not manufacture findings to populate every severity group.
- Write `None` for an empty group.
- Do not let minor grammar findings obscure structural or technical defects.
- Mention strengths only when they help the author preserve an effective choice.
- Prefer a small number of precise findings over repeated symptoms of one root cause.
