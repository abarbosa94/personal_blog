"""Deterministic abstract-context classifier for the frozen RAI dimensions."""

from __future__ import annotations

import re


CONTEXT_DIMENSIONS = {
    "privacy_data_governance": (
        r"\bprivacy (?:risk|guarantee|violation|vulnerability|policy|policies|management|protection)\b",
        r"\bprivacy (?:of|amplification)\b",
        r"\bprivate data\b",
        r"\bdifferential(?:ly)? private\b",
        r"\bdata governance\b",
        r"\bmachine unlearning\b",
        r"\bmembership inference\b",
        r"\bdata deletion\b",
        r"\bforget data\b",
        r"\bconsent[- ]driven\b",
        r"\bsensitive (?:data|information|attribute|group membership)\b",
        r"\bhomomorphic encryption\b",
        r"\bcontextual integrity\b",
        r"\balgorithmic (?:travel )?surveillance\b",
    ),
    "transparency_explainability": (
        r"\bexplainab",
        r"\bexplanation(?:s)?\b",
        r"\b(?:model|representation|policy|prediction|decision|feature) interpretability\b",
        r"\binterpretable (?:model|representation|policy|prediction|decision|feature|concept)",
        r"\btransparency\b",
        r"\btransparent\b",
        r"\btransparent (?:ai|model|algorithm|decision)",
        r"\bfeature attribution\b",
        r"\battribution (?:framework|method|score)",
        r"\bcounterfactual explanation",
        r"\bactionable (?:algorithmic )?recourse\b",
        r"\balgorithmic recourse\b",
        r"\bhuman[- ]understandable (?:prediction|decision|model)",
        r"\b(?:expos(?:e|es|ed|ing)|disclos(?:e|es|ed|ing)|display(?:s|ed|ing)?) "
        r"(?:predictive )?uncertainty\b.{0,120}\b(?:ai|algorithm|model|decision aid)",
        r"\b(?:ai|algorithm|model|decision aid).{0,120}\b"
        r"(?:expos(?:e|es|ed|ing)|disclos(?:e|es|ed|ing)|display(?:s|ed|ing)?) "
        r"(?:predictive )?uncertainty\b",
    ),
    "security_safety": (
        r"\badversarial (?:attack|example|perturbation)",
        r"\bbackdoor(?: attack)?\b",
        r"\bdata poisoning\b",
        r"\bprompt injection\b",
        r"\bjailbreak",
        r"\bai safety\b",
        r"\bmodel safety\b",
        r"\bsafety alignment\b",
        r"\bsafety guardrail",
        r"\b(?:policy|agent|autonomous driving).{0,80}\bsafety constraint",
        r"\bsafety constraint.{0,80}\b(?:policy|agent|autonomous driving)",
        r"\bharmless",
        r"\bharmful (?:content|instruction|output|fine[- ]tuning)",
        r"\btoxic(?:ity| output)",
        r"\bred team",
        r"\bmalicious (?:instruction|prompt|sample|fine[- ]tuning|user|attack)",
        r"\breward hacking\b",
        r"\badverse[- ]event reporting\b",
        r"\b(?:ai|agent) incidents?\b",
        r"\bincident analysis\b.{0,120}\b(?:ai|agent)",
        r"\b(?:private information )?exfiltration\b",
        r"\bunauthorized actions?\b",
        r"\bfrontier[- ]ai risk",
        r"\bmodel misuse\b",
        r"\bnon[- ]consensual deepfake",
        r"\bred[- ]team(?:ing)?\b",
        r"\b(?:watermark|intellectual property protection)\b.{0,300}\bmisuse\b",
        r"\b(?:ai|machine learning|algorithmic systems?|llms?).{0,180}\b"
        r"(?:computer security|safety[- ]engineering|formal verification)\b",
        r"\b(?:computer security|safety[- ]engineering|formal verification)\b"
        r".{0,180}\b(?:ai|machine learning|algorithmic systems?|llms?)\b",
        r"\b(?:documented|severe|interaction) harms?\b.{0,160}\b"
        r"(?:ai|chatbots?|companions?|llms?)\b",
        r"\b(?:ai|chatbots?|companions?|llms?).{0,160}\b"
        r"(?:documented|severe|interaction) harms?\b",
        r"\balgorithmic[- ]driven risks?\b",
    ),
    "fairness": (
        r"\bfairness\b",
        r"\balgorithmic fairness\b",
        r"\bdiscrimination\b",
        r"\bdiscriminatory\b",
        r"\bequity\b",
        r"\bdemographic parity\b",
        r"\bequalized odds\b",
        r"\bsocial bias\b",
        r"\bgender(?:ed)? bias\b",
        r"\bgender[- ]biased\b",
        r"\bracial bias\b",
        r"\bdemographic bias\b",
        r"\bnon[- ]binary gender bias\b",
        r"\bracial disparit",
        r"\bdemographic disparit",
        r"\bperformance (?:disparit|differential)",
        r"\bbias audits?\b",
        r"\bstereotype(?:s|d| assessment)?\b",
        r"\brepresentational harms?\b",
        r"\bgroup[- ]fair",
        r"\bneurodivergent marginali[sz](?:ed|ation)\b",
        r"\bharmful or unjust uses? of algorithmic systems?\b",
        r"\blinguistic colonialism\b",
        r"\bepistemic violence\b",
        r"\bcultural erasure\b",
        r"\bdialect bias\b",
        r"\bminoritized (?:english )?dialects?\b",
        r"\bquality[- ]of[- ]service harms?\b",
    ),
}

TITLE_DIMENSIONS = {
    "privacy_data_governance": (r"\bprivacy\b",),
    "transparency_explainability": (
        r"\binterpretability\b",
        r"\binterpretable\b",
        r"\bfixed predictions?\b",
    ),
    "security_safety": (),
    "fairness": (),
}

DIMENSION_EXCLUSIONS = {
    "privacy_data_governance": (),
    "transparency_explainability": (
        r"\b(?:mathematical|theoretical) explanation\b",
        r"\btheoretically[- ]grounded explanation\b",
        r"\b(?:compelling|satisfactory) explanation (?:for|of|why)\b",
        r"\bexplanation of (?:the |a )?(?:behavior|phenomenon|limitations?|normalization)",
        r"\btextual point explanation\b",
        r"\btransparent framework\b",
        r"\bbetter explainability\b",
        r"\bfair and transparent (?:democratic |decision[- ]making |governance )?(?:procedure|process|value)",
        r"\bcode explanation\b",
        r"\bproposed explanations?\b.{0,100}\b(?:generaliz|inductive bias)",
        r"\binterpretable model of (?:the )?(?:conserved|neural|biological|physical|scientific|temporal|latent)\b",
    ),
    "security_safety": (
        r"\badversarial (?:augmentation|example training)\b.{0,180}"
        r"\b(?:domain generalization|generalization|unseen environments?)\b",
    ),
    "fairness": (
        r"\b(?:instance|feature|frame|spatial) discrimination\b",
        r"\bimplicit bias of\b",
        r"\bbiased toward\b",
        r"\b(?:motivat|prior work|adjacent|example|properties).{0,140}\bfairness\b",
        r"\bfairness\b.{0,140}\b(?:motivat|prior work|adjacent|example|properties)\b",
        r"\bfairness datasets?\b",
    ),
}


def abstract_text(value: str) -> str:
    """Remove PDF title/author matter and later sections when present."""
    text = re.sub(r"\s+", " ", value).strip()
    marker = re.search(r"\babstract\b", text[:1200], flags=re.IGNORECASE)
    if marker:
        text = text[marker.end():]
        end = re.search(
            r"\b(?:1\.?\s+introduction|introduction\s+1)\b",
            text,
            flags=re.IGNORECASE,
        )
        if end:
            text = text[:end.start()]
    return text


def evidence_units(title: str, abstract: str) -> tuple[str, ...]:
    """Return title and sentence-like units for local inclusion/exclusion rules."""
    text = abstract_text(abstract).casefold()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    # Preserve an empty title as element zero because the classifier uses the
    # element index to distinguish title-only patterns from abstract evidence.
    return (
        title.casefold().strip(),
        *tuple(unit.strip() for unit in sentences if unit.strip()),
    )


def classify_context(title: str, abstract: str) -> tuple[str, ...]:
    """Return frozen RAI dimensions supported by title/abstract context."""
    units = tuple(
        re.sub(
            r"(?:fundamentally )?different problems?, such as measuring bias or fairness",
            "",
            unit,
        )
        for unit in evidence_units(title, abstract)
    )
    title_text = units[0] if units else ""
    dimensions = []
    for dimension, patterns in CONTEXT_DIMENSIONS.items():
        exclusions = DIMENSION_EXCLUSIONS[dimension]
        title_patterns = TITLE_DIMENSIONS[dimension]
        for index, unit in enumerate(units):
            unit_patterns = patterns + (title_patterns if index == 0 else ())
            if not any(re.search(pattern, unit) for pattern in unit_patterns):
                continue
            # An exclusion suppresses evidence only in its own sentence. This
            # prevents a background sentence such as "fairness is an example"
            # from erasing a later, substantive fairness contribution. The
            # title is retained as local context so a title that explicitly
            # frames adversarial examples as domain-generalization
            # augmentation can disambiguate the evidence sentence.
            exclusion_context = f"{title_text}. {unit}"
            if any(re.search(exclusion, exclusion_context) for exclusion in exclusions):
                continue
            dimensions.append(dimension)
            break
    return tuple(dimensions)
