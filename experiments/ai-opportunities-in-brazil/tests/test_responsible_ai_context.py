from conference_pipeline.responsible_ai_context import classify_context


def test_context_classifier_recognizes_frozen_dimensions():
    assert classify_context("", "We provide actionable algorithmic recourse.") == (
        "transparency_explainability",
    )
    assert classify_context("", "We study demographic bias and discriminatory pricing.") == (
        "fairness",
    )
    assert classify_context("", "Membership inference threatens privacy.") == (
        "privacy_data_governance",
    )
    assert classify_context("", "A jailbreak bypasses model safety guardrails.") == (
        "security_safety",
    )


def test_context_classifier_ignores_known_ambiguous_uses():
    assert classify_context("", "We align geometric representations robustly.") == ()
    assert classify_context("", "The optimizer has a simplicity bias.") == ()
    assert classify_context("", "Instance discrimination is our contrastive objective.") == ()
    assert classify_context("", "We provide a theoretical explanation of normalization.") == ()


def test_context_classifier_recognizes_audit_missed_harms():
    assert classify_context("", "We red-team the model to elicit harmful images.") == (
        "security_safety",
    )
    assert classify_context("", "We study non-consensual deepfake generators.") == (
        "security_safety",
    )
    assert classify_context("", "Bias audits reveal demographic performance disparities.") == (
        "fairness",
    )
    assert classify_context("", "Contextual integrity is applied to algorithmic surveillance.") == (
        "privacy_data_governance",
    )


def test_exclusions_apply_to_the_evidence_sentence_not_the_whole_abstract():
    abstract = (
        "Prior work names fairness as one example. "
        "We measure fairness for protected groups and quantify demographic disparities."
    )
    assert classify_context("", abstract) == ("fairness",)


def test_context_classifier_recognizes_sociotechnical_failure_families():
    assert classify_context(
        "",
        "AI companions have caused documented severe interaction harms.",
    ) == ("security_safety",)
    assert classify_context(
        "",
        "Machine translation can perpetuate linguistic colonialism and cultural erasure.",
    ) == ("fairness",)
    assert classify_context(
        "",
        "The AI decision aid exposes predictive uncertainty to its users.",
    ) == ("transparency_explainability",)


def test_context_classifier_excludes_technical_confirmation_failures():
    assert classify_context(
        "",
        "Adversarial example training improves domain generalization in unseen environments.",
    ) == ()
    assert classify_context(
        "",
        "We use five fairness datasets to evaluate predictive multiplicity.",
    ) == ()
    assert classify_context(
        "",
        "The method enables an interpretable model of conserved neural dynamics.",
    ) == ()
