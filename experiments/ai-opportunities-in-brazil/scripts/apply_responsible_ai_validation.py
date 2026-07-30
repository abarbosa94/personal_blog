"""Apply the manual Responsible-AI review decisions to the frozen sample."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


# Values are: manual_label, manual_dimensions, concise evidence note.
# The decisions were made from official abstracts/pages or official PDF text. ICLR
# context came from the official proceedings pages when OpenReview returned 403.
DECISIONS = {
    ("aaai", "10.1609/aaai.v39i21.34429"): ("negative", "", "Official abstract: clustering is the contribution; privacy is only a federated-learning motivation, with no privacy analysis or guarantee. Scope-borderline."),
    ("aaai", "10.1609/aaai.v39i11.33299"): ("negative", "", "Official abstract: dense-retrieval representation, storage, and search efficiency; no frozen RAI dimension."),
    ("aaai", "10.1609/aaai.v39i24.34762"): ("positive", "security_safety", "Official abstract: restores LLM safety after malicious fine-tuning by identifying safety-critical neurons."),
    ("aaai", "10.1609/aaai.v39i26.34930"): ("negative", "", "Official abstract: verifies robustness to benign blur/sharpen convolutional perturbations, not security attacks or AI safety."),
    ("aaai", "10.1609/aaai.v39i2.32212"): ("negative", "", "Official abstract: object detection and cross-modal alignment; 'bias' denotes downstream alignment error, not social bias."),
    ("aaai", "10.1609/aaai.v39i1.32106"): ("negative", "", "Official abstract: predicts scholarly impact from titles and abstracts; no frozen RAI dimension."),
    ("aaai", "10.1609/aaai.v39i21.34440"): ("positive", "privacy_data_governance", "Official abstract: develops differentially private optimization algorithms and proves privacy-risk rates."),
    ("aaai", "10.1609/aaai.v39i7.32771"): ("positive", "security_safety", "Official abstract: architecture explicitly filters adversarial noise and mitigates adversarial attacks."),
    ("aaai", "10.1609/aaai.v39i22.34518"): ("negative", "", "Official abstract: robustness to ordinary ASR transcription errors in medical summarization, not security or AI safety."),
    ("aaai", "10.1609/aaai.v39i28.35267"): ("negative", "", "Official abstract: robust medication-demand forecasting under changing conditions; no frozen RAI dimension."),
    ("aaai", "10.1609/aaai.v39i22.34521"): ("positive", "security_safety", "Official abstract: adjusts LLM safety alignment and refusal behavior while preserving defenses against malicious instructions."),
    ("aaai", "10.1609/aaai.v39i5.32572"): ("positive", "fairness", "Official abstract: measures and mitigates racial disparities in deep-forgery detection."),
    ("aaai", "10.1609/aaai.v39i20.35394"): ("positive", "transparency_explainability", "Official abstract: proposes inherently interpretable RL policies to make automated decisions transparent."),
    ("aies", "36751"): ("negative", "", "Official abstract: human-centered AI-mediated communication design; no privacy, explainability, security/safety, or fairness analysis."),
    ("aies", "36713"): ("negative", "", "Official abstract: philosophical analysis of relational selfhood and predictive AI; outside the four frozen dimensions."),
    ("aies", "36580"): ("positive", "fairness", "Official abstract: reviews harms, bias, fairness, and equity for African American English speakers in ASR."),
    ("aies", "36601"): ("positive", "transparency_explainability", "Official abstract: evaluates AI labels intended to communicate model behavior and improve transparency."),
    ("aies", "36695"): ("negative", "", "Official abstract: critiques LLM-based empirical jurisprudence and judicial discretion; it does not evaluate a frozen technical dimension. Scope-borderline."),
    ("aies", "36555"): ("negative", "", "Official abstract: proposes taxation of attention capture; no frozen RAI dimension."),
    ("aies", "36597"): ("positive", "transparency_explainability|fairness", "Official abstract: uses explainable AI to diagnose whether automated-decision bias is technical or societal."),
    ("aies", "36626"): ("negative", "", "Official abstract: assesses LLMs as research-ethics support tools; general ethics is outside the frozen four dimensions."),
    ("aies", "36607"): ("positive", "security_safety", "Official abstract: designs post-deployment adverse-event reporting for frontier-AI risks, vulnerabilities, and misuse."),
    ("aies", "36612"): ("positive", "transparency_explainability|fairness", "Official abstract: proposes explanation difference as a procedural-fairness measure alongside distributional fairness."),
    ("aies", "36706"): ("positive", "privacy_data_governance|transparency_explainability|security_safety|fairness", "Official abstract: maps AI-ethics evaluations to hazards and harms, explicitly covering privacy, transparency, safety/trust, and fairness."),
    ("aies", "36723"): ("positive", "fairness", "Official abstract: human-subject studies measure how gender-biased text-to-image outputs affect implicit bias."),
    ("facct", "10.1145/3715275.3732185"): ("positive", "privacy_data_governance", "Official CSV abstract: demonstrates sensitive group-membership inference from nominally anonymized street imagery."),
    ("facct", "10.1145/3715275.3732132"): ("negative", "", "Official CSV abstract: studies designers' gendered mental representations of generic technology users, not AI-system fairness. Scope-borderline."),
    ("facct", "10.1145/3715275.3732042"): ("positive", "fairness", "Official CSV abstract: defines a group-fairness metric and mitigation algorithms for rated preference aggregation."),
    ("facct", "10.1145/3715275.3732055"): ("negative", "", "Official CSV abstract: inclusive recruitment methods for non-WEIRD HCI research; no frozen AI dimension."),
    ("facct", "10.1145/3715275.3732008"): ("positive", "transparency_explainability", "Official CSV abstract: studies validity of actionable algorithmic recourse over time; recourse is within the explanation dimension."),
    ("facct", "10.1145/3715275.3732076"): ("positive", "fairness", "Official CSV abstract: designs a fairness-centered exchange to mitigate discriminatory personalized pricing."),
    ("facct", "10.1145/3715275.3732117"): ("positive", "transparency_explainability|fairness", "Official CSV abstract: fairness-guided pruning jointly reduces discrimination and simplifies interpretable decision trees."),
    ("facct", "10.1145/3715275.3732169"): ("positive", "fairness", "Official CSV abstract: develops and evaluates a taxonomy of gender bias and harms in AI-generated product descriptions."),
    ("facct", "10.1145/3715275.3732013"): ("positive", "fairness", "Official CSV abstract: dataset and evaluation explicitly measure non-binary gender bias in machine translation."),
    ("facct", "10.1145/3715275.3732211"): ("positive", "privacy_data_governance|security_safety|fairness", "Official CSV abstract: regulates ML societal risks including privacy/fairness violations and safety, under information asymmetry."),
    ("facct", "10.1145/3715275.3732038"): ("positive", "transparency_explainability|fairness", "Official CSV abstract: links opaque system prompts to demographic bias and argues for transparent auditing."),
    ("facct", "10.1145/3715275.3732218"): ("positive", "fairness", "Paper abstract (official DOI identified; author/arXiv context used because official CSV was blank): studies neurodivergent marginalization, model bias, and accessibility."),
    ("facct", "10.1145/3715275.3732058"): ("positive", "privacy_data_governance|fairness", "Official CSV abstract: addresses racial stereotypes and consent-driven, community-led governance of AI training collections."),
    ("iclr", "YLIsIzC74j"): ("negative", "", "Official ICLR abstract: optimizes chip macro-placement metrics; no frozen RAI dimension."),
    ("iclr", "CSj72Rr2PB"): ("negative", "", "Official ICLR abstract: 'bias' means reverse-starting and exposure error in graph diffusion, not social or demographic bias."),
    ("iclr", "d2UrCGtntF"): ("negative", "", "Official ICLR abstract: 4D novel-view synthesis with camera/time control; no frozen RAI dimension."),
    ("iclr", "KL8Sm4xRn7"): ("negative", "", "Official ICLR abstract: brain-tuning improves speech-model semantic representations; 'bias' is an inductive preference, not fairness."),
    ("iclr", "xNsIfzlefG"): ("negative", "", "Official ICLR abstract: generative modeling with hierarchical discrete distributions; no frozen RAI dimension."),
    ("iclr", "uQnvYP7yX9"): ("negative", "", "Official ICLR abstract: retrieval-based mass-spectrometry peptide sequencing; no frozen RAI dimension."),
    ("iclr", "LO4MEPoqrG"): ("positive", "security_safety", "Paper abstract: evaluates jailbreaks and toxic output from safety-trained LLMs under semantically related natural prompts."),
    ("iclr", "ptjrpEGrGg"): ("negative", "", "Official ICLR abstract: theoretical dueling-bandit learning under imperfect feedback; no frozen privacy, explanation, safety, or fairness target."),
    ("iclr", "Oh8MuCacJW"): ("positive", "transparency_explainability", "Official ICLR abstract: lexicalized motion-language features are constructed specifically to make representations interpretable."),
    ("iclr", "bDt5qc7TfO"): ("positive", "security_safety", "Official ICLR abstract: learns policies that comply with safety constraints, including autonomous-driving evaluations. Scope-borderline."),
    ("iclr", "INqLJwqUmc"): ("positive", "transparency_explainability", "Official ICLR abstract: develops an attribution framework to improve CLIP representation interpretability."),
    ("iclr", "xJXq6FkqEw"): ("positive", "transparency_explainability", "Official ICLR abstract: sparse nonnegative decision layer improves disentanglement and model interpretability."),
    ("iclr", "FEpAUnS7f7"): ("positive", "privacy_data_governance", "Official ICLR abstract: LLM agent explains privacy policies and supports informed consent and privacy management."),
    ("icml", "shen25a"): ("negative", "", "Official PDF abstract: physics-enhanced flow-field reconstruction; no frozen RAI dimension."),
    ("icml", "geirhos25a"): ("negative", "", "Official PDF abstract: flexible visual memory and editable knowledge; no frozen RAI dimension."),
    ("icml", "bai25d"): ("negative", "", "Official PDF abstract: multivariate conformal selection and FDR control; no frozen RAI dimension."),
    ("icml", "tjandrasuwita25a"): ("negative", "", "Official PDF abstract: cross-modal representation alignment; 'alignment' is geometric comparability, not safety alignment."),
    ("icml", "lawless25a"): ("positive", "transparency_explainability", "Official PDF abstract: certifies actionable recourse and gives interpretable descriptions of fixed-prediction regions."),
    ("icml", "ma25m"): ("positive", "security_safety", "Official PDF abstract: detects and purifies malicious test samples that undermine test-time model adaptation."),
    ("icml", "huang25c"): ("positive", "security_safety", "Official PDF abstract: analyzes LLM inference-time alignment failures including reward hacking."),
    ("icml", "islamov25a"): ("negative", "", "Official PDF abstract: 'safe' denotes feasibility constraints in compressed convex optimization, not AI security or societal safety."),
    ("icml", "vafa25a"): ("negative", "", "Official PDF abstract: inductive-bias probes for learned world models; 'bias' is a learning preference, not fairness."),
    ("icml", "daneshvaramoli25a"): ("negative", "", "Official PDF abstract: worst-case robustness/consistency trade-offs in online knapsack algorithms; no frozen RAI dimension."),
    ("icml", "pan25b"): ("positive", "security_safety", "Official PDF abstract: generates unrestricted adversarial examples and evaluates attack effectiveness and defense robustness."),
    ("icml", "rauba25a"): ("negative", "", "Official PDF abstract: generic distributional change testing under arbitrary interventions; it explicitly distinguishes its target from bias/fairness. Scope-borderline."),
    ("icml", "pierquin25a"): ("positive", "privacy_data_governance", "Official PDF abstract: proves when synthetic data amplifies or leaks differential-privacy guarantees."),
    ("icml", "de-castro25a"): ("positive", "privacy_data_governance", "Official PDF abstract: homomorphic encryption prevents cloud providers from learning private LLM queries."),
    ("icml", "vandenhirtz25a"): ("positive", "transparency_explainability", "Official PDF abstract: instance-wise grouped feature selection produces human-understandable model predictions."),
    ("icml", "wang25ek"): ("positive", "fairness", "Official PDF abstract: defines and mitigates feature and structural societal biases in graph generation."),
    ("neurips", "5c6c63b7ec141825ff1327563e699f76"): ("negative", "", "Official PDF abstract: coordination for household mobile manipulation; no frozen RAI dimension."),
    ("neurips", "f502981cbe221d857ad409450a7917c3"): ("negative", "", "Official PDF abstract: world simulation coverage for hazardous driving trajectories; the contribution is simulation reliability, not safety governance."),
    ("neurips", "59ea33ae3d096f3bcd5026b479710cf8"): ("negative", "", "Official PDF abstract: resource-efficient continual learning; no frozen RAI dimension."),
    ("neurips", "c6eadcc507edc04c1baf00f05cd18b1a"): ("negative", "", "Official PDF abstract: transformer loss-landscape geometry and parameter symmetries; no frozen RAI dimension."),
    ("neurips", "9a3e2737cbb40d41b4a8efe33dbf511b"): ("negative", "", "Official PDF abstract: policy-evaluation data integration under distribution shift; no frozen RAI dimension."),
    ("neurips", "0ddf14f20994636eeecc3d96fa8545cf"): ("positive", "transparency_explainability", "Official PDF abstract: discovers human-interpretable visual concepts to characterize dataset bias. Scope-borderline because the bias is not demographic."),
    ("neurips", "be3a581a2ecf289bebb563faa80e65e1"): ("positive", "privacy_data_governance", "Official PDF abstract: machine unlearning removes the influence of requested forget data."),
    ("neurips", "6c83f1f1290e80236587e4c89fa24f4e"): ("negative", "", "Official PDF abstract: object-detection augmentation; 'alignment' is content-position matching, not safety."),
    ("neurips", "af9c083d5d6bf8b860dbe57051c65984"): ("negative", "", "Official PDF abstract: optimal-transport alignment for dataset distillation; no frozen RAI dimension."),
    ("neurips", "e47619ce4fe9f5a6d09493744dc1b7de"): ("negative", "", "Official PDF abstract: robust numerical camera-pose estimation; no frozen RAI dimension."),
    ("neurips", "f539e9461f89a12caea30645605eef51"): ("positive", "security_safety", "Official PDF abstract: improves unrestricted adversarial attacks by aligning diffusion generation to attacker preferences."),
    ("neurips", "1e4b1dfc7c396205490ac8db2341f3cd"): ("negative", "", "Official PDF abstract: optimizer stability and simplicity bias; 'bias' is an optimization preference, not fairness."),
    ("neurips", "abf731c2993f9b1ee417cc3734787d7a"): ("positive", "security_safety", "Official PDF abstract: generative-AI watermarking addresses intellectual-property protection and misuse."),
    ("neurips", "903ceb0ed2d5ceec6e2c9b317b6c54a8"): ("positive", "security_safety", "Official PDF abstract: constructs multimodal jailbreaks that bypass LVLM safety guardrails."),
    ("neurips", "2656bba937d78593fbd99ace9f14e311"): ("positive", "privacy_data_governance|security_safety", "Official PDF abstract: adversarial perturbations reveal residual forgotten data, creating an explicit privacy vulnerability."),
    ("neurips", "ac1af21780d24a1345774c4a2e383972"): ("positive", "transparency_explainability", "Official PDF abstract: generates reliable visual counterfactual explanations for high-stakes model decisions."),
    ("neurips", "a1d20cc72a21ef971d7e49a90d8fa56f"): ("positive", "privacy_data_governance", "Official PDF abstract: membership-inference attacks test whether sensitive videos were used for MLLM training."),
    ("neurips", "68dced1638d71d4d21598f79ae91262e"): ("positive", "transparency_explainability", "Official PDF abstract: identifies sparse feature interactions to make LLM inference interpretable."),
    ("neurips", "0b42dd45aa4c23d3a307980c87fd87f0"): ("positive", "fairness", "Official PDF abstract: optimizes nonlinear fairness/welfare criteria in offline multi-objective RL. Scope-borderline because fairness is not protected-group specific."),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    sample_keys = {(row["venue"], row["paper_id"]) for row in rows}
    missing = sample_keys - DECISIONS.keys()
    extra = DECISIONS.keys() - sample_keys
    if missing or extra:
        raise ValueError(f"Decision/sample mismatch: missing={sorted(missing)} extra={sorted(extra)}")
    for row in rows:
        label, dimensions, notes = DECISIONS[(row["venue"], row["paper_id"])]
        row["manual_label"] = label
        row["manual_dimensions"] = dimensions
        row["review_notes"] = notes
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
