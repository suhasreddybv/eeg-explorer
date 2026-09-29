"""v4 T3 fusion-ready outputs.

The Myna EEG output schema writer, the naive-Bayes LLR combiner (flagged
unvalidated), and StackingFusion for when paired multimodal data exist.

Why the log-likelihood ratio is the exported quantity: a calibrated
probability bakes in this cohort's PD prevalence (~64%), which is an
artefact of how these research cohorts were assembled and has nothing to
do with whoever Myna eventually scores. Subtracting the training prior
leaves a prior-free measure of evidence that a downstream fusion step can
combine with its own prior.

    llr = logit(calibrated p) - logit(training prevalence)

The honest caveat, enforced in code rather than left to a footnote:
`combine_llr` assumes the modalities are conditionally independent given
diagnosis. EEG, speech and movement in Parkinson's are all downstream of
the same disease process and are certainly *not* independent, so naive
summation over-counts shared evidence and produces over-confident scores.
`demonstrate_overcounting()` exhibits this on synthetic data, and the
combiner refuses to run without an explicit acknowledgement flag.

v4 scope only. Imports from v3 modules for reuse; never edits v1/v2/v3 code.
"""

import json
from pathlib import Path

import numpy as np

RUN = Path("data/pd-runs/v4")
SCHEMA_PATH = Path("docs/myna_eeg_output_v1.schema.json")
SCHEMA_ID = "myna-eeg-output/1"


def logit(p):
    p = np.clip(np.asarray(p, dtype="float64"), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype="float64")))


def log_likelihood_ratio(calibrated_probability, training_prevalence):
    """Prior-free evidence: logit(p) - logit(training prevalence)."""
    if not 0 < training_prevalence < 1:
        raise ValueError("training_prevalence must be strictly between 0 and 1")
    return logit(calibrated_probability) - logit(training_prevalence)


def combine_llr(llrs, prior=0.5, acknowledge_unvalidated=False):
    """Naive-Bayes combination of per-modality LLRs: sum them, then add the
    prior's logit.

    UNVALIDATED. This is correct only if the modalities are conditionally
    independent given diagnosis, which is implausible for EEG/speech/
    movement in Parkinson's. Caller must pass
    `acknowledge_unvalidated=True`, so nobody can wire this into a product
    path without having read that.
    """
    if not acknowledge_unvalidated:
        raise ValueError(
            "combine_llr assumes conditional independence between modalities, which is "
            "not established for EEG/speech/movement in PD and will over-count shared "
            "evidence. Pass acknowledge_unvalidated=True to proceed anyway."
        )
    llrs = np.atleast_2d(np.asarray(llrs, dtype="float64"))
    combined = llrs.sum(axis=-1) + logit(prior)
    return {"combined_llr": combined, "probability": sigmoid(combined)}


class StackingFusion:
    """The validated alternative, for when paired multimodal data exist: fit
    a logistic regression over modality LLRs instead of assuming they are
    independent. Learns how much each modality actually adds."""

    def __init__(self, C=1.0):
        self.C = C

    def fit(self, llrs, y):
        from sklearn.linear_model import LogisticRegression

        X = np.atleast_2d(np.asarray(llrs, dtype="float64"))
        self.model_ = LogisticRegression(C=self.C, max_iter=5000).fit(X, np.asarray(y))
        return self

    def predict_proba(self, llrs):
        X = np.atleast_2d(np.asarray(llrs, dtype="float64"))
        return self.model_.predict_proba(X)


def demonstrate_overcounting(n=4000, correlation=0.95, seed=20261001):
    """Show, rather than assert, that naive-Bayes over-counts correlated
    modalities: duplicate one modality and watch the combined evidence
    double while the true information content does not change."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    signal = y * 1.5 + rng.normal(0, 1, n)
    independent = y * 1.5 + rng.normal(0, 1, n)
    # A near-duplicate of the first modality: adds almost no new information.
    duplicate = correlation * signal + np.sqrt(1 - correlation**2) * rng.normal(0, 1, n)

    single = combine_llr(signal.reshape(-1, 1), acknowledge_unvalidated=True)
    with_duplicate = combine_llr(
        np.column_stack([signal, duplicate]), acknowledge_unvalidated=True
    )
    with_independent = combine_llr(
        np.column_stack([signal, independent]), acknowledge_unvalidated=True
    )
    return {
        "mean_abs_llr_single": float(np.mean(np.abs(single["combined_llr"]))),
        "mean_abs_llr_with_near_duplicate": float(np.mean(np.abs(with_duplicate["combined_llr"]))),
        "mean_abs_llr_with_independent": float(np.mean(np.abs(with_independent["combined_llr"]))),
        "note": (
            "Adding a near-duplicate modality inflates the combined evidence almost as "
            "much as adding a genuinely independent one, although it carries little new "
            "information. That gap is the over-counting."
        ),
    }


def build_output(
    model_id,
    montage,
    regime,
    quality,
    calibrated_probability,
    training_prevalence,
    embedding,
    cognition_index=None,
):
    """One session's Myna-facing record, matching the v1 schema."""
    record = {
        "schema": SCHEMA_ID,
        "model_id": model_id,
        "montage": montage,
        "regime": regime,
        "quality": quality,
        "pd_probability": float(calibrated_probability),
        "pd_log_likelihood_ratio": float(
            log_likelihood_ratio(calibrated_probability, training_prevalence)
        ),
        "embedding": [float(v) for v in np.asarray(embedding).ravel()],
        "research_use_only": True,
    }
    if cognition_index is not None:
        record["cognition_index"] = cognition_index
    return record


def write_schema(path=SCHEMA_PATH):
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://example.invalid/myna_eeg_output_v1.schema.json",
        "title": "Myna EEG output v1",
        "description": (
            "Per-session EEG output for multimodal fusion. Research use only: this is "
            "not a diagnostic device output, not a progression measure, and not a "
            "validated wearable claim."
        ),
        "type": "object",
        "required": [
            "schema",
            "model_id",
            "montage",
            "regime",
            "quality",
            "pd_probability",
            "pd_log_likelihood_ratio",
            "embedding",
            "research_use_only",
        ],
        "additionalProperties": False,
        "properties": {
            "schema": {"const": SCHEMA_ID},
            "model_id": {"type": "string", "description": "Hash of the frozen fitted model."},
            "montage": {"type": "string", "enum": ["T1_17ch", "M4F", "M4T", "M5S", "M8"]},
            "regime": {"type": "string", "enum": ["inductive", "transductive"]},
            "quality": {
                "type": "object",
                "required": ["accepted_windows", "acceptance_fraction"],
                "properties": {
                    "accepted_windows": {"type": "integer", "minimum": 0},
                    "acceptance_fraction": {"type": "number", "minimum": 0, "maximum": 1},
                    "line_noise_ratio": {"type": "number", "minimum": 0},
                },
            },
            "pd_probability": {"type": "number", "minimum": 0, "maximum": 1},
            "pd_log_likelihood_ratio": {
                "type": "number",
                "description": (
                    "logit(calibrated p) - logit(training prevalence). Prior-free, so a "
                    "fusion step can apply its own prior instead of inheriting this "
                    "research cohort's ~64% PD rate."
                ),
            },
            "cognition_index": {
                "type": "object",
                "properties": {
                    "value": {"type": "number"},
                    "units": {"type": "string"},
                },
                "description": "Exploratory (H6); not a validated cognitive measure.",
            },
            "embedding": {
                "type": "array",
                "items": {"type": "number"},
                "description": "PCA of the stacked-feature space, fit on training people only.",
            },
            "research_use_only": {"const": True},
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(schema, indent=2) + "\n")
    return schema


def validate_output(record, schema_path=SCHEMA_PATH):
    """Structural check without adding a jsonschema dependency: required
    keys, no extras, and the constants."""
    schema = json.loads(Path(schema_path).read_text())
    missing = [key for key in schema["required"] if key not in record]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    extra = [key for key in record if key not in schema["properties"]]
    if extra:
        raise ValueError(f"Unexpected fields: {extra}")
    if record["schema"] != SCHEMA_ID:
        raise ValueError("Wrong schema identifier")
    if record["research_use_only"] is not True:
        raise ValueError("research_use_only must be true")
    if record["montage"] not in schema["properties"]["montage"]["enum"]:
        raise ValueError(f"Unknown montage {record['montage']}")
    return True


def icc_2_1(measurements):
    """ICC(2,1): two-way random effects, absolute agreement, single rater.
    `measurements` is (n_subjects, n_sessions)."""
    X = np.asarray(measurements, dtype="float64")
    n, k = X.shape
    grand = X.mean()
    between_subject = k * ((X.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    between_session = n * ((X.mean(axis=0) - grand) ** 2).sum() / (k - 1)
    residual = (
        ((X - X.mean(axis=1, keepdims=True) - X.mean(axis=0, keepdims=True) + grand) ** 2).sum()
        / ((n - 1) * (k - 1))
    )
    denominator = between_subject + (k - 1) * residual + k * (between_session - residual) / n
    if denominator <= 0:
        return float("nan")
    return float((between_subject - residual) / denominator)
