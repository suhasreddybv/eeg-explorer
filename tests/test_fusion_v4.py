"""v4 T3 fusion outputs: the prior-free LLR, the schema contract, the
naive-Bayes over-counting demonstration, and ICC(2,1)."""

import json
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

from pd_pipeline import fusion_v4 as fu  # noqa: E402


def test_llr_is_zero_at_the_training_prevalence():
    """The point of the LLR: a person scored exactly at the training prior
    carries no evidence either way, whatever that prior happens to be."""
    for prevalence in (0.1, 0.5, 0.645, 0.9):
        assert fu.log_likelihood_ratio(prevalence, prevalence) == pytest.approx(0.0, abs=1e-9)


def test_llr_removes_the_cohort_prevalence_artifact():
    # Same evidence, two cohorts with different PD rates: the probability
    # differs but the LLR is what a fusion step should see.
    p_balanced = 0.75
    llr = fu.log_likelihood_ratio(p_balanced, 0.5)
    # Re-express that evidence against a 64.5% prevalence cohort.
    p_enriched = fu.sigmoid(llr + fu.logit(0.645))
    assert p_enriched > p_balanced  # same evidence, higher prior -> higher probability
    assert fu.log_likelihood_ratio(p_enriched, 0.645) == pytest.approx(llr, abs=1e-9)


def test_llr_rejects_impossible_prevalence():
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="strictly between"):
            fu.log_likelihood_ratio(0.5, bad)


def test_combine_llr_refuses_without_acknowledgement():
    with pytest.raises(ValueError, match="conditional independence"):
        fu.combine_llr([[1.0, 2.0]])


def test_combine_llr_matches_analytic_result_for_independent_gaussians():
    """With truly independent evidence, summing LLRs is the correct Bayes
    update -- so the combiner should reproduce the analytic posterior."""
    llr_a, llr_b, prior = 1.2, -0.4, 0.3
    out = fu.combine_llr([[llr_a, llr_b]], prior=prior, acknowledge_unvalidated=True)
    expected_logit = llr_a + llr_b + fu.logit(prior)
    assert out["combined_llr"][0] == pytest.approx(expected_logit)
    assert out["probability"][0] == pytest.approx(1 / (1 + np.exp(-expected_logit)))


def test_naive_bayes_overcounts_correlated_modalities():
    """The failure mode, demonstrated rather than asserted: a near-duplicate
    modality inflates the combined evidence about as much as a genuinely
    independent one, despite adding little information."""
    result = fu.demonstrate_overcounting()
    single = result["mean_abs_llr_single"]
    duplicated = result["mean_abs_llr_with_near_duplicate"]
    independent = result["mean_abs_llr_with_independent"]
    assert duplicated > 1.7 * single  # nearly doubles on redundant evidence
    assert duplicated >= independent * 0.95  # no better than truly new evidence


def test_stacking_fusion_learns_to_discount_a_useless_modality():
    rng = np.random.default_rng(0)
    n = 600
    y = rng.integers(0, 2, n)
    informative = y * 2.0 + rng.normal(0, 1, n)
    noise = rng.normal(0, 1, n)
    model = fu.StackingFusion().fit(np.column_stack([informative, noise]), y)
    weights = np.abs(model.model_.coef_.ravel())
    assert weights[0] > 3 * weights[1]


def test_schema_round_trip_and_validation(tmp_path):
    path = tmp_path / "schema.json"
    fu.write_schema(path)
    record = fu.build_output(
        model_id="abc123",
        montage="M8",
        regime="inductive",
        quality={"accepted_windows": 15, "acceptance_fraction": 0.83, "line_noise_ratio": 0.04},
        calibrated_probability=0.71,
        training_prevalence=0.645,
        embedding=np.arange(32) / 32,
    )
    assert fu.validate_output(record, path) is True
    assert record["research_use_only"] is True
    assert record["pd_log_likelihood_ratio"] == pytest.approx(
        fu.logit(0.71) - fu.logit(0.645)
    )


def test_schema_validation_rejects_bad_records(tmp_path):
    path = tmp_path / "schema.json"
    fu.write_schema(path)
    good = fu.build_output(
        model_id="x", montage="M8", regime="inductive",
        quality={"accepted_windows": 15, "acceptance_fraction": 0.8},
        calibrated_probability=0.6, training_prevalence=0.5, embedding=[0.1],
    )
    missing = {k: v for k, v in good.items() if k != "pd_probability"}
    with pytest.raises(ValueError, match="Missing required"):
        fu.validate_output(missing, path)
    unknown = {**good, "surprise": 1}
    with pytest.raises(ValueError, match="Unexpected fields"):
        fu.validate_output(unknown, path)
    bad_montage = {**good, "montage": "M99"}
    with pytest.raises(ValueError, match="Unknown montage"):
        fu.validate_output(bad_montage, path)
    not_research = {**good, "research_use_only": False}
    with pytest.raises(ValueError, match="research_use_only"):
        fu.validate_output(not_research, path)


def test_written_schema_is_valid_json_and_marks_research_use():
    schema = fu.write_schema()
    reparsed = json.loads(json.dumps(schema))
    assert reparsed["properties"]["research_use_only"]["const"] is True
    assert "research use only" in reparsed["description"].lower()


def test_icc_is_high_for_stable_repeat_measures_and_low_for_noise():
    rng = np.random.default_rng(1)
    subject_effect = rng.normal(0, 2, 40).reshape(-1, 1)
    stable = subject_effect + rng.normal(0, 0.2, (40, 2))
    assert fu.icc_2_1(stable) > 0.9
    noise = rng.normal(0, 1, (40, 2))
    assert fu.icc_2_1(noise) < 0.4


def test_icc_matches_hand_computed_perfect_agreement():
    values = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    assert fu.icc_2_1(values) == pytest.approx(1.0, abs=1e-9)
