# Expanded PD model comparison

Status: completed on 26 September 2026. The reconciled archive, all 126 neural fits, classical baselines, locked evaluations, diagnostics and reviewed 17-page PDF are complete. See [the results](PD_COMPARISON_RESULTS.md), [verification record](PD_COMPARISON_VERIFICATION.md) and [frozen protocol](PD_PROTOCOL_V2.md).

This supersedes the starting protocol's permission to train before all downloads finish. The completed experiment in PD_PILOT_RESULTS.md remains a pilot: 80 participants passed training QC, 15 passed validation QC, and 20 of the 22 assigned internal-test participants passed QC. The six EEGNet tuning configurations were not six different model families.

## Readiness gate

Use `/Volumes/Extreme Pro/EEG data/download-plan.json` as the scope of the queued download. Require the SSD to be mounted, download status complete, all planned files verified, no pending retries or unresolved failures, and reconciliation of the plan with verification receipts and existing files. A stalled worker or a partial collection must not be interpreted as complete. Preserve and retry failed files through the independent archive service.

Archive completion does not guarantee a larger usable PD sample. Count unique people after reconciling original IDs, repeated visits, medication sessions, overlapping releases, acquisition provenance and recording fingerprints. In particular, reconcile the Iowa releases and overlapping New Mexico studies. Unresolved overlap must remain in one evaluation group or be excluded. Do not inflate controls with unrelated healthy tasks or inflate PD numbers with other diseases.

## Freeze protocol v2 before fitting

Audit all available PD collections for matched controls, valid resting segments, eye condition, montage, medication state and usable duration. Publish a cohort flow with source counts, deduplicated counts and QC exclusions by diagnosis/site. Report whether the actual cohort increased and by how much; do not promise a sample size based on archive gigabytes.

Create a new immutable participant registry and stratified, participant-grouped train/validation/test manifests. All sessions and representations of a person must remain together. Freeze shared preprocessing, QC, source fingerprints, channel order, duration cap, scaling policy and model selection criteria. Fit transformations only on training people, including within tuning folds. Report per-site and demographic balance.

Iowa and UCSD pilot test results have already been examined. They cannot be presented as untouched confirmatory tests in v2. Reserve an eligible, genuinely unexamined site/cohort where feasible, checking overlap with development participants. New Mexico resting segments need verified event timing; other candidate sites need the same compatibility audit. If no suitable fresh holdout exists, disclose that limitation and report grouped/nested validation and previously examined tests as exploratory. Never create a supposedly fresh test by reshuffling people whose outcomes were already used for development.

## Separate model families

Implement, fit and report each family independently:

1. Class-frequency/majority reference.
2. Regularized logistic regression on spectral bandpower features.
3. Age/sex-only confounding baseline, subject to metadata coverage.
4. Periodic/aperiodic spectral-feature logistic regression.
5. Compact EEGNet-style network.
6. ShallowConvNet.
7. Channel-wise CNN based on the external-validation PD study.
8. Wavelet-image CNN based on the PD wavelet study.
9. Graph structure/attention network.
10. Transformer using the reviewed EEG representation as a starting point.

The original research and primary-source links are in PD_CLASSIFIER_RESEARCH_PLAN.md. Re-read the papers and author implementations before specifying each new architecture. Record exact versions, architecture, features, parameter counts and departures from the paper. Call an adapted implementation an adaptation, not an exact reproduction. All classifiers use the same binary PD/control endpoint; medication state remains metadata. The demographic baseline is a confounding check rather than an EEG candidate.

## Fair local experiments

Use a shared eligible participant set for the primary comparison. If a representation requires extra exclusions, report coverage and a paired common-participant comparison as well. Freeze model-specific representations and bounded search spaces before opening tests. Use identical grouped training folds, comparable declared tuning budgets, three fixed final seeds for neural models, and validation-only stopping/threshold selection. Do not select the best seed from test scores. Record differences in compute budget and successful fits rather than silently giving one family a larger search.

Use this M4 Mac only, with bounded memory and streamed preprocessing. Benchmark each architecture on actual development data, including preprocessing and validation. Forecast the entire model campaign with a 25% buffer, record CPU/MPS fallback and memory use, and checkpoint progress. Notify the user before launching any continuous stage expected to exceed 96 hours; do not divide a longer uninterrupted campaign into artificial smaller stages to bypass this condition. Do not include cloud GPU costs.

Lock all model configurations, seeds, aggregation and thresholds before the final test evaluation. Preserve the pilot artifacts separately. Record failed experiments and their reasons; do not silently omit unsuccessful families or substitute tuning configurations for architectures.

## Comparison deliverables

Report participant-level AUROC with bootstrap 95% intervals, balanced accuracy, sensitivity, specificity, confusion counts, PR-AUC and class prevalence, Brier score, and coverage. Give each external site and medication analysis separately. Use paired participant-level bootstrap comparisons on common participants, account for repeated sessions, and describe multiple-comparison uncertainty. Include training/validation gaps, seed variability, label-permutation sanity checks and demographic/site/artifact confounding analyses. Avoid declaring a winner from tiny differences with overlapping uncertainty.

Save a machine-readable results manifest, locked configurations and reproducible commands, plus an expanded results document and model cards. Include cohort size, exclusions, model and preprocessing versions, device, training time and resource use. Distinguish discrimination from diagnosis or symptom tracking. Commit code and aggregate documentation to the existing private repository; keep raw EEG, participant-level predictions and weights out of Git under the existing data policy.

## Deferred execution

The macOS archive service remains independent of Codex usage. A follow-up in this Codex task checks readiness and resumes the audited comparison after completion. That research follow-up requires available Codex usage and a running local environment; it does not make model development independent of account limits. While downloads are incomplete, do not train further models. Escalate actionable download problems without treating unverified files as successes.
