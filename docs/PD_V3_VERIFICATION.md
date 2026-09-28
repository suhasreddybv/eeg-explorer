# V3 verification and delivery record

The 15-page final PDF passed visual review on 28 September 2026 UTC. All 69 tests passed, and Ruff and whitespace checks passed. Final report SHA-256: `839f4f60e2989237fc337857c1871d802de2b74412c116b9e1efc2ed54b555cf`.

The independent post-lock audit verified all 620 prescribed fit identities and weight hashes, all 7,988 accepted feature-cache hashes, and 12,858 candidate source receipts/stat records. Sources read for preprocessing were rehashed during preprocessing; the final audit does not claim a second complete content pass across the original archive. All reported outer performance point estimates were independently reproduced. Participant/site separation, auxiliary pretraining exclusion, model selection and thresholds passed audit. Fourteen additional fixed descriptive confound fits did not change the tested EEG models.

All four PD sites were previously examined, and the comparisons remain exploratory. The highest observed family cross-site mean was 0.763 for rich-feature SVM; the inner-selected procedure scored 0.690. The corresponding within-site procedure scored 0.693. Pretraining showed no demonstrated cross-site advantage. Recording metadata perfectly classified UCSD/New Mexico diagnoses; the report explains the study-design confound. Final all-data selection chose the pretrained hybrid SVM, whose inner selection score is not a new test result.

Only aggregate documentation, diagnostics, model cards, reporting/audit code and the PDF are published. Source EEG, participant identities, individual predictions, serialized estimators and neural weights remain local. V1/v2 records and the preliminary PDF are preserved.
