# Protocol v2 completion and verification

Completed on 26 September 2026. This record covers the frozen comparison and final report, not clinical validation.

- Archive reconciliation: 41,746 files, 440,590,301,533 bytes, no unresolved audit errors. Reconciliation is not a second full-content hash pass across the archive; selected signal inputs were rehashed.
- All six neural families completed 21 fits each: 18 training-fold tuning fits and three final seeds, totaling 126. Four fitted classical/confound baselines and a class-frequency reference also completed.
- All-model lock artifacts were checked against their frozen hashes. The lock preceded test preparation. Validation-selected thresholds were preserved.
- Saved private predictions independently reproduced AUROC and confusion counts for every model and endpoint. Participant keys were unique within each endpoint. New Mexico ON/OFF endpoints share participants and are not independent replications.
- Coverage, diagnosis-specific exclusions, demographic balance, participant-level uncertainty, paired comparisons and post-lock subgroup checks appear in the report. The training-only spectral permutation check completed 100 permutations; it does not validate all neural models.
- Final local validation: all 56 tests passed; Ruff and whitespace checks passed. Aggregate JSON field scans found no participant identifiers, individual predictions or fitted weights.
- All 17 final PDF pages were rendered and visually inspected. No clipped content, overlapping text or unreadable tables remained.
- Final PDF SHA-256: `6f691175c184762a29268e6795b659a7d2e78e3e97771d02951c8af49e760995`.
- Only aggregate results, model cards, documentation, reporting code and the PDF are published. Raw signals, participant records, predictions, fitted weights and local run artifacts remain excluded from Git.

The primary Transformer AUROC of 0.786 is an observed ranking, not a clinical endorsement. Its frozen threshold detected 39 of 72 retained PD cases, and primary-site QC excluded 37.9% of assigned PD participants versus 17.9% of controls. Model rankings changed across sites. Iowa remains exploratory because overlap with the previously examined pilot cannot be resolved.
