# Reviewer-objection register

| Objection | Design response | Residual limitation / decisive extension |
|---|---|---|
| The outcome is not recovery. | Name and estimate recorded satisfaction at extract only. | No cash, partial-payment, LGD, or return claim. |
| Newer rows have less time to satisfy. | Fallback includes nonlinear age and an age-only comparator; validation is age-stratified. | Cross-sectional survivor stock remains non-longitudinal. Satisfaction dates would justify the event-time extension. |
| Satisfied rows paid within one month disappear. | Define a strict post-one-calendar-month cohort. | Cohort is conditional on remaining registered after that landmark. |
| The live-only CH file excludes dissolved companies. | Define linked live-company subpopulation and compare included/excluded records. | Do not generalise; extend with historical/dissolved CH data only if a reviewer makes this decisive. |
| Accepted matches prove precision, not recall. | Add a random unmatched audit with weights and double review. | Recall is withheld unless reviewers search an exhaustive common universe. |
| Repeated firms make rows dependent. | Keep repeats, group partitions by company, bootstrap companies. | Report both judgment and unique-company counts. |
| Prior-history features are incomplete. | Rename them observable-retained history and report calendar coverage. | Cancelled/removed and failed-link records remain missing. |
| Current CH features post-date judgment. | Exclude them from locked primary evaluation; development-only exploratory analysis. | They cannot support prospective claims. |
| Random split is not temporal validation. | Call it held-out cross-sectional validation, never temporal/prospective. | A future dated snapshot is the decisive external extension. |
| The model is judged by AUC > .70. | Report discrimination, calibration, proper scores, paired improvement, and capacity metrics. | AUC .70 remains explicitly internal, not a publication gate. |
| Model selection leaked test information. | Mask test labels, withhold class counts, freeze champion/calibration, bind a one-use approval. | Custodian procedure must be followed and documented. |
| The paper exposes commercial IP. | Publish reproducible scientific design and aggregate evidence, not fitted weights or investment rules. | Editors can inspect protected artefacts under agreement if necessary. |
| Results are data dredged. | Freeze this document, settings, code, samples, and development specification before release. | Clearly label post-release analyses exploratory. |
