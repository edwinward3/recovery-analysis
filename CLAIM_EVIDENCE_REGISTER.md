# Claim–evidence register

Update this table whenever a draft adds or changes a substantive claim. No claim
enters the paper without a named source or a reproducible table/figure.

| ID | Draft claim | Evidence required | Current evidence | Status |
|---|---|---|---|---|
| M1 | Satisfied is a fully paid judgment recorded after one month with evidence. | Current Registry Trust guidance and data dictionary. | Registry Trust public guidance; RT confirmation still to archive with project records. | provisional |
| M2 | Unsatisfied includes unpaid and partially unpaid judgments. | Registry Trust definition. | User-supplied institutional fact; obtain and archive primary wording. | provisional |
| M3 | The supplied schema cannot identify satisfaction timing. | Raw headers, workbook schema, ingestion audit. | Supplied sample has no satisfaction/event date; schema audit now fails closed. | supported for sample; confirm full extract |
| M4 | The extract is current register stock, not historical inflow/outflow. | RT delivery specification plus one-row-per-ID/schema checks. | Repository construct and supplied sample are cross-sectional; written RT delivery confirmation required. | provisional |
| P1 | Recorded-satisfaction prevalence in corporate E&W register stock is X. | Locked E1 descriptive table with denominator and observation date. | Not run. | pending |
| L1 | Exact-match precision is X. | Double-reviewed accepted probability sample and weighted interval. | Sampling/estimation code complete; adjudication pending. | pending |
| L2 | Missed-link prevalence among unmatched rows is X. | Double-reviewed unmatched probability sample and weighted interval. | Sampling/estimation code complete; adjudication pending. | pending |
| L3 | Linkage recall is X. | L1/L2 plus certified common exhaustive search universe. | Deliberately withheld unless certification is met. | gated |
| S1 | Linked and excluded populations differ on X. | Prespecified included/excluded aggregate comparison. | Code/report integration pending full-data run. | pending |
| R1 | Sparse information improves on age alone by X. | One-time paired locked-test estimates and company-clustered intervals. | Test locked. | pending |
| R2 | Model calibration is adequate/inadequate. | Calibration intercept, slope, flexible curve, Brier/log loss; cautious language. | Test locked. | pending |
| R3 | Ranking at capacity k is X. | Prespecified 1/2/5/10/20% locked-test table with intervals. | Test locked. | pending |

Forbidden substitutions: recorded satisfaction = cash recovery; Unsatisfied = zero
payment; AUC = commercial value; association = causation; current live-company
linked cohort = all corporate judgment debtors.
