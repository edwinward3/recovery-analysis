# Frozen study design: Registry Trust recorded satisfaction

Status: validity-gate specification. The test set must remain unopened until the
diagnostic run, double-reviewed linkage audits, development run, manifest freeze,
and independent one-time approval are complete.

## 1. Measurement decision

The supplied Registry Trust schema is a single current-register snapshot with one
current status per judgment. It has no satisfaction date, cancellation history,
or repeated historical snapshots. The implemented study is therefore the
cross-sectional fallback, not a time-to-event or fixed-horizon recovery study.

`Satisfied` means Registry Trust records the judgment as fully paid after one
calendar month and has received supporting evidence. `Unsatisfied` combines
unpaid and partially unpaid judgments. The outcome is administrative recorded
satisfaction at the extract date. It is not cash recovery, partial recovery,
loss given default, investment return, or future satisfaction.

If a real extract contains a satisfaction date, cancellation history, or multiple
snapshots, modelling stops. Those fields reopen the design gate and require the
preferred event-time/fixed-horizon protocol; they must not be ignored to preserve
this fallback.

## 2. Research question and estimands

Question: Among corporate judgments in England and Wales that remain on the
Register more than one and no more than 48 calendar months after judgment, which
are recorded as satisfied at the extract date, and how much does sparse
administrative information improve held-out identification beyond judgment age
alone in the uniquely linked live-company subpopulation?

Three quantities are distinct:

1. Descriptive estimand: the proportion recorded satisfied at the extract date
   among all in-scope corporate England and Wales records in the observed RT
   register stock, overall and by prespecified age, amount, and vintage groups.
2. Linkage estimands: accepted-link precision and missed-link prevalence among
   algorithmically unmatched records in the same corporate England and Wales
   target population. Linkage recall is reported only if adjudicators search a
   common, sufficiently exhaustive company universe, including dissolved
   companies where relevant.
3. Predictive estimand: held-out performance in the eligible, unique-exact-linked
   live-company subpopulation, and the paired difference between the frozen
   primary model and a flexible judgment-age-only model. This is conditional on
   survival into the current register stock and successful linkage; it is not a
   population-wide causal or prospective recovery effect.

The unit is a judgment. All eligible repeated judgments are retained. Every
judgment belonging to the same company is assigned to the same partition, and
uncertainty is resampled by company.

## 3. Populations and dates

- Descriptive population: all RT records marked Corporate and England and Wales.
- Binary status population: the descriptive population restricted to Satisfied
  or Unsatisfied.
- Primary age population: binary status records strictly older than one calendar
  month and no older than 48 calendar months on the declared RT observation date.
- Predictive population: the primary age population with one date-valid unique
  exact normalized-name link to the declared Companies House live-company bulk
  snapshot.
- Linkage-validation population: all corporate England and Wales RT rows sent to
  the exact matcher, before restriction to the predictive cohort.

The RT observation date and Companies House snapshot date are mandatory inputs.
The Companies House snapshot may not post-date the RT observation date and may be
at most 35 days earlier. A date embedded in its filename must agree with the
declared date.

The 48-month ceiling is prespecified because a 24-month prior-record window can
then fall within the Register's six-year retention span. It does not make that
history complete: cancelled/removed records and records that failed linkage are
unobserved.

## 4. Linkage and selection audit

The production link remains conservative: one date-valid exact normalized
company/former/trading name candidate. Postcode describes agreement but never
creates or selects a link. The Companies House bulk file contains live companies
only; dissolved defendants are structurally excluded from the linked population.

The diagnostic run creates two outcome-blind files:

- 1,000 accepted exact links selected by a seeded equal-probability systematic
  design spread over matching characteristics; and
- up to 1,000 unmatched records selected by seeded probability sampling within
  unmatched-reason by judgment-year strata (a census if fewer are available).

Two reviewers independently label every row, disagreements are adjudicated, and
company numbers are recorded for missed links. Sampling probabilities and weights
must remain unchanged. The development and locked stages regenerate the samples
and reject altered membership or sampling metadata. They report design-weighted
precision and missed-link prevalence with declared intervals, reviewer agreement,
and guarded recall. Outcomes and Companies House current status are forbidden in
review files.

Included and excluded records are compared on prespecified observable RT
characteristics. Differences are selection evidence, not corrections for
unobserved dissolved-company outcomes.

## 5. Prediction protocol

Partitioning is outcome-blind, deterministic, age-stratified, and grouped by
company: 60% train, 15% validation, 10% calibration, and 15% locked test. This is
held-out cross-sectional validation, not chronological or fixed-horizon temporal
validation. Test outcomes are absent from the development object and test class
counts are not reported before release.

Prespecified comparators are:

1. the training prevalence;
2. flexible nonlinear judgment age at observation (logistic regression);
3. age plus sparse judgment-time/reconstructable administrative variables,
   fitted by logistic regression and LightGBM.

The primary candidate family includes judgment age, judgment amount, company age
at judgment, and explicitly named observable-retained prior-judgment measures.
Current Companies House status, accounts, and charge variables can be examined in
development only as retrospective snapshot variables; they are not evaluated on
the locked test and cannot support a prospective claim.

Logistic regression is the simplicity default. LightGBM is selected on validation
only if it reduces Brier score by at least 1% without reducing ROC-AUC. The chosen
model is refit on train plus validation. Calibration uses the separate calibration
partition: isotonic at the prespecified sample threshold, otherwise Platt scaling,
otherwise an explicit underpowered result.

The one-time locked evaluation contains only the prevalence baseline, the
age-only baseline, and the frozen primary champion. It reports ROC-AUC, average
precision, Brier score, log loss, calibration-in-the-large, calibration intercept
and slope, a reliability plot/table, and ranking performance at 1%, 2%, 5%, 10%,
and 20% capacities. Company-clustered bootstrap intervals and paired incremental
intervals against age-only are primary uncertainty summaries.

The repository's AUC 0.70 rule is an internal operational screen. It is labelled
as such, is not a publication criterion, and cannot turn a scientifically valid
null or weak result into a failed study.

## 6. Release and disclosure protocol

Development writes an outcome-blind frozen specification. The release manifest
binds the RT and Companies House bytes, both dates, settings, analysis source,
dependency lock, this design, both completed linkage-adjudication files, and the
development specification. A custodian signs that manifest with a private key
kept outside the repository. The approval is atomically consumed before the
locked run opens the test outcome; completed and failed attempts both require a
new approval.

Only aggregate, disclosure-checked tables leave the secure environment. Published
materials may describe the scientific question, matching logic, prespecified
features, validation design, and aggregate results. They do not need to disclose
production coefficients, fitted model weights, acquisition cut-offs, or any
integrated investment architecture.

## 7. Interpretation rules

- Say `recorded satisfied`, never `recovered`, unless separately evidenced.
- Do not infer payment timing without a satisfaction date.
- Do not infer partial repayment from Unsatisfied.
- Do not call the fallback prospective, longitudinal, survival, or fixed-horizon.
- Do not generalise linked-cohort performance to dissolved or unmatched companies.
- Do not treat a high accepted-link precision estimate as evidence of high recall.
- Do not select a journal until the institutional, linkage, and predictive results
  show which contribution dominates.
