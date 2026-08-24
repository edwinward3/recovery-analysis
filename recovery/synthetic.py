"""Creates the fake RT and Companies House files used by the self-test."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


@dataclass(slots=True)
class SyntheticBundle:
    judgments: pd.DataFrame
    companies_house: pd.DataFrame
    ground_truth: pd.DataFrame
    observation_date: date


def make_synthetic_bundle(
    n_companies: int = 8_000,
    seed: int = 20260618,
    *,
    include_prior_rows: bool = True,
    include_event_dates: bool = False,
) -> SyntheticBundle:
    """Return fake inputs with exact, postcode-change and non-match cases."""

    if n_companies < 100:
        raise ValueError("n_companies must be at least 100")
    rng = np.random.default_rng(seed)
    observed = pd.Timestamp("2026-06-01")
    company_ids = np.arange(n_companies)
    numbers = np.array([f"{i + 10_000_000:08d}" for i in company_ids], dtype=object)
    base_names = np.array([f"ALPHA {i:06d} SERVICES LIMITED" for i in company_ids], dtype=object)
    current_names = base_names.copy()
    current_postcodes = np.array([_postcode(i) for i in company_ids], dtype=object)
    target_dates = observed - pd.to_timedelta(rng.integers(18 * 30, 30 * 30, n_companies), unit="D")
    incorporation = target_dates - pd.to_timedelta(rng.integers(2 * 365, 25 * 365, n_companies), unit="D")
    amounts = np.exp(rng.normal(np.log(1_800), 1.0, n_companies)).clip(50, 250_000)
    prior_flag = rng.random(n_companies) < 0.28
    prior_amount = np.where(prior_flag, amounts * rng.uniform(0.4, 1.6, n_companies), 0.0)
    company_age = (target_dates - incorporation).days / 365.25
    latent = -1.9 + 0.045 * company_age - 0.000004 * amounts - 0.55 * prior_flag
    p_satisfied = 1 / (1 + np.exp(-latent))
    satisfied = rng.random(n_companies) < p_satisfied
    cancelled = (~satisfied) & (rng.random(n_companies) < 0.12)

    corruption = np.array(
        [
            (
                "clean",
                "punctuation",
                "same_postcode_wrong_name",
                "trading",
                "former",
                "postcode_drift",
                "same_postcode_ambiguous",
                "unmatched",
            )[i % 8]
            for i in company_ids
        ],
        dtype=object,
    )

    previous_names = np.full(n_companies, "", dtype=object)
    previous_change_dates = np.full(n_companies, "", dtype=object)
    source_names = base_names.copy()
    source_trading = np.full(n_companies, "", dtype=object)
    source_postcodes = current_postcodes.copy()

    for i, cls in enumerate(corruption):
        if cls == "punctuation":
            source_names[i] = f"ALPHA-{i:06d} SERVICES LTD."
        elif cls == "same_postcode_wrong_name":
            source_names[i] = f"ALFA {i:06d} SERVICE GROUP"
        elif cls == "trading":
            source_names[i] = f"TRADING STYLE {i:06d}"
            source_trading[i] = base_names[i]
        elif cls == "former":
            previous_names[i] = f"OLD ALPHA {i:06d} LIMITED"
            previous_change_dates[i] = (target_dates[i] + pd.Timedelta(days=120)).strftime("%d/%m/%Y")
            source_names[i] = previous_names[i]
            current_names[i] = f"NEW ALPHA {i:06d} LIMITED"
        elif cls == "postcode_drift":
            source_postcodes[i] = _old_postcode(i)
        elif cls == "same_postcode_ambiguous":
            source_names[i] = f"ALPHA {i:06d} SERVICE"
        elif cls == "unmatched":
            source_names[i] = f"UNRELATED DEFENDANT {i:06d}"
            source_postcodes[i] = _old_postcode(i)

    n_charges = rng.poisson(0.8, n_companies)
    n_satisfied = np.minimum(n_charges, rng.binomial(n_charges, 0.35))
    accounts_due = np.where(
        satisfied,
        observed + pd.to_timedelta(rng.integers(30, 500, n_companies), unit="D"),
        observed - pd.to_timedelta(rng.integers(1, 500, n_companies), unit="D"),
    )
    company_status = np.where(satisfied | (rng.random(n_companies) > 0.2), "Active", "Liquidation")

    ch = pd.DataFrame(
        {
            "CompanyName": current_names,
            "CompanyNumber": numbers,
            "RegAddress.PostCode": current_postcodes,
            "CompanyStatus": company_status,
            "CompanyCategory": "Private Limited Company",
            "IncorporationDate": pd.Series(incorporation).dt.strftime("%d/%m/%Y"),
            "Accounts.NextDueDate": pd.Series(accounts_due).dt.strftime("%d/%m/%Y"),
            "Mortgages.NumMortCharges": n_charges,
            "Mortgages.NumMortOutstanding": np.maximum(n_charges - n_satisfied, 0),
            "Mortgages.NumMortPartSatisfied": 0,
            "Mortgages.NumMortSatisfied": n_satisfied,
            "PreviousName_1.CompanyName": previous_names,
            "PreviousName_1.CONDATE": previous_change_dates,
        }
    )

    # Add a near-identical company at the same postcode for the ambiguity class.
    amb_idx = np.flatnonzero(corruption == "same_postcode_ambiguous")
    if len(amb_idx):
        extra = ch.iloc[amb_idx].copy()
        extra["CompanyNumber"] = [f"{i + 80_000_000:08d}" for i in amb_idx]
        extra["CompanyName"] = [f"ALPHA {i:06d} SERVICE GROUP LIMITED" for i in amb_idx]
        ch = pd.concat([ch, extra], ignore_index=True)

    target = pd.DataFrame(
        {
            "ID": [f"J-{i:07d}" for i in company_ids],
            "Date Inserted": pd.Series(target_dates + pd.Timedelta(days=1)).dt.strftime("%d/%m/%Y"),
            "JudgmentDate": pd.Series(target_dates).dt.strftime("%d/%m/%Y"),
            "JudgmentStatus": np.select(
                [satisfied, cancelled], ["Satisfied", "Cancelled"], default="Unsatisfied"
            ),
            "DefendantType": "Corporate",
            "Jurisdiction": "England and Wales",
            "Defendant Company Name": source_names,
            "Defendant_Postcode": source_postcodes,
            "Amount": amounts.round(2),
            "Defendant Trading Name": source_trading,
            "Defendant Address": "",
        }
    )
    if include_event_dates:
        landmark = pd.Series(target_dates).map(
            lambda value: value + pd.DateOffset(months=1)
        )
        event_dates = landmark + pd.to_timedelta(
            rng.integers(1, 330, n_companies), unit="D"
        )
        target["Satisfaction Date"] = pd.Series(event_dates).where(satisfied).dt.strftime(
            "%d/%m/%Y"
        )
        target["Cancellation Date"] = pd.Series(event_dates).where(cancelled).dt.strftime(
            "%d/%m/%Y"
        )

    # Add older judgments to test repeated companies.
    if include_prior_rows:
        prior_indices = np.flatnonzero(prior_flag)
        prior_dates = target_dates[prior_indices] - pd.to_timedelta(
            rng.integers(19 * 30, 23 * 30, len(prior_indices)), unit="D"
        )
        prior = pd.DataFrame(
            {
                "ID": [f"P-{i:07d}" for i in prior_indices],
                "Date Inserted": pd.Series(
                    prior_dates + pd.Timedelta(days=1)
                ).dt.strftime("%d/%m/%Y"),
                "JudgmentDate": pd.Series(prior_dates).dt.strftime("%d/%m/%Y"),
                "JudgmentStatus": "Unsatisfied",
                "DefendantType": "Corporate",
                "Jurisdiction": "England and Wales",
                "Defendant Company Name": base_names[prior_indices],
                "Defendant_Postcode": current_postcodes[prior_indices],
                "Amount": prior_amount[prior_indices].round(2),
                "Defendant Trading Name": "",
                "Defendant Address": "",
            }
        )
        if include_event_dates:
            prior["Satisfaction Date"] = ""
            prior["Cancellation Date"] = ""
        judgments = pd.concat([target, prior], ignore_index=True)
    else:
        judgments = target.copy()
    ground_truth = pd.DataFrame(
        {
            "ID": target["ID"],
            "expected_company_number": numbers,
            "corruption_class": corruption,
        }
    )
    return SyntheticBundle(judgments, ch, ground_truth, observed.date())


def write_bundle(bundle: SyntheticBundle, directory: str | Path, *, excel: bool = False) -> tuple[Path, Path, Path]:
    """Write fake judgment, zipped CH and ground-truth files for CI."""

    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    judgment_path = destination / ("synthetic judgments.xlsx" if excel else "synthetic judgments.csv")
    if excel:
        bundle.judgments.to_excel(judgment_path, index=False)
    else:
        bundle.judgments.to_csv(judgment_path, index=False)
    ch_csv = destination / "BasicCompanyDataAsOneFile-2026-06-01.csv"
    bundle.companies_house.to_csv(ch_csv, index=False)
    ch_zip = destination / "BasicCompanyDataAsOneFile-2026-06-01.zip"
    with zipfile.ZipFile(ch_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(ch_csv, arcname=ch_csv.name)
    ch_csv.unlink()
    truth_path = destination / "synthetic_ground_truth.csv"
    bundle.ground_truth.to_csv(truth_path, index=False)
    return judgment_path, ch_zip, truth_path


def _postcode(index: int) -> str:
    return _encoded_postcode(index, "ABCDEFGHIJKLM")


def _old_postcode(index: int) -> str:
    return _encoded_postcode(index, "NOPQRSTUVWXYZ")


def _encoded_postcode(index: int, first_letters: str) -> str:
    """Return a unique UK-shaped synthetic postcode for more than 20m rows."""

    if index < 0:
        raise ValueError("synthetic postcode index cannot be negative")
    value = int(index)
    first = first_letters[value % len(first_letters)]
    value //= len(first_letters)
    second = chr(65 + value % 26)
    value //= 26
    outward_digit = value % 10
    value //= 10
    inward_digit = value % 10
    value //= 10
    inward_first = chr(65 + value % 26)
    value //= 26
    inward_second = chr(65 + value % 26)
    value //= 26
    if value:
        raise ValueError("synthetic postcode space exhausted")
    return f"{first}{second}{outward_digit} {inward_digit}{inward_first}{inward_second}"
