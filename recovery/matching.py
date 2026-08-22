"""Matches the judgments to Companies House and makes the 1,000-pair file for RT."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import hashlib
import math
import re
import unicodedata

import pandas as pd
from scipy.stats import norm

from .config import Settings
from .data import iter_ch_chunks


CH_REQUIRED_COLUMNS: tuple[str, ...] = (
    "CompanyName",
    "CompanyNumber",
    "RegAddress.PostCode",
    "IncorporationDate",
)

CH_CURRENT_SNAPSHOT_COLUMNS: tuple[str, ...] = (
    "CompanyStatus",
    "CompanyCategory",
    "Accounts.NextDueDate",
    "Accounts.LastMadeUpDate",
    "Accounts.AccountCategory",
    "Mortgages.NumMortCharges",
    "Mortgages.NumMortOutstanding",
    "Mortgages.NumMortPartSatisfied",
    "Mortgages.NumMortSatisfied",
)

MATCH_TIERS: tuple[str, ...] = ("exact_unique", "unmatched")

ACCEPTED_LINKAGE_VALIDATION_FILENAME = "linkage_validation_accepted.csv"
UNMATCHED_LINKAGE_VALIDATION_FILENAME = "linkage_validation_unmatched.csv"

LINKAGE_REVIEW_COLUMNS: tuple[str, ...] = (
    "reviewer_1_label",
    "reviewer_1_company_number",
    "reviewer_2_label",
    "reviewer_2_company_number",
    "adjudicated_label",
    "adjudicated_company_number",
    "adjudication_notes",
)

_ARM_LABELS: dict[str, frozenset[str]] = {
    "accepted": frozenset(("correct_match", "incorrect_match")),
    "unmatched": frozenset(("missed_match", "true_unmatched")),
}

_POSITIVE_LABEL: dict[str, str] = {
    "accepted": "correct_match",
    "unmatched": "missed_match",
}

_FORBIDDEN_VALIDATION_COLUMNS: frozenset[str] = frozenset(
    {
        "judgmentstatus",
        "satisfactiondate",
        "cancellationdate",
        "cancellationreason",
        "companystatus",
        "outcome",
        "target",
        "issatisfied",
        "satisfied",
        "unsatisfied",
        "y",
    }
)

_LEGAL_SUFFIXES: tuple[tuple[str, ...], ...] = tuple(
    tuple(value.split())
    for value in (
        "PRIVATE LIMITED COMPANY",
        "PUBLIC LIMITED COMPANY",
        "LIMITED LIABILITY PARTNERSHIP",
        "LIMITED",
        "LTD",
        "PLC",
        "LLP",
        "LP",
        "LLC",
        "CIC",
        "INCORPORATED",
        "INC",
        "CO",
        "COMPANY",
        "IN LIQUIDATION",
        "IN ADMINISTRATION",
    )
)

_CH_HEADER_ALIASES: dict[str, str] = {
    re.sub(r"[^a-z0-9]+", "", name.lower()): name
    for name in (*CH_REQUIRED_COLUMNS, *CH_CURRENT_SNAPSHOT_COLUMNS)
}


# Prepare names and postcodes for comparison

def normalize_name(value: object) -> str:
    """Make an ASCII uppercase form, removing qualifiers, punctuation and suffixes."""

    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    text = text.upper().replace("&", " AND ")
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.split(r"\b(?:T\s*/\s*A|TRADING\s+AS|FORMERLY)\b", text, maxsplit=1)[0]
    words = re.sub(r"[^A-Z0-9]+", " ", text).split()
    changed = True
    while words and changed:
        changed = False
        for suffix in _LEGAL_SUFFIXES:
            if len(words) >= len(suffix) and tuple(words[-len(suffix) :]) == suffix:
                del words[-len(suffix) :]
                changed = True
                break
    return " ".join(words)


def normalize_postcode(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"[^A-Z0-9]+", "", str(value).upper())


@dataclass(frozen=True, slots=True)
class NamePeriod:
    raw_name: str
    normalized_name: str
    valid_from: pd.Timestamp | None
    valid_to: pd.Timestamp | None
    kind: str

    def valid_on(self, when: pd.Timestamp) -> bool:
        return (
            (self.valid_from is None or when >= self.valid_from)
            and (self.valid_to is None or when < self.valid_to)
        )


@dataclass(frozen=True, slots=True)
class CompanyRecord:
    company_number: str
    company_name: str
    postcode: str
    incorporation_date: pd.Timestamp | None
    names: tuple[NamePeriod, ...]
    all_normalized_names: frozenset[str]
    name_history_complete: bool
    current_snapshot: Mapping[str, str]

    def valid_names(self, when: pd.Timestamp) -> tuple[NamePeriod, ...]:
        if (
            self.incorporation_date is None
            or not self.name_history_complete
            or when < self.incorporation_date
        ):
            return ()
        return tuple(name for name in self.names if name.valid_on(when))


@dataclass(frozen=True, slots=True)
class CHIndex:
    companies: Mapping[str, CompanyRecord]
    by_exact_name: Mapping[str, tuple[str, ...]]
    stats: Mapping[str, object]


# Read Companies House and keep possible candidates

def _canon_header(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _previous_column(value: object) -> str | None:
    canonical = _canon_header(value)
    match = re.fullmatch(r"previousname(\d+)(companyname|condate)", canonical)
    if not match:
        return None
    suffix = "CompanyName" if match.group(2) == "companyname" else "CONDATE"
    return f"PreviousName_{int(match.group(1))}.{suffix}"


def _want_ch_column(value: object) -> bool:
    return _canon_header(value) in _CH_HEADER_ALIASES or _previous_column(value) is not None


def _standardise_ch_headers(frame: pd.DataFrame) -> pd.DataFrame:
    rename: dict[object, str] = {}
    seen: dict[str, object] = {}
    for original in frame.columns:
        standard = _CH_HEADER_ALIASES.get(_canon_header(original)) or _previous_column(original)
        if standard is None:
            continue
        if standard in seen:
            raise ValueError(
                f"duplicate Companies House columns map to {standard!r}: "
                f"{seen[standard]!r} and {original!r}"
            )
        seen[standard] = original
        rename[original] = standard
    return frame.rename(columns=rename)


def _parse_dates(series: pd.Series) -> pd.Series:
    raw = series.astype("string").fillna("").str.strip()
    iso = raw.str.fullmatch(r"\d{4}-\d{2}-\d{2}(?:[ T].*)?", na=False)
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    if iso.any():
        parsed.loc[iso] = pd.to_datetime(
            raw.loc[iso], format="ISO8601", errors="coerce"
        ).astype("datetime64[ns]")
    if (~iso).any():
        parsed.loc[~iso] = pd.to_datetime(
            raw.loc[~iso], format="mixed", dayfirst=True, errors="coerce"
        ).astype("datetime64[ns]")
    return parsed.dt.normalize()


def _name_periods(
    current_name: str,
    incorporation_date: pd.Timestamp | None,
    former: Iterable[tuple[str, pd.Timestamp | None]],
) -> tuple[tuple[NamePeriod, ...], int, int]:
    dated: list[tuple[pd.Timestamp, str, str]] = []
    undated = 0
    invalid_dates = 0
    for raw_name, change_date in former:
        normalized = normalize_name(raw_name)
        if not normalized:
            continue
        if change_date is None:
            undated += 1
            continue
        if incorporation_date is not None and change_date < incorporation_date:
            invalid_dates += 1
            continue
        dated.append((change_date, str(raw_name).strip(), normalized))
    dated.sort(key=lambda item: (item[0], item[2]))

    names: list[NamePeriod] = []
    cursor = incorporation_date
    for change_date, raw_name, normalized in dated:
        if cursor is not None and change_date <= cursor:
            invalid_dates += 1
            continue
        names.append(NamePeriod(raw_name, normalized, cursor, change_date, "former"))
        cursor = change_date

    normalized_current = normalize_name(current_name)
    if normalized_current:
        names.append(
            NamePeriod(str(current_name).strip(), normalized_current, cursor, None, "current")
        )
    return tuple(names), undated, invalid_dates


def _relevant_names(judgments: pd.DataFrame) -> set[str]:
    required = {"Defendant Company Name"}
    missing = required - set(judgments.columns)
    if missing:
        raise ValueError(f"judgment data missing matching column(s): {sorted(missing)}")
    names: set[str] = set()
    for column in ("Defendant Company Name", "Defendant Trading Name"):
        if column in judgments:
            names.update(
                normalized
                for normalized in judgments[column].map(normalize_name).unique().tolist()
                if normalized
            )
    return names


def build_relevant_ch_index(
    judgments: pd.DataFrame,
    ch_path: str | Path,
    *,
    chunksize: int = 100_000,
) -> CHIndex:
    """Stream CH once, retaining only exact normalized-name candidates."""

    relevant_names = _relevant_names(judgments)
    companies: dict[str, CompanyRecord] = {}
    rows_read = 0
    rows_retained = 0
    duplicate_company_rows = 0
    undated_former_names = 0
    invalid_former_dates = 0
    first_chunk = True
    analysis_digest = hashlib.sha256()

    for raw_chunk in iter_ch_chunks(ch_path, chunksize=chunksize, usecols=_want_ch_column):
        if first_chunk:
            analysis_digest.update(
                "\x1f".join(map(str, raw_chunk.columns)).encode("utf-8")
            )
        row_hashes = pd.util.hash_pandas_object(
            raw_chunk, index=False, categorize=False
        )
        analysis_digest.update(
            row_hashes.to_numpy(dtype="uint64", copy=False).tobytes()
        )
        chunk = _standardise_ch_headers(raw_chunk)
        if first_chunk:
            missing = set(CH_REQUIRED_COLUMNS) - set(chunk.columns)
            if missing:
                raise ValueError(
                    f"Companies House file is missing required column(s): {sorted(missing)}"
                )
            first_chunk = False
        rows_read += len(chunk)

        current_norm = chunk["CompanyName"].map(normalize_name)
        retain = current_norm.isin(relevant_names)
        former_name_columns = sorted(
            column
            for column in chunk.columns
            if re.fullmatch(r"PreviousName_\d+\.CompanyName", column)
        )
        for column in former_name_columns:
            raw_former = chunk[column].astype("string").fillna("").str.strip()
            nonempty = raw_former.ne("")
            if nonempty.any():
                retain.loc[nonempty] |= raw_former.loc[nonempty].map(normalize_name).isin(
                    relevant_names
                )
        if not retain.any():
            continue

        kept = chunk.loc[retain].copy()
        rows_retained += len(kept)
        kept["__postcode"] = kept["RegAddress.PostCode"].map(normalize_postcode)
        kept["__incorporation"] = _parse_dates(kept["IncorporationDate"])
        previous_numbers = sorted(
            {
                int(match.group(1))
                for column in kept.columns
                if (match := re.fullmatch(r"PreviousName_(\d+)\.CompanyName", column))
            }
        )
        for number in previous_numbers:
            date_column = f"PreviousName_{number}.CONDATE"
            if date_column in kept:
                kept[f"__former_date_{number}"] = _parse_dates(kept[date_column])

        arrays = {column: kept[column].to_numpy() for column in kept.columns}
        for position in range(len(kept)):
            company_number = str(arrays["CompanyNumber"][position]).strip().upper()
            if not company_number:
                continue
            raw_incorporation = arrays["__incorporation"][position]
            incorporation = None if pd.isna(raw_incorporation) else pd.Timestamp(raw_incorporation)
            former: list[tuple[str, pd.Timestamp | None]] = []
            for number in previous_numbers:
                name_column = f"PreviousName_{number}.CompanyName"
                raw_name = str(arrays[name_column][position]).strip()
                if not raw_name:
                    continue
                parsed_column = f"__former_date_{number}"
                raw_date = arrays[parsed_column][position] if parsed_column in arrays else pd.NaT
                change_date = None if pd.isna(raw_date) else pd.Timestamp(raw_date)
                former.append((raw_name, change_date))
            names, undated, invalid = _name_periods(
                str(arrays["CompanyName"][position]).strip(), incorporation, former
            )
            undated_former_names += undated
            invalid_former_dates += invalid
            current_snapshot = {
                column: str(arrays[column][position]).strip()
                for column in CH_CURRENT_SNAPSHOT_COLUMNS
                if column in arrays
            }
            record = CompanyRecord(
                company_number=company_number,
                company_name=str(arrays["CompanyName"][position]).strip(),
                postcode=str(arrays["__postcode"][position]),
                incorporation_date=incorporation,
                names=names,
                all_normalized_names=frozenset(
                    normalized
                    for normalized in (
                        normalize_name(arrays["CompanyName"][position]),
                        *(normalize_name(raw_name) for raw_name, _ in former),
                    )
                    if normalized
                ),
                name_history_complete=undated == 0 and invalid == 0,
                current_snapshot=current_snapshot,
            )
            if company_number in companies:
                duplicate_company_rows += 1
                if companies[company_number] != record:
                    raise ValueError(
                        f"Companies House company number {company_number!r} has conflicting rows"
                    )
                continue
            companies[company_number] = record

    if first_chunk:
        raise ValueError("Companies House file contains no readable rows")

    exact_name_index: dict[str, list[str]] = {}
    for company_number, record in companies.items():
        for normalized in record.all_normalized_names:
            exact_name_index.setdefault(normalized, []).append(company_number)

    by_exact_name = {
        key: tuple(sorted(set(values))) for key, values in exact_name_index.items()
    }
    stats = {
        "ch_rows_read": int(rows_read),
        "ch_rows_retained": int(rows_retained),
        "companies_retained": int(len(companies)),
        "duplicate_company_rows": int(duplicate_company_rows),
        "undated_former_names": int(undated_former_names),
        "invalid_former_dates": int(invalid_former_dates),
        "relevant_names": int(len(relevant_names)),
        "analysis_fingerprint": analysis_digest.hexdigest(),
    }
    return CHIndex(companies, by_exact_name, stats)


# Match each RT row

@dataclass(frozen=True, slots=True)
class _Candidate:
    record: CompanyRecord
    source_field: str
    matched_period: NamePeriod


def _unique_exact_candidate(
    index: CHIndex,
    source_names: Sequence[tuple[str, str, str]],
    judgment_date: pd.Timestamp,
) -> tuple[_Candidate | None, int, int, int, int]:
    exact: dict[str, _Candidate] = {}
    rejected_post_incorporation: set[str] = set()
    missing_incorporation: set[str] = set()
    incomplete_name_history: set[str] = set()
    for source_field, _raw_source, normalized_source in source_names:
        for company_number in index.by_exact_name.get(normalized_source, ()):
            record = index.companies[company_number]
            if record.incorporation_date is None:
                missing_incorporation.add(company_number)
                continue
            if not record.name_history_complete:
                incomplete_name_history.add(company_number)
                continue
            if (
                judgment_date < record.incorporation_date
            ):
                rejected_post_incorporation.add(company_number)
                continue
            for period in record.valid_names(judgment_date):
                if period.normalized_name == normalized_source:
                    candidate = _Candidate(record, source_field, period)
                    existing = exact.get(company_number)
                    if existing is None or (source_field, period.kind) < (
                        existing.source_field,
                        existing.matched_period.kind,
                    ):
                        exact[company_number] = candidate
                    break
    unresolved = missing_incorporation | incomplete_name_history
    candidate_count = len(exact) + len(unresolved)
    if len(exact) != 1 or unresolved:
        return (
            None,
            candidate_count,
            len(rejected_post_incorporation),
            len(missing_incorporation),
            len(incomplete_name_history),
        )
    return (
        next(iter(exact.values())),
        1,
        len(rejected_post_incorporation),
        0,
        0,
    )


def match_judgments(
    judgments: pd.DataFrame,
    index: CHIndex,
) -> pd.DataFrame:
    """Return one deterministic match decision per RT judgment."""

    required = {
        "ID",
        "JudgmentDate",
        "Defendant Company Name",
        "Defendant_Postcode",
    }
    missing = required - set(judgments.columns)
    if missing:
        raise ValueError(f"judgment data missing matching column(s): {sorted(missing)}")
    if judgments["ID"].astype(str).duplicated().any():
        raise ValueError("judgment ID values must be unique before matching")

    working_columns = [
        "ID",
        "JudgmentDate",
        "Defendant Company Name",
        "Defendant_Postcode",
    ]
    if "Defendant Trading Name" in judgments:
        working_columns.append("Defendant Trading Name")
    working = judgments.loc[:, working_columns].copy()
    if "Defendant Trading Name" not in working:
        working["Defendant Trading Name"] = ""
    working["_company_name"] = working["Defendant Company Name"].fillna("").astype(str)
    working["_trading_name"] = working["Defendant Trading Name"].fillna("").astype(str)
    working["_postcode"] = working["Defendant_Postcode"].map(normalize_postcode)
    dates = pd.to_datetime(
        working["JudgmentDate"], format="mixed", dayfirst=True, errors="coerce"
    ).dt.normalize()
    if dates.isna().any():
        raise ValueError(f"JudgmentDate has {int(dates.isna().sum())} unparseable row(s)")
    working["_judgment_date"] = dates

    output: list[dict[str, object]] = []
    columns = list(working.columns)
    for values in working.itertuples(index=False, name=None):
        row_map = dict(zip(columns, values))
        source_names = (
            tuple(
                (field, raw, normalized)
                for field, raw in (
                    ("company_name", str(row_map["_company_name"]).strip()),
                    ("trading_name", str(row_map["_trading_name"]).strip()),
                )
                if (normalized := normalize_name(raw))
            )
        )
        judgment_date = pd.Timestamp(row_map["_judgment_date"])
        postcode = str(row_map["_postcode"])
        (
            exact,
            exact_count,
            rejected_post_incorporation,
            missing_incorporation,
            incomplete_name_history,
        ) = _unique_exact_candidate(index, source_names, judgment_date)
        tier = "unmatched"
        selected: _Candidate | None = None
        reason: str
        if exact is not None:
            tier = "exact_unique"
            selected = exact
            if not postcode:
                reason = "unique_exact_name_postcode_missing"
            elif exact.record.postcode == postcode:
                reason = "unique_exact_name_postcode_agrees"
            else:
                reason = "unique_exact_name_postcode_differs"
        elif not source_names:
            reason = "missing_name"
        elif exact_count > 1:
            reason = "exact_name_not_uniquely_verifiable"
        elif rejected_post_incorporation:
            reason = "exact_name_post_incorporation"
        elif missing_incorporation:
            reason = "exact_name_missing_incorporation_date"
        elif incomplete_name_history:
            reason = "exact_name_incomplete_name_history"
        else:
            reason = "no_date_valid_unique_exact_name"

        base = {
            "ID": str(row_map["ID"]),
            "tier": tier,
            "reason": reason,
            "matched_company_number": selected.record.company_number if selected else "",
            "matched_company_name": selected.record.company_name if selected else "",
            "matched_company_postcode": selected.record.postcode if selected else "",
            "matched_name": selected.matched_period.raw_name if selected else "",
            "matched_name_kind": selected.matched_period.kind if selected else "",
            "matched_on": selected.source_field if selected else "",
            "postcode_agrees": bool(selected and postcode and selected.record.postcode == postcode),
            "exact_name_candidate_count": int(exact_count),
            "rejected_post_incorporation": int(rejected_post_incorporation),
            "incorporation_date_missing": bool(missing_incorporation),
            "name_history_incomplete": bool(incomplete_name_history),
            "source_company_name": row_map["Defendant Company Name"],
            "source_trading_name": row_map["Defendant Trading Name"],
            "source_postcode": row_map["Defendant_Postcode"],
        }
        if selected is not None:
            base.update(selected.record.current_snapshot)
            base["IncorporationDate"] = selected.record.incorporation_date
        else:
            base["IncorporationDate"] = pd.NaT
        output.append(base)

    return pd.DataFrame(output)


# Count matches and unmatched reasons

def _count_table(series: pd.Series, column: str) -> pd.DataFrame:
    counts = series.fillna("missing").astype(str).value_counts(dropna=False, sort=False)
    table = counts.rename_axis(column).reset_index(name="rows")
    table["share"] = table["rows"] / max(int(table["rows"].sum()), 1)
    return table.sort_values(column, kind="stable").reset_index(drop=True)


def match_diagnostics(
    judgments: pd.DataFrame,
    matches: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Return identifier-free aggregate coverage and failure diagnostics."""

    if "ID" not in judgments or "ID" not in matches:
        raise ValueError("judgments and matches must both contain ID")
    joined = judgments[[column for column in (
        "ID", "DefendantType", "Jurisdiction", "JudgmentDate"
    ) if column in judgments]].merge(matches, on="ID", how="left", validate="one_to_one")
    if joined["tier"].isna().any():
        raise ValueError("matches do not cover every judgment ID")

    tier_counts = _count_table(joined["tier"], "tier")
    tier_counts = (
        tier_counts.set_index("tier")
        .reindex(MATCH_TIERS, fill_value=0)
        .rename_axis("tier")
        .reset_index()
    )
    tier_order = {tier: position for position, tier in enumerate(MATCH_TIERS)}
    tier_counts["__order"] = tier_counts["tier"].map(tier_order).fillna(len(tier_order))
    tier_counts = tier_counts.sort_values("__order").drop(columns="__order").reset_index(drop=True)
    unmatched_reasons = _count_table(
        joined.loc[joined["tier"].eq("unmatched"), "reason"], "reason"
    )
    method = joined.loc[
        ~joined["tier"].eq("unmatched"),
        ["tier", "matched_on", "matched_name_kind"],
    ].copy()
    if method.empty:
        method_counts = pd.DataFrame(columns=["tier", "matched_on", "matched_name_kind", "rows"])
    else:
        method_counts = (
            method.groupby(["tier", "matched_on", "matched_name_kind"], dropna=False)
            .size()
            .rename("rows")
            .reset_index()
            .sort_values(["tier", "matched_on", "matched_name_kind"])
            .reset_index(drop=True)
        )
    if "DefendantType" in joined:
        by_defendant_type = (
            joined.groupby(["DefendantType", "tier"], dropna=False)
            .size()
            .rename("rows")
            .reset_index()
        )
    else:
        by_defendant_type = pd.DataFrame(columns=["DefendantType", "tier", "rows"])
    if "JudgmentDate" in joined:
        dates = pd.to_datetime(
            joined["JudgmentDate"], format="mixed", dayfirst=True, errors="coerce"
        )
        vintage = dates.dt.year.astype("Int64").astype("string").fillna("missing")
        by_judgment_vintage = (
            joined.assign(judgment_year=vintage)
            .groupby(["judgment_year", "tier"], dropna=False)
            .size()
            .rename("rows")
            .reset_index()
        )
    else:
        by_judgment_vintage = pd.DataFrame(columns=["judgment_year", "tier", "rows"])
    guard_counts = pd.DataFrame(
        [
            {
                "guard": "candidate_companies_rejected_post_incorporation",
                "rows": int(
                    pd.to_numeric(
                        joined.get(
                            "rejected_post_incorporation",
                            pd.Series(0, index=joined.index),
                        ),
                        errors="coerce",
                    ).fillna(0).sum()
                ),
            },
            {
                "guard": "judgments_with_post_incorporation_rejection",
                "rows": int(
                    pd.to_numeric(
                        joined.get(
                            "rejected_post_incorporation",
                            pd.Series(0, index=joined.index),
                        ),
                        errors="coerce",
                    ).fillna(0).gt(0).sum()
                ),
            },
            {
                "guard": "accepted_matches_missing_incorporation_date",
                "rows": int(
                    (
                        joined["tier"].ne("unmatched")
                        & joined.get(
                            "incorporation_date_missing",
                            pd.Series(False, index=joined.index),
                        ).fillna(False).astype(bool)
                    ).sum()
                ),
            },
            {
                "guard": "judgments_blocked_by_incomplete_name_history",
                "rows": int(
                    joined.get(
                        "name_history_incomplete",
                        pd.Series(False, index=joined.index),
                    ).fillna(False).astype(bool).sum()
                ),
            },
        ]
    )
    return {
        "tier_counts": tier_counts,
        "unmatched_reasons": unmatched_reasons,
        "method_counts": method_counts,
        "by_defendant_type": by_defendant_type,
        "by_judgment_vintage": by_judgment_vintage,
        "guard_counts": guard_counts,
    }


# Select 1,000 matching pairs

def _stable_rank(seed: int, tier: str, identifier: str, company_number: str) -> str:
    value = f"{seed}|{tier}|{identifier}|{company_number}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _equal_probability_systematic_sample(
    frame: pd.DataFrame,
    count: int,
    *,
    seed: int,
    tier: str,
) -> pd.DataFrame:
    """Take an equal-probability sample spread across ordered match strata.

    Sorting by the composite stratum before a random-start systematic draw
    gives the requested spread across name source, vintage and method.
    Every row within the tier has the same inclusion probability, so unweighted
    precision is unbiased. Wilson is the declared approximation because the
    systematic selections are not independent.
    """

    if count <= 0:
        return frame.head(0).copy()
    if count >= len(frame):
        return frame.copy()
    ordered = frame.sort_values(["__stratum", "__rank"], kind="stable")
    interval = len(ordered) / count
    start_digest = hashlib.sha256(
        f"{seed}|{tier}|systematic-start".encode("utf-8")
    ).digest()
    start_fraction = int.from_bytes(start_digest[:8], "big") / 2**64
    positions = [int((start_fraction + offset) * interval) for offset in range(count)]
    return ordered.iloc[positions].copy()


def pair_sample(
    judgments: pd.DataFrame,
    matches: pd.DataFrame,
    settings: Settings,
    *,
    seed: int | None = None,
) -> pd.DataFrame:
    """Select 1,000 exact-name matches across name, date and postcode groups."""

    sampling_columns = [
        "ID",
        "tier",
        "reason",
        "matched_company_number",
        "matched_name_kind",
        "matched_on",
        "postcode_agrees",
    ]
    canonical_source_columns = (
        "source_company_name",
        "source_trading_name",
        "source_postcode",
    )
    missing = set((*sampling_columns, *canonical_source_columns)) - set(matches.columns)
    if missing:
        raise ValueError(f"matches missing pair-sample column(s): {sorted(missing)}")

    accepted_mask = matches["tier"].eq("exact_unique")
    accepted = matches.loc[accepted_mask, sampling_columns].copy()
    accepted["__match_position"] = accepted_mask.to_numpy().nonzero()[0]
    if accepted["ID"].astype(str).duplicated().any():
        raise ValueError("matches contain duplicate ID values")
    if len(accepted) < settings.sample_size:
        raise ValueError(
            f"only {len(accepted):,} unique exact-name matches are available; "
            f"{settings.sample_size:,} are required for the example file"
        )
    actual_seed = settings.locked_seed if seed is None else int(seed)
    dates = judgments[["ID", "JudgmentDate"]].copy()
    dates["JudgmentDate"] = pd.to_datetime(
        dates["JudgmentDate"], format="mixed", dayfirst=True, errors="coerce"
    )
    accepted = accepted.merge(dates, on="ID", how="left", validate="one_to_one")
    accepted["__vintage"] = accepted["JudgmentDate"].dt.year.astype("Int64").astype("string")
    accepted["__stratum"] = (
        accepted[
            ["matched_on", "matched_name_kind", "reason", "postcode_agrees", "__vintage"]
        ]
        .astype("string")
        .fillna("missing")
        .agg("|".join, axis=1)
    )
    accepted["__rank"] = [
        _stable_rank(actual_seed, str(tier), str(identifier), str(company_number))
        for tier, identifier, company_number in zip(
            accepted["tier"], accepted["ID"], accepted["matched_company_number"]
        )
    ]
    selected = _equal_probability_systematic_sample(
        accepted,
        settings.sample_size,
        seed=actual_seed,
        tier="exact_unique",
    )
    selected_positions = selected["__match_position"].astype(int).to_numpy()
    sampled = matches.iloc[selected_positions].copy().reset_index(drop=True)
    sampled["JudgmentDate"] = selected["JudgmentDate"].to_numpy()

    output_source_columns = ["source_company_name"]
    if "Defendant Trading Name" in judgments:
        output_source_columns.append("source_trading_name")
    output_source_columns.append("source_postcode")
    source = sampled.loc[:, output_source_columns].copy()
    sampled = sampled.drop(
        columns=[
            *canonical_source_columns,
            "__rank",
            "__vintage",
            "__stratum",
        ],
        errors="ignore",
    )
    for column in output_source_columns:
        sampled[column] = source[column].to_numpy()
    sampled = sampled.sort_values("ID", kind="stable").reset_index(drop=True)
    sampled["match_method"] = sampled["reason"]
    columns = [
        "ID",
        "source_company_name",
        "source_trading_name",
        "source_postcode",
        "matched_company_name",
        "matched_name",
        "matched_name_kind",
        "matched_company_number",
        "matched_company_postcode",
        "IncorporationDate",
        "tier",
        "match_method",
        "matched_on",
        "postcode_agrees",
        "JudgmentDate",
    ]
    return sampled.loc[:, [column for column in columns if column in sampled]].copy()


# Make and analyse the independent linkage-validation samples

def _assert_no_outcome_columns(frame: pd.DataFrame) -> None:
    forbidden = {
        str(column)
        for column in frame.columns
        if _canon_header(column) in _FORBIDDEN_VALIDATION_COLUMNS
    }
    if forbidden:
        raise ValueError(
            "linkage-validation files must not contain outcome/status column(s): "
            f"{sorted(forbidden)}"
        )


def _sample_allocations(populations: pd.Series, target: int) -> pd.Series:
    """Allocate a fixed target to every stratum, then approximately proportionally."""

    populations = populations.astype(int).sort_index()
    if populations.empty:
        return populations.copy()
    total = int(populations.sum())
    target = min(int(target), total)
    if target < len(populations):
        raise ValueError(
            "unmatched sample size is too small to give every reason/vintage "
            f"stratum a positive inclusion probability: need at least {len(populations):,}"
        )

    allocation = pd.Series(1, index=populations.index, dtype="int64")
    remaining = target - len(allocation)
    capacity = populations - allocation
    while remaining > 0 and int(capacity.sum()) > 0:
        ideal = capacity.astype(float) * remaining / int(capacity.sum())
        addition = ideal.map(math.floor).astype(int).clip(upper=capacity)
        if int(addition.sum()) == 0:
            order = sorted(
                capacity.loc[capacity.gt(0)].index,
                key=lambda key: (-(ideal.loc[key] % 1), str(key)),
            )
            for key in order[:remaining]:
                addition.loc[key] += 1
        allocation += addition
        capacity -= addition
        remaining = target - int(allocation.sum())
    if int(allocation.sum()) != target:
        raise RuntimeError("could not allocate the requested unmatched validation sample")
    return allocation


def unmatched_pair_sample(
    judgments: pd.DataFrame,
    matches: pd.DataFrame,
    sample_size: int,
    *,
    seed: int,
) -> pd.DataFrame:
    """Draw a reproducible stratified probability sample of unmatched judgments.

    Strata cross the algorithmic unmatched reason with judgment year. Within each
    stratum, a seeded cryptographic rank acts as the random permutation. The
    design samples every non-empty stratum and records its exact first-order
    inclusion probability and inverse-probability weight.
    """

    if int(sample_size) <= 0:
        raise ValueError("unmatched sample size must be positive")
    required_matches = {
        "ID",
        "tier",
        "reason",
        "source_company_name",
        "source_trading_name",
        "source_postcode",
    }
    missing = required_matches - set(matches.columns)
    if missing:
        raise ValueError(
            f"matches missing unmatched-sample column(s): {sorted(missing)}"
        )
    if "ID" not in judgments or "JudgmentDate" not in judgments:
        raise ValueError("judgments must contain ID and JudgmentDate")
    if matches["ID"].astype(str).duplicated().any():
        raise ValueError("matches contain duplicate ID values")
    if judgments["ID"].astype(str).duplicated().any():
        raise ValueError("judgments contain duplicate ID values")

    allowed_match_columns = [
        "ID",
        "tier",
        "reason",
        "source_company_name",
        "source_trading_name",
        "source_postcode",
        "exact_name_candidate_count",
        "rejected_post_incorporation",
        "incorporation_date_missing",
        "name_history_incomplete",
    ]
    unmatched = matches.loc[
        matches["tier"].eq("unmatched"),
        [column for column in allowed_match_columns if column in matches],
    ].copy()
    if unmatched.empty:
        raise ValueError("no unmatched judgments are available for validation")

    judgment_columns = ["ID", "JudgmentDate"]
    if "Defendant Address" in judgments:
        judgment_columns.append("Defendant Address")
    dates = judgments.loc[:, judgment_columns].copy()
    dates["ID"] = dates["ID"].astype(str)
    dates["JudgmentDate"] = pd.to_datetime(
        dates["JudgmentDate"], format="mixed", dayfirst=True, errors="coerce"
    ).dt.normalize()
    unmatched["ID"] = unmatched["ID"].astype(str)
    unmatched = unmatched.merge(dates, on="ID", how="left", validate="one_to_one")
    if unmatched["JudgmentDate"].isna().any():
        raise ValueError(
            "every unmatched judgment must have a parseable JudgmentDate for vintage sampling"
        )
    if "Defendant Address" in unmatched:
        unmatched = unmatched.rename(columns={"Defendant Address": "source_address"})

    unmatched["__vintage"] = unmatched["JudgmentDate"].dt.year.astype(str)
    unmatched["sampling_stratum"] = (
        "reason="
        + unmatched["reason"].fillna("missing").astype(str)
        + "|judgment_year="
        + unmatched["__vintage"]
    )
    populations = unmatched["sampling_stratum"].value_counts(sort=False)
    allocation = _sample_allocations(populations, int(sample_size))

    selected_parts: list[pd.DataFrame] = []
    for stratum in sorted(populations.index):
        stratum_frame = unmatched.loc[
            unmatched["sampling_stratum"].eq(stratum)
        ].copy()
        stratum_frame["__rank"] = [
            _stable_rank(int(seed), f"unmatched|{stratum}", identifier, "")
            for identifier in stratum_frame["ID"]
        ]
        stratum_frame = stratum_frame.sort_values("__rank", kind="stable")
        sample_n = int(allocation.loc[stratum])
        population_n = int(populations.loc[stratum])
        selected = stratum_frame.head(sample_n).copy()
        selected["validation_arm"] = "unmatched"
        selected["sampling_design"] = "reason_vintage_stratified_hash_srs"
        selected["stratum_population_n"] = population_n
        selected["stratum_sample_n"] = sample_n
        selected["inclusion_probability"] = sample_n / population_n
        selected["sampling_weight"] = population_n / sample_n
        selected_parts.append(selected)

    sampled = pd.concat(selected_parts, ignore_index=True)
    sampled["match_method"] = sampled["reason"]
    sampled = sampled.drop(columns=["__vintage", "__rank"], errors="ignore")
    for column in LINKAGE_REVIEW_COLUMNS:
        sampled[column] = ""
    sampled = sampled.sort_values("ID", kind="stable").reset_index(drop=True)
    _assert_no_outcome_columns(sampled)
    return sampled


def accepted_validation_sample(
    judgments: pd.DataFrame,
    matches: pd.DataFrame,
    settings: Settings,
    *,
    seed: int | None = None,
) -> pd.DataFrame:
    """Add auditable sampling metadata and blank review fields to accepted pairs."""

    actual_seed = settings.locked_seed if seed is None else int(seed)
    accepted = pair_sample(judgments, matches, settings, seed=actual_seed).copy()
    accepted_population_n = int(matches["tier"].eq("exact_unique").sum())
    accepted_sample_n = len(accepted)
    accepted["validation_arm"] = "accepted"
    accepted["sampling_design"] = "equal_probability_systematic"
    accepted["sampling_stratum"] = "tier=exact_unique"
    accepted["stratum_population_n"] = accepted_population_n
    accepted["stratum_sample_n"] = accepted_sample_n
    accepted["inclusion_probability"] = accepted_sample_n / accepted_population_n
    accepted["sampling_weight"] = accepted_population_n / accepted_sample_n
    if "Defendant Address" in judgments:
        source_address = judgments[["ID", "Defendant Address"]].copy()
        source_address["ID"] = source_address["ID"].astype(str)
        accepted["ID"] = accepted["ID"].astype(str)
        accepted = accepted.merge(
            source_address.rename(columns={"Defendant Address": "source_address"}),
            on="ID",
            how="left",
            validate="one_to_one",
        )
    for column in LINKAGE_REVIEW_COLUMNS:
        accepted[column] = ""
    _assert_no_outcome_columns(accepted)
    return accepted


def unmatched_validation_sample(
    judgments: pd.DataFrame,
    matches: pd.DataFrame,
    sample_size: int,
    *,
    seed: int,
) -> pd.DataFrame:
    """Public, plainly named entry point for the unmatched validation arm."""

    return unmatched_pair_sample(judgments, matches, sample_size, seed=seed)


def linkage_validation_sample(
    judgments: pd.DataFrame,
    matches: pd.DataFrame,
    settings: Settings,
    *,
    unmatched_sample_size: int,
    seed: int | None = None,
) -> pd.DataFrame:
    """Combine the retained accepted-pair sample with an unmatched audit sample.

    Only identity, matching and sampling fields leave this function. In
    particular, no Registry Trust outcome/status or Companies House current
    status field is copied to the adjudication file.
    """

    actual_seed = settings.locked_seed if seed is None else int(seed)
    accepted = accepted_validation_sample(
        judgments, matches, settings, seed=actual_seed
    )

    unmatched = unmatched_validation_sample(
        judgments,
        matches,
        unmatched_sample_size,
        seed=actual_seed,
    )
    combined = pd.concat([accepted, unmatched], ignore_index=True, sort=False)
    text_columns = [
        "source_company_name",
        "source_trading_name",
        "source_postcode",
        "source_address",
        "matched_company_name",
        "matched_name",
        "matched_name_kind",
        "matched_company_number",
        "matched_company_postcode",
        "matched_on",
        "reason",
        "match_method",
        *LINKAGE_REVIEW_COLUMNS,
    ]
    for column in text_columns:
        if column not in combined:
            combined[column] = ""
        else:
            combined[column] = combined[column].fillna("")
    combined = combined.sort_values(
        ["validation_arm", "ID"], kind="stable"
    ).reset_index(drop=True)
    _assert_no_outcome_columns(combined)
    return combined


def _clean_label(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().lower()


def _clean_company_number(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", "", str(value).strip().upper())


def _validate_sampling_metadata(frame: pd.DataFrame) -> None:
    required = {
        "ID",
        "validation_arm",
        "sampling_stratum",
        "stratum_population_n",
        "stratum_sample_n",
        "inclusion_probability",
        "sampling_weight",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"linkage adjudications missing sampling column(s): {sorted(missing)}"
        )
    if frame.empty:
        raise ValueError("linkage adjudications are empty")
    if frame["ID"].astype(str).duplicated().any():
        raise ValueError("linkage adjudications contain duplicate ID values")
    invalid_arms = set(frame["validation_arm"].map(_clean_label)) - set(_ARM_LABELS)
    if invalid_arms:
        raise ValueError(f"invalid linkage-validation arm(s): {sorted(invalid_arms)}")

    for (arm, stratum), group in frame.groupby(
        ["validation_arm", "sampling_stratum"], dropna=False, sort=False
    ):
        if not str(stratum).strip():
            raise ValueError("sampling_stratum must not be blank")
        metadata: dict[str, float] = {}
        for column in (
            "stratum_population_n",
            "stratum_sample_n",
            "inclusion_probability",
            "sampling_weight",
        ):
            values = pd.to_numeric(group[column], errors="coerce")
            if values.isna().any() or values.nunique(dropna=False) != 1:
                raise ValueError(
                    f"sampling metadata {column!r} is invalid or inconsistent in "
                    f"{arm!r}/{stratum!r}"
                )
            metadata[column] = float(values.iloc[0])
        population_n = metadata["stratum_population_n"]
        sample_n = metadata["stratum_sample_n"]
        if (
            not population_n.is_integer()
            or not sample_n.is_integer()
            or population_n <= 0
            or sample_n <= 0
            or sample_n > population_n
            or int(sample_n) != len(group)
        ):
            raise ValueError(f"invalid sample/population sizes in {arm!r}/{stratum!r}")
        expected_probability = sample_n / population_n
        if not math.isclose(
            metadata["inclusion_probability"], expected_probability, rel_tol=1e-9
        ):
            raise ValueError(
                f"inclusion_probability is inconsistent in {arm!r}/{stratum!r}"
            )
        if not math.isclose(
            metadata["sampling_weight"], 1 / expected_probability, rel_tol=1e-9
        ):
            raise ValueError(f"sampling_weight is inconsistent in {arm!r}/{stratum!r}")


def validate_linkage_adjudications(adjudications: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalise a completed, independently double-reviewed sample."""

    _assert_no_outcome_columns(adjudications)
    _validate_sampling_metadata(adjudications)
    missing = set(LINKAGE_REVIEW_COLUMNS) - set(adjudications.columns)
    if missing:
        raise ValueError(
            f"linkage adjudications missing review column(s): {sorted(missing)}"
        )
    validated = adjudications.copy()
    validated["validation_arm"] = validated["validation_arm"].map(_clean_label)
    for column in (
        "reviewer_1_label",
        "reviewer_2_label",
        "adjudicated_label",
    ):
        validated[column] = validated[column].map(_clean_label)
    for column in (
        "matched_company_number",
        "reviewer_1_company_number",
        "reviewer_2_company_number",
        "adjudicated_company_number",
    ):
        if column not in validated:
            validated[column] = ""
        validated[column] = validated[column].map(_clean_company_number)

    for row_number, row in validated.iterrows():
        identifier = str(row["ID"])
        arm = str(row["validation_arm"])
        allowed = _ARM_LABELS[arm]
        labels = {
            role: str(row[f"{role}_label"])
            for role in ("reviewer_1", "reviewer_2", "adjudicated")
        }
        for role, label in labels.items():
            if not label:
                raise ValueError(
                    f"incomplete linkage adjudication for ID {identifier!r}: "
                    f"{role}_label is blank"
                )
            if label not in allowed:
                raise ValueError(
                    f"invalid {role}_label {label!r} for {arm} ID {identifier!r}; "
                    f"allowed labels are {sorted(allowed)}"
                )
        if labels["reviewer_1"] == labels["reviewer_2"]:
            if labels["adjudicated"] != labels["reviewer_1"]:
                raise ValueError(
                    f"adjudicated_label contradicts agreeing reviewers for ID {identifier!r}"
                )

        for role in ("reviewer_1", "reviewer_2", "adjudicated"):
            label = labels[role]
            company_number = str(row[f"{role}_company_number"])
            if arm == "unmatched":
                if label == "missed_match" and not company_number:
                    raise ValueError(
                        f"{role}_company_number is required for missed_match ID {identifier!r}"
                    )
                if label == "true_unmatched" and company_number:
                    raise ValueError(
                        f"{role}_company_number must be blank for true_unmatched ID {identifier!r}"
                    )
            elif label == "correct_match":
                proposed = str(row["matched_company_number"])
                if not proposed:
                    raise ValueError(
                        f"accepted ID {identifier!r} has no proposed matched company number"
                    )
                if company_number and company_number != proposed:
                    raise ValueError(
                        f"{role}_company_number does not equal the proposed match "
                        f"for ID {identifier!r}"
                    )

        if arm == "unmatched" and labels["reviewer_1"] == labels["reviewer_2"] == "missed_match":
            first_number = str(row["reviewer_1_company_number"])
            second_number = str(row["reviewer_2_company_number"])
            final_number = str(row["adjudicated_company_number"])
            if first_number == second_number and final_number != first_number:
                raise ValueError(
                    f"adjudicated company number contradicts agreeing reviewers "
                    f"for ID {identifier!r}"
                )
    return validated


def _wilson_interval(successes: int, sample_n: int, confidence_level: float) -> tuple[float, float]:
    if sample_n <= 0:
        raise ValueError("Wilson interval requires a positive sample size")
    alpha = 1 - confidence_level
    z = float(norm.ppf(1 - alpha / 2))
    proportion = successes / sample_n
    denominator = 1 + z * z / sample_n
    centre = (proportion + z * z / (2 * sample_n)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / sample_n
            + z * z / (4 * sample_n * sample_n)
        )
        / denominator
    )
    return max(0.0, centre - radius), min(1.0, centre + radius)


def _design_weighted_prevalence(
    frame: pd.DataFrame,
    *,
    positive_label: str,
    confidence_level: float,
) -> dict[str, object]:
    strata = list(frame.groupby("sampling_stratum", sort=False, dropna=False))
    if not strata:
        raise ValueError("cannot estimate a prevalence from an empty validation arm")
    alpha = 1 - confidence_level
    stratum_confidence = 1 - alpha / len(strata)
    population_total = 0
    estimated_positive_total = 0.0
    lower_total = 0.0
    upper_total = 0.0
    sample_total = 0
    for _stratum, group in strata:
        population_n = int(group["stratum_population_n"].iloc[0])
        sample_n = len(group)
        successes = int(group["adjudicated_label"].eq(positive_label).sum())
        estimate = successes / sample_n
        if sample_n == population_n:
            lower, upper = estimate, estimate
        else:
            lower, upper = _wilson_interval(
                successes, sample_n, stratum_confidence
            )
        population_total += population_n
        sample_total += sample_n
        estimated_positive_total += population_n * estimate
        lower_total += population_n * lower
        upper_total += population_n * upper
    return {
        "estimate": estimated_positive_total / population_total,
        "lower_ci": lower_total / population_total,
        "upper_ci": upper_total / population_total,
        "estimated_positive_total": estimated_positive_total,
        "population_n": population_total,
        "sample_n": sample_total,
        "ci_method": (
            "Bonferroni-stratified Wilson intervals (census strata exact); "
            "systematic-sample interval is an approximation"
        ),
    }


def _cohen_kappa(first: pd.Series, second: pd.Series) -> float:
    first_binary = first.astype(int)
    second_binary = second.astype(int)
    observed = float(first_binary.eq(second_binary).mean())
    first_positive = float(first_binary.mean())
    second_positive = float(second_binary.mean())
    expected = (
        first_positive * second_positive
        + (1 - first_positive) * (1 - second_positive)
    )
    if math.isclose(expected, 1.0):
        return float("nan")
    return (observed - expected) / (1 - expected)


def _reviewer_agreement(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups: list[tuple[str, pd.DataFrame]] = [
        (str(arm), group) for arm, group in frame.groupby("validation_arm", sort=True)
    ]
    groups.append(("overall", frame))
    for arm, group in groups:
        first_positive = pd.Series(
            [
                label == _POSITIVE_LABEL[row_arm]
                for label, row_arm in zip(
                    group["reviewer_1_label"], group["validation_arm"]
                )
            ],
            index=group.index,
        )
        second_positive = pd.Series(
            [
                label == _POSITIVE_LABEL[row_arm]
                for label, row_arm in zip(
                    group["reviewer_2_label"], group["validation_arm"]
                )
            ],
            index=group.index,
        )
        decision_agreement: list[bool] = []
        for _, row in group.iterrows():
            labels_agree = row["reviewer_1_label"] == row["reviewer_2_label"]
            if row["validation_arm"] == "unmatched" and row["reviewer_1_label"] == "missed_match":
                labels_agree = labels_agree and (
                    row["reviewer_1_company_number"]
                    == row["reviewer_2_company_number"]
                )
            decision_agreement.append(bool(labels_agree))
        rows.append(
            {
                "validation_arm": arm,
                "reviewed_n": int(len(group)),
                "label_agreement": float(first_positive.eq(second_positive).mean()),
                "decision_agreement": float(pd.Series(decision_agreement).mean()),
                "cohen_kappa": _cohen_kappa(first_positive, second_positive),
            }
        )
    return pd.DataFrame(rows)


def summarize_linkage_validation(
    adjudications: pd.DataFrame,
    *,
    confidence_level: float = 0.95,
    recall_denominator_supported: bool = False,
) -> dict[str, pd.DataFrame]:
    """Calculate linkage accuracy with design weights and explicit recall gating.

    Set ``recall_denominator_supported`` only when both audit arms used the same
    target population and reviewers searched a company universe capable of
    finding every target link (including dissolved companies when those are in
    scope). Otherwise the function deliberately withholds recall.
    """

    if not 0 < float(confidence_level) < 1:
        raise ValueError("confidence_level must lie strictly between zero and one")
    validated = validate_linkage_adjudications(adjudications)
    arms = set(validated["validation_arm"])
    required_arms = {"accepted", "unmatched"}
    if not required_arms.issubset(arms):
        raise ValueError(
            "completed accepted and unmatched adjudication arms are both required"
        )

    accepted = validated.loc[validated["validation_arm"].eq("accepted")]
    unmatched = validated.loc[validated["validation_arm"].eq("unmatched")]
    precision = _design_weighted_prevalence(
        accepted,
        positive_label="correct_match",
        confidence_level=float(confidence_level),
    )
    missed = _design_weighted_prevalence(
        unmatched,
        positive_label="missed_match",
        confidence_level=float(confidence_level),
    )
    estimate_rows: list[dict[str, object]] = [
        {
            "measure": "accepted_match_precision",
            **precision,
            "status": "estimated",
            "reason": "",
        },
        {
            "measure": "unmatched_missed_link_prevalence",
            **missed,
            "status": "estimated",
            "reason": "",
        },
    ]

    recall_row: dict[str, object] = {
        "measure": "linkage_recall",
        "estimate": float("nan"),
        "lower_ci": float("nan"),
        "upper_ci": float("nan"),
        "estimated_positive_total": float("nan"),
        "population_n": int(precision["population_n"]) + int(missed["population_n"]),
        "sample_n": int(precision["sample_n"]) + int(missed["sample_n"]),
        "ci_method": "",
        "status": "not_estimated",
        "reason": (
            "recall denominator not certified; confirm a common target population "
            "and an exhaustive adjudication search universe"
        ),
    }
    if recall_denominator_supported:
        accepted_true = float(precision["estimated_positive_total"])
        missed_true = float(missed["estimated_positive_total"])
        denominator = accepted_true + missed_true
        if denominator <= 0:
            recall_row["reason"] = "estimated number of true links is zero"
        else:
            component_confidence = 1 - (1 - float(confidence_level)) / 2
            precision_joint = _design_weighted_prevalence(
                accepted,
                positive_label="correct_match",
                confidence_level=component_confidence,
            )
            missed_joint = _design_weighted_prevalence(
                unmatched,
                positive_label="missed_match",
                confidence_level=component_confidence,
            )
            accepted_population = float(precision["population_n"])
            unmatched_population = float(missed["population_n"])
            true_positive_low = accepted_population * float(precision_joint["lower_ci"])
            true_positive_high = accepted_population * float(precision_joint["upper_ci"])
            false_negative_low = unmatched_population * float(missed_joint["lower_ci"])
            false_negative_high = unmatched_population * float(missed_joint["upper_ci"])
            recall_row.update(
                {
                    "estimate": accepted_true / denominator,
                    "lower_ci": true_positive_low
                    / (true_positive_low + false_negative_high),
                    "upper_ci": true_positive_high
                    / (true_positive_high + false_negative_low),
                    "estimated_positive_total": denominator,
                    "ci_method": (
                        "monotone bounds from Bonferroni-adjusted component "
                        "prevalence intervals"
                    ),
                    "status": "estimated",
                    "reason": "",
                }
            )
    estimate_rows.append(recall_row)

    stratum_rows: list[dict[str, object]] = []
    for (arm, stratum), group in validated.groupby(
        ["validation_arm", "sampling_stratum"], sort=True
    ):
        positive = _POSITIVE_LABEL[str(arm)]
        population_n = int(group["stratum_population_n"].iloc[0])
        sample_n = len(group)
        positives = int(group["adjudicated_label"].eq(positive).sum())
        stratum_rows.append(
            {
                "validation_arm": arm,
                "sampling_stratum": stratum,
                "stratum_population_n": population_n,
                "stratum_sample_n": sample_n,
                "adjudicated_positive_n": positives,
                "weighted_positive_total": population_n * positives / sample_n,
                "weighted_prevalence": positives / sample_n,
            }
        )
    return {
        "estimates": pd.DataFrame(estimate_rows),
        "reviewer_agreement": _reviewer_agreement(validated),
        "stratum_estimates": pd.DataFrame(stratum_rows),
    }


__all__ = [
    "ACCEPTED_LINKAGE_VALIDATION_FILENAME",
    "CH_CURRENT_SNAPSHOT_COLUMNS",
    "CH_REQUIRED_COLUMNS",
    "CHIndex",
    "CompanyRecord",
    "MATCH_TIERS",
    "NamePeriod",
    "UNMATCHED_LINKAGE_VALIDATION_FILENAME",
    "accepted_validation_sample",
    "build_relevant_ch_index",
    "linkage_validation_sample",
    "match_diagnostics",
    "match_judgments",
    "normalize_name",
    "normalize_postcode",
    "pair_sample",
    "summarize_linkage_validation",
    "unmatched_pair_sample",
    "unmatched_validation_sample",
    "validate_linkage_adjudications",
]
