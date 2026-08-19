"""Offline, auditable matching of RT defendants to Companies House companies.

The bulk file is streamed once and only postcode or exact-name candidates are
retained.  Historical names are usable only for dates on which they were valid;
an unknown incorporation date can never produce an automatic match.  This
module performs no network, shell, cache, or output operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import hashlib
import re
import unicodedata

import pandas as pd
from rapidfuzz import fuzz

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

MATCH_TIERS: tuple[str, ...] = ("auto", "review", "fallback_review", "unmatched")

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


def normalize_name(value: object) -> str:
    """Return a conservative comparison form, stripping only terminal suffixes."""

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
    current_snapshot: Mapping[str, str]

    def valid_names(self, when: pd.Timestamp) -> tuple[NamePeriod, ...]:
        if self.incorporation_date is not None and when < self.incorporation_date:
            return ()
        return tuple(name for name in self.names if name.valid_on(when))


@dataclass(frozen=True, slots=True)
class CHIndex:
    companies: Mapping[str, CompanyRecord]
    by_postcode: Mapping[str, tuple[str, ...]]
    by_exact_name: Mapping[str, tuple[str, ...]]
    stats: Mapping[str, object]


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
    return pd.to_datetime(
        series.astype("string").str.strip(),
        format="mixed",
        dayfirst=True,
        errors="coerce",
    ).dt.normalize()


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


def _relevant_values(judgments: pd.DataFrame) -> tuple[set[str], set[str]]:
    required = {"Defendant Company Name", "Defendant_Postcode"}
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
    postcodes = {
        normalized
        for normalized in judgments["Defendant_Postcode"].map(normalize_postcode).unique().tolist()
        if normalized
    }
    return names, postcodes


def build_relevant_ch_index(
    judgments: pd.DataFrame,
    ch_path: str | Path,
    *,
    chunksize: int = 100_000,
) -> CHIndex:
    """Stream CH once, retaining only exact-name or exact-postcode candidates."""

    relevant_names, relevant_postcodes = _relevant_values(judgments)
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
        postcode_norm = chunk["RegAddress.PostCode"].map(normalize_postcode)
        retain = current_norm.isin(relevant_names) | postcode_norm.isin(relevant_postcodes)
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
        kept["__postcode"] = postcode_norm.loc[retain].values
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

    postcode_index: dict[str, list[str]] = {}
    exact_name_index: dict[str, list[str]] = {}
    for company_number, record in companies.items():
        if record.postcode:
            postcode_index.setdefault(record.postcode, []).append(company_number)
        for normalized in {name.normalized_name for name in record.names if name.normalized_name}:
            exact_name_index.setdefault(normalized, []).append(company_number)

    by_postcode = {
        key: tuple(sorted(set(values))) for key, values in postcode_index.items()
    }
    by_exact_name = {
        key: tuple(sorted(set(values))) for key, values in exact_name_index.items()
    }
    stats = {
        "ch_rows_read": int(rows_read),
        "ch_rows_retained": int(rows_retained),
        "companies_retained": int(len(companies)),
        "duplicate_company_rows": int(duplicate_company_rows),
        "undated_former_names_ignored": int(undated_former_names),
        "invalid_former_dates_ignored": int(invalid_former_dates),
        "relevant_postcodes": int(len(relevant_postcodes)),
        "relevant_names": int(len(relevant_names)),
        "analysis_fingerprint": analysis_digest.hexdigest(),
    }
    return CHIndex(companies, by_postcode, by_exact_name, stats)


@dataclass(frozen=True, slots=True)
class _Candidate:
    record: CompanyRecord
    score: float
    source_field: str
    source_name: str
    matched_period: NamePeriod


def _best_candidate(
    record: CompanyRecord,
    source_names: Sequence[tuple[str, str, str]],
    judgment_date: pd.Timestamp,
) -> _Candidate | None:
    valid_names = record.valid_names(judgment_date)
    if not valid_names:
        return None
    best: tuple[float, str, str, NamePeriod] | None = None
    for source_field, raw_source, normalized_source in source_names:
        for period in valid_names:
            score = float(fuzz.WRatio(normalized_source, period.normalized_name)) / 100.0
            option = (score, source_field, raw_source, period)
            if best is None or score > best[0] or (
                score == best[0]
                and (source_field, period.kind, period.normalized_name)
                < (best[1], best[3].kind, best[3].normalized_name)
            ):
                best = option
    if best is None:
        return None
    return _Candidate(record, best[0], best[1], best[2], best[3])


def _unique_exact_candidate(
    index: CHIndex,
    source_names: Sequence[tuple[str, str, str]],
    judgment_date: pd.Timestamp,
) -> tuple[_Candidate | None, int]:
    exact: dict[str, _Candidate] = {}
    for source_field, raw_source, normalized_source in source_names:
        for company_number in index.by_exact_name.get(normalized_source, ()):
            record = index.companies[company_number]
            # The postcode-free fallback is deliberately conservative: without
            # an incorporation date the pre-judgment identity cannot be proved.
            if record.incorporation_date is None or judgment_date < record.incorporation_date:
                continue
            for period in record.valid_names(judgment_date):
                if period.normalized_name == normalized_source:
                    candidate = _Candidate(record, 1.0, source_field, raw_source, period)
                    existing = exact.get(company_number)
                    if existing is None or (source_field, period.kind) < (
                        existing.source_field,
                        existing.matched_period.kind,
                    ):
                        exact[company_number] = candidate
                    break
    if len(exact) != 1:
        return None, len(exact)
    return next(iter(exact.values())), 1


def match_judgments(
    judgments: pd.DataFrame,
    index: CHIndex,
    settings: Settings,
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
        same_postcode_numbers = index.by_postcode.get(postcode, ()) if postcode else ()
        candidates: list[_Candidate] = []
        rejected_post_incorporation = 0
        for company_number in same_postcode_numbers:
            record = index.companies[company_number]
            if record.incorporation_date is not None and judgment_date < record.incorporation_date:
                rejected_post_incorporation += 1
                continue
            candidate = _best_candidate(record, source_names, judgment_date)
            if candidate is not None:
                candidates.append(candidate)
        candidates.sort(key=lambda value: (-value.score, value.record.company_number))
        top = candidates[0] if candidates else None
        runner_up = candidates[1].score if len(candidates) > 1 else 0.0
        margin = top.score - runner_up if top is not None else 0.0

        exact, exact_count = _unique_exact_candidate(index, source_names, judgment_date)
        tier = "unmatched"
        selected: _Candidate | None = None
        reason: str
        if (
            top is not None
            and top.score >= settings.auto_threshold
            and margin >= settings.auto_margin
            and top.record.incorporation_date is not None
        ):
            tier = "auto"
            selected = top
            reason = "postcode_score_and_margin"
        elif exact is not None and exact.record.postcode != postcode:
            tier = "fallback_review"
            selected = exact
            reason = "unique_date_valid_exact_name"
        elif top is not None and top.score >= settings.review_threshold:
            tier = "review"
            selected = top
            if top.record.incorporation_date is None:
                reason = "incorporation_date_missing"
            elif margin < settings.auto_margin:
                reason = "ambiguous_postcode_candidates"
            else:
                reason = "score_below_auto"
        elif not source_names:
            reason = "missing_name"
        elif not postcode:
            reason = "missing_postcode_no_unique_exact_name"
        elif not same_postcode_numbers:
            reason = "no_postcode_candidate_no_unique_exact_name"
        elif rejected_post_incorporation and not candidates:
            reason = "all_postcode_candidates_post_incorporation"
        elif exact_count > 1:
            reason = "ambiguous_exact_name_and_below_review"
        else:
            reason = "best_postcode_score_below_review"

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
            "score": round(float(selected.score if selected else (top.score if top else 0.0)), 6),
            "runner_up_score": round(float(runner_up), 6),
            "margin": round(float(margin), 6),
            "postcode_agrees": bool(selected and postcode and selected.record.postcode == postcode),
            "postcode_candidate_count": int(len(candidates)),
            "exact_name_candidate_count": int(exact_count),
            "rejected_post_incorporation": int(rejected_post_incorporation),
            "incorporation_date_missing": bool(
                selected is not None and selected.record.incorporation_date is None
            ),
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
    method = joined.loc[~joined["tier"].eq("unmatched"), ["tier", "matched_on", "matched_name_kind"]].copy()
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
    gives the requested spread across score, name source, vintage and method.
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


def review_sample(
    judgments: pd.DataFrame,
    matches: pd.DataFrame,
    settings: Settings,
    *,
    seed: int | None = None,
) -> pd.DataFrame:
    """Select the confidential 500/300/200 pair sample, redistributing shortages."""

    desired = {
        "auto": settings.sample_auto,
        "review": settings.sample_review,
        "fallback_review": settings.sample_fallback,
    }
    sampling_columns = [
        "ID",
        "tier",
        "reason",
        "matched_company_number",
        "matched_name_kind",
        "matched_on",
        "score",
    ]
    canonical_source_columns = (
        "source_company_name",
        "source_trading_name",
        "source_postcode",
    )
    missing = set((*sampling_columns, *canonical_source_columns)) - set(matches.columns)
    if missing:
        raise ValueError(f"matches missing review column(s): {sorted(missing)}")

    accepted_mask = matches["tier"].isin(desired)
    accepted = matches.loc[accepted_mask, sampling_columns].copy()
    accepted["__match_position"] = accepted_mask.to_numpy().nonzero()[0]
    if accepted["ID"].astype(str).duplicated().any():
        raise ValueError("matches contain duplicate ID values")
    available = {tier: int(accepted["tier"].eq(tier).sum()) for tier in desired}
    target = min(sum(desired.values()), sum(available.values()))
    allocation = {tier: min(desired[tier], available[tier]) for tier in desired}
    deficit = target - sum(allocation.values())
    tier_order = {tier: position for position, tier in enumerate(desired)}
    while deficit:
        choices = [tier for tier in desired if available[tier] > allocation[tier]]
        if not choices:
            break
        tier = max(choices, key=lambda value: (
            available[value] - allocation[value], -tier_order[value]
        ))
        take = min(deficit, available[tier] - allocation[tier])
        allocation[tier] += take
        deficit -= take

    actual_seed = settings.locked_seed if seed is None else int(seed)
    dates = judgments[["ID", "JudgmentDate"]].copy()
    dates["JudgmentDate"] = pd.to_datetime(
        dates["JudgmentDate"], format="mixed", dayfirst=True, errors="coerce"
    )
    accepted = accepted.merge(dates, on="ID", how="left", validate="one_to_one")
    accepted["__score_band"] = pd.cut(
        pd.to_numeric(accepted["score"], errors="coerce"),
        bins=[-0.001, settings.review_threshold, settings.auto_threshold, 0.95, 1.001],
        labels=["below_review", "review", "strong", "exact"],
        include_lowest=True,
        right=False,
    ).astype("string").fillna("missing")
    accepted["__vintage"] = accepted["JudgmentDate"].dt.year.astype("Int64").astype("string")
    accepted["__stratum"] = (
        accepted[
            ["__score_band", "matched_on", "matched_name_kind", "reason", "__vintage"]
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
    pieces: list[pd.DataFrame] = []
    for tier in desired:
        piece = _equal_probability_systematic_sample(
            accepted[accepted["tier"].eq(tier)].copy(),
            allocation[tier],
            seed=actual_seed,
            tier=tier,
        )
        pieces.append(piece)
    selected = pd.concat(pieces, ignore_index=True) if pieces else accepted.head(0)
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
            "__score_band",
            "__vintage",
            "__stratum",
        ],
        errors="ignore",
    )
    for column in output_source_columns:
        sampled[column] = source[column].to_numpy()
    sampled["__tier_order"] = sampled["tier"].map(tier_order)
    sampled = sampled.sort_values(["__tier_order", "ID"], kind="stable").drop(
        columns="__tier_order"
    ).reset_index(drop=True)
    sampled.insert(0, "review_row_id", [f"R{position:04d}" for position in range(1, len(sampled) + 1)])
    sampled.insert(1, "review_tier", sampled["tier"])
    sampled.insert(2, "review_decision", "")
    sampled.insert(3, "review_notes", "")
    sampled.insert(
        4,
        "data_classification",
        "RT INTERNAL - CONTAINS IDENTIFIERS - DO NOT EGRESS",
    )
    sampled["match_method"] = sampled["reason"]
    sampled["sample_seed"] = actual_seed
    sampled["sample_allocation"] = sampled["tier"].map(allocation).astype(int)
    sampled["sampling_design"] = "equal_probability_systematic_stratified_v1"
    sampled["sampling_weight"] = sampled["tier"].map(
        {
            tier: available[tier] / allocation[tier] if allocation[tier] else 0.0
            for tier in desired
        }
    )
    return sampled


__all__ = [
    "CH_CURRENT_SNAPSHOT_COLUMNS",
    "CH_REQUIRED_COLUMNS",
    "CHIndex",
    "CompanyRecord",
    "MATCH_TIERS",
    "NamePeriod",
    "build_relevant_ch_index",
    "match_diagnostics",
    "match_judgments",
    "normalize_name",
    "normalize_postcode",
    "review_sample",
]
