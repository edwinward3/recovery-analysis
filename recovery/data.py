"""Read and check the RT extract and stream the Companies House file.

Every valid RT row is retained for matching; cohort filters belong to Run 2.
``Date Inserted`` is used only to measure registration delay. Judgment age is
measured from ``JudgmentDate`` to the supplied extract date. This file makes no
internet, shell or cache calls and writes no analysis output.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterator
import hashlib
import re
import zipfile

import pandas as pd


DAYS_PER_MONTH = 30.44

# ===== the columns and fixed values expected from RT =====

REQUIRED_RT_COLUMNS: tuple[str, ...] = (
    "ID",
    "Date Inserted",
    "JudgmentDate",
    "JudgmentStatus",
    "DefendantType",
    "Jurisdiction",
    "Defendant Company Name",
    "Defendant_Postcode",
)

OPTIONAL_RT_COLUMNS: dict[str, object] = {
    "Amount": pd.NA,
    "Defendant Trading Name": "",
    "Defendant Address": "",
}

RT_COLUMNS: tuple[str, ...] = (*REQUIRED_RT_COLUMNS, *OPTIONAL_RT_COLUMNS)

_HEADER_ALIASES: dict[str, str] = {
    "id": "ID",
    "dateinserted": "Date Inserted",
    "judgmentdate": "JudgmentDate",
    "judgmentstatus": "JudgmentStatus",
    "defendanttype": "DefendantType",
    "jurisdiction": "Jurisdiction",
    "defendantcompanyname": "Defendant Company Name",
    "companyname": "Defendant Company Name",
    "defendantpostcode": "Defendant_Postcode",
    "postcode": "Defendant_Postcode",
    "amount": "Amount",
    "defendanttradingname": "Defendant Trading Name",
    "tradingname": "Defendant Trading Name",
    "defendantaddress": "Defendant Address",
    "address": "Defendant Address",
}

_CODED_VALUE_MAPS: dict[str, dict[str, str]] = {
    "JudgmentStatus": {
        "satisfied": "Satisfied",
        "unsatisfied": "Unsatisfied",
        "cancelled": "Cancelled",
        "canceled": "Cancelled",
    },
    "DefendantType": {
        "corporate": "Corporate",
        "non corporate": "Non-Corporate",
        "noncorporate": "Non-Corporate",
        "consumer": "Consumer",
        "non consumer": "Non-Consumer",
        "nonconsumer": "Non-Consumer",
    },
    "Jurisdiction": {
        "england and wales": "England and Wales",
        "england wales": "England and Wales",
        "scotland": "Scotland",
    },
}


# ===== the identifier-free facts recorded about the RT input =====

@dataclass(frozen=True, slots=True)
class DataAudit:
    """Identifier-free facts recording how the RT input was interpreted."""

    rows: int
    observation_date: str
    date_inserted_distinct: int
    judgment_date_min: str
    judgment_date_max: str
    registration_lag_days_min: int
    registration_lag_days_median: float
    registration_lag_days_max: int
    age_at_observation_months_min: float
    age_at_observation_months_median: float
    age_at_observation_months_max: float
    invalid_amount_rows: int
    missing_company_name_rows: int
    missing_postcode_rows: int
    extra_headers: tuple[str, ...]
    absent_optional_columns: tuple[str, ...]
    analysis_fingerprint: str
    status_counts: dict[str, int]
    defendant_type_counts: dict[str, int]
    jurisdiction_counts: dict[str, int]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


# ===== read, standardise and validate the full RT extract =====

def _canon_header(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _canon_code(value: object) -> str:
    text = str(value).strip().lower().replace("&", " and ")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _observation_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    try:
        if isinstance(value, str) and not re.fullmatch(r"\s*\d{4}-\d{2}-\d{2}\s*", value):
            stamp = pd.to_datetime(value, dayfirst=True, errors="raise")
        else:
            stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid observation date: {value!r}") from exc
    if pd.isna(stamp):
        raise ValueError("observation date is missing")
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert(None)
    return stamp.normalize()


def _read_csv(path: Path) -> pd.DataFrame:
    problem: UnicodeDecodeError | None = None
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return pd.read_csv(
                path,
                dtype=str,
                encoding=encoding,
                keep_default_na=False,
                low_memory=False,
            )
        except UnicodeDecodeError as exc:
            problem = exc
    raise ValueError(
        f"{path.name} is not readable as UTF-8 or Windows-1252; re-save it as CSV UTF-8"
    ) from problem


def _read_rt_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            return _read_csv(path)
        if suffix in {".xlsx", ".xlsm"}:
            try:
                return pd.read_excel(path, dtype=str, engine="openpyxl")
            except ImportError as exc:
                raise ValueError("XLSX input requires the bundled openpyxl package") from exc
    except Exception as exc:
        raise ValueError(f"could not read {path.name}: {type(exc).__name__}: {exc}") from exc
    raise ValueError(f"unsupported RT input format {suffix!r}; expected .csv, .xlsx, or .xlsm")


def _standardise_headers(frame: pd.DataFrame) -> pd.DataFrame:
    rename: dict[object, str] = {}
    seen: dict[str, object] = {}
    for original in frame.columns:
        expected = _HEADER_ALIASES.get(_canon_header(original))
        if expected is None:
            continue
        if expected in seen:
            raise ValueError(
                f"duplicate columns map to {expected!r}: {seen[expected]!r} and {original!r}"
            )
        seen[expected] = original
        rename[original] = expected
    missing = [column for column in REQUIRED_RT_COLUMNS if column not in seen]
    if missing:
        raise ValueError(f"RT extract is missing required column(s): {missing}")
    result = frame.rename(columns=rename).copy()
    for column, default in OPTIONAL_RT_COLUMNS.items():
        if column not in result:
            result[column] = default
    extra_headers = tuple(str(column) for column in result.columns if column not in RT_COLUMNS)
    absent_optional = tuple(column for column in OPTIONAL_RT_COLUMNS if column not in seen)
    # Extra input fields are audited but never carried through the confidential
    # in-memory pipeline. This keeps the one RT frame narrow at full scale.
    result = result.loc[:, list(RT_COLUMNS)].copy()
    result.attrs["extra_headers"] = extra_headers
    result.attrs["absent_optional"] = absent_optional
    return result


def _normalise_coded_columns(frame: pd.DataFrame) -> None:
    for column, mapping in _CODED_VALUE_MAPS.items():
        raw = frame[column].astype("string").fillna("").str.strip()
        canonical = raw.map(lambda value: mapping.get(_canon_code(value), pd.NA))
        invalid = canonical.isna()
        if invalid.any():
            raise ValueError(
                f"{column} has {int(invalid.sum())} missing or unsupported row(s) "
                f"across {int(raw[invalid].nunique(dropna=False))} distinct value(s)"
            )
        frame[column] = canonical.astype(str)


def _parse_required_date(frame: pd.DataFrame, column: str) -> pd.Series:
    parsed = pd.to_datetime(frame[column], format="mixed", dayfirst=True, errors="coerce")
    invalid = parsed.isna()
    if invalid.any():
        raise ValueError(f"{column} has {int(invalid.sum())} missing or unparseable row(s)")
    return parsed.dt.normalize()


def _parse_amount(series: pd.Series) -> tuple[pd.Series, int]:
    raw = series.astype("string").fillna("").str.strip()
    cleaned = (
        raw.str.replace(",", "", regex=False)
        .str.replace("£", "", regex=False)
        .str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    )
    values = pd.to_numeric(cleaned.where(cleaned.ne("")), errors="coerce")
    invalid = raw.ne("") & values.isna()
    return values.astype(float), int(invalid.sum())


def frame_fingerprint(frame: pd.DataFrame) -> str:
    """Fingerprint a pinned-pandas frame without rereading its source file."""

    digest = hashlib.sha256()
    digest.update("\x1f".join(map(str, frame.columns)).encode("utf-8"))
    row_hashes = pd.util.hash_pandas_object(frame, index=False, categorize=False)
    digest.update(row_hashes.to_numpy(dtype="uint64", copy=False).tobytes())
    return digest.hexdigest()


def read_rt_extract(
    path: str | Path,
    observation_date: str | pd.Timestamp,
) -> tuple[pd.DataFrame, DataAudit]:
    """Read, standardise, and validate an RT extract.

    The returned frame preserves one row per judgment and adds only explicit
    timing fields.  Invalid identifiers, coded values, or dates stop the run.
    """

    source = Path(path)
    if not source.is_file():
        raise ValueError(f"RT extract does not exist: {source}")
    observed = _observation_timestamp(observation_date)
    frame = _standardise_headers(_read_rt_file(source))
    extra_headers = tuple(frame.attrs.get("extra_headers", ()))
    absent_optional = tuple(frame.attrs.get("absent_optional", ()))
    if frame.empty:
        raise ValueError("RT extract contains no judgment rows")

    ids = frame["ID"].astype("string").fillna("").str.strip()
    if ids.eq("").any():
        raise ValueError(f"ID has {int(ids.eq('').sum())} missing row(s)")
    duplicated = ids.duplicated(keep=False)
    if duplicated.any():
        raise ValueError(
            f"ID must be unique; {int(duplicated.sum())} row(s) share duplicate ID values"
        )
    frame["ID"] = ids.astype(str)

    _normalise_coded_columns(frame)
    frame["Date Inserted"] = _parse_required_date(frame, "Date Inserted")
    frame["JudgmentDate"] = _parse_required_date(frame, "JudgmentDate")

    insertion_before_judgment = frame["Date Inserted"] < frame["JudgmentDate"]
    if insertion_before_judgment.any():
        raise ValueError(
            "Date Inserted precedes JudgmentDate for "
            f"{int(insertion_before_judgment.sum())} row(s)"
        )
    after_observation = (frame["JudgmentDate"] > observed) | (frame["Date Inserted"] > observed)
    if after_observation.any():
        raise ValueError(
            f"{int(after_observation.sum())} row(s) occur after observation date "
            f"{observed.date().isoformat()}"
        )

    for column in (
        "Defendant Company Name",
        "Defendant Trading Name",
        "Defendant Address",
        "Defendant_Postcode",
    ):
        frame[column] = frame[column].astype("string").fillna("").str.strip().astype(str)

    amounts, invalid_amounts = _parse_amount(frame["Amount"])
    frame["Amount"] = amounts
    registration_lag = (frame["Date Inserted"] - frame["JudgmentDate"]).dt.days
    age_days = (observed - frame["JudgmentDate"]).dt.days
    frame["registration_lag_days"] = registration_lag.astype(int)
    frame["age_at_observation_days"] = age_days.astype(int)
    frame["age_at_observation_months"] = age_days / DAYS_PER_MONTH
    frame.attrs["observation_date"] = observed.date().isoformat()
    analysis_fingerprint = frame_fingerprint(frame.loc[:, list(RT_COLUMNS)])

    audit = DataAudit(
        rows=int(len(frame)),
        observation_date=observed.date().isoformat(),
        date_inserted_distinct=int(frame["Date Inserted"].nunique()),
        judgment_date_min=frame["JudgmentDate"].min().date().isoformat(),
        judgment_date_max=frame["JudgmentDate"].max().date().isoformat(),
        registration_lag_days_min=int(registration_lag.min()),
        registration_lag_days_median=float(registration_lag.median()),
        registration_lag_days_max=int(registration_lag.max()),
        age_at_observation_months_min=float(frame["age_at_observation_months"].min()),
        age_at_observation_months_median=float(frame["age_at_observation_months"].median()),
        age_at_observation_months_max=float(frame["age_at_observation_months"].max()),
        invalid_amount_rows=invalid_amounts,
        missing_company_name_rows=int(frame["Defendant Company Name"].eq("").sum()),
        missing_postcode_rows=int(frame["Defendant_Postcode"].eq("").sum()),
        extra_headers=extra_headers,
        absent_optional_columns=absent_optional,
        analysis_fingerprint=analysis_fingerprint,
        status_counts={
            str(key): int(value) for key, value in frame["JudgmentStatus"].value_counts().items()
        },
        defendant_type_counts={
            str(key): int(value) for key, value in frame["DefendantType"].value_counts().items()
        },
        jurisdiction_counts={
            str(key): int(value) for key, value in frame["Jurisdiction"].value_counts().items()
        },
    )
    return frame, audit


# ===== read the large Companies House CSV or ZIP a piece at a time =====

def iter_ch_chunks(
    path: str | Path,
    *,
    chunksize: int = 100_000,
    usecols: Callable[[object], bool] | None = None,
) -> Iterator[pd.DataFrame]:
    """Yield a CH CSV, or the first CSV in a ZIP, exactly once in chunks."""

    source = Path(path)
    if not source.is_file():
        raise ValueError(f"Companies House input does not exist: {source}")
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")
    read_options: dict[str, object] = {
        "dtype": str,
        "chunksize": chunksize,
        "keep_default_na": False,
        "encoding": "utf-8",
        "encoding_errors": "replace",
    }
    if usecols is not None:
        read_options["usecols"] = usecols

    if source.suffix.lower() == ".zip":
        try:
            with zipfile.ZipFile(source) as archive:
                csv_names = sorted(
                    name for name in archive.namelist() if name.lower().endswith(".csv")
                )
                if not csv_names:
                    raise ValueError(f"{source.name} contains no CSV file")
                with archive.open(csv_names[0]) as handle:
                    yield from pd.read_csv(handle, **read_options)
        except zipfile.BadZipFile as exc:
            raise ValueError(f"{source.name} is not a readable ZIP file") from exc
        return
    if source.suffix.lower() != ".csv":
        raise ValueError("Companies House input must be a .csv or .zip")
    yield from pd.read_csv(source, **read_options)


__all__ = [
    "DAYS_PER_MONTH",
    "DataAudit",
    "OPTIONAL_RT_COLUMNS",
    "REQUIRED_RT_COLUMNS",
    "RT_COLUMNS",
    "frame_fingerprint",
    "iter_ch_chunks",
    "read_rt_extract",
]
