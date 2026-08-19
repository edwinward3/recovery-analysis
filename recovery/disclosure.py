"""Stage only approved aggregate artefacts and enforce RT disclosure controls.

The scanner skips a sibling ``rt_internal`` area but never skips anything
inside an egress directory.  It performs local file inspection only: no
network or shell operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable, Mapping, Sequence
import re
import shutil

import pandas as pd


_TEXT_SUFFIXES = frozenset({".csv", ".txt", ".json", ".md", ".log"})
_POSTCODE_RE = re.compile(
    r"(?<![A-Z0-9])(?:GIR\s?0AA|[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2})(?![A-Z0-9])",
    re.IGNORECASE,
)
_LETTERED_COMPANY_NUMBER_RE = re.compile(r"(?<![A-Z0-9])[A-Z]{2}\d{6}(?![A-Z0-9])", re.I)
_PREFIXED_COMPANY_NUMBER_RE = re.compile(
    r"\b(?:company|co)\s*(?:number|no\.?)[\s:#-]*(?:[A-Z]{2}\d{6}|\d{8})\b",
    re.IGNORECASE,
)
_NUMERIC_COMPANY_NUMBER_RE = re.compile(r"\d{8}")
_COMPANY_NAME_RE = re.compile(
    r"\b(?:[A-Z0-9][A-Z0-9&.'’()/-]*\s+){1,10}(?:LTD|LIMITED|PLC|LLP)\b",
    re.IGNORECASE,
)
_SAFE_NAME_COLUMNS = frozenset(
    {"artifact_name", "feature_name", "file_name", "filename", "metric_name", "model_name"}
)
_SAFE_EIGHT_DIGIT_COLUMNS = frozenset(
    {
        "date",
        "diagnostic_seed",
        "judgment_date",
        "locked_seed",
        "observation_date",
        "run_seed",
        "seed",
        "snapshot_date",
    }
)
_EXACT_IDENTIFIER_COLUMNS = frozenset(
    {
        "id",
        "rt_id",
        "judgment_id",
        "case_id",
        "company_id",
        "company_name",
        "company_number",
        "company_postcode",
        "defendant_name",
        "defendant_company_name",
        "defendant_trading_name",
        "defendant_postcode",
        "trading_name",
        "matched_name",
        "matched_company_name",
        "matched_company_number",
        "address",
        "postcode",
    }
)


@dataclass(frozen=True, slots=True)
class IdentifierFinding:
    """One kind of potential row-level identifier found in one file."""

    relative_path: str
    kind: str
    detail: str


@dataclass(frozen=True, slots=True)
class DisclosureReport:
    """Result of staging or validating an egress directory."""

    passed: bool
    findings: tuple[IdentifierFinding, ...]
    staged_files: tuple[str, ...] = ()
    suppressed_rows: tuple[tuple[str, int], ...] = ()


class DisclosureViolation(RuntimeError):
    """Raised when a candidate egress directory is not disclosure-safe."""

    def __init__(self, report: DisclosureReport):
        self.report = report
        summary = "; ".join(
            f"{finding.relative_path}: {finding.kind}" for finding in report.findings
        )
        super().__init__("egress disclosure gate failed" + (f": {summary}" if summary else ""))


def suppress_small_cells(
    frame: pd.DataFrame,
    *,
    count_columns: str | Sequence[str],
    min_cell_n: int = 10,
) -> tuple[pd.DataFrame, int]:
    """Remove aggregate rows supported by fewer than ``min_cell_n`` records."""

    if min_cell_n < 1:
        raise ValueError("min_cell_n must be positive")
    columns = (count_columns,) if isinstance(count_columns, str) else tuple(count_columns)
    if not columns:
        raise ValueError("at least one count column is required")
    suppress = pd.Series(False, index=frame.index)
    for column in columns:
        if column not in frame.columns:
            raise ValueError(f"missing disclosure count column {column!r}")
        counts = pd.to_numeric(frame[column], errors="coerce")
        if counts.isna().any() or counts.lt(0).any():
            raise ValueError(f"disclosure count column {column!r} must be non-negative numeric")
        # A structural zero reveals no underlying record. Positive cells below
        # the threshold are suppressed conservatively at whole-row level.
        suppress |= counts.gt(0) & counts.lt(min_cell_n)
    return frame.loc[~suppress].reset_index(drop=True), int(suppress.sum())


def scan_identifiers(
    root: str | Path,
    *,
    known_identifiers: Iterable[str] = (),
    ignore_rt_internal: bool = True,
) -> tuple[IdentifierFinding, ...]:
    """Scan local text artefacts, ignoring only a top-level RT-internal sibling."""

    base = Path(root)
    if not base.exists():
        return ()
    known = tuple(value.strip() for value in known_identifiers if value and value.strip())
    findings: list[IdentifierFinding] = []
    skip_internal = ignore_rt_internal and base.name.casefold() != "egress"
    files = [base] if base.is_file() else sorted(path for path in base.rglob("*") if path.is_file())
    for path in files:
        relative = Path(path.name) if base.is_file() else path.relative_to(base)
        if skip_internal and relative.parts and relative.parts[0].casefold() == "rt_internal":
            continue
        if path.suffix.casefold() not in _TEXT_SUFFIXES:
            findings.append(
                IdentifierFinding(str(relative), "uninspectable_file", "file type is not allowlisted")
            )
            continue
        if path.suffix.casefold() == ".csv":
            findings.extend(_scan_csv(path, relative, known))
        else:
            text = path.read_text(encoding="utf-8", errors="replace")
            findings.extend(_scan_text(text, relative, known))
    return tuple(findings)


def validate_egress(
    egress_dir: str | Path, *, known_identifiers: Iterable[str] = ()
) -> DisclosureReport:
    """Validate an egress directory and raise on any possible identifier."""

    findings = scan_identifiers(
        egress_dir, known_identifiers=known_identifiers, ignore_rt_internal=False
    )
    report = DisclosureReport(passed=not findings, findings=findings)
    if findings:
        raise DisclosureViolation(report)
    return report


def stage_egress(
    source_dir: str | Path,
    egress_dir: str | Path,
    *,
    allowlist: Mapping[str, str | Sequence[str] | None],
    min_cell_n: int = 10,
    known_identifiers: Iterable[str] = (),
) -> DisclosureReport:
    """Copy only explicitly named artefacts into a newly validated egress area.

    ``allowlist`` maps each relative filename to its disclosure count column(s),
    or to ``None`` for a count-free aggregate.  Unlisted source files are never
    copied.  ``rt_internal`` paths cannot be allowlisted.
    """

    source_root = Path(source_dir).resolve()
    destination = Path(egress_dir)
    if not allowlist:
        raise ValueError("egress allowlist cannot be empty")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("egress directory must be absent or empty before staging")

    policies: list[tuple[Path, tuple[str, ...]]] = []
    for raw_name, raw_counts in allowlist.items():
        relative = Path(raw_name)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError(f"unsafe egress allowlist path: {raw_name!r}")
        if relative.parts[0].casefold() == "rt_internal":
            raise ValueError("RT-internal files cannot be staged for egress")
        if relative.suffix.casefold() not in _TEXT_SUFFIXES:
            raise ValueError(f"egress file type is not allowlisted: {raw_name!r}")
        source_path = source_root / relative
        if not source_path.is_file() or source_path.is_symlink():
            raise FileNotFoundError(f"allowlisted source file is missing or unsafe: {raw_name}")
        resolved = source_path.resolve()
        if source_root not in resolved.parents:
            raise ValueError(f"allowlisted source escapes source directory: {raw_name!r}")
        if raw_counts is None:
            counts: tuple[str, ...] = ()
        elif isinstance(raw_counts, str):
            counts = (raw_counts,)
        else:
            counts = tuple(raw_counts)
        if counts and relative.suffix.casefold() != ".csv":
            raise ValueError("small-cell count columns may only be assigned to CSV artefacts")
        policies.append((relative, counts))

    suppressed: list[tuple[str, int]] = []
    staged_names = tuple(str(relative) for relative, _ in policies)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".egress-stage-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        for relative, count_columns in policies:
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_root / relative, target)
            if count_columns:
                frame = pd.read_csv(target)
                cleaned, removed = suppress_small_cells(
                    frame, count_columns=count_columns, min_cell_n=min_cell_n
                )
                cleaned.to_csv(target, index=False)
                if removed:
                    suppressed.append((str(relative), removed))

        findings = scan_identifiers(
            staging, known_identifiers=known_identifiers, ignore_rt_internal=False
        )
        if findings:
            raise DisclosureViolation(
                DisclosureReport(
                    passed=False,
                    findings=findings,
                    staged_files=staged_names,
                    suppressed_rows=tuple(suppressed),
                )
            )

        destination.mkdir(parents=True, exist_ok=True)
        for relative, _ in policies:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(staging / relative, target)

    findings = scan_identifiers(
        destination, known_identifiers=known_identifiers, ignore_rt_internal=False
    )
    report = DisclosureReport(
        passed=not findings,
        findings=findings,
        staged_files=staged_names,
        suppressed_rows=tuple(suppressed),
    )
    if findings:
        raise DisclosureViolation(report)
    return report


def _scan_csv(
    path: Path, relative: Path, known_identifiers: tuple[str, ...]
) -> list[IdentifierFinding]:
    try:
        frame = pd.read_csv(path, dtype="string", keep_default_na=False)
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        return [IdentifierFinding(str(relative), "unreadable_csv", type(exc).__name__)]
    findings: list[IdentifierFinding] = []
    sensitive = [str(column) for column in frame.columns if _is_identifier_column(column)]
    if sensitive:
        findings.append(
            IdentifierFinding(
                str(relative),
                "identifier_column",
                "forbidden heading(s): " + ", ".join(sensitive),
            )
        )
    numeric_company_number = any(
        _column_has_numeric_company_number(frame[column])
        for column in frame.columns
        if _canon_column(column) not in _SAFE_EIGHT_DIGIT_COLUMNS
    )
    if numeric_company_number:
        findings.append(
            IdentifierFinding(
                str(relative),
                "company_number_value",
                "one or more possible values found",
            )
        )
    text = frame.astype("string").to_csv(index=False)
    findings.extend(_scan_text(text, relative, known_identifiers))
    return _deduplicate_findings(findings)


def _scan_text(
    text: str, relative: Path, known_identifiers: tuple[str, ...]
) -> list[IdentifierFinding]:
    tests = (
        ("postcode_value", _POSTCODE_RE.search(text)),
        ("company_number_value", _LETTERED_COMPANY_NUMBER_RE.search(text)),
        ("company_number_value", _PREFIXED_COMPANY_NUMBER_RE.search(text)),
        ("company_name_value", _COMPANY_NAME_RE.search(text)),
    )
    findings = [
        IdentifierFinding(str(relative), kind, "one or more possible values found")
        for kind, match in tests
        if match
    ]
    folded = text.casefold()
    if any(identifier.casefold() in folded for identifier in known_identifiers):
        findings.append(
            IdentifierFinding(
                str(relative), "known_identifier_value", "one or more supplied values found"
            )
        )
    return _deduplicate_findings(findings)


def _canon_column(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().casefold()).strip("_")


def _is_identifier_column(value: object) -> bool:
    key = _canon_column(value)
    if key in _SAFE_NAME_COLUMNS:
        return False
    if key in _EXACT_IDENTIFIER_COLUMNS:
        return True
    if "postcode" in key or "address" in key:
        return True
    if key.endswith("_id") and key not in {"model_id", "run_id"}:
        return True
    if "company" in key and ("name" in key or "number" in key):
        return True
    if "defendant" in key and "name" in key:
        return True
    if "trading" in key and "name" in key:
        return True
    return False


def _column_has_numeric_company_number(values: pd.Series) -> bool:
    return bool(
        values.astype("string")
        .str.strip()
        .str.fullmatch(_NUMERIC_COMPANY_NUMBER_RE, na=False)
        .any()
    )


def _deduplicate_findings(findings: list[IdentifierFinding]) -> list[IdentifierFinding]:
    seen: set[tuple[str, str]] = set()
    unique: list[IdentifierFinding] = []
    for finding in findings:
        key = (finding.relative_path, finding.kind)
        if key not in seen:
            seen.add(key)
            unique.append(finding)
    return unique
