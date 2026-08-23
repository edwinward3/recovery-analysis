"""Run the matching check."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Sequence
import argparse
import hashlib
import re
import shutil
import sys

import pandas as pd

from .config import Settings, load_settings
from .data import DataAudit, read_rt_extract
from .disclosure import stage_egress
from .matching import (
    ACCEPTED_LINKAGE_VALIDATION_FILENAME,
    UNMATCHED_LINKAGE_VALIDATION_FILENAME,
    CHIndex,
    accepted_validation_sample,
    build_relevant_ch_index,
    match_diagnostics,
    match_judgments,
    unmatched_validation_sample,
)
from .reporting import (
    RunPaths,
    RunRecorder,
    build_data_audit_counts,
    create_run_paths,
    source_fingerprint,
    write_e1,
    write_e2,
    write_e5,
    write_summary,
)


MAX_CH_SNAPSHOT_LAG_DAYS = 35


class RunFailure(RuntimeError):
    """A problem that stops the run."""


def analyze(
    *,
    judgments_path: str | Path,
    companies_house_path: str | Path,
    observation_date: str | pd.Timestamp | None,
    companies_house_date: str | pd.Timestamp | None = None,
    settings_path: str | Path,
    output_base: str | Path,
    run_id: str | None = None,
    _match_validator: Callable[[pd.DataFrame], None] | None = None,
) -> RunPaths:
    """Run the matching check."""

    if observation_date is None:
        raise RunFailure("RT extract date is required; it must not default to today")
    if companies_house_date is None:
        raise RunFailure("Companies House file date is required")
    observed = _declared_date(observation_date, "RT extract date")
    ch_observed = _declared_date(companies_house_date, "Companies House file date")
    companies_file = Path(companies_house_path).resolve()
    _validate_ch_snapshot(companies_file, observed, ch_observed)
    settings_source = Path(settings_path).resolve()
    settings = load_settings(settings_source)

    paths: RunPaths | None = None
    try:
        paths = create_run_paths(output_base, run_id)
        result = _analyze_created_run(
            judgments_path=judgments_path,
            companies_house_path=companies_file,
            observed=observed,
            ch_observed=ch_observed,
            settings_source=settings_source,
            settings=settings,
            paths=paths,
            _match_validator=_match_validator,
        )
        return result
    except BaseException as exc:
        if paths is not None and paths.root.exists():
            try:
                shutil.rmtree(paths.root)
            except OSError as cleanup_error:
                raise RunFailure(
                    "the run failed and its incomplete files could not be removed: "
                    f"{cleanup_error}"
                ) from exc
        raise


def _analyze_created_run(
    *,
    judgments_path: str | Path,
    companies_house_path: Path,
    observed: pd.Timestamp,
    ch_observed: pd.Timestamp,
    settings_source: Path,
    settings: Settings,
    paths: RunPaths,
    _match_validator: Callable[[pd.DataFrame], None] | None,
) -> RunPaths:
    aggregate = paths.root / ".aggregate_staging"
    aggregate.mkdir()
    recorder = RunRecorder()
    judgments_file = Path(judgments_path).resolve()

    with recorder.stage("E1_read_validate_schema"):
        judgments, audit = read_rt_extract(judgments_file, observed)

    with recorder.stage("CH_stream_index") as record:
        ch_index = build_relevant_ch_index(judgments, companies_house_path)
        record["ch_rows_read"] = ch_index.stats.get("ch_rows_read")
        record["ch_rows_retained"] = ch_index.stats.get("ch_rows_retained")

    with recorder.stage("E2_exact_linkage") as record:
        matches = match_judgments(judgments, ch_index)
        if _match_validator is not None:
            _match_validator(matches)
        linkage_judgments, linkage_matches = _linkage_target(judgments, matches)
        diagnostics = match_diagnostics(linkage_judgments, linkage_matches)
        record["judgments_matched"] = int(
            linkage_matches["tier"].eq("exact_unique").sum()
        )
    with recorder.stage("E2_probability_validation_samples") as record:
        accepted_sample = accepted_validation_sample(
            linkage_judgments,
            linkage_matches,
            settings,
            seed=settings.diagnostic_seed,
        )
        unmatched_sample = unmatched_validation_sample(
            linkage_judgments,
            linkage_matches,
            settings.sample_size,
            seed=settings.diagnostic_seed,
        )
        accepted_sample.to_csv(
            paths.working / ACCEPTED_LINKAGE_VALIDATION_FILENAME,
            index=False,
            encoding="utf-8-sig",
        )
        unmatched_sample.to_csv(
            paths.working / UNMATCHED_LINKAGE_VALIDATION_FILENAME,
            index=False,
            encoding="utf-8-sig",
        )
        record["accepted_sample_rows"] = len(accepted_sample)
        record["unmatched_sample_rows"] = len(unmatched_sample)
        record["sample_seed"] = settings.diagnostic_seed

    with recorder.stage("aggregate_reports"):
        audit_counts = build_data_audit_counts(judgments, audit)
        funnel = _matching_funnel(
            judgments, linkage_matches, len(accepted_sample), len(unmatched_sample)
        )
        write_e1(aggregate, audit_counts, funnel)
        write_e2(aggregate, diagnostics)
        write_summary(
            aggregate / "SUMMARY.txt",
            _summary_context(
                audit=audit,
                ch_observed=ch_observed,
                judgments=judgments,
                matches=matches,
                settings=settings,
            ),
        )

    allowlist = _analysis_allowlist()
    working_files = sorted(
        path.relative_to(paths.working).as_posix()
        for path in paths.working.rglob("*")
        if path.is_file()
    )
    manifest = _run_manifest(
        paths=paths,
        observed=observed,
        ch_observed=ch_observed,
        audit=audit,
        ch_index=ch_index,
        settings=settings,
        settings_source=settings_source,
        judgments_file=judgments_file,
        companies_file=companies_house_path,
        working_files=working_files,
        allowlist=allowlist,
    )
    known_identifiers = _bounded_known_identifiers(
        pd.concat([accepted_sample, unmatched_sample], ignore_index=True, sort=False)
    )
    report_allowlist = {
        name: policy
        for name, policy in allowlist.items()
        if name not in {"E5_run_log.csv", "E5_run_manifest.json"}
    }
    with recorder.stage("E5_disclosure_gate") as record:
        with TemporaryDirectory(prefix=".disclosure-preview-", dir=paths.root) as temp:
            preview = stage_egress(
                aggregate,
                Path(temp) / "egress",
                allowlist=report_allowlist,
                min_cell_n=settings.min_cell_n,
                known_identifiers=known_identifiers,
            )
        record["staged_files"] = len(allowlist)
        record["suppressed_rows"] = sum(count for _, count in preview.suppressed_rows)
    manifest["disclosure"].update(
        {
            "status": "pass",
            "staged_files": sorted(allowlist),
            "suppressed_rows": [
                {"file": filename, "rows": count}
                for filename, count in preview.suppressed_rows
            ],
        }
    )
    write_e5(aggregate, recorder, manifest, min_cell_n=settings.min_cell_n)
    disclosure = stage_egress(
        aggregate,
        paths.results,
        allowlist=allowlist,
        min_cell_n=settings.min_cell_n,
        known_identifiers=known_identifiers,
    )
    if (
        tuple(disclosure.suppressed_rows) != tuple(preview.suppressed_rows)
        or set(disclosure.staged_files) != set(allowlist)
    ):
        raise RunFailure("final disclosure copy differed from its checked preview")
    shutil.rmtree(aggregate)
    return paths


def _linkage_target(
    judgments: pd.DataFrame, matches: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = judgments["DefendantType"].eq("Corporate") & judgments["Jurisdiction"].eq(
        "England and Wales"
    )
    target = judgments.loc[mask].copy()
    identifiers = set(target["ID"].astype(str))
    decisions = matches.loc[matches["ID"].astype(str).isin(identifiers)].copy()
    if len(target) != len(decisions):
        raise RunFailure(
            "corporate England and Wales records do not have one match decision each"
        )
    return target, decisions


def _matching_funnel(
    judgments: pd.DataFrame,
    matches: pd.DataFrame,
    accepted_rows: int,
    unmatched_rows: int,
) -> pd.DataFrame:
    tiers = matches["tier"].value_counts()
    return pd.DataFrame(
        [
            {"stage": "judgments_read", "rows": int(len(judgments))},
            {"stage": "matching_decisions", "rows": int(len(matches))},
            {"stage": "unique_exact_name", "rows": int(tiers.get("exact_unique", 0))},
            {"stage": "unmatched", "rows": int(tiers.get("unmatched", 0))},
            {"stage": "accepted_validation_sample", "rows": int(accepted_rows)},
            {"stage": "unmatched_validation_sample", "rows": int(unmatched_rows)},
        ]
    )


def _summary_context(
    *,
    audit: DataAudit,
    ch_observed: pd.Timestamp,
    judgments: pd.DataFrame,
    matches: pd.DataFrame,
    settings: Settings,
) -> dict[str, Any]:
    inserted = pd.to_datetime(judgments["Date Inserted"], errors="raise")
    corporate_ew = judgments["DefendantType"].eq("Corporate") & judgments[
        "Jurisdiction"
    ].eq("England and Wales")
    target_matches = matches.loc[corporate_ew]
    exact = int(target_matches["tier"].eq("exact_unique").sum())
    denominator = int(len(target_matches))
    return {
        "status": "MATCHING COMPLETE",
        "observation_date": audit.observation_date,
        "companies_house_date": ch_observed.date().isoformat(),
        "data_construct": audit.data_construct,
        "optional_fields": {
            "Satisfaction Date": {
                "present": "Satisfaction Date" not in audit.absent_optional_columns,
                "rows": audit.satisfaction_date_present_rows,
            },
            "Cancellation Date": {
                "present": "Cancellation Date" not in audit.absent_optional_columns,
                "rows": audit.cancellation_date_present_rows,
            },
            "Cancellation Reason": {
                "present": "Cancellation Reason" not in audit.absent_optional_columns,
                "rows": audit.cancellation_reason_present_rows,
            },
            "Status Effective Date": {
                "present": "Status Effective Date" not in audit.absent_optional_columns,
                "rows": audit.status_effective_date_present_rows,
            },
            "Snapshot Date": {
                "present": "Snapshot Date" not in audit.absent_optional_columns,
                "rows": audit.snapshot_date_present_rows,
            },
        },
        "min_cell_n": settings.min_cell_n,
        "date_inserted": {
            "distinct_values": audit.date_inserted_distinct,
            "minimum": inserted.min().date().isoformat(),
            "maximum": inserted.max().date().isoformat(),
        },
        "counts": {
            "rows_read": audit.rows,
            "matching_decisions": denominator,
            "missing_company_name": audit.missing_company_name_rows,
            "missing_postcode": audit.missing_postcode_rows,
            "date_inserted_before_judgment": audit.date_inserted_before_judgment_rows,
        },
        "match": {
            "denominator": denominator,
            "exact_unique": exact,
            "unmatched": denominator - exact,
            "coverage": exact / denominator if denominator else 0.0,
        },
        "accepted_file": ACCEPTED_LINKAGE_VALIDATION_FILENAME,
        "unmatched_file": UNMATCHED_LINKAGE_VALIDATION_FILENAME,
    }


def _run_manifest(
    *,
    paths: RunPaths,
    observed: pd.Timestamp,
    ch_observed: pd.Timestamp,
    audit: DataAudit,
    ch_index: CHIndex,
    settings: Settings,
    settings_source: Path,
    judgments_file: Path,
    companies_file: Path,
    working_files: list[str],
    allowlist: dict[str, str | Sequence[str] | None],
) -> dict[str, Any]:
    return {
        "schema_version": 6,
        "status": "CONFIDENTIAL - SEND ONLY TO EDWIN",
        "run_id": paths.root.name,
        "schema_construct": audit.data_construct,
        "observation_date": observed.date().isoformat(),
        "companies_house_snapshot_date": ch_observed.date().isoformat(),
        "companies_house_snapshot_lag_days": int((observed - ch_observed).days),
        "fingerprints": {
            "rt_raw_file": audit.raw_source_sha256,
            "rt_raw_header_schema": audit.raw_header_schema_sha256,
            "rt_analysis_content": audit.analysis_fingerprint,
            "rt_provenance": audit.provenance_fingerprint,
            "companies_house_raw_file": _file_sha256(companies_file),
            "companies_house_analysis_content": ch_index.stats.get("analysis_fingerprint"),
            "code_and_settings": source_fingerprint(Path(__file__).parent, settings_source),
            "settings_file": _file_sha256(settings_source),
        },
        "settings": settings.as_dict(),
        "matching_rule": "unique_date_valid_exact_normalized_name_v1",
        "linkage_validation_seed": settings.diagnostic_seed,
        "ch_index_stats": dict(ch_index.stats),
        "package_versions": _package_versions(),
        "artifact_manifest": {
            "reports": sorted(allowlist),
            "working_files": working_files,
        },
        "input_formats": {
            "judgments": judgments_file.suffix.casefold(),
            "companies_house": companies_file.suffix.casefold(),
        },
        "disclosure": {
            "status": "pending",
            "minimum_cell": settings.min_cell_n,
            "explicit_allowlist": True,
            "identifier_scan_required": True,
        },
    }


def _analysis_allowlist() -> dict[str, str | Sequence[str] | None]:
    return {
        "SUMMARY.txt": None,
        "E1_data_audit.csv": "rows",
        "E1_data_funnel.csv": "rows",
        "E2_match_coverage.csv": "rows",
        "E2_unmatched_reasons.csv": "rows",
        "E2_match_methods.csv": "rows",
        "E2_linkage_profile.csv": "rows",
        "E2_linkage_checks.csv": "rows",
        "E5_run_log.csv": None,
        "E5_run_manifest.json": None,
    }


def _bounded_known_identifiers(sample: pd.DataFrame) -> tuple[str, ...]:
    values: list[str] = []
    for column in (
        "source_company_name",
        "source_trading_name",
        "matched_company_name",
    ):
        if column in sample:
            values.extend(sample[column].astype("string").fillna("").tolist())
    return tuple(
        dict.fromkeys(
            value.strip()
            for value in values
            if len(value.strip()) >= 8 and any(character.isalpha() for character in value)
        )
    )


def _declared_date(value: str | pd.Timestamp, label: str) -> pd.Timestamp:
    if isinstance(value, str) and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value.strip()):
        raise RunFailure(f"{label} must use YYYY-MM-DD")
    try:
        timestamp = pd.Timestamp(value.strip() if isinstance(value, str) else value)
    except (TypeError, ValueError) as exc:
        raise RunFailure(f"{label} is invalid") from exc
    if pd.isna(timestamp):
        raise RunFailure(f"{label} is invalid")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(None)
    return timestamp.normalize()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_ch_snapshot(
    path: Path, observed: pd.Timestamp, ch_observed: pd.Timestamp
) -> None:
    lag = int((observed - ch_observed).days)
    if lag < 0:
        raise RunFailure("Companies House file date is after the RT extract date")
    if lag > MAX_CH_SNAPSHOT_LAG_DAYS:
        raise RunFailure(
            f"Companies House file is more than {MAX_CH_SNAPSHOT_LAG_DAYS} days "
            "older than the RT extract"
        )
    embedded = sorted(set(re.findall(r"(?<!\d)(20\d{2}-\d{2}-\d{2})(?!\d)", path.name)))
    if not embedded:
        raise RunFailure(
            "Companies House filename must contain its date as YYYY-MM-DD"
        )
    if len(embedded) > 1:
        raise RunFailure(f"Companies House filename contains conflicting dates: {embedded}")
    if embedded and pd.Timestamp(embedded[0]).normalize() != ch_observed:
        raise RunFailure("Companies House file date does not match its filename")


def _package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for name in (
        "numpy",
        "pandas",
        "openpyxl",
        "python-dateutil",
        "tzdata",
    ):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "not-installed"
    return result


def package_results(paths: RunPaths) -> Path:
    """Put the completed run in one ZIP file."""

    token = paths.root.name.removeprefix("run_")
    archive = paths.root.parent / f"SEND_TO_EDWIN_{token}.zip"
    if archive.exists():
        raise RunFailure(f"output ZIP already exists: {archive}")
    written = shutil.make_archive(
        str(archive.with_suffix("")),
        "zip",
        root_dir=paths.root.parent,
        base_dir=paths.root.name,
    )
    return Path(written).resolve()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Registry Trust data check")
    commands = parser.add_subparsers(dest="command", required=True)
    analyze_parser = commands.add_parser("analyze")
    analyze_parser.add_argument("--judgments", required=True)
    analyze_parser.add_argument("--companies-house", required=True)
    analyze_parser.add_argument("--observation-date", required=True)
    analyze_parser.add_argument("--companies-house-date", required=True)
    analyze_parser.add_argument("--settings", default="settings.toml")
    analyze_parser.add_argument("--output-base", default="outputs")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        paths = analyze(
            judgments_path=args.judgments,
            companies_house_path=args.companies_house,
            observation_date=args.observation_date,
            companies_house_date=args.companies_house_date,
            settings_path=args.settings,
            output_base=args.output_base,
        )
        archive = package_results(paths)
        print("RUN COMPLETE")
        print(f"SEND THIS FILE TO EDWIN: {archive}")
        return 0
    except Exception as exc:
        print(f"STOP: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
