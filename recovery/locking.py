"""Freeze and authorize one manifest-bound model release.

The private key is kept outside the repository.  Freezing reads input bytes only
to calculate hashes; it does not parse the RT outcome or fit a model.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
import argparse
import hashlib
import hmac
import json
import os
import re
import secrets

import pandas as pd

from .config import load_settings


DESIGN_SCHEMA_VERSION = 2
APPROVAL_SCHEMA_VERSION = 1
MAX_CH_SNAPSHOT_LAG_DAYS = 35
_APPROVAL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{7,99}$")


class ReleaseLockError(RuntimeError):
    """Raised before a locked outcome is opened or a release is repeated."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def file_sha256(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def source_sha256(
    package_dir: str | Path,
    settings_path: str | Path,
    *,
    extra_sources: Sequence[str | Path] = (),
) -> str:
    """Hash executable analysis sources, settings and named design sources."""

    package = Path(package_dir)
    paths = sorted(package.glob("*.py"), key=lambda path: path.name)
    paths.extend([Path(settings_path), *(Path(path) for path in extra_sources)])
    digest = hashlib.sha256()
    for path in paths:
        resolved = path.resolve()
        digest.update(resolved.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(resolved.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_design_manifest(
    *,
    judgments_path: str | Path,
    companies_house_path: str | Path,
    observation_date: str | pd.Timestamp,
    companies_house_date: str | pd.Timestamp,
    settings_path: str | Path,
    package_dir: str | Path,
    extra_sources: Sequence[str | Path] = (),
    bound_files: Mapping[str, str | Path] | None = None,
    recall_denominator_supported: bool = False,
) -> dict[str, Any]:
    """Return the immutable scientific and input contract for one release."""

    judgments = Path(judgments_path).resolve()
    companies = Path(companies_house_path).resolve()
    settings_source = Path(settings_path).resolve()
    bound = {
        str(name): Path(path).resolve()
        for name, path in sorted((bound_files or {}).items())
    }
    for name in bound:
        if not re.fullmatch(r"[a-z][a-z0-9_]{2,63}", name):
            raise ReleaseLockError(f"unsafe bound-file name: {name!r}")
    for source in (judgments, companies, settings_source, *bound.values()):
        if not source.is_file():
            raise ReleaseLockError(f"manifest source does not exist: {source}")
    observed = pd.Timestamp(observation_date)
    if pd.isna(observed):
        raise ReleaseLockError("observation date is missing")
    if observed.tzinfo is not None:
        observed = observed.tz_convert(None)
    observed = observed.normalize()
    ch_observed = pd.Timestamp(companies_house_date)
    if pd.isna(ch_observed):
        raise ReleaseLockError("Companies House snapshot date is missing")
    if ch_observed.tzinfo is not None:
        ch_observed = ch_observed.tz_convert(None)
    ch_observed = ch_observed.normalize()
    lag_days = int((observed - ch_observed).days)
    if lag_days < 0:
        raise ReleaseLockError(
            "Companies House snapshot post-dates the RT observation date"
        )
    if lag_days > MAX_CH_SNAPSHOT_LAG_DAYS:
        raise ReleaseLockError(
            "Companies House snapshot is more than "
            f"{MAX_CH_SNAPSHOT_LAG_DAYS} days before the RT observation date"
        )
    _verify_filename_snapshot_date(companies, ch_observed)
    settings = load_settings(settings_source)
    package = Path(package_dir).resolve()
    resolved_extra: list[Path] = []
    candidates = [*(Path(path).resolve() for path in extra_sources)]
    for default_name in ("STUDY_DESIGN.md", "requirements.lock"):
        default = package.parent / default_name
        if default.is_file():
            candidates.append(default.resolve())
    seen_extra: set[Path] = set()
    for source in candidates:
        if source not in seen_extra:
            if not source.is_file():
                raise ReleaseLockError(f"design source does not exist: {source}")
            resolved_extra.append(source)
            seen_extra.add(source)

    payload: dict[str, Any] = {
        "schema_version": DESIGN_SCHEMA_VERSION,
        "design_status": "frozen_cross_sectional_status_at_extract",
        "observation_date": observed.date().isoformat(),
        "companies_house_snapshot_date": ch_observed.date().isoformat(),
        "companies_house_snapshot_lag_days": lag_days,
        "inputs": {
            "rt_sha256": file_sha256(judgments),
            "rt_suffix": judgments.suffix.casefold(),
            "companies_house_sha256": file_sha256(companies),
            "companies_house_suffix": companies.suffix.casefold(),
            "bound_file_sha256": {
                name: file_sha256(path) for name, path in bound.items()
            },
        },
        "source": {
            "analysis_sha256": source_sha256(
                package_dir,
                settings_source,
                extra_sources=resolved_extra,
            ),
            "settings_sha256": file_sha256(settings_source),
        },
        "settings": settings.as_dict(),
        "scientific_contract": {
            "outcome": "Registry Trust Satisfied versus Unsatisfied status at extract date",
            "not_outcomes": [
                "cash recovery",
                "partial recovery",
                "loss given default",
                "investment return",
                "future satisfaction",
            ],
            "data_construct": (
                "one current-register stock snapshot; not an historical "
                "inflow/outflow or status-transition dataset"
            ),
            "descriptive_population": "all corporate England and Wales RT records",
            "predictive_population": (
                "post-one-month through 48-month Satisfied or Unsatisfied judgments "
                "uniquely exact-matched "
                "to a company in the specified live-company Companies House snapshot"
            ),
            "unit": "judgment; repeated companies retained and grouped",
            "validation": (
                "label-blind age-stratified company-group split; development sees "
                "train, validation and calibration outcomes only; test evaluated once"
            ),
            "primary_comparator": "flexible judgment-age-only baseline",
            "current_snapshot_features": "exploratory only",
            "matching_validation": "accepted and unmatched probability samples",
            "recall_denominator_supported": bool(recall_denominator_supported),
        },
    }
    manifest_hash = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return {"manifest": payload, "manifest_sha256": manifest_hash}


def write_design_manifest(path: str | Path, envelope: Mapping[str, Any]) -> None:
    destination = Path(path)
    if destination.exists():
        raise ReleaseLockError(f"refusing to overwrite frozen manifest: {destination}")
    _validated_manifest(envelope)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(envelope, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def generate_release_key(path: str | Path) -> Path:
    """Create a private 256-bit key without printing it or overwriting a file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(destination, flags, 0o600)
    try:
        os.write(descriptor, secrets.token_bytes(32))
    finally:
        os.close(descriptor)
    return destination


def create_release_approval(
    *,
    manifest_path: str | Path,
    key_path: str | Path,
    approval_path: str | Path,
    approval_id: str,
) -> dict[str, Any]:
    """Sign one frozen manifest; the key must remain with the release custodian."""

    if not _APPROVAL_ID.fullmatch(approval_id):
        raise ReleaseLockError("approval id must be 8-100 safe characters")
    manifest = _read_json(manifest_path)
    manifest_hash = _validated_manifest(manifest)
    key = _read_key(key_path)
    authorization = {
        "schema_version": APPROVAL_SCHEMA_VERSION,
        "approval_id": approval_id,
        "authorized_action": "single_locked_release",
        "manifest_sha256": manifest_hash,
    }
    envelope = {
        "authorization": authorization,
        "hmac_sha256": hmac.new(
            key, _canonical_json(authorization), hashlib.sha256
        ).hexdigest(),
    }
    destination = Path(approval_path)
    if destination.exists():
        raise ReleaseLockError(f"refusing to overwrite approval: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(envelope, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return envelope


def verify_release_approval(
    *,
    manifest_path: str | Path,
    approval_path: str | Path,
    key_path: str | Path,
) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    manifest_hash = _validated_manifest(manifest)
    approval = _read_json(approval_path)
    authorization = approval.get("authorization")
    signature = approval.get("hmac_sha256")
    if not isinstance(authorization, dict) or not isinstance(signature, str):
        raise ReleaseLockError("release approval has an invalid structure")
    if authorization.get("schema_version") != APPROVAL_SCHEMA_VERSION:
        raise ReleaseLockError("release approval schema is unsupported")
    approval_id = authorization.get("approval_id")
    if not isinstance(approval_id, str) or not _APPROVAL_ID.fullmatch(approval_id):
        raise ReleaseLockError("release approval id is invalid")
    if authorization.get("authorized_action") != "single_locked_release":
        raise ReleaseLockError("approval does not authorize a locked release")
    if not hmac.compare_digest(
        str(authorization.get("manifest_sha256", "")), manifest_hash
    ):
        raise ReleaseLockError("approval refers to a different design manifest")
    expected = hmac.new(
        _read_key(key_path), _canonical_json(authorization), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(signature, expected):
        raise ReleaseLockError("release approval signature is invalid")
    return authorization


def verify_manifest_sources(
    *,
    manifest_path: str | Path,
    judgments_path: str | Path,
    companies_house_path: str | Path,
    observation_date: str | pd.Timestamp,
    companies_house_date: str | pd.Timestamp,
    settings_path: str | Path,
    package_dir: str | Path,
    extra_sources: Sequence[str | Path] = (),
    bound_files: Mapping[str, str | Path] | None = None,
    recall_denominator_supported: bool = False,
) -> str:
    """Refuse release if data, date, settings, code or design source changed."""

    frozen = _read_json(manifest_path)
    frozen_hash = _validated_manifest(frozen)
    current = build_design_manifest(
        judgments_path=judgments_path,
        companies_house_path=companies_house_path,
        observation_date=observation_date,
        companies_house_date=companies_house_date,
        settings_path=settings_path,
        package_dir=package_dir,
        extra_sources=extra_sources,
        bound_files=bound_files,
        recall_denominator_supported=recall_denominator_supported,
    )
    if not hmac.compare_digest(current["manifest_sha256"], frozen_hash):
        raise ReleaseLockError("data, date, settings, code or design changed after freeze")
    return frozen_hash


def reserve_release(
    *,
    registry_dir: str | Path,
    authorization: Mapping[str, Any],
) -> Path:
    """Atomically consume an approval before any test outcome is evaluated."""

    approval_id = str(authorization.get("approval_id", ""))
    if not _APPROVAL_ID.fullmatch(approval_id):
        raise ReleaseLockError("cannot reserve an invalid approval id")
    registry = Path(registry_dir)
    registry.mkdir(parents=True, exist_ok=True)
    receipt = registry / f"{approval_id}.json"
    record = {
        "approval_id": approval_id,
        "manifest_sha256": authorization.get("manifest_sha256"),
        "status": "reserved",
        "reserved_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        with receipt.open("x", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as exc:
        raise ReleaseLockError(f"approval has already been consumed: {approval_id}") from exc
    return receipt


def finalize_release_receipt(
    receipt_path: str | Path,
    *,
    status: str,
    run_id: str | None = None,
) -> None:
    if status not in {"completed", "failed"}:
        raise ValueError("release receipt status must be completed or failed")
    receipt = Path(receipt_path)
    record = _read_json(receipt)
    if record.get("status") != "reserved":
        raise ReleaseLockError("release receipt is not reserved")
    record["status"] = status
    record["finished_utc"] = datetime.now(timezone.utc).isoformat()
    if run_id is not None:
        record["run_id"] = str(run_id)
    temporary = receipt.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, receipt)


def _validated_manifest(envelope: Mapping[str, Any]) -> str:
    payload = envelope.get("manifest")
    recorded = envelope.get("manifest_sha256")
    if not isinstance(payload, dict) or not isinstance(recorded, str):
        raise ReleaseLockError("design manifest has an invalid structure")
    if payload.get("schema_version") != DESIGN_SCHEMA_VERSION:
        raise ReleaseLockError("design manifest schema is unsupported")
    calculated = hashlib.sha256(_canonical_json(payload)).hexdigest()
    if not hmac.compare_digest(recorded, calculated):
        raise ReleaseLockError("design manifest hash is invalid")
    return calculated


def _read_key(path: str | Path) -> bytes:
    key = Path(path).read_bytes()
    if len(key) < 32:
        raise ReleaseLockError("release key must contain at least 32 bytes")
    return key


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseLockError(f"could not read release file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReleaseLockError(f"release file is not a JSON object: {path}")
    return value


def _verify_filename_snapshot_date(path: Path, declared: pd.Timestamp) -> None:
    """Cross-check an ISO date embedded in the CH filename when one is present."""

    candidates = re.findall(r"(?<!\d)(20\d{2}-\d{2}-\d{2})(?!\d)", path.name)
    if not candidates:
        return
    distinct = sorted(set(candidates))
    if len(distinct) != 1:
        raise ReleaseLockError(
            f"Companies House filename contains conflicting dates: {distinct}"
        )
    embedded = pd.Timestamp(distinct[0]).normalize()
    if embedded != declared:
        raise ReleaseLockError(
            "declared Companies House snapshot date does not match its filename"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freeze and approve one model release")
    commands = parser.add_subparsers(dest="command", required=True)
    key = commands.add_parser("generate-key")
    key.add_argument("--output", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--judgments", required=True)
    freeze.add_argument("--companies-house", required=True)
    freeze.add_argument("--observation-date", required=True)
    freeze.add_argument("--companies-house-date", required=True)
    freeze.add_argument("--settings", default="settings.toml")
    freeze.add_argument("--package-dir", default="recovery")
    freeze.add_argument("--design-source", action="append", default=[])
    freeze.add_argument("--accepted-adjudications", required=True)
    freeze.add_argument("--unmatched-adjudications", required=True)
    freeze.add_argument("--development-specification", required=True)
    freeze.add_argument("--recall-denominator-supported", action="store_true")
    freeze.add_argument("--output", required=True)
    sign = commands.add_parser("sign")
    sign.add_argument("--manifest", required=True)
    sign.add_argument("--key-file", required=True)
    sign.add_argument("--approval-id", required=True)
    sign.add_argument("--output", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--manifest", required=True)
    verify.add_argument("--approval", required=True)
    verify.add_argument("--key-file", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "generate-key":
            generate_release_key(args.output)
            print(f"KEY CREATED: {args.output}")
        elif args.command == "freeze":
            envelope = build_design_manifest(
                judgments_path=args.judgments,
                companies_house_path=args.companies_house,
                observation_date=args.observation_date,
                companies_house_date=args.companies_house_date,
                settings_path=args.settings,
                package_dir=args.package_dir,
                extra_sources=args.design_source,
                bound_files={
                    "accepted_adjudications": args.accepted_adjudications,
                    "unmatched_adjudications": args.unmatched_adjudications,
                    "development_specification": args.development_specification,
                },
                recall_denominator_supported=args.recall_denominator_supported,
            )
            write_design_manifest(args.output, envelope)
            print(f"FROZEN MANIFEST: {args.output}")
            print(f"SHA256: {envelope['manifest_sha256']}")
        elif args.command == "sign":
            create_release_approval(
                manifest_path=args.manifest,
                key_path=args.key_file,
                approval_path=args.output,
                approval_id=args.approval_id,
            )
            print(f"APPROVAL CREATED: {args.output}")
        else:
            authorization = verify_release_approval(
                manifest_path=args.manifest,
                approval_path=args.approval,
                key_path=args.key_file,
            )
            print(f"APPROVAL VERIFIED: {authorization['approval_id']}")
        return 0
    except Exception as exc:
        print(f"STOP: {type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
