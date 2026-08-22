from __future__ import annotations

from pathlib import Path

import pytest

from recovery.locking import (
    ReleaseLockError,
    build_design_manifest,
    create_release_approval,
    finalize_release_receipt,
    generate_release_key,
    reserve_release,
    verify_manifest_sources,
    verify_release_approval,
    write_design_manifest,
)


ROOT = Path(__file__).resolve().parents[1]


def _sources(tmp_path: Path) -> tuple[Path, Path]:
    rt = tmp_path / "rt.csv"
    ch = tmp_path / "ch.zip"
    rt.write_bytes(b"private-rt-fixture")
    ch.write_bytes(b"private-ch-fixture")
    return rt, ch


def test_manifest_approval_source_check_and_one_time_receipt(tmp_path: Path) -> None:
    rt, ch = _sources(tmp_path)
    manifest_path = tmp_path / "design.json"
    approval_path = tmp_path / "approval.json"
    key_path = generate_release_key(tmp_path / "outside-repo.key")
    manifest = build_design_manifest(
        judgments_path=rt,
        companies_house_path=ch,
        observation_date="2026-06-01",
        companies_house_date="2026-06-01",
        settings_path=ROOT / "settings.toml",
        package_dir=ROOT / "recovery",
    )
    write_design_manifest(manifest_path, manifest)
    create_release_approval(
        manifest_path=manifest_path,
        key_path=key_path,
        approval_path=approval_path,
        approval_id="release-0001",
    )
    authorization = verify_release_approval(
        manifest_path=manifest_path,
        approval_path=approval_path,
        key_path=key_path,
    )
    assert verify_manifest_sources(
        manifest_path=manifest_path,
        judgments_path=rt,
        companies_house_path=ch,
        observation_date="2026-06-01",
        companies_house_date="2026-06-01",
        settings_path=ROOT / "settings.toml",
        package_dir=ROOT / "recovery",
    ) == manifest["manifest_sha256"]
    receipt = reserve_release(
        registry_dir=tmp_path / "receipts",
        authorization=authorization,
    )
    finalize_release_receipt(receipt, status="completed", run_id="locked_fixture")
    with pytest.raises(ReleaseLockError, match="already been consumed"):
        reserve_release(
            registry_dir=tmp_path / "receipts",
            authorization=authorization,
        )


def test_tampering_or_changed_source_is_rejected(tmp_path: Path) -> None:
    rt, ch = _sources(tmp_path)
    manifest_path = tmp_path / "design.json"
    approval_path = tmp_path / "approval.json"
    key_path = generate_release_key(tmp_path / "release.key")
    manifest = build_design_manifest(
        judgments_path=rt,
        companies_house_path=ch,
        observation_date="2026-06-01",
        companies_house_date="2026-06-01",
        settings_path=ROOT / "settings.toml",
        package_dir=ROOT / "recovery",
    )
    write_design_manifest(manifest_path, manifest)
    create_release_approval(
        manifest_path=manifest_path,
        key_path=key_path,
        approval_path=approval_path,
        approval_id="release-0002",
    )
    ch.write_bytes(b"changed")
    with pytest.raises(ReleaseLockError, match="changed after freeze"):
        verify_manifest_sources(
            manifest_path=manifest_path,
            judgments_path=rt,
            companies_house_path=ch,
            observation_date="2026-06-01",
            companies_house_date="2026-06-01",
            settings_path=ROOT / "settings.toml",
            package_dir=ROOT / "recovery",
        )

    approval = approval_path.read_text(encoding="utf-8").replace(
        "release-0002", "release-9999"
    )
    approval_path.write_text(approval, encoding="utf-8")
    with pytest.raises(ReleaseLockError, match="signature is invalid"):
        verify_release_approval(
            manifest_path=manifest_path,
            approval_path=approval_path,
            key_path=key_path,
        )


def test_short_key_and_overwrite_are_rejected(tmp_path: Path) -> None:
    key = tmp_path / "key"
    key.write_bytes(b"short")
    with pytest.raises(FileExistsError):
        generate_release_key(key)

    rt, ch = _sources(tmp_path)
    manifest = build_design_manifest(
        judgments_path=rt,
        companies_house_path=ch,
        observation_date="2026-06-01",
        companies_house_date="2026-06-01",
        settings_path=ROOT / "settings.toml",
        package_dir=ROOT / "recovery",
    )
    manifest_path = tmp_path / "manifest.json"
    write_design_manifest(manifest_path, manifest)
    with pytest.raises(ReleaseLockError, match="overwrite frozen manifest"):
        write_design_manifest(manifest_path, manifest)
