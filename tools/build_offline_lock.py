"""Build a hash-locked requirements file from reviewed Windows wheels.

Developer-only input: ``wheels/py313`` and ``wheels/py314``. Outputs are
``requirements.lock`` and an offline-bundle manifest. This never reads RT data.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import hashlib

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename


ROOT = Path(__file__).resolve().parents[1]
WHEELS = ROOT / "wheels"


def main() -> int:
    wheel_paths: list[Path] = []
    folder_packages: dict[str, dict[str, str]] = {}
    for folder in (WHEELS / "py313", WHEELS / "py314"):
        if not folder.is_dir():
            raise SystemExit(f"missing wheel directory: {folder}")
        paths = sorted(folder.glob("*.whl"))
        wheel_paths.extend(paths)
        versions: dict[str, str] = {}
        for path in paths:
            name, version, _, _ = parse_wheel_filename(path.name)
            canonical = canonicalize_name(name)
            if canonical in versions:
                raise SystemExit(f"duplicate wheel for {canonical} in {folder}")
            versions[canonical] = str(version)
        folder_packages[folder.name] = versions
    if not wheel_paths:
        raise SystemExit("no wheels found")

    packages: dict[tuple[str, str], set[str]] = defaultdict(set)
    manifest_lines: list[str] = []
    for path in wheel_paths:
        name, version, _, _ = parse_wheel_filename(path.name)
        digest = _sha256(path)
        packages[(canonicalize_name(name), str(version))].add(digest)
        manifest_lines.append(f"{digest}  {path.relative_to(ROOT).as_posix()}")

    versions_by_name: dict[str, set[str]] = defaultdict(set)
    for name, version in packages:
        versions_by_name[name].add(version)
    conflicts = {
        name: versions for name, versions in versions_by_name.items() if len(versions) != 1
    }
    if conflicts:
        raise SystemExit(f"wheel directories contain conflicting versions: {conflicts}")
    if folder_packages["py313"] != folder_packages["py314"]:
        raise SystemExit("py313 and py314 wheel folders do not contain identical packages")

    required: dict[str, str] = {}
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        requirement = Requirement(stripped)
        pins = [item.version for item in requirement.specifier if item.operator == "=="]
        if len(pins) != 1 or len(requirement.specifier) != 1:
            raise SystemExit(f"runtime requirement is not exactly pinned: {stripped}")
        required[canonicalize_name(requirement.name)] = pins[0]
    missing_or_wrong = {
        name: (version, folder_packages["py313"].get(name))
        for name, version in required.items()
        if folder_packages["py313"].get(name) != version
    }
    if missing_or_wrong:
        raise SystemExit(f"root requirements missing or wrong in wheel folders: {missing_or_wrong}")

    lock_lines = [
        "# Hash-locked Windows runtime dependencies for Python 3.13 and 3.14.",
        "# RUN.bat installs these from PyPI, or from matching reviewed wheels when supplied.",
    ]
    continuation = " " + chr(92) + "\n"
    for (name, version), hashes in sorted(packages.items()):
        hash_lines = continuation.join(
            f"    --hash=sha256:{digest}" for digest in sorted(hashes)
        )
        lock_lines.append(f"{name}=={version}" + continuation + hash_lines)

    (ROOT / "requirements.lock").write_text(
        "\n".join(lock_lines) + "\n", encoding="utf-8"
    )
    (WHEELS / "MANIFEST.sha256").write_text(
        "\n".join(sorted(manifest_lines)) + "\n", encoding="utf-8"
    )
    print(f"locked {len(packages)} package versions from {len(wheel_paths)} wheels")
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
