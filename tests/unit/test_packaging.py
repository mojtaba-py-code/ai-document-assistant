"""The wheel must be self-contained: every migration file ships inside it.

`docassist migrate` falls back to ``docassist/_alembic/alembic.ini`` when it is installed from
a wheel; the files are mapped one by one in ``pyproject.toml`` (so caches never leak into the
wheel). This test fails when a new migration is added without being mapped.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _force_include() -> dict[str, str]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    mapping = data["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    assert isinstance(mapping, dict)
    return mapping


def test_every_migration_file_is_packaged() -> None:
    mapping = _force_include()
    shipped = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "migrations").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    assert shipped, "no migration files found"
    missing = sorted(shipped - set(mapping))
    assert not missing, f"add these to [tool.hatch.build.targets.wheel.force-include]: {missing}"


def test_packaged_layout_matches_the_cli_fallback() -> None:
    mapping = _force_include()
    assert mapping["alembic.ini"] == "docassist/_alembic/alembic.ini"
    for source, target in mapping.items():
        if source.startswith("migrations/"):
            assert target == "docassist/_alembic/" + source
    cli = (ROOT / "src" / "docassist" / "cli.py").read_text(encoding="utf-8")
    assert '"_alembic" / "alembic.ini"' in cli
