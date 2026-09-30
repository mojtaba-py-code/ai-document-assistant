"""Documents wiring: storage, scanner selection and the ``malware_scanner`` override."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from docassist.api.container import build_egress_policy, build_key_ring
from docassist.documents.scanning import ClamAvScanner, NullScanner
from docassist.documents.service import DocumentService
from docassist.documents.storage import LocalEncryptedStorage
from docassist.documents.wiring import build_malware_scanner, wire
from tests.conftest import make_settings


def _fake_container(tmp_path: Path, overrides: dict[str, Any], **settings: Any) -> Any:
    config = make_settings(storage={"root": str(tmp_path / "storage")}, **settings)
    return SimpleNamespace(
        settings=config,
        ring=build_key_ring(config),
        egress=build_egress_policy(config),
        db=object(),
        audit=object(),
        limiter=object(),
        overrides=overrides,
    )


def test_scanner_selection() -> None:
    assert isinstance(build_malware_scanner(make_settings()), NullScanner)
    clamav = build_malware_scanner(
        make_settings(
            upload={"malware_scanner": "clamav", "clamav_host": "av", "clamav_port": 3311}
        )
    )
    assert isinstance(clamav, ClamAvScanner) and clamav.name == "clamav"


def test_wire_sets_storage_and_service(tmp_path: Path) -> None:
    container = _fake_container(tmp_path, {})
    wire(container)
    assert isinstance(container.storage, LocalEncryptedStorage)
    assert (tmp_path / "storage").is_dir()
    assert isinstance(container.documents, DocumentService)
    assert isinstance(container.documents.malware_scanner, NullScanner)


def test_malware_scanner_override_wins(tmp_path: Path) -> None:
    fake = NullScanner()
    container = _fake_container(
        tmp_path, {"malware_scanner": fake}, upload={"malware_scanner": "clamav"}
    )
    wire(container)
    assert container.documents.malware_scanner is fake
