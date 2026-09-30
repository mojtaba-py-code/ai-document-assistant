"""Service wiring for the documents area.

Sets ``container.storage`` (encrypted object storage) and ``container.documents``.
Honoured overrides: ``"malware_scanner"`` (any :class:`MalwareScanner`) and
``"url_import_transport"`` (an ``httpx`` transport placed *inside* the SSRF guard, for tests).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from docassist.core.config import Settings
from docassist.documents.scanning import ClamAvScanner, MalwareScanner, NullScanner
from docassist.documents.service import DocumentService
from docassist.documents.storage import LocalEncryptedStorage

if TYPE_CHECKING:
    from docassist.api.container import Container


def build_malware_scanner(settings: Settings) -> MalwareScanner:
    """``clamav`` -> :class:`ClamAvScanner` (fails closed); otherwise :class:`NullScanner`."""
    upload = settings.upload
    if upload.malware_scanner == "clamav":
        return ClamAvScanner(upload.clamav_host, upload.clamav_port, upload.clamav_timeout_seconds)
    return NullScanner()


def wire(container: Container) -> None:
    """Attach this area's services to the container."""
    settings = container.settings
    container.storage = LocalEncryptedStorage(settings.storage.root, container.ring)
    scanner: MalwareScanner = container.overrides.get("malware_scanner") or build_malware_scanner(
        settings
    )
    container.documents = DocumentService(container, malware_scanner=scanner)
