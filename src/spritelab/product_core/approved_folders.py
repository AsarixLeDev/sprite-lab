"""Opaque, session-bound approvals for read-only dataset folder access."""

from __future__ import annotations

import hashlib
import os
import re
import secrets
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from spritelab.product_core.contracts import ProjectContext

APPROVED_FOLDER_SCHEMA = "spritelab.product.approved-folder.v1"
APPROVAL_SOURCES = frozenset({"native_picker", "project_import_root", "explicit_launch_allowlist"})
DRIVE_RELATIVE_PATTERN = re.compile(r"^[A-Za-z]:[^\\/]")
DEVICE_PREFIXES = ("\\\\?\\", "\\\\.\\", "\\??\\")


class ApprovedFolderError(ValueError):
    """A folder approval or use attempt violated the local access boundary."""


@dataclass(frozen=True)
class ApprovedFolder:
    approval_id: str
    canonical_path: Path
    project_id: str
    created_at: str
    expires_at: str | None
    source: str
    read_only: bool
    session_id: str
    confinement_root: Path | None = None
    schema_version: str = APPROVED_FOLDER_SCHEMA

    def public_dict(self) -> dict[str, object]:
        """Expose the opaque identity and friendly name, never the server path."""

        return {
            "schema_version": self.schema_version,
            "approval_id": self.approval_id,
            "folder_name": self.canonical_path.name or "Selected folder",
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "source": self.source,
            "read_only": self.read_only,
        }


def project_identity(context: ProjectContext) -> str:
    material = os.path.normcase(str(context.project_root.resolve())).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def interactive_desktop_available() -> bool:
    if sys.platform == "win32":
        session = os.environ.get("SESSIONNAME", "").strip().casefold()
        return bool(session) and session not in {"service", "services"}
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def choose_native_folder() -> str | None:
    """Open a native chooser only when explicitly called by the loopback API."""

    if not interactive_desktop_available():
        raise ApprovedFolderError("Native folder selection is unavailable in this desktop session.")
    try:
        import tkinter
        from tkinter import filedialog
    except ImportError as exc:
        raise ApprovedFolderError("Native folder selection is not installed in this Python environment.") from exc
    root = tkinter.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory(title="Choose image folder", mustexist=True)
    finally:
        root.destroy()
    return str(selected) if selected else None


class ApprovedFolderStore:
    """Keep approvals in one application session's server-side memory."""

    def __init__(
        self,
        context: ProjectContext,
        *,
        session_id: str | None = None,
        import_roots: Iterable[Path] = (),
        launch_allowlist: Iterable[Path] = (),
        allow_network_paths: bool = False,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.context = context
        self.project_id = project_identity(context)
        self.session_id = session_id or secrets.token_urlsafe(24)
        self.allow_network_paths = allow_network_paths
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self._records: dict[str, ApprovedFolder] = {}
        self.import_roots = tuple(self._canonicalize(path, require_directory=True) for path in import_roots)
        for path in launch_allowlist:
            self.approve(path, source="explicit_launch_allowlist")

    def _canonicalize(
        self,
        raw_path: str | Path,
        *,
        require_directory: bool,
        confinement_root: Path | None = None,
    ) -> Path:
        raw = str(raw_path)
        if not raw or "\x00" in raw:
            raise ApprovedFolderError("The selected folder path is malformed.")
        normalized = raw.replace("/", "\\") if os.name == "nt" else raw
        lowered = normalized.casefold()
        if any(lowered.startswith(prefix.casefold()) for prefix in DEVICE_PREFIXES):
            raise ApprovedFolderError("Windows device paths cannot be approved.")
        if os.name == "nt" and normalized.startswith("\\\\") and not self.allow_network_paths:
            raise ApprovedFolderError("Network and UNC folders are disabled by local folder policy.")
        if DRIVE_RELATIVE_PATTERN.match(normalized):
            raise ApprovedFolderError("Drive-relative paths are not valid folder selections.")
        candidate = Path(raw).expanduser()
        if any(part == ".." for part in candidate.parts):
            raise ApprovedFolderError("Folder traversal is not allowed.")
        if not candidate.is_absolute():
            raise ApprovedFolderError("Folder approvals require a canonical absolute path.")
        try:
            canonical = candidate.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ApprovedFolderError("The selected folder does not exist or cannot be read.") from exc
        if require_directory and not canonical.is_dir():
            raise ApprovedFolderError("The selected path is not a folder.")
        if confinement_root is not None:
            root = confinement_root.resolve(strict=True)
            try:
                canonical.relative_to(root)
            except ValueError as exc:
                raise ApprovedFolderError("The selected folder escapes its configured import root.") from exc
        return canonical

    def approve(
        self,
        path: str | Path,
        *,
        source: str,
        ttl_seconds: float | None = None,
        confinement_root: Path | None = None,
    ) -> ApprovedFolder:
        if source not in APPROVAL_SOURCES:
            raise ApprovedFolderError(f"Unsupported folder approval source: {source}")
        root = confinement_root.resolve(strict=True) if confinement_root is not None else None
        canonical = self._canonicalize(path, require_directory=True, confinement_root=root)
        now = self.clock().astimezone(timezone.utc)
        expires = now + timedelta(seconds=float(ttl_seconds)) if ttl_seconds is not None else None
        record = ApprovedFolder(
            approval_id=secrets.token_urlsafe(32),
            canonical_path=canonical,
            project_id=self.project_id,
            created_at=now.isoformat(),
            expires_at=expires.isoformat() if expires else None,
            source=source,
            read_only=True,
            session_id=self.session_id,
            confinement_root=root,
        )
        self._records[record.approval_id] = record
        return record

    def approve_import_root_child(self, root_index: int, relative: str) -> ApprovedFolder:
        if not 0 <= root_index < len(self.import_roots):
            raise ApprovedFolderError("The configured import root is unavailable.")
        root = self.import_roots[root_index]
        relative_path = Path(relative)
        if relative_path.is_absolute() or any(part in {"", ".", ".."} for part in relative_path.parts):
            raise ApprovedFolderError("Import-root selections must use a safe relative folder name.")
        return self.approve(
            root / relative_path,
            source="project_import_root",
            confinement_root=root,
        )

    def resolve(self, approval_id: str, *, project_id: str | None = None) -> Path:
        if not isinstance(approval_id, str) or not approval_id or len(approval_id) > 256:
            raise ApprovedFolderError("The approved-folder ID is malformed.")
        record = self._records.get(approval_id)
        if record is None:
            raise ApprovedFolderError("This folder is not approved for the current Sprite Lab session.")
        if record.session_id != self.session_id:
            raise ApprovedFolderError("This folder approval belongs to another server session.")
        expected_project = project_id or self.project_id
        if record.project_id != expected_project or expected_project != self.project_id:
            raise ApprovedFolderError("This folder approval belongs to another project.")
        if record.expires_at:
            expires = datetime.fromisoformat(record.expires_at.replace("Z", "+00:00"))
            if self.clock().astimezone(timezone.utc) >= expires:
                self._records.pop(approval_id, None)
                raise ApprovedFolderError("This folder approval expired. Choose the folder again.")
        canonical = self._canonicalize(
            record.canonical_path,
            require_directory=True,
            confinement_root=record.confinement_root,
        )
        if canonical != record.canonical_path:
            raise ApprovedFolderError("The approved folder identity changed. Choose the folder again.")
        return canonical

    def record(self, approval_id: str) -> ApprovedFolder | None:
        return self._records.get(approval_id)

    def public_records(self) -> list[dict[str, object]]:
        """List only opaque approval metadata; canonical paths remain server-private."""

        values: list[dict[str, object]] = []
        for approval_id in tuple(self._records):
            try:
                self.resolve(approval_id)
            except ApprovedFolderError:
                continue
            values.append(self._records[approval_id].public_dict())
        return values


__all__ = [
    "APPROVED_FOLDER_SCHEMA",
    "ApprovedFolder",
    "ApprovedFolderError",
    "ApprovedFolderStore",
    "choose_native_folder",
    "interactive_desktop_available",
    "project_identity",
]
