"""Fail-closed helpers for destructive and replacement filesystem operations."""

from __future__ import annotations

import os
import stat
import uuid
from collections.abc import Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class UnsafeFilesystemOperation(ValueError):
    """Raised when a filesystem mutation cannot prove its target is confined."""


class ExactPublicationUnsupported(UnsafeFilesystemOperation):
    """Raised before mutation when an exact held-inode primitive is unavailable."""


@dataclass(frozen=True)
class OwnedFileIdentity:
    """Stable identity used to condition cleanup of an operation-owned file."""

    device: int
    inode: int
    file_type: int

    @classmethod
    def from_stat(cls, metadata: os.stat_result) -> OwnedFileIdentity:
        return cls(
            device=metadata.st_dev,
            inode=metadata.st_ino,
            file_type=stat.S_IFMT(metadata.st_mode),
        )

    def matches(self, metadata: os.stat_result) -> bool:
        return (
            metadata.st_dev == self.device
            and metadata.st_ino == self.inode
            and stat.S_IFMT(metadata.st_mode) == self.file_type
            and not _metadata_is_link_or_reparse(metadata)
        )


class AnchoredDirectory(AbstractContextManager["AnchoredDirectory"]):
    """Hold one mutable parent stable and address children relative to it.

    POSIX operations use a no-follow directory descriptor and ``*at`` APIs, so
    a renamed path cannot redirect a child mutation. Windows operations use a
    verified directory handle and native handle-relative NT calls. A platform
    may still permit renaming the visible parent path while it is held; child
    operations remain bound to the originally opened directory either way.
    """

    def __init__(self, directory: str | Path, root: str | Path) -> None:
        self.directory = require_confined_path(directory, root, allow_root=True)
        self.root = _absolute(root)
        self._before: os.stat_result | None = None
        self._descriptor: int | None = None
        self._windows_handle: int | None = None
        self._parent_anchor: AnchoredDirectory | None = None
        self._entry_name: str | None = None
        self._detached = False
        self._rename_capable = True

    def __enter__(self) -> AnchoredDirectory:
        if self._detached:
            self.verify()
            return self
        before = self.directory.lstat()
        _require_safe_directory(before, self.directory)
        if self.directory.is_mount() and self.directory != self.root:
            raise UnsafeFilesystemOperation(f"mutable parent may not be a mount point: {self.directory}")
        self._before = before
        if os.name == "nt":
            self._windows_handle = _open_windows_directory_handle(self.directory, before)
        else:
            flags = os.O_RDONLY | int(getattr(os, "O_DIRECTORY", 0)) | int(getattr(os, "O_NOFOLLOW", 0))
            descriptor = os.open(self.directory, flags)
            try:
                _require_same_directory(before, os.fstat(descriptor), self.directory)
                _require_same_directory(before, self.directory.lstat(), self.directory)
            except BaseException:
                os.close(descriptor)
                raise
            self._descriptor = descriptor
        self.verify()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        del exc_value, traceback
        verification_error: BaseException | None = None
        try:
            self.verify()
        except BaseException as error:
            verification_error = error
        finally:
            if self._descriptor is not None:
                os.close(self._descriptor)
                self._descriptor = None
            if self._windows_handle is not None:
                _close_windows_handle(self._windows_handle)
                self._windows_handle = None
        if exc_type is None and verification_error is not None:
            raise verification_error

    def verify(self) -> None:
        if self._before is None:
            raise UnsafeFilesystemOperation("mutable parent anchor is not open")
        if self._descriptor is None and self._windows_handle is None:
            raise UnsafeFilesystemOperation("mutable parent anchor is closed")
        if self._detached:
            current = self._before
        elif self._parent_anchor is not None and self._entry_name is not None:
            self._parent_anchor.verify()
            try:
                current = self._parent_anchor.lstat(self._entry_name)
            except FileNotFoundError as exc:
                raise UnsafeFilesystemOperation("mutable child directory disappeared while anchored") from exc
        else:
            try:
                current = self.directory.lstat()
            except FileNotFoundError as exc:
                raise UnsafeFilesystemOperation("mutable parent disappeared while anchored") from exc
        _require_same_directory(self._before, current, self.directory)
        if self._descriptor is not None:
            _require_same_directory(self._before, os.fstat(self._descriptor), self.directory)
        if self._windows_handle is not None:
            _verify_windows_directory_handle(self._windows_handle, self._before, self.directory)

    def detached_duplicate(self) -> AnchoredDirectory:
        """Duplicate the exact held directory without re-resolving its path.

        The duplicate intentionally verifies only the duplicated operating-
        system handle. This is useful when a caller must retain an inode-bound
        parent beyond the lifetime of the lexical root chain that opened it.
        """

        self.verify()
        if self._before is None:
            raise UnsafeFilesystemOperation("mutable parent anchor is not open")
        duplicate = object.__new__(AnchoredDirectory)
        duplicate.directory = self.directory
        duplicate.root = self.root
        duplicate._before = self._before
        duplicate._descriptor = None
        duplicate._windows_handle = None
        duplicate._parent_anchor = None
        duplicate._entry_name = None
        duplicate._detached = True
        duplicate._rename_capable = self._rename_capable
        if self._descriptor is not None:
            duplicate._descriptor = os.dup(self._descriptor)
        elif self._windows_handle is not None:
            duplicate._windows_handle = _duplicate_windows_handle(self._windows_handle)
        else:
            raise UnsafeFilesystemOperation("mutable parent anchor is not open")
        try:
            duplicate.verify()
        except BaseException as error:
            duplicate.__exit__(type(error), error, error.__traceback__)
            raise
        return duplicate

    def directory_metadata(self) -> os.stat_result:
        """Return metadata for the exact held directory after verification."""

        self.verify()
        if self._descriptor is not None:
            return os.fstat(self._descriptor)
        if self._before is None:
            raise UnsafeFilesystemOperation("mutable parent anchor is not open")
        return self._before

    def fixed_directory_path(self) -> Path:
        """Return a pathname that stays bound to this held directory.

        POSIX exposes the inherited descriptor through the process descriptor
        namespace. Windows keeps the lexical path stable by holding a directory
        handle that deliberately denies delete/rename sharing.
        """

        self.verify()
        if self._descriptor is None:
            if self._windows_handle is None:
                raise UnsafeFilesystemOperation("mutable parent anchor is closed")
            return self.directory
        for namespace in (Path("/proc/self/fd"), Path("/dev/fd")):
            candidate = namespace / str(self._descriptor)
            try:
                metadata = candidate.stat()
            except OSError:
                continue
            if self._before is not None and OwnedFileIdentity.from_stat(metadata) == OwnedFileIdentity.from_stat(
                self._before
            ):
                return candidate
        raise UnsafeFilesystemOperation("this POSIX platform has no stable inherited-descriptor pathname")

    @contextmanager
    def inheritable_token(self) -> Iterator[tuple[str, int]]:
        """Temporarily make the exact held directory inheritable by one child."""

        self.verify()
        if self._descriptor is not None:
            descriptor = self._descriptor
            before = os.get_inheritable(descriptor)
            os.set_inheritable(descriptor, True)
            try:
                yield "posix_fd", descriptor
            finally:
                os.set_inheritable(descriptor, before)
            return
        if self._windows_handle is not None:
            handle = self._windows_handle
            before = os.get_handle_inheritable(handle)
            os.set_handle_inheritable(handle, True)
            try:
                yield "windows_handle", handle
            finally:
                os.set_handle_inheritable(handle, before)
            return
        raise UnsafeFilesystemOperation("mutable parent anchor is closed")

    def lstat(self, name: str) -> os.stat_result:
        child = self._child_name(name)
        if self._descriptor is not None:
            return os.stat(child, dir_fd=self._descriptor, follow_symlinks=False)
        return _windows_relative_stat(self._required_windows_handle(), child)

    def lexists(self, name: str) -> bool:
        try:
            self.lstat(name)
        except FileNotFoundError:
            return False
        return True

    def open_file(self, name: str, flags: int, mode: int = 0o600) -> int:
        child = self._child_name(name)
        if self._descriptor is not None:
            return os.open(
                child,
                flags | int(getattr(os, "O_NOFOLLOW", 0)),
                mode,
                dir_fd=self._descriptor,
            )
        return _open_windows_child(self._required_windows_handle(), child, flags, mode)

    def open_file_immovable(self, name: str, flags: int, mode: int = 0o600) -> int:
        """Open one exact child while denying Windows rename/delete sharing.

        POSIX descriptor-relative opens already pin the selected inode.  On
        Windows this variant additionally omits ``FILE_SHARE_DELETE`` so the
        visible directory entry cannot be exchanged while a caller validates
        and consumes the held descriptor.
        """

        child = self._child_name(name)
        if self._descriptor is not None:
            return os.open(
                child,
                flags | int(getattr(os, "O_NOFOLLOW", 0)),
                mode,
                dir_fd=self._descriptor,
            )
        return _open_windows_child(
            self._required_windows_handle(),
            child,
            flags,
            mode,
            share_delete=False,
        )

    def open_anonymous_file(self, mode: int = 0o600) -> int:
        """Create an unnamed regular file relative to this held directory.

        This is available only where the operating system provides a genuine
        unnamed-file primitive. Callers must fail closed or retain an explicit
        named evidence file when it is unsupported.
        """

        if self._descriptor is None:
            raise UnsafeFilesystemOperation("anonymous anchored files are unsupported on this platform")
        temporary_flag = int(getattr(os, "O_TMPFILE", 0))
        if temporary_flag == 0:
            raise UnsafeFilesystemOperation("anonymous anchored files are unsupported on this platform")
        return os.open(
            ".",
            os.O_RDWR | temporary_flag | int(getattr(os, "O_BINARY", 0)),
            mode,
            dir_fd=self._descriptor,
        )

    def link(self, source_name: str, destination_name: str) -> None:
        source = self._child_name(source_name)
        destination = self._child_name(destination_name)
        if self._descriptor is not None:
            os.link(
                source,
                destination,
                src_dir_fd=self._descriptor,
                dst_dir_fd=self._descriptor,
                follow_symlinks=False,
            )
        else:
            _windows_link_child(self._required_windows_handle(), source, destination)

    def replace(self, source_name: str, destination_name: str) -> None:
        self.rename(source_name, destination_name, replace=True)

    def rename(self, source_name: str, destination_name: str, *, replace: bool) -> None:
        source = self._child_name(source_name)
        destination = self._child_name(destination_name)
        if self._descriptor is not None:
            if replace:
                os.replace(source, destination, src_dir_fd=self._descriptor, dst_dir_fd=self._descriptor)
            else:
                _posix_rename_noreplace(self._descriptor, source, destination)
        else:
            _windows_replace_child(self._required_windows_handle(), source, destination, replace=replace)
        self._sync_directory()

    def publish_held_file_no_replace(
        self,
        source_descriptor: int,
        source_name: str | None,
        destination_name: str,
        *,
        identity: OwnedFileIdentity,
    ) -> None:
        """Publish the exact held regular-file inode without replacing a name.

        Windows renames a freshly re-opened exact source handle while denying
        delete sharing. POSIX links directly from the held descriptor. The
        latter intentionally leaves a named source as a second hard link;
        callers that require a single-link artifact should stage anonymously
        where ``O_TMPFILE`` is available or retain and validate that alias.
        """

        destination = self._child_name(destination_name)
        source = self._child_name(source_name) if source_name is not None else None
        self._validate_held_file(source_descriptor, source, identity)
        if self.lexists(destination):
            raise FileExistsError(self.directory / destination)
        if self._windows_handle is not None:
            if source is None:
                raise ExactPublicationUnsupported("Windows held-file publication requires a named source")
            publish_descriptor = _open_windows_publishable_child(self._required_windows_handle(), source)
            try:
                self._validate_held_file(publish_descriptor, source, identity)
                _windows_rename_descriptor(
                    publish_descriptor,
                    self._required_windows_handle(),
                    destination,
                    replace=False,
                )
            finally:
                os.close(publish_descriptor)
        elif self._descriptor is not None:
            _posix_link_descriptor_noreplace(source_descriptor, self._descriptor, destination)
        else:
            raise UnsafeFilesystemOperation("mutable parent anchor is not open")
        if not identity.matches(self.lstat(destination)):
            raise UnsafeFilesystemOperation("held-file publication changed inode identity")
        self._sync_directory()

    def replace_held_file_if_owned(
        self,
        source_descriptor: int,
        source_name: str,
        destination_name: str,
        *,
        identity: OwnedFileIdentity,
        destination_descriptor: int,
        destination_identity: OwnedFileIdentity,
    ) -> None:
        """Replace one exact held destination with one exact held source.

        Windows implements this as two exact handle renames: the prior inode is
        first moved to an unpredictable recovery name, then the source handle
        is published no-replace. The prior handle is restored or retained as a
        recovery residue on failure. Portable POSIX has no rename-by-file-
        descriptor primitive, so the exact CAS contract fails before mutation.
        Generic atomic writes use a separately documented recovery fallback.
        """

        source = self._child_name(source_name)
        destination = self._child_name(destination_name)
        self._validate_held_file(source_descriptor, source, identity)
        self._validate_held_file(destination_descriptor, destination, destination_identity)
        if self._windows_handle is None:
            raise ExactPublicationUnsupported("exact held-file replacement is unsupported on this platform")

        parent_handle = self._required_windows_handle()
        source_publish = _open_windows_publishable_child(parent_handle, source)
        destination_publish = _open_windows_publishable_child(parent_handle, destination)
        recovery = f".spritelab-recovery-{uuid.uuid4().hex}"
        moved_destination = False
        published_source = False
        try:
            self._validate_held_file(source_publish, source, identity)
            self._validate_held_file(destination_publish, destination, destination_identity)
            _windows_rename_descriptor(destination_publish, parent_handle, recovery, replace=False)
            moved_destination = True
            if not destination_identity.matches(self.lstat(recovery)):
                raise UnsafeFilesystemOperation("held destination changed while entering recovery")
            _windows_rename_descriptor(source_publish, parent_handle, destination, replace=False)
            published_source = True
            if not identity.matches(self.lstat(destination)):
                raise UnsafeFilesystemOperation("held source changed during exact replacement")
            _windows_delete_descriptor(destination_publish, destination_identity)
        except BaseException:
            if moved_destination and not published_source and not self.lexists(destination):
                try:
                    _windows_rename_descriptor(destination_publish, parent_handle, destination, replace=False)
                    moved_destination = False
                except BaseException:
                    pass
            raise
        finally:
            os.close(destination_publish)
            os.close(source_publish)
        self._sync_directory()

    def _validate_held_file(
        self,
        descriptor: int,
        source_name: str | None,
        identity: OwnedFileIdentity,
    ) -> os.stat_result:
        self.verify()
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or not identity.matches(opened):
            raise UnsafeFilesystemOperation("held publication descriptor identity changed")
        if source_name is not None:
            current = self.lstat(source_name)
            if not _same_held_file(opened, current) or not identity.matches(current):
                raise UnsafeFilesystemOperation("held publication source name changed")
        return opened

    def rename_held_directory_noreplace(
        self,
        child: AnchoredDirectory,
        destination_name: str,
    ) -> None:
        """Rename an open direct-child directory and keep its anchor valid.

        Windows renames the exact held directory handle. POSIX performs an
        exclusive parent-relative rename and then requires the destination to
        retain the held inode identity. No destination is ever replaced.
        """

        self.verify()
        child.verify()
        destination = self._child_name(destination_name)
        if child._parent_anchor is not self or child._entry_name is None or child._detached:
            raise UnsafeFilesystemOperation("held directory is not a direct child of this anchor")
        if not child._rename_capable:
            raise UnsafeFilesystemOperation("held directory was opened as an immovable anchor")
        identity = OwnedFileIdentity.from_stat(child.directory_metadata())
        if self._windows_handle is not None:
            _set_windows_handle_name(
                child._required_windows_handle(),
                self._required_windows_handle(),
                destination,
                information_class=10,  # FileRenameInformation
                replace=False,
            )
        elif self._descriptor is not None:
            raise ExactPublicationUnsupported("exact held-directory publication is unsupported on this POSIX platform")
        else:
            raise UnsafeFilesystemOperation("mutable parent anchor is not open")
        moved = self.lstat(destination)
        if not identity.matches(moved):
            raise UnsafeFilesystemOperation("held directory identity changed during exclusive rename")
        child._entry_name = destination
        child.directory = self.directory / destination
        child.verify()

    def mkdir(
        self,
        name: str,
        mode: int = 0o700,
        *,
        exist_ok: bool = False,
    ) -> OwnedFileIdentity:
        """Create one exact direct-child directory relative to the held parent."""

        child = self._child_name(name)
        try:
            if self._descriptor is not None:
                os.mkdir(child, mode, dir_fd=self._descriptor)
            else:
                _windows_create_directory(self._required_windows_handle(), child)
        except FileExistsError:
            if not exist_ok:
                raise
        metadata = self.lstat(child)
        _require_safe_directory(metadata, self.directory / child)
        self._sync_directory()
        return OwnedFileIdentity.from_stat(metadata)

    def mkdir_unique(self, prefix: str) -> tuple[str, OwnedFileIdentity]:
        """Create an unpredictable direct child directory exclusively."""

        safe_prefix = self._child_name(prefix)
        for _attempt in range(16):
            name = f"{safe_prefix}{uuid.uuid4().hex}"
            try:
                if self._descriptor is not None:
                    os.mkdir(name, 0o700, dir_fd=self._descriptor)
                else:
                    _windows_create_directory(self._required_windows_handle(), name)
            except FileExistsError:
                continue
            metadata = self.lstat(name)
            _require_safe_directory(metadata, self.directory / name)
            return name, OwnedFileIdentity.from_stat(metadata)
        raise UnsafeFilesystemOperation("could not allocate a unique anchored directory")

    def names(self) -> tuple[str, ...]:
        """Return deterministic direct-child names from the held directory."""

        if self._descriptor is not None:
            names = os.listdir(self._descriptor)
        else:
            names = _windows_list_directory(self._required_windows_handle())
        if any(not name or name in {".", ".."} or Path(name).name != name for name in names):
            raise UnsafeFilesystemOperation("anchored directory enumeration returned an unsafe child name")
        return tuple(sorted(names, key=lambda item: (item.casefold(), item)))

    @contextmanager
    def _open_directory(self, name: str, *, movable: bool) -> Iterator[AnchoredDirectory]:
        """Open one no-follow direct-child directory relative to this anchor."""

        child_name = self._child_name(name)
        if self._before is None:
            raise UnsafeFilesystemOperation("mutable parent anchor is not open")
        before = self.lstat(child_name)
        _require_safe_directory(before, self.directory / child_name)
        if before.st_dev != self._before.st_dev:
            raise UnsafeFilesystemOperation("anchored child directory crosses a filesystem boundary")
        child = object.__new__(AnchoredDirectory)
        child.directory = self.directory / child_name
        child.root = self.root
        child._before = before
        child._descriptor = None
        child._windows_handle = None
        child._parent_anchor = self
        child._entry_name = child_name
        child._detached = False
        child._rename_capable = movable
        if self._descriptor is not None:
            flags = os.O_RDONLY | int(getattr(os, "O_DIRECTORY", 0)) | int(getattr(os, "O_NOFOLLOW", 0))
            descriptor = os.open(child_name, flags, dir_fd=self._descriptor)
            try:
                _require_same_directory(before, os.fstat(descriptor), child.directory)
            except BaseException:
                os.close(descriptor)
                raise
            child._descriptor = descriptor
        else:
            child._windows_handle = _open_windows_relative_directory_handle(
                self._required_windows_handle(), child_name, before, child.directory, movable=movable
            )
        try:
            child.verify()
            yield child
        except BaseException as error:
            child.__exit__(type(error), error, error.__traceback__)
            raise
        else:
            child.__exit__(None, None, None)

    @contextmanager
    def open_directory(self, name: str) -> Iterator[AnchoredDirectory]:
        """Open a rename-capable direct-child anchor."""

        with self._open_directory(name, movable=True) as child:
            yield child

    @contextmanager
    def open_directory_immovable(self, name: str) -> Iterator[AnchoredDirectory]:
        """Open a fixed direct-child anchor without Windows DELETE sharing.

        Use this for roots that must coexist with a later absolute fixed-root
        handle. The returned anchor cannot be passed to
        :meth:`rename_held_directory_noreplace`.
        """

        with self._open_directory(name, movable=False) as child:
            yield child

    def list_names(self) -> tuple[str, ...]:
        """Compatibility alias for :meth:`names`."""

        return self.names()

    def child_directory(self, name: str) -> AbstractContextManager[AnchoredDirectory]:
        """Compatibility alias for :meth:`open_directory`."""

        return self.open_directory(name)

    def quarantine_if_owned(
        self,
        name: str,
        identity: OwnedFileIdentity,
        *,
        prefix: str,
    ) -> str | None:
        """Move an exact owned entry to a unique residue without deleting it."""

        source = self._child_name(name)
        safe_prefix = self._child_name(prefix)
        for _attempt in range(16):
            try:
                current = self.lstat(source)
            except FileNotFoundError:
                return None
            if not identity.matches(current):
                return None
            destination = f"{safe_prefix}{uuid.uuid4().hex}"
            try:
                self.rename(source, destination, replace=False)
            except FileExistsError:
                continue
            except FileNotFoundError:
                return None
            moved = self.lstat(destination)
            if not identity.matches(moved):
                moved_identity = OwnedFileIdentity.from_stat(moved)
                if self.lexists(source):
                    raise UnsafeFilesystemOperation(
                        "raced quarantine entry was retained as residue because its original name was reused"
                    )
                try:
                    self.rename(destination, source, replace=False)
                except BaseException as exc:
                    raise UnsafeFilesystemOperation(
                        "raced quarantine entry could not be restored and was retained as residue"
                    ) from exc
                if OwnedFileIdentity.from_stat(self.lstat(source)) != moved_identity:
                    raise UnsafeFilesystemOperation("raced quarantine entry changed identity during restoration")
                return None
            return destination
        raise UnsafeFilesystemOperation("could not allocate a unique anchored residue")

    def unlink_if_owned(self, name: str, identity: OwnedFileIdentity, *, missing_ok: bool = True) -> bool:
        """Retire only when the current entry still names the owned inode.

        Windows deletes the quarantined inode through an identity-checked file
        handle. Portable POSIX has no compare-and-unlink primitive, so it keeps
        the unpredictable quarantine residue rather than risk deleting a
        foreign entry substituted after verification.
        """

        child = self._child_name(name)
        if not self.lexists(child):
            if missing_ok:
                return True
            raise FileNotFoundError(self.directory / child)
        residue = self.quarantine_if_owned(
            child,
            identity,
            prefix=f".spritelab-unlink-{uuid.uuid4().hex}-",
        )
        if residue is None:
            return False
        if self._descriptor is not None:
            # A path-based unlink after an identity check has an unavoidable
            # substitution window on POSIX. The original public name is gone;
            # retain the random residue as the fail-closed outcome.
            return True
        else:
            _windows_unlink_child(self._required_windows_handle(), residue, identity)
        self._sync_directory()
        return True

    def atomic_write_bytes(self, name: str, content: bytes) -> Path:
        """Durably replace one cooperative direct-child namespace entry.

        This bounded mutable writer leaves one single-link canonical file and
        no successful staging aliases. It is not an adversarial compare-and-
        swap authority on POSIX; audit-bound publication must use
        :meth:`publish_held_file_no_replace` or immutable commit records.
        """

        target = self._child_name(name)
        temporary = f".spritelab-{uuid.uuid4().hex}.tmp"
        descriptor = self.open_file(
            temporary,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | int(getattr(os, "O_BINARY", 0)),
        )
        identity = OwnedFileIdentity.from_stat(os.fstat(descriptor))
        destination_descriptor = -1
        try:
            writer_descriptor = os.dup(descriptor)
            try:
                handle = os.fdopen(writer_descriptor, "wb")
                writer_descriptor = -1
                with handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                    written_identity = OwnedFileIdentity.from_stat(os.fstat(handle.fileno()))
                    if written_identity != identity:
                        raise UnsafeFilesystemOperation("atomic temporary descriptor identity changed while writing")
            finally:
                if writer_descriptor >= 0:
                    os.close(writer_descriptor)
            if not identity.matches(self.lstat(temporary)):
                raise UnsafeFilesystemOperation("atomic temporary path changed before publication")
            if self._windows_handle is not None and self.lexists(target):
                destination_descriptor = self.open_file(
                    target,
                    os.O_RDONLY | int(getattr(os, "O_BINARY", 0)),
                )
                destination_identity = OwnedFileIdentity.from_stat(os.fstat(destination_descriptor))
                self.replace_held_file_if_owned(
                    descriptor,
                    temporary,
                    target,
                    identity=identity,
                    destination_descriptor=destination_descriptor,
                    destination_identity=destination_identity,
                )
            elif self._windows_handle is not None:
                self.publish_held_file_no_replace(descriptor, temporary, target, identity=identity)
            else:
                self._validate_held_file(descriptor, temporary, identity)
                self.replace(temporary, target)
            if not identity.matches(self.lstat(target)):
                raise UnsafeFilesystemOperation("atomic publication changed inode identity")
        except BaseException:
            self.unlink_if_owned(temporary, identity)
            raise
        finally:
            if destination_descriptor >= 0:
                os.close(destination_descriptor)
            os.close(descriptor)
        return self.directory / target

    def _sync_directory(self) -> None:
        if self._descriptor is not None:
            os.fsync(self._descriptor)

    def _required_windows_handle(self) -> int:
        if self._windows_handle is None:
            raise UnsafeFilesystemOperation("mutable parent anchor is not open")
        return self._windows_handle

    @staticmethod
    def _child_name(name: str) -> str:
        if not name or name in {".", ".."} or Path(name).name != name or "/" in name or "\\" in name:
            raise UnsafeFilesystemOperation("anchored mutation requires one exact child name")
        return name


def require_confined_path(
    path: str | Path,
    root: str | Path,
    *,
    allow_root: bool = False,
) -> Path:
    """Return a lexical absolute path only when it is safely below ``root``.

    Both lexical and resolved containment are checked. Existing descendant
    components may not be symbolic links or Windows reparse points. The root is
    treated as the caller-approved boundary and may itself resolve elsewhere.
    """

    root_path = _absolute(root)
    target = _absolute(path)
    try:
        relative = target.relative_to(root_path)
    except ValueError as exc:
        raise UnsafeFilesystemOperation(f"target escapes its approved root: {target}") from exc
    if not relative.parts and not allow_root:
        raise UnsafeFilesystemOperation(f"refusing to mutate the approved root itself: {root_path}")

    resolved_root = root_path.resolve(strict=False)
    resolved_target = target.resolve(strict=False)
    try:
        resolved_relative = resolved_target.relative_to(resolved_root)
    except ValueError as exc:
        raise UnsafeFilesystemOperation(f"resolved target escapes its approved root: {target}") from exc
    if not resolved_relative.parts and not allow_root:
        raise UnsafeFilesystemOperation(f"refusing to mutate the approved root itself: {root_path}")

    current = root_path
    for part in relative.parts:
        current = current / part
        if not _lexists(current):
            break
        if _is_link_or_reparse_point(current):
            raise UnsafeFilesystemOperation(f"target crosses a link or reparse point: {current}")
    return target


def remove_confined_tree(path: str | Path, root: str | Path, *, missing_ok: bool = False) -> None:
    """Retire one exact directory without recursively deleting through paths.

    Portable filesystems do not expose a compare-and-recursively-delete
    primitive. The exact owned entry is therefore moved to an unpredictable
    hidden residue below the same held parent. Callers lose the public target
    name while recovery bytes remain available for explicit audited cleanup.
    """

    target = require_confined_path(path, root)
    if not _lexists(target):
        if missing_ok:
            return
        raise FileNotFoundError(target)
    if _is_link_or_reparse_point(target):
        raise UnsafeFilesystemOperation(f"refusing to retire a linked path: {target}")
    if target.is_mount():
        raise UnsafeFilesystemOperation(f"refusing to retire a mount point: {target}")
    with open_anchored_directory(target.parent, root) as parent:
        try:
            metadata = parent.lstat(target.name)
        except FileNotFoundError:
            if missing_ok:
                return
            raise
        _require_safe_directory(metadata, target)
        parent_metadata = parent.directory_metadata()
        if metadata.st_dev != parent_metadata.st_dev:
            raise UnsafeFilesystemOperation(f"refusing to retire a filesystem boundary: {target}")
        identity = OwnedFileIdentity.from_stat(metadata)
        residue = parent.quarantine_if_owned(
            target.name,
            identity,
            prefix=".spritelab-retired-tree-",
        )
        if residue is None:
            raise UnsafeFilesystemOperation(f"directory identity changed before retirement: {target}")
        if not identity.matches(parent.lstat(residue)):
            raise UnsafeFilesystemOperation(f"retired directory identity changed unexpectedly: {target}")


def atomic_write_bytes(path: str | Path, content: bytes) -> Path:
    """Atomically replace one file via an exclusive unpredictable sibling."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with AnchoredDirectory(target.parent, target.parent) as parent:
        parent.atomic_write_bytes(target.name, content)
    return target


def atomic_write_text(path: str | Path, content: str, *, encoding: str = "utf-8") -> Path:
    """Encode and atomically replace one text file."""

    return atomic_write_bytes(path, content.encode(encoding))


@contextmanager
def open_anchored_directory(
    directory: str | Path,
    root: str | Path,
) -> Iterator[AnchoredDirectory]:
    """Open an existing descendant through a held no-follow root chain."""

    root_path = _absolute(root)
    target = require_confined_path(directory, root_path, allow_root=True)
    relative = target.relative_to(root_path)
    with ExitStack() as stack:
        anchor = stack.enter_context(AnchoredDirectory(root_path, root_path))
        for part in relative.parts:
            # Lexical root descent is a fixed trust chain, never a held child
            # prepared for rename. On Windows, omitting DELETE/SHARE_DELETE
            # lets it coexist with repository mutation-lock handles.
            anchor = stack.enter_context(anchor.open_directory_immovable(part))
        yield anchor


def _absolute(path: str | Path) -> Path:
    value = os.fspath(path)
    if not value or not value.strip():
        raise UnsafeFilesystemOperation("filesystem target must not be empty")
    return Path(os.path.abspath(os.path.expanduser(value)))


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _is_link_or_reparse_point(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(metadata.st_mode):
        return True
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & reparse_flag)


def _metadata_is_link_or_reparse(metadata: os.stat_result) -> bool:
    if stat.S_ISLNK(metadata.st_mode):
        return True
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & reparse_flag)


def _require_safe_directory(metadata: os.stat_result, path: Path) -> None:
    if not stat.S_ISDIR(metadata.st_mode) or _metadata_is_link_or_reparse(metadata):
        raise UnsafeFilesystemOperation(f"mutable parent is linked or not a directory: {path}")


def _require_same_directory(before: os.stat_result, after: os.stat_result, path: Path) -> None:
    _require_safe_directory(after, path)
    if after.st_dev != before.st_dev or after.st_ino != before.st_ino:
        raise UnsafeFilesystemOperation(f"mutable parent identity changed while anchored: {path}")


def _same_held_file(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev,
        left.st_ino,
        stat.S_IFMT(left.st_mode),
        left.st_nlink,
        left.st_size,
        left.st_mtime_ns,
        _metadata_is_link_or_reparse(left),
    ) == (
        right.st_dev,
        right.st_ino,
        stat.S_IFMT(right.st_mode),
        right.st_nlink,
        right.st_size,
        right.st_mtime_ns,
        _metadata_is_link_or_reparse(right),
    )


def _open_windows_directory_handle(path: Path, before: os.stat_result) -> int:
    import ctypes
    from ctypes import wintypes

    create_file = ctypes.WinDLL("kernel32", use_last_error=True).CreateFileW
    create_file.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create_file.restype = wintypes.HANDLE
    handle = create_file(
        str(path),
        0x00000001 | 0x80,  # FILE_LIST_DIRECTORY | FILE_READ_ATTRIBUTES
        0x00000001 | 0x00000002,  # FILE_SHARE_READ | FILE_SHARE_WRITE; deliberately no DELETE
        None,
        3,  # OPEN_EXISTING
        0x02000000 | 0x00200000,  # BACKUP_SEMANTICS | OPEN_REPARSE_POINT
        None,
    )
    invalid = ctypes.c_void_p(-1).value
    if handle == invalid:
        raise OSError(ctypes.get_last_error(), f"could not anchor mutable parent: {path}")
    numeric = int(handle)
    try:
        _verify_windows_directory_handle(numeric, before, path)
        _require_same_directory(before, path.lstat(), path)
    except BaseException:
        _close_windows_handle(numeric)
        raise
    return numeric


def _open_windows_relative_directory_handle(
    parent_handle: int,
    name: str,
    before: os.stat_result,
    path: Path,
    *,
    movable: bool,
) -> int:
    handle = _nt_open_relative(
        parent_handle,
        name,
        # FILE_LIST_DIRECTORY | optional DELETE | READ_ATTRIBUTES | SYNCHRONIZE.
        desired_access=0x00000001 | (0x00010000 if movable else 0) | 0x80 | 0x00100000,
        disposition=1,  # FILE_OPEN
        options=0x00200000 | 0x00000001 | 0x00000020,
        share_access=0x00000001 | 0x00000002 | (0x00000004 if movable else 0),
    )
    try:
        _verify_windows_directory_handle(handle, before, path)
    except BaseException:
        _close_windows_handle(handle)
        raise
    return handle


def _verify_windows_directory_handle(handle: int, before: os.stat_result, path: Path) -> None:
    attributes, file_index = _windows_handle_information(handle)
    if not attributes & 0x10 or attributes & 0x400:
        raise UnsafeFilesystemOperation(f"mutable parent handle is linked or not a directory: {path}")
    if file_index != before.st_ino:
        raise UnsafeFilesystemOperation(f"mutable parent handle identity changed while opening: {path}")


def _windows_handle_information(handle: int) -> tuple[int, int]:
    import ctypes
    from ctypes import wintypes

    class _ByHandleFileInformation(ctypes.Structure):
        _fields_ = [
            ("dwFileAttributes", wintypes.DWORD),
            ("ftCreationTime", wintypes.FILETIME),
            ("ftLastAccessTime", wintypes.FILETIME),
            ("ftLastWriteTime", wintypes.FILETIME),
            ("dwVolumeSerialNumber", wintypes.DWORD),
            ("nFileSizeHigh", wintypes.DWORD),
            ("nFileSizeLow", wintypes.DWORD),
            ("nNumberOfLinks", wintypes.DWORD),
            ("nFileIndexHigh", wintypes.DWORD),
            ("nFileIndexLow", wintypes.DWORD),
        ]

    get_information = ctypes.WinDLL("kernel32", use_last_error=True).GetFileInformationByHandle
    get_information.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ByHandleFileInformation)]
    get_information.restype = wintypes.BOOL
    information = _ByHandleFileInformation()
    if not get_information(wintypes.HANDLE(handle), ctypes.byref(information)):
        raise OSError(ctypes.get_last_error(), "could not inspect anchored Windows handle")
    file_index = (int(information.nFileIndexHigh) << 32) | int(information.nFileIndexLow)
    return int(information.dwFileAttributes), file_index


def _duplicate_windows_handle(handle: int) -> int:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_current_process = kernel32.GetCurrentProcess
    get_current_process.argtypes = []
    get_current_process.restype = wintypes.HANDLE
    current_process = get_current_process()
    duplicate_handle = kernel32.DuplicateHandle
    duplicate_handle.argtypes = [
        wintypes.HANDLE,
        wintypes.HANDLE,
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.HANDLE),
        wintypes.DWORD,
        wintypes.BOOL,
        wintypes.DWORD,
    ]
    duplicate_handle.restype = wintypes.BOOL
    duplicated = wintypes.HANDLE()
    if not duplicate_handle(
        current_process,
        wintypes.HANDLE(handle),
        current_process,
        ctypes.byref(duplicated),
        0,
        False,
        0x00000002,  # DUPLICATE_SAME_ACCESS
    ):
        raise OSError(ctypes.get_last_error(), "could not duplicate anchored Windows handle")
    return int(duplicated.value)


def _open_windows_child(
    parent_handle: int,
    name: str,
    flags: int,
    mode: int,
    *,
    share_delete: bool = True,
) -> int:
    del mode
    import msvcrt

    access_mode = flags & (os.O_RDONLY | os.O_WRONLY | os.O_RDWR)
    if access_mode == os.O_RDWR:
        desired_access = 0x80000000 | 0x40000000
    elif access_mode == os.O_WRONLY:
        desired_access = 0x40000000
    else:
        desired_access = 0x80000000
    desired_access |= 0x80 | 0x00100000  # FILE_READ_ATTRIBUTES | SYNCHRONIZE
    if flags & os.O_CREAT and flags & os.O_EXCL:
        disposition = 2  # FILE_CREATE
    elif flags & os.O_CREAT and flags & os.O_TRUNC:
        disposition = 5  # FILE_OVERWRITE_IF
    elif flags & os.O_CREAT:
        disposition = 3  # FILE_OPEN_IF
    elif flags & os.O_TRUNC:
        disposition = 4  # FILE_OVERWRITE
    else:
        disposition = 1  # FILE_OPEN
    handle = _nt_open_relative(
        parent_handle,
        name,
        desired_access=desired_access,
        disposition=disposition,
        options=0x00200000 | 0x00000040 | 0x00000020,
        share_access=0x00000001 | 0x00000002 | (0x00000004 if share_delete else 0),
    )
    try:
        attributes, _file_index = _windows_handle_information(handle)
        if attributes & (0x10 | 0x400):
            raise UnsafeFilesystemOperation(f"anchored file is a directory or reparse point: {name}")
        descriptor_flags = int(getattr(os, "O_BINARY", 0))
        if flags & os.O_APPEND:
            descriptor_flags |= os.O_APPEND
        if access_mode == os.O_RDWR:
            descriptor_flags |= os.O_RDWR
        elif access_mode == os.O_WRONLY:
            descriptor_flags |= os.O_WRONLY
        else:
            descriptor_flags |= os.O_RDONLY
        descriptor = msvcrt.open_osfhandle(handle, descriptor_flags)
        handle = -1
        return descriptor
    finally:
        if handle >= 0:
            _close_windows_handle(handle)


def _open_windows_publishable_child(parent_handle: int, name: str) -> int:
    """Open one regular child for exact handle rename without DELETE sharing."""

    import msvcrt

    handle = _nt_open_relative(
        parent_handle,
        name,
        desired_access=0x00010000 | 0x80 | 0x00100000,  # DELETE | READ_ATTRIBUTES | SYNCHRONIZE
        disposition=1,
        options=0x00200000 | 0x00000040 | 0x00000020,
        share_access=0x00000001 | 0x00000002,
    )
    try:
        attributes, _file_index = _windows_handle_information(handle)
        if attributes & (0x10 | 0x400):
            raise UnsafeFilesystemOperation(f"held publication source is unsafe: {name}")
        descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | int(getattr(os, "O_BINARY", 0)))
        handle = -1
        return descriptor
    finally:
        if handle >= 0:
            _close_windows_handle(handle)


def _windows_rename_descriptor(
    descriptor: int,
    parent_handle: int,
    destination_name: str,
    *,
    replace: bool,
) -> None:
    import msvcrt

    _set_windows_handle_name(
        int(msvcrt.get_osfhandle(descriptor)),
        parent_handle,
        destination_name,
        information_class=10,  # FileRenameInformation
        replace=replace,
    )


def _windows_delete_descriptor(descriptor: int, identity: OwnedFileIdentity) -> None:
    import ctypes
    import msvcrt

    metadata = os.fstat(descriptor)
    if not identity.matches(metadata):
        raise UnsafeFilesystemOperation("held recovery inode changed before deletion")

    class _FileDispositionInfo(ctypes.Structure):
        _fields_ = [("DeleteFile", ctypes.c_ubyte)]

    information = _FileDispositionInfo(True)
    _set_file_information(
        int(msvcrt.get_osfhandle(descriptor)),
        13,
        ctypes.byref(information),
        ctypes.sizeof(information),
    )


def _windows_relative_stat(parent_handle: int, name: str) -> os.stat_result:
    import msvcrt

    handle = _nt_open_relative(
        parent_handle,
        name,
        desired_access=0x80 | 0x00100000,
        disposition=1,
        options=0x00200000 | 0x00000020,
    )
    try:
        attributes, _file_index = _windows_handle_information(handle)
        if attributes & 0x400:
            raise UnsafeFilesystemOperation(f"anchored child is a reparse point: {name}")
        descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | int(getattr(os, "O_BINARY", 0)))
        handle = -1
        try:
            return os.fstat(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if handle >= 0:
            _close_windows_handle(handle)


def _windows_list_directory(handle: int) -> list[str]:
    import ctypes
    from ctypes import wintypes

    class _FileIdBothDirectoryInfo(ctypes.Structure):
        _fields_ = [
            ("NextEntryOffset", wintypes.DWORD),
            ("FileIndex", wintypes.DWORD),
            ("CreationTime", ctypes.c_longlong),
            ("LastAccessTime", ctypes.c_longlong),
            ("LastWriteTime", ctypes.c_longlong),
            ("ChangeTime", ctypes.c_longlong),
            ("EndOfFile", ctypes.c_longlong),
            ("AllocationSize", ctypes.c_longlong),
            ("FileAttributes", wintypes.DWORD),
            ("FileNameLength", wintypes.DWORD),
            ("EaSize", wintypes.DWORD),
            ("ShortNameLength", ctypes.c_ubyte),
            ("ShortName", wintypes.WCHAR * 12),
            ("FileId", ctypes.c_longlong),
            ("FileName", wintypes.WCHAR * 1),
        ]

    get_information = ctypes.WinDLL("kernel32", use_last_error=True).GetFileInformationByHandleEx
    get_information.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
    get_information.restype = wintypes.BOOL
    buffer_size = 64 * 1024
    buffer = ctypes.create_string_buffer(buffer_size)
    names: list[str] = []
    information_class = 11  # FileIdBothDirectoryRestartInfo
    while True:
        if not get_information(wintypes.HANDLE(handle), information_class, buffer, buffer_size):
            error = ctypes.get_last_error()
            if error == 18:  # ERROR_NO_MORE_FILES
                break
            raise OSError(error, "could not enumerate anchored Windows directory")
        information_class = 10  # FileIdBothDirectoryInfo
        offset = 0
        while True:
            address = ctypes.addressof(buffer) + offset
            record = _FileIdBothDirectoryInfo.from_address(address)
            filename_address = address + _FileIdBothDirectoryInfo.FileName.offset
            raw_name = ctypes.string_at(filename_address, int(record.FileNameLength))
            name = raw_name.decode("utf-16-le", errors="strict")
            if name not in {".", ".."}:
                names.append(name)
            next_offset = int(record.NextEntryOffset)
            if next_offset == 0:
                break
            if next_offset < _FileIdBothDirectoryInfo.FileName.offset or offset + next_offset >= buffer_size:
                raise UnsafeFilesystemOperation("Windows directory enumeration returned an invalid record")
            offset += next_offset
    return names


def _windows_link_child(parent_handle: int, source_name: str, destination_name: str) -> None:
    handle = _nt_open_relative(
        parent_handle,
        source_name,
        desired_access=0x00010000 | 0x80 | 0x00100000,  # DELETE | READ_ATTRIBUTES | SYNCHRONIZE
        disposition=1,
        options=0x00200000 | 0x00000040 | 0x00000020,
    )
    try:
        attributes, _file_index = _windows_handle_information(handle)
        if attributes & (0x10 | 0x400):
            raise UnsafeFilesystemOperation(f"anchored hard-link source is unsafe: {source_name}")
        _set_windows_handle_name(
            handle,
            parent_handle,
            destination_name,
            information_class=11,  # FileLinkInfo
            replace=False,
        )
    finally:
        _close_windows_handle(handle)


def _windows_replace_child(
    parent_handle: int,
    source_name: str,
    destination_name: str,
    *,
    replace: bool,
) -> None:
    handle = _nt_open_relative(
        parent_handle,
        source_name,
        desired_access=0x00010000 | 0x00100000,  # DELETE | SYNCHRONIZE
        disposition=1,
        options=0x00200000 | 0x00000020,
    )
    try:
        _set_windows_handle_name(
            handle,
            parent_handle,
            destination_name,
            information_class=10,  # FileRenameInformation
            replace=replace,
        )
    finally:
        _close_windows_handle(handle)


def _windows_create_directory(parent_handle: int, name: str) -> None:
    handle = _nt_open_relative(
        parent_handle,
        name,
        desired_access=0x00000001 | 0x80 | 0x00100000,
        disposition=2,  # FILE_CREATE
        options=0x00200000 | 0x00000001 | 0x00000020,
    )
    try:
        attributes, _file_index = _windows_handle_information(handle)
        if not attributes & 0x10 or attributes & 0x400:
            raise UnsafeFilesystemOperation("anchored directory creation produced an unsafe entry")
    finally:
        _close_windows_handle(handle)


def _posix_rename_noreplace(descriptor: int, source: str, destination: str) -> None:
    import ctypes
    import errno

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is not None:
        renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            descriptor,
            os.fsencode(source),
            descriptor,
            os.fsencode(destination),
            1,  # RENAME_NOREPLACE
        )
    else:
        renameatx_np = getattr(libc, "renameatx_np", None)
        if renameatx_np is None:
            raise UnsafeFilesystemOperation("exclusive anchored rename is unsupported on this platform")
        renameatx_np.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        renameatx_np.restype = ctypes.c_int
        result = renameatx_np(
            descriptor,
            os.fsencode(source),
            descriptor,
            os.fsencode(destination),
            0x00000004,  # RENAME_EXCL
        )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise FileExistsError(error, os.strerror(error), destination)
    raise OSError(error, os.strerror(error), destination)


def _posix_link_descriptor_noreplace(
    source_descriptor: int,
    destination_directory_descriptor: int,
    destination: str,
) -> None:
    """Create one exact no-replace hard link from an already held file."""

    import ctypes
    import errno

    libc = ctypes.CDLL(None, use_errno=True)
    linkat = getattr(libc, "linkat", None)
    if linkat is None:
        raise ExactPublicationUnsupported("exact held-file publication is unsupported on this platform")
    linkat.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    linkat.restype = ctypes.c_int
    destination_bytes = os.fsencode(destination)
    result = linkat(
        source_descriptor,
        b"",
        destination_directory_descriptor,
        destination_bytes,
        0x1000,  # AT_EMPTY_PATH
    )
    if result == 0:
        return
    first_error = ctypes.get_errno()
    if first_error == errno.EEXIST:
        raise FileExistsError(first_error, os.strerror(first_error), destination)

    for namespace in ("/proc/self/fd", "/dev/fd"):
        descriptor_path = os.fsencode(f"{namespace}/{source_descriptor}")
        ctypes.set_errno(0)
        result = linkat(
            -100,  # AT_FDCWD
            descriptor_path,
            destination_directory_descriptor,
            destination_bytes,
            0x400,  # AT_SYMLINK_FOLLOW
        )
        if result == 0:
            return
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(error, os.strerror(error), destination)
    raise ExactPublicationUnsupported(
        f"exact held-file publication is unsupported by this filesystem (error {first_error})"
    )


def _windows_unlink_child(parent_handle: int, name: str, identity: OwnedFileIdentity) -> None:
    import ctypes

    handle = _nt_open_relative(
        parent_handle,
        name,
        desired_access=0x00010000 | 0x80 | 0x00100000,
        disposition=1,
        options=0x00200000 | 0x00000020,
    )
    try:
        attributes, file_index = _windows_handle_information(handle)
        file_type = stat.S_IFDIR if attributes & 0x10 else stat.S_IFREG
        if attributes & 0x400 or file_index != identity.inode or file_type != identity.file_type:
            raise UnsafeFilesystemOperation("owned cleanup entry changed before deletion")

        class _FileDispositionInfo(ctypes.Structure):
            _fields_ = [("DeleteFile", ctypes.c_ubyte)]

        information = _FileDispositionInfo(True)
        _set_file_information(handle, 13, ctypes.byref(information), ctypes.sizeof(information))
    finally:
        _close_windows_handle(handle)


def _nt_open_relative(
    parent_handle: int,
    name: str,
    *,
    desired_access: int,
    disposition: int,
    options: int,
    share_access: int = 0x00000001 | 0x00000002 | 0x00000004,
) -> int:
    import ctypes
    from ctypes import wintypes

    class _UnicodeString(ctypes.Structure):
        _fields_ = [
            ("Length", wintypes.USHORT),
            ("MaximumLength", wintypes.USHORT),
            ("Buffer", wintypes.LPWSTR),
        ]

    class _ObjectAttributes(ctypes.Structure):
        _fields_ = [
            ("Length", wintypes.ULONG),
            ("RootDirectory", wintypes.HANDLE),
            ("ObjectName", ctypes.POINTER(_UnicodeString)),
            ("Attributes", wintypes.ULONG),
            ("SecurityDescriptor", wintypes.LPVOID),
            ("SecurityQualityOfService", wintypes.LPVOID),
        ]

    class _IoStatusUnion(ctypes.Union):
        _fields_ = [("Status", wintypes.LONG), ("Pointer", wintypes.LPVOID)]  # noqa: RUF012

    class _IoStatusBlock(ctypes.Structure):
        _fields_ = [("value", _IoStatusUnion), ("Information", ctypes.c_size_t)]

    buffer = ctypes.create_unicode_buffer(name)
    name_bytes = len(name.encode("utf-16-le"))
    unicode_name = _UnicodeString(name_bytes, name_bytes + 2, ctypes.cast(buffer, wintypes.LPWSTR))
    attributes = _ObjectAttributes(
        ctypes.sizeof(_ObjectAttributes),
        wintypes.HANDLE(parent_handle),
        ctypes.pointer(unicode_name),
        0x40,  # OBJ_CASE_INSENSITIVE
        None,
        None,
    )
    status_block = _IoStatusBlock()
    result = wintypes.HANDLE()
    nt_create_file = ctypes.WinDLL("ntdll").NtCreateFile
    nt_create_file.argtypes = [
        ctypes.POINTER(wintypes.HANDLE),
        wintypes.DWORD,
        ctypes.POINTER(_ObjectAttributes),
        ctypes.POINTER(_IoStatusBlock),
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    nt_create_file.restype = wintypes.LONG
    status = nt_create_file(
        ctypes.byref(result),
        desired_access,
        ctypes.byref(attributes),
        ctypes.byref(status_block),
        None,
        0x80,
        share_access,
        disposition,
        options,
        None,
        0,
    )
    if status < 0:
        _raise_windows_ntstatus(status, name)
    return int(result.value)


def _set_windows_handle_name(
    handle: int,
    parent_handle: int,
    name: str,
    *,
    information_class: int,
    replace: bool,
) -> None:
    import ctypes
    from ctypes import wintypes

    class _FileNameInformation(ctypes.Structure):
        _fields_ = [
            ("ReplaceIfExists", wintypes.BOOL),
            ("RootDirectory", wintypes.HANDLE),
            ("FileNameLength", wintypes.DWORD),
            ("FileName", wintypes.WCHAR * 1),
        ]

    encoded = name.encode("utf-16-le")
    size = _FileNameInformation.FileName.offset + len(encoded)
    buffer = ctypes.create_string_buffer(size)
    information = ctypes.cast(buffer, ctypes.POINTER(_FileNameInformation)).contents
    information.ReplaceIfExists = bool(replace)
    information.RootDirectory = wintypes.HANDLE(parent_handle)
    information.FileNameLength = len(encoded)
    ctypes.memmove(ctypes.addressof(buffer) + _FileNameInformation.FileName.offset, encoded, len(encoded))
    _set_file_information(handle, information_class, buffer, size)


def _set_file_information(handle: int, information_class: int, information: Any, size: int) -> None:
    import ctypes
    from ctypes import wintypes

    class _IoStatusUnion(ctypes.Union):
        _fields_ = [("Status", wintypes.LONG), ("Pointer", wintypes.LPVOID)]  # noqa: RUF012

    class _IoStatusBlock(ctypes.Structure):
        _fields_ = [("value", _IoStatusUnion), ("Information", ctypes.c_size_t)]

    status_block = _IoStatusBlock()
    set_information = ctypes.WinDLL("ntdll").NtSetInformationFile
    set_information.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_IoStatusBlock),
        wintypes.LPVOID,
        wintypes.ULONG,
        wintypes.DWORD,
    ]
    set_information.restype = wintypes.LONG
    status = set_information(
        wintypes.HANDLE(handle),
        ctypes.byref(status_block),
        information,
        size,
        information_class,
    )
    if status < 0:
        _raise_windows_ntstatus(status, "anchored child mutation")


def _raise_windows_ntstatus(status: int, name: str) -> None:
    import ctypes
    from ctypes import wintypes

    convert = ctypes.WinDLL("ntdll").RtlNtStatusToDosError
    convert.argtypes = [wintypes.LONG]
    convert.restype = wintypes.ULONG
    error = int(convert(status))
    if error in {80, 183}:
        raise FileExistsError(error, os.strerror(error), name)
    if error in {2, 3}:
        raise FileNotFoundError(error, os.strerror(error), name)
    raise OSError(error, os.strerror(error), name)


def _close_windows_handle(handle: int) -> None:
    import ctypes
    from ctypes import wintypes

    close_handle = ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL
    if not close_handle(wintypes.HANDLE(handle)):
        raise OSError(ctypes.get_last_error(), "could not close anchored Windows handle")


__all__ = [
    "AnchoredDirectory",
    "ExactPublicationUnsupported",
    "OwnedFileIdentity",
    "UnsafeFilesystemOperation",
    "atomic_write_bytes",
    "atomic_write_text",
    "open_anchored_directory",
    "remove_confined_tree",
    "require_confined_path",
]
