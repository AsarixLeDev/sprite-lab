"""Streaming download helpers for direct archive and image URLs."""

from __future__ import annotations

import hashlib
import http.client
import ipaddress
import os
import socket
import ssl
import stat
import tempfile
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from spritelab.utils.safe_fs import require_confined_path

DEFAULT_MAX_DOWNLOAD_BYTES = 512 * 1024 * 1024
_COPY_CHUNK_BYTES = 1 << 20


class DownloadSecurityError(ValueError):
    """Raised when a remote download cannot satisfy the acquisition policy."""


class DownloadCancelled(DownloadSecurityError):
    """Raised when a bounded download observes an explicit cancellation."""


@dataclass(frozen=True)
class DownloadReceipt:
    """Network and content evidence for one manually redirected response."""

    final_url: str
    redirect_chain: tuple[str, ...]
    http_status: int
    response_mime_type: str
    response_bytes: int
    response_sha256: str
    elapsed_seconds: float


@dataclass(frozen=True)
class ReceiptDownloadResult:
    path: Path
    receipt: DownloadReceipt


class PinnedHTTPResponse(Protocol):
    status: int
    headers: Any
    peer_ip: str

    def read(self, size: int = -1) -> bytes: ...

    def close(self) -> None: ...


class PinnedHTTPTransport(Protocol):
    def open(
        self,
        *,
        url: str,
        pinned_ip: str,
        server_hostname: str,
        port: int,
        timeout_seconds: float,
    ) -> PinnedHTTPResponse: ...


HostResolver = Callable[[str, int], Sequence[str]]
DownloadProgress = Callable[[int, int | None], None]
CancelProbe = Callable[[], bool]


def download_file_with_receipt(
    url: str,
    output_path: str | Path,
    *,
    allowed_hosts: Sequence[str],
    overwrite: bool = False,
    timeout_seconds: float = 60.0,
    max_duration_seconds: float | None = None,
    allowed_content_types: Sequence[str] = (),
    max_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES,
    expected_sha256: str | None = None,
    max_redirects: int = 5,
    require_https: bool = True,
    allow_private_hosts: bool = False,
    cancel_requested: CancelProbe | None = None,
    progress: DownloadProgress | None = None,
    resolver: HostResolver | None = None,
    transport: PinnedHTTPTransport | None = None,
) -> ReceiptDownloadResult:
    """Download with manual redirects and one pinned DNS answer per hop.

    Every address returned for a hop must be globally routable unless the
    private-host escape hatch is explicitly enabled for a local test. The
    selected address is passed directly to the transport, while the original
    hostname remains the TLS SNI and certificate-verification name.
    """

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    duration_limit = timeout_seconds if max_duration_seconds is None else max_duration_seconds
    if duration_limit <= 0:
        raise ValueError("max_duration_seconds must be positive")
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    if type(max_redirects) is not int or max_redirects < 0:
        raise ValueError("max_redirects must be a non-negative integer")
    normalized_hosts = tuple(host.casefold().rstrip(".") for host in allowed_hosts)
    if not normalized_hosts or len(set(normalized_hosts)) != len(normalized_hosts):
        raise ValueError("allowed_hosts must contain unique exact hostnames")
    expected_digest = _normalize_expected_sha256(expected_sha256)
    requested_output_path = Path(output_path)
    safe_output_path = _prepare_download_path(output_path, create_parent=False)
    if os.path.lexists(safe_output_path) and not overwrite:
        raise FileExistsError(f"output file already exists: {safe_output_path}")

    started = time.monotonic()
    deadline = started + duration_limit
    current_url = url
    redirect_chain: list[str] = []
    resolve = resolver or _resolve_host_addresses
    pinned_transport = transport or _StdlibPinnedTransport()
    part_path: Path | None = None
    while True:
        _check_download_abort(cancel_requested, deadline)
        _parsed, host, port = _validate_pinned_url(
            current_url,
            normalized_hosts,
            require_https=require_https,
        )
        addresses = tuple(resolve(host, port))
        selected_ip = _select_pinned_address(
            host,
            addresses,
            allow_private_hosts=allow_private_hosts,
        )
        _check_download_abort(cancel_requested, deadline)
        remaining = max(0.001, min(timeout_seconds, deadline - time.monotonic()))
        response = pinned_transport.open(
            url=current_url,
            pinned_ip=selected_ip,
            server_hostname=host,
            port=port,
            timeout_seconds=remaining,
        )
        try:
            _verify_peer_address(response.peer_ip, selected_ip)
            status = int(response.status)
            if status in {301, 302, 303, 307, 308}:
                locations = _header_values(response.headers, "Location")
                if len(locations) != 1 or not locations[0].strip():
                    raise DownloadSecurityError("redirect response must contain exactly one Location header")
                if len(redirect_chain) >= max_redirects:
                    raise DownloadSecurityError(f"download exceeded the {max_redirects}-redirect limit")
                redirect_chain.append(current_url)
                current_url = urllib.parse.urljoin(current_url, locations[0].strip())
                continue
            if not 200 <= status < 300:
                raise DownloadSecurityError(f"download returned HTTP status {status}")

            content_types = _header_values(response.headers, "Content-Type")
            if len(content_types) != 1:
                raise DownloadSecurityError("download response must contain exactly one Content-Type header")
            content_type = content_types[0].split(";", 1)[0].strip().lower()
            allowed_types = {value.strip().lower() for value in allowed_content_types}
            if content_type == "text/html" and content_type not in allowed_types:
                raise DownloadSecurityError("download returned HTML instead of an allowed artifact")
            if allowed_types and content_type not in allowed_types:
                raise DownloadSecurityError(f"download returned unsupported content type {content_type!r}")
            lengths = _header_values(response.headers, "Content-Length")
            if len(lengths) > 1:
                raise DownloadSecurityError("download response contains multiple Content-Length headers")
            declared_length = _parse_content_length(lengths[0] if lengths else None)
            if declared_length is not None and declared_length > max_bytes:
                raise DownloadSecurityError(
                    f"download declares {declared_length} bytes, exceeding the {max_bytes}-byte limit"
                )

            safe_output_path = _prepare_download_path(safe_output_path, create_parent=True)
            if os.path.lexists(safe_output_path) and not overwrite:
                raise FileExistsError(f"output file already exists: {safe_output_path}")
            descriptor, part_name = tempfile.mkstemp(
                prefix=f".{safe_output_path.name}.",
                suffix=".part",
                dir=safe_output_path.parent,
            )
            part_path = Path(part_name)
            digest = hashlib.sha256()
            received = 0
            with os.fdopen(descriptor, "wb") as handle:
                while True:
                    _check_download_abort(cancel_requested, deadline)
                    chunk = response.read(_COPY_CHUNK_BYTES)
                    if not chunk:
                        break
                    if received + len(chunk) > max_bytes:
                        raise DownloadSecurityError(f"download exceeded the {max_bytes}-byte limit")
                    handle.write(chunk)
                    digest.update(chunk)
                    received += len(chunk)
                    if progress is not None:
                        progress(received, declared_length)
                handle.flush()
                os.fsync(handle.fileno())
            if declared_length is not None and received != declared_length:
                raise DownloadSecurityError(
                    f"download length mismatch: expected {declared_length} bytes, received {received}"
                )
            actual_digest = digest.hexdigest()
            if expected_digest is not None and actual_digest != expected_digest:
                raise DownloadSecurityError(
                    f"download SHA256 mismatch: expected {expected_digest}, got {actual_digest}"
                )
            _check_download_abort(cancel_requested, deadline)
            _publish_download(part_path, safe_output_path, overwrite=overwrite)
            part_path = None
            elapsed = time.monotonic() - started
            return ReceiptDownloadResult(
                path=requested_output_path,
                receipt=DownloadReceipt(
                    final_url=current_url,
                    redirect_chain=tuple(redirect_chain),
                    http_status=status,
                    response_mime_type=content_type,
                    response_bytes=received,
                    response_sha256=actual_digest,
                    elapsed_seconds=elapsed,
                ),
            )
        finally:
            response.close()
            if part_path is not None:
                part_path.unlink(missing_ok=True)


def compute_sha256(
    path: str | Path,
    *,
    chunk_size: int = _COPY_CHUNK_BYTES,
    max_bytes: int | None = None,
) -> str:
    """Return the SHA256 hex digest of a file, streamed."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if max_bytes is not None and max_bytes <= 0:
        raise ValueError("max_bytes must be positive when provided")
    digest = hashlib.sha256()
    total = 0
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            total += len(chunk)
            if max_bytes is not None and total > max_bytes:
                raise ValueError(f"file exceeds the {max_bytes}-byte hashing limit")
            digest.update(chunk)
    return digest.hexdigest()


def download_file(
    url: str,
    output_path: str | Path,
    *,
    overwrite: bool = False,
    timeout_seconds: float = 60.0,
    allowed_content_types: Sequence[str] = (),
    max_bytes: int = DEFAULT_MAX_DOWNLOAD_BYTES,
    expected_sha256: str | None = None,
    allow_private_hosts: bool = False,
) -> Path:
    """Download one HTTP(S) URL with bounded, verified atomic publication.

    Both the initial URL and every redirect must resolve only to globally
    routable addresses unless ``allow_private_hosts`` is explicitly enabled.
    ``overwrite=False`` uses an atomic exclusive hard-link publication so a
    concurrently-created destination is never replaced.
    """

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    expected_digest = _normalize_expected_sha256(expected_sha256)

    requested_output_path = Path(output_path)
    output_path = _prepare_download_path(output_path, create_parent=False)
    if os.path.lexists(output_path) and not overwrite:
        raise FileExistsError(f"output file already exists: {output_path}")
    _validate_remote_url(url, allow_private_hosts=allow_private_hosts)
    output_path = _prepare_download_path(output_path, create_parent=True)
    if os.path.lexists(output_path) and not overwrite:
        raise FileExistsError(f"output file already exists: {output_path}")

    request = urllib.request.Request(url, headers={"User-Agent": "spritelab-harvest/0.1"})
    part_path: Path | None = None
    try:
        with _open_url(
            request,
            timeout_seconds=timeout_seconds,
            allow_private_hosts=allow_private_hosts,
        ) as response:
            final_url = response.geturl() if hasattr(response, "geturl") else url
            _validate_remote_url(final_url, allow_private_hosts=allow_private_hosts)
            content_type = str(response.headers.get("Content-Type", ""))
            normalized_content_type = content_type.lower().split(";", 1)[0].strip()
            if normalized_content_type == "text/html":
                raise DownloadSecurityError(
                    f"URL returned HTML instead of a file ({content_type}); "
                    "this is probably a landing page, not a direct download."
                )
            if allowed_content_types and normalized_content_type not in {
                allowed.lower().strip() for allowed in allowed_content_types
            }:
                raise DownloadSecurityError(
                    f"URL returned unsupported content type {content_type!r}; "
                    f"expected one of {list(allowed_content_types)!r}"
                )

            declared_length = _parse_content_length(response.headers.get("Content-Length"))
            if declared_length is not None and declared_length > max_bytes:
                raise DownloadSecurityError(
                    f"download declares {declared_length} bytes, exceeding the {max_bytes}-byte limit"
                )
            progress = _make_progress(declared_length, final_url)
            descriptor, part_name = tempfile.mkstemp(
                prefix=f".{output_path.name}.",
                suffix=".part",
                dir=output_path.parent,
            )
            part_path = Path(part_name)
            digest = hashlib.sha256()
            received = 0
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    while True:
                        chunk = response.read(_COPY_CHUNK_BYTES)
                        if not chunk:
                            break
                        if received + len(chunk) > max_bytes:
                            raise DownloadSecurityError(f"download exceeded the {max_bytes}-byte limit")
                        handle.write(chunk)
                        digest.update(chunk)
                        received += len(chunk)
                        if progress is not None:
                            progress.update(len(chunk))
                    handle.flush()
                    os.fsync(handle.fileno())
            finally:
                if progress is not None:
                    progress.close()

            if declared_length is not None and received != declared_length:
                raise DownloadSecurityError(
                    f"download length mismatch: expected {declared_length} bytes, received {received}"
                )
            actual_digest = digest.hexdigest()
            if expected_digest is not None and actual_digest != expected_digest:
                raise DownloadSecurityError(
                    f"download SHA256 mismatch: expected {expected_digest}, got {actual_digest}"
                )

        assert part_path is not None
        _publish_download(part_path, output_path, overwrite=overwrite)
        part_path = None
        return requested_output_path
    finally:
        if part_path is not None:
            part_path.unlink(missing_ok=True)


def _open_url(
    request: urllib.request.Request,
    *,
    timeout_seconds: float,
    allow_private_hosts: bool,
):
    opener = urllib.request.build_opener(_ValidatedRedirectHandler(allow_private_hosts=allow_private_hosts))
    return opener.open(request, timeout=timeout_seconds)


class _HTTPClientPinnedResponse:
    def __init__(
        self,
        connection: http.client.HTTPConnection,
        response: http.client.HTTPResponse,
        peer_ip: str,
    ) -> None:
        self._connection = connection
        self._response = response
        self.status = response.status
        self.headers = response.headers
        self.peer_ip = peer_ip

    def read(self, size: int = -1) -> bytes:
        return self._response.read(size)

    def close(self) -> None:
        try:
            self._response.close()
        finally:
            self._connection.close()


class _StdlibPinnedTransport:
    """HTTP/1.1 transport that connects to an IP but authenticates the URL host."""

    def __init__(self, *, ssl_context: ssl.SSLContext | None = None) -> None:
        self._ssl_context = ssl_context or ssl.create_default_context()

    def open(
        self,
        *,
        url: str,
        pinned_ip: str,
        server_hostname: str,
        port: int,
        timeout_seconds: float,
    ) -> PinnedHTTPResponse:
        parsed = urllib.parse.urlsplit(url)
        target = urllib.parse.urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
        raw_socket = socket.create_connection((pinned_ip, port), timeout=timeout_seconds)
        connection: http.client.HTTPConnection
        connected_socket: socket.socket | ssl.SSLSocket | None = None
        try:
            if parsed.scheme.casefold() == "https":
                connection = http.client.HTTPSConnection(
                    server_hostname,
                    port,
                    timeout=timeout_seconds,
                    context=self._ssl_context,
                )
                connected_socket = self._ssl_context.wrap_socket(raw_socket, server_hostname=server_hostname)
            else:
                connection = http.client.HTTPConnection(server_hostname, port, timeout=timeout_seconds)
                connected_socket = raw_socket
            connected_socket.settimeout(timeout_seconds)
            peer_ip = str(connected_socket.getpeername()[0])
            connection.sock = connected_socket
            connection.request(
                "GET",
                target,
                headers={
                    "Accept": "*/*",
                    "Connection": "close",
                    "User-Agent": "spritelab-harvest/0.1",
                },
            )
            response = connection.getresponse()
            return _HTTPClientPinnedResponse(connection, response, peer_ip)
        except BaseException:
            if connected_socket is not None:
                connected_socket.close()
            else:
                raw_socket.close()
            raise


def _validate_pinned_url(
    url: str,
    allowed_hosts: Sequence[str],
    *,
    require_https: bool,
) -> tuple[urllib.parse.SplitResult, str, int]:
    try:
        parsed = urllib.parse.urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise DownloadSecurityError(f"invalid download URL: {url!r}") from exc
    scheme = parsed.scheme.casefold()
    if scheme not in {"http", "https"} or (require_https and scheme != "https"):
        policy = "https://" if require_https else "http:// or https://"
        raise DownloadSecurityError(f"download URL must use {policy}")
    if not parsed.hostname:
        raise DownloadSecurityError("download URL must include a hostname")
    if parsed.username is not None or parsed.password is not None:
        raise DownloadSecurityError("download URLs may not contain credentials")
    if parsed.fragment:
        raise DownloadSecurityError("download URLs may not contain fragments")
    if any(ord(character) < 32 for character in url):
        raise DownloadSecurityError("download URL contains control characters")
    host = parsed.hostname.casefold().rstrip(".")
    if host not in set(allowed_hosts):
        raise DownloadSecurityError(f"download host {host!r} is not in the exact source allowlist")
    effective_port = port or (443 if scheme == "https" else 80)
    if not 1 <= effective_port <= 65535:
        raise DownloadSecurityError("download URL contains an invalid port")
    return parsed, host, effective_port


def _select_pinned_address(
    host: str,
    addresses: Sequence[str],
    *,
    allow_private_hosts: bool,
) -> str:
    parsed_addresses: dict[tuple[int, bytes], ipaddress.IPv4Address | ipaddress.IPv6Address] = {}
    for value in addresses:
        try:
            address = ipaddress.ip_address(str(value).split("%", 1)[0])
        except ValueError as exc:
            raise DownloadSecurityError(f"resolver returned an invalid address for {host!r}") from exc
        if not allow_private_hosts and not address.is_global:
            raise DownloadSecurityError(f"download host {host!r} resolves to non-public address {address}")
        parsed_addresses[(address.version, address.packed)] = address
    if not parsed_addresses:
        raise DownloadSecurityError(f"hostname did not resolve to an address: {host!r}")
    return str(parsed_addresses[min(parsed_addresses)])


def _verify_peer_address(peer_ip: str, pinned_ip: str) -> None:
    try:
        peer = ipaddress.ip_address(peer_ip.split("%", 1)[0])
        pinned = ipaddress.ip_address(pinned_ip.split("%", 1)[0])
    except ValueError as exc:
        raise DownloadSecurityError("transport returned an invalid peer address") from exc
    if peer != pinned:
        raise DownloadSecurityError(f"connected peer {peer} did not match pinned address {pinned}")


def _header_values(headers: Any, name: str) -> list[str]:
    get_all = getattr(headers, "get_all", None)
    if callable(get_all):
        values = get_all(name, [])
        return [str(value) for value in values]
    if isinstance(headers, Mapping):
        for key, value in headers.items():
            if str(key).casefold() != name.casefold():
                continue
            if isinstance(value, (list, tuple)):
                return [str(item) for item in value]
            return [str(value)]
    return []


def _check_download_abort(cancel_requested: CancelProbe | None, deadline: float) -> None:
    if cancel_requested is not None and cancel_requested():
        raise DownloadCancelled("download was cancelled")
    if time.monotonic() > deadline:
        raise DownloadSecurityError("download exceeded its duration limit")


def _prepare_download_path(path: str | Path, *, create_parent: bool) -> Path:
    raw_path = os.fspath(path)
    if not raw_path.strip() or raw_path.strip() in {".", ".."}:
        raise DownloadSecurityError("download destination must be a specific non-root path")
    output_path = Path(os.path.abspath(os.path.expanduser(raw_path)))
    existing_ancestor = output_path.parent
    while not os.path.lexists(existing_ancestor):
        parent = existing_ancestor.parent
        if parent == existing_ancestor:
            raise DownloadSecurityError(f"could not find an existing ancestor for destination: {output_path}")
        existing_ancestor = parent
    metadata = existing_ancestor.lstat()
    if _is_link_or_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode):
        raise DownloadSecurityError(f"download destination crosses an unsafe ancestor: {existing_ancestor}")
    output_path = require_confined_path(output_path, existing_ancestor)
    if create_parent:
        _create_download_parents(output_path.parent, existing_ancestor)
        output_path = require_confined_path(output_path, existing_ancestor)
    return output_path


def _create_download_parents(parent: Path, root: Path) -> None:
    current = root
    for part in parent.relative_to(root).parts:
        current = current / part
        try:
            current.mkdir()
        except FileExistsError:
            pass
        metadata = current.lstat()
        if _is_link_or_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode) or current.is_mount():
            raise DownloadSecurityError(f"download destination crosses an unsafe directory seam: {current}")
        require_confined_path(current, root)


class _ValidatedRedirectHandler(urllib.request.HTTPRedirectHandler):
    def __init__(self, *, allow_private_hosts: bool) -> None:
        super().__init__()
        self._allow_private_hosts = allow_private_hosts

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _validate_remote_url(newurl, allow_private_hosts=self._allow_private_hosts)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _validate_remote_url(url: str, *, allow_private_hosts: bool) -> None:
    try:
        parsed = urllib.parse.urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise DownloadSecurityError(f"invalid download URL: {url!r}") from exc
    if parsed.scheme.lower() not in {"http", "https"}:
        raise DownloadSecurityError("downloads require an http:// or https:// URL")
    if not parsed.hostname:
        raise DownloadSecurityError("download URL must include a hostname")
    if parsed.username is not None or parsed.password is not None:
        raise DownloadSecurityError("download URLs may not contain credentials")
    if any(ord(character) < 32 for character in url):
        raise DownloadSecurityError("download URL contains control characters")

    host = parsed.hostname.rstrip(".")
    if not host:
        raise DownloadSecurityError("download URL must include a hostname")
    effective_port = port or (443 if parsed.scheme.lower() == "https" else 80)
    addresses = _resolve_host_addresses(host, effective_port)
    if not addresses:
        raise DownloadSecurityError(f"hostname did not resolve to an address: {host!r}")
    if allow_private_hosts:
        return
    for address_text in addresses:
        try:
            address = ipaddress.ip_address(address_text)
        except ValueError as exc:
            raise DownloadSecurityError(f"resolver returned an invalid address: {address_text!r}") from exc
        if not address.is_global:
            raise DownloadSecurityError(f"download host {host!r} resolves to non-public address {address}")


def _resolve_host_addresses(host: str, port: int) -> tuple[str, ...]:
    try:
        records = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise DownloadSecurityError(f"could not resolve download host {host!r}") from exc
    return tuple(sorted({str(record[4][0]) for record in records}))


def _parse_content_length(raw_value: object) -> int | None:
    if raw_value in (None, ""):
        return None
    try:
        value = int(str(raw_value))
    except (TypeError, ValueError) as exc:
        raise DownloadSecurityError(f"invalid Content-Length header: {raw_value!r}") from exc
    if value < 0:
        raise DownloadSecurityError(f"invalid Content-Length header: {raw_value!r}")
    return value


def _normalize_expected_sha256(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError("expected_sha256 must be a 64-character hexadecimal SHA256 digest")
    return normalized


def _publish_download(part_path: Path, output_path: Path, *, overwrite: bool) -> None:
    if overwrite:
        os.replace(part_path, output_path)
        return
    try:
        os.link(part_path, output_path)
    except FileExistsError as exc:
        raise FileExistsError(f"output file already exists: {output_path}") from exc
    part_path.unlink()


def _is_link_or_reparse(metadata: os.stat_result) -> bool:
    if stat.S_ISLNK(metadata.st_mode):
        return True
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & reparse_flag)


def _make_progress(total: int | None, url: str):
    try:
        from tqdm import tqdm
    except ImportError:
        return None
    return tqdm(total=total, unit="B", unit_scale=True, desc=Path(urllib.parse.urlsplit(url).path).name or "download")


__all__ = [
    "DEFAULT_MAX_DOWNLOAD_BYTES",
    "DownloadCancelled",
    "DownloadReceipt",
    "DownloadSecurityError",
    "PinnedHTTPResponse",
    "PinnedHTTPTransport",
    "ReceiptDownloadResult",
    "compute_sha256",
    "download_file",
    "download_file_with_receipt",
]
