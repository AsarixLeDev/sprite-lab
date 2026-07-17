"""Tests for spritelab.harvest.archive and download helpers."""

from __future__ import annotations

import io
import os
import stat
import tarfile
import warnings
import zipfile

import pytest

from _harvest_testdata import make_zip_of_pngs
from spritelab.harvest.archive import ArchiveSecurityError, extract_archive, iter_archive_pngs
from spritelab.harvest.download import DownloadSecurityError, compute_sha256, download_file


class _Response(io.BytesIO):
    def __init__(
        self,
        content: bytes,
        *,
        url: str = "https://example.test/archive.zip",
        content_type: str = "application/octet-stream",
    ) -> None:
        super().__init__(content)
        self.headers = {"Content-Type": content_type, "Content-Length": str(len(content))}
        self._url = url

    def geturl(self) -> str:
        return self._url


def _stub_public_download(monkeypatch, response_factory) -> None:
    monkeypatch.setattr(
        "spritelab.harvest.download._resolve_host_addresses",
        lambda _host, _port: ("93.184.216.34",),
    )
    monkeypatch.setattr(
        "spritelab.harvest.download._open_url",
        lambda *_args, **_kwargs: response_factory(),
    )


def test_extracts_simple_zip(tmp_path):
    zip_path = make_zip_of_pngs(tmp_path / "pack.zip", ["a.png", "sub/b.png"])
    out = extract_archive(zip_path, tmp_path / "out")
    assert (out / "a.png").exists()
    assert (out / "sub" / "b.png").exists()


def test_zip_slip_rejects_entire_archive_without_publication(tmp_path):
    zip_path = tmp_path / "evil.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("../evil.txt", "bad")
        archive.writestr("/abs.txt", "bad")
        archive.writestr("ok.txt", "good")

    with pytest.raises(ArchiveSecurityError, match="unsafe archive member"):
        extract_archive(zip_path, tmp_path / "out")

    assert not (tmp_path / "evil.txt").exists()
    assert not (tmp_path / "out").exists()


def test_iterates_png_names(tmp_path):
    zip_path = make_zip_of_pngs(tmp_path / "pack.zip", ["z.png", "a.png"])
    with zipfile.ZipFile(zip_path, "a") as archive:
        archive.writestr("readme.txt", "hi")
    names = iter_archive_pngs(zip_path)
    assert names == ["a.png", "z.png"]


def test_computes_sha256(tmp_path):
    path = tmp_path / "file.bin"
    path.write_bytes(b"hello")
    assert compute_sha256(path) == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


def test_refuses_overwrite(tmp_path):
    zip_path = make_zip_of_pngs(tmp_path / "pack.zip", ["a.png"])
    out = extract_archive(zip_path, tmp_path / "out")
    with pytest.raises(FileExistsError):
        extract_archive(zip_path, out)
    extract_archive(zip_path, out, overwrite=True)


def test_failed_overwrite_preserves_existing_destination(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    marker = output / "existing.txt"
    marker.write_bytes(b"existing")
    archive_path = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("safe.txt", b"new")
        archive.writestr("../escape.txt", b"escape")

    with pytest.raises(ArchiveSecurityError):
        extract_archive(archive_path, output, overwrite=True)

    assert marker.read_bytes() == b"existing"
    assert not (output / "safe.txt").exists()
    assert not (tmp_path / "escape.txt").exists()


def test_destination_parent_symlink_is_rejected_before_outside_directory_creation(tmp_path):
    archive_path = tmp_path / "pack.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("safe.txt", b"safe")
    safe_parent = tmp_path / "safe-parent"
    safe_parent.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        os.symlink(outside, safe_parent / "linked", target_is_directory=True)
    except OSError:
        pytest.skip("directory symbolic links are unavailable in this test session")

    with pytest.raises(ArchiveSecurityError, match="linked or non-directory"):
        extract_archive(archive_path, safe_parent / "linked" / "new" / "out")

    assert not (outside / "new").exists()


@pytest.mark.parametrize(
    "members",
    [
        (("A.png", b"one"), ("a.png", b"two")),
        (("CON.txt", b"reserved"),),
        (("parent", b"file"), ("parent/child.png", b"child")),
    ],
)
def test_rejects_platform_collisions_reserved_names_and_file_prefixes(tmp_path, members):
    archive_path = tmp_path / "ambiguous.zip"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(archive_path, "w") as archive:
            for name, payload in members:
                archive.writestr(name, payload)

    with pytest.raises(ArchiveSecurityError):
        extract_archive(archive_path, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_rejects_zip_symlink_member(tmp_path):
    archive_path = tmp_path / "linked.zip"
    linked = zipfile.ZipInfo("linked.png")
    linked.create_system = 3
    linked.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(linked, "../outside.png")

    with pytest.raises(ArchiveSecurityError, match="linked or special"):
        extract_archive(archive_path, tmp_path / "out")


def test_rejects_encrypted_zip_member_before_extraction(tmp_path):
    archive_path = tmp_path / "encrypted.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("secret.png", b"secret")
    payload = bytearray(archive_path.read_bytes())
    local_header = payload.index(b"PK\x03\x04")
    central_header = payload.index(b"PK\x01\x02")
    payload[local_header + 6 : local_header + 8] = (1).to_bytes(2, "little")
    payload[central_header + 8 : central_header + 10] = (1).to_bytes(2, "little")
    archive_path.write_bytes(payload)

    with pytest.raises(ArchiveSecurityError, match="encrypted"):
        extract_archive(archive_path, tmp_path / "out")


@pytest.mark.parametrize("member_type", [tarfile.LNKTYPE, tarfile.SYMTYPE, tarfile.CHRTYPE, tarfile.FIFOTYPE])
def test_rejects_tar_link_and_special_members(tmp_path, member_type):
    archive_path = tmp_path / "special.tar"
    linked = tarfile.TarInfo("special.png")
    linked.type = member_type
    linked.linkname = "../outside.png"
    with tarfile.open(archive_path, "w") as archive:
        archive.addfile(linked)

    with pytest.raises(ArchiveSecurityError, match="linked or special"):
        extract_archive(archive_path, tmp_path / "out")


def test_extracts_bounded_regular_tar_member(tmp_path):
    archive_path = tmp_path / "pack.tar"
    payload = b"safe"
    member = tarfile.TarInfo("nested/safe.txt")
    member.size = len(payload)
    with tarfile.open(archive_path, "w") as archive:
        archive.addfile(member, io.BytesIO(payload))

    output = extract_archive(archive_path, tmp_path / "out")

    assert (output / "nested" / "safe.txt").read_bytes() == payload


def test_rejects_archive_beyond_size_and_compression_limits(tmp_path):
    archive_path = tmp_path / "bomb.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("large.txt", b"A" * 4096)

    with pytest.raises(ArchiveSecurityError, match="compression-ratio"):
        extract_archive(archive_path, tmp_path / "ratio", max_compression_ratio=2)
    with pytest.raises(ArchiveSecurityError, match="member limit"):
        extract_archive(archive_path, tmp_path / "size", max_member_bytes=1024, max_total_bytes=4096)
    with pytest.raises(ArchiveSecurityError, match="input limit"):
        extract_archive(archive_path, tmp_path / "input", max_archive_bytes=10)
    assert not (tmp_path / "ratio").exists()
    assert not (tmp_path / "size").exists()
    assert not (tmp_path / "input").exists()


def test_rejects_archive_member_count_limit(tmp_path):
    archive_path = tmp_path / "many.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("one.txt", b"one")
        archive.writestr("two.txt", b"two")

    with pytest.raises(ArchiveSecurityError, match="more than 1 members"):
        extract_archive(archive_path, tmp_path / "out", max_members=1)

    assert not (tmp_path / "out").exists()


def test_overwrite_unlinks_hardlinked_destination_without_touching_outside_file(tmp_path):
    archive_path = tmp_path / "pack.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("sprite.png", b"new")
    output = tmp_path / "out"
    output.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    try:
        os.link(outside, output / "sprite.png")
    except OSError:
        pytest.skip("hard links are unavailable in this test session")

    extract_archive(archive_path, output, overwrite=True)

    assert outside.read_bytes() == b"outside"
    assert (output / "sprite.png").read_bytes() == b"new"


def test_download_uses_unique_partial_and_preserves_preplanted_links(tmp_path, monkeypatch):
    _stub_public_download(monkeypatch, lambda: _Response(b"download"))
    output = tmp_path / "archive.zip"
    predictable_partial = tmp_path / "archive.zip.part"
    outside_output = tmp_path / "outside-output.bin"
    outside_partial = tmp_path / "outside-partial.bin"
    outside_output.write_bytes(b"old output")
    outside_partial.write_bytes(b"old partial")
    try:
        os.link(outside_output, output)
        os.link(outside_partial, predictable_partial)
    except OSError:
        pytest.skip("hard links are unavailable in this test session")

    download_file("https://example.test/archive.zip", output, overwrite=True)

    assert output.read_bytes() == b"download"
    assert outside_output.read_bytes() == b"old output"
    assert predictable_partial.read_bytes() == b"old partial"
    assert outside_partial.read_bytes() == b"old partial"


@pytest.mark.parametrize("url", ["file:///tmp/archive.zip", "ftp://example.test/archive.zip"])
def test_download_rejects_non_http_schemes_without_opening(tmp_path, monkeypatch, url):
    monkeypatch.setattr(
        "spritelab.harvest.download._open_url",
        lambda *_args, **_kwargs: pytest.fail("unsafe URL reached the opener"),
    )
    with pytest.raises(DownloadSecurityError, match="http"):
        download_file(url, tmp_path / "archive.zip")


def test_download_rejects_private_host_without_opening(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "spritelab.harvest.download._resolve_host_addresses",
        lambda _host, _port: ("127.0.0.1",),
    )
    monkeypatch.setattr(
        "spritelab.harvest.download._open_url",
        lambda *_args, **_kwargs: pytest.fail("private URL reached the opener"),
    )
    with pytest.raises(DownloadSecurityError, match="non-public"):
        download_file("http://localhost/archive.zip", tmp_path / "archive.zip")


def test_download_parent_symlink_is_rejected_before_outside_directory_creation(tmp_path, monkeypatch):
    safe_parent = tmp_path / "safe-parent"
    safe_parent.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        os.symlink(outside, safe_parent / "linked", target_is_directory=True)
    except OSError:
        pytest.skip("directory symbolic links are unavailable in this test session")
    monkeypatch.setattr(
        "spritelab.harvest.download._open_url",
        lambda *_args, **_kwargs: pytest.fail("unsafe destination reached the opener"),
    )

    with pytest.raises(DownloadSecurityError, match="unsafe ancestor"):
        download_file(
            "https://example.test/archive.zip",
            safe_parent / "linked" / "new" / "archive.zip",
        )

    assert not (outside / "new").exists()


def test_download_rejects_private_final_redirect(tmp_path, monkeypatch):
    def resolve(host, _port):
        return ("127.0.0.1",) if host == "localhost" else ("93.184.216.34",)

    monkeypatch.setattr("spritelab.harvest.download._resolve_host_addresses", resolve)
    monkeypatch.setattr(
        "spritelab.harvest.download._open_url",
        lambda *_args, **_kwargs: _Response(b"download", url="http://localhost/archive.zip"),
    )
    with pytest.raises(DownloadSecurityError, match="non-public"):
        download_file("https://example.test/archive.zip", tmp_path / "archive.zip")
    assert not (tmp_path / "archive.zip").exists()


def test_download_limits_and_checksum_fail_before_replacing_destination(tmp_path, monkeypatch):
    output = tmp_path / "archive.zip"
    output.write_bytes(b"existing")
    _stub_public_download(monkeypatch, lambda: _Response(b"download"))

    with pytest.raises(DownloadSecurityError, match="exceeding"):
        download_file("https://example.test/archive.zip", output, overwrite=True, max_bytes=4)
    assert output.read_bytes() == b"existing"

    with pytest.raises(DownloadSecurityError, match="SHA256 mismatch"):
        download_file(
            "https://example.test/archive.zip",
            output,
            overwrite=True,
            expected_sha256="0" * 64,
        )
    assert output.read_bytes() == b"existing"
    assert not list(tmp_path.glob(".archive.zip.*.part"))


def test_download_exclusive_publication_loses_race_without_replacing_winner(tmp_path, monkeypatch):
    output = tmp_path / "archive.zip"

    class RacingResponse(_Response):
        def __init__(self) -> None:
            super().__init__(b"download")
            self._published_competitor = False

        def read(self, size: int = -1) -> bytes:
            chunk = super().read(size)
            if not chunk and not self._published_competitor:
                output.write_bytes(b"competitor")
                self._published_competitor = True
            return chunk

    _stub_public_download(monkeypatch, RacingResponse)

    with pytest.raises(FileExistsError):
        download_file("https://example.test/archive.zip", output)

    assert output.read_bytes() == b"competitor"
    assert not list(tmp_path.glob(".archive.zip.*.part"))
