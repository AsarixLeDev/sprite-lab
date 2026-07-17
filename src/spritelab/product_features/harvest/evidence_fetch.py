"""Bounded, inert evidence retrieval and deterministic source verification."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any
from urllib.parse import SplitResult, urljoin, urlunsplit

from spritelab.harvest.download import (
    HARVEST_USER_AGENT,
    DownloadCancelled,
    DownloadSecurityError,
    HostResolver,
    PinnedHTTPTransport,
    ReceiptDownloadResult,
    download_file_with_receipt,
)
from spritelab.product_features.harvest.catalog import (
    AUTOMATION_ALLOW_DECLARATIONS,
    INITIAL_LICENSE_POLICY,
    automation_terms_decision_identity,
    public_url,
    url_identity,
    validate_public_evidence_url,
    validate_public_hostname,
)
from spritelab.utils.safe_fs import AnchoredDirectory

MAX_EVIDENCE_PAGE_BYTES = 2 * 1024 * 1024
MAX_ROBOTS_BYTES = 512 * 1024
MAX_VISIBLE_TEXT_CHARACTERS = 200_000
MAX_PAGE_LINKS = 20_000
ROBOTS_USER_AGENT_TOKEN = "spritelab-harvest"
ROBOTS_MISSING_STATUSES = frozenset({404, 410})

_IGNORED_HTML_ELEMENTS = frozenset({"script", "style", "noscript", "template", "svg", "canvas"})
_BLOCK_HTML_ELEMENTS = frozenset(
    {
        "article",
        "dd",
        "div",
        "dt",
        "figcaption",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "li",
        "p",
        "section",
        "title",
    }
)
_UNIT_HTML_ELEMENTS = frozenset({"article", "dd", "dt", "li", "section"})
_SPACE = re.compile(r"\s+")
_PLAIN_URL = re.compile(r"https://[^\s<>\"']+")
_ROBOTS_FIELD = re.compile(r"^([A-Za-z][A-Za-z-]*):[ \t]*(.*)$")
_AUTOMATION_BLOCK_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bautomated (?:access|downloads?|downloading)\s*(?::|is|are)?\s*(?:not (?:allowed|permitted)|prohibited|forbidden|disallowed)\b",
        r"\b(?:scraping|crawling)(?:\s*(?:and|or|/)\s*(?:scraping|crawling))?\s*(?::|is|are)\s*(?:not (?:allowed|permitted)|prohibited|forbidden|disallowed)\b",
        r"\bbots?\s*(?::|is|are)\s*(?:not (?:allowed|permitted)|prohibited|forbidden|disallowed)\b",
        r"\buse of automated (?:means|tools?|systems?)\s*(?::|is|are)\s*(?:not (?:allowed|permitted)|prohibited|forbidden|disallowed)\b",
        r"\bdo not (?:crawl|scrape|automate|use automated (?:means|tools?)|download automatically)\b",
        r"\bmay not (?:access(?:\s*(?:/|or)\s*use)?|use)\s+(?:the\s+|this\s+|our\s+)?(?:service|site|website|platform)\s+(?:by|through|using)\s+(?:automated|non-human)(?:\s*(?:or|/)\s*(?:automated|non-human))?\s+means\b",
        r"\bno (?:robots?|bots?|spiders?|scrapers?)(?:\s*(?:,|/|and|or)\s*(?:robots?|bots?|spiders?|scrapers?))*\b(?!\s+(?:is|are)\s+(?:prohibited|forbidden|disallowed|blocked))",
        r"\bno automated (?:access|downloads?|downloading)\b(?!\s+(?:is|are)\s+(?:prohibited|forbidden|disallowed|blocked))",
        r"\b(?:systematic|bulk)(?:\s*(?:and|or|/)\s*(?:systematic|bulk))?\s+downloading\s*(?::|is|are)?\s*(?:not (?:allowed|permitted)|prohibited|forbidden|disallowed)\b",
        r"\bautomated data (?:collection|extraction|mining)(?:\s*(?:,|and|or|/)\s*(?:collection|extraction|mining))*\s*(?::|is|are)?\s*(?:not (?:allowed|permitted)|prohibited|forbidden|disallowed)\b",
        r"\bmust not use (?:scripts?|robots?|bots?|spiders?|scrapers?)\s+to\s+download\b",
    )
)
_GOVERNING_TERMS_LABEL = re.compile(
    r"^(?:(?:read|view|site|website|service) )?(?:terms(?: of (?:service|use))?|tos|legal(?: use)?|"
    r"acceptable use(?: policy)?|automation (?:terms|policy)|bot policy|crawler policy|scraping policy|"
    r"usage policy|terms and conditions|conditions of use|user agreement|"
    r"privacy\s*(?:&|and|/)\s*terms|terms\s*(?:&|and|/)\s*(?:privacy|policies|legal)|"
    r"legal\s*(?:&|and|/)\s*terms)$",
    re.IGNORECASE,
)
_GOVERNING_TERMS_PATH = re.compile(
    r"(?:^|[-_/])(?:terms(?:-of-(?:service|use))?|tos|legal(?:-use)?|acceptable-use|automation-policy|"
    r"bot-policy|crawler-policy|scraping-policy|usage-policy|conditions-of-use|user-agreement|"
    r"website-terms|privacy-terms|terms-privacy)(?:[-_/.]|$)",
    re.IGNORECASE,
)
_NON_TERMS_LABEL = re.compile(r"\b(?:license|licence|legal code|copyright)\b", re.IGNORECASE)
_NON_TERMS_PATH = re.compile(r"(?:^|[-_/])(?:license|licence|licenses|licences|legalcode)(?:[-_/.]|$)", re.IGNORECASE)
_AUTOMATION_DOUBLE_NEGATIVE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bno (?:bots?|automated (?:access|downloads?|downloading))\s+(?:is|are)\s+(?:prohibited|forbidden|disallowed|blocked)\b",
        r"\b(?:bots?|automated (?:access|downloads?|downloading)|scraping|crawling)\s+(?:is|are)\s+not\s+(?:prohibited|forbidden|disallowed|blocked)\b",
        r"\b(?:we\s+)?do not (?:prohibit|forbid|disallow|block)\s+(?:bots?|automated (?:access|downloads?|downloading)|scraping|crawling)\b",
        r"\b(?:systematic|bulk)(?:\s*(?:and|or|/)\s*(?:systematic|bulk))?\s+downloading\s+(?:is|are)\s+not\s+(?:prohibited|forbidden|disallowed|blocked)\b",
        r"\bautomated data (?:collection|extraction|mining)(?:\s*(?:,|and|or|/)\s*(?:collection|extraction|mining))*\s+(?:is|are)\s+not\s+(?:prohibited|forbidden|disallowed|blocked)\b",
        r"\b(?:we\s+)?do not (?:prohibit|forbid|disallow|block)\s+(?:(?:systematic|bulk) downloading|automated data (?:collection|extraction|mining)|the use of scripts to download)\b",
    )
)


class EvidenceFetchError(ValueError):
    """Remote evidence or its deterministic interpretation failed closed."""


@dataclass(frozen=True)
class FetchSnapshot:
    request_url_sha256: str
    request_public_url: str
    final_url: str
    http_status: int
    mime_type: str
    byte_count: int
    content_sha256: str
    elapsed_seconds: float
    relative_file: str

    @classmethod
    def from_result(cls, request_url: str, relative_file: str, result: ReceiptDownloadResult) -> FetchSnapshot:
        receipt = result.receipt
        return cls(
            request_url_sha256=url_identity(request_url),
            request_public_url=public_url(request_url),
            final_url=public_url(receipt.final_url),
            http_status=receipt.http_status,
            mime_type=receipt.response_mime_type,
            byte_count=receipt.response_bytes,
            content_sha256=receipt.response_sha256,
            elapsed_seconds=receipt.elapsed_seconds,
            relative_file=relative_file,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_url_sha256": self.request_url_sha256,
            "request_public_url": self.request_public_url,
            "final_url": self.final_url,
            "http_status": self.http_status,
            "mime_type": self.mime_type,
            "byte_count": self.byte_count,
            "content_sha256": self.content_sha256,
            "elapsed_seconds": round(self.elapsed_seconds, 6),
            "relative_file": self.relative_file,
            "redirect_count": 0,
        }


@dataclass(frozen=True)
class RobotsRule:
    directive: str
    pattern: str


@dataclass(frozen=True)
class RobotsDecision:
    request_url_sha256: str
    request_public_url: str
    allowed: bool
    matched_directive: str | None
    matched_pattern: str | None
    matched_length: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_url_sha256": self.request_url_sha256,
            "request_public_url": self.request_public_url,
            "allowed": self.allowed,
            "matched_directive": self.matched_directive,
            "matched_pattern": self.matched_pattern,
            "matched_length": self.matched_length,
        }


@dataclass(frozen=True)
class RobotsSnapshot:
    origin: str
    fetch: FetchSnapshot
    policy: str
    rules: tuple[RobotsRule, ...]

    @property
    def identity(self) -> str:
        payload = (
            self.origin,
            self.fetch.request_url_sha256,
            self.fetch.http_status,
            self.fetch.content_sha256,
            self.policy,
            tuple((rule.directive, rule.pattern) for rule in self.rules),
            HARVEST_USER_AGENT,
        )
        return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()

    def evaluate(self, request_url: str) -> RobotsDecision:
        parsed = canonical_https_url(request_url, allow_query=True)
        if url_origin(parsed) != self.origin:
            raise EvidenceFetchError("Robots policy origin does not match the requested URL.")
        target = parsed.path or "/"
        if parsed.query:
            target = f"{target}?{parsed.query}"
        matches: list[tuple[int, bool, RobotsRule]] = []
        for rule in self.rules:
            matched_length = _robots_match_length(rule.pattern, target)
            if matched_length is not None:
                matches.append((matched_length, rule.directive == "allow", rule))
        if not matches:
            decision = RobotsDecision(url_identity(request_url), public_url(request_url), True, None, None, 0)
        else:
            # RFC 9309: the most specific octet match wins; Allow wins an exact tie.
            matched_length, allowed, rule = max(matches, key=lambda item: (item[0], item[1]))
            decision = RobotsDecision(
                url_identity(request_url),
                public_url(request_url),
                allowed,
                rule.directive,
                rule.pattern,
                matched_length,
            )
        if not decision.allowed:
            raise EvidenceFetchError("Robots policy explicitly disallows this Harvest request path.")
        return decision

    def to_dict(self, decisions: tuple[RobotsDecision, ...]) -> dict[str, Any]:
        return {
            "schema_version": "spritelab.harvest.robots-evidence.v1",
            "origin": self.origin,
            "user_agent": HARVEST_USER_AGENT,
            "user_agent_token": ROBOTS_USER_AGENT_TOKEN,
            "policy": self.policy,
            "fetch": self.fetch.to_dict(),
            "decisions": [decision.to_dict() for decision in decisions],
            "robots_identity": self.identity,
        }


@dataclass(frozen=True)
class VerifiedPageEvidence:
    source_snapshot: FetchSnapshot
    license_snapshot: FetchSnapshot
    license_evidence_text: str
    direct_download_url: str
    direct_download_host: str
    source_pack_evidence_text: str
    zero_cost_verified: bool
    license_conflict_checked: bool
    verification_identity: str


@dataclass(frozen=True)
class AutomationTermsEvidence:
    mode: str
    decision: str
    evidence_url: str
    content_sha256: str
    matched_declaration: str | None
    limited_evidence: bool
    decision_identity: str

    def to_dict(self, snapshot: FetchSnapshot | None) -> dict[str, Any]:
        return {
            "schema_version": "spritelab.harvest.automation-terms-evidence.v1",
            "mode": self.mode,
            "decision": self.decision,
            "evidence_url": self.evidence_url,
            "content_sha256": self.content_sha256,
            "matched_declaration": self.matched_declaration,
            "limited_evidence": self.limited_evidence,
            "fetch": snapshot.to_dict() if snapshot is not None else None,
            "decision_identity": self.decision_identity,
            "robots_permission_treated_as_terms_permission": False,
        }


@dataclass(frozen=True)
class _PageBlock:
    """One completed visible block with every proper descendant block index."""

    tag: str
    text: str
    links: tuple[str, ...]
    descendants: tuple[int, ...]


class _InertHTMLParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self._ignored_depth = 0
        self._title_depth = 0
        self._text: list[str] = []
        self._title: list[str] = []
        self.links: list[str] = []
        self.link_labels: list[tuple[str, str]] = []
        self.segments: list[str] = []
        self.linked_segments: list[_PageBlock] = []
        self._segment_buffers: list[tuple[str, list[str], list[str], list[int]]] = []
        self._anchor_buffers: list[tuple[str, list[str]]] = []
        self._characters = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized = tag.casefold()
        if normalized in _IGNORED_HTML_ELEMENTS:
            self._ignored_depth += 1
            return
        if self._ignored_depth:
            return
        if normalized == "title":
            self._title_depth += 1
        if normalized in _BLOCK_HTML_ELEMENTS:
            self._segment_buffers.append((normalized, [], [], []))
        if normalized == "a" and len(self.links) < MAX_PAGE_LINKS:
            hrefs = [value for name, value in attrs if name.casefold() == "href" and value]
            if len(hrefs) == 1:
                try:
                    link = canonical_url_string(urljoin(self.base_url, hrefs[0]), allow_query=True)
                except ValueError:
                    pass
                else:
                    self.links.append(link)
                    for _tag, _values, block_links, _descendants in self._segment_buffers:
                        block_links.append(link)
                    self._anchor_buffers.append((link, []))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag.casefold() in _IGNORED_HTML_ELEMENTS:
            self._ignored_depth = max(0, self._ignored_depth - 1)
        elif tag.casefold() == "a" and self._anchor_buffers:
            link, values = self._anchor_buffers.pop()
            self.link_labels.append((link, _normalize_text(" ".join(values))))

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.casefold()
        if normalized in _IGNORED_HTML_ELEMENTS:
            self._ignored_depth = max(0, self._ignored_depth - 1)
            return
        if not self._ignored_depth and normalized == "title":
            self._title_depth = max(0, self._title_depth - 1)
        if not self._ignored_depth and normalized == "a" and self._anchor_buffers:
            link, values = self._anchor_buffers.pop()
            self.link_labels.append((link, _normalize_text(" ".join(values))))
        if not self._ignored_depth and normalized in _BLOCK_HTML_ELEMENTS:
            for index in range(len(self._segment_buffers) - 1, -1, -1):
                block_tag, values, block_links, descendants = self._segment_buffers[index]
                if block_tag == normalized:
                    del self._segment_buffers[index]
                    segment = _normalize_text(" ".join(values))
                    if segment:
                        block_index = len(self.linked_segments)
                        self.segments.append(segment)
                        self.linked_segments.append(
                            _PageBlock(block_tag, segment, tuple(sorted(set(block_links))), tuple(descendants))
                        )
                        for _tag, _values, _links, open_descendants in self._segment_buffers[:index]:
                            open_descendants.append(block_index)
                    break

    def handle_data(self, data: str) -> None:
        if self._ignored_depth or self._characters >= MAX_VISIBLE_TEXT_CHARACTERS:
            return
        remaining = MAX_VISIBLE_TEXT_CHARACTERS - self._characters
        retained = data[:remaining]
        self._characters += len(retained)
        self._text.append(retained)
        for _tag, values, _links, _descendants in self._segment_buffers:
            values.append(retained)
        for _link, values in self._anchor_buffers:
            values.append(retained)
        if self._title_depth:
            self._title.append(retained)

    @property
    def text(self) -> str:
        return _normalize_text(" ".join((*self._title, *self._text)))


def canonical_https_url(value: str, *, allow_query: bool) -> SplitResult:
    parsed = validate_public_evidence_url(value)
    if parsed.scheme.casefold() != "https":
        raise ValueError("Harvest onboarding URLs must use HTTPS.")
    if parsed.fragment:
        raise ValueError("Harvest onboarding URLs cannot contain fragments.")
    if not allow_query and parsed.query:
        raise ValueError("Harvest evidence page URLs cannot contain private query data.")
    return parsed


def canonical_url_string(value: str, *, allow_query: bool) -> str:
    parsed = canonical_https_url(value, allow_query=allow_query)
    host = (parsed.hostname or "").casefold().rstrip(".")
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    return urlunsplit(("https", host, parsed.path or "/", parsed.query if allow_query else "", ""))


def url_origin(value: str | SplitResult) -> str:
    parsed = canonical_https_url(value, allow_query=True) if isinstance(value, str) else value
    host = (parsed.hostname or "").casefold().rstrip(".")
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    return f"https://{host}"


def fetch_robots_snapshot(
    request_url: str,
    destination_anchor: AnchoredDirectory,
    destination_name: str,
    *,
    cancel_requested: Any,
    progress: Any = None,
    resolver: HostResolver | None = None,
    transport: PinnedHTTPTransport | None = None,
    timeout_seconds: float = 30.0,
    downloader: Any = download_file_with_receipt,
) -> RobotsSnapshot:
    parsed = canonical_https_url(request_url, allow_query=True)
    origin = url_origin(parsed)
    robots_url = f"{origin}/robots.txt"
    host = (parsed.hostname or "").casefold().rstrip(".")
    result = _download(
        downloader,
        robots_url,
        destination_anchor,
        destination_name,
        allowed_hosts=(host,),
        allowed_content_types=("text/plain", "text/html"),
        accepted_http_statuses=tuple(sorted(ROBOTS_MISSING_STATUSES)),
        max_bytes=MAX_ROBOTS_BYTES,
        timeout_seconds=timeout_seconds,
        cancel_requested=cancel_requested,
        progress=progress,
        resolver=resolver,
        transport=transport,
    )
    snapshot = FetchSnapshot.from_result(robots_url, destination_name, result)
    if snapshot.http_status in ROBOTS_MISSING_STATUSES:
        return RobotsSnapshot(origin, snapshot, "missing_policy_allow", ())
    if snapshot.http_status != 200 or snapshot.mime_type != "text/plain":
        raise EvidenceFetchError("Robots policy was not an unambiguous successful text/plain response.")
    payload = _read_anchored_bytes(destination_anchor, destination_name, MAX_ROBOTS_BYTES)
    rules = _parse_robots(payload)
    return RobotsSnapshot(origin, snapshot, "parsed_policy", rules)


def fetch_evidence_page(
    request_url: str,
    destination_anchor: AnchoredDirectory,
    destination_name: str,
    *,
    cancel_requested: Any,
    progress: Any = None,
    resolver: HostResolver | None = None,
    transport: PinnedHTTPTransport | None = None,
    timeout_seconds: float = 60.0,
    downloader: Any = download_file_with_receipt,
) -> FetchSnapshot:
    parsed = canonical_https_url(request_url, allow_query=False)
    host = (parsed.hostname or "").casefold().rstrip(".")
    result = _download(
        downloader,
        canonical_url_string(request_url, allow_query=False),
        destination_anchor,
        destination_name,
        allowed_hosts=(host,),
        allowed_content_types=("text/html", "text/plain"),
        accepted_http_statuses=(),
        max_bytes=MAX_EVIDENCE_PAGE_BYTES,
        timeout_seconds=timeout_seconds,
        cancel_requested=cancel_requested,
        progress=progress,
        resolver=resolver,
        transport=transport,
    )
    return FetchSnapshot.from_result(request_url, destination_name, result)


def verify_evidence_pages(
    *,
    source_url: str,
    source_snapshot: FetchSnapshot,
    source_bytes: bytes,
    license_url: str,
    license_snapshot: FetchSnapshot,
    license_bytes: bytes,
    title: str,
    creator: str,
    license_id: str,
    direct_download_url: str,
) -> VerifiedPageEvidence:
    _source_text, source_links, source_segments, _source_link_labels, source_blocks = _parse_page(
        source_bytes, source_snapshot.mime_type, source_url
    )
    license_text, _license_links, _license_segments, _license_link_labels, _license_blocks = _parse_page(
        license_bytes, license_snapshot.mime_type, license_url
    )
    normalized_title = _normalize_text(title)
    normalized_creator = _normalize_text(creator)
    if not _provenance_segment_matches(normalized_title, source_segments, label="title"):
        raise EvidenceFetchError("The source page does not visibly identify the submitted title.")
    if not _provenance_segment_matches(normalized_creator, source_segments, label="creator"):
        raise EvidenceFetchError("The source page does not visibly identify the submitted creator.")
    normalized_license = license_id.strip().casefold()
    if normalized_license not in INITIAL_LICENSE_POLICY:
        raise EvidenceFetchError("Only CC0-1.0 or explicit public-domain onboarding is allowed.")
    folded_license = license_text.casefold()
    if _license_conflict(folded_license, normalized_license):
        raise EvidenceFetchError("The retained license page contains conflicting or restrictive rights language.")
    declared = _license_declared(folded_license, normalized_license)
    if not declared:
        raise EvidenceFetchError("The retained license page does not declare the selected permissive license.")
    canonical_license_url = canonical_url_string(license_url, allow_query=False)
    canonical_direct = canonical_url_string(direct_download_url, allow_query=True)
    if canonical_direct not in source_links:
        raise EvidenceFetchError("The creator source page does not contain the exact submitted direct download link.")
    link_block_indices = {index for index, block in enumerate(source_blocks) if canonical_direct in block.links}
    minimal_link_blocks = {
        index for index in link_block_indices if not link_block_indices.intersection(source_blocks[index].descendants)
    }
    if len(minimal_link_blocks) > 1:
        raise EvidenceFetchError(
            "The submitted direct download link appears in multiple distinct source-page blocks; "
            "the source-pack binding is ambiguous."
        )
    # Ancestor containers must not compose provenance from one sibling card
    # with license/zero-cost/download evidence from another. Any strict
    # descendant that is a self-contained unit element and supplies required
    # evidence disqualifies the enclosing block; plain wrappers (div/p/...)
    # inside a single card carry no unit boundary and still bind normally.
    evidence_unit_indices = {
        index
        for index, block in enumerate(source_blocks)
        if block.tag in _UNIT_HTML_ELEMENTS
        and _supplies_pack_evidence(
            block,
            direct_url=canonical_direct,
            license_page_url=canonical_license_url,
            title=normalized_title,
            creator=normalized_creator,
            license_id=normalized_license,
        )
    }
    matching_blocks = [
        block
        for block in source_blocks
        if canonical_direct in block.links
        and _contains_bound_phrase(block.text, normalized_title)
        and _contains_bound_phrase(block.text, normalized_creator)
    ]
    unit_bound_blocks = [
        block for block in matching_blocks if not evidence_unit_indices.intersection(block.descendants)
    ]
    if matching_blocks and not unit_bound_blocks:
        raise EvidenceFetchError(
            "The source page composes provenance, license, or zero-cost evidence across sibling "
            "source-pack units instead of one unambiguous unit."
        )
    if not unit_bound_blocks:
        raise EvidenceFetchError(
            "The exact title, creator, license, zero-cost declaration, and download link are not bound to one source-pack block."
        )
    chosen = min(unit_bound_blocks, key=lambda block: (len(block.text), block.tag, block.text))
    pack_text, pack_links = chosen.text, chosen.links
    folded_pack = pack_text.casefold()
    if _license_conflict(folded_pack, normalized_license):
        raise EvidenceFetchError("The retained source-pack block contains conflicting or restrictive license language.")
    if not _license_declared(folded_pack, normalized_license) and canonical_license_url not in pack_links:
        raise EvidenceFetchError(
            "The retained source-pack block neither declares the selected license nor links its exact evidence page."
        )
    if _paid_conflict(folded_pack) or not _zero_cost_declared(folded_pack):
        raise EvidenceFetchError(
            "The retained source-pack block does not provide conflict-free explicit zero-cost evidence."
        )
    direct = canonical_https_url(canonical_direct, allow_query=True)
    host = (direct.hostname or "").casefold().rstrip(".")
    validate_public_hostname(host)
    license_excerpt = license_text[:4000].strip()
    if len(license_excerpt) < 2:
        raise EvidenceFetchError("The retained license evidence text is empty.")
    source_pack_excerpt = pack_text[:4000].strip()
    identity_payload = "\n".join(
        (
            source_snapshot.content_sha256,
            license_snapshot.content_sha256,
            normalized_title,
            normalized_creator,
            normalized_license,
            url_identity(canonical_direct),
            license_excerpt,
            source_pack_excerpt,
            "zero_cost_verified",
            "license_conflict_checked",
        )
    )
    return VerifiedPageEvidence(
        source_snapshot=source_snapshot,
        license_snapshot=license_snapshot,
        license_evidence_text=license_excerpt,
        direct_download_url=canonical_direct,
        direct_download_host=host,
        source_pack_evidence_text=source_pack_excerpt,
        zero_cost_verified=True,
        license_conflict_checked=True,
        verification_identity=hashlib.sha256(identity_payload.encode("utf-8")).hexdigest(),
    )


def verify_automation_terms(
    *,
    source_url: str,
    source_bytes: bytes,
    source_mime_type: str,
    source_content_sha256: str,
    terms_url: str | None,
    terms_bytes: bytes | None = None,
    terms_snapshot: FetchSnapshot | None = None,
) -> AutomationTermsEvidence:
    """Classify retained source-bound terms without treating silence as permission."""

    source_text, source_links, source_segments, source_link_labels, _source_blocks = _parse_page(
        source_bytes,
        source_mime_type,
        source_url,
    )
    governing_links = _governing_terms_links(source_link_labels)
    if len(governing_links) > 1:
        raise EvidenceFetchError(
            "The source page exposes multiple possible governing terms links; single-page verification is ambiguous."
        )
    if terms_url is None:
        if governing_links:
            raise EvidenceFetchError(
                "The source page links a likely governing terms or legal-use page; submit that exact URL for review."
            )
        evidence_url = canonical_url_string(source_url, allow_query=False)
        segments = source_segments
        evidence_text = source_text
        content_sha256 = source_content_sha256
        mode = "source_page_no_governing_terms_link"
    else:
        canonical_terms = canonical_url_string(terms_url, allow_query=False)
        if canonical_terms not in source_links:
            raise EvidenceFetchError("The source page does not link the exact submitted automation terms page.")
        if not governing_links:
            raise EvidenceFetchError(
                "A submitted automation terms URL must be the independently detected governing terms link."
            )
        if canonical_terms not in governing_links:
            raise EvidenceFetchError("The submitted automation terms page is not the detected governing terms link.")
        if terms_bytes is None or terms_snapshot is None:
            raise EvidenceFetchError("Automation terms evidence is incomplete.")
        evidence_text, _terms_links, segments, _terms_link_labels, _terms_blocks = _parse_page(
            terms_bytes,
            terms_snapshot.mime_type,
            canonical_terms,
        )
        evidence_url = canonical_terms
        content_sha256 = terms_snapshot.content_sha256
        mode = "linked_terms_page"
    source_folded_segments = tuple(_normalize_text(segment).casefold() for segment in source_segments)
    folded_segments = tuple(_normalize_text(segment).casefold() for segment in segments)
    # Prohibitions fail closed over the complete visible page text as well as
    # exact block segments. HTMLParser only finishes a segment when it sees a
    # matching block end tag, so direct body/table text and malformed or
    # unclosed blocks must not disappear from the prohibition decision. Exact
    # segments remain the sole ALLOW input: surrounding prose cannot fabricate
    # an affirmative declaration.
    block_segments = tuple(
        dict.fromkeys(
            (
                _normalize_text(source_text).casefold(),
                _normalize_text(evidence_text).casefold(),
                *source_folded_segments,
                *folded_segments,
            )
        )
    )
    blocked = sorted({segment for segment in block_segments if _automation_prohibition(segment)})
    allowed = sorted(set(folded_segments) & AUTOMATION_ALLOW_DECLARATIONS)
    if blocked:
        declaration = blocked[0]
        decision = "BLOCK"
    elif allowed:
        declaration = allowed[0]
        decision = "ALLOW"
    else:
        declaration = None
        decision = "NO_PROHIBITION_OBSERVED"
    limited = decision == "NO_PROHIBITION_OBSERVED"
    identity = automation_terms_decision_identity(
        mode=mode,
        evidence_url=evidence_url,
        content_sha256=content_sha256,
        matched_declaration=declaration,
        decision=decision,
    )
    return AutomationTermsEvidence(
        mode=mode,
        decision=decision,
        evidence_url=evidence_url,
        content_sha256=content_sha256,
        matched_declaration=declaration,
        limited_evidence=limited,
        decision_identity=identity,
    )


def read_snapshot_bytes(anchor: AnchoredDirectory, name: str, *, max_bytes: int) -> bytes:
    return _read_anchored_bytes(anchor, name, max_bytes)


def recover_direct_link(
    source_bytes: bytes,
    *,
    mime_type: str,
    source_url: str,
    expected_url_sha256: str,
) -> str:
    """Recover one exact creator-posted link from retained inert page bytes."""

    _text, links, _segments, _link_labels, _blocks = _parse_page(source_bytes, mime_type, source_url)
    matches = tuple(link for link in links if url_identity(link) == expected_url_sha256)
    if len(matches) != 1:
        raise EvidenceFetchError("Retained source evidence does not identify one exact direct download link.")
    return matches[0]


def rebuild_robots_snapshot(origin: str, fetch: FetchSnapshot, payload: bytes) -> RobotsSnapshot:
    """Reparse a retained robots snapshot for promotion-time verification."""

    if hashlib.sha256(payload).hexdigest() != fetch.content_sha256 or len(payload) != fetch.byte_count:
        raise EvidenceFetchError("Retained robots evidence changed after the probe.")
    if fetch.http_status in ROBOTS_MISSING_STATUSES:
        return RobotsSnapshot(origin, fetch, "missing_policy_allow", ())
    if fetch.http_status != 200 or fetch.mime_type != "text/plain":
        raise EvidenceFetchError("Retained robots policy is not an unambiguous text policy.")
    return RobotsSnapshot(origin, fetch, "parsed_policy", _parse_robots(payload))


def _download(downloader: Any, url: str, anchor: AnchoredDirectory, name: str, **kwargs: Any) -> ReceiptDownloadResult:
    anchor.verify()
    try:
        return downloader(
            url,
            anchor.directory / name,
            overwrite=False,
            require_https=True,
            max_redirects=0,
            destination_anchor=anchor,
            **kwargs,
        )
    except DownloadCancelled:
        raise
    except (DownloadSecurityError, OSError, ValueError) as exc:
        raise EvidenceFetchError("A bounded pinned Harvest evidence request failed.") from exc


def _read_anchored_bytes(anchor: AnchoredDirectory, name: str, max_bytes: int) -> bytes:
    before = anchor.lstat(name)
    if before.st_nlink != 1 or before.st_size < 0 or before.st_size > max_bytes:
        raise EvidenceFetchError("Retained Harvest evidence is unsafe or oversized.")
    descriptor = anchor.open_file(name, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    try:
        opened = os.fstat(descriptor)
        if _file_identity(before) != _file_identity(opened):
            raise EvidenceFetchError("Retained Harvest evidence changed while opening.")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read(max_bytes + 1)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    path_after = anchor.lstat(name)
    if (
        len(payload) != before.st_size
        or _file_identity(before) != _file_identity(after)
        or _file_identity(before) != _file_identity(path_after)
    ):
        raise EvidenceFetchError("Retained Harvest evidence changed while reading.")
    return payload


def _parse_page(
    payload: bytes,
    mime_type: str,
    base_url: str,
) -> tuple[
    str,
    tuple[str, ...],
    tuple[str, ...],
    tuple[tuple[str, str], ...],
    tuple[_PageBlock, ...],
]:
    try:
        decoded = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise EvidenceFetchError("Evidence pages must be strict UTF-8.") from exc
    if mime_type == "text/plain":
        text = _normalize_text(decoded[:MAX_VISIBLE_TEXT_CHARACTERS])
        links: set[str] = set()
        for match in _PLAIN_URL.finditer(decoded):
            try:
                links.add(canonical_url_string(match.group(0).rstrip(".,);]"), allow_query=True))
            except ValueError:
                continue
        segments = tuple(_normalize_text(line) for line in decoded.splitlines() if _normalize_text(line))
        blocks: list[_PageBlock] = []
        for line in segments:
            line_links = tuple(sorted(link for link in links if link in line))
            blocks.append(_PageBlock("line", line, line_links, ()))
        return (
            text,
            tuple(sorted(links)),
            segments,
            tuple((link, "") for link in sorted(links)),
            tuple(blocks),
        )
    if mime_type != "text/html":
        raise EvidenceFetchError("Evidence page MIME type is not supported.")
    parser = _InertHTMLParser(base_url)
    try:
        parser.feed(decoded)
        parser.close()
    except Exception as exc:
        raise EvidenceFetchError("Evidence HTML is malformed or ambiguous.") from exc
    segments = tuple(dict.fromkeys(_normalize_text(value) for value in parser.segments if _normalize_text(value)))
    labels = dict.fromkeys(parser.links, "")
    for link, label in parser.link_labels:
        if label:
            labels[link] = label
    return (
        parser.text,
        tuple(sorted(labels)),
        segments,
        tuple(sorted(labels.items())),
        tuple(parser.linked_segments),
    )


def _governing_terms_links(link_labels: tuple[tuple[str, str], ...]) -> frozenset[str]:
    candidates: set[str] = set()
    for link, label in link_labels:
        path = canonical_https_url(link, allow_query=True).path.casefold()
        normalized_label = _normalize_text(label).casefold()
        if _NON_TERMS_LABEL.search(normalized_label) is not None or _NON_TERMS_PATH.search(path) is not None:
            continue
        if (
            _GOVERNING_TERMS_LABEL.fullmatch(normalized_label) is not None
            or _GOVERNING_TERMS_PATH.search(path) is not None
        ):
            candidates.add(link)
    return frozenset(candidates)


def _automation_prohibition(segment: str) -> bool:
    candidate = segment
    for pattern in _AUTOMATION_DOUBLE_NEGATIVE_PATTERNS:
        candidate = pattern.sub(" ", candidate)
    return any(pattern.search(candidate) is not None for pattern in _AUTOMATION_BLOCK_PATTERNS)


def _parse_robots(payload: bytes) -> tuple[RobotsRule, ...]:
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise EvidenceFetchError("Robots policy must be strict UTF-8.") from exc
    groups: list[tuple[list[str], list[RobotsRule]]] = []
    agents: list[str] = []
    rules: list[RobotsRule] = []
    saw_rule = False
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            if agents:
                groups.append((agents, rules))
                agents, rules, saw_rule = [], [], False
            continue
        match = _ROBOTS_FIELD.fullmatch(line)
        if match is None:
            raise EvidenceFetchError("Robots policy contains a malformed directive.")
        field, raw_value = match.group(1).casefold(), match.group(2).strip()
        if any(ord(character) < 32 and character not in "\t" for character in raw_value):
            raise EvidenceFetchError("Robots policy contains control characters.")
        if field == "user-agent":
            token = raw_value.casefold()
            if not token or any(character.isspace() for character in token):
                raise EvidenceFetchError("Robots policy contains an ambiguous user-agent token.")
            if saw_rule:
                groups.append((agents, rules))
                agents, rules, saw_rule = [], [], False
            agents.append(token)
        elif field in {"allow", "disallow"}:
            if not agents:
                raise EvidenceFetchError("Robots policy rule has no user-agent group.")
            saw_rule = True
            if field == "disallow" and raw_value == "":
                continue
            if not raw_value.startswith("/"):
                raise EvidenceFetchError("Robots policy path rules must be absolute.")
            rules.append(RobotsRule(field, raw_value))
        elif field in {"sitemap", "crawl-delay", "host", "clean-param"}:
            continue
        else:
            # RFC extension fields are ignored only when syntactically valid.
            continue
    if agents:
        groups.append((agents, rules))
    exact = [tuple(group_rules) for group_agents, group_rules in groups if ROBOTS_USER_AGENT_TOKEN in group_agents]
    wildcard = [tuple(group_rules) for group_agents, group_rules in groups if "*" in group_agents]
    selected = exact if exact else wildcard
    return tuple(rule for group in selected for rule in group)


def _robots_match_length(pattern: str, target: str) -> int | None:
    anchored = pattern.endswith("$")
    body = pattern[:-1] if anchored else pattern
    expression = "^" + re.escape(body).replace(r"\*", ".*") + ("$" if anchored else "")
    match = re.search(expression, target)
    if match is None:
        return None
    return len(body.replace("*", "").encode("utf-8"))


def _normalize_text(value: str) -> str:
    return _SPACE.sub(" ", value).strip()


def _license_declared(folded_text: str, license_id: str) -> bool:
    if _license_conflict(folded_text, license_id):
        return False
    if license_id == "cc0-1.0":
        return "cc0" in folded_text and ("1.0" in folded_text or "public domain" in folded_text)
    return "public domain" in folded_text or "public-domain" in folded_text


def _license_conflict(folded_text: str, license_id: str) -> bool:
    patterns = (
        r"\ball rights reserved\b",
        r"\bpersonal use only\b",
        r"\bnon[- ]commercial(?: use)?(?: only)?\b",
        r"\bno[- ]derivatives?\b",
        r"\bno (?:copying|redistribution|adaptation|modification)\b",
        r"\b(?:copying|redistribution|adaptation|modification) (?:is|are) (?:not permitted|prohibited|forbidden)\b",
    )
    if any(re.search(pattern, folded_text) is not None for pattern in patterns):
        return True
    if license_id == "cc0-1.0":
        return re.search(r"\b(?:not|isn't|is not)\s+(?:licensed\s+(?:as|under)\s+)?cc0\b", folded_text) is not None
    return re.search(r"\b(?:not|isn't|is not)\s+(?:in\s+the\s+)?public[- ]domain\b", folded_text) is not None


def _zero_cost_declared(folded_text: str) -> bool:
    return any(
        re.search(pattern, folded_text) is not None
        for pattern in (
            r"\bzero[- ]cost\b",
            r"\bno[- ]cost\b",
            r"\bfree (?:download|to download|asset|pack|dataset)\b",
            r"\bprice\s*(?::|=|-)?\s*(?:0(?:\.00)?|[$€£]\s*0(?:\.00)?)\b",
        )
    )


def _paid_conflict(folded_text: str) -> bool:
    return any(
        re.search(pattern, folded_text) is not None
        for pattern in (
            r"\bnot (?:free|zero[- ]cost)\b",
            r"\b(?:purchase|required purchase|paid|subscription|trial)\b",
            r"\b(?:buy|purchase) (?:now|this|the|pack|asset|dataset)\b",
            r"(?:[$€£]\s*[1-9][0-9]*(?:\.[0-9]{1,2})?)",
            r"\bprice\s*(?::|=|-)\s*[1-9][0-9]*(?:\.[0-9]{1,2})?\b",
        )
    )


def _supplies_pack_evidence(
    block: _PageBlock,
    *,
    direct_url: str,
    license_page_url: str,
    title: str,
    creator: str,
    license_id: str,
) -> bool:
    """Report whether one unit block carries any evidence the binding relies on."""

    if direct_url in block.links or license_page_url in block.links:
        return True
    if _contains_bound_phrase(block.text, title) or _contains_bound_phrase(block.text, creator):
        return True
    folded = block.text.casefold()
    if _zero_cost_declared(folded):
        return True
    if license_id == "cc0-1.0":
        return "cc0" in folded
    return "public domain" in folded or "public-domain" in folded


def _contains_bound_phrase(text: str, value: str) -> bool:
    folded_text = text.casefold()
    folded_value = value.casefold()
    return re.search(rf"(?<!\w){re.escape(folded_value)}(?!\w)", folded_text) is not None


def _provenance_segment_matches(value: str, segments: tuple[str, ...], *, label: str) -> bool:
    minimum = 4 if label == "title" else 3
    if len(value) < minimum or not any(character.isalnum() for character in value):
        return False
    folded = value.casefold()
    if any(segment.casefold() == folded for segment in segments):
        return True
    escaped = re.escape(folded)
    if label == "creator":
        expression = re.compile(
            rf"(?:^|[^\w])(?:by|author|creator|artist)\s*(?::|-)?\s*(?<!\w){escaped}(?!\w)(?:$|\s*[|\u2022])",
            re.IGNORECASE,
        )
    else:
        expression = re.compile(
            rf"(?:^|[^\w])(?:title|name|asset|submission)\s*(?::|-)\s*(?<!\w){escaped}(?!\w)(?:$|\s*[|\u2022])",
            re.IGNORECASE,
        )
    return any(expression.search(segment.casefold()) is not None for segment in segments)


def _file_identity(metadata: Any) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
    )


__all__ = [
    "MAX_EVIDENCE_PAGE_BYTES",
    "MAX_ROBOTS_BYTES",
    "ROBOTS_USER_AGENT_TOKEN",
    "AutomationTermsEvidence",
    "EvidenceFetchError",
    "FetchSnapshot",
    "RobotsDecision",
    "RobotsSnapshot",
    "VerifiedPageEvidence",
    "canonical_https_url",
    "canonical_url_string",
    "fetch_evidence_page",
    "fetch_robots_snapshot",
    "read_snapshot_bytes",
    "rebuild_robots_snapshot",
    "recover_direct_link",
    "url_origin",
    "verify_automation_terms",
    "verify_evidence_pages",
]
