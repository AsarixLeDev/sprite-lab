"""Pluggable structural embeddings, durable cache, and exact retrieval."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import struct
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from PIL import Image

from spritelab.hierarchical_labeling.contracts import (
    HumanReferenceLabel,
    RetrievalEvidence,
    RetrievalNeighbor,
    TechnicalVisualEvidence,
)
from spritelab.hierarchical_labeling.json_utils import (
    ContractValidationError,
    StrictRecord,
    content_identity,
    require_finite,
    require_text,
    require_unique_text,
)
from spritelab.hierarchical_labeling.renders import RenderType, RenderView

EMBEDDING_STORE_SCHEMA_VERSION = "spritelab-embedding-store-v1"
EXACT_INDEX_VERSION = "spritelab-numpy-exact-retrieval-v1"


@dataclass(frozen=True)
class EmbeddingCapabilities:
    representations: tuple[str, ...]
    maximum_batch_size: int
    device_requirements: str
    resumable: bool = True

    def __post_init__(self) -> None:
        require_unique_text(self.representations, "embedding representations")
        if type(self.maximum_batch_size) is not int or self.maximum_batch_size < 1:
            raise ContractValidationError("embedding maximum batch size must be positive")
        require_text(self.device_requirements, "embedding device requirements")


@dataclass(frozen=True)
class EmbeddingSample:
    record_identity: str
    image_identity: str
    technical: TechnicalVisualEvidence
    views: tuple[RenderView, ...] = ()

    def __post_init__(self) -> None:
        require_text(self.record_identity, "embedding record identity")
        require_text(self.image_identity, "embedding image identity")
        if (
            self.technical.record_identity != self.record_identity
            or self.technical.image_identity != self.image_identity
        ):
            raise ContractValidationError("embedding sample technical identity mismatch")


@dataclass(frozen=True, eq=False)
class EmbeddingVector(StrictRecord):
    SCHEMA_VERSION = "spritelab.labeling.embedding-vector.v1"
    IDENTITY_FIELDS = ("record_identity", "image_identity", "backend_identity", "representation")

    record_identity: str
    image_identity: str
    backend_identity: str
    model_identity: str
    representation: str
    vector: tuple[float, ...]

    def __post_init__(self) -> None:
        for name in ("record_identity", "image_identity", "backend_identity", "model_identity", "representation"):
            require_text(getattr(self, name), name.replace("_", " "))
        if not self.vector:
            raise ContractValidationError("embedding vector cannot be empty")
        for index, value in enumerate(self.vector):
            require_finite(value, f"embedding vector[{index}]")
        self.validate_record()

    def array(self) -> np.ndarray:
        return np.asarray(self.vector, dtype=np.float32)


@runtime_checkable
class EmbeddingBackend(Protocol):
    backend_id: str
    model_identity: str
    capabilities: EmbeddingCapabilities

    @property
    def cache_identity(self) -> str: ...

    def embed_images(self, samples: Sequence[EmbeddingSample]) -> tuple[EmbeddingVector, ...]: ...

    def embed_views(self, samples: Sequence[EmbeddingSample]) -> tuple[EmbeddingVector, ...]: ...

    def health(self) -> Mapping[str, Any]: ...

    def cancel(self) -> bool: ...


class DeterministicMockEmbeddingBackend:
    """Small deterministic backend used only for tests and synthetic fixtures."""

    backend_id = "deterministic_mock"

    def __init__(self, *, dimensions: int = 16, model_identity: str = "mock-embedding-v1") -> None:
        if not 2 <= dimensions <= 1024:
            raise ValueError("mock embedding dimensions must be from 2 through 1024")
        self.dimensions = dimensions
        self.model_identity = model_identity
        self.capabilities = EmbeddingCapabilities(("mock",), 4096, "cpu_only")
        self._cancelled = False

    @property
    def cache_identity(self) -> str:
        return content_identity(
            "spritelab-embedding-backend-v1",
            {"backend_id": self.backend_id, "model_identity": self.model_identity, "dimensions": self.dimensions},
        )

    def _vector(self, sample: EmbeddingSample, suffix: str) -> EmbeddingVector:
        digest = hashlib.shake_256(f"{sample.image_identity}:{suffix}".encode()).digest(self.dimensions * 2)
        values = np.frombuffer(digest, dtype=">u2").astype(np.float32) / 32767.5 - 1.0
        values = _normalized(values)
        return EmbeddingVector(
            sample.record_identity,
            sample.image_identity,
            self.cache_identity,
            self.model_identity,
            "mock",
            tuple(float(value) for value in values),
        )

    def embed_images(self, samples: Sequence[EmbeddingSample]) -> tuple[EmbeddingVector, ...]:
        if self._cancelled:
            return ()
        return tuple(self._vector(sample, "image") for sample in samples)

    def embed_views(self, samples: Sequence[EmbeddingSample]) -> tuple[EmbeddingVector, ...]:
        if self._cancelled:
            return ()
        return tuple(self._vector(sample, ":".join(view.identity for view in sample.views)) for sample in samples)

    def health(self) -> Mapping[str, Any]:
        return {"state": "cancelled" if self._cancelled else "available", "device": "cpu"}

    def cancel(self) -> bool:
        already = self._cancelled
        self._cancelled = True
        return not already


class StructuralEmbeddingBackend:
    """Dependency-light CPU backend using deterministic Sprite Lab evidence."""

    backend_id = "structural_cpu"
    model_identity = "spritelab-structural-embedding-v1"
    capabilities = EmbeddingCapabilities(
        ("technical_feature", "alpha_silhouette", "palette_composition"), 2048, "cpu_only"
    )

    def __init__(self) -> None:
        self._cancelled = False

    @property
    def cache_identity(self) -> str:
        return content_identity(
            "spritelab-embedding-backend-v1",
            {
                "backend_id": self.backend_id,
                "model_identity": self.model_identity,
                "representations": self.capabilities.representations,
                "feature_order": _TECHNICAL_FEATURE_ORDER,
                "silhouette_grid": [8, 8],
                "palette_bins": [4, 4, 4],
            },
        )

    def embed_images(self, samples: Sequence[EmbeddingSample]) -> tuple[EmbeddingVector, ...]:
        if self._cancelled:
            return ()
        result: list[EmbeddingVector] = []
        for sample in samples:
            result.extend((self._technical(sample), self._palette(sample)))
        return tuple(result)

    def embed_views(self, samples: Sequence[EmbeddingSample]) -> tuple[EmbeddingVector, ...]:
        if self._cancelled:
            return ()
        return tuple(self._silhouette(sample) for sample in samples)

    def _technical(self, sample: EmbeddingSample) -> EmbeddingVector:
        values: list[float] = []
        width = max(1.0, float(sample.technical.feature("image_width", 1)))
        height = max(1.0, float(sample.technical.feature("image_height", 1)))
        for name in _TECHNICAL_FEATURE_ORDER:
            value = sample.technical.feature(name, 0.0)
            if name == "aspect_log2":
                number = math.log2(width / height)
            elif name == "component_log1p":
                number = math.log1p(float(sample.technical.feature("connected_component_count", 0)))
            elif name == "palette_log1p":
                number = math.log1p(float(sample.technical.feature("palette_size", 0)))
            elif name in {"symmetry_horizontal", "symmetry_vertical"}:
                symmetry = sample.technical.feature("symmetry_estimates", {})
                key = name.removeprefix("symmetry_")
                number = float(symmetry.get(key) or 0.0) if isinstance(symmetry, Mapping) else 0.0
            elif name == "animation":
                number = float(bool(sample.technical.feature("animation_status", False)))
            elif name == "blank":
                number = float(bool(sample.technical.feature("empty_blank_status", False)))
            else:
                number = float(value or 0.0)
            values.append(number)
        return self._record(sample, "technical_feature", np.asarray(values, dtype=np.float32))

    def _palette(self, sample: EmbeddingSample) -> EmbeddingVector:
        bins = np.zeros((4, 4, 4), dtype=np.float32)
        rows = sample.technical.feature("dominant_colors", [])
        total = 0.0
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                rgba = row.get("rgba")
                count = row.get("count")
                if not isinstance(rgba, list) or len(rgba) < 3 or isinstance(count, bool):
                    continue
                weight = max(0.0, float(count or 0.0))
                red, green, blue = (max(0, min(255, int(value))) for value in rgba[:3])
                bins[min(3, red // 64), min(3, green // 64), min(3, blue // 64)] += weight
                total += weight
        if total:
            bins /= total
        return self._record(sample, "palette_composition", bins.reshape(-1))

    def _silhouette(self, sample: EmbeddingSample) -> EmbeddingVector:
        selected = next(
            (view for view in sample.views if view.render_type == RenderType.SILHOUETTE.value),
            next((view for view in sample.views if view.render_type == RenderType.NATIVE.value), None),
        )
        if selected is None:
            vector = np.zeros(64, dtype=np.float32)
        else:
            with Image.open(selected.artifact_path) as opened:
                alpha = opened.convert("RGBA").getchannel("A").resize((8, 8), Image.Resampling.BOX)
                vector = np.asarray(alpha, dtype=np.float32).reshape(-1) / 255.0
        return self._record(sample, "alpha_silhouette", vector)

    def _record(self, sample: EmbeddingSample, representation: str, values: np.ndarray) -> EmbeddingVector:
        values = _normalized(values)
        return EmbeddingVector(
            sample.record_identity,
            sample.image_identity,
            self.cache_identity,
            self.model_identity,
            representation,
            tuple(float(value) for value in values),
        )

    def health(self) -> Mapping[str, Any]:
        return {"state": "cancelled" if self._cancelled else "available", "device": "cpu", "network": False}

    def cancel(self) -> bool:
        already = self._cancelled
        self._cancelled = True
        return not already


class PluginEmbeddingBackend:
    """Adapter seam for future local pixel-art encoders; no dependency is mandatory."""

    def __init__(
        self,
        *,
        backend_id: str,
        model_identity: str,
        representation: str,
        embed: Callable[[Sequence[EmbeddingSample]], Sequence[Sequence[float]]],
        maximum_batch_size: int = 128,
        device_requirements: str = "plugin_declared",
    ) -> None:
        self.backend_id = require_text(backend_id, "plugin embedding backend ID")
        self.model_identity = require_text(model_identity, "plugin embedding model identity")
        self.representation = require_text(representation, "plugin embedding representation")
        self.capabilities = EmbeddingCapabilities((representation,), maximum_batch_size, device_requirements)
        self._embed = embed
        self._cancelled = False

    @property
    def cache_identity(self) -> str:
        return content_identity(
            "spritelab-plugin-embedding-backend-v1",
            {
                "backend_id": self.backend_id,
                "model_identity": self.model_identity,
                "representation": self.representation,
                "device_requirements": self.capabilities.device_requirements,
            },
        )

    def embed_images(self, samples: Sequence[EmbeddingSample]) -> tuple[EmbeddingVector, ...]:
        if self._cancelled:
            return ()
        vectors = tuple(self._embed(samples))
        if len(vectors) != len(samples):
            raise ContractValidationError("plugin embedding backend returned the wrong batch size")
        return tuple(
            EmbeddingVector(
                sample.record_identity,
                sample.image_identity,
                self.cache_identity,
                self.model_identity,
                self.representation,
                tuple(float(value) for value in vector),
            )
            for sample, vector in zip(samples, vectors, strict=True)
        )

    def embed_views(self, samples: Sequence[EmbeddingSample]) -> tuple[EmbeddingVector, ...]:
        return self.embed_images(samples)

    def health(self) -> Mapping[str, Any]:
        return {
            "state": "cancelled" if self._cancelled else "available",
            "device": self.capabilities.device_requirements,
        }

    def cancel(self) -> bool:
        already = self._cancelled
        self._cancelled = True
        return not already


_TECHNICAL_FEATURE_ORDER = (
    "aspect_log2",
    "alpha_coverage",
    "opaque_area_ratio",
    "component_log1p",
    "palette_log1p",
    "color_entropy",
    "symmetry_horizontal",
    "symmetry_vertical",
    "edge_density",
    "detail_density",
    "object_count_estimate",
    "animation",
    "blank",
)


def _normalized(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(values)):
        raise ContractValidationError("embedding backend produced a non-finite value")
    norm = float(np.linalg.norm(values))
    return values / norm if norm > 0 else values


class EmbeddingStore:
    """SQLite cache supporting incremental updates, invalidation, and deletes."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS embeddings (
                    record_identity TEXT NOT NULL,
                    image_identity TEXT NOT NULL,
                    backend_identity TEXT NOT NULL,
                    model_identity TEXT NOT NULL,
                    representation TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    vector BLOB NOT NULL,
                    vector_sha256 TEXT NOT NULL,
                    PRIMARY KEY (record_identity, backend_identity, representation)
                );
                CREATE INDEX IF NOT EXISTS embeddings_image ON embeddings(image_identity, backend_identity);
                """
            )
            existing = connection.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()
            if existing and existing[0] != EMBEDDING_STORE_SCHEMA_VERSION:
                raise ContractValidationError("embedding store schema version is incompatible")
            connection.execute(
                "INSERT OR REPLACE INTO metadata(key,value) VALUES('schema_version',?)",
                (EMBEDDING_STORE_SCHEMA_VERSION,),
            )

    def put_batch(self, vectors: Sequence[EmbeddingVector]) -> int:
        rows = []
        for vector in vectors:
            array = np.asarray(vector.vector, dtype="<f4")
            payload = array.tobytes(order="C")
            rows.append(
                (
                    vector.record_identity,
                    vector.image_identity,
                    vector.backend_identity,
                    vector.model_identity,
                    vector.representation,
                    len(vector.vector),
                    payload,
                    hashlib.sha256(payload).hexdigest(),
                )
            )
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO embeddings VALUES(?,?,?,?,?,?,?,?)
                ON CONFLICT(record_identity,backend_identity,representation) DO UPDATE SET
                    image_identity=excluded.image_identity,
                    model_identity=excluded.model_identity,
                    dimensions=excluded.dimensions,
                    vector=excluded.vector,
                    vector_sha256=excluded.vector_sha256
                """,
                rows,
            )
        return len(rows)

    def get(
        self,
        record_identity: str,
        *,
        image_identity: str,
        backend_identity: str,
    ) -> tuple[EmbeddingVector, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT image_identity,model_identity,representation,dimensions,vector,vector_sha256
                   FROM embeddings WHERE record_identity=? AND backend_identity=? ORDER BY representation""",
                (record_identity, backend_identity),
            ).fetchall()
        result: list[EmbeddingVector] = []
        for stored_image, model, representation, dimensions, payload, digest in rows:
            if stored_image != image_identity:
                continue
            if hashlib.sha256(payload).hexdigest() != digest or len(payload) != int(dimensions) * 4:
                raise ContractValidationError("embedding cache row is corrupt")
            array = np.frombuffer(payload, dtype="<f4")
            result.append(
                EmbeddingVector(
                    record_identity,
                    stored_image,
                    backend_identity,
                    model,
                    representation,
                    tuple(float(value) for value in array),
                )
            )
        return tuple(result)

    def delete_missing(self, valid_record_identities: Sequence[str], *, backend_identity: str) -> int:
        valid = set(valid_record_identities)
        with self._connect() as connection:
            existing = {
                row[0]
                for row in connection.execute(
                    "SELECT DISTINCT record_identity FROM embeddings WHERE backend_identity=?", (backend_identity,)
                )
            }
            stale = sorted(existing - valid)
            connection.executemany(
                "DELETE FROM embeddings WHERE record_identity=? AND backend_identity=?",
                ((record_identity, backend_identity) for record_identity in stale),
            )
        return len(stale)

    def stats(self, *, backend_identity: str | None = None) -> dict[str, Any]:
        clause = " WHERE backend_identity=?" if backend_identity else ""
        parameters = (backend_identity,) if backend_identity else ()
        with self._connect() as connection:
            row_count = connection.execute(f"SELECT COUNT(*) FROM embeddings{clause}", parameters).fetchone()[0]
            records = connection.execute(
                f"SELECT COUNT(DISTINCT record_identity) FROM embeddings{clause}", parameters
            ).fetchone()[0]
        return {"schema_version": EMBEDDING_STORE_SCHEMA_VERSION, "rows": row_count, "records": records}


@dataclass(frozen=True)
class RetrievalIndexRecord:
    record_identity: str
    image_identity: str
    taxonomy_identity: str
    embeddings: Mapping[str, tuple[float, ...]]
    review_status: str = "unreviewed"
    verified_taxonomy_path: tuple[str, ...] = ()
    proposal_taxonomy_path: tuple[str, ...] = ()
    reference_cohort_identity: str | None = None
    truth_projection_identity: str | None = None
    review_log_identity: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_text(self.record_identity, "retrieval record identity")
        require_text(self.image_identity, "retrieval image identity")
        require_text(self.taxonomy_identity, "retrieval taxonomy identity")
        if self.review_status not in {"reviewed", "proposal", "unreviewed"}:
            raise ContractValidationError("retrieval review status is invalid")
        require_unique_text(self.verified_taxonomy_path, "verified retrieval path")
        require_unique_text(self.proposal_taxonomy_path, "proposal retrieval path")
        if self.review_status == "reviewed" and not self.verified_taxonomy_path:
            raise ContractValidationError("reviewed retrieval records require verified labels")
        if self.review_status != "reviewed" and self.verified_taxonomy_path:
            raise ContractValidationError("non-reviewed retrieval records cannot carry verified labels")
        if self.review_status == "reviewed":
            if (
                self.reference_cohort_identity is None
                or self.truth_projection_identity is None
                or self.review_log_identity is None
            ):
                raise ContractValidationError(
                    "reviewed retrieval records require cohort, review-log, and truth projection identities"
                )
            require_text(self.reference_cohort_identity, "retrieval reference cohort identity")
            require_text(self.truth_projection_identity, "retrieval truth projection identity")
            require_text(self.review_log_identity, "retrieval review log identity")
        elif self.truth_projection_identity is not None or self.review_log_identity is not None:
            raise ContractValidationError("non-reviewed retrieval records cannot carry authoritative truth identities")
        if self.reference_cohort_identity is not None:
            require_text(self.reference_cohort_identity, "retrieval reference cohort identity")
        if not self.embeddings:
            raise ContractValidationError("retrieval record requires at least one representation")
        for representation, vector in self.embeddings.items():
            require_text(representation, "retrieval representation")
            if not vector:
                raise ContractValidationError("retrieval representation vector cannot be empty")
            for value in vector:
                require_finite(value, "retrieval vector value")

    @classmethod
    def from_vectors(
        cls,
        vectors: Sequence[EmbeddingVector],
        *,
        taxonomy_identity: str,
        reference_cohort_identity: str | None = None,
        human_label: HumanReferenceLabel | None = None,
        proposal_taxonomy_path: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> RetrievalIndexRecord:
        if not vectors:
            raise ContractValidationError("cannot index an empty embedding vector set")
        records = {vector.record_identity for vector in vectors}
        images = {vector.image_identity for vector in vectors}
        if len(records) != 1 or len(images) != 1:
            raise ContractValidationError("embedding vectors for one index record must share identities")
        require_text(taxonomy_identity, "retrieval index taxonomy identity")
        if human_label is not None:
            if not isinstance(human_label, HumanReferenceLabel):
                raise ContractValidationError("synthetic oracle labels cannot become authoritative retrieval truth")
            if human_label.record_identity != next(iter(records)):
                raise ContractValidationError("reviewed label identity does not match the indexed record")
            if (
                human_label.verification is None
                or not human_label.verified_append_only
                or human_label.partition != "reference"
            ):
                raise ContractValidationError("authoritative retrieval requires verified reference-partition truth")
            if human_label.taxonomy_identity != taxonomy_identity:
                raise ContractValidationError("reviewed retrieval label taxonomy does not match the index")
            if reference_cohort_identity is None:
                reference_cohort_identity = human_label.verification.cohort_identity
            elif human_label.verification.cohort_identity != reference_cohort_identity:
                raise ContractValidationError("reviewed retrieval label does not bind the reference cohort")
            if human_label.verification.image_identity != next(iter(images)):
                raise ContractValidationError("reviewed retrieval label does not bind the indexed image")
        return cls(
            next(iter(records)),
            next(iter(images)),
            taxonomy_identity,
            {vector.representation: vector.vector for vector in vectors},
            "reviewed" if human_label else "proposal" if proposal_taxonomy_path else "unreviewed",
            human_label.taxonomy_path if human_label else (),
            tuple(proposal_taxonomy_path),
            reference_cohort_identity,
            human_label.verification.identity if human_label and human_label.verification else None,
            human_label.verification.review_log_identity if human_label and human_label.verification else None,
            dict(metadata or {}),
        )


class ExactRetrievalIndex:
    """Reliable NumPy baseline with chunked O(ND) queries and no GPU."""

    def __init__(
        self,
        records: Sequence[RetrievalIndexRecord],
        *,
        backend_identity: str,
        fusion_weights: Mapping[str, float],
        search_batch_size: int = 4096,
    ) -> None:
        if not records:
            raise ContractValidationError("retrieval index cannot be empty")
        identities = [record.record_identity for record in records]
        if len(identities) != len(set(identities)):
            raise ContractValidationError("retrieval index record identities cannot repeat")
        require_text(backend_identity, "retrieval backend identity")
        taxonomy_identities = {record.taxonomy_identity for record in records}
        if len(taxonomy_identities) != 1:
            raise ContractValidationError("retrieval index records must bind one taxonomy identity")
        cohort_identities = {
            record.reference_cohort_identity for record in records if record.reference_cohort_identity is not None
        }
        if len(cohort_identities) > 1:
            raise ContractValidationError("retrieval index records must bind one reference cohort identity")
        if any(record.review_status == "reviewed" for record in records) and not cohort_identities:
            raise ContractValidationError("reviewed retrieval index requires a reference cohort identity")
        review_log_identities = {
            record.review_log_identity for record in records if record.review_log_identity is not None
        }
        if len(review_log_identities) > 1:
            raise ContractValidationError("retrieval index records must bind one review-log snapshot")
        if any(record.review_status == "reviewed" for record in records) and not review_log_identities:
            raise ContractValidationError("reviewed retrieval index requires a review-log snapshot")
        if type(search_batch_size) is not int or search_batch_size < 1:
            raise ContractValidationError("search batch size must be positive")
        weights = {str(name): float(value) for name, value in fusion_weights.items()}
        if not weights or any(not math.isfinite(value) or value < 0 for value in weights.values()):
            raise ContractValidationError("retrieval fusion weights must be finite non-negative values")
        if sum(weights.values()) <= 0:
            raise ContractValidationError("at least one retrieval fusion weight must be positive")
        available = set.intersection(*(set(record.embeddings) for record in records))
        if not set(weights).issubset(available):
            raise ContractValidationError("retrieval fusion weights name an unavailable representation")
        self.records = tuple(records)
        self.backend_identity = backend_identity
        self.taxonomy_identity = next(iter(taxonomy_identities))
        self.reference_cohort_identity = next(iter(cohort_identities), None)
        self.review_log_identity = next(iter(review_log_identities), None)
        total = sum(weights.values())
        self.fusion_weights = {name: value / total for name, value in sorted(weights.items())}
        self.search_batch_size = search_batch_size
        self._matrices: dict[str, np.ndarray] = {}
        for representation in self.fusion_weights:
            dimensions = {len(record.embeddings[representation]) for record in records}
            if len(dimensions) != 1:
                raise ContractValidationError("retrieval vectors have inconsistent dimensions")
            matrix = np.asarray([record.embeddings[representation] for record in records], dtype=np.float32)
            self._matrices[representation] = _row_normalized(matrix)
        self.identity = content_identity(
            EXACT_INDEX_VERSION,
            {
                "backend_identity": backend_identity,
                "taxonomy_identity": self.taxonomy_identity,
                "reference_cohort_identity": self.reference_cohort_identity,
                "review_log_identity": self.review_log_identity,
                "fusion_weights": self.fusion_weights,
                "records": [
                    {
                        "record_identity": record.record_identity,
                        "image_identity": record.image_identity,
                        "review_status": record.review_status,
                        "taxonomy_identity": record.taxonomy_identity,
                        "reference_cohort_identity": record.reference_cohort_identity,
                        "truth_projection_identity": record.truth_projection_identity,
                        "review_log_identity": record.review_log_identity,
                        "verified_taxonomy_path": record.verified_taxonomy_path,
                        "proposal_taxonomy_path": record.proposal_taxonomy_path,
                        "embedding_hashes": {
                            name: hashlib.sha256(struct.pack(f"<{len(vector)}f", *vector)).hexdigest()
                            for name, vector in sorted(record.embeddings.items())
                            if name in self.fusion_weights
                        },
                    }
                    for record in records
                ],
            },
        )

    @staticmethod
    def estimated_memory_bytes(record_count: int, dimensions_by_representation: Mapping[str, int]) -> int:
        if type(record_count) is not int or record_count < 0:
            raise ContractValidationError("record count must be a non-negative integer")
        return record_count * sum(int(value) for value in dimensions_by_representation.values()) * 4

    def _scores(self, query_embeddings: Mapping[str, Sequence[float]]) -> np.ndarray:
        scores = np.zeros(len(self.records), dtype=np.float32)
        active_weight = 0.0
        for representation, weight in self.fusion_weights.items():
            raw = query_embeddings.get(representation)
            if raw is None:
                continue
            query = _normalized(np.asarray(raw, dtype=np.float32))
            matrix = self._matrices[representation]
            if query.size != matrix.shape[1]:
                raise ContractValidationError("query embedding dimension does not match the index")
            active_weight += weight
            for start in range(0, len(self.records), self.search_batch_size):
                stop = min(len(self.records), start + self.search_batch_size)
                scores[start:stop] += weight * (matrix[start:stop] @ query)
        if active_weight <= 0:
            raise ContractValidationError("query provides none of the indexed representations")
        return scores / active_weight

    def nearest_neighbors(
        self,
        query_embeddings: Mapping[str, Sequence[float]],
        *,
        k: int = 10,
        exclude_record_identity: str | None = None,
    ) -> tuple[RetrievalNeighbor, ...]:
        if type(k) is not int or k < 1:
            raise ContractValidationError("neighbor count k must be positive")
        scores = self._scores(query_embeddings)
        candidates = [
            (float(score), index)
            for index, score in enumerate(scores)
            if self.records[index].record_identity != exclude_record_identity
        ]
        candidates.sort(key=lambda item: (-item[0], self.records[item[1]].record_identity))
        neighbors: list[RetrievalNeighbor] = []
        for similarity, index in candidates[:k]:
            record = self.records[index]
            neighbors.append(
                RetrievalNeighbor(
                    record.record_identity,
                    record.image_identity,
                    self.backend_identity,
                    record.taxonomy_identity,
                    max(0.0, 1.0 - similarity),
                    max(-1.0, min(1.0, similarity)),
                    record.review_status,
                    record.verified_taxonomy_path,
                    record.reference_cohort_identity if record.review_status == "reviewed" else None,
                    record.truth_projection_identity if record.review_status == "reviewed" else None,
                    record.review_log_identity if record.review_status == "reviewed" else None,
                    record.proposal_taxonomy_path,
                    dict(record.metadata),
                )
            )
        return tuple(neighbors)

    def novelty_score(
        self, query_embeddings: Mapping[str, Sequence[float]], *, exclude_record_identity: str | None = None
    ) -> float:
        neighbors = self.nearest_neighbors(query_embeddings, k=1, exclude_record_identity=exclude_record_identity)
        return 1.0 if not neighbors else round(max(0.0, min(1.0, 1.0 - neighbors[0].similarity)), 6)

    def cluster_assignments(self, *, similarity_threshold: float = 0.9) -> dict[str, str]:
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ContractValidationError("cluster similarity threshold must be from 0 through 1")
        fused = self._fused_matrix()
        medoid_indices: list[int] = []
        assignments: dict[str, str] = {}
        for index, record in enumerate(self.records):
            if not medoid_indices:
                medoid_indices.append(index)
                assignments[record.record_identity] = "cluster-000001"
                continue
            scores = fused[medoid_indices] @ fused[index]
            best_position = int(np.argmax(scores))
            if float(scores[best_position]) >= similarity_threshold:
                cluster_index = best_position + 1
            else:
                medoid_indices.append(index)
                cluster_index = len(medoid_indices)
            assignments[record.record_identity] = f"cluster-{cluster_index:06d}"
        return assignments

    def cluster_medoids(self, assignments: Mapping[str, str]) -> dict[str, str]:
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(self.records):
            cluster_id = assignments.get(record.record_identity)
            if cluster_id:
                grouped[cluster_id].append(index)
        fused = self._fused_matrix()
        medoids: dict[str, str] = {}
        for cluster_id, indices in sorted(grouped.items()):
            matrix = fused[indices]
            similarities = matrix @ matrix.T
            best = max(
                range(len(indices)), key=lambda position: (float(similarities[position].sum()), -indices[position])
            )
            medoids[cluster_id] = self.records[indices[best]].record_identity
        return medoids

    def _fused_matrix(self) -> np.ndarray:
        parts = [self._matrices[name] * math.sqrt(weight) for name, weight in self.fusion_weights.items()]
        return _row_normalized(np.concatenate(parts, axis=1))

    def build_evidence(
        self,
        record_identity: str,
        query_embeddings: Mapping[str, Sequence[float]],
        *,
        query_image_identity: str,
        query_embedding_identity: str,
        k: int = 10,
    ) -> RetrievalEvidence:
        neighbors = self.nearest_neighbors(query_embeddings, k=k, exclude_record_identity=record_identity)
        novelty = 1.0 if not neighbors else max(0.0, min(1.0, 1.0 - neighbors[0].similarity))
        return RetrievalEvidence(
            record_identity,
            query_image_identity,
            query_embedding_identity,
            self.identity,
            self.taxonomy_identity,
            self.reference_cohort_identity,
            self.review_log_identity,
            neighbors,
            tuple(self.fusion_weights.items()),
            round(novelty, 6),
        )


def _row_normalized(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ContractValidationError("retrieval matrix must be finite and two-dimensional")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)


def write_index_manifest(path: str | Path, index: ExactRetrievalIndex) -> None:
    """Persist identity/metadata only; embedding vectors stay in the durable store."""

    payload = {
        "schema_version": "spritelab.labeling.retrieval-index-manifest.v1",
        "index_identity": index.identity,
        "backend_identity": index.backend_identity,
        "taxonomy_identity": index.taxonomy_identity,
        "reference_cohort_identity": index.reference_cohort_identity,
        "review_log_identity": index.review_log_identity,
        "fusion_weights": index.fusion_weights,
        "record_count": len(index.records),
        "records": [
            {
                "record_identity": record.record_identity,
                "image_identity": record.image_identity,
                "review_status": record.review_status,
                "taxonomy_identity": record.taxonomy_identity,
                "reference_cohort_identity": record.reference_cohort_identity,
                "truth_projection_identity": record.truth_projection_identity,
                "review_log_identity": record.review_log_identity,
                "verified_taxonomy_path": list(record.verified_taxonomy_path),
                "proposal_taxonomy_path": list(record.proposal_taxonomy_path),
            }
            for record in index.records
        ],
    }
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
