"""Deterministic, identity-bound, metadata-free multi-view rendering."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from spritelab.harvest.label_v4.pixel_evidence import exact_rgba_content_hash
from spritelab.hierarchical_labeling.contracts import TechnicalVisualEvidence
from spritelab.hierarchical_labeling.json_utils import (
    ContractValidationError,
    StrictRecord,
    content_identity,
    require_text,
    require_unique_text,
)

RENDER_IMPLEMENTATION_VERSION = "spritelab-multi-view-render-v1"


class RenderType(str, Enum):
    NATIVE = "native"
    ENLARGED = "nearest_neighbor_enlarged"
    CHECKERBOARD = "checkerboard"
    LIGHT_BACKGROUND = "neutral_light_background"
    DARK_BACKGROUND = "neutral_dark_background"
    SILHOUETTE = "alpha_silhouette"
    BOUNDING_BOX_CROP = "opaque_bounding_box_crop"
    SOURCE_SHEET_CONTEXT = "source_sheet_context"
    ANIMATION_CONTACT_SHEET = "animation_contact_sheet"
    DUPLICATE_CLUSTER_CONTACT_SHEET = "duplicate_cluster_contact_sheet"
    PACK_CONTACT_SHEET = "pack_contact_sheet"


@dataclass(frozen=True, eq=False)
class RenderView(StrictRecord):
    SCHEMA_VERSION = "spritelab.labeling.render-view.v1"
    IDENTITY_FIELDS = ("source_image_identity", "render_type", "render_sha256")

    source_image_identity: str
    render_type: str
    parameters: dict[str, Any]
    background_parameters: dict[str, Any]
    scale: int
    crop: tuple[int, int, int, int] | None
    context_source_identities: tuple[str, ...]
    implementation_version: str
    render_sha256: str
    width: int
    height: int
    artifact_path: str

    def __post_init__(self) -> None:
        for name in (
            "source_image_identity",
            "render_type",
            "implementation_version",
            "render_sha256",
            "artifact_path",
        ):
            require_text(getattr(self, name), name.replace("_", " "))
        if type(self.scale) is not int or self.scale < 1 or type(self.width) is not int or type(self.height) is not int:
            raise ContractValidationError("render scale and dimensions must be positive integers")
        if self.width < 1 or self.height < 1:
            raise ContractValidationError("render dimensions must be positive")
        if self.crop is not None and len(self.crop) != 4:
            raise ContractValidationError("render crop must be a four-value rectangle")
        require_unique_text(self.context_source_identities, "render context source identities")
        self.validate_record()

    def blind_manifest(self) -> dict[str, Any]:
        """Provider-safe manifest: no filesystem path or source metadata."""

        return {
            "view_identity": self.identity,
            "render_type": self.render_type,
            "parameters": self.parameters,
            "background_parameters": self.background_parameters,
            "scale": self.scale,
            "crop": list(self.crop) if self.crop else None,
            "context_source_identities": list(self.context_source_identities),
            "implementation_version": self.implementation_version,
            "render_sha256": self.render_sha256,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True, eq=False)
class MultiViewRenderBundle(StrictRecord):
    SCHEMA_VERSION = "spritelab.labeling.multi-view-render-bundle.v1"
    IDENTITY_FIELDS = ("record_identity", "source_image_identity", "policy_identity")

    record_identity: str
    source_image_identity: str
    profile: str
    blind_visual_only: bool
    policy_identity: str
    views: tuple[RenderView, ...]

    def __post_init__(self) -> None:
        for name in ("record_identity", "source_image_identity", "profile", "policy_identity"):
            require_text(getattr(self, name), name.replace("_", " "))
        if not self.views:
            raise ContractValidationError("render bundle must contain at least one view")
        types = [view.render_type for view in self.views]
        if len(types) != len(set(types)):
            raise ContractValidationError("render bundle cannot contain duplicate render types")
        if any(view.source_image_identity != self.source_image_identity for view in self.views):
            raise ContractValidationError("render view source identity does not match its bundle")
        if self.blind_visual_only and any(view.context_source_identities for view in self.views):
            raise ContractValidationError("blind visual-only bundles cannot contain external context sources")
        self.validate_record()


@dataclass(frozen=True)
class RenderPolicy:
    profile: str
    render_types: tuple[RenderType, ...]
    blind_visual_only: bool
    allow_sheet_context: bool = False
    allow_animation_context: bool = False
    allow_duplicate_context: bool = False
    allow_pack_context: bool = False

    @property
    def identity(self) -> str:
        return content_identity(
            "spritelab-render-policy-v1",
            {
                "profile": self.profile,
                "render_types": [item.value for item in self.render_types],
                "blind_visual_only": self.blind_visual_only,
                "allow_sheet_context": self.allow_sheet_context,
                "allow_animation_context": self.allow_animation_context,
                "allow_duplicate_context": self.allow_duplicate_context,
                "allow_pack_context": self.allow_pack_context,
            },
        )


BLIND_VISUAL_POLICY = RenderPolicy(
    "blind_visual",
    (
        RenderType.NATIVE,
        RenderType.ENLARGED,
        RenderType.CHECKERBOARD,
        RenderType.LIGHT_BACKGROUND,
        RenderType.DARK_BACKGROUND,
        RenderType.SILHOUETTE,
        RenderType.BOUNDING_BOX_CROP,
    ),
    True,
)
FAST_LOCAL_POLICY = RenderPolicy("fast_local", BLIND_VISUAL_POLICY.render_types, True)
BALANCED_POLICY = RenderPolicy(
    "balanced",
    (*BLIND_VISUAL_POLICY.render_types, RenderType.ANIMATION_CONTACT_SHEET, RenderType.SOURCE_SHEET_CONTEXT),
    False,
    allow_sheet_context=True,
    allow_animation_context=True,
)
HIGH_QUALITY_POLICY = RenderPolicy(
    "high_quality",
    (
        *BALANCED_POLICY.render_types,
        RenderType.DUPLICATE_CLUSTER_CONTACT_SHEET,
        RenderType.PACK_CONTACT_SHEET,
    ),
    False,
    allow_sheet_context=True,
    allow_animation_context=True,
    allow_duplicate_context=True,
    allow_pack_context=True,
)
RENDER_POLICIES = {
    policy.profile: policy for policy in (BLIND_VISUAL_POLICY, FAST_LOCAL_POLICY, BALANCED_POLICY, HIGH_QUALITY_POLICY)
}


def _png_bytes(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=False, compress_level=9)
    return output.getvalue()


def _checkerboard(size: tuple[int, int], cell: int = 8) -> Image.Image:
    background = Image.new("RGBA", size, (224, 224, 224, 255))
    draw = ImageDraw.Draw(background)
    for y in range(0, size[1], cell):
        for x in range(0, size[0], cell):
            if (x // cell + y // cell) % 2:
                draw.rectangle(
                    (x, y, min(size[0], x + cell) - 1, min(size[1], y + cell) - 1), fill=(176, 176, 176, 255)
                )
    return background


def _background_view(source: Image.Image, color: tuple[int, int, int, int]) -> Image.Image:
    background = Image.new("RGBA", source.size, color)
    background.alpha_composite(source)
    return background


def _silhouette(source: Image.Image) -> Image.Image:
    alpha = source.getchannel("A")
    result = Image.new("RGBA", source.size, (255, 255, 255, 0))
    result.putalpha(alpha.point(lambda value: 255 if value > 0 else 0))
    return result


def _contact_sheet(
    paths: tuple[Path, ...], *, scale: int = 4, maximum: int = 16
) -> tuple[Image.Image, tuple[str, ...]]:
    selected = paths[:maximum]
    if not selected:
        raise ContractValidationError("a context contact sheet requires at least one image")
    opened: list[Image.Image] = []
    identities: list[str] = []
    for path in selected:
        with Image.open(path) as image:
            opened.append(image.convert("RGBA"))
        identities.append(exact_rgba_content_hash(path))
    cell_width = max(image.width for image in opened)
    cell_height = max(image.height for image in opened)
    columns = min(4, len(opened))
    rows = (len(opened) + columns - 1) // columns
    sheet = Image.new("RGBA", (cell_width * columns, cell_height * rows), (128, 128, 128, 255))
    for index, image in enumerate(opened):
        x = (index % columns) * cell_width + (cell_width - image.width) // 2
        y = (index // columns) * cell_height + (cell_height - image.height) // 2
        sheet.alpha_composite(image, (x, y))
    if scale > 1:
        sheet = sheet.resize((sheet.width * scale, sheet.height * scale), Image.Resampling.NEAREST)
    return sheet, tuple(identities)


def _build_image(
    render_type: RenderType,
    source: Image.Image,
    technical: TechnicalVisualEvidence,
    *,
    scale: int,
    sheet_context: Path | None,
    animation_frames: tuple[Path, ...],
    duplicate_context: tuple[Path, ...],
    pack_context: tuple[Path, ...],
) -> tuple[Image.Image, dict[str, Any], dict[str, Any], tuple[int, int, int, int] | None, tuple[str, ...]]:
    parameters: dict[str, Any] = {}
    background: dict[str, Any] = {}
    crop: tuple[int, int, int, int] | None = None
    contexts: tuple[str, ...] = ()
    if render_type == RenderType.NATIVE:
        result = source.copy()
    elif render_type == RenderType.ENLARGED:
        parameters["resampling"] = "nearest"
        parameters["scale"] = scale
        result = source.resize((source.width * scale, source.height * scale), Image.Resampling.NEAREST)
    elif render_type == RenderType.CHECKERBOARD:
        background = {"kind": "checkerboard", "colors": ["#e0e0e0", "#b0b0b0"], "cell": 8}
        result = _checkerboard(source.size)
        result.alpha_composite(source)
    elif render_type == RenderType.LIGHT_BACKGROUND:
        background = {"kind": "solid", "rgba": [240, 240, 240, 255]}
        result = _background_view(source, (240, 240, 240, 255))
    elif render_type == RenderType.DARK_BACKGROUND:
        background = {"kind": "solid", "rgba": [32, 32, 32, 255]}
        result = _background_view(source, (32, 32, 32, 255))
    elif render_type == RenderType.SILHOUETTE:
        parameters["foreground"] = "opaque_white"
        result = _silhouette(source)
    elif render_type == RenderType.BOUNDING_BOX_CROP:
        raw = technical.feature("opaque_bounding_box", [0, 0, 0, 0])
        if not isinstance(raw, list) or len(raw) != 4 or raw[2] <= raw[0] or raw[3] <= raw[1]:
            crop = (0, 0, source.width, source.height)
        else:
            crop = tuple(int(value) for value in raw)
        result = source.crop(crop)
    elif render_type == RenderType.SOURCE_SHEET_CONTEXT:
        if sheet_context is None:
            raise ContractValidationError("source-sheet context was selected without a source sheet")
        with Image.open(sheet_context) as image:
            result = image.convert("RGBA")
        contexts = (exact_rgba_content_hash(sheet_context),)
    elif render_type == RenderType.ANIMATION_CONTACT_SHEET:
        result, contexts = _contact_sheet(animation_frames)
    elif render_type == RenderType.DUPLICATE_CLUSTER_CONTACT_SHEET:
        result, contexts = _contact_sheet(duplicate_context)
    elif render_type == RenderType.PACK_CONTACT_SHEET:
        result, contexts = _contact_sheet(pack_context)
    else:  # pragma: no cover - exhaustive enum guard
        raise ContractValidationError(f"unsupported render type: {render_type}")
    return result, parameters, background, crop, contexts


def build_render_bundle(
    image_path: str | Path,
    technical: TechnicalVisualEvidence,
    output_root: str | Path,
    *,
    policy: RenderPolicy = BLIND_VISUAL_POLICY,
    scale: int = 8,
    sheet_context: str | Path | None = None,
    animation_frames: tuple[str | Path, ...] = (),
    duplicate_context: tuple[str | Path, ...] = (),
    pack_context: tuple[str | Path, ...] = (),
    pack_context_permitted: bool = False,
) -> MultiViewRenderBundle:
    path = Path(image_path)
    if exact_rgba_content_hash(path) != technical.image_identity:
        raise ContractValidationError("render source no longer matches technical evidence identity")
    if type(scale) is not int or not 1 <= scale <= 64:
        raise ContractValidationError("render scale must be an integer from 1 through 64")
    if sheet_context and not policy.allow_sheet_context:
        raise ContractValidationError("selected render policy does not permit source-sheet context")
    if animation_frames and not policy.allow_animation_context:
        raise ContractValidationError("selected render policy does not permit animation context")
    if duplicate_context and not policy.allow_duplicate_context:
        raise ContractValidationError("selected render policy does not permit duplicate context")
    if pack_context and (not policy.allow_pack_context or not pack_context_permitted):
        raise ContractValidationError("pack context requires an explicit permitting policy")
    context_availability = {
        RenderType.SOURCE_SHEET_CONTEXT: bool(sheet_context),
        RenderType.ANIMATION_CONTACT_SHEET: bool(animation_frames),
        RenderType.DUPLICATE_CLUSTER_CONTACT_SHEET: bool(duplicate_context),
        RenderType.PACK_CONTACT_SHEET: bool(pack_context),
    }
    render_types = tuple(
        item for item in policy.render_types if item not in context_availability or context_availability[item]
    )
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    views: list[RenderView] = []
    with Image.open(path) as opened:
        source = opened.convert("RGBA")
    for index, render_type in enumerate(render_types):
        rendered, parameters, background, crop, contexts = _build_image(
            render_type,
            source,
            technical,
            scale=scale,
            sheet_context=Path(sheet_context) if sheet_context else None,
            animation_frames=tuple(Path(item) for item in animation_frames),
            duplicate_context=tuple(Path(item) for item in duplicate_context),
            pack_context=tuple(Path(item) for item in pack_context),
        )
        payload = _png_bytes(rendered)
        render_sha256 = hashlib.sha256(payload).hexdigest()
        view_seed = {
            "source_image_identity": technical.image_identity,
            "render_type": render_type.value,
            "parameters": parameters,
            "background_parameters": background,
            "scale": scale,
            "crop": list(crop) if crop else None,
            "context_source_identities": contexts,
            "implementation_version": RENDER_IMPLEMENTATION_VERSION,
            "render_sha256": render_sha256,
        }
        view_identity = content_identity("spritelab-render-view-v1", view_seed)
        artifact = output / f"{index:02d}-{render_type.value}-{view_identity[:12]}.png"
        artifact.write_bytes(payload)
        views.append(
            RenderView(
                technical.image_identity,
                render_type.value,
                parameters,
                background,
                scale,
                crop,
                contexts,
                RENDER_IMPLEMENTATION_VERSION,
                render_sha256,
                rendered.width,
                rendered.height,
                str(artifact),
            )
        )
    return MultiViewRenderBundle(
        technical.record_identity,
        technical.image_identity,
        policy.profile,
        policy.blind_visual_only,
        policy.identity,
        tuple(views),
    )


def visual_only_provider_payload(bundle: MultiViewRenderBundle) -> tuple[dict[str, Any], ...]:
    """Return only pixels and controlled render descriptors, never source names/paths."""

    payload: list[dict[str, Any]] = []
    for view in bundle.views:
        payload.append(
            {**view.blind_manifest(), "media_type": "image/png", "data": Path(view.artifact_path).read_bytes()}
        )
    audit_visual_only_payload(payload)
    return tuple(payload)


def audit_visual_only_payload(value: Any) -> None:
    forbidden = {
        "filename",
        "file_name",
        "filesystem_path",
        "source_path",
        "pack_name",
        "creator",
        "source_url",
        "license",
        "old_label",
        "metadata_prediction",
        "retrieval_labels",
        "prior_pass_outputs",
    }

    def inspect(item: Any, path: str = "$") -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if str(key).casefold() in forbidden:
                    raise ContractValidationError(f"visual-only payload leaks forbidden key at {path}.{key}")
                inspect(child, f"{path}.{key}")
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                inspect(child, f"{path}[{index}]")
        elif isinstance(item, str):
            lowered = item.casefold()
            if any(token in lowered for token in ("file://", "source_url=", "license=")):
                raise ContractValidationError(f"visual-only payload leaks forbidden text at {path}")

    inspect(value)
