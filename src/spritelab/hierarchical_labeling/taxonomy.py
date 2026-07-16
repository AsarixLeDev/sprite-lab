"""Versioned controlled taxonomy graph with safe hierarchical backoff."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from spritelab.hierarchical_labeling.json_utils import (
    ContractValidationError,
    StrictRecord,
    content_identity,
    require_text,
    require_unique_text,
    strict_json_value,
)

TAXONOMY_VERSION = "sprite_lab_visual_taxonomy_v1"
TAXONOMY_SCHEMA_VERSION = "spritelab.labeling.taxonomy-graph.v1"
NODE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")


@dataclass(frozen=True, eq=False)
class TaxonomyNode(StrictRecord):
    SCHEMA_VERSION = "spritelab.labeling.taxonomy-node.v1"
    IDENTITY_FIELDS = ("node_id", "taxonomy_version")

    node_id: str
    display_name: str
    parent_id: str | None
    definition: str
    positive_visual_criteria: tuple[str, ...]
    negative_visual_criteria: tuple[str, ...]
    allowed_children: tuple[str, ...]
    minimum_visual_evidence: tuple[str, ...]
    may_be_automatically_accepted: bool
    human_truth_required: bool
    deprecated_aliases: tuple[str, ...]
    taxonomy_version: str = TAXONOMY_VERSION

    def __post_init__(self) -> None:
        if not NODE_ID_PATTERN.fullmatch(self.node_id) or self.node_id == "unknown":
            raise ContractValidationError("taxonomy node ID is invalid; unknown is abstention, not a node")
        require_text(self.display_name, "taxonomy display name")
        if self.parent_id is not None and not NODE_ID_PATTERN.fullmatch(self.parent_id):
            raise ContractValidationError("taxonomy parent ID is invalid")
        require_text(self.definition, "taxonomy definition")
        require_text(self.taxonomy_version, "taxonomy version")
        for name in (
            "positive_visual_criteria",
            "negative_visual_criteria",
            "allowed_children",
            "minimum_visual_evidence",
            "deprecated_aliases",
        ):
            require_unique_text(getattr(self, name), name)
        if not self.positive_visual_criteria or not self.minimum_visual_evidence:
            raise ContractValidationError("taxonomy nodes require positive criteria and minimum visual evidence")
        if any(not NODE_ID_PATTERN.fullmatch(value) for value in self.allowed_children):
            raise ContractValidationError("allowed child IDs must be controlled node identifiers")
        if any(not NODE_ID_PATTERN.fullmatch(value) for value in self.deprecated_aliases):
            raise ContractValidationError("deprecated aliases must be controlled identifiers")
        self.validate_record()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TaxonomyNode:
        expected = {
            "schema_version",
            "node_id",
            "display_name",
            "parent_id",
            "definition",
            "positive_visual_criteria",
            "negative_visual_criteria",
            "allowed_children",
            "minimum_visual_evidence",
            "may_be_automatically_accepted",
            "human_truth_required",
            "deprecated_aliases",
            "taxonomy_version",
        }
        if set(value) != expected or value.get("schema_version") != cls.SCHEMA_VERSION:
            raise ContractValidationError("taxonomy node does not match the exact v1 schema")
        for name in ("may_be_automatically_accepted", "human_truth_required"):
            if type(value[name]) is not bool:
                raise ContractValidationError(f"{name} must be a JSON boolean")
        for name in (
            "positive_visual_criteria",
            "negative_visual_criteria",
            "allowed_children",
            "minimum_visual_evidence",
            "deprecated_aliases",
        ):
            if not isinstance(value[name], list) or not all(isinstance(item, str) for item in value[name]):
                raise ContractValidationError(f"{name} must be an array of strings")
        return cls(
            node_id=value["node_id"],
            display_name=value["display_name"],
            parent_id=value["parent_id"],
            definition=value["definition"],
            positive_visual_criteria=tuple(value["positive_visual_criteria"]),
            negative_visual_criteria=tuple(value["negative_visual_criteria"]),
            allowed_children=tuple(value["allowed_children"]),
            minimum_visual_evidence=tuple(value["minimum_visual_evidence"]),
            may_be_automatically_accepted=value["may_be_automatically_accepted"],
            human_truth_required=value["human_truth_required"],
            deprecated_aliases=tuple(value["deprecated_aliases"]),
            taxonomy_version=value["taxonomy_version"],
        )


class TaxonomyGraph:
    """Validated immutable hierarchy; malformed graphs never partially load."""

    def __init__(self, version: str, nodes: Iterable[TaxonomyNode]):
        require_text(version, "taxonomy version")
        supplied = tuple(nodes)
        if not supplied:
            raise ContractValidationError("taxonomy graph cannot be empty")
        by_id = {node.node_id: node for node in supplied}
        if len(by_id) != len(supplied):
            raise ContractValidationError("taxonomy graph contains duplicate node IDs")
        if any(node.taxonomy_version != version for node in supplied):
            raise ContractValidationError("every taxonomy node must bind the graph version")
        roots = [node.node_id for node in supplied if node.parent_id is None]
        if len(roots) != 1:
            raise ContractValidationError("taxonomy graph must contain exactly one root")
        for node in supplied:
            if node.parent_id is not None and node.parent_id not in by_id:
                raise ContractValidationError(f"orphan taxonomy node: {node.node_id}")
        expected_children: dict[str, list[str]] = defaultdict(list)
        for node in supplied:
            if node.parent_id:
                expected_children[node.parent_id].append(node.node_id)
        for node in supplied:
            if tuple(sorted(node.allowed_children)) != tuple(sorted(expected_children[node.node_id])):
                raise ContractValidationError(f"allowed children disagree with parent pointers for {node.node_id}")
        aliases: dict[str, str] = {}
        for node in supplied:
            for alias in node.deprecated_aliases:
                if alias in by_id or alias in aliases:
                    raise ContractValidationError(f"duplicate or colliding deprecated alias: {alias}")
                aliases[alias] = node.node_id
        self.version = version
        self._nodes = by_id
        self._aliases = aliases
        self.root_id = roots[0]
        self._validate_acyclic()

    def _validate_acyclic(self) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node_id: str) -> None:
            if node_id in visiting:
                raise ContractValidationError(f"taxonomy cycle reaches {node_id}")
            if node_id in visited:
                return
            visiting.add(node_id)
            for child in self._nodes[node_id].allowed_children:
                visit(child)
            visiting.remove(node_id)
            visited.add(node_id)

        visit(self.root_id)
        if visited != set(self._nodes):
            missing = ", ".join(sorted(set(self._nodes) - visited))
            raise ContractValidationError(f"taxonomy nodes are disconnected from the root: {missing}")

    @property
    def identity(self) -> str:
        return content_identity(TAXONOMY_SCHEMA_VERSION, self.to_dict())

    @property
    def nodes(self) -> tuple[TaxonomyNode, ...]:
        return tuple(sorted(self._nodes.values(), key=lambda node: (self.depth(node.node_id), node.node_id)))

    def to_dict(self) -> dict[str, Any]:
        return strict_json_value(
            {
                "schema_version": TAXONOMY_SCHEMA_VERSION,
                "taxonomy_version": self.version,
                "root_id": self.root_id,
                "unknown_policy": "abstention_not_a_node",
                "nodes": [node.to_dict() for node in self.nodes],
            }
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TaxonomyGraph:
        if set(value) != {"schema_version", "taxonomy_version", "root_id", "unknown_policy", "nodes"}:
            raise ContractValidationError("taxonomy graph does not match the exact v1 schema")
        if value.get("schema_version") != TAXONOMY_SCHEMA_VERSION:
            raise ContractValidationError("taxonomy graph schema version is unsupported")
        if value.get("unknown_policy") != "abstention_not_a_node":
            raise ContractValidationError("unknown must remain abstention and cannot become a taxonomy node")
        raw_nodes = value.get("nodes")
        if not isinstance(raw_nodes, list) or not all(isinstance(item, Mapping) for item in raw_nodes):
            raise ContractValidationError("taxonomy nodes must be a JSON array of objects")
        graph = cls(str(value.get("taxonomy_version", "")), (TaxonomyNode.from_dict(item) for item in raw_nodes))
        if value.get("root_id") != graph.root_id:
            raise ContractValidationError("declared taxonomy root does not match the graph")
        return graph

    def node(self, node_id: str) -> TaxonomyNode:
        resolved = self.resolve(node_id)
        if resolved is None:
            raise ContractValidationError(f"unknown taxonomy node: {node_id!r}")
        return self._nodes[resolved]

    def resolve(self, value: str | None) -> str | None:
        if value in (None, "", "unknown"):
            return None
        if not isinstance(value, str) or value != value.strip():
            return None
        return value if value in self._nodes else self._aliases.get(value)

    def migrate_deprecated(self, value: str) -> dict[str, str] | None:
        target = self._aliases.get(value)
        if target is None:
            return None
        return {"deprecated_node": value, "current_node": target, "taxonomy_version": self.version}

    def depth(self, node_id: str) -> int:
        current = self.node(node_id)
        depth = 0
        while current.parent_id is not None:
            depth += 1
            current = self._nodes[current.parent_id]
        return depth

    def path(self, node_id: str, *, include_root: bool = True) -> tuple[str, ...]:
        current = self.node(node_id)
        path = [current.node_id]
        while current.parent_id is not None:
            current = self._nodes[current.parent_id]
            path.append(current.node_id)
        path.reverse()
        if not include_root and path and path[0] == self.root_id:
            path = path[1:]
        return tuple(path)

    def parent(self, node_id: str) -> TaxonomyNode | None:
        parent_id = self.node(node_id).parent_id
        return self._nodes[parent_id] if parent_id else None

    def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        ancestor_id = self.resolve(ancestor)
        descendant_id = self.resolve(descendant)
        return bool(ancestor_id and descendant_id and ancestor_id in self.path(descendant_id))

    def lowest_common_ancestor(self, *node_ids: str) -> str | None:
        resolved = [self.resolve(node_id) for node_id in node_ids]
        if not resolved or any(node_id is None for node_id in resolved):
            return None
        paths = [self.path(node_id or "") for node_id in resolved]
        shared: str | None = None
        for values in zip(*paths, strict=False):
            if len(set(values)) != 1:
                break
            shared = values[0]
        return shared

    def deepest_defensible_node(
        self,
        evidence_by_node: Mapping[str, float],
        threshold_by_depth: Mapping[int, float],
    ) -> str | None:
        """Return the deepest node whose complete ancestor chain is defensible."""

        accepted: list[str] = []
        for candidate, score in evidence_by_node.items():
            resolved = self.resolve(candidate)
            if resolved is None or isinstance(score, bool) or not isinstance(score, (int, float)):
                continue
            path = self.path(resolved)
            if all(
                float(evidence_by_node.get(node_id, 0.0)) >= float(threshold_by_depth.get(self.depth(node_id), 1.0))
                for node_id in path
            ):
                accepted.append(resolved)
        return max(accepted, key=lambda node_id: (self.depth(node_id), node_id), default=None)


def _default_nodes() -> tuple[TaxonomyNode, ...]:
    """Controlled additive taxonomy derived from repository v2/v3 categories."""

    specs: dict[str, dict[str, Any]] = {}

    def add(
        node_id: str,
        parent_id: str | None,
        definition: str,
        positive: Sequence[str],
        negative: Sequence[str],
        minimum: Sequence[str],
        *,
        auto: bool = True,
        human: bool = False,
        aliases: Sequence[str] = (),
    ) -> None:
        specs[node_id] = {
            "node_id": node_id,
            "display_name": node_id.replace("_", " ").title(),
            "parent_id": parent_id,
            "definition": definition,
            "positive_visual_criteria": tuple(positive),
            "negative_visual_criteria": tuple(negative),
            "minimum_visual_evidence": tuple(minimum),
            "may_be_automatically_accepted": auto,
            "human_truth_required": human,
            "deprecated_aliases": tuple(aliases),
        }

    add(
        "entity",
        None,
        "Any visibly bounded sprite subject; this synthetic root does not assert an exact object identity.",
        ("one or more visible foreground forms",),
        ("blank or technically unusable image",),
        ("nonblank deterministic alpha evidence",),
    )
    add(
        "object",
        "entity",
        "A discrete icon, subject, prop, tile, effect, or interface form represented by the sprite.",
        ("visually coherent bounded subject",),
        ("unsegmented multi-asset source sheet",),
        ("coherent silhouette or visible-part observation",),
        aliases=("item_icon",),
    )
    add(
        "equipment",
        "object",
        "Portable-looking equipment, including weapons, armor, and tools.",
        ("graspable or wearable form with functional parts",),
        ("pure scenery or abstract interface mark",),
        ("equipment-like outline and at least one functional visible part",),
    )
    add(
        "weapon",
        "equipment",
        "Equipment whose visible form is consistent with striking, cutting, or ranged use.",
        ("blade, striking head, projectile channel, or weapon-like haft",),
        ("ordinary container or food silhouette",),
        ("weapon-like form plus supporting visible part",),
    )
    add(
        "sword",
        "weapon",
        "A long bladed weapon with handle and guard cues.",
        ("elongated blade and handle",),
        ("separate bow limbs or axe head",),
        ("blade and handle visible",),
    )
    add(
        "axe",
        "weapon",
        "A hafted cutting weapon or tool with a transverse head.",
        ("haft plus offset cutting head",),
        ("continuous sword blade",),
        ("head-to-haft relationship visible",),
    )
    add(
        "bow",
        "weapon",
        "A curved ranged weapon with two limbs and a string-like span.",
        ("paired curved limbs",),
        ("solid blade or compact shield",),
        ("bow-like curvature visible",),
    )
    add(
        "dagger",
        "weapon",
        "A short bladed weapon with a compact handle.",
        ("short blade and handle",),
        ("long staff-like form",),
        ("short blade boundary visible",),
    )
    add(
        "hammer",
        "weapon",
        "A hafted striking object with a heavy head.",
        ("haft and blunt head",),
        ("thin cutting edge as primary cue",),
        ("head-to-haft relationship visible",),
    )
    add(
        "spear",
        "weapon",
        "A long shaft with a pointed terminal head.",
        ("long shaft and point",),
        ("wide transverse axe head",),
        ("shaft and terminal point visible",),
    )
    add(
        "armor",
        "equipment",
        "Wearable protective equipment.",
        ("wearable shell, plate, or protective contour",),
        ("tool handle or consumable contents",),
        ("wearable protective form visible",),
    )
    add(
        "helmet",
        "armor",
        "Head-shaped protective armor with opening or rim cues.",
        ("domed head covering",),
        ("open bottle neck or bag handles",),
        ("head-covering contour visible",),
    )
    add(
        "body_armor",
        "armor",
        "Torso-shaped protective armor.",
        ("torso plate or wearable body shell",),
        ("small head-only contour",),
        ("torso-like wearable contour visible",),
        aliases=("chestplate",),
    )
    add(
        "boots",
        "armor",
        "Foot or lower-leg protective equipment.",
        ("foot or paired boot contour",),
        ("handheld haft",),
        ("footwear contour visible",),
    )
    add(
        "shield",
        "armor",
        "A broad handheld protective plate.",
        ("broad plate-like silhouette with rim or boss",),
        ("narrow blade-dominant form",),
        ("protective plate silhouette visible",),
    )
    add(
        "tool",
        "equipment",
        "Portable equipment whose visible form indicates a work function.",
        ("functional head, gripping shaft, or working end",),
        ("wearable protective shell",),
        ("working end and grip relationship visible",),
    )
    add(
        "pickaxe",
        "tool",
        "A long-handled tool with a transverse pointed head.",
        ("haft plus two-sided or pointed head",),
        ("single continuous blade",),
        ("pick head and haft visible",),
    )
    add(
        "shovel",
        "tool",
        "A long-handled tool with a broad scoop-like end.",
        ("shaft plus broad terminal scoop",),
        ("paired bow limbs",),
        ("scoop and shaft visible",),
    )
    add(
        "scissors",
        "tool",
        "A paired-blade cutting tool with loop or pivot cues.",
        ("crossed paired blades or loops",),
        ("single blade and handle",),
        ("paired working parts visible",),
    )
    add(
        "fishing_rod",
        "tool",
        "A slender rod with line, reel, or hook cues.",
        ("long flexible rod-like form",),
        ("wide solid blade",),
        ("rod plus line-like detail visible",),
    )
    add(
        "container",
        "object",
        "A vessel or receptacle whose form visibly encloses or carries contents.",
        ("enclosed body with opening, lid, neck, or handles",),
        ("solid blade or flat tile",),
        ("container body and opening/closure cue",),
    )
    add(
        "bottle",
        "container",
        "A narrow-necked vessel with an enclosed body.",
        ("body, narrow neck, and mouth or stopper",),
        ("wide hinged lid or soft bag body",),
        ("neck-to-body relationship visible",),
        aliases=("vial", "flask"),
    )
    add(
        "chest",
        "container",
        "A rigid box-like container with lid or latch cues.",
        ("box body plus lid, bands, or latch",),
        ("narrow bottle neck",),
        ("rigid body and closure cue visible",),
    )
    add(
        "bag",
        "container",
        "A soft-sided container with gathered top, opening, or handles.",
        ("soft pouch body and opening/handle cue",),
        ("rigid plate or blade",),
        ("pouch body visible",),
    )
    add(
        "consumable",
        "object",
        "An item visibly presented as food, drink, or another expendable object.",
        ("portion, edible form, or filled vessel cue",),
        ("wearable armor or scenery",),
        ("consumable-like form without relying on role metadata",),
    )
    add(
        "food",
        "consumable",
        "A visibly edible portion, produce item, or prepared food icon.",
        ("organic portion, produce, or prepared-food form",),
        ("weapon parts or abstract glyph",),
        ("edible-form cues visible",),
    )
    add(
        "fruit",
        "food",
        "A fruit-like organic form, potentially with stem or leaf cues.",
        ("rounded organic body with stem/leaf cues",),
        ("mineral facets or mechanical parts",),
        ("fruit-like body plus organic detail",),
    )
    add(
        "vegetable",
        "food",
        "A vegetable-like organic form with stalk, root, or leafy cues.",
        ("root, stalk, or leafy edible form",),
        ("weapon blade or gem facets",),
        ("vegetable-like structure visible",),
    )
    add(
        "meat",
        "food",
        "A meat-like prepared or raw portion with bone or cut cues.",
        ("portion contour with bone or cut-face cue",),
        ("whole fruit stem/leaf structure",),
        ("portion and meat-like visible cue",),
    )
    add(
        "potion",
        "consumable",
        "A bottle-like consumable specifically interpreted as a potion; contents/role are context sensitive.",
        ("bottle with visually distinctive liquid or magical-effect cue",),
        ("plain empty bottle without contents cue",),
        ("bottle form and distinctive contents/effect cue",),
        human=True,
    )
    add(
        "resource",
        "object",
        "A raw, crafted, or collectible material-like object.",
        ("ore, ingot, crystal, wood, or material bundle form",),
        ("character body or interface control",),
        ("resource-like form or repeated material unit",),
        aliases=("material", "crafting_material"),
    )
    add(
        "mineral",
        "resource",
        "A rock, ore, or mineral-like chunk.",
        ("irregular rocky body or ore inclusions",),
        ("organic leaf/stem structure",),
        ("rocky silhouette and mineral-like facets",),
        aliases=("ore",),
    )
    add(
        "gem",
        "resource",
        "A cut or naturally faceted gem-like object.",
        ("symmetrical crystalline facets",),
        ("soft pouch or organic food texture",),
        ("faceted crystal geometry visible",),
        aliases=("gemstone",),
    )
    add(
        "crystal_cluster",
        "resource",
        "A cluster of multiple crystal-like points.",
        ("multiple pointed crystalline forms",),
        ("single smooth rounded object",),
        ("at least two crystal points visible",),
    )
    add(
        "ingot",
        "resource",
        "A cast bar-like resource with beveled or stacked-bar cues.",
        ("compact bar with beveled top",),
        ("natural irregular rock",),
        ("bar-like cast form visible",),
    )
    add(
        "wood_resource",
        "resource",
        "A wood-like log, plank, or bundled timber resource.",
        ("cylindrical log, rings, grain-like stripes, or plank form",),
        ("metallic faceted gem",),
        ("wood-like visual cue and resource form",),
    )
    add(
        "environment",
        "object",
        "A scenery, terrain, plant, block, or environmental prop.",
        ("grounded scenery or world-building form",),
        ("isolated interface glyph",),
        ("environmental silhouette or terrain repetition",),
        aliases=("environment_prop",),
    )
    add(
        "prop",
        "environment",
        "A discrete environmental prop not primarily a tile or living plant.",
        ("grounded standalone scenery object",),
        ("wearable or handheld equipment",),
        ("prop-like grounded form visible",),
    )
    add(
        "plant",
        "environment",
        "A plant-like environmental or collectible form.",
        ("stem, leaves, petals, cap, or branching organic growth",),
        ("mechanical straight-edged equipment",),
        ("organic growth structure visible",),
    )
    add(
        "flower",
        "plant",
        "A flowering plant with petals around a center or bloom cluster.",
        ("petals or bloom around center",),
        ("mushroom cap and stalk",),
        ("bloom structure visible",),
    )
    add(
        "mushroom",
        "plant",
        "A fungus-like form with a cap and stalk.",
        ("cap-over-stalk silhouette",),
        ("petaled flower center",),
        ("cap and stalk visible",),
        aliases=("fungus",),
    )
    add(
        "tree",
        "plant",
        "A tree-like form with trunk and canopy or branching cues.",
        ("trunk plus canopy or branches",),
        ("small single flower stem",),
        ("trunk-to-canopy relationship visible",),
    )
    add(
        "block",
        "environment",
        "A discrete block-like terrain or construction unit.",
        ("cuboid, square, or isometric block form",),
        ("thin interface glyph",),
        ("block boundary and face cues visible",),
    )
    add(
        "character",
        "object",
        "A humanoid or character-like figure.",
        ("head/body/limb arrangement or character portrait",),
        ("inanimate tool silhouette",),
        ("character body or portrait structure visible",),
    )
    add(
        "creature",
        "object",
        "A non-humanoid animal, monster, or creature-like figure.",
        ("body plus limbs, wings, tail, or creature face",),
        ("inanimate container geometry",),
        ("creature anatomy cues visible",),
    )
    add(
        "effect",
        "object",
        "A visual effect such as burst, aura, spark, flame, or status mark.",
        ("non-solid radiating, wispy, or transient form",),
        ("solid functional object with stable parts",),
        ("effect-like silhouette or radiating detail",),
        aliases=("effect_icon",),
    )
    add(
        "projectile",
        "object",
        "A discrete projectile or ammunition form.",
        ("small directional body with point, fins, or trail",),
        ("broad stationary scenery",),
        ("directional projectile form visible",),
    )
    add(
        "arrow",
        "projectile",
        "A shafted projectile with a pointed head and fletching cues.",
        ("shaft, point, and rear fins",),
        ("curved bow limbs",),
        ("shaft and terminal point visible",),
        aliases=("bolt",),
    )
    add(
        "tile",
        "object",
        "A repeatable terrain or pattern tile whose boundaries dominate the image.",
        ("edge-to-edge repeatable surface or tile boundary",),
        ("isolated centered object with transparent margin",),
        ("tile-like coverage or repetition cue",),
    )
    add(
        "interface_element",
        "object",
        "An interface icon, control, marker, or abstract status glyph.",
        ("symbolic flat mark or control-like framing",),
        ("physically articulated object with unambiguous parts",),
        ("symbolic or interface framing cue visible",),
        aliases=("ui_icon", "abstract_icon"),
    )

    children: dict[str, list[str]] = defaultdict(list)
    for node_id, value in specs.items():
        if value["parent_id"] is not None:
            children[value["parent_id"]].append(node_id)
    return tuple(
        TaxonomyNode(**value, allowed_children=tuple(sorted(children[node_id]))) for node_id, value in specs.items()
    )


_DEFAULT_TAXONOMY: TaxonomyGraph | None = None


def load_default_taxonomy() -> TaxonomyGraph:
    global _DEFAULT_TAXONOMY
    if _DEFAULT_TAXONOMY is None:
        _DEFAULT_TAXONOMY = TaxonomyGraph(TAXONOMY_VERSION, _default_nodes())
    return _DEFAULT_TAXONOMY


def taxonomy_migration_matrix(graph: TaxonomyGraph | None = None) -> dict[str, Any]:
    selected = graph or load_default_taxonomy()
    rows = [
        {
            "deprecated_node": alias,
            "current_node": node.node_id,
            "migration_kind": "deprecated_alias",
            "human_review_required": False,
        }
        for node in selected.nodes
        for alias in node.deprecated_aliases
    ]
    return {
        "schema_version": "spritelab.labeling.taxonomy-migration-matrix.v1",
        "taxonomy_identity": selected.identity,
        "rows": rows,
    }


def taxonomy_validation_report(graph: TaxonomyGraph | None = None) -> dict[str, Any]:
    selected = graph or load_default_taxonomy()
    return {
        "schema_version": "spritelab.labeling.taxonomy-validation-report.v1",
        "taxonomy_version": selected.version,
        "taxonomy_identity": selected.identity,
        "node_count": len(selected.nodes),
        "root_count": 1,
        "cycle_count": 0,
        "orphan_count": 0,
        "duplicate_alias_count": 0,
        "unknown_is_node": False,
        "verdict": "PASS",
    }
