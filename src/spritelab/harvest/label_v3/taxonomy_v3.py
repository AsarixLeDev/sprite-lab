"""Auto-Labeling v3: hierarchical semantic taxonomy.

Versioned, additive taxonomy that sits alongside the Labeling v2 taxonomy
without modifying its behavior. Supports:

- Domain → broad category → canonical object hierarchy
- Optional surface/display aliases
- Parent/child relationships
- Explicit open-set states
- Independent colors, materials, shapes, roles
- Impossible-combination rules
- Canonical synonym and alias mappings
"""

from __future__ import annotations

from dataclasses import dataclass

from spritelab.harvest.label_taxonomy import normalize_object_name

TAXONOMY_VERSION = "v3.1"

V3_DOMAINS: tuple[str, ...] = (
    "unknown",
    "item_icon",
    "weapon",
    "armor",
    "material",
    "effect_icon",
    "tool",
    "plant",
    "block",
    "environment_prop",
    "ui_icon",
    "entity",
    "character",
)

BROAD_CATEGORIES: tuple[str, ...] = (
    "unknown",
    "weapon",
    "bladed_weapon",
    "ranged_weapon",
    "blunt_weapon",
    "armor",
    "head_armor",
    "body_armor",
    "shield",
    "container",
    "potion",
    "food",
    "fruit",
    "vegetable",
    "meat",
    "crafting_material",
    "gem",
    "mineral",
    "tool",
    "cutting_tool",
    "crafting_tool",
    "plant",
    "flower",
    "mushroom",
    "block",
    "environment_prop",
    "round_object",
    "elongated_object",
    "tile",
    "abstract_icon",
)

MATERIAL_VALUES: tuple[str, ...] = (
    "unknown",
    "metal",
    "wood",
    "stone",
    "glass",
    "cloth",
    "leather",
    "bone",
    "crystal",
    "gemstone",
    "liquid",
    "organic",
    "paper",
    "magical",
    "plastic",
    "ceramic",
)

SHAPE_VALUES: tuple[str, ...] = (
    "unknown",
    "round",
    "elongated",
    "flat",
    "multipart",
    "hollow",
    "symmetric",
    "compact",
    "tile_like",
    "container_like",
    "blade_like",
    "irregular",
    "rectangular",
    "triangular",
)

ROLE_VALUES: tuple[str, ...] = (
    "unknown",
    "item",
    "resource",
    "crafting_material",
    "consumable",
    "equippable",
    "crafting_component",
    "quest_item",
    "currency",
    "decorative",
    "obstacle",
    "interactable",
    "status_indicator",
    "navigation",
)


@dataclass(frozen=True)
class HierarchyNode:
    name: str
    parent: str = ""
    children: tuple[str, ...] = ()
    depth: int = 0
    domain: str = ""
    broader_nodes: tuple[str, ...] = ()
    synonyms: tuple[str, ...] = ()
    open_set_allowed: bool = False

    def is_ancestor_of(self, other: HierarchyNode) -> bool:
        return self.name in other.broader_nodes or self.name == other.parent

    def is_descendant_of(self, other: HierarchyNode) -> bool:
        return other.name in self.broader_nodes

    def shared_ancestor_depth(self, other: HierarchyNode) -> int:
        common = set(self.broader_nodes) & set(other.broader_nodes)
        common.add(self.name if self.name == other.name else "")
        return max(
            (_HIERARCHY_NODES.get(n, HierarchyNode(name=n)).depth for n in common if n),
            default=0,
        )


_HIERARCHY_NODES: dict[str, HierarchyNode] = {}
_HIERARCHY_INITIALIZED = False

# The synthetic root all domains hang off of. Sharing only this ancestor does
# not make two nodes "siblings" for ambiguity purposes.
_ROOT_NODE = "object"


def _build_initial_hierarchy() -> dict[str, HierarchyNode]:
    nodes: dict[str, HierarchyNode] = {}

    def add(
        name: str,
        parent: str = "",
        children: tuple[str, ...] = (),
        domain: str = "unknown",
        synonyms: tuple[str, ...] = (),
    ) -> HierarchyNode:
        depth = 0
        broader: list[str] = []
        current = parent
        while current:
            if current in nodes:
                broader.append(current)
                depth = max(depth, nodes[current].depth + 1)
                current = nodes[current].parent
            else:
                break
        node = HierarchyNode(
            name=name,
            parent=parent,
            children=children,
            depth=depth,
            domain=domain,
            broader_nodes=tuple(broader),
            synonyms=synonyms,
            open_set_allowed=not bool(children),
        )
        nodes[name] = node
        return node

    # Domain level
    add("object", domain="unknown")
    add("weapon", parent="object", domain="weapon")
    add("armor", parent="object", domain="armor")
    add("tool", parent="object", domain="tool")
    add("consumable", parent="object", domain="item_icon")
    add("material", parent="object", domain="material")
    add("plant", parent="object", domain="plant")
    add("block", parent="object", domain="block")
    add("environment_prop", parent="object", domain="environment_prop")
    add("ui_icon", parent="object", domain="ui_icon")
    add("entity", parent="object", domain="entity")
    add("character", parent="object", domain="character")
    add("effect_icon", parent="object", domain="effect_icon")

    # Weapon hierarchy
    add("bladed_weapon", parent="weapon", domain="weapon")
    add("ranged_weapon", parent="weapon", domain="weapon")
    add("blunt_weapon", parent="weapon", domain="weapon")
    for child, names in [
        ("sword", ("sword", "blade")),
        ("dagger", ("dagger", "knife")),
        ("axe", ("axe", "hatchet")),
        ("bow", ("bow", "longbow")),
        ("arrow", ("arrow", "bolt")),
        ("hammer", ("hammer", "mace")),
        ("spear", ("spear", "lance", "pike")),
    ]:
        parent = (
            "bladed_weapon"
            if child in {"sword", "dagger"}
            else "ranged_weapon"
            if child in {"bow", "arrow"}
            else "blunt_weapon"
        )
        add(child, parent=parent, domain="weapon", synonyms=names)

    # Armor hierarchy
    add("head_armor", parent="armor", domain="armor")
    add("body_armor", parent="armor", domain="armor")
    add("shield", parent="armor", domain="armor")
    for child, names in [
        ("helmet", ("helmet", "helm", "headgear")),
        ("chestplate", ("chestplate", "breastplate", "plate_armor")),
        ("boots", ("boots", "greaves")),
    ]:
        parent_name = (
            "head_armor"
            if child == "helmet"
            else "body_armor"
            if child == "chestplate"
            else "body_armor"
            if child == "boots"
            else "armor"
        )
        add(child, parent=parent_name, domain="armor", synonyms=names)

    # Container/Food
    add("container", parent="consumable", domain="item_icon")
    add("potion", parent="container", domain="item_icon", synonyms=("potion", "vial", "flask", "bottle"))
    add("food", parent="consumable", domain="item_icon")
    add("fruit", parent="food", domain="item_icon")
    add("vegetable", parent="food", domain="item_icon")
    add("meat", parent="food", domain="item_icon")

    # Material
    add("gem", parent="material", domain="material", synonyms=("gem", "gemstone"))
    add(
        "crystal_cluster",
        parent="material",
        domain="material",
        synonyms=("crystal", "crystal cluster", "cluster of crystals", "crystalline cluster"),
    )
    add("mineral", parent="material", domain="material", synonyms=("mineral", "ore", "ingot"))

    # Tool
    add("cutting_tool", parent="tool", domain="tool")
    add("crafting_tool", parent="tool", domain="tool")
    for child in ("pickaxe", "fishing_rod", "shovel", "scissors"):
        add(child, parent="tool", domain="tool")

    # Plant
    add("flower", parent="plant", domain="plant")
    add("mushroom", parent="plant", domain="plant", synonyms=("mushroom", "fungus"))
    add("herb", parent="plant", domain="plant")
    add("tree", parent="plant", domain="plant")

    # Populate children from the parent pointers so that leaf-vs-internal
    # detection is correct. ``add`` runs before children are known, so
    # ``open_set_allowed`` (== "is a leaf that may be accepted directly") must be
    # recomputed here. Without this pass every node looks like a leaf and the
    # hierarchy-backoff branch in fusion is unreachable.
    children_map: dict[str, list[str]] = {}
    for node in nodes.values():
        if node.parent:
            children_map.setdefault(node.parent, []).append(node.name)

    for name, node in list(nodes.items()):
        child_names = tuple(sorted(children_map.get(name, ())))
        nodes[name] = HierarchyNode(
            name=node.name,
            parent=node.parent,
            children=child_names,
            depth=node.depth,
            domain=node.domain,
            broader_nodes=node.broader_nodes,
            synonyms=node.synonyms,
            open_set_allowed=not bool(child_names),
        )

    return nodes


def _init_hierarchy() -> None:
    global _HIERARCHY_NODES, _HIERARCHY_INITIALIZED
    if not _HIERARCHY_INITIALIZED:
        _HIERARCHY_NODES = _build_initial_hierarchy()
        _HIERARCHY_INITIALIZED = True


def get_hierarchy_node(name: str) -> HierarchyNode | None:
    _init_hierarchy()
    normalized = normalize_object_name(name)
    if normalized in _HIERARCHY_NODES:
        return _HIERARCHY_NODES[normalized]
    for node in _HIERARCHY_NODES.values():
        if normalized in node.synonyms:
            return node
    return None


Relation = str  # "agree" | "compatible" | "sibling" | "contradict" | "unknown"


def taxonomy_relation(a: str, b: str) -> Relation:
    """Classify how two same-field values relate via the taxonomy.

    * ``agree``       — same value / same node (incl. synonyms).
    * ``compatible``  — one is an ancestor/descendant of the other (broad vs
      specific agreement, e.g. ``sword`` and ``bladed_weapon``). NOT a conflict.
    * ``sibling``     — distinct nodes sharing the same immediate parent
      (e.g. ``sword`` and ``dagger``): ambiguous, but not a hard contradiction.
    * ``contradict``  — both are known nodes in different subtrees
      (e.g. ``sword`` and ``shield``): a genuine contradiction.
    * ``unknown``     — at least one value is not a taxonomy node, so
      incompatibility cannot be proven (e.g. ``rapier`` vs ``sword``): treated as
      non-contradictory.
    """
    _init_hierarchy()
    a_norm = normalize_object_name(a) if a else ""
    b_norm = normalize_object_name(b) if b else ""
    if not a_norm or not b_norm:
        return "unknown"
    if a_norm == b_norm:
        return "agree"

    node_a = get_hierarchy_node(a)
    node_b = get_hierarchy_node(b)
    if node_a is not None and node_b is not None and node_a.name == node_b.name:
        return "agree"
    if node_a is None or node_b is None:
        return "unknown"

    if node_a.is_ancestor_of(node_b) or node_b.is_ancestor_of(node_a):
        return "compatible"
    if node_a.name in node_b.broader_nodes or node_b.name in node_a.broader_nodes:
        return "compatible"

    # Distinct nodes are siblings only if they share a *non-root* immediate
    # parent (e.g. sword/dagger under bladed_weapon). Two top-level domains that
    # merely share the "object" root (e.g. weapon/armor) are a genuine
    # contradiction, not ambiguity.
    if node_a.parent and node_a.parent == node_b.parent and node_a.parent != _ROOT_NODE:
        return "sibling"
    return "contradict"


def deepest_supported_node(
    candidate_object: str,
    *,
    category: str = "unknown",
    min_depth: int = 0,
) -> HierarchyNode | None:
    _init_hierarchy()
    node = get_hierarchy_node(candidate_object)
    if node is None:
        return None
    if node.depth < min_depth:
        return None
    return node


def broader_hierarchy_node(node: HierarchyNode, depth_increment: int = 1) -> HierarchyNode | None:
    _init_hierarchy()
    if node.parent and node.parent in _HIERARCHY_NODES:
        parent = _HIERARCHY_NODES[node.parent]
        if depth_increment > 1:
            return broader_hierarchy_node(parent, depth_increment - 1)
        return parent
    return node


def all_hierarchy_nodes() -> dict[str, HierarchyNode]:
    _init_hierarchy()
    return dict(_HIERARCHY_NODES)


def taxonomy_version_hash() -> str:
    import hashlib

    _init_hierarchy()
    data = ";".join(
        f"{n.name}:{n.parent}:{','.join(sorted(n.children))}:{','.join(sorted(n.synonyms))}"
        for n in sorted(_HIERARCHY_NODES.values(), key=lambda x: x.name)
    )
    return hashlib.sha256(data.encode()).hexdigest()[:16]
