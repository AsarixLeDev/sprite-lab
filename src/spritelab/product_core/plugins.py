"""Validation and composition helpers for feature-owned product plugins."""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Iterator
from types import ModuleType

from spritelab.product_core.cli import ProductCliRegistry
from spritelab.product_core.contracts import ProductPlugin


class PluginContractError(TypeError):
    """A feature module does not implement the product plugin contract."""


class DuplicatePluginIdError(ValueError):
    """Two independently supplied plugins claim the same stable ID."""


class ProductPluginRegistry:
    """An instance-scoped plugin collection; this module has no global registry."""

    def __init__(self, plugins: Iterable[ProductPlugin] = ()) -> None:
        self._plugins: dict[str, ProductPlugin] = {}
        for plugin in plugins:
            self.register(plugin)

    def register(self, plugin: ProductPlugin) -> None:
        if not isinstance(plugin, ProductPlugin):
            raise PluginContractError("build_plugin() must return ProductPlugin.")
        if plugin.plugin_id in self._plugins:
            raise DuplicatePluginIdError(f"Duplicate product plugin ID: {plugin.plugin_id}")
        self._plugins[plugin.plugin_id] = plugin

    def register_cli(self, registry: ProductCliRegistry) -> None:
        for plugin in self._plugins.values():
            plugin.cli_registration(registry)

    def __iter__(self) -> Iterator[ProductPlugin]:
        return iter(self._plugins.values())

    def __len__(self) -> int:
        return len(self._plugins)


def load_plugin(module: str | ModuleType) -> ProductPlugin:
    """Load and validate the required ``build_plugin() -> ProductPlugin`` export."""

    loaded = importlib.import_module(module) if isinstance(module, str) else module
    builder = getattr(loaded, "build_plugin", None)
    if not callable(builder):
        raise PluginContractError(f"Product plugin module {loaded.__name__!r} must export build_plugin().")
    plugin = builder()
    if not isinstance(plugin, ProductPlugin):
        raise PluginContractError(f"{loaded.__name__}.build_plugin() must return ProductPlugin.")
    return plugin
