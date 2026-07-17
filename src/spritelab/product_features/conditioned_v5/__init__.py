"""Conditioned Dataset-v5 product plugin with lazy public imports."""

__all__ = [
    "PLUGIN_ID",
    "CandidatePolicy",
    "ConditionedDatasetError",
    "ConditionedDatasetImportAdapter",
    "ConditionedDatasetService",
    "build_plugin",
    "create_plugin",
]


def __getattr__(name: str) -> object:
    if name == "ConditionedDatasetImportAdapter":
        from spritelab.product_features.conditioned_v5.intake import ConditionedDatasetImportAdapter

        return ConditionedDatasetImportAdapter
    if name in {"CandidatePolicy", "ConditionedDatasetError", "ConditionedDatasetService"}:
        from spritelab.product_features.conditioned_v5.service import (
            CandidatePolicy,
            ConditionedDatasetError,
            ConditionedDatasetService,
        )

        return {
            "CandidatePolicy": CandidatePolicy,
            "ConditionedDatasetError": ConditionedDatasetError,
            "ConditionedDatasetService": ConditionedDatasetService,
        }[name]
    if name in {"PLUGIN_ID", "build_plugin", "create_plugin"}:
        from spritelab.product_features.conditioned_v5.plugin import PLUGIN_ID, build_plugin, create_plugin

        return {"PLUGIN_ID": PLUGIN_ID, "build_plugin": build_plugin, "create_plugin": create_plugin}[name]
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(__all__)
