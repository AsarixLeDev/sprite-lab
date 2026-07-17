"""User-facing folder intake and exception review for Dataset-v3."""

__all__ = ["DatasetIntakeService", "DatasetReviewStore", "build_dataset", "build_plugin"]


def __getattr__(name: str) -> object:
    if name in {"DatasetIntakeService", "build_dataset"}:
        from spritelab.product_features.dataset.intake import DatasetIntakeService, build_dataset

        return {"DatasetIntakeService": DatasetIntakeService, "build_dataset": build_dataset}[name]
    if name == "DatasetReviewStore":
        from spritelab.product_features.dataset.review import DatasetReviewStore

        return DatasetReviewStore
    if name == "build_plugin":
        from spritelab.product_features.dataset.plugin import build_plugin

        return build_plugin
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(__all__)
