from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import pytest

from spritelab.product_features.conditioned_v5.publication_commit import (
    CAMPAIGN_COMMIT_KIND,
    CAMPAIGN_COMMIT_SCHEMA,
    CAMPAIGN_COMMIT_STATUS,
    DATASET_COMMIT_KIND,
    DATASET_COMMIT_SCHEMA,
    DATASET_COMMIT_STATUS,
    PUBLICATION_JOURNAL_KIND,
    PUBLICATION_JOURNAL_NAME,
    PUBLICATION_JOURNAL_SCHEMA,
    PUBLICATION_JOURNAL_STATUS,
    PublicationCommitError,
    build_campaign_commit,
    build_dataset_commit,
    build_publication_journal,
    campaign_commit_name,
    canonical_publication_commit_bytes,
    dataset_commit_name,
    validate_campaign_commit,
    validate_dataset_commit,
    validate_publication_journal,
)
from spritelab.training.campaign import stable_hash

PUBLICATION_IDENTITY = "a" * 64
DATASET_INVENTORY = {
    "activation.json": {"sha256": "1" * 64, "byte_count": 701},
    "evidence/label_audit.json": {"sha256": "2" * 64, "byte_count": 902},
    "view_manifest.json": {"sha256": "3" * 64, "byte_count": 503},
}
CAMPAIGN_INVENTORY = {
    "campaign.json": {"sha256": "4" * 64, "byte_count": 1104},
}


def _documents(
    *, publication_identity: str = PUBLICATION_IDENTITY
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    journal = build_publication_journal(
        publication_identity=publication_identity,
        dataset_inventory=DATASET_INVENTORY,
        campaign_inventory=CAMPAIGN_INVENTORY,
    )
    dataset = build_dataset_commit(
        journal=journal,
        dataset_inventory=DATASET_INVENTORY,
        campaign_inventory=CAMPAIGN_INVENTORY,
    )
    campaign = build_campaign_commit(
        journal=journal,
        dataset_commit=dataset,
        dataset_inventory=DATASET_INVENTORY,
        campaign_inventory=CAMPAIGN_INVENTORY,
    )
    return journal, dataset, campaign


def _reidentify(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    payload = dict(value)
    payload.pop(field, None)
    return {**payload, field: stable_hash(payload)}


def test_builders_fix_paths_status_kinds_and_pair_authority() -> None:
    journal, dataset, campaign = _documents()

    name = f"conditioned-v5-{PUBLICATION_IDENTITY}.commit.json"
    assert PUBLICATION_JOURNAL_NAME == "publication-journal.json"
    assert dataset_commit_name(PUBLICATION_IDENTITY) == name
    assert campaign_commit_name(PUBLICATION_IDENTITY) == name
    assert journal["schema_version"] == PUBLICATION_JOURNAL_SCHEMA
    assert journal["status"] == PUBLICATION_JOURNAL_STATUS
    assert journal["kind"] == PUBLICATION_JOURNAL_KIND
    assert journal["dataset_relative_path"] == f"datasets/conditioned-v5-{PUBLICATION_IDENTITY}"
    assert journal["campaign_relative_path"] == f"campaigns/conditioned-v5-{PUBLICATION_IDENTITY}"
    assert journal["dataset_commit_relative_path"] == f"datasets/{name}"
    assert journal["campaign_commit_relative_path"] == f"campaigns/{name}"
    assert journal["pair_authority"] is False

    assert dataset["schema_version"] == DATASET_COMMIT_SCHEMA
    assert dataset["status"] == DATASET_COMMIT_STATUS
    assert dataset["kind"] == DATASET_COMMIT_KIND
    assert dataset["pair_authority"] is False
    assert campaign["schema_version"] == CAMPAIGN_COMMIT_SCHEMA
    assert campaign["status"] == CAMPAIGN_COMMIT_STATUS
    assert campaign["kind"] == CAMPAIGN_COMMIT_KIND
    assert campaign["pair_authority"] is True
    assert campaign["dataset_marker_identity"] == dataset["marker_identity"]
    dataset_bytes = canonical_publication_commit_bytes(dataset)
    assert campaign["dataset_marker_sha256"] == hashlib.sha256(dataset_bytes).hexdigest()
    assert campaign["dataset_marker_byte_count"] == len(dataset_bytes)


def test_documents_and_canonical_bytes_are_deterministic() -> None:
    journal, dataset, campaign = _documents()
    reverse_dataset = {
        path: {"byte_count": record["byte_count"], "sha256": record["sha256"]}
        for path, record in reversed(tuple(DATASET_INVENTORY.items()))
    }
    reverse_campaign = {
        path: {"byte_count": record["byte_count"], "sha256": record["sha256"]}
        for path, record in reversed(tuple(CAMPAIGN_INVENTORY.items()))
    }
    second_journal = build_publication_journal(
        publication_identity=PUBLICATION_IDENTITY,
        dataset_inventory=reverse_dataset,
        campaign_inventory=reverse_campaign,
    )
    second_dataset = build_dataset_commit(
        journal=second_journal,
        dataset_inventory=reverse_dataset,
        campaign_inventory=reverse_campaign,
    )
    second_campaign = build_campaign_commit(
        journal=second_journal,
        dataset_commit=second_dataset,
        dataset_inventory=reverse_dataset,
        campaign_inventory=reverse_campaign,
    )

    assert (second_journal, second_dataset, second_campaign) == (journal, dataset, campaign)
    for first, second in zip(
        (journal, dataset, campaign),
        (second_journal, second_dataset, second_campaign),
        strict=True,
    ):
        first_bytes = canonical_publication_commit_bytes(first)
        assert first_bytes == canonical_publication_commit_bytes(second)
        assert first_bytes.endswith(b"\n")


@pytest.mark.parametrize("document_name", ["journal", "dataset", "campaign"])
@pytest.mark.parametrize("mutation", ["extra", "missing"])
def test_validators_reject_malformed_or_extra_top_level_keys(document_name: str, mutation: str) -> None:
    journal, dataset, campaign = _documents()
    documents = {"journal": journal, "dataset": dataset, "campaign": campaign}
    changed = dict(documents[document_name])
    if mutation == "extra":
        changed["unexpected"] = False
    else:
        changed.pop("kind")

    with pytest.raises(PublicationCommitError, match="exact schema"):
        if document_name == "journal":
            validate_publication_journal(
                changed,
                dataset_inventory=DATASET_INVENTORY,
                campaign_inventory=CAMPAIGN_INVENTORY,
            )
        elif document_name == "dataset":
            validate_dataset_commit(
                changed,
                journal=journal,
                dataset_inventory=DATASET_INVENTORY,
                campaign_inventory=CAMPAIGN_INVENTORY,
            )
        else:
            validate_campaign_commit(
                changed,
                journal=journal,
                dataset_commit=dataset,
                dataset_inventory=DATASET_INVENTORY,
                campaign_inventory=CAMPAIGN_INVENTORY,
            )


def test_self_rehashed_fixed_path_swap_is_rejected() -> None:
    journal, _, _ = _documents()
    changed = dict(journal)
    changed["dataset_relative_path"] = journal["campaign_relative_path"]
    changed = _reidentify(changed, "journal_identity")

    with pytest.raises(PublicationCommitError, match="fixed paths"):
        validate_publication_journal(
            changed,
            dataset_inventory=DATASET_INVENTORY,
            campaign_inventory=CAMPAIGN_INVENTORY,
        )


def test_swapped_exact_inventories_are_rejected() -> None:
    journal, _, _ = _documents()

    with pytest.raises(PublicationCommitError, match="dataset inventory"):
        validate_publication_journal(
            journal,
            dataset_inventory=CAMPAIGN_INVENTORY,
            campaign_inventory=DATASET_INVENTORY,
        )


def test_campaign_rejects_another_valid_dataset_marker() -> None:
    journal, dataset, campaign = _documents()
    other_journal, other_dataset, _ = _documents(publication_identity="b" * 64)
    del other_journal

    with pytest.raises(PublicationCommitError):
        validate_campaign_commit(
            campaign,
            journal=journal,
            dataset_commit=other_dataset,
            dataset_inventory=DATASET_INVENTORY,
            campaign_inventory=CAMPAIGN_INVENTORY,
        )
    with pytest.raises(PublicationCommitError, match="exact schema"):
        validate_dataset_commit(
            campaign,
            journal=journal,
            dataset_inventory=DATASET_INVENTORY,
            campaign_inventory=CAMPAIGN_INVENTORY,
        )
    assert dataset["marker_identity"] != other_dataset["marker_identity"]


def test_self_rehashed_dataset_marker_sha_swap_is_rejected() -> None:
    journal, dataset, campaign = _documents()
    changed = dict(campaign)
    changed["dataset_marker_sha256"] = "f" * 64
    changed = _reidentify(changed, "marker_identity")

    with pytest.raises(PublicationCommitError, match="pair-authority"):
        validate_campaign_commit(
            changed,
            journal=journal,
            dataset_commit=dataset,
            dataset_inventory=DATASET_INVENTORY,
            campaign_inventory=CAMPAIGN_INVENTORY,
        )


def test_self_rehashed_embedded_inventory_swap_is_rejected() -> None:
    journal, _, _ = _documents()
    changed = dict(journal)
    changed["dataset_inventory"] = journal["campaign_inventory"]
    changed = _reidentify(changed, "journal_identity")

    with pytest.raises(PublicationCommitError, match="dataset inventory"):
        validate_publication_journal(
            changed,
            dataset_inventory=DATASET_INVENTORY,
            campaign_inventory=CAMPAIGN_INVENTORY,
        )


@pytest.mark.parametrize("identity", ["a" * 63, "A" * 64, "g" * 64, "", None])
def test_marker_names_reject_noncanonical_publication_identity(identity: Any) -> None:
    with pytest.raises(PublicationCommitError, match="lowercase SHA-256"):
        dataset_commit_name(identity)
    with pytest.raises(PublicationCommitError, match="lowercase SHA-256"):
        campaign_commit_name(identity)


@pytest.mark.parametrize(
    "inventory",
    [
        {},
        {"../escape.json": {"sha256": "1" * 64, "byte_count": 1}},
        {"C:/escape.json": {"sha256": "1" * 64, "byte_count": 1}},
        {"bad\\path.json": {"sha256": "1" * 64, "byte_count": 1}},
        {"item.json": {"sha256": "1" * 64, "byte_count": True}},
        {"item.json": {"sha256": "1" * 64, "byte_count": 1, "extra": False}},
        {
            "Evidence/item.json": {"sha256": "1" * 64, "byte_count": 1},
            "evidence/ITEM.json": {"sha256": "2" * 64, "byte_count": 1},
        },
    ],
)
def test_builder_rejects_malformed_exact_inventory(inventory: Mapping[str, Mapping[str, Any]]) -> None:
    with pytest.raises(PublicationCommitError):
        build_publication_journal(
            publication_identity=PUBLICATION_IDENTITY,
            dataset_inventory=inventory,
            campaign_inventory=CAMPAIGN_INVENTORY,
        )


def test_validator_rejects_self_rehashed_noninteger_binding_byte_count() -> None:
    journal, dataset, _ = _documents()
    changed = dict(dataset)
    changed["journal_byte_count"] = float(dataset["journal_byte_count"])
    changed = _reidentify(changed, "marker_identity")

    with pytest.raises(PublicationCommitError, match="bindings"):
        validate_dataset_commit(
            changed,
            journal=journal,
            dataset_inventory=DATASET_INVENTORY,
            campaign_inventory=CAMPAIGN_INVENTORY,
        )
