from spritelab.dataset_maker.prefill import build_vlm_descriptor_prompt


def test_prefix_family_prompt_guides_candidates_without_exact_source_obedience() -> None:
    prompt = build_vlm_descriptor_prompt(
        {
            "object_name": "shoes",
            "confidence": 0.75,
            "source_profile_name": "oga_496_rpg_icons",
            "filename_trust": "prefix_family",
        },
        candidate_object_names=("shoes", "boots", "boot", "footwear"),
    )
    lowered = prompt.lower()

    assert "filename gives a family hint" in lowered
    assert "- boots" in lowered
    assert "prefer one candidate" in lowered
    assert "do not invent unrelated objects outside the candidate family" in lowered
    assert "mushrooms" in lowered
    assert "magnifying_glass" in lowered
    assert "possible_object_name must equal the source object_name: 'shoes'" not in lowered


def test_exact_trusted_prompt_keeps_strong_source_metadata() -> None:
    prompt = build_vlm_descriptor_prompt(
        {
            "object_name": "butter",
            "confidence": 0.95,
            "source_profile_name": "cc0_food",
            "filename_trust": "exact",
        }
    )
    lowered = prompt.lower()

    assert "filename gives a family hint" not in lowered
    assert "treat this as strong source metadata" in lowered
    assert "possible_object_name must equal the source object_name: 'butter'" in lowered
