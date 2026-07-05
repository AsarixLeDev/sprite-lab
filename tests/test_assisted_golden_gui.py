"""Tests for pure assisted golden GUI state helpers."""

from __future__ import annotations

from pathlib import Path

from spritelab.harvest.assisted_golden import AssistedGoldenCandidate, AssistedGoldenLabel
from spritelab.harvest.assisted_golden_gui import (
    accept_as_is,
    append_note,
    filter_candidates,
    make_initial_state,
    mark_unknown_fields,
    move_next,
    move_previous,
    progress_counts,
    save_current_label,
    skip_current,
)


def test_initial_state_starts_at_first_unlabeled_candidate():
    candidates = (_candidate("a"), _candidate("b"), _candidate("c"))
    labels = {"a": AssistedGoldenLabel(sprite_id="a", category="weapon", object_name="axe", tags=("axe",))}

    state = make_initial_state(candidates, labels=labels, order="source_order")

    assert state.index == 1
    assert state.candidates[state.index].sprite_id == "b"


def test_save_label_advances_index(tmp_path):
    state = make_initial_state((_candidate("a"), _candidate("b")), order="source_order")

    next_state, label = save_current_label(
        state,
        category="weapon",
        object_name="axe",
        tags="axe",
        labels_path=tmp_path / "golden_labels.jsonl",
    )

    assert label is not None
    assert next_state.index == 1
    assert "a" in next_state.labels


def test_accept_as_is_saves_prefilled_fields(tmp_path):
    state = make_initial_state((_candidate("a"), _candidate("b")), order="source_order")

    next_state, label = accept_as_is(state, labels_path=tmp_path / "golden_labels.jsonl")

    assert label is not None
    assert label.category == "weapon"
    assert label.object_name == "axe"
    assert label.tags == ("axe", "weapon")
    assert label.prefill_was_corrected is False
    assert next_state.index == 1


def test_skip_advances_without_label():
    state = make_initial_state((_candidate("a"), _candidate("b")), order="source_order")

    next_state = skip_current(state)

    assert next_state.index == 1
    assert "a" in next_state.skipped
    assert "a" not in next_state.labels


def test_previous_next_navigation_bounds():
    state = make_initial_state((_candidate("a"), _candidate("b")), order="source_order")

    assert move_previous(state).index == 0
    assert move_next(move_next(state)).index == 1


def test_filters_return_matching_candidates():
    candidates = (_candidate("a", category="weapon"), _candidate("b", category="plant"))
    labels = {"a": AssistedGoldenLabel(sprite_id="a", category="weapon", object_name="axe", tags=("axe",))}

    result = filter_candidates(candidates, labels, unlabeled_only=True, category="plant")

    assert [candidate.sprite_id for candidate in result] == ["b"]


def test_progress_counts_are_correct():
    state = make_initial_state((_candidate("a"), _candidate("b"), _candidate("c")), order="source_order")
    state, _ = accept_as_is(state)
    state, _ = save_current_label(state, category="plant", object_name="leaf", tags="leaf")
    state = skip_current(state)

    counts = progress_counts(state)

    assert counts == {
        "total": 3,
        "labeled": 2,
        "corrected": 1,
        "accepted_as_is": 1,
        "skipped": 1,
        "remaining": 0,
    }


def test_resume_loads_existing_labels_and_starts_at_first_unlabeled():
    labels = {
        "a": AssistedGoldenLabel(sprite_id="a", category="weapon", object_name="axe", tags=("axe",)),
        "b": AssistedGoldenLabel(sprite_id="b", category="plant", object_name="leaf", tags=("leaf",)),
    }

    state = make_initial_state((_candidate("a"), _candidate("b"), _candidate("c")), labels=labels, order="source_order")

    assert state.index == 2
    assert state.candidates[state.index].sprite_id == "c"


def test_mark_unknown_fills_unknown_fields():
    assert mark_unknown_fields() == ("unknown", "", "")


def test_note_buttons_append_notes_without_duplicates():
    notes = append_note("", "bad_crop")
    notes = append_note(notes, "bad crop")

    assert notes == "bad_crop"


def _candidate(sprite_id: str, *, category: str = "weapon") -> AssistedGoldenCandidate:
    return AssistedGoldenCandidate(
        sprite_id=sprite_id,
        final_png_path=Path(f"{sprite_id}.png"),
        source_id="src",
        source_name="Source",
        relative_path=f"{sprite_id}.png",
        suggested_category=category,
        suggested_object_name="axe" if category == "weapon" else "leaf",
        suggested_tags=("axe", "weapon") if category == "weapon" else ("leaf", "plant"),
        suggested_source="fusion",
    )
