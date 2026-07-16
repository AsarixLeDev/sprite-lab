from spritelab.harvest.label_schema import LabelSuggestion
from spritelab.harvest.label_taxonomy import canonicalize_object_name


def test_footwear_and_ambiguous_objects_do_not_over_singularize() -> None:
    assert canonicalize_object_name("shoes") == "shoes"
    assert canonicalize_object_name("boots") == "boots"
    assert canonicalize_object_name("scissors") == "scissors"
    assert canonicalize_object_name("wiresnips") == "wiresnips"
    assert canonicalize_object_name("ambiguous") == "ambiguous"
    assert canonicalize_object_name("ambiguous_object") == "ambiguous_object"
    assert canonicalize_object_name("ambiguous_shape") == "ambiguous_shape"


def test_typos_and_low_information_objects_canonicalize() -> None:
    assert canonicalize_object_name("armour") == "armor"
    assert canonicalize_object_name("amethist") == "amethyst"
    assert canonicalize_object_name("saphire") == "sapphire"
    assert canonicalize_object_name("ovale") == "oval"
    assert canonicalize_object_name("ambiguou") == "ambiguous"
    assert canonicalize_object_name("ambiguou_object") == "ambiguous_object"
    assert canonicalize_object_name("ambiguou_shape") == "ambiguous_shape"
    assert canonicalize_object_name("unidentified_object") == "unknown"


def test_bad_canonical_forms_do_not_survive_label_normalization() -> None:
    assert LabelSuggestion("armor", "shoes").object_name != "sho"
    assert LabelSuggestion("armor", "armour").object_name == "armor"
    assert LabelSuggestion("unknown", "ambiguou_object").object_name == "ambiguous_object"
    assert LabelSuggestion("unknown", "ambiguou_shape").object_name == "ambiguous_shape"


def test_elm_is_not_globally_rewritten_to_helmet() -> None:
    assert canonicalize_object_name("elm") == "elm"
