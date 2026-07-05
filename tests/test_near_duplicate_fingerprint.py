from __future__ import annotations

import pytest
from PIL import Image

from spritelab.data.dedupe_report import (
    average_hash_image,
    difference_hash_image,
    hamming_distance_hex,
)


def test_average_and_difference_hash_are_deterministic() -> None:
    image = _center_square((220, 40, 40))

    assert average_hash_image(image) == average_hash_image(image)
    assert difference_hash_image(image) == difference_hash_image(image)


def test_hamming_distance_hex_behaviour() -> None:
    image = _vertical_split()
    other = _horizontal_split()
    average = average_hash_image(image)
    other_average = average_hash_image(other)

    assert hamming_distance_hex(average, average) == 0
    assert hamming_distance_hex(average, other_average) > 0
    with pytest.raises(ValueError):
        hamming_distance_hex("0f", "0")


def test_identical_images_have_same_hashes() -> None:
    first = _center_square((120, 200, 255))
    second = _center_square((120, 200, 255))

    assert average_hash_image(first) == average_hash_image(second)
    assert difference_hash_image(first) == difference_hash_image(second)


def test_slightly_changed_image_has_small_hash_distance() -> None:
    base = _center_square((220, 40, 40))
    changed = base.copy()
    changed.putpixel((16, 16), (240, 80, 80, 255))

    average_distance = hamming_distance_hex(average_hash_image(base), average_hash_image(changed))
    difference_distance = hamming_distance_hex(difference_hash_image(base), difference_hash_image(changed))

    assert min(average_distance, difference_distance) <= 8


def test_different_images_can_have_larger_average_hash_distance() -> None:
    slight = _center_square((220, 40, 40))
    slight_changed = slight.copy()
    slight_changed.putpixel((16, 16), (240, 80, 80, 255))
    different = _horizontal_split()

    slight_distance = hamming_distance_hex(average_hash_image(slight), average_hash_image(slight_changed))
    different_distance = hamming_distance_hex(average_hash_image(slight), average_hash_image(different))

    assert different_distance >= slight_distance


def _center_square(color: tuple[int, int, int]) -> Image.Image:
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    for y in range(10, 22):
        for x in range(10, 22):
            image.putpixel((x, y), (*color, 255))
    return image


def _vertical_split() -> Image.Image:
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 255))
    for y in range(32):
        for x in range(16, 32):
            image.putpixel((x, y), (255, 255, 255, 255))
    return image


def _horizontal_split() -> Image.Image:
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 255))
    for y in range(16, 32):
        for x in range(32):
            image.putpixel((x, y), (255, 255, 255, 255))
    return image
