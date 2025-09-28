from __future__ import annotations

from verify_images import (
    DEFAULT_CARD_DIRECTORY,
    DEFAULT_TABLE_BACKGROUND,
    inspect_card_assets,
    inspect_table_background,
)


def test_card_images_complete():
    report = inspect_card_assets()
    assert report.directory_exists, f"Card directory missing: {DEFAULT_CARD_DIRECTORY}"
    assert report.missing_cards == []
    assert report.unexpected_files == []
    assert report.found == report.total_expected
    assert all(size > 0 for size in report.file_sizes.values())


def test_table_background_exists():
    report = inspect_table_background()
    assert report.exists, f"Expected poker table background at {DEFAULT_TABLE_BACKGROUND}"
    assert report.size_bytes is not None and report.size_bytes > 0
