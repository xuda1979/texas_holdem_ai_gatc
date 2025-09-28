"""Utilities for verifying bundled GUI assets."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List

DEFAULT_CARD_DIRECTORY = os.path.join("src", "poker_ai", "gui", "card_images")
DEFAULT_TABLE_BACKGROUND = os.path.join(
    "src", "poker_ai", "gui", "assets", "poker_table_background.png"
)

SUITS: tuple[str, ...] = ("h", "d", "c", "s")
RANKS: tuple[str, ...] = tuple("23456789TJQKA")
EXPECTED_CARD_FILENAMES: tuple[str, ...] = tuple(
    f"{rank}{suit}.png" for suit in SUITS for rank in RANKS
)


@dataclass
class CardAssetReport:
    """Structured result describing the card image directory."""

    directory: str
    directory_exists: bool
    missing_cards: List[str] = field(default_factory=list)
    unexpected_files: List[str] = field(default_factory=list)
    file_sizes: Dict[str, int] = field(default_factory=dict)

    @property
    def total_expected(self) -> int:
        return len(EXPECTED_CARD_FILENAMES)

    @property
    def found(self) -> int:
        return len(self.file_sizes)

    @property
    def is_complete(self) -> bool:
        return self.directory_exists and not self.missing_cards


@dataclass
class TableBackgroundReport:
    """Information about the poker table background asset."""

    path: str
    exists: bool
    size_bytes: int | None = None


def inspect_card_assets(card_dir: str | None = None) -> CardAssetReport:
    """Return a :class:`CardAssetReport` describing the card assets."""

    directory = card_dir or DEFAULT_CARD_DIRECTORY
    if not os.path.isdir(directory):
        return CardAssetReport(directory=directory, directory_exists=False)

    file_sizes: Dict[str, int] = {}
    missing_cards: List[str] = []
    for filename in EXPECTED_CARD_FILENAMES:
        path = os.path.join(directory, filename)
        if os.path.exists(path):
            file_sizes[filename[:-4]] = os.path.getsize(path)
        else:
            missing_cards.append(filename[:-4])

    all_pngs = [f for f in os.listdir(directory) if f.endswith(".png")]
    unexpected_files = sorted(f for f in all_pngs if f not in EXPECTED_CARD_FILENAMES)

    return CardAssetReport(
        directory=directory,
        directory_exists=True,
        missing_cards=missing_cards,
        unexpected_files=unexpected_files,
        file_sizes={name: size for name, size in file_sizes.items()},
    )


def inspect_table_background(path: str | None = None) -> TableBackgroundReport:
    """Return metadata describing the poker table background asset."""

    background_path = path or DEFAULT_TABLE_BACKGROUND
    if os.path.exists(background_path):
        return TableBackgroundReport(
            path=background_path,
            exists=True,
            size_bytes=os.path.getsize(background_path),
        )
    return TableBackgroundReport(path=background_path, exists=False)


def _warn_on_anomalies(report: CardAssetReport) -> None:
    for card, size in sorted(report.file_sizes.items()):
        if size < 100:
            print(f"⚠️  {card}.png is very small ({size} bytes)")
        elif size > 50000:
            print(f"⚠️  {card}.png is very large ({size} bytes)")


def verify_card_images(card_dir: str | None = None) -> bool:
    """Verify all 52 card images are present with correct names."""

    print("Verifying poker card images...")
    print("=" * 40)

    report = inspect_card_assets(card_dir)

    if not report.directory_exists:
        print(f"❌ Card directory {report.directory} does not exist!")
        return False

    print(f"✅ Found {report.found}/{report.total_expected} card images")

    if report.missing_cards:
        missing_preview = ", ".join(report.missing_cards[:10])
        suffix = "..." if len(report.missing_cards) > 10 else ""
        print(f"❌ Missing {len(report.missing_cards)} cards:")
        print(f"   {missing_preview}{suffix}")
        return False

    _warn_on_anomalies(report)

    print("\nExample cards with sizes:")
    for card in ("2h", "7d", "Tc", "Jh", "Qs", "Kc", "As"):
        size = report.file_sizes.get(card)
        if size is not None:
            print(f"   {card}.png: {size} bytes")

    if report.unexpected_files:
        print(f"\nUnexpected files found: {report.unexpected_files}")

    print("\n🎉 SUCCESS! All 52 poker card images are present and ready!")
    return True


def check_poker_table_background(path: str | None = None) -> bool:
    """Check if poker table background exists."""

    report = inspect_table_background(path)
    if report.exists:
        size = report.size_bytes or 0
        print(f"✅ Poker table background found ({size} bytes)")
        return True

    print(f"⚠️  Poker table background not found at {report.path}")
    return False


def main() -> None:
    """Run the verification suite as a standalone script."""

    print("Poker Image Assets Verification")
    print("=" * 50)

    cards_ok = verify_card_images()
    print()
    bg_ok = check_poker_table_background()

    print("\n" + "=" * 50)
    print("FINAL RESULT:")
    print("=" * 50)

    if cards_ok:
        print("🎉 EXCELLENT! All poker card images are ready!")
        print("\nYour Texas Hold'em AI now has:")
        print("   ✅ All 52 poker card images")
        if bg_ok:
            print("   ✅ Poker table background")
        else:
            print("   ⚠️  Basic poker table background")

        print("\n🚀 Ready to play! Run the GUI:")
        print("   python play/gui.py")
        print("\nThe game will now show actual card images instead of text!")

    else:
        print("❌ Some card images are missing. Please run:")
        print("   python setup_card_images.py")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
