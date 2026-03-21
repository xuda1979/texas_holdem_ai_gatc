from __future__ import annotations

"""Betting tree representation for multi-street poker games."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BettingNode:
    action: str | None = None
    parent: Optional["BettingNode"] = None
    children: dict[str, "BettingNode"] = field(default_factory=dict)
    pot: int = 0
    street: str = "pre-flop"

    def add_child(self, action: str, pot: int, street: str) -> "BettingNode":
        """Add a child node representing an action transition."""
        node = BettingNode(action=action, parent=self, pot=pot, street=street)
        self.children[action] = node
        return node


class BettingTree:
    """Simple betting tree with utility methods for traversal."""

    def __init__(self):
        self.root = BettingNode()

    def add_path(
        self, actions: list[str], starting_pot: int = 0, street: str = "pre-flop"
    ) -> BettingNode:
        """Create nodes following the sequence of actions."""
        node = self.root
        pot = starting_pot
        current_street = street
        for act in actions:
            pot += self._pot_increment(act)
            if act.lower() in {"flop", "turn", "river"}:
                current_street = act.lower()
                continue
            node = node.children.get(act) or node.add_child(act, pot, current_street)
        return node

    @staticmethod
    def _pot_increment(action: str) -> int:
        if action in {"bet", "raise"}:
            # Simplified increment for demonstration purposes
            return 10
        return 0

    def traverse(self, node: BettingNode | None = None) -> list[BettingNode]:
        node = node or self.root
        nodes = [node]
        for child in node.children.values():
            nodes.extend(self.traverse(child))
        return nodes
