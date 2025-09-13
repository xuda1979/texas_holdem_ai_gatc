import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_path = os.path.join(project_root, "src")
for p in (src_path, project_root):
    if p not in sys.path:
        sys.path.insert(0, p)

from poker_ai.engine.betting_tree import BettingTree


def test_add_and_traverse():
    tree = BettingTree()
    tree.add_path(["bet", "call", "flop", "check", "bet"])
    nodes = tree.traverse()
    actions = [n.action for n in nodes if n.action]
    assert "bet" in actions
    assert "call" in actions
    assert any(n.street == "flop" for n in nodes)
