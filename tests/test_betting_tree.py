from game_engine.betting_tree import BettingTree


def test_add_and_traverse():
    tree = BettingTree()
    tree.add_path(["bet", "call", "flop", "check", "bet"])
    nodes = tree.traverse()
    actions = [n.action for n in nodes if n.action]
    assert "bet" in actions
    assert "call" in actions
    assert any(n.street == "flop" for n in nodes)
