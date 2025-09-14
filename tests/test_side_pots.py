from poker_ai.rules.side_pots import compute_side_pots, total_pot


def test_three_way_all_in_side_pots():
    # Contributions on the street from three active players:
    # A=100, B=300, C=600 (e.g., A open-shoves 100, B calls 100 then shoves to 300, C covers)
    contribs = [100, 300, 600]
    pots = compute_side_pots(contribs)
    # (amount, eligible players)
    assert pots == [
        (300, [0, 1, 2]),  # main: 100 * 3
        (400, [1, 2]),     # side 1: (300-100) * 2
        (300, [2]),        # side 2: (600-300) * 1
    ]
    assert total_pot(contribs) == sum(p for p, _ in pots)
