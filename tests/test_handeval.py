from gatc_poker.handeval import best_of


def test_quads_beat_flush() -> None:
    # Board gives many diamonds; P0 has quads Aces; P1 has a diamond flush.
    board = ["Ah", "Ad", "2d", "3d", "4d"]
    players = {
        0: ["As", "Ac"],  # quads A
        1: ["7d", "8d"],  # diamond flush
    }
    winners = best_of(players, board)
    assert winners == [0]


def test_board_plays_everyone_ties() -> None:
    # Royal flush on the board -> all players tie.
    board = ["As", "Ks", "Qs", "Js", "Ts"]
    players = {
        0: ["2c", "3d"],
        1: ["4c", "5d"],
        2: ["9h", "9d"],
    }
    winners = best_of(players, board)
    assert set(winners) == {0, 1, 2}
