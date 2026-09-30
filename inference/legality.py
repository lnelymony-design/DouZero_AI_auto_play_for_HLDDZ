"""Reuse the project's Dou Dizhu move rules to test whether a hidden hand can beat a move."""
from douzero.env import move_detector as md
from douzero.env import move_selector as ms
from douzero.env.move_generator import MovesGener


def _candidate_responses(hand_cards, rival_move):
    """Return non-pass legal responses from hand_cards against rival_move."""
    hand_cards = sorted(hand_cards)
    rival_move = sorted(rival_move)

    if not rival_move:
        return []

    mg = MovesGener(hand_cards)
    rival_type = md.get_move_type(rival_move)
    move_type = rival_type["type"]
    move_len = rival_type.get("len", 1)

    if move_type == md.TYPE_5_KING_BOMB:
        return []
    if move_type == md.TYPE_1_SINGLE:
        moves = ms.filter_type_1_single(mg.gen_type_1_single(), rival_move)
    elif move_type == md.TYPE_2_PAIR:
        moves = ms.filter_type_2_pair(mg.gen_type_2_pair(), rival_move)
    elif move_type == md.TYPE_3_TRIPLE:
        moves = ms.filter_type_3_triple(mg.gen_type_3_triple(), rival_move)
    elif move_type == md.TYPE_4_BOMB:
        moves = ms.filter_type_4_bomb(
            mg.gen_type_4_bomb() + mg.gen_type_5_king_bomb(), rival_move
        )
    elif move_type == md.TYPE_6_3_1:
        moves = ms.filter_type_6_3_1(mg.gen_type_6_3_1(), rival_move)
    elif move_type == md.TYPE_7_3_2:
        moves = ms.filter_type_7_3_2(mg.gen_type_7_3_2(), rival_move)
    elif move_type == md.TYPE_8_SERIAL_SINGLE:
        moves = ms.filter_type_8_serial_single(
            mg.gen_type_8_serial_single(repeat_num=move_len), rival_move
        )
    elif move_type == md.TYPE_9_SERIAL_PAIR:
        moves = ms.filter_type_9_serial_pair(
            mg.gen_type_9_serial_pair(repeat_num=move_len), rival_move
        )
    elif move_type == md.TYPE_10_SERIAL_TRIPLE:
        moves = ms.filter_type_10_serial_triple(
            mg.gen_type_10_serial_triple(repeat_num=move_len), rival_move
        )
    elif move_type == md.TYPE_11_SERIAL_3_1:
        moves = ms.filter_type_11_serial_3_1(
            mg.gen_type_11_serial_3_1(repeat_num=move_len), rival_move
        )
    elif move_type == md.TYPE_12_SERIAL_3_2:
        moves = ms.filter_type_12_serial_3_2(
            mg.gen_type_12_serial_3_2(repeat_num=move_len), rival_move
        )
    elif move_type == md.TYPE_13_4_2:
        moves = ms.filter_type_13_4_2(mg.gen_type_13_4_2(), rival_move)
    elif move_type == md.TYPE_14_4_22:
        moves = ms.filter_type_14_4_22(mg.gen_type_14_4_22(), rival_move)
    else:
        moves = []

    # Any ordinary non-bomb move can also be beaten by a bomb/rocket.
    if move_type not in (md.TYPE_4_BOMB, md.TYPE_5_KING_BOMB):
        moves = moves + mg.gen_type_4_bomb() + mg.gen_type_5_king_bomb()

    return [m for m in moves if m]


def can_beat(hand_cards, rival_move):
    """Whether hand_cards contains at least one legal non-pass response."""
    return bool(_candidate_responses(hand_cards, rival_move))
