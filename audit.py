"""Independent audit of an chess-rules dataset using python-chess (Chess960 mode).

For a random sample of records it rebuilds the position, then re-derives
  * that the legal move is legal (or that the position has no legal move for resign),
  * that the illegal move is not legal,
  * the legal kind and the illegal reason, with its own classifier (same
    precedence rules as the C++ generator, different code),
and reports every disagreement. It also reports bucket balance, duplicates and
split leakage.

    python3 tools/audit.py train.bin [--sample 200000] [--leak val.bin test.bin]
"""

from __future__ import annotations

import argparse
import collections
import random
import sys

import chess
import numpy as np

RECORD_BYTES = 80
DTYPE = np.dtype(
    [
        ("bb", "<u8", 9),
        ("legal", "<u2"),
        ("illegal", "<u2"),
        ("ep", "u1"),
        ("wtm", "u1"),
        ("kind", "u1"),
        ("reason", "u1"),
    ]
)
if DTYPE.itemsize != RECORD_BYTES:
    message = f"record layout is {DTYPE.itemsize} bytes, expected {RECORD_BYTES}"
    raise SystemExit(message)

KINDS = [
    "pawn_single",
    "pawn_double",
    "pawn_attack",
    "pawn_enpassant",
    "pawn_promotion",
    "knight",
    "bishop",
    "rook",
    "queen",
    "king",
    "castle_king",
    "castle_queen",
    "resign",
]
REASONS = [
    "same_square",
    "empty_from",
    "opponent_piece",
    "own_capture",
    "geometry",
    "blocked",
    "pawn_blocked",
    "pawn_rule",
    "promotion_rule",
    "castling_rule",
    "ks_king_step",
    "ks_pinned",
    "ks_unresolved_check",
    "ks_castling",
    "ks_ep",
]
R = {name: i for i, name in enumerate(REASONS)}
K = {name: i for i, name in enumerate(KINDS)}

RESIGN_CODE = 65
NO_EP = 255
NUM_SQUARES = 64
DOUBLE_PUSH = 2
MAX_SHOWN = 3  # examples printed per kind of problem
PROGRESS_EVERY = 50_000
BOARD_SAMPLE = 500_000  # records used to count distinct boards
DUPLICATE_SCAN_LIMIT = (
    3_000_000  # above this many records the exact duplicate scan is skipped
)
TOP_CONFUSIONS = 10

PROMO = {0: None, 1: chess.KNIGHT, 2: chess.BISHOP, 3: chess.ROOK, 4: chess.QUEEN}
PIECE_KIND = {
    chess.KNIGHT: K["knight"],
    chess.BISHOP: K["bishop"],
    chess.ROOK: K["rook"],
    chess.QUEEN: K["queen"],
    chess.KING: K["king"],
}
SLIDERS = (chess.BISHOP, chess.ROOK, chess.QUEEN)


def build_board(rec):
    """Rebuild the position of a record as a Chess960 python-chess board."""
    bb = [int(x) for x in rec["bb"]]
    board = chess.Board(None, chess960=True)
    types = [
        chess.PAWN,
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
        chess.QUEEN,
        chess.KING,
    ]
    for sq in range(NUM_SQUARES):
        mask = 1 << sq
        if not (bb[0] | bb[1]) & mask:
            continue
        color = chess.WHITE if bb[0] & mask else chess.BLACK
        for i, piece_type in enumerate(types):
            if bb[2 + i] & mask:
                board.set_piece_at(sq, chess.Piece(piece_type, color))
                break
    board.castling_rights = bb[8]
    board.turn = chess.WHITE if rec["wtm"] else chess.BLACK
    ep = int(rec["ep"])
    if ep != NO_EP:
        board.ep_square = (5 if board.turn == chess.WHITE else 2) * 8 + ep
    return board


def decode(code):
    """Split a move code into (from, to, promotion)."""
    code = int(code)
    return code >> 6 & 63, code & 63, code >> 12 & 7


def castling_geometry_ok(board, king_sq, rook_sq):
    """Right present, both on the back rank, path empty (Chess960 rules)."""
    back = 0 if board.turn == chess.WHITE else 7
    if not board.castling_rights & chess.BB_SQUARES[rook_sq]:
        return False
    if chess.square_rank(king_sq) != back or chess.square_rank(rook_sq) != back:
        return False
    a_side = rook_sq < king_sq
    king_to = chess.square(2 if a_side else 6, back)
    rook_to = chess.square(3 if a_side else 5, back)
    occupied = board.occupied & ~chess.BB_SQUARES[king_sq] & ~chess.BB_SQUARES[rook_sq]
    path = (
        chess.between(king_sq, king_to)
        | chess.BB_SQUARES[king_to]
        | chess.between(rook_sq, rook_to)
        | chess.BB_SQUARES[rook_to]
    )
    return not occupied & path


def empty_board_attacks(piece, sq):
    """Squares the piece would attack on an otherwise empty board."""
    board = chess.Board(None)
    board.set_piece_at(sq, piece)
    return board.attacks_mask(sq)


def basic_reason(board, f, t):
    """Return same-square, empty-from or opponent-piece as a reason, else None."""
    if f == t:
        return R["same_square"]
    piece = board.piece_at(f)
    if piece is None:
        return R["empty_from"]
    if piece.color != board.turn:
        return R["opponent_piece"]
    return None


def promotion_reason(board, piece, f, t, promo):
    """Promotion mistakes, and the move whose movement rules are checked next."""
    is_pawn = piece.piece_type == chess.PAWN
    last_rank = 7 if board.turn == chess.WHITE else 0
    to_last = chess.square_rank(t) == last_rank
    if promo:
        invalid = not is_pawn or not to_last
        return (R["promotion_rule"] if invalid else None), chess.Move(
            f, t, PROMO[promo]
        )
    if is_pawn and to_last:
        # promotion piece missing: judge the move as if it had been a queen
        queen = chess.Move(f, t, chess.QUEEN)
        return (R["promotion_rule"] if board.is_pseudo_legal(queen) else None), queen
    return None, chess.Move(f, t)


def king_safety_reason(board, piece, f, t):
    """Explain a move that follows the movement rules but breaks king safety."""
    if piece.piece_type == chess.KING:
        return R["ks_king_step"]
    on_other_file = chess.square_file(f) != chess.square_file(t)
    if piece.piece_type == chess.PAWN and on_other_file and board.piece_at(t) is None:
        return R["ks_ep"]
    return R["ks_unresolved_check"] if board.is_check() else R["ks_pinned"]


def pawn_violation(board, f, t, to_own):
    """Pawn movement mistakes."""
    up = 8 if board.turn == chess.WHITE else -8
    if chess.square_file(f) == chess.square_file(t):
        if t == f + up:
            return R["pawn_blocked"]
        if t == f + 2 * up:
            start = 1 if board.turn == chess.WHITE else 6
            return (
                R["pawn_blocked"] if chess.square_rank(f) == start else R["pawn_rule"]
            )
        return R["pawn_rule"]
    if to_own and chess.BB_PAWN_ATTACKS[board.turn][f] & chess.BB_SQUARES[t]:
        return R["own_capture"]
    return R["pawn_rule"]


def piece_violation(board, piece, f, t):
    """Movement mistakes of pieces other than pawns."""
    if board.attacks_mask(f) & chess.BB_SQUARES[t]:
        return R["own_capture"]
    if (
        piece.piece_type in SLIDERS
        and empty_board_attacks(piece, f) & chess.BB_SQUARES[t]
    ):
        return R["blocked"]
    return R["geometry"]


def classify(board, f, t, promo):
    """Why the move is illegal. Same precedence as the C++ generator, other code."""
    reason = basic_reason(board, f, t)
    if reason is not None:
        return reason
    piece = board.piece_at(f)
    reason, check_move = promotion_reason(board, piece, f, t, promo)
    if reason is not None:
        return reason

    target = board.piece_at(t)
    to_own = target is not None and target.color == board.turn
    if piece.piece_type == chess.KING and to_own and target.piece_type == chess.ROOK:
        # king onto an own rook: a castling attempt
        return (
            R["ks_castling"]
            if castling_geometry_ok(board, f, t)
            else R["castling_rule"]
        )
    if board.is_pseudo_legal(check_move):
        return king_safety_reason(board, piece, f, t)
    if piece.piece_type == chess.PAWN:
        return pawn_violation(board, f, t, to_own)
    return piece_violation(board, piece, f, t)


def pawn_kind(f, t, target, promo):
    """Kind of a legal pawn move."""
    if promo:
        return K["pawn_promotion"]
    same_file = chess.square_file(f) == chess.square_file(t)
    if not same_file and target is None:
        return K["pawn_enpassant"]
    if abs(chess.square_rank(t) - chess.square_rank(f)) == DOUBLE_PUSH:
        return K["pawn_double"]
    return K["pawn_single"] if same_file else K["pawn_attack"]


def kind_of(board, f, t, promo):
    """Kind of a legal move."""
    piece_type = board.piece_at(f).piece_type
    target = board.piece_at(t)
    to_own_rook = (
        target is not None
        and target.color == board.turn
        and target.piece_type == chess.ROOK
    )
    if piece_type == chess.KING and to_own_rook:
        return K["castle_king"] if t > f else K["castle_queen"]
    if piece_type == chess.PAWN:
        return pawn_kind(f, t, target, promo)
    return PIECE_KIND[piece_type]


def position_key(rec):
    """Identity of a position (board, side to move, en passant file)."""
    return (tuple(int(x) for x in rec["bb"]), int(rec["wtm"]), int(rec["ep"]))


def check_legal_move(rec, board, legal_set):
    """Problems with the legal move of a record."""
    code = int(rec["legal"])
    kind = int(rec["kind"])
    f, t, promo = decode(code)
    if kind == K["resign"]:
        problems = []
        if code != RESIGN_CODE:
            problems.append("resign record without the resign code")
        if legal_set:
            problems.append("resign in a position that has legal moves")
        return problems
    if code == RESIGN_CODE:
        return ["resign code used for a non-resign kind"]
    if promo not in PROMO:
        return ["invalid promotion code in the legal move"]
    if chess.Move(f, t, PROMO[promo]) not in legal_set:
        return ["legal move is not legal"]
    got = kind_of(board, f, t, promo)
    if got != kind:
        return [f"kind mismatch (python says {KINDS[got]})"]
    return []


def check_illegal_move(rec, board, legal_set):
    """Problems with the illegal move of a record, and the reason confusion if any."""
    code = int(rec["illegal"])
    f, t, promo = decode(code)
    if code == RESIGN_CODE:
        return ["illegal move is the resign code"], None
    if promo not in PROMO:
        return ["invalid promotion code in the illegal move"], None
    if chess.Move(f, t, PROMO[promo]) in legal_set:
        return ["illegal move is legal"], None
    stored = int(rec["reason"])
    got = classify(board, f, t, promo)
    if got != stored:
        return [f"reason mismatch (python says {REASONS[got]})"], (
            REASONS[stored],
            REASONS[got],
        )
    return [], None


def check_en_passant(board):
    """Check that a recorded double push has the pusher's pawn in front of the file."""
    if board.ep_square is None:
        return []
    sq = board.ep_square
    pawn = board.piece_at(sq + (-8 if board.turn == chess.WHITE else 8))
    if pawn is None or pawn.piece_type != chess.PAWN or board.piece_at(sq) is not None:
        return ["inconsistent en passant flag"]
    return []


def describe(index, rec, board):
    """One line describing a record, for problem reports."""
    lf, lt, lp = decode(rec["legal"])
    xf, xt, xp = decode(rec["illegal"])
    return (
        f"rec {index} fen={board.fen()} legal={chess.Move(lf, lt, PROMO.get(lp))}"
        f" illegal={chess.Move(xf, xt, PROMO.get(xp))}"
        f" kind={KINDS[int(rec['kind'])]} reason={REASONS[int(rec['reason'])]}"
    )


def print_balance(data):
    """Print the counts per legal kind and per illegal reason."""
    kinds = np.bincount(data["kind"], minlength=len(KINDS))
    reasons = np.bincount(data["reason"], minlength=len(REASONS))
    print("legal kinds   :", {KINDS[i]: int(kinds[i]) for i in range(len(KINDS))})
    print("illegal reason:", {REASONS[i]: int(reasons[i]) for i in range(len(REASONS))})
    print(
        f"kind spread  : min {kinds.min()} max {kinds.max()}"
        f"   reason spread: min {reasons.min()} max {reasons.max()}"
    )


def print_duplicates(data):
    """Exact duplicate records and the number of distinct boards."""
    n = len(data)
    if n <= DUPLICATE_SCAN_LIMIT:
        flat = data.view(np.uint8).reshape(n, RECORD_BYTES)
        duplicates = str(n - len({row.tobytes() for row in flat}))
    else:
        duplicates = "skipped"
    sample = min(n, BOARD_SAMPLE)
    boards = {position_key(r) for r in data[:sample]}
    print(
        f"exact duplicate examples: {duplicates};"
        f" distinct boards in the first {sample} records: {len(boards)}"
    )


def audit_sample(data, sample, seed):
    """Audits a random sample of records. Returns the number of problems found."""
    n = len(data)
    rng = random.Random(seed)
    indices = list(range(n)) if sample >= n else rng.sample(range(n), sample)
    problems = collections.Counter()
    shown = collections.Counter()
    confusion = collections.Counter()

    for count, index in enumerate(indices):
        rec = data[index]
        board = build_board(rec)
        legal_set = set(board.legal_moves)
        found = check_legal_move(rec, board, legal_set)
        illegal_problems, pair = check_illegal_move(rec, board, legal_set)
        found += illegal_problems
        found += check_en_passant(board)
        if pair is not None:
            confusion[pair] += 1

        for name in found:
            problems[name] += 1
            if shown[name] < MAX_SHOWN:
                shown[name] += 1
                print(f"PROBLEM {name}: {describe(index, rec, board)}")
        if (count + 1) % PROGRESS_EVERY == 0:
            print(f"  audited {count + 1}/{len(indices)}", file=sys.stderr)

    total = sum(problems.values())
    print(f"audited {len(indices)} records; problems: {total}")
    for name, count in problems.most_common():
        print(f"  {count:8d}  {name}")
    for pair, count in confusion.most_common(TOP_CONFUSIONS):
        print(f"  reason confusion {pair}: {count}")
    return total


def check_leakage(data, other_files):
    """Report positions that also occur in other split files."""
    mine = {position_key(r) for r in data}
    for other in other_files:
        other_data = np.fromfile(other, dtype=DTYPE)
        shared = sum(1 for r in other_data if position_key(r) in mine)
        print(f"positions shared with {other}: {shared} of {len(other_data)}")


def main():
    """Run the audit; the exit code is 1 if any problem was found."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("file")
    parser.add_argument("--sample", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--leak",
        nargs="*",
        default=[],
        help="other split files to check for shared positions",
    )
    args = parser.parse_args()

    data = np.fromfile(args.file, dtype=DTYPE)
    print(f"{args.file}: {len(data)} records")
    print_balance(data)
    print_duplicates(data)
    total = audit_sample(data, args.sample, args.seed)
    if args.leak:
        check_leakage(data, args.leak)
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
