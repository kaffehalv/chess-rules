"""Tests for chess_rules. Run from this directory: python3 -m pytest -q

The loader is tested with synthetic records, `unpack` with hand-built and synthetic
records and with real records against python-chess. The tests with real data build a
small dataset with the chess-rules binary (build/chess-rules, or the path in $CR_BIN) and are
skipped if there is none.
"""

from __future__ import annotations

import importlib.util
import os
import pickle
import subprocess
from pathlib import Path

import chess
import numpy as np
import pytest
import torch

import chess_rules as cr

ROOT = Path(__file__).resolve().parent.parent
MAX_PICKLE_BYTES = 10_000
MIN_BLACK_ROWS, MAX_BLACK_ROWS = 100, 300
PIECE_TYPES = [
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
]


def load_audit():
    """Load the independent audit script, the reference implementation."""
    spec = importlib.util.spec_from_file_location("audit", ROOT / "tools" / "audit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = load_audit()


def make_record(**fields):
    """One raw record (uint8 [1, 80]) with the given fields set."""
    record = np.zeros(1, dtype=cr.RECORD_DTYPE)
    record["ep"] = cr.NO_EP
    for name, value in fields.items():
        record[name] = value
    return torch.from_numpy(record.view(np.uint8).reshape(1, cr.RECORD_BYTES).copy())


def bit(rank, file):
    """Return the bitboard with one square set (a1 = bit 0, h1 = 7, a8 = 56)."""
    return 1 << (rank * 8 + file)


def code(origin, to, promo=0):
    """Return the move code (to | from << 6 | promotion << 12) of a move."""
    return to | origin << 6 | promo << 12


def write_records(path, n):
    """Write a file of n records; record i carries i in its first bitboard."""
    records = np.zeros(n, dtype=cr.RECORD_DTYPE)
    ids = np.arange(n)
    records["bb"][:, 0] = ids
    records["legal"] = ids % 20480
    records["illegal"] = (ids * 7) % 20480
    records["illegal"][records["illegal"] == cr.RESIGN_CODE] += 1
    records["ep"] = cr.NO_EP
    records["wtm"] = ids % 2
    records["kind"] = ids % len(cr.KINDS)
    records["reason"] = ids % len(cr.REASONS)
    records.tofile(path)
    return path


def read_raw(path, n):
    """Read n records of a file as a raw uint8 tensor [n, 80]."""
    return torch.from_numpy(
        np.fromfile(path, dtype=np.uint8).reshape(n, cr.RECORD_BYTES)
    )


def record_ids(raw):
    """Return the ids written by write_records."""
    return raw[:, :8].contiguous().view(torch.int64).flatten()


def epoch_ids(dataset, epoch):
    """Ids of all records of an epoch, in the order the batches are visited."""
    return torch.cat([record_ids(dataset[(epoch, k)]) for k in range(len(dataset))])


def hot(vector):
    """Return the indices of the non-zero entries of a vector."""
    return torch.nonzero(vector).flatten().tolist()


def plane_squares(plane):
    """Return the (rank, file) pairs that are set in an 8x8 plane."""
    return [tuple(index) for index in torch.nonzero(plane).tolist()]


# ----------------------------------------
# layout
# ----------------------------------------
def test_layout_matches_the_audit_script():
    assert audit.DTYPE == cr.RECORD_DTYPE
    assert audit.KINDS == cr.KINDS
    assert audit.REASONS == cr.REASONS
    assert audit.RESIGN_CODE == cr.RESIGN_CODE
    assert audit.NO_EP == cr.NO_EP


# ----------------------------------------
# dataset
# ----------------------------------------
def test_unshuffled_batches_are_consecutive_records(tmp_path):
    dataset = cr.ChessRulesDataset(
        write_records(tmp_path / "d.bin", 1000), 64, shuffle=False
    )

    assert len(dataset) == 1000 // 64
    raw = dataset[(0, 2)]
    assert raw.shape == (64, cr.RECORD_BYTES)
    assert raw.dtype == torch.uint8
    assert record_ids(raw).tolist() == list(range(128, 192))


def test_shuffled_epoch_uses_every_record_at_most_once(tmp_path):
    n, batch = 1000, 64
    dataset = cr.ChessRulesDataset(write_records(tmp_path / "d.bin", n), batch, seed=3)

    assert len(dataset) == (n - batch + 1) // batch
    for epoch in range(5):
        ids = epoch_ids(dataset, epoch)
        assert len(ids) == len(dataset) * batch
        assert len(set(ids.tolist())) == len(ids)
        assert (
            len(ids) > n - 2 * batch
        )  # the offset and the remainder leave out less than two batches
        assert ids.min() >= 0
        assert ids.max() < n


def test_epochs_are_reproducible_and_different(tmp_path):
    path = write_records(tmp_path / "d.bin", 1000)
    first = cr.ChessRulesDataset(path, 64, seed=3)
    again = cr.ChessRulesDataset(path, 64, seed=3)
    other_seed = cr.ChessRulesDataset(path, 64, seed=4)

    assert torch.equal(epoch_ids(first, 2), epoch_ids(again, 2))
    assert not torch.equal(epoch_ids(first, 2), epoch_ids(first, 3))
    assert not torch.equal(epoch_ids(first, 2), epoch_ids(other_seed, 2))

    # the batch boundaries move: the set of records in a batch changes between epochs
    epochs = 10
    batches = {frozenset(record_ids(first[(e, 0)]).tolist()) for e in range(epochs)}
    assert len(batches) == epochs


def test_the_returned_batch_is_a_writable_copy(tmp_path):
    path = write_records(tmp_path / "d.bin", 200)
    dataset = cr.ChessRulesDataset(path, 32, shuffle=False)

    raw = dataset[(0, 0)]
    raw[0, 0] = 255  # would raise or change the file if this were the memory map itself
    assert record_ids(dataset[(0, 0)])[0].item() == 0


def test_invalid_items_and_files_are_refused(tmp_path):
    path = write_records(tmp_path / "d.bin", 200)
    dataset = cr.ChessRulesDataset(path, 32, shuffle=False)
    with pytest.raises(IndexError):
        dataset[(0, len(dataset))]

    with pytest.raises(ValueError, match="does not fit"):
        cr.ChessRulesDataset(path, 201)

    truncated = tmp_path / "truncated.bin"
    truncated.write_bytes(path.read_bytes() + b"garbage")
    with pytest.raises(ValueError, match="whole number"):
        cr.ChessRulesDataset(truncated, 32)

    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match="whole number"):
        cr.ChessRulesDataset(empty, 32)


def test_the_dataset_pickles_without_its_memory_map(tmp_path):
    dataset = cr.ChessRulesDataset(write_records(tmp_path / "d.bin", 2000), 64)
    dataset[(0, 0)]  # opens the memory map

    payload = pickle.dumps(dataset)
    assert len(payload) < MAX_PICKLE_BYTES
    restored = pickle.loads(payload)
    assert torch.equal(restored[(1, 1)], dataset[(1, 1)])


# ----------------------------------------
# sampler and loader
# ----------------------------------------
def test_sampler_advances_one_epoch_per_iteration(tmp_path):
    dataset = cr.ChessRulesDataset(
        write_records(tmp_path / "d.bin", 500), 50, shuffle=False
    )
    sampler = cr.EpochSampler(dataset, start_epoch=2)

    assert list(sampler) == [(2, k) for k in range(10)]
    assert list(sampler) == [(3, k) for k in range(10)]
    sampler.set_epoch(7)
    assert next(iter(sampler)) == (7, 0)


def test_sampler_splits_an_epoch_between_ranks(tmp_path):
    dataset = cr.ChessRulesDataset(
        write_records(tmp_path / "d.bin", 500), 50, shuffle=False
    )

    parts = [list(cr.EpochSampler(dataset, rank=r, world_size=3)) for r in range(3)]
    assert sorted(item for part in parts for item in part) == [
        (0, k) for k in range(10)
    ]
    assert [len(cr.EpochSampler(dataset, rank=r, world_size=3)) for r in range(3)] == [
        4,
        3,
        3,
    ]

    with pytest.raises(ValueError, match="rank"):
        cr.EpochSampler(dataset, rank=3, world_size=3)


@pytest.mark.parametrize("num_workers", [0, 2])
def test_loader_yields_the_epochs_of_the_dataset(tmp_path, num_workers):
    path = write_records(tmp_path / "d.bin", 1000)
    reference = cr.ChessRulesDataset(path, 64, seed=5)
    loader = cr.make_loader(path, 64, num_workers=num_workers, seed=5, pin_memory=False)

    assert len(loader) == len(reference)
    for epoch in range(3):  # persistent workers must follow the epochs as well
        batches = list(loader)
        assert len(batches) == len(reference)
        assert all(batch.shape == (64, cr.RECORD_BYTES) for batch in batches)
        assert torch.equal(
            torch.cat([record_ids(b) for b in batches]), epoch_ids(reference, epoch)
        )


def test_a_resumed_loader_continues_with_the_right_epoch(tmp_path):
    path = write_records(tmp_path / "d.bin", 1000)
    fresh = cr.make_loader(path, 64, num_workers=0, seed=5, pin_memory=False)
    list(fresh)
    second_epoch = torch.cat([record_ids(b) for b in fresh])

    resumed = cr.make_loader(
        path, 64, num_workers=0, seed=5, start_epoch=1, pin_memory=False
    )
    assert torch.equal(torch.cat([record_ids(b) for b in resumed]), second_epoch)


# ----------------------------------------
# unpack: hand-built records
# ----------------------------------------
def test_white_to_move_keeps_the_orientation():
    # white: pawn a2 and king e1, black: pawn d7, en passant file c
    white = bit(1, 0) | bit(0, 4)
    black = bit(6, 3)
    pawns = bit(1, 0) | bit(6, 3)
    raw = make_record(bb=[white, black, pawns, 0, 0, 0, 0, bit(0, 4), 0], wtm=1, ep=2)
    batch = cr.unpack(raw)

    assert batch.board.shape == (1, cr.NUM_BOARD_PLANES, 8, 8)
    assert plane_squares(batch.board[0, 0]) == [(1, 0)]  # our pawn a2
    assert plane_squares(batch.board[0, 5]) == [(0, 4)]  # our king e1
    assert plane_squares(batch.board[0, 6]) == [(6, 3)]  # their pawn d7
    assert plane_squares(batch.board[0, cr.EN_PASSANT_TARGET]) == [(5, 2)]
    assert not batch.black_to_move.any()


def test_black_to_move_mirrors_the_board_and_swaps_the_colours():
    # black to move: black pawn a7, black rook h8 with a castling right,
    # white pawn h2, white rook a1 with a castling right, en passant file d
    white = bit(1, 7) | bit(0, 0)
    black = bit(6, 0) | bit(7, 7)
    pawns = bit(6, 0) | bit(1, 7)
    rooks = bit(7, 7) | bit(0, 0)
    raw = make_record(bb=[white, black, pawns, 0, 0, rooks, 0, 0, rooks], wtm=0, ep=3)
    batch = cr.unpack(raw)
    board = batch.board[0]

    assert plane_squares(board[0]) == [
        (1, 0)
    ]  # our pawn: black a7 seen from our side is a2
    assert plane_squares(board[6]) == [(6, 7)]  # their pawn: white h2 is h7
    assert plane_squares(board[3]) == [(0, 7)]  # our rook: black h8 is h1
    assert plane_squares(board[9]) == [(7, 0)]  # their rook: white a1 is a8
    assert plane_squares(board[cr.US_CASTLING_ROOKS]) == [(0, 7)]
    assert plane_squares(board[cr.THEM_CASTLING_ROOKS]) == [(7, 0)]
    assert plane_squares(board[cr.EN_PASSANT_TARGET]) == [(5, 3)]  # always on rank 5
    assert batch.black_to_move.all()


def test_no_en_passant_gives_an_empty_plane():
    batch = cr.unpack(make_record(wtm=0))

    assert batch.board[0, cr.EN_PASSANT_TARGET].sum() == 0


def test_the_highest_bit_survives():
    raw = make_record(bb=[bit(7, 7), 0, bit(7, 7), 0, 0, 0, 0, 0, 0], wtm=1)

    assert plane_squares(cr.unpack(raw).board[0, 0]) == [(7, 7)]


def test_moves_are_mirrored_for_black_only():
    e7e5 = code(52, 36)  # e7 -> e5
    black = cr.unpack(make_record(wtm=0, legal=e7e5, illegal=code(52, 20)))
    white = cr.unpack(make_record(wtm=1, legal=code(12, 28), illegal=code(12, 20)))

    # black e7e5 seen from black's side is e2e4
    assert hot(black.legal[0, cr.MOVE_FROM]) == [12]
    assert hot(black.legal[0, cr.MOVE_TO]) == [28]
    assert hot(black.illegal[0, cr.MOVE_TO]) == [44]  # e7e3 is e2e6
    assert hot(white.legal[0, cr.MOVE_FROM]) == [12]
    assert hot(white.legal[0, cr.MOVE_TO]) == [28]
    # the codes in the batch are the ones of the record, not flipped
    assert black.legal_code.tolist() == [e7e5]


def test_promotion_and_resign_entries():
    vectors = cr.move_vectors(
        torch.tensor([code(48, 56, 3), code(12, 28), cr.RESIGN_CODE]),
        torch.tensor([False, False, False]),
    )

    assert hot(vectors[0, cr.MOVE_PROMOTION]) == [2]  # knight, bishop, rook
    assert hot(vectors[1, cr.MOVE_PROMOTION]) == [4]  # none is last
    assert vectors[0, cr.MOVE_RESIGN] == 0
    assert hot(vectors[2]) == [
        cr.MOVE_PROMOTION.start + 4,
        cr.MOVE_RESIGN,
    ]  # no squares
    assert vectors.shape == (3, cr.MOVE_FEATURES) == (3, 134)


def test_split_move_cuts_planes_and_scalars():
    vectors = cr.move_vectors(torch.tensor([code(12, 28, 4)]), torch.tensor([False]))
    planes, scalars = cr.split_move(vectors)

    assert planes.shape == (1, 2, 8, 8)
    assert scalars.shape == (1, cr.MOVE_SCALARS)
    assert plane_squares(planes[0, 0]) == [(1, 4)]  # from e2
    assert plane_squares(planes[0, 1]) == [(3, 4)]  # to e4
    assert hot(scalars[0]) == [3]  # queen


def test_unpack_refuses_malformed_input():
    with pytest.raises(ValueError, match="uint8"):
        cr.unpack(torch.zeros(2, cr.RECORD_BYTES, dtype=torch.int64))
    with pytest.raises(ValueError, match="uint8"):
        cr.unpack(torch.zeros(2, cr.RECORD_BYTES - 1, dtype=torch.uint8))


# ----------------------------------------
# unpack: augmentation
# ----------------------------------------
AUGMENTATION_RECORDS = 2000
BEFORE_FLAG = slice(0, cr.MOVE_RESIGN)
SQUARE_PLANES = slice(0, cr.MOVE_PROMOTION.start)  # the from and the to plane
SQUARE_MARKS = 2  # one set bit in each of them


def synthetic_raw(tmp_path):
    """Return the synthetic records (both colours to move) as a raw uint8 tensor."""
    n = AUGMENTATION_RECORDS
    return read_raw(write_records(tmp_path / "d.bin", n), n)


def seeded(seed):
    return torch.Generator().manual_seed(seed)


def assert_nothing_else_changed(batch, plain):
    """Check everything but the legal slot: the pair keeps its labels (1, 0)."""
    assert torch.equal(batch.board, plain.board)
    assert torch.equal(
        batch.illegal, plain.illegal
    )  # the illegal move is never augmented
    assert torch.equal(batch.legal_code, plain.legal_code)
    assert torch.equal(batch.illegal_code, plain.illegal_code)
    assert torch.equal(batch.legal_kind, plain.legal_kind)
    assert torch.equal(batch.illegal_reason, plain.illegal_reason)
    assert torch.equal(batch.black_to_move, plain.black_to_move)


def test_without_augmentation_nothing_is_marked(tmp_path):
    raw = synthetic_raw(tmp_path)
    plain, batch = cr.unpack(raw), cr.unpack(raw, resign_prob=0.0)

    assert plain.black_to_move.any()
    assert not plain.black_to_move.all()
    assert not batch.augmented.any()
    assert not batch.legal_from_illegal.any()
    assert torch.equal(
        batch.legal, cr.move_vectors(plain.legal_code, plain.black_to_move)
    )
    assert torch.equal(batch.legal, plain.legal)
    assert_nothing_else_changed(batch, plain)


def test_legal_and_resign(tmp_path):
    """(legal move, illegal move) -> (legal move & resign, illegal move)"""
    raw = synthetic_raw(tmp_path)
    plain = cr.unpack(raw)
    batch = cr.unpack(raw, resign_prob=1.0, illegal_prob=0.0)
    natural = plain.legal_code == cr.RESIGN_CODE  # record 65 is a resign to begin with
    assert natural.sum() == 1

    assert torch.equal(batch.augmented, ~natural)  # a resign is not "augmented"
    assert not batch.legal_from_illegal.any()
    changed = batch.augmented
    assert torch.all(batch.legal[changed][:, cr.MOVE_RESIGN] == 1)
    # only the flag differs: the squares and the promotion of the legal move are kept
    assert torch.equal(
        batch.legal[changed][:, BEFORE_FLAG], plain.legal[changed][:, BEFORE_FLAG]
    )
    kept = batch.legal[changed][:, SQUARE_PLANES]
    assert torch.all(kept.sum(dim=1) == SQUARE_MARKS)  # not the squareless resign
    assert torch.equal(batch.legal[~changed], plain.legal[~changed])
    assert_nothing_else_changed(batch, plain)


def test_illegal_and_resign(tmp_path):
    """(legal move, illegal move) -> (illegal move & resign, illegal move)"""
    raw = synthetic_raw(tmp_path)
    plain = cr.unpack(raw)
    batch = cr.unpack(raw, resign_prob=1.0, illegal_prob=1.0)

    assert batch.augmented.all()
    assert batch.legal_from_illegal.all()
    # the legal slot has the squares and promotion of the illegal move, plus the flag
    expected = cr.move_vectors(
        plain.illegal_code, plain.black_to_move, resign=batch.augmented
    )
    assert torch.equal(batch.legal, expected)
    assert torch.equal(batch.legal[:, BEFORE_FLAG], plain.illegal[:, BEFORE_FLAG])
    assert torch.all(batch.legal[:, cr.MOVE_RESIGN] == 1)
    assert torch.all(batch.illegal[:, cr.MOVE_RESIGN] == 0)  # the flag tells them apart
    assert torch.all(batch.legal[:, SQUARE_PLANES].sum(dim=1) == SQUARE_MARKS)
    assert_nothing_else_changed(batch, plain)


def test_each_augmented_record_uses_one_of_its_two_moves(tmp_path):
    raw = synthetic_raw(tmp_path)
    plain = cr.unpack(raw)
    batch = cr.unpack(raw, resign_prob=1.0, generator=seeded(1))
    natural = plain.legal_code == cr.RESIGN_CODE

    assert torch.all(batch.augmented[~natural])
    assert not (batch.legal_from_illegal & ~batch.augmented).any()
    from_illegal = batch.legal_from_illegal
    from_legal = batch.augmented & ~from_illegal
    assert from_illegal.float().mean().item() == pytest.approx(0.5, abs=0.1)

    black = plain.black_to_move
    expected_from_legal = cr.move_vectors(plain.legal_code, black, resign=from_legal)
    expected_from_illegal = cr.move_vectors(
        plain.illegal_code, black, resign=from_illegal
    )
    assert torch.equal(batch.legal[from_legal], expected_from_legal[from_legal])
    assert torch.equal(batch.legal[from_illegal], expected_from_illegal[from_illegal])
    assert torch.equal(batch.legal[~batch.augmented], plain.legal[~batch.augmented])
    assert_nothing_else_changed(batch, plain)

    again = cr.unpack(raw, resign_prob=1.0, generator=seeded(1))
    assert torch.equal(batch.augmented, again.augmented)
    assert torch.equal(batch.legal_from_illegal, again.legal_from_illegal)
    assert torch.equal(batch.legal, again.legal)


def test_resign_prob_is_the_share_of_augmented_records(tmp_path):
    batch = cr.unpack(synthetic_raw(tmp_path), resign_prob=0.3, generator=seeded(2))

    assert batch.augmented.float().mean().item() == pytest.approx(0.3, abs=0.06)


def test_move_vectors_can_set_the_resign_flag_without_touching_the_rest():
    codes = torch.tensor([code(12, 28), code(48, 56, 3), cr.RESIGN_CODE])
    black = torch.tensor([False, True, False])
    plain = cr.move_vectors(codes, black)
    flagged = cr.move_vectors(codes, black, resign=torch.tensor([True, False, True]))

    assert flagged[0, cr.MOVE_RESIGN] == 1
    assert torch.equal(flagged[0, : cr.MOVE_RESIGN], plain[0, : cr.MOVE_RESIGN])
    assert torch.equal(flagged[1], plain[1])
    assert torch.equal(
        flagged[2], plain[2]
    )  # already a resign: the vector stays as it is


def test_unpack_converts_to_the_requested_dtype():
    batch = cr.unpack(make_record(legal=code(12, 28)), dtype=torch.bfloat16)

    assert batch.board.dtype == torch.bfloat16
    assert batch.legal.dtype == torch.bfloat16
    assert batch.illegal.dtype == torch.bfloat16


@pytest.fixture(scope="module")
def real_dataset(tmp_path_factory):
    binary = Path(os.environ.get("CR_BIN", ROOT / "build" / "chess-rules"))
    if not binary.exists():
        pytest.skip(f"no chess-rules binary at {binary}")
    path = tmp_path_factory.mktemp("real") / "real.bin"
    subprocess.run(
        [
            binary,
            "--split",
            "train",
            "--samples-per-kind",
            "60",
            "--out",
            path,
            "--threads",
            "1",
            "--seed",
            "1",
        ],
        check=True,
        capture_output=True,
    )
    return path


# ----------------------------------------
# unpack: real records against python-chess
# ----------------------------------------
def expected_planes(board):
    """Derive the planes of a position seen from the side to move from python-chess."""
    reference = board if board.turn == chess.WHITE else board.mirror()
    planes = np.zeros((cr.NUM_BOARD_PLANES, 8, 8), dtype=np.float32)
    for color_index, color in enumerate((chess.WHITE, chess.BLACK)):
        for type_index, piece_type in enumerate(PIECE_TYPES):
            for square in reference.pieces(piece_type, color):
                planes[6 * color_index + type_index, square // 8, square % 8] = 1
    for square in chess.SquareSet(reference.castling_rights):
        ours = reference.piece_at(square).color == chess.WHITE
        plane = cr.US_CASTLING_ROOKS if ours else cr.THEM_CASTLING_ROOKS
        planes[plane, square // 8, square % 8] = 1
    if reference.ep_square is not None:
        planes[
            cr.EN_PASSANT_TARGET, reference.ep_square // 8, reference.ep_square % 8
        ] = 1
    return planes


def expected_move(move_code, black_to_move):
    """Derive a flipped move vector from a code with plain arithmetic."""
    vector = np.zeros(cr.MOVE_FEATURES, dtype=np.float32)
    if move_code == cr.RESIGN_CODE:
        vector[cr.MOVE_PROMOTION.start + cr.NUM_PROMOTIONS - 1] = 1
        vector[cr.MOVE_RESIGN] = 1
        return vector
    origin, to, promo = move_code >> 6 & 63, move_code & 63, move_code >> 12
    if black_to_move:
        origin, to = chess.square_mirror(origin), chess.square_mirror(to)
    vector[cr.MOVE_FROM.start + origin] = 1
    vector[cr.MOVE_TO.start + to] = 1
    vector[cr.MOVE_PROMOTION.start + (promo - 1) % cr.NUM_PROMOTIONS] = 1
    return vector


def test_planes_agree_with_python_chess_mirror_on_real_records(real_dataset):
    records = np.memmap(real_dataset, dtype=cr.RECORD_DTYPE, mode="r")
    raw = torch.from_numpy(
        np.array(records[:400]).view(np.uint8).reshape(-1, cr.RECORD_BYTES)
    )
    batch = cr.unpack(raw)

    black_rows = 0
    for i, record in enumerate(records[:400]):
        board = audit.build_board(record)
        black_rows += board.turn == chess.BLACK
        assert np.array_equal(batch.board[i].numpy(), expected_planes(board)), i
        black = board.turn == chess.BLACK
        assert np.array_equal(
            batch.legal[i].numpy(), expected_move(int(record["legal"]), black)
        ), i
        assert np.array_equal(
            batch.illegal[i].numpy(), expected_move(int(record["illegal"]), black)
        ), i
    assert MIN_BLACK_ROWS < black_rows < MAX_BLACK_ROWS  # both colours occur


def test_the_flip_is_a_symmetry_of_the_rules(real_dataset):
    """Legality is the same when the position is mirrored and the colours swapped."""
    records = np.memmap(real_dataset, dtype=cr.RECORD_DTYPE, mode="r")

    for i, record in enumerate(records):
        board = audit.build_board(record)
        mirrored = board.mirror()
        legal = set(board.legal_moves)
        legal_mirrored = set(mirrored.legal_moves)
        flipped = {
            chess.Move(
                chess.square_mirror(move.from_square),
                chess.square_mirror(move.to_square),
                move.promotion,
            )
            for move in legal
        }
        assert flipped == legal_mirrored, (i, board.fen())
        assert board.is_check() == mirrored.is_check()


def test_a_real_epoch_contains_every_kind_and_reason(real_dataset):
    loader = cr.make_loader(real_dataset, 256, num_workers=2, pin_memory=False)
    kinds = torch.zeros(len(cr.KINDS), dtype=torch.int64)
    reasons = torch.zeros(len(cr.REASONS), dtype=torch.int64)
    for raw in loader:
        batch = cr.unpack(raw)
        kinds += torch.bincount(batch.legal_kind, minlength=len(cr.KINDS))
        reasons += torch.bincount(batch.illegal_reason, minlength=len(cr.REASONS))

    assert torch.all(kinds > 0)
    assert torch.all(reasons > 0)
    assert kinds.sum() == reasons.sum() == len(loader) * 256
