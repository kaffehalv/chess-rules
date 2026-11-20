"""Tests for chess_rules. Run from this directory: python3 -m pytest -q

The tests that need real data build a small dataset with the chess-rules binary
(build/chess-rules, or the path in $CR_BIN) and are skipped if there is none.
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
SQUARE_GROUPS = (cr.MOVE_TO_RANK, cr.MOVE_TO_FILE, cr.MOVE_FROM_RANK, cr.MOVE_FROM_FILE)


def load_audit():
    """Load the independent audit script, the reference implementation."""
    spec = importlib.util.spec_from_file_location("audit", ROOT / "tools" / "audit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = load_audit()


def write_records(path, n):
    """Write a file of n records; record i carries i in its first bitboard."""
    records = np.zeros(n, dtype=cr.RECORD_DTYPE)
    ids = np.arange(n)
    records["bb"][:, 0] = ids
    records["legal"] = ids % 20480
    records["illegal"] = (ids * 7) % 20480
    records["illegal"][records["illegal"] == cr.RESIGN_CODE] += 1
    records["ep"] = cr.NO_EP
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
    return to | origin << 6 | promo << 12


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
# unpack
# ----------------------------------------


def hot(vector):
    """Return the indices of the non-zero entries of a vector."""
    return torch.nonzero(vector).flatten().tolist()


def test_unpack_places_every_field_on_the_right_feature():
    # bit 63 is the highest bit of a bitboard: the shift must still give the right bit
    white, black = bit(0, 0) | bit(7, 7), bit(1, 1)
    raw = make_record(
        bb=[white, black, bit(0, 0) | bit(7, 7), bit(1, 1), 0, 0, 0, 0, bit(0, 7)],
        legal=code(12, 28),
        illegal=code(48, 56, cr.NUM_PROMOTIONS - 1),
        ep=3,
        wtm=1,
        kind=cr.KINDS.index("pawn_double"),
        reason=cr.REASONS.index("blocked"),
    )

    batch = cr.unpack(raw)
    board = batch.board[0]

    assert board.shape == (cr.BOARD_FEATURES,)
    # square s of bitboard b is feature 64 * b + s
    assert hot(board[cr.BOARD_BITBOARDS]) == [
        0,
        63,
        64 + 9,
        128,
        128 + 63,
        192 + 9,
        512 + 7,
    ]
    assert hot(board[cr.BOARD_EN_PASSANT]) == [3]
    assert board[cr.BOARD_WHITE_TO_MOVE] == 1
    assert board.sum() == 7 + 1 + 1

    assert hot(batch.legal[0, cr.MOVE_FROM_RANK]) == [1]  # e2
    assert hot(batch.legal[0, cr.MOVE_FROM_FILE]) == [4]
    assert hot(batch.legal[0, cr.MOVE_TO_RANK]) == [3]  # e4
    assert hot(batch.legal[0, cr.MOVE_TO_FILE]) == [4]
    assert hot(batch.illegal[0, cr.MOVE_PROMOTION]) == [3]  # a7a8 = queen, entry 3
    assert batch.legal_code.tolist() == [code(12, 28)]
    assert batch.illegal_code.tolist() == [code(48, 56, 4)]
    assert batch.legal_kind.tolist() == [cr.KINDS.index("pawn_double")]
    assert batch.illegal_reason.tolist() == [cr.REASONS.index("blocked")]
    assert not batch.legal_augmented.any()
    assert not batch.illegal_augmented.any()
    assert batch.legal_label.tolist() == [1.0]
    assert batch.illegal_label.tolist() == [0.0]


def test_unpack_without_en_passant_and_black_to_move():
    batch = cr.unpack(make_record(wtm=0))

    assert hot(batch.board[0, cr.BOARD_EN_PASSANT]) == [cr.BOARD_SIZE]  # the none entry
    assert cr.BOARD_EN_PASSANT.start + cr.BOARD_SIZE == cr.BOARD_EN_PASSANT_NONE
    assert batch.board[0, cr.BOARD_WHITE_TO_MOVE] == 0


def test_the_en_passant_group_is_always_one_hot():
    raw = torch.cat(
        [make_record(ep=file) for file in range(cr.BOARD_SIZE)]
        + [make_record(ep=cr.NO_EP)]
    )
    group = cr.unpack(raw).board[:, cr.BOARD_EN_PASSANT]

    assert group.shape == (cr.BOARD_SIZE + 1, cr.BOARD_SIZE + 1)
    assert torch.equal(group, torch.eye(cr.BOARD_SIZE + 1))  # files a-h, then none


def test_unpack_converts_to_the_requested_dtype():
    batch = cr.unpack(make_record(legal=code(12, 28)), dtype=torch.bfloat16)

    assert batch.board.dtype == torch.bfloat16
    assert batch.legal.dtype == torch.bfloat16
    assert batch.illegal.dtype == torch.bfloat16


def test_unpack_refuses_malformed_input():
    with pytest.raises(ValueError, match="uint8"):
        cr.unpack(torch.zeros(4, cr.RECORD_BYTES, dtype=torch.int64))
    with pytest.raises(ValueError, match="uint8"):
        cr.unpack(torch.zeros(4, cr.RECORD_BYTES - 1, dtype=torch.uint8))


def test_move_vectors_have_the_documented_layout():
    codes = torch.tensor([code(12, 28), code(48, 56, 3), cr.RESIGN_CODE, code(1, 1, 4)])
    vectors = cr.move_vectors(codes)

    assert vectors.shape == (4, cr.MOVE_FEATURES)
    assert cr.MOVE_FEATURES == 8 + 8 + 8 + 8 + 5 + 1
    assert cr.MOVE_PROMOTION.start + 4 == cr.MOVE_PROMOTION_NONE
    assert cr.MOVE_RESIGN == cr.MOVE_FEATURES - 1

    e2e4 = vectors[0]
    assert hot(e2e4[cr.MOVE_TO_RANK]) == [3]
    assert hot(e2e4[cr.MOVE_TO_FILE]) == [4]
    assert hot(e2e4[cr.MOVE_FROM_RANK]) == [1]
    assert hot(e2e4[cr.MOVE_FROM_FILE]) == [4]
    assert hot(e2e4[cr.MOVE_PROMOTION]) == [4]  # none is the last entry
    assert e2e4[cr.MOVE_RESIGN] == 0

    rook_promotion = vectors[1]  # a7a8r
    assert hot(rook_promotion[cr.MOVE_FROM_RANK]) == [6]
    assert hot(rook_promotion[cr.MOVE_FROM_FILE]) == [0]
    assert hot(rook_promotion[cr.MOVE_TO_RANK]) == [7]
    assert hot(rook_promotion[cr.MOVE_TO_FILE]) == [0]
    assert hot(rook_promotion[cr.MOVE_PROMOTION]) == [2]  # knight, bishop, rook

    resign = vectors[2]  # no squares, promotion none, resign flag
    assert hot(resign) == [cr.MOVE_PROMOTION_NONE, cr.MOVE_RESIGN]

    # b1b1 with a promotion piece is an ordinary (illegal) move, not resign
    odd = vectors[3]
    assert hot(odd[cr.MOVE_TO_RANK]) == [0]
    assert hot(odd[cr.MOVE_TO_FILE]) == [1]
    assert hot(odd[cr.MOVE_PROMOTION]) == [3]  # queen
    assert odd[cr.MOVE_RESIGN] == 0

    assert vectors.sum(dim=1).tolist() == [5, 5, 2, 5]


AUGMENTATION_RECORDS = 2000
BEFORE_FLAG = slice(0, cr.MOVE_RESIGN)
SQUARES = slice(cr.MOVE_TO_RANK.start, cr.MOVE_PROMOTION.start)


def assert_only_the_flag_changed(batch, plain, side, mask):
    """Check that the marked moves of a side differ in the flag only."""
    moves, plain_moves = getattr(batch, side), getattr(plain, side)
    assert torch.all(moves[mask][:, cr.MOVE_RESIGN] == 1)
    assert torch.equal(moves[mask][:, BEFORE_FLAG], plain_moves[mask][:, BEFORE_FLAG])
    assert torch.equal(moves[~mask], plain_moves[~mask])  # everything else is as it was
    assert torch.equal(getattr(batch, f"{side}_code"), getattr(plain, f"{side}_code"))


def test_without_augmentation_the_targets_are_legal_then_illegal(tmp_path):
    n = AUGMENTATION_RECORDS
    plain = cr.unpack(read_raw(write_records(tmp_path / "d.bin", n), n))

    assert torch.all(plain.legal_label == 1)
    assert torch.all(plain.illegal_label == 0)
    assert not plain.legal_augmented.any()
    assert not plain.illegal_augmented.any()


def test_a_legal_move_can_be_turned_into_resign(tmp_path):
    n = AUGMENTATION_RECORDS
    raw = read_raw(write_records(tmp_path / "d.bin", n), n)
    plain = cr.unpack(raw)
    natural = plain.legal_code == cr.RESIGN_CODE  # record 65 is a resign to begin with
    assert natural.sum() == 1

    batch = cr.unpack(raw, resign_prob=1.0, illegal_prob=0.0)

    assert torch.equal(batch.legal_augmented, ~natural)  # a resign is not "augmented"
    assert not batch.illegal_augmented.any()
    assert_only_the_flag_changed(batch, plain, "legal", batch.legal_augmented)
    assert torch.all(
        batch.legal[batch.legal_augmented][:, SQUARES].sum(dim=1) == len(SQUARE_GROUPS)
    )
    assert torch.equal(batch.legal_kind, plain.legal_kind)
    assert torch.equal(batch.illegal, plain.illegal)  # the illegal move is untouched
    assert torch.all(batch.legal_label == 1)
    assert torch.all(batch.illegal_label == 0)
    assert torch.equal(batch.board, plain.board)


def test_an_illegal_move_can_be_turned_into_a_legal_resign(tmp_path):
    n = AUGMENTATION_RECORDS
    raw = read_raw(write_records(tmp_path / "d.bin", n), n)
    plain = cr.unpack(raw)

    batch = cr.unpack(raw, resign_prob=1.0, illegal_prob=1.0)

    assert batch.illegal_augmented.all()
    assert not batch.legal_augmented.any()
    assert_only_the_flag_changed(batch, plain, "illegal", batch.illegal_augmented)
    squares = batch.illegal[:, SQUARES]
    assert torch.all(squares.sum(dim=1) == len(SQUARE_GROUPS))  # the squares are kept
    assert torch.equal(batch.illegal_reason, plain.illegal_reason)
    assert torch.equal(batch.legal, plain.legal)
    assert torch.all(
        batch.illegal_label == 1
    )  # resign is legal, whatever the rest says
    assert torch.all(batch.legal_label == 1)
    assert torch.equal(batch.board, plain.board)


def test_each_augmented_record_changes_one_of_its_two_moves(tmp_path):
    n = AUGMENTATION_RECORDS
    raw = read_raw(write_records(tmp_path / "d.bin", n), n)
    plain = cr.unpack(raw)
    natural = plain.legal_code == cr.RESIGN_CODE

    batch = cr.unpack(raw, resign_prob=1.0, generator=torch.Generator().manual_seed(1))

    assert not (batch.legal_augmented & batch.illegal_augmented).any()
    assert torch.all((batch.legal_augmented | batch.illegal_augmented)[~natural])
    assert batch.illegal_augmented.float().mean().item() == pytest.approx(0.5, abs=0.1)
    assert_only_the_flag_changed(batch, plain, "legal", batch.legal_augmented)
    assert_only_the_flag_changed(batch, plain, "illegal", batch.illegal_augmented)
    assert torch.equal(batch.illegal_label, batch.illegal_augmented.float())
    assert torch.all(batch.legal_label == 1)

    again = cr.unpack(raw, resign_prob=1.0, generator=torch.Generator().manual_seed(1))
    assert torch.equal(batch.legal_augmented, again.legal_augmented)
    assert torch.equal(batch.illegal_augmented, again.illegal_augmented)


def test_resign_prob_is_the_share_of_augmented_records(tmp_path):
    n = AUGMENTATION_RECORDS
    raw = read_raw(write_records(tmp_path / "d.bin", n), n)

    batch = cr.unpack(raw, resign_prob=0.3, generator=torch.Generator().manual_seed(2))
    changed = batch.legal_augmented | batch.illegal_augmented
    assert changed.float().mean().item() == pytest.approx(0.3, abs=0.06)

    nothing = cr.unpack(raw, resign_prob=0.0)
    assert not nothing.legal_augmented.any()
    assert not nothing.illegal_augmented.any()


def test_move_vectors_can_set_the_resign_flag_without_touching_the_rest():
    codes = torch.tensor([code(12, 28), code(48, 56, 3), cr.RESIGN_CODE])
    plain = cr.move_vectors(codes)
    flagged = cr.move_vectors(codes, resign=torch.tensor([True, False, True]))

    assert flagged[0, cr.MOVE_RESIGN] == 1
    assert torch.equal(flagged[0, : cr.MOVE_RESIGN], plain[0, : cr.MOVE_RESIGN])
    assert torch.equal(flagged[1], plain[1])
    assert torch.equal(
        flagged[2], plain[2]
    )  # already a resign: the vector stays the resign vector


# ----------------------------------------
# real data
# ----------------------------------------


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


def expected_board(board, ep_file):
    """Derive the board vector of a position from python-chess."""
    bitboards = [
        board.occupied_co[chess.WHITE],
        board.occupied_co[chess.BLACK],
        board.pawns,
        board.knights,
        board.bishops,
        board.rooks,
        board.queens,
        board.kings,
        board.castling_rights,
    ]
    vector = np.zeros(cr.BOARD_FEATURES, dtype=np.float32)
    for index, bitboard in enumerate(bitboards):
        for square in chess.SquareSet(bitboard):
            vector[cr.NUM_SQUARES * index + square] = 1
    ep_entry = (
        cr.BOARD_SIZE if ep_file == cr.NO_EP else ep_file
    )  # the last entry is "none"
    vector[cr.BOARD_EN_PASSANT.start + ep_entry] = 1
    vector[cr.BOARD_WHITE_TO_MOVE] = float(board.turn == chess.WHITE)
    return vector


def expected_move(move_code):
    """Compute the move vector of a code with plain arithmetic."""
    vector = np.zeros(cr.MOVE_FEATURES, dtype=np.float32)
    if move_code == cr.RESIGN_CODE:
        vector[cr.MOVE_PROMOTION_NONE] = 1
        vector[cr.MOVE_RESIGN] = 1
        return vector
    origin, to, promo = move_code >> 6 & 63, move_code & 63, move_code >> 12
    vector[cr.MOVE_TO_RANK.start + to // 8] = 1
    vector[cr.MOVE_TO_FILE.start + to % 8] = 1
    vector[cr.MOVE_FROM_RANK.start + origin // 8] = 1
    vector[cr.MOVE_FROM_FILE.start + origin % 8] = 1
    # codes have 0 = none, 1-4 = knight .. queen; vectors have knight .. queen, none
    vector[cr.MOVE_PROMOTION.start + (promo - 1) % cr.NUM_PROMOTIONS] = 1
    return vector


def test_unpack_agrees_with_python_chess_on_real_records(real_dataset):
    records = np.memmap(real_dataset, dtype=cr.RECORD_DTYPE, mode="r")
    loader = cr.make_loader(
        real_dataset, 128, num_workers=0, shuffle=False, pin_memory=False
    )
    batch = cr.unpack(next(iter(loader)))

    for i in range(128):
        record = records[i]
        board = audit.build_board(record)
        assert np.array_equal(
            batch.board[i].numpy(), expected_board(board, int(record["ep"]))
        ), i
        assert np.array_equal(
            batch.legal[i].numpy(), expected_move(int(record["legal"]))
        ), i
        assert np.array_equal(
            batch.illegal[i].numpy(), expected_move(int(record["illegal"]))
        ), i
        assert batch.legal_kind[i].item() == int(record["kind"])
        assert batch.illegal_reason[i].item() == int(record["reason"])


def test_real_records_have_exactly_one_en_passant_entry(real_dataset):
    loader = cr.make_loader(
        real_dataset, 256, num_workers=0, shuffle=False, pin_memory=False
    )
    group = cr.unpack(next(iter(loader))).board[:, cr.BOARD_EN_PASSANT]

    assert torch.all(group.sum(dim=1) == 1)
    assert group[:, cr.BOARD_SIZE].sum() < len(
        group
    )  # some records do have an en passant file


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
