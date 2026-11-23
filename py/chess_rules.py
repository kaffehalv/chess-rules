"""PyTorch input pipeline for chess-rules datasets.

A dataset file is a flat array of 80 byte records (see the README). The loader
keeps the CPU work to a minimum: worker processes only copy a batch of raw
records out of a memory map. `unpack` then turns the raw bytes into tensors on
whatever device they are on, normally the GPU, so a batch of 8192 examples costs
640 KB of transfer and no Python per-example work.

    loader = make_loader("train.bin", batch_size=8192, num_workers=4)
    for _ in range(epochs):
        for raw in loader:  # uint8 [B, 80]
            batch = unpack(raw.to(device, non_blocking=True), resign_prob=0.1)
            ...

Shuffling: chess-rules writes its files already shuffled. Every epoch the batch
boundaries are shifted by a random offset and the batches are visited in a
random order, so batches change from epoch to epoch without any random access
to the file. Everything is a pure function of (seed, epoch), which makes runs
reproducible and lets persistent workers and resumed runs agree.

Layout of the tensors: `unpack` gives 8x8 planes seen from the side to move. The
board is mirrored vertically and the colours are swapped when black is to move, so
"us" always plays up the board and the side to move is never an input. This is a
symmetry of the rules, also for Chess960 and for castling (castling is "king takes
own rook", files do not change), see test_chess_rules.py.

Board planes [B, 15, 8, 8], plane [rank, file], rank 0 is our back rank:

    0-5    our pawns, knights, bishops, rooks, queens, kings
    6-11   their pawns, knights, bishops, rooks, queens, kings
    12     our rooks that still have a castling right
    13     their rooks that still have a castling right
    14     the en passant target square (always on rank 5), all zero if there is none

A move is a vector [B, 134], flipped the same way: a from-square plane (64), a
to-square plane (64), the promotion one-hot (knight, bishop, rook, queen, none) and
the resign flag. Resign has no squares. `split_move` cuts it into planes and scalars.
The legal slot can be augmented into a resign, see `unpack`.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional
from torch.utils.data import DataLoader, Dataset, Sampler

RECORD_BYTES = 80
RECORD_DTYPE = np.dtype(
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
if RECORD_DTYPE.itemsize != RECORD_BYTES:
    message = f"record layout is {RECORD_DTYPE.itemsize} bytes, expected {RECORD_BYTES}"
    raise ImportError(message)

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

BOARD_SIZE = 8
NUM_SQUARES = BOARD_SIZE * BOARD_SIZE
NUM_PROMOTIONS = 5  # in a move code: 0 none, 1 knight, 2 bishop, 3 rook, 4 queen
NUM_BITBOARDS = (
    9  # white, black, pawns, knights, bishops, rooks, queens, kings, castling rooks
)
RESIGN_CODE = 65  # b1b1
NO_EP = 255

# Byte layout of a record.
BITBOARD_BYTES = NUM_BITBOARDS * 8
MOVE_BYTES = slice(BITBOARD_BYTES, BITBOARD_BYTES + 4)  # legal, illegal as uint16
EP_BYTE = BITBOARD_BYTES + 4
WHITE_TO_MOVE_BYTE = BITBOARD_BYTES + 5
KIND_BYTE = BITBOARD_BYTES + 6
REASON_BYTE = BITBOARD_BYTES + 7

NUM_BOARD_PLANES = 15
US_PIECES = slice(0, 6)
THEM_PIECES = slice(6, 12)
US_CASTLING_ROOKS = 12
THEM_CASTLING_ROOKS = 13
EN_PASSANT_TARGET = 14
EN_PASSANT_RANK = 5  # the target square of an en passant capture, from our side

MOVE_FROM = slice(0, NUM_SQUARES)
MOVE_TO = slice(NUM_SQUARES, 2 * NUM_SQUARES)
MOVE_PROMOTION = slice(2 * NUM_SQUARES, 2 * NUM_SQUARES + NUM_PROMOTIONS)
MOVE_RESIGN = MOVE_PROMOTION.stop
MOVE_FEATURES = MOVE_RESIGN + 1  # 134
MOVE_SCALARS = NUM_PROMOTIONS + 1  # promotion one-hot and the resign flag
MIRROR = BOARD_SIZE * (BOARD_SIZE - 1)  # square ^ 56 mirrors the ranks


class ChessRulesDataset(Dataset):
    """A dataset file as a map-style dataset whose items are whole batches.

    The item is a pair `(epoch, k)`: batch `k` of that epoch, returned as a
    uint8 tensor of shape [batch_size, 80]. Use it through `make_loader`, which
    supplies the items with an `EpochSampler`.
    """

    def __init__(self, path, batch_size, *, shuffle=True, seed=0):
        self.path = Path(path)
        size = self.path.stat().st_size
        if size == 0 or size % RECORD_BYTES:
            message = (
                f"{path}: {size} bytes is not a whole number of "
                f"{RECORD_BYTES} byte records"
            )
            raise ValueError(message)
        self.num_records = size // RECORD_BYTES
        if not 0 < batch_size <= self.num_records:
            message = f"batch size {batch_size} does not fit {self.num_records} records"
            raise ValueError(message)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self._records = None  # opened lazily in every process
        self._plan = None  # (epoch, offset, order) of the last epoch asked for

    def __len__(self):
        """Return the batches per epoch (shuffling may lose one to the offset)."""
        if self.shuffle:
            return (self.num_records - (self.batch_size - 1)) // self.batch_size
        return self.num_records // self.batch_size

    def __getitem__(self, index):
        epoch, k = index
        if not 0 <= k < len(self):
            raise IndexError(k)
        offset, order = self.plan(epoch)
        start = offset + int(order[k]) * self.batch_size
        rows = self._open()[start : start + self.batch_size]
        # copy out of the read-only memory map so that the tensor owns its memory
        raw = np.array(rows.view(np.uint8).reshape(-1, RECORD_BYTES))
        return torch.from_numpy(raw)

    def plan(self, epoch):
        """Return the offset of the first batch and the order of the batches."""
        if self._plan is not None and self._plan[0] == epoch:
            return self._plan[1], self._plan[2]
        if self.shuffle:
            rng = np.random.default_rng([self.seed, epoch])
            offset = int(rng.integers(0, self.batch_size))
            order = rng.permutation(len(self))
        else:
            offset = 0
            order = np.arange(len(self))
        self._plan = (epoch, offset, order)
        return offset, order

    def _open(self):
        if self._records is None:
            self._records = np.memmap(self.path, dtype=RECORD_DTYPE, mode="r")
        return self._records

    def __getstate__(self):
        # a memory map must not be pickled: workers started with "spawn" open their own
        state = self.__dict__.copy()
        state["_records"] = None
        return state


class EpochSampler(Sampler):
    """Yields the items `(epoch, k)` of an epoch and moves on to the next epoch.

    `rank` and `world_size` split every epoch between processes of a
    distributed run: rank r gets batches r, r + world_size, ... of the order.
    """

    def __init__(self, dataset, *, start_epoch=0, rank=0, world_size=1):
        super().__init__()
        if not 0 <= rank < world_size:
            message = f"rank {rank} is not in [0, {world_size})"
            raise ValueError(message)
        self.dataset = dataset
        self.epoch = start_epoch
        self.rank = rank
        self.world_size = world_size

    def set_epoch(self, epoch):
        """Make the next iteration run epoch `epoch` (for resuming a run)."""
        self.epoch = epoch

    def __iter__(self):
        epoch = self.epoch
        self.epoch += 1
        for k in range(self.rank, len(self.dataset), self.world_size):
            yield (epoch, k)

    def __len__(self):
        return len(range(self.rank, len(self.dataset), self.world_size))


def make_loader(
    path,
    batch_size,
    *,
    num_workers=4,
    shuffle=True,
    seed=0,
    start_epoch=0,
    rank=0,
    world_size=1,
    pin_memory=None,
    prefetch_factor=4,
):
    """Build a DataLoader whose batches are uint8 tensors [B, 80].

    Every iteration of the loader is one epoch, and the next iteration is the
    next epoch. Pass `start_epoch` to continue a run. The loader never
    collates, shuffles or batches by itself: that is all done by the dataset.
    """
    dataset = ChessRulesDataset(path, batch_size, shuffle=shuffle, seed=seed)
    sampler = EpochSampler(
        dataset, start_epoch=start_epoch, rank=rank, world_size=world_size
    )
    return DataLoader(
        dataset,
        batch_size=None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available() if pin_memory is None else pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )


def check_records(records):
    """Validate raw records (uint8 [B, 80]) and return their count and device."""
    if records.dtype != torch.uint8 or tuple(records.shape[1:]) != (RECORD_BYTES,):
        message = (
            f"expected a uint8 tensor of shape [B, {RECORD_BYTES}], "
            f"got {records.dtype} {tuple(records.shape)}"
        )
        raise ValueError(message)
    return records.shape[0], records.device


def decode_moves(records):
    """Return the legal and illegal move codes, the kind and the reason of records."""
    move_bytes = records[:, MOVE_BYTES].to(
        torch.int64
    )  # legal and illegal, 2 bytes each
    return (
        move_bytes[:, 0] | move_bytes[:, 1] << 8,
        move_bytes[:, 2] | move_bytes[:, 3] << 8,
        records[:, KIND_BYTE].to(torch.int64),
        records[:, REASON_BYTE].to(torch.int64),
    )


def choose_legal_slot(legal_code, illegal_code, resign_prob, illegal_prob, generator):
    """Choose what fills the legal slot of every record (see `unpack`).

    Return the move code of the legal slot, `augmented` and `legal_from_illegal`.
    """
    n, device = legal_code.shape[0], legal_code.device
    augmented = torch.zeros(n, dtype=torch.bool, device=device)
    legal_from_illegal = torch.zeros(n, dtype=torch.bool, device=device)
    if resign_prob > 0:
        chosen = torch.rand(n, generator=generator, device=device) < resign_prob
        use_illegal = torch.rand(n, generator=generator, device=device) < illegal_prob
        legal_from_illegal = chosen & use_illegal
        # a legal move that already is a resign stays as it is
        augmented = chosen & (use_illegal | (legal_code != RESIGN_CODE))
    legal_source = torch.where(legal_from_illegal, illegal_code, legal_code)
    return legal_source, augmented, legal_from_illegal


@dataclasses.dataclass
class Batch:
    """A batch of records as planes, on the device of the raw bytes."""

    board: torch.Tensor  # float [B, 15, 8, 8], seen from the side to move
    legal: torch.Tensor  # float [B, 134], the legal move, flipped like the board
    illegal: torch.Tensor  # float [B, 134], the illegal move, never augmented
    legal_code: torch.Tensor  # int64 [B], the codes in the record (not flipped)
    illegal_code: torch.Tensor  # int64 [B]
    legal_kind: torch.Tensor  # int64 [B], index into chess_rules.KINDS
    illegal_reason: torch.Tensor  # int64 [B], index into chess_rules.REASONS
    augmented: (
        torch.Tensor
    )  # bool [B], `legal` has the resign flag set by the augmentation
    legal_from_illegal: (
        torch.Tensor
    )  # bool [B], `legal` has the squares of the illegal move
    black_to_move: torch.Tensor  # bool [B], the position was mirrored


def unpack(
    records, *, resign_prob=0.0, illegal_prob=0.5, generator=None, dtype=torch.float32
):
    """Turn raw records (uint8 [B, 80], on any device) into a `Batch`.

    Augmentation: a fraction `resign_prob` of the records is chosen. In each of them
    the legal slot becomes a resign: `legal move & resign` (probability
    `1 - illegal_prob`) or `illegal move & resign` (probability `illegal_prob`), where
    `& resign` sets the resign flag and keeps the squares and the promotion. The
    illegal slot and the targets (1 for `legal`, 0 for `illegal`) never change, so the
    network learns that the rest of a resign move does not matter. `augmented` marks
    the rows where the legal slot was changed and `legal_from_illegal` those that took
    the squares of the illegal move. Leave `augmented` rows out of the accuracy per
    legal kind; the illegal side is never changed, so the accuracy per reason can use
    every row.
    """
    n, device = check_records(records)
    white_to_move = records[:, WHITE_TO_MOVE_BYTE].bool()
    black_to_move = ~white_to_move

    # Byte j of a bitboard is rank j, so reversing the bytes mirrors the board.
    bitboard_bytes = records[:, :BITBOARD_BYTES].view(n, NUM_BITBOARDS, 8)
    bitboard_bytes = torch.where(
        white_to_move.view(n, 1, 1), bitboard_bytes, bitboard_bytes.flip(-1)
    )
    shifts = torch.arange(8, dtype=torch.uint8, device=device)
    bits = ((bitboard_bytes.unsqueeze(-1) >> shifts) & 1).view(
        n, NUM_BITBOARDS, NUM_SQUARES
    )

    white, black = bits[:, 0], bits[:, 1]
    us = torch.where(white_to_move.view(n, 1), white, black)
    them = torch.where(white_to_move.view(n, 1), black, white)
    pieces = bits[:, 2:8]
    castling = bits[:, 8]

    files = torch.arange(BOARD_SIZE, dtype=torch.uint8, device=device)
    en_passant = torch.zeros(n, NUM_SQUARES, dtype=torch.uint8, device=device)
    first = EN_PASSANT_RANK * BOARD_SIZE
    en_passant[:, first : first + BOARD_SIZE] = files == records[:, EP_BYTE].unsqueeze(
        1
    )

    planes = torch.cat(
        (
            pieces * us.unsqueeze(1),
            pieces * them.unsqueeze(1),
            (castling * us).unsqueeze(1),
            (castling * them).unsqueeze(1),
            en_passant.unsqueeze(1),
        ),
        dim=1,
    )
    board = planes.to(dtype).view(n, NUM_BOARD_PLANES, BOARD_SIZE, BOARD_SIZE)

    legal_code, illegal_code, legal_kind, illegal_reason = decode_moves(records)
    legal_source, augmented, legal_from_illegal = choose_legal_slot(
        legal_code, illegal_code, resign_prob, illegal_prob, generator
    )
    return Batch(
        board=board,
        legal=move_vectors(legal_source, black_to_move, dtype, resign=augmented),
        illegal=move_vectors(illegal_code, black_to_move, dtype),
        legal_code=legal_code,
        illegal_code=illegal_code,
        legal_kind=legal_kind,
        illegal_reason=illegal_reason,
        augmented=augmented,
        legal_from_illegal=legal_from_illegal,
        black_to_move=black_to_move,
    )


def move_vectors(codes, black_to_move, dtype=torch.float32, resign=None):
    """Return the move vectors [N, 134] of move codes (int64 [N]) from the mover's side.

    The resign code gets the squareless resign vector. `resign` (bool [N]) additionally
    sets the resign flag of the marked rows and keeps their squares and promotion.
    """
    is_resign = codes == RESIGN_CODE
    flag = is_resign if resign is None else is_resign | resign
    mirror = black_to_move.to(torch.int64) * MIRROR
    origin = ((codes >> 6) & (NUM_SQUARES - 1)) ^ mirror
    to = (codes & (NUM_SQUARES - 1)) ^ mirror

    squares = torch.arange(NUM_SQUARES, device=codes.device)
    keep = ~is_resign.unsqueeze(1)
    from_plane = (squares == origin.unsqueeze(1)) & keep
    to_plane = (squares == to.unsqueeze(1)) & keep
    # the code has 0 = none, 1-4 = knight .. queen; the vector has knight .. queen, none
    entry = ((codes >> 12) + NUM_PROMOTIONS - 1) % NUM_PROMOTIONS
    promotion = functional.one_hot(entry, NUM_PROMOTIONS).to(torch.bool)
    return torch.cat((from_plane, to_plane, promotion, flag.unsqueeze(1)), dim=1).to(
        dtype
    )


def split_move(move):
    """Cut move vectors [N, 134] into from/to planes [N, 2, 8, 8] and scalars [N, 6]."""
    planes = move[:, : MOVE_TO.stop]
    scalars = move[:, MOVE_PROMOTION.start :]
    return planes.reshape(-1, 2, BOARD_SIZE, BOARD_SIZE), scalars
