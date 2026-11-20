"""PyTorch input pipeline for chess-rules datasets.

A dataset file is a flat array of 80 byte records (see the README). The loader
keeps the CPU work to a minimum: worker processes only copy a batch of raw
records out of a memory map. `unpack` then turns the raw bytes into tensors on
whatever device they are on, normally the GPU, so a batch of 8192 examples
costs 640 KB of transfer and no Python per-example work. Records are decoded
byte by byte, so the code does not depend on the byte order of the machine.

    loader = make_loader("train.bin", batch_size=8192, num_workers=4)
    for _ in range(epochs):
        for raw in loader:  # uint8 [B, 80]
            batch = unpack(raw.to(device, non_blocking=True), resign_prob=0.1)
            legal_logits = model(batch.board, batch.legal)
            illegal_logits = model(batch.board, batch.illegal)
            loss = bce(legal_logits, batch.legal_label)
            loss += bce(illegal_logits, batch.illegal_label)

Shuffling: chess-rules writes its files already shuffled. Every epoch the batch
boundaries are shifted by a random offset and the batches are visited in a
random order, so batches change from epoch to epoch without any random access
to the file. Everything is a pure function of (seed, epoch), which makes runs
reproducible and lets persistent workers and resumed runs agree.
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
RESIGN_KIND = KINDS.index("resign")

BOARD_SIZE = 8
NUM_SQUARES = BOARD_SIZE * BOARD_SIZE
NUM_PROMOTIONS = 5  # none, knight, bishop, rook, queen
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

# Layout of `Batch.board` [B, 585]: the nine bitboards of the record flattened (square
# a1 first, h8 last), the file of the last double pawn push one-hot, and a
# white-to-move flag.
BOARD_BITBOARDS = slice(0, NUM_BITBOARDS * NUM_SQUARES)  # 576
BOARD_EN_PASSANT = slice(
    BOARD_BITBOARDS.stop, BOARD_BITBOARDS.stop + BOARD_SIZE
)  # zero if none
BOARD_WHITE_TO_MOVE = BOARD_EN_PASSANT.stop
BOARD_FEATURES = BOARD_WHITE_TO_MOVE + 1  # 585

# Layout of a move vector [N, 38]: one-hot to rank, to file, from rank, from file,
# promotion (none, knight, bishop, rook, queen) and a resign flag. Resign has no
# squares: its four square groups are zero, its promotion is "none" and its resign
# flag is set.
MOVE_TO_RANK = slice(0, BOARD_SIZE)
MOVE_TO_FILE = slice(MOVE_TO_RANK.stop, MOVE_TO_RANK.stop + BOARD_SIZE)
MOVE_FROM_RANK = slice(MOVE_TO_FILE.stop, MOVE_TO_FILE.stop + BOARD_SIZE)
MOVE_FROM_FILE = slice(MOVE_FROM_RANK.stop, MOVE_FROM_RANK.stop + BOARD_SIZE)
MOVE_PROMOTION = slice(MOVE_FROM_FILE.stop, MOVE_FROM_FILE.stop + NUM_PROMOTIONS)
MOVE_RESIGN = MOVE_PROMOTION.stop
MOVE_FEATURES = MOVE_RESIGN + 1  # 38


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

    def __getitem__(self, item):
        epoch, k = item
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


@dataclasses.dataclass
class Batch:
    """A batch of records as tensors, on the device of the raw bytes."""

    board: torch.Tensor  # float [B, 585], see BOARD_*
    legal: torch.Tensor  # float [B, 38], move vector of the legal move, see MOVE_*
    illegal: torch.Tensor  # float [B, 38], move vector of the illegal move
    legal_label: torch.Tensor  # float [B], target of the legal move: always 1
    illegal_label: (
        torch.Tensor
    )  # float [B], target of the illegal move: 0, or 1 if augmented
    legal_code: torch.Tensor  # int64 [B], the move code in the record
    illegal_code: torch.Tensor  # int64 [B]
    legal_kind: torch.Tensor  # int64 [B], index into KINDS
    illegal_reason: torch.Tensor  # int64 [B], index into REASONS
    legal_augmented: torch.Tensor  # bool [B], the resign flag of `legal` was set
    illegal_augmented: torch.Tensor  # bool [B], the resign flag of `illegal` was set


def unpack(
    records, *, resign_prob=0.0, illegal_prob=0.5, generator=None, dtype=torch.float32
):
    """Turn raw records (uint8 [B, 80], on any device) into a `Batch`.

    Augmentation: a fraction `resign_prob` of the records is chosen. For each of them
    the legal move (probability 1 - `illegal_prob`) or the illegal move (probability
    `illegal_prob`) gets its resign flag set and nothing else changes: squares,
    promotion, code, kind and reason stay as they are, so the network learns that the
    rest of a resign move does not matter. Resign is legal, so an augmented illegal
    move gets the target 1. `legal_augmented` and `illegal_augmented` mark the changed
    moves; leave them out of metrics per kind and per reason.
    """
    if records.dtype != torch.uint8 or tuple(records.shape[1:]) != (RECORD_BYTES,):
        message = (
            f"expected a uint8 tensor of shape [B, {RECORD_BYTES}], "
            f"got {records.dtype} {tuple(records.shape)}"
        )
        raise ValueError(message)
    n = records.shape[0]
    device = records.device

    # Bit k of byte j of a bitboard is square 8 * j + k (a1 = 0, h1 = 7, a8 = 56). Bytes
    # are shifted as uint8, which is 8 times less memory than going through int64 and
    # does not depend on byte order.
    shifts = torch.arange(8, dtype=torch.uint8, device=device)
    bitboard_bytes = records[:, :BITBOARD_BYTES].view(n, NUM_BITBOARDS, 8)
    bits = ((bitboard_bytes.unsqueeze(-1) >> shifts) & 1).view(n, -1)

    files = torch.arange(BOARD_SIZE, dtype=torch.uint8, device=device)
    en_passant = (files == records[:, EP_BYTE].unsqueeze(1)).to(
        torch.uint8
    )  # zero for NO_EP
    white_to_move = records[:, WHITE_TO_MOVE_BYTE].unsqueeze(1)
    board = torch.cat((bits, en_passant, white_to_move), dim=1).to(dtype)

    move_bytes = records[:, MOVE_BYTES].to(
        torch.int64
    )  # legal and illegal, 2 bytes each
    legal_code = move_bytes[:, 0] | move_bytes[:, 1] << 8
    illegal_code = move_bytes[:, 2] | move_bytes[:, 3] << 8
    legal_kind = records[:, KIND_BYTE].to(torch.int64)
    illegal_reason = records[:, REASON_BYTE].to(torch.int64)

    legal_augmented = torch.zeros(n, dtype=torch.bool, device=device)
    illegal_augmented = torch.zeros(n, dtype=torch.bool, device=device)
    if resign_prob > 0:
        chosen = torch.rand(n, generator=generator, device=device) < resign_prob
        use_illegal = torch.rand(n, generator=generator, device=device) < illegal_prob
        illegal_augmented = chosen & use_illegal
        # a legal move that already is a resign stays as it is
        legal_augmented = chosen & ~use_illegal & (legal_code != RESIGN_CODE)

    return Batch(
        board=board,
        legal=move_vectors(legal_code, dtype, resign=legal_augmented),
        illegal=move_vectors(illegal_code, dtype, resign=illegal_augmented),
        legal_label=torch.ones(n, dtype=dtype, device=device),
        illegal_label=illegal_augmented.to(dtype),
        legal_code=legal_code,
        illegal_code=illegal_code,
        legal_kind=legal_kind,
        illegal_reason=illegal_reason,
        legal_augmented=legal_augmented,
        illegal_augmented=illegal_augmented,
    )


def move_vectors(codes, dtype=torch.float32, resign=None):
    """Return the move vectors [N, 38] of move codes (int64 [N]), see MOVE_*.

    The resign code gets the resign vector. `resign` (bool [N]) additionally sets the
    resign flag of the marked rows and leaves their squares and promotion as they are.
    """
    is_resign = codes == RESIGN_CODE
    flag = is_resign if resign is None else is_resign | resign
    origin = (codes >> 6) & (NUM_SQUARES - 1)
    to = codes & (NUM_SQUARES - 1)
    ranks_and_files = torch.arange(BOARD_SIZE, device=codes.device)

    def one_hot(index):
        return ranks_and_files == index.unsqueeze(1)

    squares = torch.cat(
        (one_hot(to >> 3), one_hot(to & 7), one_hot(origin >> 3), one_hot(origin & 7)),
        dim=1,
    )
    squares &= ~is_resign.unsqueeze(1)
    promotion = functional.one_hot(codes >> 12, NUM_PROMOTIONS).to(torch.bool)
    return torch.cat((squares, promotion, flag.unsqueeze(1)), dim=1).to(dtype)
