# chess-rules

Chess rules dataset generator in C++

Generates a bucket-balanced dataset of (board, legal move, illegal move) for training a neural network on the rules of
chess, for standard chess and Chess960. Header-only library in `include/chess-rules`, command line tool in `src`,
[Catch2](https://github.com/catchorg/Catch2) tests in `tests`, and an independent python-chess audit in `tools`.

## Dependencies

```shell
sudo pacman -S cmake git
```

## Get the sources

[chess-library](https://github.com/Disservin/chess-library) is a git submodule in `external/chess-library`:

```shell
git submodule update --init external/chess-library
```

Configuring without it also works: CMake runs the `git submodule update` itself when `chess.hpp` is missing.

## Build and test

```shell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

CMake downloads Catch2 v3.5.4 unless it is installed. Use another copy of chess-library with
`-DCR_CHESS_LIBRARY_DIR=/path/to/dir/containing/chess.hpp` (note: the directory that contains `chess.hpp`, for
example `/path/to/chess-library/include`), skip the tests with `-DCR_BUILD_TESTS=OFF`, and build with ASan and
UBSan using `-DCR_SANITIZE=ON`.

## Generate

```shell
build/chess-rules --dir data --split train --samples-per-kind 100000
build/chess-rules --dir data --split eval  --samples-per-kind 10000
build/chess-rules --dir data --split test  --samples-per-kind 10000
```

## Audit

```shell
pip install -r tools/requirements.txt
python3 tools/audit.py data/train.bin --sample 100000 --leak data/eval.bin data/test.bin
```

## Training data in PyTorch

`py/chess_rules.py` is the input pipeline. Worker processes only copy batches of raw 80 byte records out of a
memory map; `unpack` turns them into tensors on the device they are on, normally the GPU.

```shell
pip install -r py/requirements.txt
cd py && python3 -m pytest -q
```

```python
import chess_rules as cr

loader = cr.make_loader("train.bin", batch_size=8192, num_workers=4)   # uint8 [B, 80] per batch
for epoch in range(epochs):                                            # one iteration = one epoch
    for raw in loader:
        batch = cr.unpack(raw.to(device, non_blocking=True), resign_prob=0.1)
        legal_logits = model(batch.board, batch.legal)                 # [B, 586], [B, 38]
        illegal_logits = model(batch.board, batch.illegal)
        loss = bce(legal_logits, torch.ones_like(legal_logits))        # legal: target 1
        loss += bce(illegal_logits, torch.zeros_like(illegal_logits))  # illegal: target 0
```

`batch` has the board once per record and the two moves separately, plus the bucket ids for metrics:

| field                          | shape      | content                                                                                                           |
| ------------------------------ | ---------- | ----------------------------------------------------------------------------------------------------------------- |
| `board`                        | `[B, 586]` | the 9 bitboards flattened (576, square a1 first), en passant one-hot (9: files a-h, then none), white to move (1) |
| `legal`, `illegal`             | `[B, 38]`  | to rank (8), to file (8), from rank (8), from file (8), promotion N / B / R / Q / none (5), resign (1)            |
| `legal_code`, `illegal_code`   | `[B]`      | the integer move codes in the record                                                                              |
| `legal_kind`, `illegal_reason` | `[B]`      | bucket ids, names in `cr.KINDS` and `cr.REASONS`                                                                  |
| `augmented`                    | `[B]`      | the augmentation replaced the legal slot of this row by a resign                                                  |
| `legal_from_illegal`           | `[B]`      | ... and it took the squares and the promotion of the illegal move                                                 |

The targets never change: `legal` is always a legal move (target 1) and `illegal` always an illegal one (target 0).

The slices `BOARD_*` and `MOVE_*` name the parts. Edit `move_vectors` if you want another convention.

- **Shuffling:** chess-rules writes shuffled files. Each epoch the batch boundaries move by a random offset and the batches
  are visited in a random order, so there is no random access to the file. The order is a pure function of
  `(seed, epoch)`: runs are reproducible, persistent workers follow the epochs, and `start_epoch=` resumes a run.
- **Resign augmentation:** `resign_prob` is the share of records that is augmented. In each of them the legal slot
  becomes a resign, either `legal move & resign` (probability `1 - illegal_prob`) or `illegal move & resign`
  (probability `illegal_prob`, default 0.5), where `& resign` sets the resign flag and keeps the squares and the
  promotion. The illegal slot and the targets (1, 0) never change, so the network learns that the rest of a resign
  move does not matter; with `illegal_move & resign` the two moves of a row differ only in the flag. A legal move
  that already is a resign is not changed by the first kind. `augmented` and `legal_from_illegal` mark the rows;
  leave `augmented` rows out of the accuracy per legal kind. The illegal side is never changed, so the accuracy per
  illegal reason can use every row. A natural resign (no legal move in the position) has no squares: its square
  groups are zero, its promotion is "none" and its resign flag is set.
- **Distributed:** `rank=` and `world_size=` give every process its own share of every epoch.
- **Per-bucket metrics:** accuracy per legal kind from the legal pass grouped by `legal_kind` (without the `augmented`
  rows), accuracy per illegal reason from the illegal pass grouped by `illegal_reason` (all rows).

## Record layout (80 bytes, little endian)

    np.dtype([("bb", "<u8", 9), ("legal", "<u2"), ("illegal", "<u2"),
              ("ep", "u1"), ("wtm", "u1"), ("kind", "u1"), ("reason", "u1")])

    bb[0..8]  white, black, pawns, knights, bishops, rooks, queens, kings, castling rooks   (bit i = square i, a1 = 0)
    move      to | from << 6 | promo << 12      promo: 0 none, 1 N, 2 B, 3 R, 4 Q
              castling = king takes own rook (Chess960 style); resign = 65 (b1b1)
    ep        file of the last double pawn push (255 = last move was not one)
    kind      0 pawn_single 1 pawn_double 2 pawn_attack 3 pawn_enpassant 4 pawn_promotion 5 knight 6 bishop
              7 rook 8 queen 9 king 10 castle_king 11 castle_queen 12 resign
    reason    0 same_square 1 empty_from 2 opponent_piece 3 own_capture 4 geometry 5 blocked 6 pawn_blocked
              7 pawn_rule 8 promotion_rule 9 castling_rule 10 ks_king_step 11 ks_pinned 12 ks_unresolved_check
              13 ks_castling 14 ks_ep

## Disclaimer

This is a complete rewrite, by Claude, of my previous code hosted at [Codeberg](https://codeberg.org/).
The old version was based on random play and checking which types of moves were present.
But e.g. en passant discovered checks do not arise very often. It was there fore not very efficient.
So after discussing with Claude I let it rewrite it.
Due to Codebergs strict no-LLM policy I migrated the data generation code to this repo.
