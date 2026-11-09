# chess-rules

Chess rules dataset generator in C++

Generates a bucket-balanced dataset of (board, legal move, illegal move) for training a neural network on the rules of
chess, for standard chess and Chess960. Header-only library in `include/chess-rules`, command line tool in `src`,
[Catch2](https://github.com/catchorg/Catch2) tests in `tests`, and an independent python-chess `audit.py`.

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
`-DCRULES_CHESS_LIBRARY_DIR=/path/to/dir/containing/chess.hpp` (note: the directory that contains `chess.hpp`, for
example `/path/to/chess-library/include`), skip the tests with `-DCRULES_BUILD_TESTS=OFF`, and build with ASan and
UBSan using `-DCRULES_SANITIZE=ON`.

## Generate

```shell
build/chess-rules --out train.bin --samples-per-kind 1000000 --split train --variant mixed --threads 8 --seed 1
build/chess-rules --out val.bin   --samples-per-kind 100000  --split val   --variant mixed --threads 8 --seed 2
build/chess-rules --out test.bin  --samples-per-kind 100000  --split test  --variant mixed --threads 8 --seed 3
```

## Audit

Install Python, create a virtual environment, activate it, install dependencies

```shell
sudo pacman -S python
python -m venv .venv
source .venv/bin/activate
pip install numpy chess
```

and run e.g.

```shell
python audit.py train.bin --sample 200000 --leak val.bin test.bin
```

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
