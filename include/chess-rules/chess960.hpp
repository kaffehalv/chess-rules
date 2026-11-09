#pragma once
#include <array>
#include <cstdint>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

// Scharnagl numbering (0..959); 518 is the standard start position.
inline std::string chess960_fen(const int idx) {
    std::array<char, 8> rank;
    rank.fill(' ');
    int n = idx;

    rank[2 * (n % 4) + 1] = 'B';  // light-squared bishop on b, d, f or h
    n /= 4;
    rank[2 * (n % 4)] = 'B';  // dark-squared bishop on a, c, e or g
    n /= 4;

    const auto place = [&rank](int k, const char piece) {  // k-th still-empty square
        for (char& c : rank) {
            if (c == ' ' && k-- == 0) {
                c = piece;
                return;
            }
        }
    };

    place(n % 6, 'Q');
    n /= 6;

    static constexpr int KNIGHTS[10][2] = {{0, 1}, {0, 2}, {0, 3}, {0, 4}, {1, 2},
                                           {1, 3}, {1, 4}, {2, 3}, {2, 4}, {3, 4}};
    place(KNIGHTS[n][1], 'N');  // higher index first so the lower one doesn't shift
    place(KNIGHTS[n][0], 'N');

    const char rest[3] = {'R', 'K', 'R'};
    int j              = 0;
    for (char& c : rank) {
        if (c == ' ') {
            c = rest[j++];
        }
    }

    std::string white(rank.begin(), rank.end());
    std::string black = white;
    for (char& c : black) {
        c = static_cast<char>(c - 'A' + 'a');
    }

    return black + "/pppppppp/8/8/8/8/PPPPPPPP/" + white + " w KQkq - 0 1";
}

// Deterministic train / val / test partition of the 960 start positions (768 / 96 / 96).
// Own Fisher-Yates over raw mt19937 output, since std::shuffle and the standard distributions
// are not guaranteed to give the same result across standard libraries.
// NOTE: do not split by idx % k: the Scharnagl index encodes the bishop squares in its low digits.
inline std::array<std::vector<int>, 3> split_chess960_starts(const std::uint32_t seed = 0) {
    std::vector<int> idx(960);
    std::iota(idx.begin(), idx.end(), 0);

    std::mt19937 rng(seed);
    for (int i = 959; i > 0; --i) {
        std::swap(idx[i], idx[rng() % (i + 1)]);
    }

    return {{std::vector<int>(idx.begin(), idx.begin() + 768), std::vector<int>(idx.begin() + 768, idx.begin() + 864),
             std::vector<int>(idx.begin() + 864, idx.end())}};
}
