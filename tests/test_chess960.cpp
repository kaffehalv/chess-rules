#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <set>
#include <string>

#include "chess-rules/chess960.hpp"
#include "chess.hpp"

namespace {

std::string white_back_rank(const std::string& fen) {
    const std::size_t slash = fen.rfind('/');
    return fen.substr(slash + 1, 8);
}

}  // namespace

TEST_CASE("start position 518 is the standard chess position", "[chess960]") {
    REQUIRE(chess960_fen(518) == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}

TEST_CASE("the 960 start positions are distinct and follow the Chess960 rules", "[chess960]") {
    std::set<std::string> back_ranks;

    for (int i = 0; i < 960; ++i) {
        const std::string rank = white_back_rank(chess960_fen(i));
        back_ranks.insert(rank);

        int bishops_on_light  = 0;
        int bishops_on_dark   = 0;
        int rooks_before_king = 0;
        int rooks_after_king  = 0;
        int king_file         = -1;
        for (int file = 0; file < 8; ++file) {
            if (rank[file] == 'K') {
                king_file = file;
            }
        }
        for (int file = 0; file < 8; ++file) {
            if (rank[file] == 'B') {
                // a1 is dark, so file b (index 1) on rank 1 is light
                if (file % 2 == 1) {
                    ++bishops_on_light;
                } else {
                    ++bishops_on_dark;
                }
            }
            if (rank[file] == 'R') {
                if (file < king_file) {
                    ++rooks_before_king;
                } else {
                    ++rooks_after_king;
                }
            }
        }
        INFO("start position " << i << ": " << rank);
        REQUIRE(bishops_on_light == 1);
        REQUIRE(bishops_on_dark == 1);
        REQUIRE(rooks_before_king == 1);
        REQUIRE(rooks_after_king == 1);
        REQUIRE(std::count(rank.begin(), rank.end(), 'Q') == 1);
        REQUIRE(std::count(rank.begin(), rank.end(), 'N') == 2);
    }

    REQUIRE(back_ranks.size() == 960);
}

TEST_CASE("the library accepts every start position with all four castling rights", "[chess960]") {
    for (int i = 0; i < 960; ++i) {
        const chess::Board board(chess960_fen(i), true);
        const auto rights    = board.castlingRights();
        const auto king_file = static_cast<int>(board.kingSq(chess::Color::WHITE).file());

        INFO("start position " << i);
        for (const chess::Color color : {chess::Color::WHITE, chess::Color::BLACK}) {
            REQUIRE(rights.has(color, chess::Board::CastlingRights::Side::KING_SIDE));
            REQUIRE(rights.has(color, chess::Board::CastlingRights::Side::QUEEN_SIDE));
            REQUIRE(static_cast<int>(rights.getRookFile(color, chess::Board::CastlingRights::Side::KING_SIDE)) >
                    king_file);
            REQUIRE(static_cast<int>(rights.getRookFile(color, chess::Board::CastlingRights::Side::QUEEN_SIDE)) <
                    king_file);
        }
    }
}

TEST_CASE("the start positions are split into disjoint train, validation and test sets", "[chess960]") {
    const auto split = split_chess960_starts(0);

    REQUIRE(split[0].size() == 768);
    REQUIRE(split[1].size() == 96);
    REQUIRE(split[2].size() == 96);

    std::set<int> all;
    for (const auto& part : split) {
        all.insert(part.begin(), part.end());
    }
    REQUIRE(all.size() == 960);
    REQUIRE(*all.begin() == 0);
    REQUIRE(*all.rbegin() == 959);

    SECTION("the split is deterministic") {
        REQUIRE(split_chess960_starts(0) == split);
    }

    SECTION("a different seed gives a different split") {
        REQUIRE(split_chess960_starts(1) != split);
    }
}
