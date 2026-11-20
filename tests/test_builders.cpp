#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "chess-rules/generator.hpp"
#include "chess.hpp"

namespace {

cr::Config small_config(const std::uint64_t seed) {
    cr::Config config;
    config.kind_target       = 100;
    config.seed              = seed;
    config.calibration_iters = 1000;
    return config;
}

constexpr int ATTEMPTS             = 20000;
constexpr int MIN_BUILT            = 1000;
constexpr std::uint64_t BACK_RANKS = 0xFF000000000000FFull;

}  // namespace

TEST_CASE("every builder produces valid positions", "[builders]") {
    const auto source = GENERATE(cr::Shard::SPARSE, cr::Shard::CASTLE, cr::Shard::EP, cr::Shard::PROMO, cr::Shard::PIN,
                                 cr::Shard::TERMINAL);
    INFO("source: " << cr::Shard::source_name(source));

    cr::Shard shard(small_config(5));
    int built = 0;

    for (int i = 0; i < ATTEMPTS; ++i) {
        if (!shard.build(source)) {
            continue;
        }
        ++built;

        const chess::Board& board = shard.scratch_;
        const chess::Color stm    = board.sideToMove();

        REQUIRE(board.pieces(chess::PieceType::KING, chess::Color::WHITE).count() == 1);
        REQUIRE(board.pieces(chess::PieceType::KING, chess::Color::BLACK).count() == 1);
        REQUIRE_FALSE(
            chess::attacks::king(board.kingSq(chess::Color::WHITE)).check(board.kingSq(chess::Color::BLACK).index()));
        REQUIRE((board.pieces(chess::PieceType::PAWN).getBits() & BACK_RANKS) == 0);

        if (source != cr::Shard::TERMINAL) {
            // the side that is not to move must not be in check (the terminal builder has already played its move)
            REQUIRE_FALSE(board.isAttacked(board.kingSq(~stm), stm));
        }
    }

    REQUIRE(built > MIN_BUILT);
}

TEST_CASE("the terminal builder only produces checkmates and stalemates", "[builders]") {
    cr::Shard shard(small_config(6));
    int mates      = 0;
    int stalemates = 0;

    for (int i = 0; i < ATTEMPTS; ++i) {
        if (!shard.build(cr::Shard::TERMINAL)) {
            continue;
        }
        shard.an_.analyze_legal(shard.scratch_);
        REQUIRE(shard.an_.no_moves);
        if (shard.an_.in_check) {
            ++mates;
        } else {
            ++stalemates;
        }
    }

    REQUIRE(mates > 100);
    REQUIRE(stalemates > 100);
}

TEST_CASE("the en passant builder records the file of the double push", "[builders]") {
    cr::Shard shard(small_config(7));
    int built              = 0;
    int illegal_en_passant = 0;

    for (int i = 0; i < ATTEMPTS; ++i) {
        if (!shard.build(cr::Shard::EP)) {
            continue;
        }
        ++built;

        REQUIRE(shard.ep_file_ >= 0);
        REQUIRE(shard.ep_file_ < 8);

        // the pusher's pawn now stands on the fourth (white) or fifth (black) rank of that file
        const chess::Board& board = shard.scratch_;
        const int rank            = board.sideToMove() == chess::Color::WHITE ? 4 : 3;
        REQUIRE(board.at<chess::PieceType>(chess::Square(rank * 8 + shard.ep_file_)) == chess::PieceType::PAWN);

        shard.an_.analyze_legal(board);
        shard.an_.analyze_pseudo(board);
        if (!shard.an_.ks[cr::KS_EP - cr::KS_FIRST].empty()) {
            ++illegal_en_passant;
        }
    }

    REQUIRE(built > MIN_BUILT);
    REQUIRE(illegal_en_passant > 100);
}

TEST_CASE("the castle builder produces castling moves and castling mistakes", "[builders]") {
    cr::Shard shard(small_config(8));
    int legal_castles   = 0;
    int unsafe_castles  = 0;
    int rule_violations = 0;

    for (int i = 0; i < ATTEMPTS; ++i) {
        if (!shard.build(cr::Shard::CASTLE)) {
            continue;
        }
        shard.an_.analyze_legal(shard.scratch_);
        shard.an_.analyze_pseudo(shard.scratch_);

        legal_castles +=
            shard.an_.kind_available(cr::CASTLING_KINGSIDE) || shard.an_.kind_available(cr::CASTLING_QUEENSIDE);
        unsafe_castles += !shard.an_.ks[cr::KS_CASTLING - cr::KS_FIRST].empty();
        rule_violations += shard.propose(cr::CASTLING_RULE, shard.scratch_).has_value();
    }

    REQUIRE(legal_castles > 1000);
    REQUIRE(unsafe_castles > 100);
    REQUIRE(rule_violations > 1000);
}

TEST_CASE("the pin builder produces pinned pieces", "[builders]") {
    cr::Shard shard(small_config(9));
    int pins = 0;

    for (int i = 0; i < ATTEMPTS; ++i) {
        if (!shard.build(cr::Shard::PIN)) {
            continue;
        }
        shard.an_.analyze_legal(shard.scratch_);
        shard.an_.analyze_pseudo(shard.scratch_);
        pins += !shard.an_.ks[cr::KS_PINNED - cr::KS_FIRST].empty();
    }

    REQUIRE(pins > 1000);
}

TEST_CASE("the promotion builder produces promotions", "[builders]") {
    cr::Shard shard(small_config(10));
    int promotions = 0;

    for (int i = 0; i < ATTEMPTS; ++i) {
        if (!shard.build(cr::Shard::PROMO)) {
            continue;
        }
        shard.an_.analyze_legal(shard.scratch_);
        promotions += shard.an_.kind_available(cr::PAWN_PROMOTION);
    }

    REQUIRE(promotions > 1000);
}
