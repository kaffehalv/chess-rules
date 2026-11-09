#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "chess-rules/generator.hpp"
#include "chess.hpp"

namespace {

nw::Config small_config(const std::uint64_t seed) {
    nw::Config config;
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
    const auto source = GENERATE(nw::Shard::SPARSE, nw::Shard::CASTLE, nw::Shard::EP, nw::Shard::PROMO, nw::Shard::PIN,
                                 nw::Shard::TERMINAL);
    INFO("source: " << nw::Shard::source_name(source));

    nw::Shard shard(small_config(5));
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

        if (source != nw::Shard::TERMINAL) {
            // the side that is not to move must not be in check (the terminal builder has already played its move)
            REQUIRE_FALSE(board.isAttacked(board.kingSq(~stm), stm));
        }
    }

    REQUIRE(built > MIN_BUILT);
}

TEST_CASE("the terminal builder only produces checkmates and stalemates", "[builders]") {
    nw::Shard shard(small_config(6));
    int mates      = 0;
    int stalemates = 0;

    for (int i = 0; i < ATTEMPTS; ++i) {
        if (!shard.build(nw::Shard::TERMINAL)) {
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
    nw::Shard shard(small_config(7));
    int built              = 0;
    int illegal_en_passant = 0;

    for (int i = 0; i < ATTEMPTS; ++i) {
        if (!shard.build(nw::Shard::EP)) {
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
        if (!shard.an_.ks[nw::KS_EP - nw::KS_FIRST].empty()) {
            ++illegal_en_passant;
        }
    }

    REQUIRE(built > MIN_BUILT);
    REQUIRE(illegal_en_passant > 100);
}

TEST_CASE("the castle builder produces castling moves and castling mistakes", "[builders]") {
    nw::Shard shard(small_config(8));
    int legal_castles   = 0;
    int unsafe_castles  = 0;
    int rule_violations = 0;

    for (int i = 0; i < ATTEMPTS; ++i) {
        if (!shard.build(nw::Shard::CASTLE)) {
            continue;
        }
        shard.an_.analyze_legal(shard.scratch_);
        shard.an_.analyze_pseudo(shard.scratch_);

        legal_castles +=
            shard.an_.kind_available(nw::CASTLING_KINGSIDE) || shard.an_.kind_available(nw::CASTLING_QUEENSIDE);
        unsafe_castles += !shard.an_.ks[nw::KS_CASTLING - nw::KS_FIRST].empty();
        rule_violations += shard.propose(nw::CASTLING_RULE, shard.scratch_).has_value();
    }

    REQUIRE(legal_castles > 1000);
    REQUIRE(unsafe_castles > 100);
    REQUIRE(rule_violations > 1000);
}

TEST_CASE("the pin builder produces pinned pieces", "[builders]") {
    nw::Shard shard(small_config(9));
    int pins = 0;

    for (int i = 0; i < ATTEMPTS; ++i) {
        if (!shard.build(nw::Shard::PIN)) {
            continue;
        }
        shard.an_.analyze_legal(shard.scratch_);
        shard.an_.analyze_pseudo(shard.scratch_);
        pins += !shard.an_.ks[nw::KS_PINNED - nw::KS_FIRST].empty();
    }

    REQUIRE(pins > 1000);
}

TEST_CASE("the promotion builder produces promotions", "[builders]") {
    nw::Shard shard(small_config(10));
    int promotions = 0;

    for (int i = 0; i < ATTEMPTS; ++i) {
        if (!shard.build(nw::Shard::PROMO)) {
            continue;
        }
        shard.an_.analyze_legal(shard.scratch_);
        promotions += shard.an_.kind_available(nw::PAWN_PROMOTION);
    }

    REQUIRE(promotions > 1000);
}
