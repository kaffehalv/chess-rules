#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include <set>
#include <tuple>
#include <vector>

#include "chess-rules/generator.hpp"

namespace {

using BoardKey = std::tuple<std::array<std::uint64_t, 9>, int, int>;

nw::Config small_config(const std::uint64_t kind_target, const std::uint64_t seed, const int split = 0) {
    nw::Config config;
    config.kind_target       = kind_target;
    config.seed              = seed;
    config.split             = split;
    config.calibration_iters = 3000;
    return config;
}

BoardKey board_key(const nw::Example& example) {
    std::array<std::uint64_t, 9> bitboards{};
    for (int i = 0; i < 9; ++i) {
        bitboards[i] = example.bitboards[i];
    }
    return {bitboards, example.white_to_move, example.ep_file};
}

std::vector<nw::Example> generate(const nw::Config& config) {
    nw::Shard shard(config);
    std::vector<nw::Example> examples;
    shard.run(examples);
    return examples;
}

}  // namespace

TEST_CASE("every legal kind gets exactly the requested number of examples", "[generator]") {
    constexpr std::uint64_t N = 400;
    nw::Shard shard(small_config(N, 9));
    std::vector<nw::Example> examples;
    shard.run(examples);

    REQUIRE(examples.size() == N * nw::NUM_KINDS);

    std::array<std::uint64_t, nw::NUM_KINDS> kinds{};
    for (const nw::Example& example : examples) {
        ++kinds[example.legal_kind];
    }
    for (int kind = 0; kind < nw::NUM_KINDS; ++kind) {
        INFO(nw::kind_name(kind));
        REQUIRE(kinds[kind] == N);
        REQUIRE(shard.stats().kind[kind] == N);
    }
}

TEST_CASE("the illegal reasons are balanced too", "[generator]") {
    constexpr std::uint64_t N = 400;
    nw::Shard shard(small_config(N, 9));
    std::vector<nw::Example> examples;
    shard.run(examples);

    const double target = static_cast<double>(N) * nw::NUM_KINDS / nw::NUM_REASONS;
    for (int reason = 0; reason < nw::NUM_REASONS; ++reason) {
        INFO(nw::reason_name(reason));
        REQUIRE(shard.stats().reason[reason] >= target * 0.9);
        REQUIRE(shard.stats().reason[reason] <= target * 1.1);
    }
}

TEST_CASE("the quota matrix has the requested row sums and no cell for an impossible pairing", "[generator]") {
    nw::Shard shard(small_config(1000, 3));

    for (int kind = 0; kind < nw::NUM_KINDS; ++kind) {
        std::uint64_t row = 0;
        for (int reason = 0; reason < nw::NUM_REASONS; ++reason) {
            row += shard.quota_[kind][reason];
        }
        INFO(nw::kind_name(kind));
        REQUIRE(row == 1000);
    }

    // a position without legal moves cannot also contain a legal castling move or a legal en passant capture
    REQUIRE(shard.quota_[nw::RESIGN][nw::KS_EP] == 0);
    REQUIRE(shard.quota_[nw::CASTLING_KINGSIDE][nw::KS_EP] == 0);
}

TEST_CASE("every record is internally consistent", "[generator]") {
    const std::vector<nw::Example> examples = generate(small_config(300, 21));

    for (const nw::Example& example : examples) {
        REQUIRE(example.legal_move != example.illegal_move);
        REQUIRE((example.legal_move == nw::RESIGN_CODE) == (example.legal_kind == nw::RESIGN));
        REQUIRE(example.illegal_move != nw::RESIGN_CODE);
        REQUIRE(example.legal_kind < nw::NUM_KINDS);
        REQUIRE(example.illegal_reason < nw::NUM_REASONS);
        REQUIRE(example.white_to_move <= 1);
        REQUIRE((example.ep_file < 8 || example.ep_file == 255));
        REQUIRE(nw::code_promo(example.legal_move) <= 4);
        REQUIRE(nw::code_promo(example.illegal_move) <= 4);

        const std::uint64_t white = example.bitboards[0];
        const std::uint64_t black = example.bitboards[1];
        const std::uint64_t kings = example.bitboards[7];
        REQUIRE((white & black) == 0);
        REQUIRE(__builtin_popcountll(kings & white) == 1);
        REQUIRE(__builtin_popcountll(kings & black) == 1);
        REQUIRE((example.bitboards[8] & ~example.bitboards[5]) == 0);  // castling rooks are rooks
    }
}

TEST_CASE("the same seed gives byte-identical output", "[generator]") {
    const std::vector<nw::Example> first  = generate(small_config(150, 77));
    const std::vector<nw::Example> second = generate(small_config(150, 77));
    const std::vector<nw::Example> other  = generate(small_config(150, 78));

    REQUIRE(first.size() == second.size());
    REQUIRE(std::memcmp(first.data(), second.data(), first.size() * sizeof(nw::Example)) == 0);
    REQUIRE(std::memcmp(first.data(), other.data(), first.size() * sizeof(nw::Example)) != 0);
}

TEST_CASE("the splits never share a position", "[generator]") {
    const std::vector<nw::Example> train = generate(small_config(150, 5, 0));
    const std::vector<nw::Example> val   = generate(small_config(150, 6, 1));
    const std::vector<nw::Example> test  = generate(small_config(150, 7, 2));

    std::set<BoardKey> train_boards;
    for (const nw::Example& example : train) {
        train_boards.insert(board_key(example));
    }

    int shared = 0;
    for (const std::vector<nw::Example>* other : {&val, &test}) {
        for (const nw::Example& example : *other) {
            shared += static_cast<int>(train_boards.count(board_key(example)));
        }
    }
    REQUIRE(shared == 0);
}

TEST_CASE("the split of a position depends only on its hash", "[generator]") {
    int counts[3] = {0, 0, 0};
    for (std::uint64_t hash = 0; hash < 100000; ++hash) {
        const int split = nw::split_of(hash * 0x9E3779B97F4A7C15ull);
        REQUIRE(split >= 0);
        REQUIRE(split <= 2);
        ++counts[split];
    }

    REQUIRE(counts[0] > 78000);
    REQUIRE(counts[0] < 82000);
    REQUIRE(counts[1] > 8000);
    REQUIRE(counts[2] > 8000);
}

TEST_CASE("a position without any pseudo-legal move still yields a resign example", "[generator]") {
    // stalemate in which every white piece is blocked
    const std::string fen = "7k/8/8/8/1r6/rPq5/BRP5/KB6 w - - 0 1";
    chess::Board board(fen, true);

    nw::Config config = small_config(50, 1, nw::split_of(board.hash()));
    nw::Shard shard(config);
    shard.an_.analyze_legal(board);
    std::vector<nw::Example> examples;

    bool emitted = false;
    for (int attempt = 0; attempt < 100 && !emitted; ++attempt) {
        emitted = shard.consider(board, -1, examples);
    }

    REQUIRE(emitted);
    REQUIRE(examples.size() == 1);
    REQUIRE(examples[0].legal_kind == nw::RESIGN);
    REQUIRE(examples[0].legal_move == nw::RESIGN_CODE);
}
