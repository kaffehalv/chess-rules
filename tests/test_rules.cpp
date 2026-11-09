#include <catch2/catch_test_macros.hpp>
#include <set>
#include <string>

#include "chess-rules/chess960.hpp"
#include "chess-rules/rules.hpp"
#include "chess.hpp"

namespace {

int sq(const char* name) {
    return chess::Square(name).index();
}

// A position together with its analysis, so tests can ask "why is this move illegal?" directly.
struct Position {
    explicit Position(const std::string& fen) : board(fen, true) {
        analysis.analyze_legal(board);
        analysis.analyze_pseudo(board);
    }

    nw::Reason classify(const char* from, const char* to, int promo = 0) const {
        return nw::classify(board, analysis, nw::make_code(sq(from), sq(to), promo));
    }

    bool is_legal(const char* from, const char* to, int promo = 0) const {
        return analysis.legal_set.contains(nw::make_code(sq(from), sq(to), promo));
    }

    chess::Board board;
    nw::Analysis analysis;
};

constexpr int KNIGHT_PROMO = 1;
constexpr int ROOK_PROMO   = 3;
constexpr int QUEEN_PROMO  = 4;

}  // namespace

TEST_CASE("move codes", "[rules]") {
    REQUIRE(sizeof(nw::Example) == 80);
    REQUIRE(nw::make_code(1, 1) == nw::RESIGN_CODE);

    const nw::Code code = nw::make_code(sq("e7"), sq("e8"), QUEEN_PROMO);
    REQUIRE(nw::code_from(code) == sq("e7"));
    REQUIRE(nw::code_to(code) == sq("e8"));
    REQUIRE(nw::code_promo(code) == QUEEN_PROMO);
}

TEST_CASE("library moves are encoded the way the dataset describes", "[rules]") {
    const Position position("4k3/P7/8/8/8/8/8/R3K2R w KQ - 0 1");

    SECTION("castling is king takes own rook") {
        REQUIRE(position.is_legal("e1", "h1"));
        REQUIRE(position.is_legal("e1", "a1"));
    }

    SECTION("a promotion carries the piece, a plain a7a8 is not a move") {
        REQUIRE(position.is_legal("a7", "a8", QUEEN_PROMO));
        REQUIRE(position.is_legal("a7", "a8", KNIGHT_PROMO));
        REQUIRE_FALSE(position.is_legal("a7", "a8"));
    }
}

TEST_CASE("the random number generator is reproducible and in range", "[rules]") {
    nw::Rng first(42);
    nw::Rng second(42);
    nw::Rng other(43);

    bool differs = false;
    for (int i = 0; i < 100; ++i) {
        const std::uint64_t value = first.next();
        REQUIRE(value == second.next());
        differs = differs || value != other.next();
    }
    REQUIRE(differs);

    nw::Rng rng(1);
    for (int i = 0; i < 10000; ++i) {
        REQUIRE(rng.below(7) < 7);
        const double u = rng.uniform();
        REQUIRE(u >= 0.0);
        REQUIRE(u < 1.0);
        const int r = rng.range(-3, 3);
        REQUIRE(r >= -3);
        REQUIRE(r <= 3);
    }

    const std::uint64_t bits = 0b1010'0100'0000;
    for (int i = 0; i < 100; ++i) {
        REQUIRE(((bits >> nw::pick_bit(bits, rng)) & 1) == 1);
    }
}

TEST_CASE("CodeSet inserts, finds and clears in constant time", "[rules]") {
    nw::CodeSet set;
    REQUIRE_FALSE(set.contains(123));

    set.insert(123);
    set.insert(123);
    set.insert(32767);
    REQUIRE(set.contains(123));
    REQUIRE(set.contains(32767));
    REQUIRE(set.size() == 2);

    set.clear();
    REQUIRE_FALSE(set.contains(123));
    REQUIRE_FALSE(set.contains(32767));
    REQUIRE(set.size() == 0);
}

TEST_CASE("every illegal reason is recognised", "[rules]") {
    SECTION("same square") {
        const Position position(chess::constants::STARTPOS);
        REQUIRE(position.classify("e2", "e2") == nw::SAME_SQUARE);
    }

    SECTION("empty from square") {
        const Position position(chess::constants::STARTPOS);
        REQUIRE(position.classify("e4", "e5") == nw::EMPTY_FROM);
    }

    SECTION("opponent piece") {
        const Position position(chess::constants::STARTPOS);
        REQUIRE(position.classify("e7", "e5") == nw::OPPONENT_PIECE);
    }

    SECTION("own capture") {
        const Position position(chess::constants::STARTPOS);
        REQUIRE(position.classify("a1", "a2") == nw::OWN_CAPTURE);
        REQUIRE(position.classify("e2", "d3") == nw::PAWN_RULE);  // not an own capture: nothing stands on d3
    }

    SECTION("geometry") {
        const Position position(chess::constants::STARTPOS);
        REQUIRE(position.classify("b1", "b4") == nw::GEOMETRY);
        REQUIRE(position.classify("e1", "e3") == nw::GEOMETRY);
    }

    SECTION("blocked slider") {
        const Position position(chess::constants::STARTPOS);
        REQUIRE(position.classify("a1", "a5") == nw::BLOCKED);
        REQUIRE(position.classify("c1", "h6") == nw::BLOCKED);
    }

    SECTION("blocked pawn") {
        const Position position("4k3/8/8/8/8/4p3/4P3/4K3 w - - 0 1");
        REQUIRE(position.classify("e2", "e3") == nw::PAWN_BLOCKED);
        REQUIRE(position.classify("e2", "e4") == nw::PAWN_BLOCKED);
    }

    SECTION("other pawn mistakes") {
        const Position position(chess::constants::STARTPOS);
        REQUIRE(position.classify("e2", "e1") == nw::PAWN_RULE);
        REQUIRE(position.classify("e2", "f2") == nw::PAWN_RULE);
        REQUIRE(position.classify("e2", "e5") == nw::PAWN_RULE);
    }

    SECTION("promotion rule") {
        const Position position("4k3/P7/8/8/8/8/8/4K3 w - - 0 1");
        REQUIRE(position.classify("a7", "a8") == nw::PROMOTION_RULE);
        REQUIRE(position.classify("a7", "a6", ROOK_PROMO) == nw::PROMOTION_RULE);
        REQUIRE(position.classify("e1", "f1", QUEEN_PROMO) == nw::PROMOTION_RULE);
        REQUIRE(position.is_legal("a7", "a8", QUEEN_PROMO));
    }

    SECTION("castling rule") {
        const Position no_rights("r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1");
        REQUIRE(no_rights.classify("e1", "h1") == nw::CASTLING_RULE);

        const Position blocked("r3k2r/8/8/8/8/8/8/RN2K2R w KQkq - 0 1");
        REQUIRE(blocked.classify("e1", "a1") == nw::CASTLING_RULE);
        REQUIRE(blocked.is_legal("e1", "h1"));
    }

    SECTION("king steps into check") {
        const Position position("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1");
        REQUIRE(position.classify("e1", "d2") == nw::KS_KING_STEP);
        REQUIRE(position.is_legal("e1", "e2"));
    }

    SECTION("pinned piece") {
        const Position position("4k3/8/8/8/4r3/8/4R3/4K3 w - - 0 1");
        REQUIRE(position.classify("e2", "a2") == nw::KS_PINNED);
        REQUIRE(position.is_legal("e2", "e3"));
    }

    SECTION("check is not resolved") {
        const Position position("4k3/8/8/8/8/8/P3r3/4K3 w - - 0 1");
        REQUIRE(position.analysis.in_check);
        REQUIRE(position.classify("a2", "a3") == nw::KS_UNRESOLVED_CHECK);
    }

    SECTION("castling through an attacked square") {
        const Position position("r3k2r/8/8/8/8/8/5r2/R3K2R w KQkq - 0 1");
        REQUIRE(position.is_legal("e1", "a1"));
        REQUIRE(position.classify("e1", "h1") == nw::KS_CASTLING);
    }

    SECTION("en passant that exposes the king along the rank") {
        // The library only keeps the en passant square after the double push has been played.
        chess::Board board("7k/4p3/8/2KP3r/8/8/8/8 b - - 0 1", true);
        chess::Movelist moves;
        chess::movegen::legalmoves(moves, board);
        for (const chess::Move move : moves) {
            if (nw::encode(move) == nw::make_code(sq("e7"), sq("e5"))) {
                board.makeMove(move);
                break;
            }
        }
        nw::Analysis analysis;
        analysis.analyze_legal(board);
        analysis.analyze_pseudo(board);

        const nw::Code ep = nw::make_code(sq("d5"), sq("e6"));
        REQUIRE_FALSE(analysis.legal_set.contains(ep));
        REQUIRE(analysis.pseudo_set.contains(ep));
        REQUIRE(nw::classify(board, analysis, ep) == nw::KS_EP);
    }
}

TEST_CASE("legal moves and the resign code are never illegal", "[rules]") {
    const Position position(chess::constants::STARTPOS);
    REQUIRE(position.classify("e2", "e4") == nw::R_LEGAL);
    REQUIRE(position.classify("b1", "c3") == nw::R_LEGAL);
    REQUIRE(nw::classify(position.board, position.analysis, nw::RESIGN_CODE) == nw::R_LEGAL);
}

TEST_CASE("legal move kinds", "[rules]") {
    const auto kind_of_move = [](const std::string& fen, const std::string& uci) {
        chess::Board board(fen, true);
        return nw::kind_of(board, chess::uci::uciToMove(board, uci));
    };

    REQUIRE(kind_of_move(chess::constants::STARTPOS, "e2e3") == nw::PAWN_SINGLE);
    REQUIRE(kind_of_move(chess::constants::STARTPOS, "e2e4") == nw::PAWN_DOUBLE);
    REQUIRE(kind_of_move(chess::constants::STARTPOS, "b1c3") == nw::KNIGHT);
    REQUIRE(kind_of_move("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5") == nw::PAWN_ATTACK);
    REQUIRE(kind_of_move("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8q") == nw::PAWN_PROMOTION);
    REQUIRE(kind_of_move("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6") == nw::PAWN_ENPASSANT);
    REQUIRE(kind_of_move("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1h1") == nw::CASTLING_KINGSIDE);
    REQUIRE(kind_of_move("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1a1") == nw::CASTLING_QUEENSIDE);
    REQUIRE(kind_of_move("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "a1a5") == nw::ROOK);
    REQUIRE(kind_of_move("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1d1") == nw::KING);
}

TEST_CASE("the classifier agrees with the library on every possible move code", "[rules][slow]") {
    nw::Rng rng(7);
    nw::Analysis analysis;
    long positions  = 0;
    long violations = 0;

    for (int game = 0; game < 40; ++game) {
        chess::Board board(chess960_fen(static_cast<int>(rng.below(960))), true);
        for (int ply = 0; ply < 80; ++ply) {
            analysis.analyze_legal(board);
            analysis.analyze_pseudo(board);
            ++positions;

            for (int from = 0; from < 64; ++from) {
                for (int to = 0; to < 64; ++to) {
                    for (int promo = 0; promo <= 4; ++promo) {
                        const nw::Code code     = nw::make_code(from, to, promo);
                        const nw::Reason reason = nw::classify(board, analysis, code);
                        const bool legal        = analysis.legal_set.contains(code) || code == nw::RESIGN_CODE;
                        const bool pseudo_illegal =
                            analysis.pseudo_set.contains(code) && !analysis.legal_set.contains(code);

                        if ((reason == nw::R_LEGAL) != legal) {
                            ++violations;
                        }
                        // king safety reasons are exactly the moves that follow the movement rules but are illegal
                        if (reason != nw::R_LEGAL && (reason >= nw::KS_FIRST) != pseudo_illegal) {
                            ++violations;
                        }
                    }
                }
            }

            if (analysis.no_moves) {
                break;
            }
            board.makeMove(analysis.legal[rng.below(static_cast<std::uint32_t>(analysis.legal.size()))]);
        }
    }

    REQUIRE(positions > 1000);
    REQUIRE(violations == 0);
}

TEST_CASE("every legal move is also pseudo-legal", "[rules][slow]") {
    nw::Rng rng(11);
    nw::Analysis analysis;
    long missing = 0;

    for (int game = 0; game < 200; ++game) {
        chess::Board board(chess960_fen(static_cast<int>(rng.below(960))), true);
        for (int ply = 0; ply < 80; ++ply) {
            analysis.analyze_legal(board);
            analysis.analyze_pseudo(board);
            for (const chess::Move move : analysis.legal) {
                if (!analysis.pseudo_set.contains(nw::encode(move))) {
                    ++missing;
                }
            }
            if (analysis.no_moves) {
                break;
            }
            board.makeMove(analysis.legal[rng.below(static_cast<std::uint32_t>(analysis.legal.size()))]);
        }
    }

    REQUIRE(missing == 0);
}

TEST_CASE("a position without any pseudo-legal move is a stalemate that still gets analysed", "[rules]") {
    const Position position("7k/8/8/8/1r6/rPq5/BRP5/KB6 w - - 0 1");

    REQUIRE(position.analysis.no_moves);
    REQUIRE(position.analysis.pseudo.size() == 0);
    REQUIRE_FALSE(position.analysis.in_check);
    REQUIRE(position.analysis.kind_available(nw::RESIGN));
    REQUIRE_FALSE(position.analysis.kind_available(nw::KING));
}

TEST_CASE("positions the library mishandles are flagged as untrusted", "[rules]") {
    SECTION("two diagonal checkers") {
        for (const char* fen :
             {"8/3B1P2/2k2q2/2qB4/3P4/R3KB2/8/2n3R1 b - - 0 1", "8/1k6/2bbQ3/2K2R2/8/Q7/5q2/1NnN4 w - - 0 1",
              "2q5/1PP1k3/6q1/4NK2/N7/8/8/8 w - - 0 1"}) {
            const Position position(fen);
            INFO(fen);
            REQUIRE_FALSE(position.analysis.trusted);
        }
    }

    SECTION("a single check and a position without check are fine") {
        REQUIRE(Position("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1").analysis.trusted);
        REQUIRE(Position(chess::constants::STARTPOS).analysis.trusted);
    }
}
