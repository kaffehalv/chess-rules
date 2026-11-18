#include <catch2/catch_test_macros.hpp>
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
        return nw::classify(board, analysis, nw::Code(sq(from), sq(to), promo));
    }

    bool is_legal(const char* from, const char* to, int promo = 0) const {
        return analysis.legal_set.contains(nw::Code(sq(from), sq(to), promo));
    }

    chess::Board board;
    nw::Analysis analysis;
};

constexpr int KNIGHT_PROMO = 1;
constexpr int ROOK_PROMO   = 3;
constexpr int QUEEN_PROMO  = 4;

}  // namespace

TEST_CASE("a Code keeps from, to and promotion apart", "[rules][code]") {
    const nw::Code code(sq("e7"), sq("e8"), nw::QUEEN_PROMOTION);
    REQUIRE(code.from() == sq("e7"));
    REQUIRE(code.to() == sq("e8"));
    REQUIRE(code.promo() == nw::QUEEN_PROMOTION);

    SECTION("a promotion can be added and removed") {
        REQUIRE(code.without_promo() == nw::Code(sq("e7"), sq("e8")));
        REQUIRE(code.without_promo().with_promo(nw::QUEEN_PROMOTION) == code);
        REQUIRE(code.with_promo(nw::KNIGHT_PROMOTION) != code);
    }

    SECTION("the raw value has the documented layout") {
        REQUIRE(code.raw() == (sq("e8") | sq("e7") << 6 | nw::QUEEN_PROMOTION << 12));
    }

    SECTION("resign is b1b1") {
        static_assert(nw::Code::resign() == nw::Code(1, 1));
        REQUIRE(nw::Code::resign() == nw::Code(sq("b1"), sq("b1")));
        REQUIRE(nw::Code::resign().raw() == 65);
    }

    SECTION("the raw values are a dense index and the codes are ordered") {
        REQUIRE(nw::Code(0, 0).raw() == 0);
        REQUIRE(nw::Code(63, 63, nw::QUEEN_PROMOTION).raw() == nw::Code::COUNT - 1);
        REQUIRE(nw::Code(0, 1) < nw::Code(0, 2));
    }
}

TEST_CASE("only valid values become codes", "[rules][code]") {
    REQUIRE(nw::Code::from_raw(0).has_value());
    REQUIRE(nw::Code::from_raw(nw::Code::COUNT - 1).has_value());
    REQUIRE(nw::Code::from_raw(65).value() == nw::Code::resign());

    REQUIRE_FALSE(nw::Code::from_raw(nw::Code::COUNT).has_value());
    REQUIRE_FALSE(nw::Code::from_raw(5 << 12).has_value());  // a promotion piece that does not exist
    REQUIRE_FALSE(nw::Code::from_raw(0xFFFF).has_value());
}

TEST_CASE("a Code does not depend on the type of the library move", "[rules][code]") {
    const auto e5 = chess::Square("e5");
    const auto d6 = chess::Square("d6");
    const auto e1 = chess::Square("e1");
    const auto h1 = chess::Square("h1");

    SECTION("en passant and the plain move between the same squares") {
        const auto plain      = chess::Move::make<chess::Move::NORMAL>(e5, d6);
        const auto en_passant = chess::Move::make<chess::Move::ENPASSANT>(e5, d6);
        REQUIRE(plain != en_passant);  // what made illegal moves collide with legal ones
        REQUIRE(nw::Code::of(plain) == nw::Code::of(en_passant));
    }

    SECTION("castling and king takes own rook") {
        const auto plain    = chess::Move::make<chess::Move::NORMAL>(e1, h1);
        const auto castling = chess::Move::make<chess::Move::CASTLING>(e1, h1);
        REQUIRE(plain != castling);
        REQUIRE(nw::Code::of(plain) == nw::Code::of(castling));
    }

    SECTION("the promotion piece is kept") {
        const auto a7 = chess::Square("a7");
        const auto a8 = chess::Square("a8");
        REQUIRE(nw::Code::of(chess::Move::make<chess::Move::PROMOTION>(a7, a8, chess::PieceType::KNIGHT)).promo() ==
                nw::KNIGHT_PROMOTION);
        REQUIRE(nw::Code::of(chess::Move::make<chess::Move::PROMOTION>(a7, a8, chess::PieceType::BISHOP)).promo() ==
                nw::BISHOP_PROMOTION);
        REQUIRE(nw::Code::of(chess::Move::make<chess::Move::PROMOTION>(a7, a8, chess::PieceType::ROOK)).promo() ==
                nw::ROOK_PROMOTION);
        REQUIRE(nw::Code::of(chess::Move::make<chess::Move::PROMOTION>(a7, a8, chess::PieceType::QUEEN)).promo() ==
                nw::QUEEN_PROMOTION);
        REQUIRE(nw::Code::of(chess::Move::make<chess::Move::NORMAL>(a7, a8)).promo() == nw::NO_PROMOTION);
    }
}

TEST_CASE("the record has the documented size", "[rules][code]") {
    REQUIRE(sizeof(nw::Example) == 80);
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
    const nw::Code first(sq("e2"), sq("e4"));
    const nw::Code last(63, 63, nw::QUEEN_PROMOTION);  // the largest possible code

    nw::CodeSet set;
    REQUIRE_FALSE(set.contains(first));

    set.insert(first);
    set.insert(first);
    set.insert(last);
    REQUIRE(set.contains(first));
    REQUIRE(set.contains(last));
    REQUIRE_FALSE(set.contains(nw::Code(sq("e2"), sq("e3"))));
    REQUIRE(set.size() == 2);

    set.clear();
    REQUIRE_FALSE(set.contains(first));
    REQUIRE_FALSE(set.contains(last));
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
            if (nw::Code::of(move) == nw::Code(sq("e7"), sq("e5"))) {
                board.makeMove(move);
                break;
            }
        }
        nw::Analysis analysis;
        analysis.analyze_legal(board);
        analysis.analyze_pseudo(board);

        const nw::Code ep(sq("d5"), sq("e6"));
        REQUIRE_FALSE(analysis.legal_set.contains(ep));
        REQUIRE(analysis.pseudo_set.contains(ep));
        REQUIRE(nw::classify(board, analysis, ep) == nw::KS_EP);
    }
}

TEST_CASE("legal moves and the resign code are never illegal", "[rules]") {
    const Position position(chess::constants::STARTPOS);
    REQUIRE(position.classify("e2", "e4") == nw::R_LEGAL);
    REQUIRE(position.classify("b1", "c3") == nw::R_LEGAL);
    REQUIRE(nw::classify(position.board, position.analysis, nw::Code::resign()) == nw::R_LEGAL);
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
                        const nw::Code code(from, to, promo);
                        const nw::Reason reason = nw::classify(board, analysis, code);
                        const bool legal        = analysis.legal_set.contains(code) || code == nw::Code::resign();
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
                if (!analysis.pseudo_set.contains(nw::Code::of(move))) {
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

TEST_CASE("legal play never reaches a position the library mishandles", "[rules][slow]") {
    nw::Rng rng(2024);
    nw::Analysis analysis;
    long double_checks = 0;
    long untrusted     = 0;

    for (int game = 0; game < 3000; ++game) {
        chess::Board board(chess960_fen(static_cast<int>(rng.below(960))), true);
        for (int ply = 0; ply < 120; ++ply) {
            analysis.analyze_legal(board);
            if (analysis.in_check) {
                double_checks += board.checkers().count() == 2;
                untrusted += !analysis.trusted;
            }
            if (analysis.no_moves) {
                break;
            }

            // prefer checking moves so that double checks actually occur
            chess::Move pick = analysis.legal[rng.below(static_cast<std::uint32_t>(analysis.legal.size()))];
            for (int attempt = 0; attempt < 6; ++attempt) {
                const chess::Move move = analysis.legal[rng.below(static_cast<std::uint32_t>(analysis.legal.size()))];
                if (board.givesCheck(move) != chess::CheckType::NO_CHECK) {
                    pick = move;
                    break;
                }
            }
            board.makeMove(pick);
        }
    }

    REQUIRE(double_checks > 0);  // the test can see double checks at all
    REQUIRE(untrusted == 0);
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
