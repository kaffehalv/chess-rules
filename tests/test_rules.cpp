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

    cr::Reason classify(const char* from, const char* to, int promo = 0) const {
        return cr::classify(board, analysis, cr::Code(sq(from), sq(to), promo));
    }

    bool is_legal(const char* from, const char* to, int promo = 0) const {
        return analysis.legal_set.contains(cr::Code(sq(from), sq(to), promo));
    }

    chess::Board board;
    cr::Analysis analysis;
};

constexpr int KNIGHT_PROMO = 1;
constexpr int ROOK_PROMO   = 3;
constexpr int QUEEN_PROMO  = 4;

}  // namespace

TEST_CASE("a Code keeps from, to and promotion apart", "[rules][code]") {
    const cr::Code code(sq("e7"), sq("e8"), cr::QUEEN_PROMOTION);
    REQUIRE(code.from() == sq("e7"));
    REQUIRE(code.to() == sq("e8"));
    REQUIRE(code.promo() == cr::QUEEN_PROMOTION);

    SECTION("a promotion can be added and removed") {
        REQUIRE(code.without_promo() == cr::Code(sq("e7"), sq("e8")));
        REQUIRE(code.without_promo().with_promo(cr::QUEEN_PROMOTION) == code);
        REQUIRE(code.with_promo(cr::KNIGHT_PROMOTION) != code);
    }

    SECTION("the raw value has the documented layout") {
        REQUIRE(code.raw() == (sq("e8") | sq("e7") << 6 | cr::QUEEN_PROMOTION << 12));
    }

    SECTION("resign is b1b1") {
        static_assert(cr::Code::resign() == cr::Code(1, 1));
        REQUIRE(cr::Code::resign() == cr::Code(sq("b1"), sq("b1")));
        REQUIRE(cr::Code::resign().raw() == 65);
    }

    SECTION("the raw values are a dense index and the codes are ordered") {
        REQUIRE(cr::Code(0, 0).raw() == 0);
        REQUIRE(cr::Code(63, 63, cr::QUEEN_PROMOTION).raw() == cr::Code::COUNT - 1);
        REQUIRE(cr::Code(0, 1) < cr::Code(0, 2));
    }
}

TEST_CASE("only valid values become codes", "[rules][code]") {
    REQUIRE(cr::Code::from_raw(0).has_value());
    REQUIRE(cr::Code::from_raw(cr::Code::COUNT - 1).has_value());
    REQUIRE(cr::Code::from_raw(65).value() == cr::Code::resign());

    REQUIRE_FALSE(cr::Code::from_raw(cr::Code::COUNT).has_value());
    REQUIRE_FALSE(cr::Code::from_raw(5 << 12).has_value());  // a promotion piece that does not exist
    REQUIRE_FALSE(cr::Code::from_raw(0xFFFF).has_value());
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
        REQUIRE(cr::Code::of(plain) == cr::Code::of(en_passant));
    }

    SECTION("castling and king takes own rook") {
        const auto plain    = chess::Move::make<chess::Move::NORMAL>(e1, h1);
        const auto castling = chess::Move::make<chess::Move::CASTLING>(e1, h1);
        REQUIRE(plain != castling);
        REQUIRE(cr::Code::of(plain) == cr::Code::of(castling));
    }

    SECTION("the promotion piece is kept") {
        const auto a7 = chess::Square("a7");
        const auto a8 = chess::Square("a8");
        REQUIRE(cr::Code::of(chess::Move::make<chess::Move::PROMOTION>(a7, a8, chess::PieceType::KNIGHT)).promo() ==
                cr::KNIGHT_PROMOTION);
        REQUIRE(cr::Code::of(chess::Move::make<chess::Move::PROMOTION>(a7, a8, chess::PieceType::BISHOP)).promo() ==
                cr::BISHOP_PROMOTION);
        REQUIRE(cr::Code::of(chess::Move::make<chess::Move::PROMOTION>(a7, a8, chess::PieceType::ROOK)).promo() ==
                cr::ROOK_PROMOTION);
        REQUIRE(cr::Code::of(chess::Move::make<chess::Move::PROMOTION>(a7, a8, chess::PieceType::QUEEN)).promo() ==
                cr::QUEEN_PROMOTION);
        REQUIRE(cr::Code::of(chess::Move::make<chess::Move::NORMAL>(a7, a8)).promo() == cr::NO_PROMOTION);
    }
}

TEST_CASE("the record has the documented size", "[rules][code]") {
    REQUIRE(sizeof(cr::Example) == 80);
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
    cr::Rng first(42);
    cr::Rng second(42);
    cr::Rng other(43);

    bool differs = false;
    for (int i = 0; i < 100; ++i) {
        const std::uint64_t value = first.next();
        REQUIRE(value == second.next());
        differs = differs || value != other.next();
    }
    REQUIRE(differs);

    cr::Rng rng(1);
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
        REQUIRE(((bits >> cr::pick_bit(bits, rng)) & 1) == 1);
    }
}

TEST_CASE("CodeSet inserts, finds and clears in constant time", "[rules]") {
    const cr::Code first(sq("e2"), sq("e4"));
    const cr::Code last(63, 63, cr::QUEEN_PROMOTION);  // the largest possible code

    cr::CodeSet set;
    REQUIRE_FALSE(set.contains(first));

    set.insert(first);
    set.insert(first);
    set.insert(last);
    REQUIRE(set.contains(first));
    REQUIRE(set.contains(last));
    REQUIRE_FALSE(set.contains(cr::Code(sq("e2"), sq("e3"))));
    REQUIRE(set.size() == 2);

    set.clear();
    REQUIRE_FALSE(set.contains(first));
    REQUIRE_FALSE(set.contains(last));
    REQUIRE(set.size() == 0);
}

TEST_CASE("every illegal reason is recognised", "[rules]") {
    SECTION("same square") {
        const Position position(chess::constants::STARTPOS);
        REQUIRE(position.classify("e2", "e2") == cr::SAME_SQUARE);
    }

    SECTION("empty from square") {
        const Position position(chess::constants::STARTPOS);
        REQUIRE(position.classify("e4", "e5") == cr::EMPTY_FROM);
    }

    SECTION("opponent piece") {
        const Position position(chess::constants::STARTPOS);
        REQUIRE(position.classify("e7", "e5") == cr::OPPONENT_PIECE);
    }

    SECTION("own capture") {
        const Position position(chess::constants::STARTPOS);
        REQUIRE(position.classify("a1", "a2") == cr::OWN_CAPTURE);
        REQUIRE(position.classify("e2", "d3") == cr::PAWN_RULE);  // not an own capture: nothing stands on d3
    }

    SECTION("geometry") {
        const Position position(chess::constants::STARTPOS);
        REQUIRE(position.classify("b1", "b4") == cr::GEOMETRY);
        REQUIRE(position.classify("e1", "e3") == cr::GEOMETRY);
    }

    SECTION("blocked slider") {
        const Position position(chess::constants::STARTPOS);
        REQUIRE(position.classify("a1", "a5") == cr::BLOCKED);
        REQUIRE(position.classify("c1", "h6") == cr::BLOCKED);
    }

    SECTION("blocked pawn") {
        const Position position("4k3/8/8/8/8/4p3/4P3/4K3 w - - 0 1");
        REQUIRE(position.classify("e2", "e3") == cr::PAWN_BLOCKED);
        REQUIRE(position.classify("e2", "e4") == cr::PAWN_BLOCKED);
    }

    SECTION("other pawn mistakes") {
        const Position position(chess::constants::STARTPOS);
        REQUIRE(position.classify("e2", "e1") == cr::PAWN_RULE);
        REQUIRE(position.classify("e2", "f2") == cr::PAWN_RULE);
        REQUIRE(position.classify("e2", "e5") == cr::PAWN_RULE);
    }

    SECTION("promotion rule") {
        const Position position("4k3/P7/8/8/8/8/8/4K3 w - - 0 1");
        REQUIRE(position.classify("a7", "a8") == cr::PROMOTION_RULE);
        REQUIRE(position.classify("a7", "a6", ROOK_PROMO) == cr::PROMOTION_RULE);
        REQUIRE(position.classify("e1", "f1", QUEEN_PROMO) == cr::PROMOTION_RULE);
        REQUIRE(position.is_legal("a7", "a8", QUEEN_PROMO));
    }

    SECTION("castling rule") {
        const Position no_rights("r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1");
        REQUIRE(no_rights.classify("e1", "h1") == cr::CASTLING_RULE);

        const Position blocked("r3k2r/8/8/8/8/8/8/RN2K2R w KQkq - 0 1");
        REQUIRE(blocked.classify("e1", "a1") == cr::CASTLING_RULE);
        REQUIRE(blocked.is_legal("e1", "h1"));
    }

    SECTION("king steps into check") {
        const Position position("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1");
        REQUIRE(position.classify("e1", "d2") == cr::KS_KING_STEP);
        REQUIRE(position.is_legal("e1", "e2"));
    }

    SECTION("pinned piece") {
        const Position position("4k3/8/8/8/4r3/8/4R3/4K3 w - - 0 1");
        REQUIRE(position.classify("e2", "a2") == cr::KS_PINNED);
        REQUIRE(position.is_legal("e2", "e3"));
    }

    SECTION("check is not resolved") {
        const Position position("4k3/8/8/8/8/8/P3r3/4K3 w - - 0 1");
        REQUIRE(position.analysis.in_check);
        REQUIRE(position.classify("a2", "a3") == cr::KS_UNRESOLVED_CHECK);
    }

    SECTION("castling through an attacked square") {
        const Position position("r3k2r/8/8/8/8/8/5r2/R3K2R w KQkq - 0 1");
        REQUIRE(position.is_legal("e1", "a1"));
        REQUIRE(position.classify("e1", "h1") == cr::KS_CASTLING);
    }

    SECTION("en passant that exposes the king along the rank") {
        // The library only keeps the en passant square after the double push has been played.
        chess::Board board("7k/4p3/8/2KP3r/8/8/8/8 b - - 0 1", true);
        chess::Movelist moves;
        chess::movegen::legalmoves(moves, board);
        for (const chess::Move move : moves) {
            if (cr::Code::of(move) == cr::Code(sq("e7"), sq("e5"))) {
                board.makeMove(move);
                break;
            }
        }
        cr::Analysis analysis;
        analysis.analyze_legal(board);
        analysis.analyze_pseudo(board);

        const cr::Code ep(sq("d5"), sq("e6"));
        REQUIRE_FALSE(analysis.legal_set.contains(ep));
        REQUIRE(analysis.pseudo_set.contains(ep));
        REQUIRE(cr::classify(board, analysis, ep) == cr::KS_EP);
    }
}

TEST_CASE("legal moves and the resign code are never illegal", "[rules]") {
    const Position position(chess::constants::STARTPOS);
    REQUIRE(position.classify("e2", "e4") == cr::R_LEGAL);
    REQUIRE(position.classify("b1", "c3") == cr::R_LEGAL);
    REQUIRE(cr::classify(position.board, position.analysis, cr::Code::resign()) == cr::R_LEGAL);
}

TEST_CASE("legal move kinds", "[rules]") {
    const auto kind_of_move = [](const std::string& fen, const std::string& uci) {
        chess::Board board(fen, true);
        return cr::kind_of(board, chess::uci::uciToMove(board, uci));
    };

    REQUIRE(kind_of_move(chess::constants::STARTPOS, "e2e3") == cr::PAWN_SINGLE);
    REQUIRE(kind_of_move(chess::constants::STARTPOS, "e2e4") == cr::PAWN_DOUBLE);
    REQUIRE(kind_of_move(chess::constants::STARTPOS, "b1c3") == cr::KNIGHT);
    REQUIRE(kind_of_move("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5") == cr::PAWN_ATTACK);
    REQUIRE(kind_of_move("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8q") == cr::PAWN_PROMOTION);
    REQUIRE(kind_of_move("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6") == cr::PAWN_ENPASSANT);
    REQUIRE(kind_of_move("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1h1") == cr::CASTLING_KINGSIDE);
    REQUIRE(kind_of_move("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1a1") == cr::CASTLING_QUEENSIDE);
    REQUIRE(kind_of_move("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "a1a5") == cr::ROOK);
    REQUIRE(kind_of_move("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1d1") == cr::KING);
}

TEST_CASE("the classifier agrees with the library on every possible move code", "[rules][slow]") {
    cr::Rng rng(7);
    cr::Analysis analysis;
    long positions  = 0;
    long violations = 0;

    for (int game = 0; game < 40; ++game) {
        chess::Board board(cr::chess960_fen(static_cast<int>(rng.below(960))), true);
        for (int ply = 0; ply < 80; ++ply) {
            analysis.analyze_legal(board);
            analysis.analyze_pseudo(board);
            ++positions;

            for (int from = 0; from < 64; ++from) {
                for (int to = 0; to < 64; ++to) {
                    for (int promo = 0; promo <= 4; ++promo) {
                        const cr::Code code(from, to, promo);
                        const cr::Reason reason = cr::classify(board, analysis, code);
                        const bool legal        = analysis.legal_set.contains(code) || code == cr::Code::resign();
                        const bool pseudo_illegal =
                            analysis.pseudo_set.contains(code) && !analysis.legal_set.contains(code);

                        if ((reason == cr::R_LEGAL) != legal) {
                            ++violations;
                        }
                        // king safety reasons are exactly the moves that follow the movement rules but are illegal
                        if (reason != cr::R_LEGAL && (reason >= cr::KS_FIRST) != pseudo_illegal) {
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
    cr::Rng rng(11);
    cr::Analysis analysis;
    long missing = 0;

    for (int game = 0; game < 200; ++game) {
        chess::Board board(cr::chess960_fen(static_cast<int>(rng.below(960))), true);
        for (int ply = 0; ply < 80; ++ply) {
            analysis.analyze_legal(board);
            analysis.analyze_pseudo(board);
            for (const chess::Move move : analysis.legal) {
                if (!analysis.pseudo_set.contains(cr::Code::of(move))) {
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
    REQUIRE(position.analysis.kind_available(cr::RESIGN));
    REQUIRE_FALSE(position.analysis.kind_available(cr::KING));
}

TEST_CASE("legal play never reaches a position the library mishandles", "[rules][slow]") {
    cr::Rng rng(2024);
    cr::Analysis analysis;
    long double_checks = 0;
    long untrusted     = 0;

    for (int game = 0; game < 3000; ++game) {
        chess::Board board(cr::chess960_fen(static_cast<int>(rng.below(960))), true);
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
