#pragma once
// Rules layer for the dataset generator: move encoding, pseudo-legal generation,
// per-position analysis and the classifier that explains *why* a move is illegal.
// Works for standard chess and Chess960 (castling is "king takes own rook").

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include "chess.hpp"

namespace nw {

// ---------------------------------------------------------------------------------------------
// Random numbers (xoshiro256**): fast and identical on every platform, unlike <random> distributions.
// ---------------------------------------------------------------------------------------------
class Rng {
   public:
    explicit Rng(std::uint64_t seed = 1) {
        for (auto& s : s_) {
            seed += 0x9E3779B97F4A7C15ull;
            std::uint64_t z = seed;
            z               = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
            z               = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
            s               = z ^ (z >> 31);
        }
    }

    std::uint64_t next() {
        const std::uint64_t result = rotl(s_[1] * 5, 7) * 9;
        const std::uint64_t t      = s_[1] << 17;
        s_[2] ^= s_[0];
        s_[3] ^= s_[1];
        s_[1] ^= s_[2];
        s_[0] ^= s_[3];
        s_[2] ^= t;
        s_[3] = rotl(s_[3], 45);
        return result;
    }

    // Uniform in [0, n). Multiply-shift; the bias is below n / 2^32.
    std::uint32_t below(std::uint32_t n) { return static_cast<std::uint32_t>(((next() >> 32) * n) >> 32); }
    double uniform() { return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0); }
    bool chance(double p) { return uniform() < p; }
    int range(int lo, int hi) { return lo + static_cast<int>(below(static_cast<std::uint32_t>(hi - lo + 1))); }

   private:
    static std::uint64_t rotl(std::uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    std::uint64_t s_[4];
};

inline int pick_bit(std::uint64_t bits, Rng& rng) {
    int k = static_cast<int>(rng.below(static_cast<std::uint32_t>(__builtin_popcountll(bits))));
    while (k-- > 0) {
        bits &= bits - 1;
    }
    return __builtin_ctzll(bits);
}

// ---------------------------------------------------------------------------------------------
// Move code: to (bits 0-5) | from (6-11) | promotion piece (12-14: 0 none, 1 N, 2 B, 3 R, 4 Q).
// Castling is king -> own rook (Chess960 style). En passant is the plain diagonal from/to.
// RESIGN is b1->b1 (65): it is never a legal "ordinary" move and is never used as an illegal one.
// ---------------------------------------------------------------------------------------------
using Code                 = std::uint16_t;
constexpr Code RESIGN_CODE = 65;

inline Code make_code(int from, int to, int promo = 0) {
    return static_cast<Code>(to | (from << 6) | (promo << 12));
}
inline int code_from(Code c) {
    return (c >> 6) & 63;
}
inline int code_to(Code c) {
    return c & 63;
}
inline int code_promo(Code c) {
    return (c >> 12) & 7;
}

inline Code encode(const chess::Move m) {
    int promo = 0;
    if (m.typeOf() == chess::Move::PROMOTION) {
        promo = 1 + static_cast<int>(m.promotionType()) - static_cast<int>(chess::PieceType::KNIGHT);
    }
    return make_code(m.from().index(), m.to().index(), promo);
}

// ---------------------------------------------------------------------------------------------
// Buckets
// ---------------------------------------------------------------------------------------------
enum Kind : std::uint8_t {
    PAWN_SINGLE,
    PAWN_DOUBLE,
    PAWN_ATTACK,
    PAWN_ENPASSANT,
    PAWN_PROMOTION,
    KNIGHT,
    BISHOP,
    ROOK,
    QUEEN,
    KING,
    CASTLING_KINGSIDE,
    CASTLING_QUEENSIDE,
    RESIGN,  // the side to move has no legal move (checkmate or stalemate)
    NUM_KINDS
};

// Why an illegal move is illegal. When several rules are broken the first one in this order wins,
// so every example has exactly one, deterministic, reason.
enum Reason : std::uint8_t {
    R_LEGAL     = 255,
    SAME_SQUARE = 0,      // from == to
    EMPTY_FROM,           // nothing on the from-square
    OPPONENT_PIECE,       // the piece belongs to the side not to move
    OWN_CAPTURE,          // the piece could reach the square, but one of its own pieces stands there
    GEOMETRY,             // the piece cannot move like that, even on an empty board
    BLOCKED,              // slider whose path is blocked
    PAWN_BLOCKED,         // pawn push into / through an occupied square
    PAWN_RULE,            // other pawn mistakes (sideways, backwards, capture on an empty square, ...)
    PROMOTION_RULE,       // missing / unwanted / wrong promotion
    CASTLING_RULE,        // king onto own rook without a valid castling right or with a blocked path
    KS_KING_STEP,         // the king walks into (or stays in) check
    KS_PINNED,            // a pinned piece leaves the pin line (not in check)
    KS_UNRESOLVED_CHECK,  // in check, and a non-king move that does not resolve it
    KS_CASTLING,          // castling out of / through / into check
    KS_EP,                // en passant that exposes the king
    NUM_REASONS
};
constexpr int KS_FIRST = KS_KING_STEP;
constexpr int NUM_KS   = NUM_REASONS - KS_FIRST;

inline const char* kind_name(int k) {
    static const char* n[] = {"pawn_single", "pawn_double",  "pawn_attack", "pawn_enpassant", "pawn_promotion",
                              "knight",      "bishop",       "rook",        "queen",          "king",
                              "castle_king", "castle_queen", "resign"};
    return n[k];
}
inline const char* reason_name(int r) {
    static const char* n[] = {"same_square",  "empty_from",   "opponent_piece",      "own_capture",    "geometry",
                              "blocked",      "pawn_blocked", "pawn_rule",           "promotion_rule", "castling_rule",
                              "ks_king_step", "ks_pinned",    "ks_unresolved_check", "ks_castling",    "ks_ep"};
    return n[r];
}

// ---------------------------------------------------------------------------------------------
// One training example (80 bytes, natural alignment, no padding).
//   bitboards: white, black, pawns, knights, bishops, rooks, queens, kings, castling rooks
//              (the rooks that still hold a castling right; this also disambiguates Chess960)
// ---------------------------------------------------------------------------------------------
struct Example {
    std::uint64_t bitboards[9];
    std::uint16_t legal_move;
    std::uint16_t illegal_move;
    std::uint8_t ep_file;         // file of the last double pawn push, 255 if the last move was not one
    std::uint8_t white_to_move;   // 1 / 0
    std::uint8_t legal_kind;      // nw::Kind
    std::uint8_t illegal_reason;  // nw::Reason
};
static_assert(sizeof(Example) == 80, "Example must be 80 bytes");

// ---------------------------------------------------------------------------------------------
// Fast membership set for move codes, O(1) insert/lookup and O(size) clear.
// ---------------------------------------------------------------------------------------------
class CodeSet {
   public:
    CodeSet() { items_.reserve(512); }
    void insert(Code c) {
        std::uint64_t& w      = bits_[c >> 6];
        const std::uint64_t m = 1ull << (c & 63);
        if (!(w & m)) {
            w |= m;
            items_.push_back(c);
        }
    }
    bool contains(Code c) const { return (bits_[c >> 6] >> (c & 63)) & 1; }
    void clear() {
        for (const Code c : items_) {
            bits_[c >> 6] = 0;
        }
        items_.clear();
    }
    std::size_t size() const { return items_.size(); }

   private:
    std::array<std::uint64_t, 512> bits_{};  // 32768 codes
    std::vector<Code> items_;
};

struct MoveBuf {
    std::array<chess::Move, 1024> m;
    int n = 0;
    void clear() { n = 0; }
    void add(const chess::Move mv) { m[n++] = mv; }
    const chess::Move* begin() const { return m.data(); }
    const chess::Move* end() const { return m.data() + n; }
    int size() const { return n; }
};

// Pseudo-legal moves: each piece follows its movement rules and may not land on its own piece, but
// the mover's king safety is not checked. Castling needs the right and an empty path only.
inline void generate_pseudolegal(const chess::Board& board, MoveBuf& moves) {
    using namespace chess;
    moves.clear();

    const Color us       = board.sideToMove();
    const Color them     = ~us;
    const Bitboard own   = board.us(us);
    const Bitboard occ   = board.occ();
    const Bitboard enemy = board.us(them);

    auto add_targets = [&](Square from, Bitboard targets) {
        while (targets) {
            moves.add(Move::make<Move::NORMAL>(from, Square(targets.pop())));
        }
    };

    Bitboard bb = board.pieces(PieceType::KNIGHT, us);
    while (bb) {
        const Square from = bb.pop();
        add_targets(from, attacks::knight(from) & ~own);
    }
    bb = board.pieces(PieceType::BISHOP, us);
    while (bb) {
        const Square from = bb.pop();
        add_targets(from, attacks::bishop(from, occ) & ~own);
    }
    bb = board.pieces(PieceType::ROOK, us);
    while (bb) {
        const Square from = bb.pop();
        add_targets(from, attacks::rook(from, occ) & ~own);
    }
    bb = board.pieces(PieceType::QUEEN, us);
    while (bb) {
        const Square from = bb.pop();
        add_targets(from, attacks::queen(from, occ) & ~own);
    }

    const Square king_sq = board.kingSq(us);
    add_targets(king_sq, attacks::king(king_sq) & ~own);

    const auto rights = board.castlingRights();
    if (rights.has(us) && Square::back_rank(king_sq, us)) {
        for (const auto side : {Board::CastlingRights::Side::KING_SIDE, Board::CastlingRights::Side::QUEEN_SIDE}) {
            if (!rights.has(us, side)) {
                continue;
            }
            const bool king_side = side == Board::CastlingRights::Side::KING_SIDE;
            if (occ & board.getCastlingPath(us, king_side)) {
                continue;
            }
            const Square rook_sq = Square(rights.getRookFile(us, side), king_sq.rank());
            moves.add(Move::make<Move::CASTLING>(king_sq, rook_sq));
        }
    }

    const int up         = us == Color::WHITE ? 8 : -8;
    const int start_rank = us == Color::WHITE ? 1 : 6;
    const int last_rank  = us == Color::WHITE ? 7 : 0;

    auto add_pawn_move = [&](Square from, Square to) {
        if ((to.index() >> 3) == last_rank) {
            moves.add(Move::make<Move::PROMOTION>(from, to, PieceType::QUEEN));
            moves.add(Move::make<Move::PROMOTION>(from, to, PieceType::ROOK));
            moves.add(Move::make<Move::PROMOTION>(from, to, PieceType::BISHOP));
            moves.add(Move::make<Move::PROMOTION>(from, to, PieceType::KNIGHT));
        } else {
            moves.add(Move::make<Move::NORMAL>(from, to));
        }
    };

    Bitboard pawns = board.pieces(PieceType::PAWN, us);
    while (pawns) {
        const Square from = pawns.pop();
        const int one     = from.index() + up;
        if (!occ.check(one)) {
            add_pawn_move(from, Square(one));
            if ((from.index() >> 3) == start_rank && !occ.check(one + up)) {
                moves.add(Move::make<Move::NORMAL>(from, Square(one + up)));
            }
        }
        Bitboard caps = attacks::pawn(us, from) & enemy;
        while (caps) {
            add_pawn_move(from, Square(caps.pop()));
        }
    }

    const Square ep = board.enpassantSq();
    if (ep != Square::NO_SQ) {
        Bitboard candidates = attacks::pawn(them, ep) & board.pieces(PieceType::PAWN, us);
        while (candidates) {
            moves.add(Move::make<Move::ENPASSANT>(Square(candidates.pop()), ep));
        }
    }
}

inline Kind kind_of(const chess::Board& b, const chess::Move m) {
    switch (m.typeOf()) {
        case chess::Move::ENPASSANT:
            return PAWN_ENPASSANT;
        case chess::Move::PROMOTION:
            return PAWN_PROMOTION;
        case chess::Move::CASTLING:
            return m.to().index() > m.from().index() ? CASTLING_KINGSIDE : CASTLING_QUEENSIDE;
        default:
            break;
    }
    switch (b.at<chess::PieceType>(m.from()).internal()) {
        case chess::PieceType::PAWN: {
            const int dr = std::abs((m.to().index() >> 3) - (m.from().index() >> 3));
            if (dr == 2) {
                return PAWN_DOUBLE;
            }
            return (m.to().index() & 7) == (m.from().index() & 7) ? PAWN_SINGLE : PAWN_ATTACK;
        }
        case chess::PieceType::KNIGHT:
            return KNIGHT;
        case chess::PieceType::BISHOP:
            return BISHOP;
        case chess::PieceType::ROOK:
            return ROOK;
        case chess::PieceType::QUEEN:
            return QUEEN;
        default:
            return KING;
    }
}

// chess-library counts a check by two diagonal sliders (bishop / queen) as a single check, and likewise two
// knights or two pawns, so in those positions it also returns moves that leave the king in check. Such positions
// are skipped (they are rare, but real: a queen can discover a diagonal attack by moving onto the other diagonal).
inline bool library_handles_checks(const chess::Board& b) {
    const chess::Color us     = b.sideToMove();
    const chess::Color them   = ~us;
    const chess::Square k     = b.kingSq(us);
    const chess::Bitboard occ = b.occ();
    if ((chess::attacks::knight(k) & b.pieces(chess::PieceType::KNIGHT, them)).count() > 1) {
        return false;
    }
    if ((chess::attacks::pawn(us, k) & b.pieces(chess::PieceType::PAWN, them)).count() > 1) {
        return false;
    }
    const chess::Bitboard diagonal = b.pieces(chess::PieceType::BISHOP, them) | b.pieces(chess::PieceType::QUEEN, them);
    return (chess::attacks::bishop(k, occ) & diagonal).count() <= 1;
}

// ---------------------------------------------------------------------------------------------
// Per-position analysis (buffers are reused between positions)
// ---------------------------------------------------------------------------------------------
struct Analysis {
    chess::Movelist legal;
    MoveBuf pseudo;
    CodeSet legal_set, pseudo_set;
    std::array<std::vector<chess::Move>, NUM_KINDS> by_kind;  // legal moves per kind (RESIGN stays empty)
    std::array<std::vector<Code>, NUM_KS> ks;                 // pseudo-legal but illegal moves per king-safety reason
    bool in_check    = false;
    bool no_moves    = false;  // checkmate or stalemate
    bool pseudo_done = false;
    bool trusted     = true;  // false if the library's legal move generation is known to be wrong here

    void analyze_legal(const chess::Board& b) {
        legal_set.clear();
        pseudo_set.clear();
        for (auto& v : by_kind) {
            v.clear();
        }
        for (auto& v : ks) {
            v.clear();
        }
        pseudo_done = false;

        chess::movegen::legalmoves(legal, b);
        no_moves = legal.empty();
        for (const chess::Move m : legal) {
            legal_set.insert(encode(m));
            by_kind[kind_of(b, m)].push_back(m);
        }
        in_check = b.inCheck();
        trusted  = !in_check || library_handles_checks(b);
    }

    bool kind_available(int k) const { return k == RESIGN ? no_moves : !by_kind[k].empty(); }

    // Needs analyze_legal() first. Fills the pseudo-legal set and the king-safety pools.
    void analyze_pseudo(const chess::Board& b);
};

inline chess::Bitboard attack_pattern(chess::PieceType pt, chess::Color us, int sq, chess::Bitboard occ) {
    using chess::PieceType;
    const chess::Square s(sq);
    switch (pt.internal()) {
        case PieceType::PAWN:
            return chess::attacks::pawn(us, s);
        case PieceType::KNIGHT:
            return chess::attacks::knight(s);
        case PieceType::BISHOP:
            return chess::attacks::bishop(s, occ);
        case PieceType::ROOK:
            return chess::attacks::rook(s, occ);
        case PieceType::QUEEN:
            return chess::attacks::queen(s, occ);
        default:
            return chess::attacks::king(s);
    }
}

inline bool is_slider(chess::PieceType pt) {
    return pt == chess::PieceType::BISHOP || pt == chess::PieceType::ROOK || pt == chess::PieceType::QUEEN;
}

// Explains why `c` is not a legal move in the position. Returns R_LEGAL if it is legal (or the resign code).
inline Reason classify(const chess::Board& b, const Analysis& a, const Code c) {
    using namespace chess;
    if (c == RESIGN_CODE || a.legal_set.contains(c)) {
        return R_LEGAL;
    }

    const int from  = code_from(c);
    const int to    = code_to(c);
    const int promo = code_promo(c);
    if (promo > 4) {
        return PROMOTION_RULE;
    }
    if (from == to) {
        return SAME_SQUARE;
    }

    const Piece p = b.at<Piece>(Square(from));
    if (p.type() == PieceType::NONE) {
        return EMPTY_FROM;
    }

    const Color us = b.sideToMove();
    if (p.color() != us) {
        return OPPONENT_PIECE;
    }

    const PieceType pt  = p.type();
    const bool is_pawn  = pt == PieceType::PAWN;
    const int last_rank = us == Color::WHITE ? 7 : 0;
    const bool to_last  = (to >> 3) == last_rank;
    Code lookup         = c;

    if (promo != 0) {
        if (!is_pawn || !to_last) {
            return PROMOTION_RULE;
        }
    } else if (is_pawn && to_last) {
        // Pawn reaches the last rank without saying what it becomes.
        if (a.pseudo_set.contains(static_cast<Code>(c | (4 << 12)))) {
            return PROMOTION_RULE;
        }
        lookup = static_cast<Code>(c | (4 << 12));
    }

    const Piece tp      = b.at<Piece>(Square(to));
    const bool to_empty = tp.type() == PieceType::NONE;
    const bool to_own   = !to_empty && tp.color() == us;

    if (a.pseudo_set.contains(lookup)) {
        // Follows the movement rules, so the only thing wrong is king safety.
        if (pt == PieceType::KING) {
            return (to_own && tp.type() == PieceType::ROOK) ? KS_CASTLING : KS_KING_STEP;
        }
        if (is_pawn && ((from ^ to) & 7) != 0 && to_empty) {
            return KS_EP;
        }
        return a.in_check ? KS_UNRESOLVED_CHECK : KS_PINNED;
    }

    // Movement violation.
    if (is_pawn) {
        const int dir        = us == Color::WHITE ? 8 : -8;
        const int start_rank = us == Color::WHITE ? 1 : 6;
        if ((from & 7) == (to & 7)) {
            if (to == from + dir) {
                return PAWN_BLOCKED;  // an empty target would have been pseudo-legal
            }
            if (to == from + 2 * dir) {
                return (from >> 3) == start_rank ? PAWN_BLOCKED : PAWN_RULE;
            }
            return PAWN_RULE;
        }
        if (to_own && attacks::pawn(us, Square(from)).check(to)) {
            return OWN_CAPTURE;
        }
        return PAWN_RULE;
    }

    if (pt == PieceType::KING && to_own && tp.type() == PieceType::ROOK) {
        return CASTLING_RULE;
    }

    const Bitboard occ = b.occ();
    if (attack_pattern(pt, us, from, occ).check(to)) {
        return OWN_CAPTURE;  // reachable, so it must be own-occupied
    }
    if (is_slider(pt) && attack_pattern(pt, us, from, Bitboard(0)).check(to)) {
        return BLOCKED;
    }
    return GEOMETRY;
}

inline void Analysis::analyze_pseudo(const chess::Board& b) {
    generate_pseudolegal(b, pseudo);
    for (const chess::Move m : pseudo) {
        pseudo_set.insert(encode(m));
    }
    pseudo_done = true;
    for (const chess::Move m : pseudo) {
        const Code c = encode(m);
        if (legal_set.contains(c)) {
            continue;
        }
        const Reason r = classify(b, *this, c);
        if (r < KS_FIRST || r >= NUM_REASONS) {
            std::cerr << "internal error: pseudo-legal illegal move classified as " << reason_name(r) << "\n";
            std::abort();
        }
        ks[r - KS_FIRST].push_back(c);
    }
}

}  // namespace nw
