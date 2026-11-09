#pragma once
// Bucket-balanced dataset generator for legal / illegal move classification (standard chess and Chess960).
//
// Every example is (board, one legal move, one illegal move). The legal move belongs to one of 13 kinds
// (RESIGN = the side to move has no legal move at all), the illegal move to one of 15 reasons it is illegal.
// Every example fills one cell (kind, reason). A short calibration run measures which cells each position
// source can supply, cell quotas are fitted so that every kind has the same number of examples and every
// reason has about the same number, and then the generator spends its time on whichever sources can still
// fill open cells. Positions come from random play, sparse random boards and builders for rare situations
// (castling, en passant, promotion, pins, checkmate / stalemate).

#include <numeric>
#include <optional>
#include <string>

#include "chess-rules/chess960.hpp"
#include "chess-rules/rules.hpp"

namespace nw {

enum class Variant { STANDARD, CHESS960, MIXED };

struct Config {
    std::uint64_t kind_target = 1000;  // examples per legal kind
    std::uint64_t seed        = 1;
    int split                 = 0;  // 0 train, 1 validation, 2 test (by position hash, 80 / 10 / 10)
    Variant variant           = Variant::MIXED;
    std::vector<int> starts;  // Chess960 start indices used for play; empty = all 960
    std::uint32_t max_plies = 250;
    int calibration_iters   = 10000;  // iterations per source used to measure what each source can supply
    double min_play_share   = 0.25;   // minimum share of iterations spent on plain random play ...
    double min_sparse_share = 0.10;   // ... and on sparse random boards, to keep the data close to real positions
    bool verbose            = false;
};

struct Stats {
    std::array<std::uint64_t, NUM_KINDS> kind{};
    std::array<std::uint64_t, NUM_REASONS> reason{};
    std::array<std::array<std::uint64_t, NUM_REASONS>, NUM_KINDS> joint{};
    std::array<std::uint64_t, 8> source_iterations{};
    std::array<std::uint64_t, 8> source_examples{};
    std::uint64_t iterations     = 0;
    std::uint64_t rejected_split = 0, rejected_no_kind = 0, rejected_no_illegal = 0;

    void merge(const Stats& o) {
        for (int i = 0; i < NUM_KINDS; ++i) {
            kind[i] += o.kind[i];
        }
        for (int i = 0; i < NUM_REASONS; ++i) {
            reason[i] += o.reason[i];
        }
        for (int i = 0; i < NUM_KINDS; ++i) {
            for (int j = 0; j < NUM_REASONS; ++j) {
                joint[i][j] += o.joint[i][j];
            }
        }
        for (int i = 0; i < 8; ++i) {
            source_iterations[i] += o.source_iterations[i];
            source_examples[i] += o.source_examples[i];
        }
        iterations += o.iterations;
        rejected_split += o.rejected_split;
        rejected_no_kind += o.rejected_no_kind;
        rejected_no_illegal += o.rejected_no_illegal;
    }
};

// Which of the three splits a position belongs to. Hash-based, so a position can never be in two splits.
inline int split_of(std::uint64_t hash) {
    std::uint64_t z = hash + 0x9E3779B97F4A7C15ull;
    z               = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z               = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    const int r = static_cast<int>(z % 100);
    return r < 80 ? 0 : (r < 90 ? 1 : 2);
}

// Board under construction: FEN characters, 0 = empty.
struct Grid {
    std::array<char, 64> c;
    std::uint64_t occ = 0;
    Grid() { c.fill(0); }
    bool empty(int sq) const { return !((occ >> sq) & 1); }
    void put(int sq, char ch) {
        c[sq] = ch;
        occ |= 1ull << sq;
    }
    void remove(int sq) {
        c[sq] = 0;
        occ &= ~(1ull << sq);
    }
    int count(char ch) const {
        int n = 0;
        for (const char x : c) {
            n += x == ch;
        }
        return n;
    }
};

inline Grid grid_from_placement(const std::string& placement) {
    Grid g;
    int rank = 7, file = 0;
    for (const char ch : placement) {
        if (ch == '/') {
            --rank;
            file = 0;
        } else if (ch >= '1' && ch <= '8') {
            file += ch - '0';
        } else {
            g.put(rank * 8 + file, ch);
            ++file;
        }
    }
    return g;
}

inline std::string grid_fen(const Grid& g, bool white_to_move, const std::string& castling) {
    std::string s;
    for (int r = 7; r >= 0; --r) {
        int empty = 0;
        for (int f = 0; f < 8; ++f) {
            const char ch = g.c[r * 8 + f];
            if (!ch) {
                ++empty;
            } else {
                if (empty) {
                    s += static_cast<char>('0' + empty);
                }
                empty = 0;
                s += ch;
            }
        }
        if (empty) {
            s += static_cast<char>('0' + empty);
        }
        if (r) {
            s += '/';
        }
    }
    s += white_to_move ? " w " : " b ";
    s += castling.empty() ? "-" : castling;
    s += " - 0 1";
    return s;
}

inline int chebyshev(int a, int b) {
    return std::max(std::abs((a & 7) - (b & 7)), std::abs((a >> 3) - (b >> 3)));
}

class Shard {
   public:
    enum Source { PLAY, SPARSE, CASTLE, EP, PROMO, PIN, TERMINAL, NUM_SOURCES };
    static const char* source_name(int s) {
        static const char* n[] = {"play", "sparse", "castle", "en_passant", "promotion", "pin", "terminal"};
        return n[s];
    }

    explicit Shard(Config cfg) : cfg_(std::move(cfg)), rng_(cfg_.seed) {
        if (cfg_.starts.empty()) {
            cfg_.starts.resize(960);
            std::iota(cfg_.starts.begin(), cfg_.starts.end(), 0);
        }
        scratch_.set960(true);
        play_.set960(true);
        calibrate(cfg_.calibration_iters);
        compute_quota();
    }

    const Stats& stats() const { return stats_; }

    // Generates until every cell has reached its quota (so every legal kind has kind_target examples).
    void run(std::vector<Example>& out) {
        out.reserve(out.size() + cfg_.kind_target * NUM_KINDS);
        std::uint64_t last_progress = 0, total = 0, next_report = 1;
        while (!all_full()) {
            const Source src = pick_source();
            ++stats_.iterations;
            ++stats_.source_iterations[src];
            if (step(src, out)) {
                ++stats_.source_examples[src];
                last_progress = stats_.iterations;
                ++total;
            } else if (stats_.iterations - last_progress > 3'000'000ull) {
                // The remaining cells are too rare to fill; hand their quota to better supplied cells of the same row.
                if (!rebalance_stalled()) {
                    std::cerr << "no progress and nothing left to rebalance\n";
                    std::abort();
                }
                if (cfg_.verbose) {
                    std::cerr << "stalled: rebalanced the rarest open cell\n";
                }
                last_progress = stats_.iterations;
                weight_age_   = 1 << 30;
            }
            if (cfg_.verbose && total >= next_report) {
                report(total);
                next_report = total + std::max<std::uint64_t>(1, cfg_.kind_target * NUM_KINDS / 20);
            }
        }
    }

    bool rebalance_stalled() {
        int bk = -1, br = -1;
        double best = 1e300;
        for (int k = 0; k < NUM_KINDS; ++k) {
            if (!kind_open(k)) {
                continue;
            }
            for (int r = 0; r < NUM_REASONS; ++r) {
                if (!cell_open(k, r)) {
                    continue;
                }
                double sup = 0;
                for (int s = 0; s < NUM_SOURCES; ++s) {
                    sup += supply(s, k, r);
                }
                if (sup < best) {
                    best = sup;
                    bk   = k;
                    br   = r;
                }
            }
        }
        if (bk < 0) {
            return false;
        }
        const std::uint64_t left = quota_[bk][br] - stats_.joint[bk][br];
        quota_[bk][br]           = stats_.joint[bk][br];
        int target               = -1;
        double top               = -1;
        for (int r = 0; r < NUM_REASONS; ++r) {
            if (r == br) {
                continue;
            }
            double sup = 0;
            for (int s = 0; s < NUM_SOURCES; ++s) {
                sup += supply(s, bk, r);
            }
            if (sup > top) {
                top    = sup;
                target = r;
            }
        }
        if (target < 0 || top <= 0) {
            return false;
        }
        quota_[bk][target] += left;
        return true;
    }

   public:  // internals are public on purpose: every builder and helper is unit-testable
    Config cfg_;
    Rng rng_;
    chess::Board scratch_, play_;
    Analysis an_;
    Stats stats_;
    int ep_file_            = -1;  // for scratch_ positions
    int play_ep_            = -1;
    std::uint32_t play_ply_ = 0;
    bool game_active_       = false;

    // ------------------------------------------------------------------ quotas and scheduling
    //
    // Every example fills one cell (legal kind, illegal reason). Cell quotas are chosen so that every kind
    // has kind_target examples and every reason has about the same number, using only cells that some
    // source can actually produce. Cells fill independently, so there is no lockstep pacing; the scheduler
    // simply sends effort to the sources that can supply the cells that are still open.
    static constexpr double SOURCE_COST[NUM_SOURCES] = {1.5, 1.4,  3.2, 1.5,
                                                        1.6, 0.95, 2.3};  // microseconds / iteration

    std::array<std::array<std::uint64_t, NUM_REASONS>, NUM_KINDS> quota_{};
    std::vector<double> supply_;  // [source][kind][reason]: P(a position offers this cell)
    double weight_[NUM_SOURCES] = {};
    int weight_age_             = 1 << 30;

    double& supply(int s, int k, int r) { return supply_[(s * NUM_KINDS + k) * NUM_REASONS + r]; }

    bool cell_open(int k, int r) const { return stats_.joint[k][r] < quota_[k][r]; }
    bool kind_open(int k) const { return stats_.kind[k] < cfg_.kind_target; }

    bool all_full() const {
        for (int k = 0; k < NUM_KINDS; ++k) {
            if (stats_.kind[k] < cfg_.kind_target) {
                return false;
            }
        }
        return true;
    }

    // One pass of every source without emitting anything: measures which (kind, reason) cells each source offers.
    void calibrate(int iterations) {
        supply_.assign(static_cast<std::size_t>(NUM_SOURCES) * NUM_KINDS * NUM_REASONS, 0.0);
        std::vector<int> counts(supply_.size(), 0);
        for (int s = 0; s < NUM_SOURCES; ++s) {
            for (int i = 0; i < iterations; ++i) {
                const chess::Board* board = &scratch_;
                ep_file_                  = -1;
                if (s == PLAY) {
                    if (!game_active_ && !new_game()) {
                        continue;
                    }
                    an_.analyze_legal(play_);
                    board = &play_;
                } else {
                    if (!build(static_cast<Source>(s))) {
                        continue;
                    }
                    an_.analyze_legal(scratch_);
                }
                an_.analyze_pseudo(*board);
                bool ok[NUM_REASONS];
                for (int r = 0; r < NUM_REASONS; ++r) {
                    ok[r] = propose(r, *board).has_value();
                }
                for (int k = 0; k < NUM_KINDS; ++k) {
                    if (!an_.kind_available(k)) {
                        continue;
                    }
                    for (int r = 0; r < NUM_REASONS; ++r) {
                        if (ok[r]) {
                            ++counts[(s * NUM_KINDS + k) * NUM_REASONS + r];
                        }
                    }
                }
                if (s == PLAY) {
                    advance_game();
                }
            }
            game_active_ = false;
            for (int k = 0; k < NUM_KINDS; ++k) {
                for (int r = 0; r < NUM_REASONS; ++r) {
                    supply(s, k, r) = double(counts[(s * NUM_KINDS + k) * NUM_REASONS + r]) / iterations;
                }
            }
        }
    }

    // Iterative proportional fitting: rows sum to kind_target, columns to about the same number each.
    void compute_quota() {
        const double row_total = double(cfg_.kind_target);
        const double col_total = row_total * double(NUM_KINDS) / double(NUM_REASONS);
        double q[NUM_KINDS][NUM_REASONS];
        for (int k = 0; k < NUM_KINDS; ++k) {
            for (int r = 0; r < NUM_REASONS; ++r) {
                double best = 0;
                for (int s = 0; s < NUM_SOURCES; ++s) {
                    best = std::max(best, supply(s, k, r));
                }
                // cells that are too rare to fill are dropped; rare-ish cells get a smaller prior
                q[k][r] = best >= 2e-3 ? std::clamp(best / 0.02, 0.05, 1.0) : 0.0;
            }
        }
        for (int it = 0; it < 500; ++it) {
            for (int k = 0; k < NUM_KINDS; ++k) {
                double sum = 0;
                for (int r = 0; r < NUM_REASONS; ++r) {
                    sum += q[k][r];
                }
                if (sum > 0) {
                    for (int r = 0; r < NUM_REASONS; ++r) {
                        q[k][r] *= row_total / sum;
                    }
                }
            }
            for (int r = 0; r < NUM_REASONS; ++r) {
                double sum = 0;
                for (int k = 0; k < NUM_KINDS; ++k) {
                    sum += q[k][r];
                }
                if (sum > 0) {
                    for (int k = 0; k < NUM_KINDS; ++k) {
                        q[k][r] *= col_total / sum;
                    }
                }
            }
        }
        // round each row to integers that sum to exactly kind_target (largest remainder)
        for (int k = 0; k < NUM_KINDS; ++k) {
            double sum = 0;
            for (int r = 0; r < NUM_REASONS; ++r) {
                sum += q[k][r];
            }
            std::array<std::pair<double, int>, NUM_REASONS> frac;
            std::uint64_t assigned = 0;
            for (int r = 0; r < NUM_REASONS; ++r) {
                const double v = sum > 0 ? q[k][r] * row_total / sum : 0.0;
                quota_[k][r]   = static_cast<std::uint64_t>(v);
                assigned += quota_[k][r];
                frac[r] = {-(v - std::floor(v)), r};
            }
            std::sort(frac.begin(), frac.end());
            for (std::uint64_t i = 0; assigned < cfg_.kind_target && i < NUM_REASONS && sum > 0; ++i, ++assigned) {
                ++quota_[k][frac[i].second];
            }
        }
    }

    void refresh_weights() {
        double total = 0;
        for (int s = 0; s < NUM_SOURCES; ++s) {
            double none = 1.0;
            for (int k = 0; k < NUM_KINDS; ++k) {
                if (!kind_open(k)) {
                    continue;
                }
                for (int r = 0; r < NUM_REASONS; ++r) {
                    if (cell_open(k, r)) {
                        none *= 1.0 - supply(s, k, r);
                    }
                }
            }
            weight_[s] = ((1.0 - none) + 1e-4) / SOURCE_COST[s];
            total += weight_[s];
        }
        // random play and sparse boards keep a minimum share so the data stays close to real positions
        const double play_floor   = cfg_.min_play_share / (1.0 - cfg_.min_play_share - cfg_.min_sparse_share);
        const double sparse_floor = cfg_.min_sparse_share / (1.0 - cfg_.min_play_share - cfg_.min_sparse_share);
        double others             = total - weight_[PLAY] - weight_[SPARSE];
        weight_[PLAY]             = std::max(weight_[PLAY], play_floor * others);
        weight_[SPARSE]           = std::max(weight_[SPARSE], sparse_floor * others);
    }

    Source pick_source() {
        if (++weight_age_ >= 256) {
            refresh_weights();
            weight_age_ = 0;
        }
        double total = 0;
        for (const double w : weight_) {
            total += w;
        }
        double pick = rng_.uniform() * total;
        for (int s = 0; s < NUM_SOURCES; ++s) {
            if (pick < weight_[s]) {
                return static_cast<Source>(s);
            }
            pick -= weight_[s];
        }
        return PLAY;
    }

    void report(std::uint64_t total) const {
        std::cerr << "[" << total << " examples] kinds:";
        for (int k = 0; k < NUM_KINDS; ++k) {
            std::cerr << " " << int(100 * stats_.kind[k] / cfg_.kind_target) << "%";
        }
        std::cerr << "\n";
    }

    // ------------------------------------------------------------------ one iteration
    bool build(Source src) {
        ep_file_ = -1;
        switch (src) {
            case SPARSE:
                return build_sparse();
            case CASTLE:
                return build_castle();
            case EP:
                return build_ep();
            case PROMO:
                return build_promo();
            case PIN:
                return build_pin();
            default:
                return build_terminal();
        }
    }

    bool step(Source src, std::vector<Example>& out) {
        if (src == PLAY) {
            if (!game_active_ && !new_game()) {
                return false;
            }
            an_.analyze_legal(play_);
            const bool emitted = consider(play_, play_ep_, out);
            advance_game();
            return emitted;
        }
        if (!build(src)) {
            return false;
        }
        an_.analyze_legal(scratch_);
        return consider(scratch_, ep_file_, out);
    }

    // an_.analyze_legal(b) has been run. Emits at most one example.
    bool consider(const chess::Board& b, int ep_file, std::vector<Example>& out) {
        if (!an_.trusted) {
            return false;
        }
        if (split_of(b.hash()) != cfg_.split) {
            ++stats_.rejected_split;
            return false;
        }

        // Candidate kinds: available here and still open, the emptiest rows first.
        std::array<std::pair<double, int>, NUM_KINDS> kinds;
        int nk = 0;
        for (int k = 0; k < NUM_KINDS; ++k) {
            if (!kind_open(k) || !an_.kind_available(k)) {
                continue;
            }
            kinds[nk++] = {-double(stats_.kind[k]) / double(cfg_.kind_target) + rng_.uniform() * 1e-6, k};
        }
        if (nk == 0) {
            ++stats_.rejected_no_kind;
            return false;
        }
        std::sort(kinds.begin(), kinds.begin() + nk);
        an_.analyze_pseudo(b);

        for (int i = 0; i < nk; ++i) {
            const int kind = kinds[i].second;

            // Reasons whose cell in this row is open, the emptiest cells first.
            std::array<std::pair<double, int>, NUM_REASONS> reasons;
            int nr = 0;
            for (int r = 0; r < NUM_REASONS; ++r) {
                if (!cell_open(kind, r)) {
                    continue;
                }
                const double left = double(quota_[kind][r] - stats_.joint[kind][r]) / double(quota_[kind][r]);
                reasons[nr++]     = {-left + rng_.uniform() * 1e-6, r};
            }
            std::sort(reasons.begin(), reasons.begin() + nr);
            for (int j = 0; j < nr; ++j) {
                const int reason = reasons[j].second;
                const auto c     = propose(reason, b);
                if (!c) {
                    continue;
                }
                if (classify(b, an_, *c) != reason) {
                    std::cerr << "internal error: proposal for " << reason_name(reason) << " classified as "
                              << reason_name(classify(b, an_, *c)) << "\n";
                    std::abort();
                }
                Code legal = RESIGN_CODE;
                if (kind != RESIGN) {
                    const auto& moves = an_.by_kind[kind];
                    legal             = encode(moves[rng_.below(static_cast<std::uint32_t>(moves.size()))]);
                }
                out.push_back(make_example(b, ep_file, legal, *c, kind, reason));
                ++stats_.kind[kind];
                ++stats_.reason[reason];
                ++stats_.joint[kind][reason];
                return true;
            }
        }
        ++stats_.rejected_no_illegal;
        return false;
    }

    static Example make_example(const chess::Board& b, int ep_file, Code legal, Code illegal, int kind, int reason) {
        using chess::Color;
        using chess::PieceType;
        Example e;
        std::memset(&e, 0, sizeof(e));
        e.bitboards[0] = b.us(Color::WHITE).getBits();
        e.bitboards[1] = b.us(Color::BLACK).getBits();
        e.bitboards[2] = b.pieces(PieceType::PAWN).getBits();
        e.bitboards[3] = b.pieces(PieceType::KNIGHT).getBits();
        e.bitboards[4] = b.pieces(PieceType::BISHOP).getBits();
        e.bitboards[5] = b.pieces(PieceType::ROOK).getBits();
        e.bitboards[6] = b.pieces(PieceType::QUEEN).getBits();
        e.bitboards[7] = b.pieces(PieceType::KING).getBits();

        const auto rights   = b.castlingRights();
        std::uint64_t rooks = 0;
        for (const Color c : {Color::WHITE, Color::BLACK}) {
            for (const auto side :
                 {chess::Board::CastlingRights::Side::KING_SIDE, chess::Board::CastlingRights::Side::QUEEN_SIDE}) {
                if (rights.has(c, side)) {
                    rooks |= 1ull << ((c == Color::WHITE ? 0 : 56) + static_cast<int>(rights.getRookFile(c, side)));
                }
            }
        }
        e.bitboards[8]   = rooks;
        e.legal_move     = legal;
        e.illegal_move   = illegal;
        e.ep_file        = ep_file < 0 ? 255 : static_cast<std::uint8_t>(ep_file);
        e.white_to_move  = b.sideToMove() == Color::WHITE;
        e.legal_kind     = static_cast<std::uint8_t>(kind);
        e.illegal_reason = static_cast<std::uint8_t>(reason);
        return e;
    }

    // ------------------------------------------------------------------ illegal move proposals
    int other_square(int from) {
        int s = static_cast<int>(rng_.below(63));
        return s >= from ? s + 1 : s;
    }

    int near_square(int from, int d) {
        for (int t = 0; t < 8; ++t) {
            const int dx = rng_.range(-d, d), dy = rng_.range(-d, d);
            const int f = (from & 7) + dx, r = (from >> 3) + dy;
            if ((dx || dy) && f >= 0 && f < 8 && r >= 0 && r < 8) {
                return r * 8 + f;
            }
        }
        return other_square(from);
    }

    std::optional<Code> propose(int r, const chess::Board& b) {
        using chess::PieceType;
        const chess::Color us   = b.sideToMove();
        const chess::Color them = ~us;
        const std::uint64_t occ = b.occ().getBits();
        const std::uint64_t own = b.us(us).getBits();
        const std::uint64_t opp = b.us(them).getBits();
        const int dir           = us == chess::Color::WHITE ? 8 : -8;
        const int start_rank    = us == chess::Color::WHITE ? 1 : 6;
        const int last_rank     = us == chess::Color::WHITE ? 7 : 0;
        constexpr int TRIES     = 16;

        auto accept = [&](Code c) -> std::optional<Code> {
            return classify(b, an_, c) == r ? std::optional<Code>(c) : std::nullopt;
        };
        auto type_at = [&](int sq) { return b.at<chess::Piece>(chess::Square(sq)).type(); };

        switch (r) {
            case SAME_SQUARE: {
                int s = static_cast<int>(rng_.below(63));
                if (s >= 1) {
                    ++s;  // never b1: b1->b1 is the resign code
                }
                return accept(make_code(s, s));
            }
            case EMPTY_FROM: {
                const std::uint64_t empty = ~occ;
                for (int t = 0; t < TRIES; ++t) {
                    const int from = pick_bit(empty, rng_);
                    const int to   = rng_.chance(0.5) ? other_square(from) : near_square(from, 3);
                    if (const auto c = accept(make_code(from, to))) {
                        return c;
                    }
                }
                return std::nullopt;
            }
            case OPPONENT_PIECE: {
                if (!opp) {
                    return std::nullopt;
                }
                for (int t = 0; t < TRIES; ++t) {
                    const int from = pick_bit(opp, rng_);
                    int to         = other_square(from);
                    if (rng_.chance(0.5)) {
                        const std::uint64_t pat =
                            attack_pattern(type_at(from), them, from, chess::Bitboard(occ)).getBits() & ~opp;
                        if (pat) {
                            to = pick_bit(pat, rng_);
                        }
                    }
                    if (const auto c = accept(make_code(from, to))) {
                        return c;
                    }
                }
                return std::nullopt;
            }
            case OWN_CAPTURE: {
                for (int t = 0; t < TRIES; ++t) {
                    const int from = pick_bit(own, rng_);
                    const std::uint64_t pat =
                        attack_pattern(type_at(from), us, from, chess::Bitboard(occ)).getBits() & own;
                    if (!pat) {
                        continue;
                    }
                    if (const auto c = accept(make_code(from, pick_bit(pat, rng_)))) {
                        return c;
                    }
                }
                return std::nullopt;
            }
            case GEOMETRY: {
                for (int t = 0; t < TRIES; ++t) {
                    const int from = pick_bit(own, rng_);
                    const int to   = rng_.chance(0.5) ? other_square(from) : near_square(from, 2);
                    if (const auto c = accept(make_code(from, to))) {
                        return c;
                    }
                }
                return std::nullopt;
            }
            case BLOCKED: {
                std::array<std::pair<int, std::uint64_t>, 32> cand;
                int n = 0;
                for (const PieceType pt :
                     {PieceType(PieceType::BISHOP), PieceType(PieceType::ROOK), PieceType(PieceType::QUEEN)}) {
                    std::uint64_t bb = b.pieces(pt, us).getBits();
                    while (bb && n < 32) {
                        const int sq = __builtin_ctzll(bb);
                        bb &= bb - 1;
                        const std::uint64_t blocked = attack_pattern(pt, us, sq, chess::Bitboard(0)).getBits() &
                                                      ~attack_pattern(pt, us, sq, chess::Bitboard(occ)).getBits() &
                                                      ~own;
                        if (blocked) {
                            cand[n++] = {sq, blocked};
                        }
                    }
                }
                if (!n) {
                    return std::nullopt;
                }
                const auto& [from, targets] = cand[rng_.below(n)];
                return accept(make_code(from, pick_bit(targets, rng_)));
            }
            case PAWN_BLOCKED: {
                std::array<Code, 64> cand;
                int n               = 0;
                std::uint64_t pawns = b.pieces(PieceType::PAWN, us).getBits();
                while (pawns && n < 60) {
                    const int from = __builtin_ctzll(pawns);
                    pawns &= pawns - 1;
                    const int f1 = from + dir, f2 = f1 + dir;
                    const bool o1 = (occ >> f1) & 1;
                    auto add      = [&](int to) {
                        const int promo = ((to >> 3) == last_rank && rng_.chance(0.5)) ? rng_.range(1, 4) : 0;
                        cand[n++]       = make_code(from, to, promo);
                    };
                    if (o1) {
                        add(f1);
                    }
                    if ((from >> 3) == start_rank && (o1 || ((occ >> f2) & 1))) {
                        add(f2);
                    }
                }
                if (!n) {
                    return std::nullopt;
                }
                return accept(cand[rng_.below(n)]);
            }
            case PAWN_RULE: {
                const std::uint64_t pawns = b.pieces(PieceType::PAWN, us).getBits();
                if (!pawns) {
                    return std::nullopt;
                }
                for (int t = 0; t < TRIES; ++t) {
                    const int from = pick_bit(pawns, rng_);
                    const int to   = near_square(from, 2);
                    if (const auto c = accept(make_code(from, to))) {
                        return c;
                    }
                }
                return std::nullopt;
            }
            case PROMOTION_RULE: {
                for (int t = 0; t < TRIES; ++t) {
                    Code c;
                    const auto& promos = an_.by_kind[PAWN_PROMOTION];
                    if (rng_.chance(0.4) && !promos.empty()) {
                        // A legal promotion without saying what the pawn becomes.
                        c = static_cast<Code>(encode(promos[rng_.below(static_cast<std::uint32_t>(promos.size()))]) &
                                              0x0FFF);
                    } else {
                        if (an_.pseudo.size() == 0) {
                            return std::nullopt;
                        }
                        c = encode(an_.pseudo.m[rng_.below(static_cast<std::uint32_t>(an_.pseudo.size()))]);
                        if (code_promo(c) != 0) {
                            c = static_cast<Code>(c & 0x0FFF);
                        } else {
                            c = static_cast<Code>(c | (rng_.range(1, 4) << 12));  // a promotion piece nobody asked for
                        }
                    }
                    if (const auto a = accept(c)) {
                        return a;
                    }
                }
                return std::nullopt;
            }
            case CASTLING_RULE: {
                const std::uint64_t rooks = b.pieces(PieceType::ROOK, us).getBits();
                if (!rooks) {
                    return std::nullopt;
                }
                const int king = b.kingSq(us).index();
                for (int t = 0; t < 8; ++t) {
                    if (const auto c = accept(make_code(king, pick_bit(rooks, rng_)))) {
                        return c;
                    }
                }
                return std::nullopt;
            }
            default: {
                const auto& pool = an_.ks[r - KS_FIRST];
                if (pool.empty()) {
                    return std::nullopt;
                }
                return pool[rng_.below(static_cast<std::uint32_t>(pool.size()))];
            }
        }
    }

    // ------------------------------------------------------------------ random play
    int pick_start() {
        switch (cfg_.variant) {
            case Variant::STANDARD:
                return 518;
            case Variant::CHESS960:
                return cfg_.starts[rng_.below(static_cast<std::uint32_t>(cfg_.starts.size()))];
            default:
                return rng_.chance(0.125) ? 518
                                          : cfg_.starts[rng_.below(static_cast<std::uint32_t>(cfg_.starts.size()))];
        }
    }

    static int double_push_file(const chess::Board& b, const chess::Move m) {
        if (b.at<chess::PieceType>(m.from()) == chess::PieceType::PAWN &&
            std::abs((m.to().index() >> 3) - (m.from().index() >> 3)) == 2) {
            return m.from().index() & 7;
        }
        return -1;
    }

    bool new_game() {
        const int start = pick_start();
        if (!play_.setFen(chess960_fen(start))) {
            return false;
        }
        play_ep_  = -1;
        play_ply_ = 0;
        // Random opening plies so that games from the same start do not share their first positions.
        const int warmup = start == 518 ? rng_.range(4, 10) : rng_.range(0, 6);
        chess::Movelist moves;
        for (int i = 0; i < warmup; ++i) {
            chess::movegen::legalmoves(moves, play_);
            if (moves.empty()) {
                return false;
            }
            const chess::Move m = moves[rng_.below(static_cast<std::uint32_t>(moves.size()))];
            play_ep_            = double_push_file(play_, m);
            play_.makeMove(m);
            ++play_ply_;
        }
        game_active_ = true;
        return true;
    }

    // an_ holds the legal moves of play_.
    void advance_game() {
        if (an_.no_moves || play_ply_ >= cfg_.max_plies || play_.isInsufficientMaterial()) {
            game_active_ = false;
            return;
        }
        const chess::Move m = an_.legal[rng_.below(static_cast<std::uint32_t>(an_.legal.size()))];
        play_ep_            = double_push_file(play_, m);
        play_.makeMove(m);
        ++play_ply_;
    }

    // ------------------------------------------------------------------ position builders
    bool place_random_piece(Grid& g, bool white) {
        static const char kinds[] = {'P', 'N', 'B', 'R', 'Q'};
        static const int caps[]   = {8, 4, 4, 4, 3};
        for (int t = 0; t < 10; ++t) {
            const int i = static_cast<int>(rng_.below(5));
            char ch     = kinds[i];
            if (!white) {
                ch = static_cast<char>(ch - 'A' + 'a');
            }
            if (g.count(ch) >= caps[i]) {
                continue;
            }
            for (int u = 0; u < 20; ++u) {
                const int sq = static_cast<int>(rng_.below(64));
                if (!g.empty(sq)) {
                    continue;
                }
                if (i == 0 && ((sq >> 3) == 0 || (sq >> 3) == 7)) {
                    continue;
                }
                g.put(sq, ch);
                return true;
            }
        }
        return false;
    }

    bool place_kings(Grid& g) {
        int wk = -1;
        for (int t = 0; t < 50 && wk < 0; ++t) {
            const int sq = static_cast<int>(rng_.below(64));
            if (g.empty(sq)) {
                wk = sq;
            }
        }
        if (wk < 0) {
            return false;
        }
        g.put(wk, 'K');
        for (int t = 0; t < 50; ++t) {
            const int bk = static_cast<int>(rng_.below(64));
            if (g.empty(bk) && chebyshev(wk, bk) >= 2) {
                g.put(bk, 'k');
                return true;
            }
        }
        return false;
    }

    // Loads the grid into scratch_ and checks that the side not to move is not in check.
    bool load(const Grid& g, bool white_to_move, const std::string& castling = "") {
        if (!scratch_.setFen(grid_fen(g, white_to_move, castling))) {
            return false;
        }
        const chess::Color stm = scratch_.sideToMove();
        if (scratch_.isAttacked(scratch_.kingSq(~stm), stm)) {
            return false;  // the side not to move is in check
        }
        return scratch_.checkers().count() <= 2;  // the library asserts on 3+ checkers
    }

    int random_extra_count() {
        const double r = rng_.uniform();
        return r < 0.55 ? rng_.range(0, 4) : (r < 0.9 ? rng_.range(5, 9) : rng_.range(10, 14));
    }

    bool build_sparse() {
        Grid g;
        if (!place_kings(g)) {
            return false;
        }
        const int n = random_extra_count();
        for (int i = 0; i < n; ++i) {
            place_random_piece(g, rng_.chance(0.5));
        }
        return load(g, rng_.chance(0.5));
    }

    bool build_castle() {
        const std::string fen = chess960_fen(pick_start());
        Grid g                = grid_from_placement(fen.substr(0, fen.find(' ')));

        std::array<std::array<int, 2>, 2> rook_file{};  // [color][0 = queen side, 1 = king side]
        for (int c = 0; c < 2; ++c) {
            const int base = c == 0 ? 0 : 56;
            int seen       = 0;
            for (int f = 0; f < 8; ++f) {
                if (g.c[base + f] == (c == 0 ? 'R' : 'r')) {
                    rook_file[c][seen++] = f;
                }
            }
        }

        for (int c = 0; c < 2; ++c) {
            const int base = c == 0 ? 0 : 56;
            const double p = 0.3 + 0.7 * rng_.uniform();
            for (int f = 0; f < 8; ++f) {
                const char ch = g.c[base + f];
                if (ch == 'K' || ch == 'k' || ch == 'R' || ch == 'r') {
                    continue;
                }
                if (rng_.chance(p)) {
                    g.remove(base + f);
                }
            }
            // pawns in front of the back rank
            for (int f = 0; f < 8; ++f) {
                const int sq = (c == 0 ? 8 : 48) + f;
                if (rng_.chance(0.3)) {
                    g.remove(sq);
                }
            }
        }
        bool has_rook[2][2] = {{true, true}, {true, true}};
        for (int c = 0; c < 2; ++c) {
            for (int s = 0; s < 2; ++s) {
                if (rng_.chance(0.08)) {
                    g.remove((c == 0 ? 0 : 56) + rook_file[c][s]);
                    has_rook[c][s] = false;
                }
            }
        }

        const int extras = rng_.range(0, 4);
        for (int i = 0; i < extras; ++i) {
            place_random_piece(g, rng_.chance(0.5));
        }
        if (rng_.chance(0.6)) {
            const int attackers = rng_.range(1, 2);
            for (int i = 0; i < attackers; ++i) {
                const bool white = rng_.chance(0.5);
                const char ch    = "RQB"[rng_.below(3)];
                for (int t = 0; t < 20; ++t) {
                    const int sq = rng_.range(1, 6) * 8 + static_cast<int>(rng_.below(8));
                    if (g.empty(sq)) {
                        g.put(sq, white ? ch : static_cast<char>(ch - 'A' + 'a'));
                        break;
                    }
                }
            }
        }

        std::string castling;
        for (int c = 0; c < 2; ++c) {
            for (int s = 0; s < 2; ++s) {
                if (has_rook[c][s] && rng_.chance(0.9)) {
                    castling += static_cast<char>((c == 0 ? 'A' : 'a') + rook_file[c][s]);
                }
            }
        }
        return load(g, rng_.chance(0.5), castling);
    }

    bool build_promo() {
        Grid g;
        const bool white = rng_.chance(0.5);
        const int rank7  = white ? 6 : 1;
        const int last   = white ? 7 : 0;
        const int pawns  = rng_.range(1, 2);
        for (int i = 0; i < pawns; ++i) {
            const int f = static_cast<int>(rng_.below(8));
            if (g.empty(rank7 * 8 + f)) {
                g.put(rank7 * 8 + f, white ? 'P' : 'p');
            }
            if (rng_.chance(0.5)) {  // something to capture on the promotion rank
                const int tf = f + (rng_.chance(0.5) ? 1 : -1);
                if (tf >= 0 && tf < 8 && g.empty(last * 8 + tf)) {
                    const char ch = "NBRQ"[rng_.below(4)];
                    g.put(last * 8 + tf, white ? static_cast<char>(ch - 'A' + 'a') : ch);
                }
            }
        }
        // kings must not stand on a square we already used
        if (!place_kings(g)) {
            return false;
        }
        const int n = rng_.range(0, 6);
        for (int i = 0; i < n; ++i) {
            place_random_piece(g, rng_.chance(0.5));
        }
        return load(g, white);
    }

    bool build_pin() {
        static const int dx[8] = {1, -1, 0, 0, 1, 1, -1, -1};
        static const int dy[8] = {0, 0, 1, -1, 1, -1, 1, -1};
        Grid g;
        const bool white = rng_.chance(0.5);
        const int king   = static_cast<int>(rng_.below(64));
        g.put(king, white ? 'K' : 'k');

        const int d = static_cast<int>(rng_.below(8));
        std::array<int, 7> ray;
        int len = 0;
        for (int s = 1; s < 8; ++s) {
            const int f = (king & 7) + dx[d] * s, r = (king >> 3) + dy[d] * s;
            if (f < 0 || f > 7 || r < 0 || r > 7) {
                break;
            }
            ray[len++] = r * 8 + f;
        }
        if (len < 2) {
            return false;
        }
        const int i = rng_.range(0, len - 2);
        const int j = rng_.range(i + 1, len - 1);

        static const char own_types[] = {'P', 'N', 'B', 'R', 'Q'};
        const char own_ch             = own_types[rng_.below(5)];
        if (own_ch == 'P' && ((ray[i] >> 3) == 0 || (ray[i] >> 3) == 7)) {
            return false;
        }
        g.put(ray[i], white ? own_ch : static_cast<char>(own_ch - 'A' + 'a'));

        const bool orthogonal = d < 4;
        const char pinner     = rng_.chance(0.3) ? 'Q' : (orthogonal ? 'R' : 'B');
        g.put(ray[j], white ? static_cast<char>(pinner - 'A' + 'a') : pinner);

        // the opposing king
        bool placed = false;
        for (int t = 0; t < 50 && !placed; ++t) {
            const int sq = static_cast<int>(rng_.below(64));
            if (g.empty(sq) && chebyshev(sq, king) >= 2) {
                g.put(sq, white ? 'k' : 'K');
                placed = true;
            }
        }
        if (!placed) {
            return false;
        }
        const int n = rng_.range(0, 5);
        for (int k = 0; k < n; ++k) {
            place_random_piece(g, rng_.chance(0.5));
        }
        return load(g, white);
    }

    bool build_ep() {
        Grid g;
        const bool white_push = rng_.chance(0.5);
        const int f           = static_cast<int>(rng_.below(8));
        const int start       = white_push ? 1 : 6;
        const int mid         = white_push ? 2 : 5;
        const int dest        = white_push ? 3 : 4;
        const char pusher     = white_push ? 'P' : 'p';
        const char enemy      = white_push ? 'p' : 'P';

        g.put(start * 8 + f, pusher);
        g.put(mid * 8 + f, '*');  // must stay empty
        g.put(dest * 8 + f, '*');

        const bool disc = rng_.chance(0.4);  // arrange a discovered check along the rank
        int fe          = f + (rng_.chance(0.5) ? 1 : -1);
        if (fe < 0 || fe > 7) {
            fe = f + (f == 0 ? 1 : -1);
        }
        g.put(dest * 8 + fe, enemy);
        if (!disc && rng_.chance(0.3)) {
            const int other = 2 * f - fe;
            if (other >= 0 && other < 8) {
                g.put(dest * 8 + other, enemy);
            }
        }

        bool kings_placed = false;
        if (disc) {
            const int lo = std::min(f, fe), hi = std::max(f, fe);
            const bool king_left = rng_.chance(0.5);
            const int kf = king_left ? rng_.range(0, lo - 1 < 0 ? 0 : lo - 1) : rng_.range(hi + 1 > 7 ? 7 : hi + 1, 7);
            const int sf = king_left ? rng_.range(hi + 1 > 7 ? 7 : hi + 1, 7) : rng_.range(0, lo - 1 < 0 ? 0 : lo - 1);
            const bool feasible = king_left ? (lo >= 1 && hi <= 6) : (lo >= 1 && hi <= 6);
            if (feasible) {
                // everything between king, pawns and slider on this rank must be empty
                const int a = std::min(kf, sf), z = std::max(kf, sf);
                bool clear = true;
                for (int x = a; x <= z; ++x) {
                    const int sq = dest * 8 + x;
                    if (x == kf || x == sf) {
                        continue;
                    }
                    if (x == f || x == fe) {
                        continue;
                    }
                    if (!g.empty(sq)) {
                        clear = false;
                    } else {
                        g.put(sq, '*');
                    }
                }
                if (clear && g.empty(dest * 8 + kf) && g.empty(dest * 8 + sf)) {
                    g.put(dest * 8 + kf, white_push ? 'k' : 'K');
                    g.put(dest * 8 + sf, white_push ? (rng_.chance(0.5) ? 'R' : 'Q') : (rng_.chance(0.5) ? 'r' : 'q'));
                    // the pusher's own king somewhere else
                    for (int t = 0; t < 50 && !kings_placed; ++t) {
                        const int sq = static_cast<int>(rng_.below(64));
                        if (g.empty(sq) && chebyshev(sq, dest * 8 + kf) >= 2) {
                            g.put(sq, white_push ? 'K' : 'k');
                            kings_placed = true;
                        }
                    }
                }
            }
            if (!kings_placed) {
                return false;
            }
        } else if (!place_kings(g)) {
            return false;
        }

        const int n = rng_.range(0, 6);
        for (int i = 0; i < n; ++i) {
            place_random_piece(g, rng_.chance(0.5));
        }
        for (char& ch : g.c) {
            if (ch == '*') {
                ch = 0;
            }
        }
        g.occ = 0;
        for (int sq = 0; sq < 64; ++sq) {
            if (g.c[sq]) {
                g.occ |= 1ull << sq;
            }
        }

        if (!load(g, white_push)) {
            return false;
        }
        const Code push = make_code(start * 8 + f, dest * 8 + f);
        chess::Movelist legal;
        chess::movegen::legalmoves(legal, scratch_);
        for (const chess::Move m : legal) {
            if (encode(m) == push) {
                scratch_.makeMove(m);  // the library records the en passant square: an enemy pawn is adjacent
                ep_file_ = f;
                return true;
            }
        }
        return false;
    }

    // A position in which the side to move has no legal move: a position with a heavily armed side plus one
    // mating (or stalemating) move for it. Mates and stalemates live near the edges, so the defending king
    // is often placed there and the attacking king close by.
    bool build_terminal() {
        Grid g;
        const bool white = rng_.chance(0.5);  // the side that delivers
        auto lower       = [](char ch) { return static_cast<char>(ch - 'A' + 'a'); };

        int dk = static_cast<int>(rng_.below(64));
        if (rng_.chance(terminal_edge_p)) {
            const int e = static_cast<int>(rng_.below(28));  // the 28 edge squares
            const int t = e % 7;
            dk          = e < 7 ? t : (e < 14 ? 56 + t : (e < 21 ? (t + 1) * 8 : (t + 1) * 8 + 7));
        }
        g.put(dk, white ? 'k' : 'K');

        int ak = -1;
        for (int t = 0; t < 50 && ak < 0; ++t) {
            const int sq = rng_.chance(0.6) ? near_square(dk, 3) : static_cast<int>(rng_.below(64));
            if (g.empty(sq) && chebyshev(sq, dk) >= 2) {
                ak = sq;
            }
        }
        if (ak < 0) {
            return false;
        }
        g.put(ak, white ? 'K' : 'k');

        const int heavy = rng_.range(1, 2);
        for (int i = 0; i < heavy; ++i) {
            const char ch = "QQRRBN"[rng_.below(6)];
            for (int t = 0; t < 20; ++t) {
                const int sq = static_cast<int>(rng_.below(64));
                if (g.empty(sq)) {
                    g.put(sq, white ? ch : lower(ch));
                    break;
                }
            }
        }
        const int more = rng_.range(0, 3);
        for (int i = 0; i < more; ++i) {
            place_random_piece(g, rng_.chance(terminal_defender_p) ? !white : white);
        }
        if (!load(g, white)) {
            return false;
        }

        chess::Movelist moves, replies;
        chess::movegen::legalmoves(moves, scratch_);
        std::array<chess::Move, 64> terminal;
        int count = 0;
        for (const chess::Move m : moves) {
            scratch_.makeMove(m);
            chess::movegen::legalmoves(replies, scratch_);
            const bool none = replies.empty();
            scratch_.unmakeMove(m);
            if (none && count < 64) {
                terminal[count++] = m;
            }
        }
        if (!count) {
            return false;
        }
        const chess::Move m = terminal[rng_.below(count)];
        ep_file_            = double_push_file(scratch_, m);
        scratch_.makeMove(m);
        return true;
    }

    double terminal_edge_p     = 0.75;
    double terminal_defender_p = 0.45;
};

}  // namespace nw
