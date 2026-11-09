// chess-rules: generates the bucket-balanced legal / illegal move dataset.
//
//   chess-rules --out train.bin --samples-per-kind 1000000 --split train --variant mixed --threads 8 --seed 1
//
// Writes <out> (80-byte records, see nw::Example) and <out>.json (layout, counts per bucket).

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

#include "chess-rules/generator.hpp"

namespace {

[[noreturn]] void usage(const char* msg = nullptr) {
    if (msg) {
        std::cerr << "error: " << msg << "\n\n";
    }
    std::cerr << "usage: chess-rules --out FILE --samples-per-kind N [options]\n"
                 "  --split train|val|test     which position split to generate (default train)\n"
                 "  --variant standard|960|mixed   start positions (default mixed)\n"
                 "  --threads T                independent shards (default: hardware threads)\n"
                 "  --seed S                   (default 1)\n"
                 "  --min-play-share X         minimum share of iterations on plain random play (default 0.25)\n"
                 "  --no-shuffle               keep generation order\n"
                 "  --verbose\n";
    std::exit(msg ? 1 : 0);
}

}  // namespace

int main(int argc, char** argv) {
    using namespace nw;

    std::filesystem::path out_path;
    std::uint64_t per_kind = 0, seed = 1;
    int split       = 0;
    Variant variant = Variant::MIXED;
    int threads     = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    bool shuffle = true, verbose = false;
    double play_share = 0.25;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next           = [&]() -> std::string {
            if (i + 1 >= argc) {
                usage("missing value");
            }
            return argv[++i];
        };
        if (a == "--out") {
            out_path = next();
        } else if (a == "--samples-per-kind") {
            per_kind = std::stoull(next());
        } else if (a == "--seed") {
            seed = std::stoull(next());
        } else if (a == "--threads") {
            threads = std::max(1, std::stoi(next()));
        } else if (a == "--split") {
            const std::string v = next();
            split               = v == "train" ? 0 : (v == "val" ? 1 : (v == "test" ? 2 : -1));
            if (split < 0) {
                usage("split must be train, val or test");
            }
        } else if (a == "--variant") {
            const std::string v = next();
            variant = v == "standard" ? Variant::STANDARD : (v == "960" ? Variant::CHESS960 : Variant::MIXED);
            if (v != "standard" && v != "960" && v != "mixed") {
                usage("variant must be standard, 960 or mixed");
            }
        } else if (a == "--min-play-share") {
            play_share = std::stod(next());
        } else if (a == "--no-shuffle") {
            shuffle = false;
        } else if (a == "--verbose") {
            verbose = true;
        } else if (a == "--help" || a == "-h") {
            usage();
        } else {
            usage(("unknown option " + a).c_str());
        }
    }
    if (out_path.empty() || per_kind == 0) {
        usage("--out and --samples-per-kind are required");
    }
    threads = static_cast<int>(std::min<std::uint64_t>(threads, per_kind));

    const auto starts = split_chess960_starts(0)[split];

    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::vector<Example>> parts(threads);
    std::vector<Stats> stats(threads);
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t) {
        pool.emplace_back([&, t] {
            Config cfg;
            cfg.kind_target = per_kind / threads + (static_cast<std::uint64_t>(t) < per_kind % threads ? 1 : 0);
            cfg.seed        = seed * 0x9E3779B97F4A7C15ull + static_cast<std::uint64_t>(t) * 0xD1B54A32D192ED03ull + 1;
            cfg.split       = split;
            cfg.variant     = variant;
            cfg.starts      = starts;
            cfg.min_play_share = play_share;
            cfg.verbose        = verbose && t == 0;
            Shard shard(cfg);
            shard.run(parts[t]);
            stats[t] = shard.stats();
        });
    }
    for (auto& th : pool) {
        th.join();
    }

    std::vector<Example> all;
    all.reserve(per_kind * NUM_KINDS);
    Stats total;
    for (int t = 0; t < threads; ++t) {
        all.insert(all.end(), parts[t].begin(), parts[t].end());
        total.merge(stats[t]);
        std::vector<Example>().swap(parts[t]);
    }
    if (shuffle) {
        Rng rng(seed ^ 0xABCDEF);
        for (std::size_t i = all.size() - 1; i > 0; --i) {
            std::swap(all[i], all[rng.below(static_cast<std::uint32_t>(i + 1))]);
        }
    }

    {
        std::ofstream f(out_path, std::ios::binary);
        if (!f) {
            usage("cannot open output file");
        }
        f.write(reinterpret_cast<const char*>(all.data()), static_cast<std::streamsize>(all.size() * sizeof(Example)));
    }

    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    std::filesystem::path json_path = out_path;
    json_path.replace_extension("json");
    std::ofstream j(json_path);
    j << "{\n  \"record_bytes\": 80,\n  \"examples\": " << all.size() << ",\n  \"seed\": " << seed
      << ",\n  \"split\": \""
      << (split == 0   ? "train"
          : split == 1 ? "val"
                       : "test")
      << "\",\n  \"variant\": \""
      << (variant == Variant::STANDARD   ? "standard"
          : variant == Variant::CHESS960 ? "960"
                                         : "mixed")
      << "\",\n  \"threads\": " << threads << ",\n  \"seconds\": " << secs << ",\n";
    j << "  \"layout\": {\n"
         "    \"bitboards\": \"uint64[9]: white, black, pawns, knights, bishops, rooks, queens, kings, castling rooks; "
         "bit i = square i, a1 = 0, h1 = 7, a8 = 56\",\n"
         "    \"legal_move, illegal_move\": \"uint16: to | from << 6 | promo << 12 (0 none, 1 N, 2 B, 3 R, 4 Q); "
         "castling is king "
         "takes own rook; resign is 65 (b1 b1)\",\n"
         "    \"ep_file\": \"uint8: file of the last double pawn push, 255 = none\",\n"
         "    \"white_to_move\": \"uint8\",\n"
         "    \"legal_kind, illegal_reason\": \"uint8 bucket ids, names below\"\n  },\n";
    j << "  \"kinds\": {";
    for (int k = 0; k < NUM_KINDS; ++k) {
        j << (k ? ", " : "") << "\"" << kind_name(k) << "\": " << total.kind[k];
    }
    j << "},\n  \"reasons\": {";
    for (int r = 0; r < NUM_REASONS; ++r) {
        j << (r ? ", " : "") << "\"" << reason_name(r) << "\": " << total.reason[r];
    }
    j << "},\n  \"kind_names\": [";
    for (int k = 0; k < NUM_KINDS; ++k) {
        j << (k ? ", " : "") << "\"" << kind_name(k) << "\"";
    }
    j << "],\n  \"reason_names\": [";
    for (int r = 0; r < NUM_REASONS; ++r) {
        j << (r ? ", " : "") << "\"" << reason_name(r) << "\"";
    }
    j << "],\n  \"examples_by_source\": {";
    for (int s = 0; s < Shard::NUM_SOURCES; ++s) {
        j << (s ? ", " : "") << "\"" << Shard::source_name(s) << "\": " << total.source_examples[s];
    }
    j << "}\n}\n";

    std::cerr << "wrote " << all.size() << " examples to " << out_path << " in " << secs << " s ("
              << static_cast<std::uint64_t>(all.size() / secs) << " examples/s)\n";
    return 0;
}
