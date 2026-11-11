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
#include <utility>
#include <vector>

#include "chess-rules/generator.hpp"
#include "chess-rules/rules.hpp"

[[noreturn]] void exit_with_message(const char* msg = nullptr) {
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
    std::exit(msg ? EXIT_FAILURE : EXIT_SUCCESS);
}

struct Args {
    std::filesystem::path out_path;
    std::uint64_t per_kind = 0;
    std::uint64_t seed     = 1;
    int split              = 0;
    nw::Variant variant    = nw::Variant::MIXED;
    int threads            = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    bool shuffle           = true;
    bool verbose           = false;
    double play_share      = 0.25;
};

Args parse_args(int argc, char** argv) {
    Args args;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];

        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                exit_with_message("missing value");
            }
            return argv[++i];
        };

        if (a == "--out") {
            args.out_path = next();
        } else if (a == "--samples-per-kind") {
            args.per_kind = std::stoull(next());
        } else if (a == "--seed") {
            args.seed = std::stoull(next());
        } else if (a == "--threads") {
            args.threads = std::max(1, std::stoi(next()));
        } else if (a == "--split") {
            const std::string v = next();
            args.split          = v == "train" ? 0 : (v == "val" ? 1 : (v == "test" ? 2 : -1));
            if (args.split < 0) {
                exit_with_message("split must be train, val or test");
            }
        } else if (a == "--variant") {
            const std::string v = next();
            args.variant =
                v == "standard" ? nw::Variant::STANDARD : (v == "960" ? nw::Variant::CHESS960 : nw::Variant::MIXED);
            if (v != "standard" && v != "960" && v != "mixed") {
                exit_with_message("variant must be standard, 960 or mixed");
            }
        } else if (a == "--min-play-share") {
            args.play_share = std::stod(next());
        } else if (a == "--no-shuffle") {
            args.shuffle = false;
        } else if (a == "--verbose") {
            args.verbose = true;
        } else if (a == "--help" || a == "-h") {
            exit_with_message();
        } else {
            exit_with_message(("unknown option " + a).c_str());
        }
    }

    if (args.out_path.empty() || args.per_kind == 0) {
        exit_with_message("--out and --samples-per-kind are required");
    }

    args.threads = static_cast<int>(std::min<std::uint64_t>(args.threads, args.per_kind));

    return args;
}

std::pair<std::vector<nw::Example>, nw::Stats> run(const Args& args) {
    const auto starts = split_chess960_starts(0)[args.split];

    std::vector<std::vector<nw::Example>> parts(args.threads);
    std::vector<nw::Stats> stats(args.threads);
    std::vector<std::thread> pool;
    for (int t = 0; t < args.threads; ++t) {
        pool.emplace_back([&, t] {
            nw::Config cfg;
            cfg.kind_target =
                args.per_kind / args.threads + (static_cast<std::uint64_t>(t) < args.per_kind % args.threads ? 1 : 0);
            cfg.seed    = args.seed * 0x9E3779B97F4A7C15ull + static_cast<std::uint64_t>(t) * 0xD1B54A32D192ED03ull + 1;
            cfg.split   = args.split;
            cfg.variant = args.variant;
            cfg.starts  = starts;
            cfg.min_play_share = args.play_share;
            cfg.verbose        = args.verbose && t == 0;
            nw::Shard shard(cfg);
            shard.run(parts[t]);
            stats[t] = shard.stats();
        });
    }

    for (auto& th : pool) {
        th.join();
    }

    std::vector<nw::Example> all;
    all.reserve(args.per_kind * nw::NUM_KINDS);
    nw::Stats total;
    for (int t = 0; t < args.threads; ++t) {
        all.insert(all.end(), parts[t].begin(), parts[t].end());
        total.merge(stats[t]);
        std::vector<nw::Example>().swap(parts[t]);
    }

    if (args.shuffle) {
        nw::Rng rng(args.seed ^ 0xABCDEF);
        for (std::size_t i = all.size() - 1; i > 0; --i) {
            std::swap(all[i], all[rng.below(static_cast<std::uint32_t>(i + 1))]);
        }
    }

    return {all, total};
}

void write_bin(const std::filesystem::path& path, const std::vector<nw::Example>& examples) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        exit_with_message("cannot open output file");
    }
    f.write(reinterpret_cast<const char*>(examples.data()),
            static_cast<std::streamsize>(examples.size() * sizeof(nw::Example)));
}

void write_json(const std::filesystem::path& path, const Args& args, const nw::Stats stats,
                const std::size_t num_examples, const double secs) {
    std::ofstream j(path);
    j << "{\n  \"record_bytes\": 80,\n  \"examples\": " << num_examples << ",\n  \"seed\": " << args.seed
      << ",\n  \"split\": \""
      << (args.split == 0   ? "train"
          : args.split == 1 ? "val"
                            : "test")
      << "\",\n  \"variant\": \""
      << (args.variant == nw::Variant::STANDARD   ? "standard"
          : args.variant == nw::Variant::CHESS960 ? "960"
                                                  : "mixed")
      << "\",\n  \"threads\": " << args.threads << ",\n  \"seconds\": " << secs << ",\n";
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
    for (int k = 0; k < nw::NUM_KINDS; ++k) {
        j << (k ? ", " : "") << "\"" << nw::kind_name(k) << "\": " << stats.kind[k];
    }
    j << "},\n  \"reasons\": {";
    for (int r = 0; r < nw::NUM_REASONS; ++r) {
        j << (r ? ", " : "") << "\"" << nw::reason_name(r) << "\": " << stats.reason[r];
    }
    j << "},\n  \"kind_names\": [";
    for (int k = 0; k < nw::NUM_KINDS; ++k) {
        j << (k ? ", " : "") << "\"" << nw::kind_name(k) << "\"";
    }
    j << "],\n  \"reason_names\": [";
    for (int r = 0; r < nw::NUM_REASONS; ++r) {
        j << (r ? ", " : "") << "\"" << nw::reason_name(r) << "\"";
    }
    j << "],\n  \"examples_by_source\": {";
    for (int s = 0; s < nw::Shard::NUM_SOURCES; ++s) {
        j << (s ? ", " : "") << "\"" << nw::Shard::source_name(s) << "\": " << stats.source_examples[s];
    }
    j << "}\n}\n";
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    const auto t0                = std::chrono::steady_clock::now();
    const auto [examples, stats] = run(args);
    write_bin(args.out_path, examples);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    std::filesystem::path json_path = args.out_path;
    json_path.replace_extension("json");
    write_json(json_path, args, stats, examples.size(), secs);

    std::cerr << "wrote " << examples.size() << " examples to " << args.out_path << " in " << secs << " s ("
              << static_cast<std::uint64_t>(examples.size() / secs) << " examples/s)\n";

    return EXIT_SUCCESS;
}
