#include "bpe.h"
#include "absl/log/log.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <omp.h>
#include <string>
#include <unordered_map>
#include <vector>
#ifdef BPE_HAVE_MPI
#include <mpi.h>
#endif

namespace bpe {
namespace {

std::int64_t elapsed_ms(const std::chrono::steady_clock::time_point& start,
                        const std::chrono::steady_clock::time_point& end) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
        .count();
}

inline bool is_ws(Byte b) {
    return b == 0x20 || b == 0x09 || b == 0x0A || b == 0x0D;
}

#ifdef BPE_HAVE_MPI
// Serialize word_counts into a flat byte buffer.
// Per entry: [uint32 word_len][word bytes][uint64 count]
std::vector<Byte> pack_word_counts(const std::vector<WordCount>& wcs) {
    std::vector<Byte> buf;
    buf.reserve(wcs.size() * 16);
    for (const WordCount& wc : wcs) {
        uint32_t len = static_cast<uint32_t>(wc.word.size());
        buf.insert(buf.end(), reinterpret_cast<const Byte*>(&len),
                   reinterpret_cast<const Byte*>(&len) + 4);
        buf.insert(buf.end(), wc.word.begin(), wc.word.end());
        uint64_t cnt = static_cast<uint64_t>(wc.count);
        buf.insert(buf.end(), reinterpret_cast<const Byte*>(&cnt),
                   reinterpret_cast<const Byte*>(&cnt) + 8);
    }
    return buf;
}
#endif

// Write per-rank step timings to timing-<job>.txt.
// local_times layout (one row per rank, NSTEPS columns):
//   [scatter, split_words, word_count, allgatherv, merge, sort, allgatherv2, build_results]
void write_timing(int rank, int nranks,
                  const std::vector<std::int64_t>& local_times) {
    constexpr int NSTEPS = 8;
    static const char* NAMES[NSTEPS] = {
        "scatter      ", "split_words  ", "word_count   ", "allgatherv   ",
        "merge        ", "sort         ", "allgatherv2  ", "build_results"};

#ifdef BPE_HAVE_MPI
    std::vector<std::int64_t> all_times(nranks * NSTEPS);
    MPI_Gather(local_times.data(), NSTEPS, MPI_INT64_T,
               all_times.data(), NSTEPS, MPI_INT64_T,
               0, MPI_COMM_WORLD);
#else
    (void)rank;
    (void)nranks;
    const std::vector<std::int64_t>& all_times = local_times;
#endif

    if (rank == 0) {
        const char* job_id = std::getenv("SLURM_JOB_ID");
        std::string fname =
            std::string("timing-") + (job_id ? job_id : "local") + ".txt";
        std::ofstream f(fname, std::ios::app);
        for (int r = 0; r < nranks; ++r) {
            f << "=== rank " << r << " of " << nranks << " ===\n";
            std::int64_t total = 0;
            for (int s = 0; s < NSTEPS; ++s) {
                std::int64_t t = all_times[r * NSTEPS + s];
                f << "  " << NAMES[s] << ": " << t << " ms\n";
                total += t;
            }
            f << "  total        : " << total << " ms\n\n";
        }
    }
}

}  // namespace

void parallel_task1(std::vector<Byte>& input, Results& results) {
#ifdef BPE_HAVE_MPI
    const int rank   = mpi_rank();
    const int nranks = mpi_size();
#else
    const int rank   = 0;
    const int nranks = 1;
#endif

    // per-step wall times: scatter/split/count/allgather/merge/sort/allgather2/build
    std::vector<std::int64_t> step_ms(8, 0);

    // -------------------------------------------------------------------------
    // Step 1: scatter corpus shards (word-boundary aligned)
    // -------------------------------------------------------------------------
    auto tp = std::chrono::steady_clock::now();

#ifdef BPE_HAVE_MPI
    std::vector<int> snd_counts(nranks, 0);
    std::vector<int> snd_displs(nranks, 0);

    if (rank == 0) {
        const std::size_t total = input.size();
        std::size_t pos = 0;
        for (int r = 0; r < nranks; ++r) {
            snd_displs[r] = static_cast<int>(pos);
            std::size_t ideal =
                total * static_cast<std::size_t>(r + 1) / nranks;
            std::size_t end = (r < nranks - 1) ? ideal : total;
            while (end < total && !is_ws(input[end])) ++end;
            snd_counts[r] = static_cast<int>(end - pos);
            pos = end;
        }
        snd_counts[nranks - 1] =
            static_cast<int>(total - snd_displs[nranks - 1]);
    }

    MPI_Bcast(snd_counts.data(), nranks, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(snd_displs.data(), nranks, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<Byte> local_buf(snd_counts[rank]);
    MPI_Scatterv(rank == 0 ? input.data() : nullptr,
                 snd_counts.data(), snd_displs.data(), MPI_BYTE,
                 local_buf.data(), snd_counts[rank], MPI_BYTE,
                 0, MPI_COMM_WORLD);
#else
    std::vector<Byte> local_buf = std::move(input);
#endif

    step_ms[0] = elapsed_ms(tp, std::chrono::steady_clock::now());

    // -------------------------------------------------------------------------
    // Step 2: split_words
    // -------------------------------------------------------------------------
    tp = std::chrono::steady_clock::now();
    const std::vector<Word> words = split_words(local_buf);
    step_ms[1] = elapsed_ms(tp, std::chrono::steady_clock::now());
    LOG(INFO) << "split words: " << step_ms[1] << " ms";

    // -------------------------------------------------------------------------
    // Step 3: parallel word counting (OpenMP per-thread hash maps)
    // -------------------------------------------------------------------------
    tp = std::chrono::steady_clock::now();

    int nthreads = 1;
#pragma omp parallel
    {
#pragma omp single
        nthreads = omp_get_num_threads();
    }

    std::vector<std::unordered_map<std::string, std::size_t>> tmaps(nthreads);

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < words.size(); ++i) {
        tmaps[omp_get_thread_num()]
            [reinterpret_cast<const char*>(words[i].bytes)]++;
    }

    std::unordered_map<std::string, std::size_t> local_map;
    for (auto& tm : tmaps) {
        for (auto& [k, v] : tm) local_map[k] += v;
    }

    std::vector<WordCount> local_wcs;
    local_wcs.reserve(local_map.size());
    for (const auto& [word, count] : local_map) {
        local_wcs.push_back(
            WordCount{std::vector<Byte>(word.begin(), word.end()), count});
    }
    std::sort(local_wcs.begin(), local_wcs.end(),
              [](const WordCount& a, const WordCount& b) {
                  return std::lexicographical_compare(a.word.begin(),
                                                      a.word.end(),
                                                      b.word.begin(),
                                                      b.word.end());
              });

    step_ms[2] = elapsed_ms(tp, std::chrono::steady_clock::now());

    // -------------------------------------------------------------------------
    // Step 4: allgatherv — share every rank's local word counts
    // -------------------------------------------------------------------------
    tp = std::chrono::steady_clock::now();

#ifdef BPE_HAVE_MPI
    std::vector<Byte> packed = pack_word_counts(local_wcs);
    int local_size = static_cast<int>(packed.size());

    std::vector<int> rcv_counts(nranks);
    MPI_Allgather(&local_size, 1, MPI_INT,
                  rcv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    std::vector<int> rcv_displs(nranks);
    rcv_displs[0] = 0;
    for (int r = 1; r < nranks; ++r)
        rcv_displs[r] = rcv_displs[r - 1] + rcv_counts[r - 1];
    const int total_rcv = rcv_displs[nranks - 1] + rcv_counts[nranks - 1];

    std::vector<Byte> all_packed(total_rcv);
    MPI_Allgatherv(packed.data(), local_size, MPI_BYTE,
                   all_packed.data(), rcv_counts.data(), rcv_displs.data(),
                   MPI_BYTE, MPI_COMM_WORLD);

    step_ms[3] = elapsed_ms(tp, std::chrono::steady_clock::now());

    // -------------------------------------------------------------------------
    // Step 5: parallel merge+sort — each rank owns a disjoint byte-range slice
    // of the global vocabulary, so the per-rank sort is O(V/N log(V/N)) and
    // concatenating in rank order yields a globally sorted list.
    // -------------------------------------------------------------------------
    tp = std::chrono::steady_clock::now();

    // Split 52 English letters (A-Z:65-90, a-z:97-122) evenly across ranks.
    // Non-letter bytes fall into whichever contiguous range covers them.
    // The Z/a gap (91-96) stays with Z's rank (boundary pushed to 97).
    const int total_letters    = 52;
    const int letters_per_rank = (total_letters + nranks - 1) / nranks;

    auto letter_to_byte = [](int idx) -> int {
        return idx < 26 ? 65 + idx : 97 + (idx - 26);
    };

    std::vector<int> boundary(nranks + 1);
    boundary[0] = 0;
    for (int r = 1; r < nranks; ++r) {
        int last_letter = std::min(r * letters_per_rank, total_letters) - 1;
        int last_byte   = letter_to_byte(last_letter);
        boundary[r]     = (last_byte == 90) ? 97 : last_byte + 1;
    }
    boundary[nranks] = 256;

    const int range_start = boundary[rank];
    const int range_end   = boundary[rank + 1];

    // Merge only entries whose first byte falls in [range_start, range_end).
    std::unordered_map<std::string, std::size_t> partial_map;
    for (int r = 0; r < nranks; ++r) {
        const Byte* p  = all_packed.data() + rcv_displs[r];
        const Byte* ep = p + rcv_counts[r];
        while (p < ep) {
            uint32_t len;
            std::memcpy(&len, p, 4);
            p += 4;
            const unsigned char first = static_cast<unsigned char>(p[0]);
            const char* wptr = reinterpret_cast<const char*>(p);
            p += len;
            uint64_t cnt;
            std::memcpy(&cnt, p, 8);
            p += 8;
            if (first >= range_start && first < range_end)
                partial_map[std::string(wptr, len)] += static_cast<std::size_t>(cnt);
        }
    }

    step_ms[4] = elapsed_ms(tp, std::chrono::steady_clock::now());
    tp = std::chrono::steady_clock::now();

    // Sort this rank's slice (size ≈ V/N).
    std::vector<std::pair<std::string, std::size_t>> sorted_partial(
        partial_map.begin(), partial_map.end());
    std::sort(sorted_partial.begin(), sorted_partial.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    step_ms[5] = elapsed_ms(tp, std::chrono::steady_clock::now());
    tp = std::chrono::steady_clock::now();

    // -------------------------------------------------------------------------
    // Step 6: allgatherv sorted slices → concatenate in rank order = globally
    //         sorted (no further sort needed because ranges are disjoint).
    // -------------------------------------------------------------------------
    // Pack this rank's sorted slice.
    std::vector<Byte> packed_partial;
    packed_partial.reserve(sorted_partial.size() * 16);
    for (const auto& [word, count] : sorted_partial) {
        uint32_t wlen = static_cast<uint32_t>(word.size());
        packed_partial.insert(packed_partial.end(),
                              reinterpret_cast<const Byte*>(&wlen),
                              reinterpret_cast<const Byte*>(&wlen) + 4);
        packed_partial.insert(packed_partial.end(),
                              reinterpret_cast<const Byte*>(word.data()),
                              reinterpret_cast<const Byte*>(word.data()) + word.size());
        uint64_t cnt = static_cast<uint64_t>(count);
        packed_partial.insert(packed_partial.end(),
                              reinterpret_cast<const Byte*>(&cnt),
                              reinterpret_cast<const Byte*>(&cnt) + 8);
    }

    int part_size = static_cast<int>(packed_partial.size());
    std::vector<int> part_counts(nranks), part_displs(nranks);
    MPI_Allgather(&part_size, 1, MPI_INT,
                  part_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    part_displs[0] = 0;
    for (int r = 1; r < nranks; ++r)
        part_displs[r] = part_displs[r - 1] + part_counts[r - 1];
    const int total_part = part_displs[nranks - 1] + part_counts[nranks - 1];

    std::vector<Byte> all_sorted(total_part);
    MPI_Allgatherv(packed_partial.data(), part_size, MPI_BYTE,
                   all_sorted.data(), part_counts.data(), part_displs.data(),
                   MPI_BYTE, MPI_COMM_WORLD);

    // Unpack in rank order → globally sorted list.
    std::vector<std::pair<std::string, std::size_t>> sorted_global;
    sorted_global.reserve(partial_map.size() * nranks);
    for (int r = 0; r < nranks; ++r) {
        const Byte* p  = all_sorted.data() + part_displs[r];
        const Byte* ep = p + part_counts[r];
        while (p < ep) {
            uint32_t wlen;
            std::memcpy(&wlen, p, 4); p += 4;
            std::string word(reinterpret_cast<const char*>(p), wlen); p += wlen;
            uint64_t cnt;
            std::memcpy(&cnt, p, 8); p += 8;
            sorted_global.push_back({std::move(word), static_cast<std::size_t>(cnt)});
        }
    }

    step_ms[6] = elapsed_ms(tp, std::chrono::steady_clock::now());

    // -------------------------------------------------------------------------
    // Step 7: build results
    // -------------------------------------------------------------------------
    tp = std::chrono::steady_clock::now();

    // all ranks get full char_splits (Task 2 input)
    results.char_splits.clear();
    results.char_splits.reserve(sorted_global.size());
    if (rank == 0) {
        results.word_counts.clear();
        results.word_counts.reserve(sorted_global.size());
    }
    for (const auto& [word, count] : sorted_global) {
        std::vector<Byte> bytes(word.begin(), word.end());
        results.char_splits.push_back(CharSplit{bytes, count});
        if (rank == 0) {
            results.word_counts.push_back(WordCount{std::move(bytes), count});
        }
    }

#else
    // no-MPI: single-rank path, local_wcs is already the global result
    step_ms[3] = 0;
    step_ms[6] = 0;  // no second allgatherv

    tp = std::chrono::steady_clock::now();

    std::vector<std::pair<std::string, std::size_t>> sorted_global;
    sorted_global.reserve(local_wcs.size());
    for (const WordCount& wc : local_wcs) {
        sorted_global.push_back(
            {std::string(wc.word.begin(), wc.word.end()), wc.count});
    }

    step_ms[4] = elapsed_ms(tp, std::chrono::steady_clock::now());

    tp = std::chrono::steady_clock::now();

    results.word_counts.clear();
    results.char_splits.clear();
    results.word_counts.reserve(sorted_global.size());
    results.char_splits.reserve(sorted_global.size());
    for (const auto& [word, count] : sorted_global) {
        std::vector<Byte> bytes(word.begin(), word.end());
        results.word_counts.push_back(WordCount{bytes, count});
        results.char_splits.push_back(CharSplit{std::move(bytes), count});
    }
#endif

    step_ms[7] = elapsed_ms(tp, std::chrono::steady_clock::now());

    write_timing(rank, nranks, step_ms);
}

}  // namespace bpe
