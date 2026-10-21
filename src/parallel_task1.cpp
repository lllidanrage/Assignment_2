#include "bpe.h"
#include "absl/log/log.h"
#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <omp.h>
#include <stdexcept>
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
template <typename Prepare>
void prepare_mpi_buffers(Prepare prepare) {
    int failed = 0;
    try {
        prepare();
    } catch (const std::exception& error) {
        LOG(ERROR) << "task1 rank " << mpi_rank() << ": " << error.what();
        failed = 1;
    }
    int any_failed = 0;
    MPI_Allreduce(&failed, &any_failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (any_failed)
        throw std::runtime_error("task1: MPI buffer allocation or size limit failed");
}

void scatter_bytes(const Byte* input, Byte* output,
                   const std::vector<std::uint64_t>& offsets, int rank,
                   int nranks, std::uint64_t chunk_limit = INT_MAX) {
    if (rank == 0) {
        if (offsets[1] != 0)
            std::memcpy(output, input, static_cast<std::size_t>(offsets[1]));
        for (int r = 1; r < nranks; ++r) {
            for (auto offset = offsets[r]; offset < offsets[r + 1];) {
                const int count = static_cast<int>(
                    std::min(chunk_limit, offsets[r + 1] - offset));
                MPI_Send(input + offset, count, MPI_BYTE, r, 0, MPI_COMM_WORLD);
                offset += count;
            }
        }
    } else {
        const auto size = offsets[rank + 1] - offsets[rank];
        for (std::uint64_t offset = 0; offset < size;) {
            const int count = static_cast<int>(std::min(chunk_limit, size - offset));
            MPI_Recv(output + offset, count, MPI_BYTE, 0, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            offset += count;
        }
    }
}

void allgather_bytes(const std::vector<Byte>& local, std::vector<Byte>& gathered,
                     std::vector<std::uint64_t>& offsets, int nranks) {
    std::vector<std::uint64_t> sizes;
    std::vector<int> counts, displs;
    prepare_mpi_buffers([&] {
        sizes.resize(nranks);
        offsets.assign(static_cast<std::size_t>(nranks) + 1, 0);
        counts.resize(nranks);
        displs.resize(nranks);
    });
    const std::uint64_t local_size = local.size();
    MPI_Allgather(&local_size, 1, MPI_UINT64_T, sizes.data(), 1,
                  MPI_UINT64_T, MPI_COMM_WORLD);
    for (int r = 0; r < nranks; ++r) {
        if (sizes[r] > static_cast<std::uint64_t>(INT_MAX) - offsets[r])
            throw std::length_error(
                "task1: serialized word counts exceed MPI_Allgatherv INT_MAX bytes");
        offsets[r + 1] = offsets[r] + sizes[r];
        counts[r] = static_cast<int>(sizes[r]);
        displs[r] = static_cast<int>(offsets[r]);
    }
    prepare_mpi_buffers([&] {
        gathered.resize(static_cast<std::size_t>(offsets.back()));
    });
    MPI_Allgatherv(local.data(), static_cast<int>(local_size), MPI_BYTE,
                   gathered.data(), counts.data(), displs.data(), MPI_BYTE,
                   MPI_COMM_WORLD);
}

// Serialize word_counts into a flat byte buffer.
// Per entry: [uint32 word_len][word bytes][uint64 count]
std::vector<Byte> pack_word_counts(const std::vector<WordCount>& wcs) {
    std::vector<Byte> buf;
    buf.reserve(wcs.size() * 16);
    for (const WordCount& wc : wcs) {
        if (wc.word.size() > UINT32_MAX)
            throw std::length_error("task1 word exceeds serialized length limit");
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
    std::vector<std::int64_t> all_times;
    if (nranks > 1) {
        all_times.resize(nranks * NSTEPS);
        MPI_Gather(local_times.data(), NSTEPS, MPI_INT64_T,
                   all_times.data(), NSTEPS, MPI_INT64_T,
                   0, MPI_COMM_WORLD);
    } else {
        all_times = local_times;
    }
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

    std::vector<Byte> local_buf;
#ifdef BPE_HAVE_MPI
    if (nranks > 1) {
        std::vector<std::uint64_t> offsets;
        prepare_mpi_buffers([&] { offsets.resize(static_cast<std::size_t>(nranks) + 1); });

        if (rank == 0) {
            const std::size_t total = input.size();
            std::size_t pos = 0;
            for (int r = 0; r < nranks; ++r) {
                offsets[r] = pos;
                std::size_t ideal =
                    total / nranks * (r + 1) +
                    (total % nranks) * static_cast<std::uint64_t>(r + 1) / nranks;
                std::size_t end = std::max(pos, ideal);
                while (end < total && !is_ws(input[end])) ++end;
                pos = end;
            }
            offsets[nranks] = total;
        }

        MPI_Bcast(offsets.data(), nranks, MPI_UINT64_T, 0, MPI_COMM_WORLD);
        MPI_Bcast(&offsets[nranks], 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
        prepare_mpi_buffers([&] {
            const auto size = offsets[rank + 1] - offsets[rank];
            if (size > local_buf.max_size())
                throw std::length_error("task1 corpus shard too large");
            local_buf.resize(static_cast<std::size_t>(size));
        });
        scatter_bytes(input.data(), local_buf.data(), offsets, rank, nranks);
    } else {
        local_buf = std::move(input);
    }
#else
    local_buf = std::move(input);
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
    if (nranks > 1) {
    std::vector<Byte> packed;
    prepare_mpi_buffers([&] { packed = pack_word_counts(local_wcs); });
    std::vector<std::uint64_t> rcv_offsets;
    std::vector<Byte> all_packed;
    allgather_bytes(packed, all_packed, rcv_offsets, nranks);

    step_ms[3] = elapsed_ms(tp, std::chrono::steady_clock::now());

    // -------------------------------------------------------------------------
    // Step 5: parallel merge+sort — each rank owns a disjoint byte-range slice
    // of the global vocabulary, so the per-rank sort is O(V/N log(V/N)) and
    // concatenating in rank order yields a globally sorted list.
    // -------------------------------------------------------------------------
    tp = std::chrono::steady_clock::now();

    // Divide [0, 128) evenly across ranks; the last rank extends to 256
    // so that all non-ASCII bytes are covered.
    const int bytes_per_rank = (128 + nranks - 1) / nranks;
    const int range_start = rank * bytes_per_rank;
    const int range_end = (rank == nranks - 1)? 256 : range_start + bytes_per_rank;

    // Merge only entries whose first byte falls in [range_start, range_end).
    std::unordered_map<std::string, std::size_t> partial_map;
    for (int r = 0; r < nranks; ++r) {
        if (rcv_offsets[r] == rcv_offsets[r + 1]) continue;
        const Byte* p  = all_packed.data() + rcv_offsets[r];
        const Byte* ep = all_packed.data() + rcv_offsets[r + 1];
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
    prepare_mpi_buffers([&] {
        packed_partial.reserve(sorted_partial.size() * 16);
        for (const auto& [word, count] : sorted_partial) {
            if (word.size() > UINT32_MAX)
                throw std::length_error("task1 word exceeds serialized length limit");
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
    });

    std::vector<std::uint64_t> part_offsets;
    std::vector<Byte> all_sorted;
    allgather_bytes(packed_partial, all_sorted, part_offsets, nranks);

    // Unpack in rank order → globally sorted list.
    std::vector<std::pair<std::string, std::size_t>> sorted_global;
    sorted_global.reserve(partial_map.size() * nranks);
    for (int r = 0; r < nranks; ++r) {
        if (part_offsets[r] == part_offsets[r + 1]) continue;
        const Byte* p  = all_sorted.data() + part_offsets[r];
        const Byte* ep = all_sorted.data() + part_offsets[r + 1];
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

    } else {
        // nranks == 1 but compiled with MPI: skip all MPI collectives
        step_ms[3] = 0;
        step_ms[6] = 0;

        tp = std::chrono::steady_clock::now();

        std::vector<std::pair<std::string, std::size_t>> sg;
        sg.reserve(local_wcs.size());
        for (const WordCount& wc : local_wcs)
            sg.push_back({std::string(wc.word.begin(), wc.word.end()), wc.count});

        step_ms[4] = elapsed_ms(tp, std::chrono::steady_clock::now());

        tp = std::chrono::steady_clock::now();

        results.word_counts.clear();
        results.char_splits.clear();
        results.word_counts.reserve(sg.size());
        results.char_splits.reserve(sg.size());
        for (const auto& [word, count] : sg) {
            std::vector<Byte> bytes(word.begin(), word.end());
            results.word_counts.push_back(WordCount{bytes, count});
            results.char_splits.push_back(CharSplit{std::move(bytes), count});
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
