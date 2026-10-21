#include "bpe.h"
#include "absl/log/log.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#ifdef BPE_HAVE_MPI
#include <mpi.h>
#endif

namespace bpe {
namespace {

using u8  = std::uint8_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

constexpr u32 no_pos  = std::numeric_limits<u32>::max();
constexpr u32 kVocab  = 256;
constexpr std::size_t kBudget = 256;

u64 pack_pair(u32 l, u32 r) { return (static_cast<u64>(l) << 32) | r; }
u32 pair_left (u64 k) { return static_cast<u32>(k >> 32); }
u32 pair_right(u64 k) { return static_cast<u32>(k); }

u64 text_fingerprint(const std::string& l, const std::string& r) {
    u64 fp = 0;
    std::size_t i = 0;
    const std::size_t nl = std::min<std::size_t>(l.size(), 8);
    for (; i < nl; ++i) fp |= static_cast<u64>(static_cast<u8>(l[i])) << (8*(7-i));
    const std::size_t nr = std::min<std::size_t>(r.size(), 8-i);
    for (std::size_t j = 0; j < nr; ++j, ++i)
        fp |= static_cast<u64>(static_cast<u8>(r[j])) << (8*(7-i));
    return fp;
}

struct pair_state {
    u64 key = 0, count = 0;
    u32 word_count = 0;
    std::vector<u32> positions;
    u32 last_word = no_pos, last_group = no_pos;
    u64 fingerprint = 0;
};

struct t2_state {
    std::vector<u64> word_freq;
    std::vector<u32> token, prev, next, word_of, edge_group;
    std::vector<u8>  alive;
    std::vector<u64> token_count;
    std::vector<std::string> vocab;
    std::vector<pair_state>  pair_states;
    std::vector<u32> group_state, group_count;
    std::vector<u32> left_stamp, left_state, right_stamp, right_state;
    u32 same_stamp = no_pos, same_state = no_pos;
    std::vector<u32> born_states;
    u32 free_head = no_pos;
};

// ---- helpers ----------------------------------------------------------------

bool is_live(const t2_state& s, u32 p) { return p != no_pos && s.alive[p] != 0; }

u32 create_ps(t2_state& s, u64 key) {
    u32 id = static_cast<u32>(s.pair_states.size());
    s.pair_states.push_back({});
    auto& ps = s.pair_states.back();
    ps.key = key;
    ps.fingerprint = text_fingerprint(s.vocab[pair_left(key)], s.vocab[pair_right(key)]);
    s.born_states.push_back(id);
    return id;
}

u32 get_left_ps(t2_state& s, u32 left, u32 merged) {
    if (left == merged) {
        if (s.same_stamp != merged) { s.same_stamp=merged; s.same_state=create_ps(s,pack_pair(merged,merged)); }
        return s.same_state;
    }
    if (s.left_stamp[left] != merged) { s.left_stamp[left]=merged; s.left_state[left]=create_ps(s,pack_pair(left,merged)); }
    return s.left_state[left];
}

u32 get_right_ps(t2_state& s, u32 merged, u32 right) {
    if (right == merged) {
        if (s.same_stamp != merged) { s.same_stamp=merged; s.same_state=create_ps(s,pack_pair(merged,merged)); }
        return s.same_state;
    }
    if (s.right_stamp[right] != merged) { s.right_stamp[right]=merged; s.right_state[right]=create_ps(s,pack_pair(merged,right)); }
    return s.right_state[right];
}

u32 get_word_group(t2_state& s, u32 sid, u32 word) {
    auto& ps = s.pair_states[sid];
    if (ps.last_word == word) return ps.last_group;
    u32 g;
    if (s.free_head != no_pos) { g=s.free_head; s.free_head=s.group_count[g]; s.group_count[g]=0; s.group_state[g]=sid; }
    else { g=static_cast<u32>(s.group_state.size()); s.group_state.push_back(sid); s.group_count.push_back(0); }
    ps.last_word=word; ps.last_group=g;
    return g;
}

void remove_edge(t2_state& s, u32 start, u64 freq) {
    u32 g = s.edge_group[start];
    if (g==no_pos || s.group_count[g]==0) std::abort();
    auto& ps = s.pair_states[s.group_state[g]];
    --s.group_count[g]; ps.count -= freq;
    if (s.group_count[g]==0) {
        --ps.word_count;
        if (ps.last_group==g) { ps.last_group=no_pos; ps.last_word=no_pos; }
        s.group_count[g]=s.free_head; s.free_head=g;
    }
    s.edge_group[start]=no_pos;
}

void add_edge(t2_state& s, u32 start, u32 sid, u32 word, u64 freq) {
    u32 g = get_word_group(s, sid, word);
    auto& ps = s.pair_states[sid];
    if (s.group_count[g]++==0) ++ps.word_count;
    ps.count += freq;
    ps.positions.push_back(start);
    s.edge_group[start]=g;
}

bool pair_is_at(const t2_state& s, u32 p, u32 left, u32 right) {
    if (!is_live(s,p) || s.token[p]!=left) return false;
    u32 np = s.next[p];
    return is_live(s,np) && s.token[np]==right;
}

// ---- build_state ------------------------------------------------------------

void build_state(const std::vector<CharSplit>& splits, t2_state& s) {
    if (splits.size() >= no_pos) throw std::length_error("too many distinct words");

    s.vocab.resize(kVocab);
    s.token_count.assign(kVocab, 0);
    for (u32 v = 1; v < kVocab; ++v) s.vocab[v].assign(1, static_cast<char>(v));

    s.word_freq.reserve(splits.size());
    std::size_t slots = 0;
    for (const auto& sp : splits) slots += sp.chars.size()+1;
    if (slots >= no_pos) throw std::length_error("too many byte positions");

    s.token.reserve(slots); s.prev.reserve(slots); s.next.reserve(slots);
    s.word_of.reserve(slots); s.edge_group.reserve(slots); s.alive.reserve(slots);
    s.group_state.reserve(slots); s.group_count.reserve(slots);

    std::vector<u32> init(kVocab*kVocab, no_pos);
    for (u32 w = 0; w < static_cast<u32>(splits.size()); ++w) {
        const auto& sp = splits[w];
        s.word_freq.push_back(sp.count);
        u32 first = static_cast<u32>(s.token.size());

        for (std::size_t i = 0; i < sp.chars.size(); ++i) {
            u32 pos = static_cast<u32>(s.token.size());
            u32 val = sp.chars[i];
            if (val==0) throw std::invalid_argument("NUL byte in word");
            s.token.push_back(val);
            s.prev.push_back(i==0 ? no_pos : pos-1);
            s.next.push_back(pos+1);
            s.word_of.push_back(w);
            s.edge_group.push_back(no_pos);
            s.alive.push_back(1);
            s.token_count[val] += sp.count;
        }
        u32 sent = static_cast<u32>(s.token.size());
        s.token.push_back(0);
        s.prev.push_back(sp.chars.empty() ? no_pos : sent-1);
        s.next.push_back(no_pos);
        s.word_of.push_back(w); s.edge_group.push_back(no_pos); s.alive.push_back(0);

        if (sp.chars.empty()) continue;
        for (u32 pos = first; is_live(s,pos); pos=s.next[pos]) {
            u32 np = s.next[pos];
            if (!is_live(s,np)) break;
            u32 l=s.token[pos], r=s.token[np];
            u32& sid = init[l*kVocab+r];
            if (sid==no_pos) sid=create_ps(s, pack_pair(l,r));
            auto& ps = s.pair_states[sid];
            u32 g;
            if (ps.last_word==w) { g=ps.last_group; }
            else { g=static_cast<u32>(s.group_state.size()); s.group_state.push_back(sid); s.group_count.push_back(0); ps.last_word=w; ps.last_group=g; ++ps.word_count; }
            ++s.group_count[g]; ps.count+=sp.count; ps.positions.push_back(pos); s.edge_group[pos]=g;
        }
    }
    s.born_states.clear();
    std::size_t cache = s.vocab.size()+1024;
    s.left_stamp.assign(cache,no_pos); s.left_state.assign(cache,no_pos);
    s.right_stamp.assign(cache,no_pos); s.right_state.assign(cache,no_pos);
}

struct GP { u64 key, cnt, fp; };  // candidate pair for selection

void run_merge_loop_parallel(t2_state& state, int nranks) {
    // Build key → local pair_state id map.
    std::unordered_map<u64, u32> key_to_id;
    key_to_id.reserve(state.pair_states.size() * 2);
    for (u32 id = 0; id < static_cast<u32>(state.pair_states.size()); ++id)
        key_to_id[state.pair_states[id].key] = id;

    for (;;) {
        // 1. Pack local active pairs as triples: (key, count, word_count).
        std::vector<u64> local_buf;
        for (const auto& ps : state.pair_states) {
            if (ps.count > 0 && ps.word_count > 0) {
                local_buf.push_back(ps.key);
                local_buf.push_back(ps.count);
                local_buf.push_back(static_cast<u64>(ps.word_count));
            }
        }
        int local_n = static_cast<int>(local_buf.size());

        // 2. Allgatherv all pairs.
        std::vector<u64> all_buf;
#ifdef BPE_HAVE_MPI
        if (nranks > 1) {
            std::vector<int> rc(nranks), rd(nranks);
            MPI_Allgather(&local_n, 1, MPI_INT,
                          rc.data(), 1, MPI_INT, MPI_COMM_WORLD);
            rd[0] = 0;
            for (int r = 1; r < nranks; ++r) rd[r] = rd[r-1] + rc[r-1];
            int tot = rd[nranks-1] + rc[nranks-1];
            all_buf.resize(tot);
            MPI_Allgatherv(local_buf.data(), local_n, MPI_UINT64_T,
                           all_buf.data(), rc.data(), rd.data(),
                           MPI_UINT64_T, MPI_COMM_WORLD);
        } else {
            all_buf = std::move(local_buf);
        }
#else
        all_buf = std::move(local_buf);
#endif

        // 3. Build global key → (count, word_count).
        std::unordered_map<u64, std::pair<u64,u64>> gmap;
        for (std::size_t i = 0; i < all_buf.size(); i += 3) {
            auto& g = gmap[all_buf[i]];
            g.first  += all_buf[i+1];
            g.second += all_buf[i+2];
        }

        // 4. Build candidate list (global word_count >= 2).
        std::vector<GP> cands;
        cands.reserve(gmap.size());
        for (const auto& [key, cw] : gmap) {
            if (cw.second < 2) continue;
            cands.push_back({key, cw.first,
                text_fingerprint(state.vocab[pair_left(key)],
                                 state.vocab[pair_right(key)])});
        }
        if (cands.empty()) break;

        // 5. Select top kBudget — same ordering as queue_compare (deterministic).
        auto cmp_gp = [&state](const GP& a, const GP& b) {
            if (a.cnt != b.cnt) return a.cnt > b.cnt;
            if (a.fp  != b.fp)  return a.fp  < b.fp;
            const std::string ta = state.vocab[pair_left(a.key)]
                                 + state.vocab[pair_right(a.key)];
            const std::string tb = state.vocab[pair_left(b.key)]
                                 + state.vocab[pair_right(b.key)];
            const int c = std::strcmp(ta.c_str(), tb.c_str());
            if (c != 0) return c < 0;
            return a.key < b.key;
        };
        const std::size_t take = std::min(cands.size(), kBudget);
        if (cands.size() > take) {
            std::partial_sort(cands.begin(),
                              cands.begin() + static_cast<std::ptrdiff_t>(take),
                              cands.end(), cmp_gp);
            cands.resize(take);
        } else {
            std::sort(cands.begin(), cands.end(), cmp_gp);
        }

        // 6. Apply merges.
        for (const auto& gp : cands) {
            const u32 lt = pair_left(gp.key), rt = pair_right(gp.key);
            const u32 mt = static_cast<u32>(state.vocab.size());

            // All ranks: extend vocab + token_count to keep token IDs in sync.
            {
                std::string mg;
                mg.reserve(state.vocab[lt].size() + state.vocab[rt].size());
                mg.append(state.vocab[lt]); mg.append(state.vocab[rt]);
                state.vocab.push_back(std::move(mg));
                state.token_count.push_back(0);
            }
            if (state.left_stamp.size() <= static_cast<std::size_t>(mt)) {
                std::size_t ns = std::max<std::size_t>(
                    state.left_stamp.size() * 2, mt + 1024);
                state.left_stamp.resize(ns, no_pos);
                state.left_state.resize(ns, no_pos);
                state.right_stamp.resize(ns, no_pos);
                state.right_state.resize(ns, no_pos);
            }

            // Local merge: only if this rank has occurrences of this pair.
            auto it = key_to_id.find(gp.key);
            if (it != key_to_id.end()) {
                u32 sid = it->second;
                std::vector<u32> positions =
                    std::move(state.pair_states[sid].positions);
                u32 cw = no_pos; u64 freq = 0;
                for (u32 pos : positions) {
                    if (!pair_is_at(state, pos, lt, rt)) continue;
                    u32 rp = state.next[pos], w = state.word_of[pos];
                    if (w != cw) { cw = w; freq = state.word_freq[w]; }
                    u32 lp = state.prev[pos], ap = state.next[rp];
                    if (is_live(state, lp)) remove_edge(state, lp, freq);
                    remove_edge(state, pos, freq);
                    if (is_live(state, ap)) remove_edge(state, rp, freq);
                    state.token_count[lt] -= freq;
                    state.token_count[rt] -= freq;
                    state.token_count[mt] += freq;
                    state.token[pos] = mt;
                    state.alive[rp] = 0;
                    state.next[pos] = ap;
                    if (ap != no_pos) state.prev[ap] = pos;
                    state.prev[rp] = no_pos;
                    state.next[rp] = no_pos;
                    state.edge_group[rp] = no_pos;
                    if (is_live(state, lp))
                        add_edge(state, lp,
                                 get_left_ps(state, state.token[lp], mt), w, freq);
                    if (is_live(state, ap))
                        add_edge(state, pos,
                                 get_right_ps(state, mt, state.token[ap]), w, freq);
                }
            }
        }

        // Register newly born pair_states (created during merges above).
        for (u32 id : state.born_states)
            key_to_id[state.pair_states[id].key] = id;
        state.born_states.clear();
    }
}

// ---- finalize ---------------------------------------------------------------

void finalize_results(const t2_state& s, Results& results) {
    std::vector<u32> live;
    live.reserve(s.token_count.size());
    for (u32 id = 1; id < static_cast<u32>(s.token_count.size()); ++id)
        if (s.token_count[id]!=0) live.push_back(id);
    std::sort(live.begin(), live.end(), [&s](u32 a, u32 b){
        if (s.token_count[a]!=s.token_count[b]) return s.token_count[a]>s.token_count[b];
        return std::strcmp(s.vocab[a].c_str(), s.vocab[b].c_str())<0;
    });
    results.tokens.clear(); results.tokens.reserve(live.size());
    for (u32 id : live) {
        const std::string& txt = s.vocab[id];
        results.tokens.push_back({std::vector<Byte>(txt.begin(),txt.end()),
                                   static_cast<std::size_t>(s.token_count[id])});
    }
}

void finalize_results_parallel(t2_state& state, Results& results,
                                int rank, int nranks) {
#ifdef BPE_HAVE_MPI
    if (nranks > 1) {
        // Sum token_count across all ranks to rank 0.
        std::vector<u64> gtc(state.token_count.size(), 0);
        MPI_Reduce(state.token_count.data(), gtc.data(),
                   static_cast<int>(state.token_count.size()),
                   MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) state.token_count = std::move(gtc);
    }
#else
    (void)nranks;
#endif
    if (rank == 0) finalize_results(state, results);
}

// ---- timing -----------------------------------------------------------------

std::int64_t elapsed_ms(const std::chrono::steady_clock::time_point& a,
                        const std::chrono::steady_clock::time_point& b) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(b-a).count();
}

void write_timing_task2(int rank, int nranks,
                        const std::vector<std::int64_t>& local_times) {
    constexpr int NSTEPS = 3;
    static const char* NAMES[NSTEPS] = {
        "build_state  ", "merge_loop   ", "finalize     "};

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
    (void)rank; (void)nranks;
    const std::vector<std::int64_t>& all_times = local_times;
#endif

    if (rank == 0) {
        const char* job_id = std::getenv("SLURM_JOB_ID");
        std::string fname =
            std::string("timing-") + (job_id ? job_id : "local") + ".txt";
        std::ofstream f(fname, std::ios::app);
        f << "\n--- task2 ---\n";
        for (int r = 0; r < nranks; ++r) {
            f << "=== rank " << r << " of " << nranks << " ===\n";
            std::int64_t total = 0;
            for (int step = 0; step < NSTEPS; ++step) {
                std::int64_t t = all_times[r * NSTEPS + step];
                f << "  " << NAMES[step] << ": " << t << " ms\n";
                total += t;
            }
            f << "  total        : " << total << " ms\n\n";
        }
    }
}

}  // namespace

void parallel_task2(const std::vector<CharSplit>& splits, Results& results) {
#ifdef BPE_HAVE_MPI
    const int rank   = mpi_rank();
    const int nranks = mpi_size();
#else
    const int rank   = 0;
    const int nranks = 1;
#endif

    std::vector<std::int64_t> step_ms(3, 0);

    // Shard splits across ranks; each rank builds state on its own slice.
    const std::size_t total  = splits.size();
    const std::size_t stride = (total + static_cast<std::size_t>(nranks) - 1)
                               / static_cast<std::size_t>(nranks);
    const std::size_t start  = static_cast<std::size_t>(rank) * stride;
    const std::size_t end    = std::min(start + stride, total);
    const std::vector<CharSplit> local_splits(splits.begin() + start,
                                              splits.begin() + end);

    t2_state state;

    auto tp = std::chrono::steady_clock::now();
    build_state(local_splits, state);
    step_ms[0] = elapsed_ms(tp, std::chrono::steady_clock::now());
    LOG(INFO) << "task2 build_state: " << step_ms[0] << " ms";

    tp = std::chrono::steady_clock::now();
    run_merge_loop_parallel(state, nranks);
    step_ms[1] = elapsed_ms(tp, std::chrono::steady_clock::now());
    LOG(INFO) << "task2 merge_loop: " << step_ms[1] << " ms";

    tp = std::chrono::steady_clock::now();
    finalize_results_parallel(state, results, rank, nranks);
    step_ms[2] = elapsed_ms(tp, std::chrono::steady_clock::now());
    LOG(INFO) << "task2 finalize: " << step_ms[2] << " ms";

    write_timing_task2(rank, nranks, step_ms);
}

}  // namespace bpe
