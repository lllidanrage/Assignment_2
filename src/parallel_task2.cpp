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
#include <utility>
#include <vector>
#ifdef BPE_HAVE_MPI
#include <mpi.h>
#endif

namespace bpe {
namespace {

// ---- types (mirror task2.cpp) -----------------------------------------------

using u8  = std::uint8_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

constexpr u32 no_pos   = std::numeric_limits<u32>::max();
constexpr u32 kVocab   = 256;
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

struct queue_entry { u64 count = 0, fingerprint = 0; u32 state = 0; };

struct t2_state;
struct queue_cmp {
    const t2_state* s = nullptr;
    bool operator()(const queue_entry& a, const queue_entry& b) const;
};

struct queue_heap {
    std::vector<queue_entry> data;
    queue_cmp comp;
    bool empty() const { return data.empty(); }
    const queue_entry& top() const { return data.front(); }
    void push(queue_entry e) { data.push_back(e); sift_up(data.size()-1); }
    void pop() {
        data[0] = std::move(data.back()); data.pop_back();
        if (!data.empty()) sift_down(0);
    }
   private:
    static std::size_t par(std::size_t i) { return (i-1)/4; }
    static std::size_t fc (std::size_t i) { return 4*i+1; }
    void sift_up(std::size_t i) {
        while (i > 0) { auto p = par(i); if (!comp(data[p],data[i])) break; std::swap(data[p],data[i]); i=p; }
    }
    void sift_down(std::size_t i) {
        for (;;) {
            auto f = fc(i); if (f >= data.size()) break;
            auto best = f, end = std::min(f+4, data.size());
            for (auto c = f+1; c < end; ++c) if (comp(data[best],data[c])) best=c;
            if (!comp(data[i],data[best])) break;
            std::swap(data[i],data[best]); i=best;
        }
    }
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

bool queue_cmp::operator()(const queue_entry& a, const queue_entry& b) const {
    if (a.count != b.count) return a.count < b.count;
    if (a.fingerprint != b.fingerprint) return a.fingerprint > b.fingerprint;
    const pair_state& pa = s->pair_states[a.state];
    const pair_state& pb = s->pair_states[b.state];
    const std::string ta = s->vocab[pair_left(pa.key)]  + s->vocab[pair_right(pa.key)];
    const std::string tb = s->vocab[pair_left(pb.key)]  + s->vocab[pair_right(pb.key)];
    const int cmp = std::strcmp(ta.c_str(), tb.c_str());
    if (cmp != 0) return cmp > 0;
    return pa.key > pb.key;
}

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

void push_born(t2_state& s, queue_heap& q) {
    for (u32 id : s.born_states) {
        const auto& ps = s.pair_states[id];
        if (ps.word_count>=2 && ps.count!=0) q.push({ps.count,ps.fingerprint,id});
    }
    s.born_states.clear();
}

u32 pop_best(t2_state& s, queue_heap& q) {
    while (!q.empty()) {
        auto e = q.top(); q.pop();
        const auto& ps = s.pair_states[e.state];
        if (ps.word_count<2 || ps.count==0) continue;
        if (e.count!=ps.count) { q.push({ps.count,ps.fingerprint,e.state}); continue; }
        return e.state;
    }
    return no_pos;
}

// ---- three phases -----------------------------------------------------------

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
    for (u32 w = 0; w < splits.size(); ++w) {
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

void run_merge_loop(t2_state& s) {
    queue_heap q; q.comp.s = &s;
    for (u32 id = 0; id < s.pair_states.size(); ++id) {
        const auto& ps = s.pair_states[id];
        if (ps.word_count>=2) q.push({ps.count,ps.fingerprint,id});
    }
    std::vector<u32> sel_stamp(s.pair_states.size(), 0);
    u32 round_stamp = 0;
    std::vector<u32> batch;

    for (;;) {
        ++round_stamp; batch.clear();
        while (batch.size() < kBudget) {
            u32 best = pop_best(s, q);
            if (best==no_pos) break;
            if (best >= sel_stamp.size()) sel_stamp.resize(s.pair_states.size(), 0);
            if (sel_stamp[best]==round_stamp) continue;
            sel_stamp[best]=round_stamp; batch.push_back(best);
        }
        if (batch.empty()) break;

        for (u32 sel : batch) {
            u64 key = s.pair_states[sel].key;
            std::vector<u32> positions = std::move(s.pair_states[sel].positions);
            u32 lt=pair_left(key), rt=pair_right(key);
            u32 mt=static_cast<u32>(s.vocab.size());
            std::string merged; merged.reserve(s.vocab[lt].size()+s.vocab[rt].size());
            merged.append(s.vocab[lt]); merged.append(s.vocab[rt]);
            s.vocab.push_back(std::move(merged)); s.token_count.push_back(0);
            if (s.left_stamp.size()<=mt) {
                std::size_t ns=std::max<std::size_t>(s.left_stamp.size()*2, mt+1024);
                s.left_stamp.resize(ns,no_pos); s.left_state.resize(ns,no_pos);
                s.right_stamp.resize(ns,no_pos); s.right_state.resize(ns,no_pos);
            }
            u32 cw=no_pos; u64 freq=0;
            for (u32 pos : positions) {
                if (!pair_is_at(s,pos,lt,rt)) continue;
                u32 rp=s.next[pos], w=s.word_of[pos];
                if (w!=cw) { cw=w; freq=s.word_freq[w]; }
                u32 lp=s.prev[pos], ap=s.next[rp];
                if (is_live(s,lp)) remove_edge(s,lp,freq);
                remove_edge(s,pos,freq);
                if (is_live(s,ap)) remove_edge(s,rp,freq);
                s.token_count[lt]-=freq; s.token_count[rt]-=freq; s.token_count[mt]+=freq;
                s.token[pos]=mt; s.alive[rp]=0; s.next[pos]=ap;
                if (ap!=no_pos) s.prev[ap]=pos;
                s.prev[rp]=no_pos; s.next[rp]=no_pos; s.edge_group[rp]=no_pos;
                if (is_live(s,lp)) add_edge(s,lp,get_left_ps(s,s.token[lp],mt),w,freq);
                if (is_live(s,ap)) add_edge(s,pos,get_right_ps(s,mt,s.token[ap]),w,freq);
            }
        }
        push_born(s, q);
    }
}

void finalize_results(const t2_state& s, Results& results) {
    std::vector<u32> live;
    live.reserve(s.token_count.size());
    for (u32 id = 1; id < s.token_count.size(); ++id)
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
    std::vector<std::int64_t> all_times(nranks * NSTEPS);
    MPI_Gather(local_times.data(), NSTEPS, MPI_INT64_T,
               all_times.data(), NSTEPS, MPI_INT64_T,
               0, MPI_COMM_WORLD);
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

    t2_state state;

    auto tp = std::chrono::steady_clock::now();
    build_state(splits, state);
    step_ms[0] = elapsed_ms(tp, std::chrono::steady_clock::now());
    LOG(INFO) << "task2 build_state: " << step_ms[0] << " ms";

    tp = std::chrono::steady_clock::now();
    run_merge_loop(state);
    step_ms[1] = elapsed_ms(tp, std::chrono::steady_clock::now());
    LOG(INFO) << "task2 merge_loop: " << step_ms[1] << " ms";

    tp = std::chrono::steady_clock::now();
    finalize_results(state, results);
    step_ms[2] = elapsed_ms(tp, std::chrono::steady_clock::now());
    LOG(INFO) << "task2 finalize: " << step_ms[2] << " ms";

    write_timing_task2(rank, nranks, step_ms);
}

}  // namespace bpe
