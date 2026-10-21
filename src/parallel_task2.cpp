#include "bpe.h"

#include "absl/container/inlined_vector.h"
#include "absl/log/log.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <omp.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#ifdef BPE_HAVE_MPI
#include <mpi.h>
#endif

namespace bpe {
namespace {

inline int configure_parallel_threads(int rank, int nranks) {
#ifdef BPE_HAVE_MPI
    const char* tuple = std::getenv("BPE_THREADS");
    if (tuple != nullptr && *tuple != '\0') {
        const char* cursor = tuple;
        int requested = 0;
        for (int r = 0; r < nranks; ++r) {
            char* end = nullptr;
            errno = 0;
            const long count = std::strtol(cursor, &end, 10);
            if (end == cursor || errno == ERANGE || count < 1 ||
                count > INT_MAX ||
                (r + 1 < nranks ? *end != ',' : *end != '\0')) {
                throw std::invalid_argument(
                    "BPE_THREADS must contain one positive integer per MPI "
                    "rank, separated by commas");
            }
            if (r == rank) requested = static_cast<int>(count);
            cursor = r + 1 < nranks ? end + 1 : end;
        }
        omp_set_dynamic(0);
        omp_set_num_threads(requested);
    }
#else
    (void)rank;
    (void)nranks;
#endif
    return omp_get_max_threads();
}

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
    absl::InlinedVector<u32, 2> positions;
    u32 last_word = no_pos, last_group = no_pos;
    u64 fingerprint = 0;
    u64 sent_count = 0;
    u32 sent_word_count = 0;
    bool dirty = false;
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

class pair_index {
    struct entry { u64 key = 0; u32 value = no_pos; u32 generation = 0; };
    std::vector<entry> entries;
    std::size_t used = 0;
    u32 generation = 1;
    static std::size_t hash(u64 key) {
        key ^= key >> 30; key *= UINT64_C(0xbf58476d1ce4e5b9);
        key ^= key >> 27; key *= UINT64_C(0x94d049bb133111eb);
        return static_cast<std::size_t>(key ^ (key >> 31));
    }
    void place(u64 key, u32 value) {
        std::size_t i = hash(key) & (entries.size() - 1);
        while (entries[i].generation == generation) i = (i + 1) & (entries.size() - 1);
        entries[i] = {key, value, generation};
    }
   public:
    u32 find(u64 key) const {
        if (entries.empty()) return no_pos;
        std::size_t i = hash(key) & (entries.size() - 1);
        while (entries[i].generation == generation) {
            if (entries[i].key == key) return entries[i].value;
            i = (i + 1) & (entries.size() - 1);
        }
        return no_pos;
    }
    void insert(u64 key, u32 value) {
        if ((used + 1) * 10 >= entries.size() * 7) {
            std::vector<entry> old;
            old.swap(entries);
            entries.resize(std::max<std::size_t>(1024, old.size() * 2));
            for (const auto& item : old)
                if (item.generation == generation) place(item.key, item.value);
        }
        place(key, value);
        ++used;
    }
    void clear() {
        used = 0;
        if (++generation == 0) {
            std::fill(entries.begin(), entries.end(), entry{});
            generation = 1;
        }
    }
};

struct alignas(64) t2_state {
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
    bool distributed = false;
    pair_index by_key;
    const std::vector<std::string>* shared_vocab = nullptr;
    std::vector<u32> dirty_states;
};

bool queue_cmp::operator()(const queue_entry& a, const queue_entry& b) const {
    if (a.count != b.count) return a.count < b.count;
    if (a.fingerprint != b.fingerprint) return a.fingerprint > b.fingerprint;
    const auto& vocab = s->shared_vocab ? *s->shared_vocab : s->vocab;
    const pair_state& pa = s->pair_states[a.state];
    const pair_state& pb = s->pair_states[b.state];
    const auto& al = vocab[pair_left(pa.key)];
    const auto& ar = vocab[pair_right(pa.key)];
    const auto& bl = vocab[pair_left(pb.key)];
    const auto& br = vocab[pair_right(pb.key)];
    const std::size_t an = al.size() + ar.size(), bn = bl.size() + br.size();
    for (std::size_t i = 8; i < std::min(an, bn); ++i) {
        const auto ac = static_cast<u8>(i < al.size() ? al[i] : ar[i - al.size()]);
        const auto bc = static_cast<u8>(i < bl.size() ? bl[i] : br[i - bl.size()]);
        if (ac != bc) return ac > bc;
    }
    if (an != bn) return an > bn;
    return pa.key > pb.key;
}

// ---- helpers ----------------------------------------------------------------

bool is_live(const t2_state& s, u32 p) { return p != no_pos && s.alive[p] != 0; }

void mark_dirty(t2_state& s, u32 id) {
    if (s.distributed && !s.pair_states[id].dirty) {
        s.pair_states[id].dirty = true;
        s.dirty_states.push_back(id);
    }
}

u32 create_ps(t2_state& s, u64 key) {
    u32 id = static_cast<u32>(s.pair_states.size());
    s.pair_states.push_back({});
    auto& ps = s.pair_states.back();
    ps.key = key;
    if (!s.distributed) {
        const auto& vocab = s.shared_vocab ? *s.shared_vocab : s.vocab;
        ps.fingerprint = text_fingerprint(vocab[pair_left(key)], vocab[pair_right(key)]);
    }
    s.born_states.push_back(id);
    if (s.distributed) {
        mark_dirty(s, id);
    }
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
    mark_dirty(s, s.group_state[g]);
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
    mark_dirty(s, sid);
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

void build_state(const std::vector<CharSplit>& splits, t2_state& s,
                 std::size_t begin, std::size_t end) {
    if (splits.size() >= no_pos) throw std::length_error("too many distinct words");

    s.vocab.resize(kVocab);
    s.token_count.assign(kVocab, 0);
    for (u32 v = 1; v < kVocab; ++v) s.vocab[v].assign(1, static_cast<char>(v));

    s.word_freq.reserve(end - begin);
    std::size_t slots = 0;
    for (std::size_t w = begin; w < end; ++w) slots += splits[w].chars.size()+1;
    if (slots >= no_pos) throw std::length_error("too many byte positions");

    s.token.reserve(slots); s.prev.reserve(slots); s.next.reserve(slots);
    s.word_of.reserve(slots); s.edge_group.reserve(slots); s.alive.reserve(slots);
    s.group_state.reserve(slots); s.group_count.reserve(slots);

    std::vector<u32> init(kVocab*kVocab, no_pos);
    for (u32 w = 0; w < end - begin; ++w) {
        const auto& sp = splits[begin + w];
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
    if (!s.distributed) s.born_states.clear();
    std::size_t cache = s.vocab.size()+1024;
    s.left_stamp.assign(cache,no_pos); s.left_state.assign(cache,no_pos);
    s.right_stamp.assign(cache,no_pos); s.right_state.assign(cache,no_pos);
}

void merge_pair(t2_state& s, u64 key, u32 sel, u32 mt = no_pos) {
    absl::InlinedVector<u32, 2> positions;
    if (sel != no_pos) positions = std::move(s.pair_states[sel].positions);
    u32 lt=pair_left(key), rt=pair_right(key);
    if (mt == no_pos) {
        mt = static_cast<u32>(s.vocab.size());
        s.vocab.push_back(s.vocab[lt] + s.vocab[rt]);
    }
    s.token_count.push_back(0);
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

        for (u32 sel : batch) merge_pair(s, s.pair_states[sel].key, sel);
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

struct pair_delta {
    u64 key;
    u64 count;
    u64 words;
};
static_assert(sizeof(pair_delta) == 3 * sizeof(u64), "packed MPI delta");

void collect_changes(t2_state& s, std::vector<pair_delta>& changes) {
    for (u32 id : s.born_states)
        if (s.pair_states[id].word_count != 0)
            s.by_key.insert(s.pair_states[id].key, id);
    changes.clear();
    changes.reserve(s.dirty_states.size());
    for (u32 id : s.dirty_states) {
        auto& ps = s.pair_states[id];
        const u64 dc = ps.count - ps.sent_count;
        const u64 dw = static_cast<u64>(ps.word_count) - ps.sent_word_count;
        if (dc != 0 || dw != 0) changes.push_back({ps.key, dc, dw});
        ps.sent_count = ps.count;
        ps.sent_word_count = ps.word_count;
        ps.dirty = false;
    }
    s.dirty_states.clear();
    s.born_states.clear();
}

void exchange_changes(const std::vector<std::vector<pair_delta>>& changes,
                      std::vector<pair_delta>& received, int rank, int nranks) {
    std::vector<pair_delta> local;
    std::size_t total = 0;
    for (const auto& part : changes) total += part.size();
    local.reserve(total);
    for (const auto& part : changes)
        local.insert(local.end(), part.begin(), part.end());
#ifdef BPE_HAVE_MPI
    if (nranks > 1) {
        int too_large = total > static_cast<std::size_t>(INT_MAX) / 3;
        int any_large = 0;
        MPI_Allreduce(&too_large, &any_large, 1, MPI_INT, MPI_MAX,
                      MPI_COMM_WORLD);
        if (any_large) throw std::length_error("task2 delta message too large");
        int count = static_cast<int>(total * 3);
        std::vector<int> counts(nranks), offsets(nranks);
        MPI_Gather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0,
                   MPI_COMM_WORLD);
        std::size_t size = 0;
        if (rank == 0) {
            for (int r = 0; r < nranks; ++r) {
                offsets[r] = static_cast<int>(size);
                size += counts[r];
            }
            too_large = size > INT_MAX;
        }
        MPI_Bcast(&too_large, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (too_large) throw std::length_error("task2 gathered deltas too large");
        received.resize(size / 3);
        MPI_Gatherv(local.data(), count, MPI_UINT64_T, received.data(),
                    counts.data(), offsets.data(), MPI_UINT64_T, 0,
                    MPI_COMM_WORLD);
        return;
    }
#else
    (void)rank;
    (void)nranks;
#endif
    received.swap(local);
}

void run_distributed_loop(std::vector<t2_state>& workers, int rank, int nranks) {
    t2_state global;
    global.shared_vocab = &workers.front().vocab;
    queue_heap queue;
    queue.comp.s = &global;
    std::vector<std::vector<pair_delta>> changes(workers.size());
    std::vector<pair_delta> received;
    std::vector<u64> batch;
    pair_index newborn_index;
    std::vector<pair_delta> newborn;
    u32 first_new_token = 0;
    const int nworkers = static_cast<int>(workers.size());
    std::vector<std::exception_ptr> errors(nworkers);

    using clock = std::chrono::steady_clock;
    double collect_ms = 0, exchange_ms = 0, select_ms = 0, apply_ms = 0;
    std::size_t rounds = 0, updates = 0;
    auto add_time = [](double& total, clock::time_point start) {
        total += std::chrono::duration<double, std::milli>(clock::now() - start).count();
    };
    for (;;) {
        auto stage = clock::now();
#pragma omp parallel for num_threads(nworkers) schedule(static)
        for (int w = 0; w < nworkers; ++w)
            collect_changes(workers[w], changes[w]);
        add_time(collect_ms, stage); stage = clock::now();
        exchange_changes(changes, received, rank, nranks);
        add_time(exchange_ms, stage); stage = clock::now();
        updates += received.size();
        batch.clear();
        if (rank == 0) {
            newborn_index.clear();
            newborn.clear();
            for (const auto& d : received) {
                if (pair_left(d.key) >= first_new_token ||
                    pair_right(d.key) >= first_new_token) {
                    u32 id = newborn_index.find(d.key);
                    if (id == no_pos) {
                        id = static_cast<u32>(newborn.size());
                        newborn_index.insert(d.key, id);
                        newborn.push_back({d.key, 0, 0});
                    }
                    newborn[id].count += d.count;
                    newborn[id].words += d.words;
                } else {
                    u32 id = global.by_key.find(d.key);
                    if (id == no_pos) continue;
                    auto& ps = global.pair_states[id];
                    ps.count += d.count;
                    ps.word_count = static_cast<u32>(ps.word_count + d.words);
                }
            }
            for (const auto& d : newborn) {
                if (d.words < 2 || d.count == 0) continue;
                u32 id = create_ps(global, d.key);
                global.by_key.insert(d.key, id);
                global.pair_states[id].count = d.count;
                global.pair_states[id].word_count = static_cast<u32>(d.words);
            }
            push_born(global, queue);
            while (batch.size() < kBudget) {
                u32 best = pop_best(global, queue);
                if (best == no_pos) break;
                batch.push_back(global.pair_states[best].key);
            }
        }
#ifdef BPE_HAVE_MPI
        if (nranks > 1) {
            int size = static_cast<int>(batch.size());
            MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
            batch.resize(size);
            MPI_Bcast(batch.data(), size, MPI_UINT64_T, 0, MPI_COMM_WORLD);
        }
#endif
        add_time(select_ms, stage); stage = clock::now();
        if (batch.empty()) break;
        ++rounds;
        auto& vocab = workers.front().vocab;
        const u32 first_token = static_cast<u32>(vocab.size());
        first_new_token = first_token;
        for (u64 key : batch) vocab.push_back(vocab[pair_left(key)] + vocab[pair_right(key)]);
#pragma omp parallel for num_threads(nworkers) schedule(static)
        for (int w = 0; w < nworkers; ++w) {
            try {
                auto& s = workers[w];
                for (std::size_t i = 0; i < batch.size(); ++i)
                    merge_pair(s, batch[i], s.by_key.find(batch[i]),
                               first_token + static_cast<u32>(i));
            } catch (...) { errors[w] = std::current_exception(); }
        }
        for (const auto& error : errors) if (error) std::rethrow_exception(error);
        add_time(apply_ms, stage);
    }
    LOG(INFO) << "task2 rounds: " << rounds << "; collect: " << collect_ms
              << " ms; exchange: " << exchange_ms << " ms; select: " << select_ms
              << " ms; apply: " << apply_ms << " ms; updates: " << updates
              << "; global states: " << global.pair_states.size();
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
    if (nranks > 1) {
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

    int nworkers = configure_parallel_threads(rank, nranks);
    nworkers = std::max(1, std::min(nworkers, static_cast<int>(
        std::min<std::size_t>(splits.size(), INT_MAX))));
    std::vector<int> counts(nranks, nworkers);
#ifdef BPE_HAVE_MPI
    if (nranks > 1)
        MPI_Allgather(&nworkers, 1, MPI_INT, counts.data(), 1, MPI_INT,
                      MPI_COMM_WORLD);
#endif
    int total_workers = 0, first_worker = 0;
    for (int r = 0; r < nranks; ++r) {
        if (r < rank) first_worker += counts[r];
        total_workers += counts[r];
    }
    std::vector<std::int64_t> step_ms(3, 0);
    auto tp = std::chrono::steady_clock::now();
    std::vector<std::size_t> prefix(splits.size() + 1, 0);
    for (std::size_t w = 0; w < splits.size(); ++w) {
        prefix[w + 1] = prefix[w] + splits[w].chars.size() + 1;
        if (std::find(splits[w].chars.begin(), splits[w].chars.end(), Byte(0))
            != splits[w].chars.end()) throw std::invalid_argument("NUL byte in word");
    }
    auto boundary = [&](int worker) {
        if (worker == total_workers) return splits.size();
        const std::size_t target = prefix.back() / total_workers * worker +
            (prefix.back() % total_workers) * worker / total_workers;
        return static_cast<std::size_t>(std::lower_bound(prefix.begin(),
                                     prefix.end(), target) - prefix.begin());
    };
    std::vector<t2_state> workers(nworkers);
    std::vector<std::exception_ptr> errors(nworkers);
#pragma omp parallel for num_threads(nworkers) schedule(static)
    for (int w = 0; w < nworkers; ++w) {
        try {
            workers[w].distributed = total_workers > 1;
            build_state(splits, workers[w], boundary(first_worker + w),
                        boundary(first_worker + w + 1));
        } catch (...) { errors[w] = std::current_exception(); }
    }
    for (const auto& error : errors) if (error) std::rethrow_exception(error);
    step_ms[0] = elapsed_ms(tp, std::chrono::steady_clock::now());
    LOG(INFO) << "task2 build_state: " << step_ms[0] << " ms";
    LOG(INFO) << "task2 workers: " << nworkers << "; MPI rank " << rank
              << " of " << nranks;

    tp = std::chrono::steady_clock::now();
    if (total_workers == 1) run_merge_loop(workers.front());
    else run_distributed_loop(workers, rank, nranks);
    step_ms[1] = elapsed_ms(tp, std::chrono::steady_clock::now());
    LOG(INFO) << "task2 merge_loop: " << step_ms[1] << " ms";

    tp = std::chrono::steady_clock::now();
    auto& state = workers.front();
    for (int w = 1; w < nworkers; ++w)
        for (std::size_t t = 0; t < state.token_count.size(); ++t)
            state.token_count[t] += workers[w].token_count[t];
#ifdef BPE_HAVE_MPI
    if (nranks > 1) {
        if (state.token_count.size() > INT_MAX)
            throw std::length_error("task2 token count message too large");
        MPI_Allreduce(MPI_IN_PLACE, state.token_count.data(),
                      static_cast<int>(state.token_count.size()), MPI_UINT64_T,
                      MPI_SUM, MPI_COMM_WORLD);
    }
#endif
    finalize_results(state, results);
    step_ms[2] = elapsed_ms(tp, std::chrono::steady_clock::now());
    LOG(INFO) << "task2 finalize: " << step_ms[2] << " ms";

    write_timing_task2(rank, nranks, step_ms);
}

}  // namespace bpe
