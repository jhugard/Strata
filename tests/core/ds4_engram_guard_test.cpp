// tests/core/ds4_engram_guard_test.cpp - the Engram table header's guard
// (include/strata/dsv41/ds4_engram_guard.hpp) on hand-written headers and pairs: no model, no GPU, no file.
//
// Why these cases.  engram.bin's six header values used to reach four resize() calls with nothing checking them, so
// a negative ngram or heads made cols negative, `(size_t) layers * cols` turned that into an enormous resize, and a
// header could claim tables far bigger than the file.  cols then reached eg_issue as `(size_t) li * cols + c` (a
// negative heap index), eg_ngram wrote past a fixed `int64_t prod[8]`, and every (prime, offset) pair went into
// `id = rolling % prime + off`, which eg_issue uses as `table_off + id * row_bytes` - a row address in the model
// shards tied to nothing.  The guard has to reject every header tools/dsv41/pack_gguf.py could not have written, and
// accept everything it does - a guard that refused a real pack would be worse than no guard, which is why the first
// group below is what the writer emits.
//
// The sizes come from write_engram()'s own byte counts (the guard's header comment): vocab*4 of token map,
// layers*ngram*8 of multipliers, and layers*cols*8 twice over for primes and offsets, with cols = (ngram-1)*heads.

#include "strata/dsv41/ds4_engram_guard.hpp"

#include <cstdio>
#include <string>
#include <vector>

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-92s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

using strata::dsv41::EngramGuard;
using strata::dsv41::EngramHeader;
using strata::dsv41::EngramLayout;

// A pack the writer could have produced: 64 tokens of map, 2 Engram layers, a max ngram of 4 over 2 heads (so
// cols = 6), a head of 128 values, and a pad token.  Tables: 256 + 2*4*8 + 2*6*8 + 2*6*8 = 704 bytes of body.
EngramHeader valid() {
    EngramHeader h;
    h.vocab = 64; h.layers = 2; h.ngram = 4; h.heads = 2; h.hd = 128; h.pad = 3;
    return h;
}
const uint64_t kValidBody = 64ull * 4 + 2ull * 4 * 8 + 2ull * 6 * 8 + 2ull * 6 * 8;

/// accept() for one mutated header against a body of `body` bytes; `err` gets the guard's words.
bool accept(EngramHeader h, uint64_t body, std::string* err, EngramLayout* lay = nullptr) {
    EngramLayout tmp;
    return EngramGuard(body).accept(h, lay ? lay : &tmp, err);
}

/// pairs_in_table over one layer of `cols` columns, all of them addressing rows [0, rows) harmlessly.
std::vector<int64_t> good_primes(int cols, int64_t rows) {
    return std::vector<int64_t>((size_t) cols, rows);
}
std::vector<int64_t> good_offs(int cols) {
    return std::vector<int64_t>((size_t) cols, 0);
}

bool pairs(const std::vector<int64_t>& primes, const std::vector<int64_t>& offs, uint64_t cols, uint64_t rows,
           std::string* err) {
    return EngramGuard::pairs_in_table(primes, offs, 0, cols, rows, "Engram layer 0", err);
}
}  // namespace

int main() {
    std::printf("ds4_engram_guard_test\n");
    std::string err;

    std::printf("what the writer emits\n");
    EngramLayout lay;
    check(accept(valid(), kValidBody, &err, &lay), "a header with exactly the bytes it implies is accepted");
    check(lay.cols == 6 && lay.map_elems == 64 && lay.mult_elems == 8 && lay.table_elems == 12,
          "the layout it reports is (ngram-1)*heads cols and the four table element counts");
    check(accept(valid(), kValidBody, &err) && err.empty(), "an accepted header sets no error");
    // Every field at the largest value the writer's own checkpoints could plausibly carry, sizes recomputed.
    {
        EngramHeader big = valid();
        big.vocab = 1 << 20; big.layers = 64; big.ngram = 8; big.heads = 4096; big.hd = 7168;
        const uint64_t cols = (uint64_t) (big.ngram - 1) * big.heads;
        const uint64_t body = (uint64_t) big.vocab * 4 + (uint64_t) big.layers * big.ngram * 8 +
                              (uint64_t) big.layers * cols * 8 * 2;
        check(accept(big, body, &err), "a wide real-shaped header (8-gram, 4096 heads, hd 7168) is accepted");
    }

    std::printf("\nevery shape that used to reach a resize()\n");
    {
        EngramHeader h = valid(); h.ngram = -1;
        check(!accept(h, kValidBody, &err), "a negative ngram is refused");
    }
    {
        EngramHeader h = valid(); h.heads = -2;
        check(!accept(h, kValidBody, &err), "a negative heads (which makes cols negative) is refused");
    }
    {
        EngramHeader h = valid(); h.vocab = 0;
        check(!accept(h, kValidBody, &err), "a zero vocab is refused");
    }
    {
        EngramHeader h = valid(); h.layers = 0;
        check(!accept(h, kValidBody, &err), "a zero layer count is refused");
    }
    {
        EngramHeader h = valid(); h.hd = 0;
        check(!accept(h, kValidBody, &err), "a zero head size is refused");
    }
    {
        EngramHeader h = valid(); h.ngram = 9;
        check(!accept(h, kValidBody, &err), "an ngram past the 8 look-backs eg_issue's prod[8] holds is refused");
    }
    {
        EngramHeader h = valid(); h.ngram = 1000000000;
        check(!accept(h, kValidBody, &err), "an enormous ngram is refused before it can size a product");
    }
    {
        EngramHeader h = valid(); h.vocab = 1ll << 40;
        check(!accept(h, kValidBody, &err), "a vocab past the int the loader stores it in is refused");
    }
    {
        EngramHeader h = valid(); h.layers = 1ll << 50;
        check(!accept(h, kValidBody, &err), "a layer count that would overflow layers * cols is refused");
    }
    {
        EngramHeader h = valid(); h.heads = 1ll << 50;
        check(!accept(h, kValidBody, &err), "a heads count that would overflow (ngram-1) * heads is refused");
    }
    {
        EngramHeader h = valid(); h.hd = 100;
        check(!accept(h, kValidBody, &err),
              "a head size that is not a whole number of 32-value blocks is refused (hd / 32 sizes the scales)");
    }

    std::printf("\nthe file's own length\n");
    check(!accept(valid(), kValidBody - 8, &err), "a file 8 bytes short of its header is refused");
    check(!accept(valid(), kValidBody + 8, &err), "a file with 8 bytes more than its header implies is refused");
    check(!accept(valid(), 0, &err), "an empty body is refused");
    {
        // body_len itself: 100 bytes of file, 28 consumed, so 72 remain; and a file shorter than its header.
        std::FILE* f = std::tmpfile();
        if (f != nullptr) {
            for (int i = 0; i < 100; ++i) std::fputc(0, f);
            uint64_t b = 0;
            check(strata::dsv41::body_len(f, 28, &b) && b == 72, "body_len reports the bytes after a 28-byte header");
            check(!strata::dsv41::body_len(f, 101, &b), "body_len refuses a file shorter than the header it consumed");
            std::fclose(f);
        }
    }

    std::printf("\nthe (prime, offset) pairs that pick a row\n");
    check(pairs(good_primes(6, 1000), good_offs(6), 6, 1000, &err), "pairs addressing rows 0..999 of 1000 are accepted");
    {
        std::vector<int64_t> p = good_primes(6, 1000), o = good_offs(6);
        p[0] = 1000; o[0] = 0;
        check(pairs(p, o, 6, 1000, &err), "a prime spanning the whole table is accepted");
    }
    {
        std::vector<int64_t> p = good_primes(6, 1000), o = good_offs(6);
        p[0] = 10; o[0] = 995;
        check(!pairs(p, o, 6, 1000, &err), "a pair whose last row is 1004 of 1000 is refused");
    }
    {
        std::vector<int64_t> p = good_primes(6, 1000), o = good_offs(6);
        o[2] = -1;
        check(!pairs(p, o, 6, 1000, &err), "a negative offset is refused (it became a huge row address)");
    }
    {
        std::vector<int64_t> p = good_primes(6, 1000), o = good_offs(6);
        p[2] = 0;
        check(!pairs(p, o, 6, 1000, &err), "a prime of 0 is refused (the row hash divides by it)");
    }
    {
        std::vector<int64_t> p = good_primes(6, 1000), o = good_offs(6);
        p[2] = -7;
        check(!pairs(p, o, 6, 1000, &err), "a negative prime is refused");
    }
    {
        std::vector<int64_t> p = good_primes(6, 1000), o = good_offs(6);
        p[2] = 1ll << 62;
        check(!pairs(p, o, 6, 1000, &err), "a prime that would overflow off + prime is refused");
    }
    {
        std::vector<int64_t> p = good_primes(1, 1000), o = good_offs(1);
        check(!EngramGuard::pairs_in_table(p, o, 5, 6, 1000, "Engram layer 5", &err),
              "a layer whose entries run past the table is refused");
    }

    std::printf("\nthe hash's modulo\n");
    check(strata::dsv41::nonneg_mod(-1, 7) == 6, "nonneg_mod(-1, 7) is 6, not -1 (a negative id left the table)");
    check(strata::dsv41::nonneg_mod(-7, 7) == 0, "nonneg_mod(-7, 7) is 0");
    check(strata::dsv41::nonneg_mod(8, 7) == 1, "nonneg_mod matches %% on a positive dividend");
    check(strata::dsv41::nonneg_mod(0, 7) == 0, "nonneg_mod(0, 7) is 0");

    std::printf("\nthe checked arithmetic\n");
    {
        uint64_t o = 0;
        check(strata::dsv41::mul_overflow_u64(1ull << 32, 1ull << 32, &o) && o == 0,
              "mul_overflow_u64 flags 2^32 * 2^32");
        check(!strata::dsv41::mul_overflow_u64(1ull << 32, 1ull << 31, &o) && o == (1ull << 63),
              "mul_overflow_u64 lets 2^32 * 2^31 through");
        check(!strata::dsv41::mul_overflow_u64(0, 1ull << 63, &o) && o == 0, "mul_overflow_u64 handles a zero factor");
        check(strata::dsv41::add_overflow_u64(UINT64_MAX, 1, &o), "add_overflow_u64 flags UINT64_MAX + 1");
        check(!strata::dsv41::add_overflow_u64(UINT64_MAX - 1, 1, &o) && o == UINT64_MAX,
              "add_overflow_u64 lets UINT64_MAX - 1 + 1 through");
    }

    std::printf("\n%s (%d failing)\n", g_fail == 0 ? "all ok" : "FAILURES", g_fail);
    return g_fail == 0 ? 0 : 1;
}
