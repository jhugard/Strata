// include/strata/dsv41/ds4_engram_guard.hpp - the Engram table header's guard for the DeepSeek-V4.1 engine
// (src/dsv41/ds4_model.cu's Pack::open, its Engram layer loop, and eg_issue).
//
// WHY THIS EXISTS.  engram.bin starts with "DSE1" and six int32 values:
//
//     vocab, layers, ngram, heads, hd, pad
//
// and the load used to hand each straight to four resize() calls before anything checked them:
//
//     eg_map.resize(vocab);  eg_mult.resize(layers * ngram);  eg_primes.resize(layers * cols);  eg_off.resize(...)
//     where cols = (ngram - 1) * heads
//
// Three things followed from that.  A negative ngram or heads makes cols NEGATIVE, and `(size_t) layers * cols`
// turns that into an enormous resize, so the file is read into a vector the machine cannot hold.  cols reaching
// eg_issue is used as `(size_t) li * cols + c`, which with a negative cols is a negative index into the heap.
// eg_ngram indexes a fixed `int64_t prod[8]` in eg_issue, so ngram > 8 writes off the stack.  And nothing bounded
// the four tables' sizes against the bytes the file actually holds, so a header could claim a table of any size
// while the fread below it simply failed - or, worse, the primes and offsets inside the file went unchecked into
// `id = rolling % primes[j] + offs[j]`, which eg_issue uses as `table_off + id * row_bytes`: a row address in the
// model shards that nothing tied to the table that holds it.
//
// engram.bin is written by tools/dsv41/pack_gguf.py from a GGUF's general.engram.* KV arrays, and that Python side
// checks the arrays' LENGTHS but never their VALUES (pack_gguf.py:423-427), so the six header values and every
// prime and offset reach this C++ straight from a community GGUF.  Same class as the DF2P pack guard
// (include/strata/core/df2_pack_guard.hpp), which is the file that named this site as the one it did not touch.
//
// WHAT MAKES A HEADER VALID.  The writer is the contract again.  write_engram() emits the magic, the six values,
// then exactly four tables and nothing else:
//
//     vocab * 4 bytes of token map
//     layers * ngram * 8 bytes of multipliers
//     layers * cols * 8 bytes of primes
//     layers * cols * 8 bytes of offsets
//
// So the rule is that the bytes following the header must EQUAL what the header implies, exactly as the DF2P guard
// requires a tensor's byte count to equal what its geometry implies.  That one rule bounds everything downstream
// without a table of arbitrary ceilings doing the work: the four resizes cannot exceed the file, cols and the
// table counts cannot overflow, and a header that lies about a size simply does not match the file's length.
// The ceilings below are secondary - they keep each value inside the `int` the loader stores it in and keep
// ngram inside the fixed prod[8], which a file length cannot bound.
//
// WHAT MAKES THE PAIRS VALID.  For one layer, eg_issue computes
//
//     id = rolling % primes[j] + offs[j]      then reads at  table_off + id * row_bytes
//
// rolling is a hash of the tokens and may be negative, and C++'s % of a negative dividend is negative, so the row
// must be taken with nonneg_mod() below (that is the loader's fix); once it is, the row lies in
// [0, primes[j] - 1].  So the pair addresses rows [offs[j], offs[j] + primes[j] - 1], and requiring those to lie
// inside the table tensor bounds every read the hash can produce.  row_bytes follows the format's own row layout
// in eg_issue, and the row count is derived from the tensor's byte count, so a table with padding only makes the
// bound looser rather than refusing a real pack.
//
// The checked add and multiply are written out by hand because MSVC has no __builtin_add_overflow or
// __builtin_mul_overflow, and ds4_model.cu's host pass is MSVC on Windows even in a CUDA build.  They mirror the
// helpers in df2_pack_guard.hpp rather than including it, so that the DeepSeek engine keeps no dependency on the
// drafter's header.
//
// It reports the first problem and says which value or which pair is wrong, so a bad file names itself.

#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace strata {
namespace dsv41 {

/// True when `a * b` overflows uint64_t; `*out` holds the product when it does not.
inline bool mul_overflow_u64(uint64_t a, uint64_t b, uint64_t* out) {
    if (a == 0 || b == 0) { *out = 0; return false; }
    if (a > UINT64_MAX / b) return true;
    *out = a * b;
    return false;
}

/// True when `a + b` overflows uint64_t; `*out` holds the sum when it does not.
inline bool add_overflow_u64(uint64_t a, uint64_t b, uint64_t* out) {
    const uint64_t s = a + b;
    *out = s;
    return s < a;
}

/// The bytes that follow `consumed` bytes of header in `f`, false when the size cannot be had.  Puts the read
/// position back at `consumed` on the way out, so the caller can carry on reading the tables it just sized.
inline bool body_len(std::FILE* f, uint64_t consumed, uint64_t* out) {
#if defined(_WIN32)
    if (std::fseek(f, 0, SEEK_END) != 0) return false;
    const long long sz = _ftelli64(f);
#else
    if (fseeko(f, 0, SEEK_END) != 0) return false;
    const off_t sz = ftello(f);
#endif
    if (sz < 0 || (uint64_t) sz < consumed) return false;
    if (std::fseek(f, (long) consumed, SEEK_SET) != 0) return false;
    *out = (uint64_t) sz - consumed;
    return true;
}

/// `v % p` for a positive p, always in [0, p): C++'s own % keeps the dividend's sign, and eg_issue folds the
/// result straight into a row index.
inline int64_t nonneg_mod(int64_t v, int64_t p) {
    const int64_t r = v % p;
    return r < 0 ? r + p : r;
}

/// The six header values, still signed, still unchecked.
struct EngramHeader {
    long long vocab = 0;
    long long layers = 0;
    long long ngram = 0;
    long long heads = 0;
    long long hd = 0;
    long long pad = 0;
};

/// What a header implies, in elements.  The loader resizes to these, so the guard has already proven the four
/// totals fit inside the file's body when accept() returns true.
struct EngramLayout {
    uint64_t vocab = 0;
    uint64_t layers = 0;
    uint64_t ngram = 0;
    uint64_t heads = 0;
    uint64_t hd = 0;
    int64_t pad = 0;
    uint64_t cols = 0;          ///< (ngram - 1) * heads: the Engram columns a token's row holds
    uint64_t map_elems = 0;
    uint64_t mult_elems = 0;
    uint64_t table_elems = 0;   ///< primes and offsets, one entry per layer and column
};

/// The ceilings that a file's length cannot bound.  Every one sits far above what a DeepSeek-V4.1 checkpoint
/// carries (a vocabulary of 67 million, 65536 Engram layers, 65536 heads, a head of a million values), so no
/// pack the writer produced can reach them, while a header that lies stops being an integer the loader stores.
/// ngram is the exception: it is bound to the fixed `int64_t prod[8]` in eg_issue, so it is the one real limit.
enum : uint64_t {
    kMaxVocab = 1ull << 26,
    kMaxLayers = 1ull << 16,
    kMaxNgram = 8,
    kMaxHeads = 1ull << 16,
    kMaxHd = 1ull << 20
};

/// Checks one engram.bin's header against the bytes that follow it.
class EngramGuard {
  public:
    /// `body_bytes` is what the file holds after the magic and the six values.
    explicit EngramGuard(uint64_t body_bytes) : body_bytes_(body_bytes) {}

    /// False when the writer could not have produced this header with this many bytes after it; `err` says how.
    bool accept(const EngramHeader& h, EngramLayout* out, std::string* err) const {
        EngramLayout lay;
        if (h.vocab <= 0 || h.layers <= 0 || h.ngram <= 0 || h.heads <= 0 || h.hd <= 0)
            return fail("its header carries a zero or negative vocab/layers/ngram/heads/hd", err);
        if ((uint64_t) h.vocab > kMaxVocab) return fail("it claims a vocabulary of " + std::to_string(h.vocab), err);
        if ((uint64_t) h.layers > kMaxLayers) return fail("it claims " + std::to_string(h.layers) + " Engram layers", err);
        if ((uint64_t) h.ngram > kMaxNgram)
            return fail("it declares a max ngram of " + std::to_string(h.ngram) + ", past the " +
                        std::to_string(kMaxNgram) + " look-backs the row's hash holds", err);
        if ((uint64_t) h.heads > kMaxHeads) return fail("it claims " + std::to_string(h.heads) + " Engram heads", err);
        if ((uint64_t) h.hd > kMaxHd) return fail("it claims a head of " + std::to_string(h.hd) + " values", err);
        // A head whose size is not a whole number of 32-value blocks cannot size the scale bytes that sit after the
        // codes (hd / 32 is used in every format's row layout), so the truncation would mis-size the reads.
        if (h.hd % 32 != 0)
            return fail("its head size " + std::to_string(h.hd) + " is not a whole number of 32-value blocks", err);

        lay.vocab = (uint64_t) h.vocab;
        lay.layers = (uint64_t) h.layers;
        lay.ngram = (uint64_t) h.ngram;
        lay.heads = (uint64_t) h.heads;
        lay.hd = (uint64_t) h.hd;
        lay.pad = (int64_t) h.pad;

        uint64_t cols = 0, map_bytes = 0, mult = 0, mult_bytes = 0, tables = 0, table_bytes = 0, body = 0;
        if (mul_overflow_u64(lay.ngram - 1, lay.heads, &cols)) return fail("its (ngram - 1) * heads overflows", err);
        if (mul_overflow_u64(lay.vocab, 4, &map_bytes)) return fail("its token map overflows", err);
        if (mul_overflow_u64(lay.layers, lay.ngram, &mult)) return fail("its layers * ngram overflows", err);
        if (mul_overflow_u64(mult, 8, &mult_bytes)) return fail("its multipliers overflow", err);
        if (mul_overflow_u64(lay.layers, cols, &tables)) return fail("its layers * cols overflows", err);
        if (mul_overflow_u64(tables, 8, &table_bytes)) return fail("its primes and offsets overflow", err);
        if (add_overflow_u64(map_bytes, mult_bytes, &body)) return fail("its tables overflow", err);
        if (add_overflow_u64(body, table_bytes, &body)) return fail("its tables overflow", err);
        // primes and offsets are two tables of the same count
        if (add_overflow_u64(body, table_bytes, &body)) return fail("its tables overflow", err);
        if (body != body_bytes_)
            return fail("its header implies " + std::to_string(body) + " bytes of tables, but the file holds " +
                        std::to_string(body_bytes_), err);

        lay.cols = cols;
        lay.map_elems = lay.vocab;
        lay.mult_elems = mult;
        lay.table_elems = tables;
        if (out) *out = lay;
        return true;
    }

    /// False when one layer's (prime, offset) pairs could address a row outside a table of `rows` rows, because
    /// eg_issue reads at `table_off + id * row_bytes` with id from a pair.  `primes` and `offs` hold one entry per
    /// layer and column, so the layer's own entries start at `layer * cols`.
    static bool pairs_in_table(const std::vector<int64_t>& primes, const std::vector<int64_t>& offs,
                               uint64_t layer, uint64_t cols, uint64_t rows, const std::string& what,
                               std::string* err) {
        for (uint64_t c = 0; c < cols; ++c) {
            const uint64_t j = layer * cols + c;
            if (j >= primes.size() || j >= offs.size())
                return fail(what + " has no prime and offset for column " + std::to_string(c), err);
            const int64_t prime = primes[j], off = offs[j];
            if (prime < 1)
                return fail(what + " carries a prime of " + std::to_string(prime) +
                            " in column " + std::to_string(c) + ", which the row hash divides by", err);
            if (off < 0)
                return fail(what + " carries a negative row offset " + std::to_string(off) + " in column " +
                            std::to_string(c), err);
            if ((uint64_t) off >= rows || (uint64_t) prime > rows - (uint64_t) off)
                return fail(what + " addresses rows " + std::to_string(off) + " to " +
                            std::to_string((uint64_t) off + (uint64_t) prime - 1) + " of a table of " +
                            std::to_string(rows) + " rows (column " + std::to_string(c) + ")", err);
        }
        return true;
    }

  private:
    static bool fail(const std::string& what, std::string* err) {
        if (err) *err = "engram.bin " + what;
        return false;
    }
    uint64_t body_bytes_ = 0;
};

}  // namespace dsv41
}  // namespace strata
