// include/strata/core/df2_pack_guard.hpp - the DF2P drafter pack's index guard, shared by the CUDA/HIP, SYCL and
// Metal loaders (src/core/dflash2.cpp, sycl/src/core/dflash2.cpp, metal/src/engine/dflash2.inc).
//
// WHY THIS EXISTS.  All three loaders read a text index line
//
//     t <name> <dtype> <rows> <cols> <offset> <bytes>
//
// with sscanf into SIGNED long long and then cast the offset and the byte count straight to uint64_t.  A negative
// value becomes a huge one, so `total = std::max(total, off + bytes)` can wrap round to a small number while the
// loader still seeks to `data_off + off` and writes at `blob_ + off`: a small allocation and a copy far outside it.
// The quantized branches make it worse, because the tile kernels take their destination and their element count
// from rows and cols (df2_q8_tile / df2_q4_tile), so free rows and cols mean free writes on the device.  Before
// this guard the only gate ahead of the allocation checked the cfg scalars, never the per-tensor geometry.
//
// A pack is a file the person may obtain from outside: --draft-model takes any path, tools/dflash2_pack.py
// --convert-selector rebuilds one from a community GGUF, and STRATA_DFLASH2_TUNED_URL names an alternate download
// address (strata_control/drafting.py).  The fetched packs are SHA-256-pinned there, which is why this guard is a
// wall behind a lock rather than the only one.
//
// WHAT MAKES A PACK VALID.  The writer (tools/dflash2_pack.py) is the contract: tensors follow each other with no
// overlap, each starting on a 256-byte boundary, and for the two quantized types the declared byte count is exactly
// the size of the two planes the kernels write:
//
//     q8_0   rows * (cols / 32) * 34   = the qs plane (rows*cols bytes) + the d plane (rows*cols/32 * 2 bytes)
//     q4_0   rows * (cols / 32) * 18   = the nibble plane (rows*cols/2)  + the d plane (rows*cols/32 * 2 bytes)
//     f32    rows * cols * 4           (a vector or a folded row set carries its leading dims in rows, so cols may be 1)
//     bf16   rows * cols * 2
//
// So requiring the declared byte count to EQUAL what the geometry implies, and requiring the tensor to sit inside
// the data section, bounds every plane the kernels touch by construction: the extents they derive from rows and cols
// land on bytes already proven to be inside blob_.  Same style as the GGUF reader in src/split/src/gguf.cpp, with
// one difference - that file is built by GCC/Clang, while src/core/dflash2.cpp is host-compiled by MSVC on Windows,
// where __builtin_mul_overflow does not exist.  Hence the checked helpers below.
//
// It reports the first problem and names the tensor, so a bad pack says which line is wrong.

#pragma once

#include <cstdint>
#include <cstdio>
#include <string>

namespace strata {
namespace df2 {

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

/// The size of `f` in bytes, false when it cannot be had.
inline bool file_size(std::FILE* f, uint64_t* out) {
#if defined(_WIN32)
    if (std::fseek(f, 0, SEEK_END) != 0) return false;
    const long long sz = _ftelli64(f);
#else
    if (fseeko(f, 0, SEEK_END) != 0) return false;
    const off_t sz = ftello(f);
#endif
    if (sz < 0) return false;
    *out = (uint64_t) sz;
    return true;
}

/// The geometry of one `t` line as parsed: still signed, still unchecked.
struct TensorDecl {
    std::string name;
    std::string dtype;
    long long rows = 0;
    long long cols = 0;
    long long off = 0;
    long long bytes = 0;
};

/// Checks the `t` lines of one pack's index as they are read, in the order they appear (the writer emits them in
/// offset order, so a monotonic run rules out overlap too).  `data_len` is the number of bytes that follow the
/// index, which is what blob_ is sized from once every declaration has been accepted.
class IndexGuard {
  public:
    explicit IndexGuard(uint64_t data_len) : data_len_(data_len) {}

    /// False on the first declaration the writer could not have produced; `err` then names it.
    /// On success `end` receives off+bytes, which is what the caller folds into the allocation total.
    bool accept(const TensorDecl& t, uint64_t* end, std::string* err) {
        if (end) *end = 0;
        if (t.rows < 0 || t.cols < 0 || t.off < 0 || t.bytes < 0) {
            return fail("tensor " + t.name + " has a negative rows/cols/offset/bytes", err);
        }
        const uint64_t rows = (uint64_t) t.rows, cols = (uint64_t) t.cols, off = (uint64_t) t.off, bytes = (uint64_t) t.bytes;
        if (rows == 0 || cols == 0) return fail("tensor " + t.name + " has an empty shape", err);

        uint64_t want = 0;
        if (t.dtype == "q8_0" || t.dtype == "q4_0") {
            // The tile kernels split each row into blocks of 32 and divide by the block count, so a row that is not
            // a whole number of blocks is a divide-by-zero on the device before it is anything else.
            if (cols % 32 != 0) {
                return fail("tensor " + t.name + " (" + t.dtype + ") has " + std::to_string(cols) +
                            " columns, not a whole number of 32-value blocks", err);
            }
            uint64_t blocks = 0;
            if (mul_overflow_u64(rows, cols / 32, &blocks)) return fail("tensor " + t.name + "'s block count overflows", err);
            if (mul_overflow_u64(blocks, t.dtype == "q8_0" ? 34ull : 18ull, &want)) {
                return fail("tensor " + t.name + "'s size overflows", err);
            }
        } else if (t.dtype == "f32" || t.dtype == "bf16") {
            uint64_t elements = 0;
            if (mul_overflow_u64(rows, cols, &elements)) return fail("tensor " + t.name + "'s shape overflows", err);
            if (mul_overflow_u64(elements, t.dtype == "f32" ? 4ull : 2ull, &want)) {
                return fail("tensor " + t.name + "'s size overflows", err);
            }
        }
        // Any other type reaches the loader's plain copy, which only needs its bytes inside the blob, so the exact
        // byte rule is skipped for it rather than refusing a pack the loader could otherwise read.
        const bool exact = want != 0;

        // The byte count the index declares is the one the copy uses, and for the four types the writer emits it is
        // what the planes derived from rows and cols are compared against, so an exact match bounds their writes.
        if (exact && bytes != want) {
            return fail("tensor " + t.name + " (" + t.dtype + ", " + std::to_string(rows) + "x" + std::to_string(cols) +
                        ") declares " + std::to_string(bytes) + " bytes; its shape needs " + std::to_string(want), err);
        }

        uint64_t e = 0;
        if (add_overflow_u64(off, bytes, &e)) return fail("tensor " + t.name + "'s offset and size overflow", err);
        if (off % 256 != 0) {
            return fail("tensor " + t.name + " starts at " + std::to_string(off) +
                        ", not 256-byte aligned as the pack writer leaves it", err);
        }
        if (e > data_len_) {
            return fail("tensor " + t.name + " runs to " + std::to_string(e) + " bytes, past the " +
                        std::to_string(data_len_) + " bytes the pack holds after its index", err);
        }
        if (off < next_off_) {
            return fail("tensor " + t.name + " starts at " + std::to_string(off) + ", inside the tensor before it", err);
        }
        next_off_ = e;
        if (end) *end = e;
        return true;
    }

  private:
    static bool fail(const std::string& what, std::string* err) {
        if (err) *err = "dflash2: " + what;
        return false;
    }
    uint64_t data_len_ = 0;
    uint64_t next_off_ = 0;
};

}  // namespace df2
}  // namespace strata
