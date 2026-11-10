// tests/core/df2_pack_guard_test.cpp - the DF2P drafter pack's index guard (include/strata/core/df2_pack_guard.hpp)
// on hand-written index lines: no model, no GPU, no pack file.
//
// Why these cases.  The loaders used to cast the index's offset and byte count straight from signed long long to
// uint64_t, so a negative offset became a huge one, `total = std::max(total, off + bytes)` wrapped round to a small
// number, blob_ was allocated that small, and the loader still wrote at `blob_ + off`.  The quantized branches went
// further: df2_q8_tile and df2_q4_tile take their destination and their element count from the index's rows and
// cols, so free geometry means free writes.  The guard has to reject every line the writer in
// tools/dflash2_pack.py could not have produced, and accept everything it does produce - a guard that refused a real
// pack would be worse than no guard, which is why the first group below is a pack's worth of valid lines.
//
// The sizes come from the writer's own byte counts (df2_pack_guard.hpp's header comment): q8_0 is rows*(cols/32)*34,
// q4_0 is rows*(cols/32)*18, f32 is rows*cols*4, bf16 is rows*cols*2.  A vector carries its leading dims in rows and
// cols == 1, which the block rule must NOT reject.

#include "strata/core/df2_pack_guard.hpp"

#include <cstdio>
#include <string>
#include <vector>

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-92s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

using strata::df2::IndexGuard;
using strata::df2::TensorDecl;

const uint64_t kDataLen = 0x10000;   // the bytes a pack holds after its index

TensorDecl t(std::string name, std::string dtype, long long rows, long long cols, long long off, long long bytes) {
    TensorDecl d;
    d.name = std::move(name); d.dtype = std::move(dtype);
    d.rows = rows; d.cols = cols; d.off = off; d.bytes = bytes;
    return d;
}

/// Accepts every line in order; returns the guard's verdict on the run, the last error and the largest off+bytes.
bool run(const std::vector<TensorDecl>& ts, uint64_t data_len, std::string* err, uint64_t* total) {
    IndexGuard guard(data_len);
    uint64_t tot = 0;
    for (const TensorDecl& d : ts) {
        uint64_t end = 0;
        if (!guard.accept(d, &end, err)) { if (total) *total = tot; return false; }
        if (end > tot) tot = end;
    }
    if (total) *total = tot;
    return true;
}

/// A pack the writer could have produced: a Q8_0 matrix, an f32 vector (cols == 1), an f32 folded row set, a Q4_0
/// matrix, a Q4_0 selector codebook and a bf16 codebook.  256-aligned offsets with the writer's padding between them.
std::vector<TensorDecl> valid_pack() {
    return {
        t("fc.weight", "q8_0", 4, 2560, 0, 4 * (2560 / 32) * 34),                    //      0 .. 10880
        t("norm.weight", "f32", 2560, 1, 11008, 2560 * 4),                           //  11008 .. 21248
        t("layers.0.attention_conv.base_kernel", "f32", 2, 16, 21504, 2 * 16 * 4),   //  21504 .. 21632
        t("mlp.gate_proj.weight", "q4_0", 2, 2560, 21760, 2 * (2560 / 32) * 18),     //  21760 .. 24640
        t("candidate_selector.predecessor_codebook", "q4_0", 2, 256, 24832, 2 * (256 / 32) * 18),   // 24832 .. 25120
        t("candidate_selector.successor_codebook", "bf16", 8, 256, 25344, 8 * 256 * 2),             // 25344 .. 29440
    };
}
}  // namespace

int main() {
    std::printf("df2_pack_guard_test\n");

    // ---- what the writer produces must pass ----------------------------------
    {
        std::string err;
        uint64_t total = 0;
        const bool ok = run(valid_pack(), kDataLen, &err, &total);
        check(ok, "a pack the writer could have produced is accepted" + (ok ? std::string() : (": " + err)));
        check(total == 29440, "the total is the largest off+bytes (29440), got " + std::to_string(total));
    }
    {
        // The rule the loaders depend on: cols == 1 for a vector is normal, and only the quantized types need
        // whole 32-value blocks.
        std::string err;
        check(run({t("v", "f32", 2560, 1, 0, 2560 * 4)}, kDataLen, &err, nullptr), "an f32 vector with cols == 1 is accepted");
    }
    {
        std::string err;
        check(run({t("c", "bf16", 128, 256, 0, 128 * 256 * 2)}, kDataLen, &err, nullptr), "a bf16 selector codebook is accepted");
    }
    {
        // Compatibility: an index naming a type the loaders copy through their plain branch still loads, as long as
        // its bytes sit inside the blob.  A plain copy cannot write more than it copies, so containment is enough
        // for it; refusing such a pack would change what the engines accept today.
        std::string err;
        check(run({t("odd", "f16", 8, 8, 0, 96)}, kDataLen, &err, nullptr),
              "a type the exact-size rule does not cover still loads when it is in bounds");
    }

    // ---- the shapes that used to reach the allocator -------------------------
    {
        // The audit's case: off == -1 and bytes == 16 wrap off+bytes to 8, so blob_ is 8 bytes while the copy lands
        // at blob_ - 1.
        std::string err;
        const bool ok = !run({t("fc.weight", "q8_0", 4, 2560, -1, 16)}, kDataLen, &err, nullptr);
        check(ok, "a negative offset is refused");
        check(err.find("fc.weight") != std::string::npos, "the message names the tensor: " + err);
    }
    {
        std::string err;
        check(!run({t("a", "q8_0", 4, 2560, 0, -64)}, kDataLen, &err, nullptr), "a negative byte count is refused");
        std::string err2;
        check(!run({t("a", "q8_0", -4, 2560, 0, 4 * 80 * 34)}, kDataLen, &err2, nullptr), "a negative row count is refused");
    }
    {
        // Bytes that disagree with the geometry: rows and cols are what the tile kernels derive their writes from,
        // so the declared count and the shape have to agree exactly.
        std::string err;
        const bool ok = !run({t("fc.weight", "q8_0", 4, 2560, 0, 4 * (2560 / 32) * 34 + 4096)}, kDataLen, &err, nullptr);
        check(ok, "a q8_0 tensor declaring more bytes than its shape needs is refused");
        check(err.find("needs") != std::string::npos, "the message states the size the shape needs: " + err);
        std::string err2;
        check(!run({t("m", "q4_0", 2, 2560, 0, 2 * (2560 / 32) * 18 - 18)}, kDataLen, &err2, nullptr),
              "a q4_0 tensor declaring fewer bytes than its shape needs is refused");
    }
    {
        // Free geometry: the kernels' plane extents are rows*cols and rows*cols/2, which the exact-size rule ties to
        // the declared bytes, and the declared bytes must then sit inside the blob.  A row count no drafter has
        // cannot pass both, so the kernel can no longer be handed an extent past the allocation.
        std::string err;
        const long long rows = (long long) (1ull << 40);
        const long long bytes = (long long) ((1ull << 40) * 80ull * 18ull);
        const bool ok = !run({t("m", "q4_0", rows, 2560, 0, bytes)}, kDataLen, &err, nullptr);
        check(ok, "a q4_0 tensor with an enormous row count is refused");
        check(err.find("past") != std::string::npos, "the message says it runs past the data: " + err);
    }
    {
        // A row that is not a whole number of 32-value blocks makes the tile kernels divide by a zero block count.
        std::string err;
        check(!run({t("m", "q4_0", 2, 33, 0, 2 * (33 / 32) * 18)}, kDataLen, &err, nullptr),
              "a q4_0 tensor whose columns are not whole blocks is refused");
        std::string err2;
        check(!run({t("m", "q8_0", 2, 48, 0, 2 * 16 * 34)}, kDataLen, &err2, nullptr),
              "a q8_0 tensor whose columns are not whole blocks is refused");
    }
    {
        std::string err;
        check(!run({t("a", "q8_0", 1, 256, 100, 1 * 8 * 34)}, kDataLen, &err, nullptr),
              "a tensor that does not start on a 256-byte boundary is refused");
    }
    {
        // off+bytes past the bytes the pack actually holds: blob_ is sized from the data section, so anything beyond
        // it is outside the allocation.
        std::string err;
        check(!run({t("a", "q8_0", 64, 2560, 0, 64 * 80 * 34)}, kDataLen, &err, nullptr),
              "a tensor running past the pack's data section is refused");
    }
    {
        std::string err;
        check(!run({t("a", "f32", 128, 4, 0, 128 * 4 * 4), t("b", "f32", 128, 4, 512, 128 * 4 * 4)}, kDataLen, &err, nullptr),
              "a tensor starting inside the one before it is refused");
    }
    {
        // The largest offset and size the index's signed %lld parse can produce.  Their sum is the biggest thing the
        // old code folded into the allocation total, and it sits far past the pack, so the containment rule is what
        // refuses it.  It is worth stating out loud that it cannot wrap: two values at most LLONG_MAX add to
        // 2*LLONG_MAX, one short of UINT64_MAX, which is why the loaders' old `off + bytes` blew up by growing huge
        // rather than by turning small - and why the guard's own add check (fed through the helpers below) is
        // defensive against a future unsigned parse rather than the thing that saves today's loaders.
        std::string err;
        const bool ok = !run({t("a", "f16", 1, 1, 0x7fffffffffffff00LL, 0x7fffffffffffffffLL)}, kDataLen, &err, nullptr);
        check(ok, "an offset and size at the top of the signed parse are refused");
        check(err.find("past") != std::string::npos, "the message says it runs past the data: " + err);
    }

    // ---- the arithmetic helpers themselves ----------------------------------
    {
        uint64_t out = 0;
        check(strata::df2::mul_overflow_u64(1ULL << 32, 1ULL << 32, &out), "mul_overflow_u64 catches 2^32 * 2^32");
        check(!strata::df2::mul_overflow_u64(0, 1ULL << 63, &out) && out == 0, "mul_overflow_u64 lets a zero product through");
        check(!strata::df2::mul_overflow_u64(3, 5, &out) && out == 15, "mul_overflow_u64 returns the product when it fits");
        check(strata::df2::add_overflow_u64(~0ULL, 1, &out), "add_overflow_u64 catches UINT64_MAX + 1");
        check(!strata::df2::add_overflow_u64(1, 1, &out) && out == 2, "add_overflow_u64 returns the sum when it fits");
    }

    std::printf("%s\n", g_fail == 0 ? "df2_pack_guard_test: all cases pass" : "df2_pack_guard_test: FAILURES");
    return g_fail == 0 ? 0 : 1;
}
