// tests/core/metal_tensor_bounds_test.cpp - the Metal reader's tensor sizing and bounds arithmetic
// (metal/src/gguf_tensor_bounds.h) on hand-written shapes and offsets: no model, no GPU, no Metal.
//
// Why these cases.  metal/src/gguf_lite.cpp used to size a tensor with `n = n * d` over the shape and then
// `n / block_elems * block_bytes`, with no check on either product, no check that the row is a whole number
// of blocks, and no check of the result (or the tensor's declared offset) against the file.  A file could
// therefore claim a payload size that wrapped to a small number, or an offset past EOF.  Both numbers feed
// the readers downstream: gguf_lite's Tensor::bytes is the byte count of every pread the engine issues, and
// metal/src/engine/stream.cpp sums it to size the buffer those reads land in.
//
// The table must also still agree with metal/src/ref_quant.cpp's block_elems / block_bytes for every type
// the dequantizers handle, because a tensor sized from one table and dequantized against the other would
// read the wrong number of bytes.  ref_quant's block_bytes exits on an unknown type, so the table lives in
// the header and ref_quant.cpp reads it; the first group below pins both halves of that contract.

#include "gguf_tensor_bounds.h"

#include <cstdio>
#include <string>
#include <vector>

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-92s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

const uint64_t kMax = (std::numeric_limits<uint64_t>::max)();

using gguf::pad_up256;
using gguf::tensor_in_bounds;
using gguf::tensor_payload_bytes;

std::vector<uint64_t> shape2(uint64_t a, uint64_t b) { return {a, b}; }
std::vector<uint64_t> shape3(uint64_t a, uint64_t b, uint64_t c) { return {a, b, c}; }

// ---- the table against ref_quant's block_elems / block_bytes --------------------------------------------
//
// Every type the Metal dequantizers know, with the (elems, bytes) pair metal/src/ref_quant.cpp's block_elems
// / block_bytes held inline before this table was extracted from them.  Those two functions now read this
// table (block_bytes returns its byte count, block_elems keeps its 256 default for an unknown type), so the
// two agree by construction and a snapshot of the old values is the check that the extraction was faithful.
// block_bytes exits on an unknown type, so it cannot be called from a portable test.
void table_matches_ref_quant() {
    std::printf("block_geometry carries the table ref_quant's block_elems / block_bytes had:\n");
    struct Row { int type; int elems; int bytes; const char* name; };
    const Row rows[] = {
        {ref::F32, 1, 4, "F32"},          {ref::F16, 1, 2, "F16"},          {ref::BF16, 1, 2, "BF16"},
        {ref::Q4_0, 32, 18, "Q4_0"},      {ref::Q5_0, 32, 22, "Q5_0"},      {ref::Q5_1, 32, 24, "Q5_1"},
        {ref::Q8_0, 32, 34, "Q8_0"},      {ref::IQ4_NL, 32, 18, "IQ4_NL"},  {ref::Q2_0, 64, 18, "Q2_0"},
        {ref::TQ2_T, 128, 34, "TQ2_T"},   {ref::TQK6, 128, 26, "TQK6"},     {ref::TQK7, 128, 30, "TQK7"},
        {ref::Q3_K, 256, 110, "Q3_K"},    {ref::Q4_K, 256, 144, "Q4_K"},    {ref::Q5_K, 256, 176, "Q5_K"},
        {ref::Q6_K, 256, 210, "Q6_K"},    {ref::IQ2_XXS, 256, 66, "IQ2_XXS"}, {ref::IQ2_XS, 256, 74, "IQ2_XS"},
        {ref::IQ2_S, 256, 82, "IQ2_S"},   {ref::IQ3_XXS, 256, 98, "IQ3_XXS"}, {ref::IQ3_S, 256, 110, "IQ3_S"},
        {ref::IQ4_XS, 256, 136, "IQ4_XS"}, {ref::IQ1_S, 256, 50, "IQ1_S"},  {ref::IQ1_M, 256, 56, "IQ1_M"},
    };
    for (const Row& r : rows) {
        int be = 0, bb = 0;
        const bool ok = ref::block_geometry(r.type, be, bb);
        check(ok && be == r.elems && bb == r.bytes,
              std::string(r.name) + " (" + std::to_string(r.type) + ") is " + std::to_string(r.elems) +
                  " elems / " + std::to_string(r.bytes) + " B");
    }
    // the count above is every case in ref_quant.cpp's two switches
    check(sizeof(rows) / sizeof(rows[0]) == 24, "the snapshot covers all 24 types in ref_quant's table");
}

// An unknown type has no geometry and must never reach block_bytes' exit(2) through the reader.
void table_refuses_unknown_types() {
    std::printf("block_geometry refuses a type this build has no dequantizer for:\n");
    for (const int t : {-1, 4, 5, 15, 24, 25, 26, 27, 28, 31, 41, 43, 100, 143, 148, 999}) {
        int be = 0, bb = 0;
        check(!ref::block_geometry(t, be, bb), "type " + std::to_string(t) + " is refused");
        check(be == 0 && bb == 0, "type " + std::to_string(t) + " leaves the outputs untouched");
    }
}

// ---- tensor_payload_bytes -------------------------------------------------------------------------------
void sizes_known_types() {
    std::printf("tensor_payload_bytes sizes a well-formed tensor as its blocks:\n");
    // F32: 1 element per block, 4 B each.
    check(tensor_payload_bytes(ref::F32, {4096}) == 4096 * 4, "F32 [4096] is 16384 B");
    check(tensor_payload_bytes(ref::F16, shape2(2048, 2048)) == 2048ull * 2048 * 2, "F16 [2048,2048]");
    check(tensor_payload_bytes(ref::BF16, shape2(2048, 2048)) == 2048ull * 2048 * 2, "BF16 [2048,2048]");
    // 32-blocks: n / 32 * bytes.
    check(tensor_payload_bytes(ref::Q4_0, shape2(4096, 4096)) == 4096ull * 4096 / 32 * 18, "Q4_0 [4096,4096]");
    check(tensor_payload_bytes(ref::Q8_0, shape2(2560, 2560)) == 2560ull * 2560 / 32 * 34, "Q8_0 [2560,2560]");
    check(tensor_payload_bytes(ref::IQ4_NL, shape2(4096, 1)) == 4096ull / 32 * 18, "IQ4_NL [4096,1]");
    // Q2_0 is the 64-element, 18-byte block (docs/q2_0-contract.md).
    check(tensor_payload_bytes(ref::Q2_0, shape2(4096, 4096)) == 4096ull * 4096 / 64 * 18, "Q2_0 [4096,4096]");
    // 256-blocks, the quants a Q3-era model carries.
    check(tensor_payload_bytes(ref::IQ3_S, shape2(4096, 4096)) == 4096ull * 4096 / 256 * 110, "IQ3_S [4096,4096]");
    check(tensor_payload_bytes(ref::Q3_K, shape2(4096, 4096)) == 4096ull * 4096 / 256 * 110, "Q3_K [4096,4096]");
    check(tensor_payload_bytes(ref::Q4_K, shape2(4096, 4096)) == 4096ull * 4096 / 256 * 144, "Q4_K [4096,4096]");
    check(tensor_payload_bytes(ref::Q6_K, shape2(4096, 4096)) == 4096ull * 4096 / 256 * 210, "Q6_K [4096,4096]");
    check(tensor_payload_bytes(ref::IQ1_M, shape2(4096, 4096)) == 4096ull * 4096 / 256 * 56, "IQ1_M [4096,4096]");
    // the trellis types (Gyro): 128 values a block
    check(tensor_payload_bytes(ref::TQK6, shape2(4096, 4096)) == 4096ull * 4096 / 128 * 26, "TQK6 [4096,4096]");
    // a 3-D expert tensor, the shape the MoE loader takes
    check(tensor_payload_bytes(ref::Q4_K, shape3(4096, 4096, 512)) == 4096ull * 4096 / 256 * 144 * 512,
          "Q4_K [4096,4096,512]");
    // a single block
    check(tensor_payload_bytes(ref::Q4_K, {256}) == 144, "Q4_K [256] is one block");
}

void refuses_unknown_types() {
    std::printf("tensor_payload_bytes is 0 for a type this build cannot size (the old code exited):\n");
    for (const int t : {-1, 4, 5, 15, 24, 31, 41, 43, 999})
        check(tensor_payload_bytes(t, shape2(4096, 4096)) == 0, "type " + std::to_string(t) + " gives 0 bytes");
}

void refuses_partial_rows() {
    std::printf("tensor_payload_bytes is 0 when the contiguous row is not whole blocks:\n");
    check(tensor_payload_bytes(ref::Q4_K, {4095, 4096}) == 0, "Q4_K [4095,4096] (4095 % 256)");
    check(tensor_payload_bytes(ref::Q4_0, {33, 128}) == 0, "Q4_0 [33,128] (33 % 32)");
    check(tensor_payload_bytes(ref::Q2_0, {194, 64}) == 0, "Q2_0 [194,64] (194 % 64)");
    check(tensor_payload_bytes(ref::TQK6, {200, 512}) == 0, "TQK6 [200,512] (200 % 128)");
    // shape[1] may be anything; only the contiguous row must divide.
    check(tensor_payload_bytes(ref::Q4_K, shape2(4096, 7)) == 4096ull * 7 / 256 * 144, "Q4_K [4096,7] still sizes");
}

void refuses_degenerate_shapes() {
    std::printf("tensor_payload_bytes is 0 for a shape it cannot multiply out:\n");
    check(tensor_payload_bytes(ref::F32, {}) == 0, "an empty shape gives 0 bytes");
    check(tensor_payload_bytes(ref::F32, {0}) == 0, "a zero dimension gives 0 bytes (not an empty tensor)");
    check(tensor_payload_bytes(ref::F32, shape2(4096, 0)) == 0, "a zero in a later dimension gives 0 bytes");
}

void refuses_overflow() {
    std::printf("tensor_payload_bytes is 0 when the element count or the product overflows:\n");
    // the old code wrapped these to a small number and the reader happily "read" that many bytes.
    check(tensor_payload_bytes(ref::F32, shape2(kMax, 4)) == 0, "F32 [2^64-1,4] overflows the element count");
    check(tensor_payload_bytes(ref::F32, {kMax}) == 0, "F32 [2^64-1] overflows the product");
    check(tensor_payload_bytes(ref::Q8_0, {kMax & ~31ull}) == 0, "Q8_0 [~2^64] overflows the byte product");
    check(tensor_payload_bytes(ref::F32, shape3(1ull << 34, 1ull << 34, 4)) == 0, "F32 [2^34,2^34,4] overflows");
    // Just under each limit must still size, or the guard is a ceiling and not an overflow check.
    check(tensor_payload_bytes(ref::F32, {1ull << 32}) == (1ull << 32) * 4, "F32 [2^32] sizes (2^34 B)");
}

// ---- tensor_in_bounds -----------------------------------------------------------------------------------
void bounds_accept_inside() {
    std::printf("tensor_in_bounds accepts a span wholly inside the payload:\n");
    check(tensor_in_bounds(0, 1024, 1024), "a tensor filling the payload");
    check(tensor_in_bounds(0, 1, 1), "one byte at offset 0 of a one-byte payload");
    check(tensor_in_bounds(1024, 1024, 4096), "a tensor in the middle");
    check(tensor_in_bounds(4096 - 1024, 1024, 4096), "a tensor ending exactly at the end");
    check(tensor_in_bounds(4096, 0, 4096) == false, "zero bytes is refused (every caller reads what it is given)");
}

void bounds_refuse_outside() {
    std::printf("tensor_in_bounds refuses a span that leaves the payload:\n");
    check(!tensor_in_bounds(1024, 1024, 1024), "one whole tensor past the end");
    check(!tensor_in_bounds(1023, 2, 1024), "one byte past the end");
    check(!tensor_in_bounds(kMax, 1, 4096), "an offset at the top of the range");
    check(!tensor_in_bounds(0, kMax, 4096), "a byte count of 2^64-1");
    // the wrapping the old `offset + bytes <= payload` form would have allowed
    check(!tensor_in_bounds(4096, kMax - 4095, 4096), "offset+bytes that wraps round to a small number");
    check(!tensor_in_bounds(2048, kMax, 4096), "a byte count that wraps past the offset");
    check(!tensor_in_bounds(0, 16ull << 30, 4ull << 30), "a 16 GiB tensor in a 4 GiB file");
}

void bounds_on_an_empty_payload() {
    std::printf("tensor_in_bounds refuses everything in a file with no data section:\n");
    check(!tensor_in_bounds(0, 1, 0), "one byte in an empty payload");
    check(!tensor_in_bounds(0, 0, 0), "not even a zero-length tensor");
}

// ---- the two together, the way the reader uses them ------------------------------------------------------
//
// What the fixed open() loop does per tensor, and what the old one did not: size it, then bound it.  These
// are the shapes an attacker writes into a small file.
void reader_rejects_a_small_file() {
    std::printf("a tensor an attacker wrote into a 4 MiB file is refused:\n");
    const uint64_t payload = 4ull << 20;
    auto refuses = [&](int type, const std::vector<uint64_t>& dims, uint64_t off, const char* what) {
        const uint64_t bytes = tensor_payload_bytes(type, dims);
        const bool bad = bytes == 0 || !tensor_in_bounds(off, bytes, payload);
        check(bad, what);
    };
    refuses(ref::Q4_K, shape2(4096, 4096), 0, "a 4096x4096 Q4_K (23.6 MB) claimed at offset 0");
    refuses(ref::IQ3_S, shape2(4096, 4096), payload - 16, "an IQ3_S matrix starting 16 B before EOF");
    refuses(ref::F32, shape2(1024, 1024), payload, "an F32 matrix starting exactly at EOF");
    refuses(ref::F32, shape2(1024, 1024), kMax, "an F32 matrix at offset 2^64-1");
    refuses(ref::Q4_K, shape2(4096, 4096), kMax - 100, "an offset that wraps when added to");
    refuses(999, shape2(4096, 4096), 0, "a made-up tensor type");
    refuses(ref::Q4_K, {4095, 4096}, 0, "a row that is not whole blocks");
    // and the same file's honest tensor is still accepted
    const uint64_t bytes = tensor_payload_bytes(ref::Q4_0, shape2(4096, 256));
    check(bytes == 4096ull * 256 / 32 * 18, "a 4096x256 Q4_0 is 737280 B");
    check(tensor_in_bounds(0, bytes, payload), "and fits the 4 MiB file");
}

// What stream.cpp sums: the padded per-tensor total that sizes the buffer the reads land in.
void padded_total_is_checked() {
    std::printf("the padded total in load_dense cannot wrap unseen:\n");
    check(pad_up256(1) == 256 && pad_up256(256) == 256 && pad_up256(257) == 512, "a byte count rounds up to 256");
    check(pad_up256(kMax) == 0, "a count whose padding itself would overflow reads as 0 (a refusal)");
    check(pad_up256(kMax & ~255ull) == (kMax & ~255ull), "a large padded count is stable");
    // the check the loop now makes before each +=
    const uint64_t big = kMax & ~255ull;
    check(pad_up256(256) > kMax - big, "adding one more tensor to a total at the top of the range is detected");
}

}  // namespace

int main() {
    table_matches_ref_quant();
    table_refuses_unknown_types();
    sizes_known_types();
    refuses_unknown_types();
    refuses_partial_rows();
    refuses_degenerate_shapes();
    refuses_overflow();
    bounds_accept_inside();
    bounds_refuse_outside();
    bounds_on_an_empty_payload();
    reader_rejects_a_small_file();
    padded_total_is_checked();
    std::printf("%s\n", g_fail == 0 ? "all metal tensor bounds checks passed" : "FAILED");
    return g_fail == 0 ? 0 : 1;
}
