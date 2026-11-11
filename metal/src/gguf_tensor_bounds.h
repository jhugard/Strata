// metal/src/gguf_tensor_bounds.h - the block geometry table, and the arithmetic that turns a GGUF tensor's
// declared shape into a payload size without overflowing and without exiting on a type this build does not
// know.  Header-only and free of system headers on purpose: metal/src/gguf_lite.cpp needs mmap, so the CPU
// test (tests/core/metal_tensor_bounds_test.cpp) checks this logic directly instead, on any platform.
//
// The table is the one metal/src/ref_quant.cpp held inline in ref::block_elems / ref::block_bytes
// (transcribed from ggml's block structs at the pinned 3cf03257).  It moved here unchanged so a reader can
// ask the same question without block_bytes' exit(2) on an unknown type, which a file chosen by an
// attacker must never be able to trigger in a process that is loading it.  ref_quant.cpp now reads the
// table through these two lines, so there is still one copy of it.
//
// The same rule already ships in the CUDA / CPU reader (include/strata/artifact/gguf_reader.hpp:
// block_geometry, tensor_payload_bytes, GgufModel::in_bounds); this is the Metal reader's copy of it.
#pragma once

#include "ref_quant.h"

#include <cstdint>
#include <limits>
#include <vector>

namespace ref {

// Elements per block and bytes per block for a GGML type id; false when this build has no dequantizer for
// the type.  Equivalent to the old block_elems / block_bytes pair, which is still what the kernels use.
inline bool block_geometry(int t, int& elems, int& bytes) {
    switch (t) {
        case F32:                          elems = 1;   bytes = 4;   return true;
        case F16: case BF16:               elems = 1;   bytes = 2;   return true;
        case Q4_0: case IQ4_NL:            elems = 32;  bytes = 18;  return true;
        case Q5_0:                         elems = 32;  bytes = 22;  return true;
        case Q5_1:                         elems = 32;  bytes = 24;  return true;
        case Q8_0:                         elems = 32;  bytes = 34;  return true;
        case Q2_0:                         elems = 64;  bytes = 18;  return true;
        case TQ2_T:                        elems = 128; bytes = 34;  return true;
        case TQK6:                         elems = 128; bytes = 26;  return true;
        case TQK7:                         elems = 128; bytes = 30;  return true;
        case Q3_K:                         elems = 256; bytes = 110; return true;
        case Q4_K:                         elems = 256; bytes = 144; return true;
        case Q5_K:                         elems = 256; bytes = 176; return true;
        case Q6_K:                         elems = 256; bytes = 210; return true;
        case IQ2_XXS:                      elems = 256; bytes = 66;  return true;
        case IQ2_XS:                       elems = 256; bytes = 74;  return true;
        case IQ2_S:                        elems = 256; bytes = 82;  return true;
        case IQ3_XXS:                      elems = 256; bytes = 98;  return true;
        case IQ3_S:                        elems = 256; bytes = 110; return true;
        case IQ4_XS:                       elems = 256; bytes = 136; return true;
        case IQ1_S:                        elems = 256; bytes = 50;  return true;
        case IQ1_M:                        elems = 256; bytes = 56;  return true;
    }
    return false;
}

}  // namespace ref

namespace gguf {

// Bytes a tensor's payload occupies, from its type and shape: elements divided into whole blocks, times the
// block size.  0 when the type is unknown, a row is not a whole number of blocks (ggml asserts it is for a
// quantized tensor, so a file that says otherwise is not readable by this engine), or the count or the
// product overflows.
inline uint64_t tensor_payload_bytes(int type, const std::vector<uint64_t>& shape) {
    int be = 0, bb = 0;
    if (shape.empty() || !ref::block_geometry(type, be, bb) || shape[0] % (uint64_t) be) return 0;
    uint64_t elements = 1;
    for (uint64_t d : shape) {
        if (d == 0 || elements > (std::numeric_limits<uint64_t>::max)() / d) return 0;
        elements *= d;
    }
    const uint64_t blocks = elements / (uint64_t) be;
    if (blocks > (std::numeric_limits<uint64_t>::max)() / (uint64_t) bb) return 0;
    return blocks * (uint64_t) bb;
}

// True when [offset, offset + bytes) lies wholly inside a payload of `payload` bytes, without overflowing.
// A zero-length tensor is out of bounds: every caller here reads the tensor it is given.
inline bool tensor_in_bounds(uint64_t offset, uint64_t bytes, uint64_t payload) {
    return bytes != 0 && offset <= payload && bytes <= payload - offset;
}

// A byte count rounded up to the 256-byte multiple every reader here stages at; 0 when the rounding itself
// would overflow, which a caller must treat as a refusal rather than as a zero-sized tensor.
inline uint64_t pad_up256(uint64_t bytes) {
    return bytes > (std::numeric_limits<uint64_t>::max)() - 255 ? 0 : (bytes + 255) / 256 * 256;
}

}  // namespace gguf
