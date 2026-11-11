// tests/core/gguf_tensor_bounds_test.cpp - GgufFile::in_bounds() and tensor_payload_bytes() on the two tensor
// directories the --control-vector-scaled loader used to accept and then read out of the file (audit finding 6).
// No model, no GPU.
//
// The loader checked metadata only: general.architecture, the `direction.<l>` name, the layer range, type f32 and
// `elements() == n_embd`.  `elements()` multiplies the shape unchecked and `tensor_data()` adds the declared
// offset with no bound of its own, so two shapes passed every check it made.  Both are reproduced here as bytes
// on disk, in the form the loader saw them, and the guard must refuse both.  A control-vector file that
// tools/../llama.cpp's writer would actually produce must still pass, unchanged.
#include "gguf_fixture.hpp"
#include "strata/artifact/gguf_reader.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-90s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

struct TempDir {
    fs::path path;
    TempDir() {
        path = fs::temp_directory_path() /
               ("strata-gguf-bounds-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

const uint64_t kN = 2560;  // the embedding length the loader compares elements() against

// --- raw GGUF v3 writer, for the directories the fixture writer cannot express -----------------------------
// fixture::write computes each tensor's offset and byte count from its shape, so it cannot emit a bogus offset
// or a shape whose product wraps round.  These two need the bytes spelled out.
struct Raw {
    std::vector<uint8_t> b;
    void put(const void* p, size_t n) { b.insert(b.end(), (const uint8_t*) p, (const uint8_t*) p + n); }
    void u32(uint32_t v) { put(&v, 4); }
    void u64(uint64_t v) { put(&v, 8); }
    void s(const std::string& v) { u64(v.size()); put(v.data(), v.size()); }
};

/// One tensor directory entry with an explicit offset, and the payload bytes to follow the aligned data start.
/// The path comes back as a string because that is what GgufFile opens.
std::string write_raw(const fs::path& path, const std::vector<uint64_t>& shape, uint64_t offset, size_t payload_bytes) {
    Raw r;
    r.u32(0x46554747u);  // "GGUF"
    r.u32(3);            // v3
    r.u64(1);            // one tensor
    r.u64(1);            // one metadata key
    r.s("general.architecture");
    r.u32(8);  // STRING
    r.s("controlvector");
    r.s("direction.1");
    r.u32((uint32_t) shape.size());
    for (uint64_t d : shape) r.u64(d);
    r.u32(0);  // f32
    r.u64(offset);
    while (r.b.size() % 32) r.b.push_back(0);  // the default 32-byte alignment
    r.b.resize(r.b.size() + payload_bytes, 0);
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write((const char*) r.b.data(), (std::streamsize) r.b.size());
    return path.string();
}

/// What the loader does with one tensor: metadata checks first, then the guard, then the read it used to make.
/// Returns "" when the tensor would have been read, and the loader's error text when the guard stops it.
std::string read_as_loader_does(const strata::GgufFile& f, int64_t L, int64_t N) {
    for (const strata::TensorInfo& t : f.tensors()) {
        if (t.name.rfind("direction.", 0) != 0) continue;
        const long l = std::strtol(t.name.c_str() + 10, nullptr, 10);
        if (l < 1 || l >= L) continue;
        if (t.type != 0 || t.elements() != (uint64_t) N) return "must be " + std::to_string(N) + " f32";
        const uint64_t bytes = strata::tensor_payload_bytes(t);
        if (!f.in_bounds(t) || bytes != (uint64_t) N * 4ull) return "does not fit in the file";
        const float* src = reinterpret_cast<const float*>(f.tensor_data(t));
        float sum = 0.0f;
        for (int64_t j = 0; j < N; ++j) sum += src[j];  // the read that used to run off the mapping
        (void) sum;
    }
    return "";
}

}  // namespace

int main() {
    TempDir tmp;
    std::printf("gguf_tensor_bounds_test\n");

    // A control-vector file the way a writer produces it: 48 layers' worth of direction.<l>, each 2560 f32.
    {
        std::vector<fixture::Tensor> ts;
        for (int l = 1; l <= 4; ++l)
            ts.push_back({"direction." + std::to_string(l), {kN}, 0, (uint8_t) l});
        const fs::path p = tmp.path / "good.gguf";
        fixture::write(p, {fixture::str("general.architecture", "controlvector")}, ts);
        strata::GgufFile f(p.string());
        for (const auto& t : f.tensors()) check(f.in_bounds(t), "a direction.<l> a writer produced is in bounds: " + t.name);
        check(read_as_loader_does(f, 48, (int64_t) kN).empty(), "the loader reads every tensor of a good control-vector file");
    }

    // Case 1: an offset far past the end of the file.  elements() == N and type f32, so every check the loader
    // made passed, and tensor_data() pointed 9.2 exabytes past the mapping.
    {
        const std::string p = write_raw(tmp.path / "bad_offset.gguf", {kN}, 0x7FFFFFFFFFFFFFFFull, 0);
        strata::GgufFile f(p);
        const strata::TensorInfo& t = f.tensors().front();
        check(t.elements() == kN, "bad_offset: elements() still reads 2560, so the metadata check passes");
        check(!f.in_bounds(t), "bad_offset: the offset past EOF is refused");
        check(!read_as_loader_does(f, 48, (int64_t) kN).empty(), "bad_offset: the loader refuses it instead of reading");
    }

    // Case 2: the shape product wraps round.  (2^64-1) * (2^64-2560) == 2560 (mod 2^64), so an enormous tensor
    // reads as "2560 f32" and the loader reads 10240 bytes from a 4096-byte payload.
    {
        const std::string p = write_raw(tmp.path / "bad_wrap.gguf", {(uint64_t) 0 - 1, (uint64_t) 0 - kN}, 0, 4096);
        strata::GgufFile f(p);
        const strata::TensorInfo& t = f.tensors().front();
        check(t.elements() == kN, "bad_wrap: the wrapped product reads 2560, so the metadata check passes");
        check(strata::tensor_payload_bytes(t) == 0, "bad_wrap: tensor_payload_bytes reports the overflow as 0");
        check(!f.in_bounds(t), "bad_wrap: the wrapping shape is refused");
        check(!read_as_loader_does(f, 48, (int64_t) kN).empty(), "bad_wrap: the loader refuses it instead of reading");
    }

    // The boundary the guard must keep accepting: a tensor that ends exactly at the end of the payload.
    {
        const fs::path p = tmp.path / "exact.gguf";
        fixture::write(p, {fixture::str("general.architecture", "controlvector")}, {{"direction.1", {kN}, 0, 7}});
        strata::GgufFile f(p.string());
        const strata::TensorInfo& t = f.tensors().front();
        check(f.in_bounds(t), "a tensor ending exactly at the end of the payload is in bounds");
        check(t.offset + strata::tensor_payload_bytes(t) == f.file_size() - f.data_start(),
              "and it does end exactly there");
    }

    // A tensor starting exactly at the end of the payload, and one that starts inside it but runs past the end.
    {
        const std::string p = write_raw(tmp.path / "one_past.gguf", {kN}, 4096, 4096);
        strata::GgufFile f(p);
        check(!f.in_bounds(f.tensors().front()), "a tensor starting at the end of the payload is refused");
    }
    {
        const std::string p = write_raw(tmp.path / "runs_off.gguf", {kN}, 192, 4096);
        strata::GgufFile f(p);
        check(!f.in_bounds(f.tensors().front()), "a tensor running past the end of the payload is refused");
    }

    // tensor_payload_bytes() on its own: the exact size the loader's f32 tensor must declare.
    {
        strata::TensorInfo t;
        t.type = 0;
        t.shape = {kN};
        check(strata::tensor_payload_bytes(t) == kN * 4ull, "tensor_payload_bytes gives 10240 for 2560 f32");
        t.shape = {kN, 2};
        check(strata::tensor_payload_bytes(t) == kN * 8ull, "and 20480 for 2560x2 f32");
        t.shape = {};
        check(strata::tensor_payload_bytes(t) == 0, "an empty shape has no byte count");
        t.shape = {0};
        check(strata::tensor_payload_bytes(t) == 0, "a zero row has no byte count");
        t.shape = {kN, (uint64_t) 0 - 1};
        check(strata::tensor_payload_bytes(t) == 0, "a product that overflows has no byte count");
        t.shape = {kN};
        t.type = 999;
        check(strata::tensor_payload_bytes(t) == 0, "an unknown type has no byte count");
    }

    std::printf("gguf_tensor_bounds_test: %s\n", g_fail ? "FAILED" : "all cases pass");
    return g_fail ? 1 : 0;
}
