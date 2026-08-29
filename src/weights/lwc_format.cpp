// llm-on-cpu :: weights/lwc_format.cpp
#include "weights/lwc_format.h"

#include <algorithm>
#include <cstring>
#include <fstream>

#include "common/log.h"

namespace llmoc::lwc {

namespace {

constexpr size_t kPrefaceBytes = 4 + 4 + 8 + 8;  // magic + version + len + crc

void put_u32(std::vector<uint8_t>& b, uint32_t v) {
    for (int i = 0; i < 4; ++i) b.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
}
void put_u64(std::vector<uint8_t>& b, uint64_t v) {
    for (int i = 0; i < 8; ++i) b.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
}
uint32_t get_u32(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}
uint64_t get_u64(const uint8_t* p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= static_cast<uint64_t>(p[i]) << (8 * i);
    return v;
}
void put_str(std::vector<uint8_t>& b, const std::string& s) {
    put_u32(b, static_cast<uint32_t>(s.size()));
    b.insert(b.end(), s.begin(), s.end());
}
std::string get_str(const uint8_t*& p) {
    const uint32_t n = get_u32(p);
    p += 4;
    std::string s(reinterpret_cast<const char*>(p), n);
    p += n;
    return s;
}

uint64_t align_up_u64(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

[[noreturn]] void corrupt(const char* why) {
    throw std::runtime_error(std::string("LWC corrupted: ") + why);
}

std::vector<uint8_t> build_catalog(const Header& h) {
    std::vector<uint8_t> cat;
    cat.reserve(4096);
    put_u32(cat, static_cast<uint32_t>(h.dtype));
    put_u32(cat, h.block_align);
    put_u64(cat, h.tensors.size());
    put_u64(cat, h.groups.size());
    for (const auto& t : h.tensors) {
        put_str(cat, t.name);
        put_u64(cat, t.shape.size());
        for (uint64_t d : t.shape) put_u64(cat, d);
        put_u64(cat, t.offset);
        put_u64(cat, t.nbytes);
        put_u64(cat, t.checksum);
    }
    for (const auto& g : h.groups) {
        put_u32(cat, g.layer);
        put_u32(cat, g.expert_id);
        put_u64(cat, g.tensor_names.size());
        for (const auto& n : g.tensor_names) put_str(cat, n);
    }
    return cat;
}

}  // namespace

const TensorMeta* Header::find(std::string_view name) const {
    for (const auto& t : tensors)
        if (t.name == name) return &t;
    return nullptr;
}

uint64_t fnv1a64(const void* data, size_t nbytes) {
    const auto* p = static_cast<const uint8_t*>(data);
    uint64_t h = 14695981039346656037ull;
    for (size_t i = 0; i < nbytes; ++i) h = (h ^ p[i]) * 1099511628211ull;
    return h;
}

Header Write(const std::filesystem::path& file, const Header& partial,
             const std::vector<std::pair<std::string, std::vector<uint8_t>>>& payloads) {
    if (payloads.size() != partial.tensors.size())
        corrupt("payload count != tensors declared");

    // 1) 占位目录求长度(目录长度与 offset 取值无关, u64 定长) -> 数据区起点
    Header out = partial;
    const auto probe = build_catalog(out);
    const uint64_t data_start =
        align_up_u64(kPrefaceBytes + probe.size(), out.block_align);

    // 2) 布局回填: 偏移/字节数/校验和
    uint64_t off = data_start;
    for (size_t i = 0; i < out.tensors.size(); ++i) {
        auto& t = out.tensors[i];
        const auto& pl = payloads[i].second;
        t.offset = off;
        t.nbytes = pl.size();
        t.checksum = fnv1a64(pl.data(), pl.size());
        off = align_up_u64(off + t.nbytes, out.block_align);
    }

    // 3) 最终目录 + 落盘
    const auto cat = build_catalog(out);
    std::ofstream f(file, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("LWC cannot open for write: " + file.string());

    std::vector<uint8_t> preface;
    preface.insert(preface.end(), kMagic, kMagic + 4);
    put_u32(preface, kVersion);
    put_u64(preface, cat.size());
    put_u64(preface, fnv1a64(cat.data(), cat.size()));
    f.write(reinterpret_cast<const char*>(preface.data()), preface.size());
    f.write(reinterpret_cast<const char*>(cat.data()),
            static_cast<std::streamsize>(cat.size()));
    if (!f) throw std::runtime_error("LWC header write failed");

    for (const auto& pl : payloads) {
        const auto* meta = out.find(pl.first);
        if (!meta) corrupt("internal: payload without meta");
        f.seekp(static_cast<std::streamoff>(meta->offset));
        f.write(reinterpret_cast<const char*>(pl.second.data()),
                static_cast<std::streamsize>(pl.second.size()));
        if (!f) throw std::runtime_error("LWC block write failed");
    }
    return out;
}

void RewriteCatalog(const std::filesystem::path& file, const Header& hdr) {
    const auto cat = build_catalog(hdr);
    std::fstream f(file, std::ios::binary | std::ios::in | std::ios::out);
    if (!f) throw std::runtime_error("LWC cannot open: " + file.string());

    uint8_t pre[kPrefaceBytes]{};
    f.read(reinterpret_cast<char*>(pre), kPrefaceBytes);
    if (!f || std::memcmp(pre, kMagic, 4) != 0) corrupt("bad magic");
    if (get_u64(pre + 8) != cat.size())
        corrupt("RewriteCatalog: catalog length changed - offsets layout unsafe");

    std::vector<uint8_t> preface;
    preface.insert(preface.end(), kMagic, kMagic + 4);
    put_u32(preface, kVersion);
    put_u64(preface, cat.size());
    put_u64(preface, fnv1a64(cat.data(), cat.size()));
    f.seekp(0);
    f.write(reinterpret_cast<const char*>(preface.data()), preface.size());
    f.write(reinterpret_cast<const char*>(cat.data()),
            static_cast<std::streamsize>(cat.size()));
    if (!f) throw std::runtime_error("LWC catalog rewrite failed");
}

Header ReadHeader(const std::filesystem::path& file) {
    std::ifstream f(file, std::ios::binary);
    if (!f) throw std::runtime_error("LWC cannot open: " + file.string());

    uint8_t pre[kPrefaceBytes]{};
    f.read(reinterpret_cast<char*>(pre), kPrefaceBytes);
    if (!f || std::memcmp(pre, kMagic, 4) != 0) corrupt("bad magic");
    if (get_u32(pre + 4) != kVersion) corrupt("unsupported version");

    const uint64_t clen = get_u64(pre + 8);
    const uint64_t ccrc = get_u64(pre + 16);
    std::vector<uint8_t> cat(clen);
    f.read(reinterpret_cast<char*>(cat.data()), static_cast<std::streamsize>(clen));
    if (!f) corrupt("short catalog");
    if (fnv1a64(cat.data(), cat.size()) != ccrc) corrupt("catalog crc mismatch");

    const uint8_t* p = cat.data();
    Header h;
    h.dtype = static_cast<Dtype>(get_u32(p)); p += 4;
    h.block_align = get_u32(p); p += 4;
    const uint64_t nt = get_u64(p); p += 8;
    const uint64_t ng = get_u64(p); p += 8;
    h.tensors.resize(nt);
    for (auto& t : h.tensors) {
        t.name = get_str(p);
        const uint64_t nd = get_u64(p); p += 8;
        t.shape.resize(nd);
        for (auto& d : t.shape) { d = get_u64(p); p += 8; }
        t.offset = get_u64(p); p += 8;
        t.nbytes = get_u64(p); p += 8;
        t.checksum = get_u64(p); p += 8;
        if (t.offset % h.block_align) corrupt("unaligned block offset");
    }
    h.groups.resize(ng);
    for (auto& g : h.groups) {
        g.layer = get_u32(p); p += 4;
        g.expert_id = get_u32(p); p += 4;
        const uint64_t nn = get_u64(p); p += 8;
        g.tensor_names.resize(nn);
        for (auto& n : g.tensor_names) n = get_str(p);
    }
    return h;
}

std::vector<uint8_t> ReadTensor(const std::filesystem::path& file,
                                const Header& hdr, const std::string& name) {
    const auto* t = hdr.find(name);
    if (!t) throw std::runtime_error("LWC tensor not found: " + name);
    std::ifstream f(file, std::ios::binary);
    if (!f) throw std::runtime_error("LWC cannot open: " + file.string());
    std::vector<uint8_t> buf(t->nbytes);
    f.seekg(static_cast<std::streamoff>(t->offset));
    f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(t->nbytes));
    if (!f) throw std::runtime_error("LWC short read: " + name);
    // checksum==0 哨兵: 转换工具未回填校验和(lwc_verify --update 负责), 跳过校验
    if (t->checksum != 0 && fnv1a64(buf.data(), buf.size()) != t->checksum)
        throw std::runtime_error("LWC checksum mismatch: " + name);
    return buf;
}

}  // namespace llmoc::lwc
