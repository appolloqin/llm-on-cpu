// llm-on-cpu :: weights/qlwc_format.cpp
#include "weights/qlwc_format.h"

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace llmoc::qlwc {
namespace {

void put_u32(std::vector<uint8_t>& b, uint32_t v) {
  for (int i = 0; i < 4; ++i) b.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
}
void put_u64(std::vector<uint8_t>& b, uint64_t v) {
  for (int i = 0; i < 8; ++i) b.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
}
uint32_t get_u32(const uint8_t*& p) {
  uint32_t v = static_cast<uint32_t>(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) |
               (uint32_t(p[3]) << 24);
  p += 4;
  return v;
}
uint64_t get_u64(const uint8_t*& p) {
  uint64_t v = 0;
  for (int i = 0; i < 8; ++i) v |= uint64_t(p[i]) << (8 * i);
  p += 8;
  return v;
}
void put_str(std::vector<uint8_t>& b, const std::string& s) {
  put_u32(b, static_cast<uint32_t>(s.size()));
  b.insert(b.end(), s.begin(), s.end());
}
std::string get_str(const uint8_t*& p) {
  const uint32_t n = get_u32(p);
  std::string s(reinterpret_cast<const char*>(p), n);
  p += n;
  return s;
}
uint64_t align_up(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

std::vector<uint8_t> build_catalog(const Header& h) {
  std::vector<uint8_t> cat;
  put_u32(cat, static_cast<uint32_t>(h.scheme));
  put_u32(cat, h.group_size);
  put_u32(cat, h.block_align);
  put_u64(cat, h.tensors.size());
  for (const auto& t : h.tensors) {
    put_str(cat, t.name);
    put_u32(cat, static_cast<uint32_t>(t.kind));
    put_u64(cat, t.shape.size());
    for (auto d : t.shape) put_u64(cat, d);
    if (t.kind == TensorKind::kPassthrough) {
      put_u32(cat, static_cast<uint32_t>(t.pass_dtype));
      put_u64(cat, t.data_offset);
      put_u64(cat, t.data_nbytes);
    } else {
      put_u32(cat, t.group_size);
      put_u64(cat, t.q_offset);
      put_u64(cat, t.q_nbytes);
      put_u64(cat, t.scales_offset);
      put_u64(cat, t.scales_nbytes);
      put_u64(cat, t.zeros_offset);
      put_u64(cat, t.zeros_nbytes);
    }
  }
  return cat;
}

}  // namespace

const TensorMeta* Header::find(std::string_view name) const {
  for (const auto& t : tensors)
    if (t.name == name) return &t;
  return nullptr;
}

uint64_t fnv1a64(const void* data, size_t n) {
  const auto* p = static_cast<const uint8_t*>(data);
  uint64_t h = 14695981039346656037ull;
  for (size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 1099511628211ull;
  return h;
}

Header Write(const std::filesystem::path& file, Scheme scheme, uint32_t group_size,
             std::vector<WriteItem>& items) {
  Header h;
  h.scheme = scheme;
  h.group_size = group_size;
  h.block_align = 4096;
  h.tensors.resize(items.size());

  // 第一遍: 估 catalog 长度(偏移占位 0)
  for (size_t i = 0; i < items.size(); ++i) {
    h.tensors[i] = items[i].meta;
    h.tensors[i].group_size = group_size;
  }
  auto probe = build_catalog(h);
  uint64_t off = align_up(24 + probe.size(), h.block_align);

  for (size_t i = 0; i < items.size(); ++i) {
    auto& t = h.tensors[i];
    t = items[i].meta;
    t.group_size = group_size;
    if (t.kind == TensorKind::kPassthrough) {
      t.data_offset = off;
      t.data_nbytes = items[i].pass.bytes.size();
      off = align_up(off + t.data_nbytes, h.block_align);
    } else {
      t.q_offset = off;
      t.q_nbytes = items[i].int4.qweight.size();
      off = align_up(off + t.q_nbytes, 64);
      t.scales_offset = off;
      t.scales_nbytes = items[i].int4.scales_f16.size() * sizeof(uint16_t);
      off = align_up(off + t.scales_nbytes, 64);
      t.zeros_offset = off;
      t.zeros_nbytes = items[i].int4.zeros_f16.size() * sizeof(uint16_t);
      off = align_up(off + t.zeros_nbytes, h.block_align);
    }
    items[i].meta = t;
  }

  auto catalog = build_catalog(h);
  std::vector<uint8_t> preface;
  preface.insert(preface.end(), kMagic, kMagic + 4);
  put_u32(preface, kVersion);
  put_u64(preface, catalog.size());
  put_u64(preface, fnv1a64(catalog.data(), catalog.size()));

  std::ofstream out(file, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("QLWC write open failed");
  out.write(reinterpret_cast<const char*>(preface.data()), preface.size());
  out.write(reinterpret_cast<const char*>(catalog.data()), catalog.size());
  const uint64_t data0 = align_up(preface.size() + catalog.size(), h.block_align);
  while (static_cast<uint64_t>(out.tellp()) < data0) out.put('\0');

  for (size_t i = 0; i < items.size(); ++i) {
    const auto& t = h.tensors[i];
    const auto& it = items[i];
    if (t.kind == TensorKind::kPassthrough) {
      out.seekp(static_cast<std::streamoff>(t.data_offset));
      out.write(reinterpret_cast<const char*>(it.pass.bytes.data()),
                static_cast<std::streamsize>(it.pass.bytes.size()));
    } else {
      out.seekp(static_cast<std::streamoff>(t.q_offset));
      out.write(reinterpret_cast<const char*>(it.int4.qweight.data()),
                static_cast<std::streamsize>(it.int4.qweight.size()));
      out.seekp(static_cast<std::streamoff>(t.scales_offset));
      out.write(reinterpret_cast<const char*>(it.int4.scales_f16.data()),
                static_cast<std::streamsize>(t.scales_nbytes));
      if (t.zeros_nbytes) {
        out.seekp(static_cast<std::streamoff>(t.zeros_offset));
        out.write(reinterpret_cast<const char*>(it.int4.zeros_f16.data()),
                  static_cast<std::streamsize>(t.zeros_nbytes));
      }
    }
  }
  return h;
}

Header ReadHeader(const std::filesystem::path& file) {
  std::ifstream in(file, std::ios::binary);
  if (!in) throw std::runtime_error("QLWC open failed: " + file.string());
  char magic[4];
  in.read(magic, 4);
  if (std::memcmp(magic, kMagic, 4) != 0) throw std::runtime_error("QLWC bad magic");
  // preface after magic: version(u32) + catalog_len(u64) + catalog_crc(u64) = 20 bytes
  std::vector<uint8_t> pref(20);
  in.read(reinterpret_cast<char*>(pref.data()), 20);
  if (!in) throw std::runtime_error("QLWC short preface");
  const uint8_t* p = pref.data();
  const uint32_t ver = get_u32(p);
  if (ver != kVersion) throw std::runtime_error("QLWC unsupported version");
  const uint64_t cat_len = get_u64(p);
  const uint64_t cat_crc = get_u64(p);
  std::vector<uint8_t> cat(static_cast<size_t>(cat_len));
  in.read(reinterpret_cast<char*>(cat.data()), static_cast<std::streamsize>(cat_len));
  if (static_cast<uint64_t>(in.gcount()) != cat_len) throw std::runtime_error("QLWC short catalog");
  if (fnv1a64(cat.data(), cat.size()) != cat_crc) throw std::runtime_error("QLWC catalog crc");

  Header h;
  const uint8_t* c = cat.data();
  h.scheme = static_cast<Scheme>(get_u32(c));
  h.group_size = get_u32(c);
  h.block_align = get_u32(c);
  const uint64_t n = get_u64(c);
  h.tensors.resize(n);
  for (uint64_t i = 0; i < n; ++i) {
    auto& t = h.tensors[i];
    t.name = get_str(c);
    t.kind = static_cast<TensorKind>(get_u32(c));
    const uint64_t nd = get_u64(c);
    t.shape.resize(nd);
    for (uint64_t d = 0; d < nd; ++d) t.shape[d] = get_u64(c);
    if (t.kind == TensorKind::kPassthrough) {
      t.pass_dtype = static_cast<PassDtype>(get_u32(c));
      t.data_offset = get_u64(c);
      t.data_nbytes = get_u64(c);
    } else {
      t.group_size = get_u32(c);
      t.q_offset = get_u64(c);
      t.q_nbytes = get_u64(c);
      t.scales_offset = get_u64(c);
      t.scales_nbytes = get_u64(c);
      t.zeros_offset = get_u64(c);
      t.zeros_nbytes = get_u64(c);
    }
  }
  return h;
}

std::vector<uint8_t> ReadBlob(const std::filesystem::path& file, uint64_t offset, uint64_t nbytes) {
  std::ifstream in(file, std::ios::binary);
  if (!in) throw std::runtime_error("QLWC read open failed");
  std::vector<uint8_t> buf(nbytes);
  in.seekg(static_cast<std::streamoff>(offset));
  in.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(nbytes));
  if (static_cast<uint64_t>(in.gcount()) != nbytes) throw std::runtime_error("QLWC short read");
  return buf;
}

}  // namespace llmoc::qlwc
