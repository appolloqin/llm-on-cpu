// llm-on-cpu :: glm/weights/glm_weight_store.cpp
#include "glm/weights/glm_weight_store.h"

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace llmoc::glm {
namespace {

constexpr size_t kHeaderBytes = sizeof(GlmqFileHeader);

}  // namespace

bool write_glmq_file(const std::string& path, const GlmqFileHeader& hdr_in,
                     const std::vector<GlmqTensorRec>& catalog, const std::vector<uint8_t>& data) {
  GlmqFileHeader hdr = hdr_in;
  hdr.magic = kGlmqMagic;
  hdr.version = kGlmqVersion;
  hdr.n_tensors = static_cast<uint32_t>(catalog.size());
  hdr.catalog_bytes = sizeof(GlmqTensorRec) * catalog.size();
  hdr.data_bytes = data.size();
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) return false;
  out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
  if (!catalog.empty())
    out.write(reinterpret_cast<const char*>(catalog.data()),
              static_cast<std::streamsize>(hdr.catalog_bytes));
  if (!data.empty())
    out.write(reinterpret_cast<const char*>(data.data()),
              static_cast<std::streamsize>(data.size()));
  return static_cast<bool>(out);
}

void GlmWeightStore::close() {
  open_ = false;
  catalog_.clear();
  index_.clear();
  data_base_ = nullptr;
  data_bytes_ = 0;
  owned_blob_.clear();
  owned_blob_.shrink_to_fit();

#if defined(_WIN32)
  if (map_view_) {
    UnmapViewOfFile(map_view_);
    map_view_ = nullptr;
  }
  if (win_mapping_) {
    CloseHandle(static_cast<HANDLE>(win_mapping_));
    win_mapping_ = nullptr;
  }
  if (win_file_) {
    CloseHandle(static_cast<HANDLE>(win_file_));
    win_file_ = nullptr;
  }
#else
  if (map_view_ && map_view_ != MAP_FAILED) {
    munmap(map_view_, map_bytes_);
    map_view_ = nullptr;
  }
  if (map_fd_ >= 0) {
    ::close(map_fd_);
    map_fd_ = -1;
  }
#endif
  map_bytes_ = 0;
  hdr_ = {};
}

void GlmWeightStore::open(const std::string& path, QuantKind expect) {
  close();
  (void)expect;

  // Prefer mmap so NVFP4/AWQ MoE packs (tens–100+ GiB) do not need full RAM.
  bool mapped = false;
  size_t file_size = 0;

#if defined(_WIN32)
  HANDLE hf = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                          FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS, nullptr);
  if (hf != INVALID_HANDLE_VALUE) {
    LARGE_INTEGER sz{};
    if (GetFileSizeEx(hf, &sz) && sz.QuadPart > 0) {
      file_size = static_cast<size_t>(sz.QuadPart);
      HANDLE hm = CreateFileMappingA(hf, nullptr, PAGE_READONLY, 0, 0, nullptr);
      if (hm) {
        void* view = MapViewOfFile(hm, FILE_MAP_READ, 0, 0, 0);
        if (view) {
          win_file_ = hf;
          win_mapping_ = hm;
          map_view_ = view;
          map_bytes_ = file_size;
          mapped = true;
        } else {
          CloseHandle(hm);
          CloseHandle(hf);
        }
      } else {
        CloseHandle(hf);
      }
    } else {
      CloseHandle(hf);
    }
  }
#else
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd >= 0) {
    struct stat st {};
    if (fstat(fd, &st) == 0 && st.st_size > 0) {
      file_size = static_cast<size_t>(st.st_size);
      void* view = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
      if (view != MAP_FAILED) {
#ifdef MADV_WILLNEED
        // Do not madvise WILLNEED for whole MoE — would thrash. SEQUENTIAL/RANDOM optional.
        madvise(view, file_size, MADV_RANDOM);
#endif
        map_fd_ = fd;
        map_view_ = view;
        map_bytes_ = file_size;
        mapped = true;
      } else {
        ::close(fd);
      }
    } else {
      ::close(fd);
    }
  }
#endif

  if (mapped) {
    if (map_bytes_ < kHeaderBytes)
      throw std::runtime_error("glm: GLMQ file too small: " + path);
    std::memcpy(&hdr_, map_view_, kHeaderBytes);
  } else {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("glm: cannot open weights: " + path);
    in.read(reinterpret_cast<char*>(&hdr_), sizeof(hdr_));
    if (!in) throw std::runtime_error("glm: short GLMQ header: " + path);
  }

  if (!is_glmq_magic(hdr_.magic))
    throw std::runtime_error("glm: bad GLMQ magic (run tools/glm/* to build .glmq): " + path);
  if (hdr_.version < 1 || hdr_.version > kGlmqVersion)
    throw std::runtime_error("glm: unsupported GLMQ version in " + path);

  const auto q = static_cast<GlmqQuant>(hdr_.quant);
  if (q == GlmqQuant::kAwqInt4) quant_ = QuantKind::kAwqInt4;
  else if (q == GlmqQuant::kNvfp4) quant_ = QuantKind::kNvfp4;
  else quant_ = QuantKind::kBf16;

  uint32_t gs = 0;
  std::memcpy(&gs, hdr_.reserved, 4);
  if (gs > 0) awq_group_size_ = static_cast<int>(gs);

  const size_t cat_bytes = static_cast<size_t>(hdr_.catalog_bytes);
  const size_t data_off = kHeaderBytes + cat_bytes;
  const size_t need = data_off + static_cast<size_t>(hdr_.data_bytes);
  if (mapped && map_bytes_ < need) {
    throw std::runtime_error("glm: truncated GLMQ (mmap size < header+catalog+data): " + path);
  }

  catalog_.resize(hdr_.n_tensors);
  if (hdr_.n_tensors) {
    if (cat_bytes < sizeof(GlmqTensorRec) * hdr_.n_tensors)
      throw std::runtime_error("glm: catalog_bytes too small: " + path);
    if (mapped) {
      std::memcpy(catalog_.data(), static_cast<const uint8_t*>(map_view_) + kHeaderBytes,
                  sizeof(GlmqTensorRec) * hdr_.n_tensors);
    } else {
      std::ifstream in(path, std::ios::binary);
      in.seekg(static_cast<std::streamoff>(kHeaderBytes));
      in.read(reinterpret_cast<char*>(catalog_.data()),
              static_cast<std::streamsize>(sizeof(GlmqTensorRec) * hdr_.n_tensors));
      if (!in) throw std::runtime_error("glm: short GLMQ catalog: " + path);
    }
  }

  data_bytes_ = static_cast<size_t>(hdr_.data_bytes);
  if (mapped) {
    data_base_ = static_cast<const uint8_t*>(map_view_) + data_off;
  } else {
    // Fallback: full copy (only for tiny packs / environments without mmap).
    if (data_bytes_ > (size_t{1} << 30)) {
      throw std::runtime_error(
          "glm: cannot mmap " + path + " and file is >1GiB — refusing RAM load. "
          "Check path/permissions; NVFP4 MoE must be memory-mapped.");
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("glm: cannot reopen weights: " + path);
    in.seekg(static_cast<std::streamoff>(data_off));
    owned_blob_.resize(data_bytes_);
    if (data_bytes_) {
      in.read(reinterpret_cast<char*>(owned_blob_.data()),
              static_cast<std::streamsize>(data_bytes_));
      if (!in) throw std::runtime_error("glm: truncated GLMQ payload: " + path);
    }
    data_base_ = owned_blob_.data();
  }

  for (size_t i = 0; i < catalog_.size(); ++i) index_[catalog_[i].name] = i;
  open_ = true;
}

const GlmqTensorRec* GlmWeightStore::find(const std::string& name) const {
  auto it = index_.find(name);
  if (it == index_.end()) return nullptr;
  return &catalog_[it->second];
}

const uint8_t* GlmWeightStore::data_of(const GlmqTensorRec& t) const {
  if (!data_base_) throw std::runtime_error("glm: store not open");
  if (t.offset + t.nbytes > data_bytes_) throw std::runtime_error("glm: tensor OOB");
  return data_base_ + static_cast<size_t>(t.offset);
}

const uint16_t* GlmWeightStore::bf16(const std::string& name) const {
  const auto* t = find(name);
  if (!t) return nullptr;
  return reinterpret_cast<const uint16_t*>(data_of(*t));
}

const uint16_t* GlmWeightStore::require_bf16(const std::string& name) const {
  const uint16_t* p = bf16(name);
  if (!p) throw std::runtime_error("glm: missing tensor " + name);
  return p;
}

bool GlmWeightStore::awq_view(const std::string& name, hal::AwqView& out) const {
  const auto* t = find(name);
  if (!t || t->dtype != static_cast<uint16_t>(GlmqDtype::kAWQ4)) return false;
  const int M = static_cast<int>(t->shape[0]);
  const int K = static_cast<int>(t->shape[1]);
  int gs = t->ndim >= 3 && t->shape[2] > 0 ? static_cast<int>(t->shape[2]) : awq_group_size_;
  if (gs <= 0) gs = K;
  const int rb = (K + 1) / 2;
  const size_t q_bytes = static_cast<size_t>(M) * rb;
  const int ng = K / gs;
  const size_t s_bytes = static_cast<size_t>(M) * ng * 2;
  if (t->nbytes < q_bytes + s_bytes) throw std::runtime_error("glm: AWQ tensor truncated " + name);
  const uint8_t* base = data_of(*t);
  out.qweight = base;
  out.scales = reinterpret_cast<const uint16_t*>(base + q_bytes);
  out.scales_f32 = nullptr;
  out.M = M;
  out.K = K;
  out.group_size = gs;
  return true;
}

bool GlmWeightStore::nvfp4_view(const std::string& name, hal::Nvfp4View& out) const {
  const auto* t = find(name);
  if (!t || t->dtype != static_cast<uint16_t>(GlmqDtype::kNVFP4)) return false;
  const int M = static_cast<int>(t->shape[0]);
  const int K = static_cast<int>(t->shape[1]);
  int gs = t->ndim >= 3 && t->shape[2] > 0 ? static_cast<int>(t->shape[2]) : 16;
  if (gs <= 0) gs = 16;
  const int rb = (K + 1) / 2;
  const size_t q_bytes = static_cast<size_t>(M) * rb;
  const int ng = (K + gs - 1) / gs;
  const size_t s_bytes = static_cast<size_t>(M) * ng;
  if (t->nbytes < q_bytes + s_bytes + 4)
    throw std::runtime_error("glm: NVFP4 tensor truncated " + name);
  const uint8_t* base = data_of(*t);
  out.qweight = base;
  out.scales_fp8 = base + q_bytes;
  float g = 1.f;
  std::memcpy(&g, base + q_bytes + s_bytes, 4);
  out.global_scale = g;
  out.M = M;
  out.K = K;
  out.group_size = gs;
  return true;
}

}  // namespace llmoc::glm
