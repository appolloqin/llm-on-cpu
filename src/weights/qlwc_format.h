#pragma once
// llm-on-cpu :: weights/qlwc_format.h
// QLWC v1 — 独立于 LWC 的 INT4 量化容器(GPTQ/AWQ 风格布局)。
// 不修改原 LWC 路径；BF16 引擎继续只用 LWC1。
//
// 布局:
//   preface: magic 'QLW1' + version + group_size + scheme + catalog_len + catalog_crc
//   catalog + payload(qweight/scales/zeros 或透传 BF16/F16)

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace llmoc::qlwc {

inline constexpr char kMagic[5] = "QLW1";
inline constexpr uint32_t kVersion = 1;

enum class Scheme : uint32_t { kGptqAsym = 1, kAwqSym = 2 };
enum class TensorKind : uint32_t { kPassthrough = 0, kInt4 = 1 };
enum class PassDtype : uint32_t { kBF16 = 1, kF16 = 2 };

struct TensorMeta {
  std::string name;
  TensorKind kind = TensorKind::kPassthrough;
  std::vector<uint64_t> shape;  // 逻辑形状(量化前)
  // passthrough
  PassDtype pass_dtype = PassDtype::kBF16;
  uint64_t data_offset = 0;
  uint64_t data_nbytes = 0;
  // int4 [M,K]
  uint32_t group_size = 128;
  uint64_t q_offset = 0;
  uint64_t q_nbytes = 0;
  uint64_t scales_offset = 0;
  uint64_t scales_nbytes = 0;  // fp16
  uint64_t zeros_offset = 0;
  uint64_t zeros_nbytes = 0;  // fp16, GPTQ; AWQ 可为 0
};

struct Header {
  Scheme scheme = Scheme::kAwqSym;
  uint32_t group_size = 128;
  uint32_t block_align = 4096;
  std::vector<TensorMeta> tensors;
  const TensorMeta* find(std::string_view name) const;
};

uint64_t fnv1a64(const void* data, size_t n);
Header ReadHeader(const std::filesystem::path& file);
std::vector<uint8_t> ReadBlob(const std::filesystem::path& file, uint64_t offset, uint64_t nbytes);

// 写盘: tensors 与 payloads 一一对应; Int4 的 payload 顺序为 q|scales|zeros(可空)
struct Int4Payload {
  std::vector<uint8_t> qweight;
  std::vector<uint16_t> scales_f16;
  std::vector<uint16_t> zeros_f16;  // empty if AWQ
};
struct PassPayload {
  std::vector<uint8_t> bytes;
};

struct WriteItem {
  TensorMeta meta;  // offset/nbytes 由 Write 回填
  Int4Payload int4;
  PassPayload pass;
};

Header Write(const std::filesystem::path& file, Scheme scheme, uint32_t group_size,
             std::vector<WriteItem>& items);

}  // namespace llmoc::qlwc
