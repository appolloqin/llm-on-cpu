#pragma once
// llm-on-cpu :: weights/lwc_format.h
//
// LWC (LLM Weight Container) v1 — 专家粒度、O_DIRECT 友好的权重容器。
// 设计要点对应 docs/IMPLEMENTATION.md §4：
//   * 每个张量块起始偏移按 block_align(默认4096) 对齐 → 可直读
//   * 专家组(gate/up/down 三张量)在文件内相邻排列 → 单段 READV 可取整专家
//   * 头部为自描述二进制目录(零外部依赖)，fnv1a64 做完整性校验(非加密)
//
// 文件布局:
//   [0 ..24)                preface: magic 'LWC1' + version(u32) +
//                                    catalog_len(u64) + catalog_crc(u64)
//   [24 ..24+catalog_len)   catalog 二进制区
//   [align_up(end,4096)..)  tensor blocks (每块独立对齐, 块间空洞读为零)

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace llmoc::lwc {

inline constexpr char kMagic[5] = "LWC1";
inline constexpr uint32_t kVersion = 1;

enum class Dtype : uint32_t { BF16 = 1, F16 = 2, F32 = 3 };

struct TensorMeta {
    std::string name;
    std::vector<uint64_t> shape;  // 元素数 per dim
    uint64_t offset = 0;          // 文件内字节偏移(已对齐)
    uint64_t nbytes = 0;          // 字节数
    uint64_t checksum = 0;        // payload 的 fnv1a64
};

struct GroupMeta {
    uint32_t layer = 0;
    uint32_t expert_id = 0;
    std::vector<std::string> tensor_names;  // 约定顺序 {gate, up, down}
};

struct Header {
    Dtype dtype = Dtype::BF16;
    uint32_t block_align = 4096;
    std::vector<TensorMeta> tensors;
    std::vector<GroupMeta> groups;

    const TensorMeta* find(std::string_view name) const;
};

// 抛 std::runtime_error 表示格式损坏 / IO 错误 / 校验失败。
uint64_t fnv1a64(const void* data, size_t nbytes);

// 依 payloads 的顺序(layout 顺序即数组顺序)计算并回填 offset/checksum 后写盘。
// 返回写入的 Header(已带布局信息)。
Header Write(const std::filesystem::path& file, const Header& partial,
             const std::vector<std::pair<std::string, std::vector<uint8_t>>>& payloads);

Header ReadHeader(const std::filesystem::path& file);
std::vector<uint8_t> ReadTensor(const std::filesystem::path& file,
                                const Header& hdr, const std::string& name);

// 依现有 header(偏移保持不变)重写 preface+catalog 区。
// 用途: 转换工具写 checksum=0(未校验)落盘后, 由 lwc_verify --update 以
// C++ 速度回填真实校验和 —— 校验和=0 视为"跳过校验"哨兵。
void RewriteCatalog(const std::filesystem::path& file, const Header& hdr);

}  // namespace llmoc::lwc
