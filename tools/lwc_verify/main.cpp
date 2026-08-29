// llm-on-cpu :: tools/lwc_verify/main.cpp
// LWC 文件校验 / 结构核对 / 内存预算工具。
//
//   lwc_verify <file>                          校验: 目录CRC + 全量张量校验和(0=跳过)
//   lwc_verify <file> --info                   仅打印模型结构摘要
//   lwc_verify <file> --update                 回填 checksum=0 的张量校验和(重写目录)
//   lwc_verify <file> --config config.json     与 HF config.json 交叉核对(MODEL_*.md 校准清单自动化)
//   lwc_verify <file> --config ... --host-gb 64 --kv-gb 6
//                                              额外输出档A(全常驻)内存预算判定
//   lwc_verify <file> --set-dtype BF16|F16|F32  原位改写目录中的 dtype(字节布局不变时用于纠偏)
//
// 退出码: 0=通过  1=权重校验失败  2=用法错误  3=config 交叉核对不匹配
#include <cstdio>
#include <cstring>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <regex>
#include <string>
#include <vector>

#include "weights/lwc_format.h"

namespace fs = std::filesystem;

namespace {

int g_layers = -1, g_experts = -1, g_topk = -1, g_hidden = -1, g_moe_inter = -1,
    g_fkdr = 0, g_vocab = -1;
std::string g_torch_dtype;

// 极简扁平键提取: config.json 里这些键都是顶层整数/字符串, 正则足够(避免引依赖)
void parse_config(const fs::path& p) {
    std::ifstream f(p);
    if (!f) {
        std::fprintf(stderr, "CONFIG ERROR: cannot open %s\n", p.string().c_str());
        std::exit(2);
    }
    std::string text((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
    auto grab_int = [&](const char* key, int* dst) {
        if (*dst != -1) return;
        std::regex re("\"" + std::string(key) + "\"\\s*:\\s*(-?\\d+)");
        std::smatch m;
        if (std::regex_search(text, m, re)) *dst = std::atoi(m[1].str().c_str());
    };
    grab_int("num_hidden_layers", &g_layers);
    grab_int("n_routed_experts", &g_experts);
    grab_int("num_experts", &g_experts);           // 命名兜底
    grab_int("num_experts_per_tok", &g_topk);
    grab_int("hidden_size", &g_hidden);
    grab_int("moe_intermediate_size", &g_moe_inter);
    grab_int("first_k_dense_replace", &g_fkdr);
    grab_int("vocab_size", &g_vocab);
    // 优先 torch_dtype; 新模型(如 Qwen3.5)常把精度写在 text_config.dtype
    std::regex dre("\"torch_dtype\"\\s*:\\s*\"([^\"]+)\"");
    std::smatch dm;
    if (std::regex_search(text, dm, dre)) {
        g_torch_dtype = dm[1].str();
    } else {
        std::regex dre2("\"dtype\"\\s*:\\s*\"(bfloat16|float16|float32)\"");
        if (std::regex_search(text, dm, dre2)) g_torch_dtype = dm[1].str();
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: lwc_verify <file.lwc> [--info|--update]"
                     " [--config config.json] [--host-gb N] [--kv-gb N]"
                     " [--set-dtype BF16|F16|F32]\n");
        return 2;
    }
    const fs::path file = argv[1];
    bool do_update = false, info_only = false;
    fs::path cfg_path;
    double host_gb = 64.0, kv_gb = 6.0;
    const char* set_dtype = nullptr;
    for (int i = 2; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--update")) do_update = true;
        else if (!std::strcmp(argv[i], "--info")) info_only = true;
        else if (!std::strcmp(argv[i], "--config") && i + 1 < argc) cfg_path = argv[++i];
        else if (!std::strcmp(argv[i], "--host-gb") && i + 1 < argc) host_gb = atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--kv-gb") && i + 1 < argc) kv_gb = atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--set-dtype") && i + 1 < argc) set_dtype = argv[++i];
    }

    llmoc::lwc::Header h;
    try {
        h = llmoc::lwc::ReadHeader(file);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "HEADER ERROR: %s\n", e.what());
        return 1;
    }

    if (set_dtype) {
        llmoc::lwc::Dtype nd = llmoc::lwc::Dtype::BF16;
        if (!std::strcmp(set_dtype, "BF16") || !std::strcmp(set_dtype, "bf16") ||
            !std::strcmp(set_dtype, "bfloat16"))
            nd = llmoc::lwc::Dtype::BF16;
        else if (!std::strcmp(set_dtype, "F16") || !std::strcmp(set_dtype, "fp16") ||
                 !std::strcmp(set_dtype, "float16"))
            nd = llmoc::lwc::Dtype::F16;
        else if (!std::strcmp(set_dtype, "F32") || !std::strcmp(set_dtype, "fp32") ||
                 !std::strcmp(set_dtype, "float32"))
            nd = llmoc::lwc::Dtype::F32;
        else {
            std::fprintf(stderr, "ERROR: unknown --set-dtype %s\n", set_dtype);
            return 2;
        }
        if (h.dtype != nd) {
            h.dtype = nd;
            llmoc::lwc::RewriteCatalog(file, h);
            std::printf("set-dtype   : catalog dtype -> %s\n", set_dtype);
        } else {
            std::printf("set-dtype   : already %s\n", set_dtype);
        }
    }

    // ---------- 结构摘要 ----------
    uint32_t layers = 0, experts = 0;
    for (const auto& g : h.groups) {
        if (g.layer + 1 > layers) layers = g.layer + 1;
        if (g.expert_id + 1 > experts) experts = g.expert_id + 1;
    }
    uint64_t total = 0, expert_bytes = 0;
    int zero_ck = 0;
    size_t expert_tensors = 0;
    std::map<std::string, const llmoc::lwc::TensorMeta*> by_name;
    for (const auto& t : h.tensors) by_name[t.name] = &t;
    for (const auto& g : h.groups)
        for (const auto& n : g.tensor_names) {
            auto it = by_name.find(n);
            if (it != by_name.end()) {
                expert_tensors++;
                expert_bytes += it->second->nbytes;
            }
        }
    const size_t dense_tensors = h.tensors.size() - expert_tensors;
    for (const auto& t : h.tensors) {
        total += t.nbytes;
        if (t.checksum == 0) ++zero_ck;
    }

    const char* dtype = h.dtype == llmoc::lwc::Dtype::BF16 ? "BF16"
                        : h.dtype == llmoc::lwc::Dtype::F16 ? "F16"
                                                            : "F32";
    std::printf("file        : %s (%.2f GiB)\n", file.string().c_str(),
                static_cast<double>(total) / (1024.0 * 1024.0 * 1024.0));
    std::printf("dtype/align : %s / %u\n", dtype, h.block_align);
    std::printf("tensors     : %zu (dense=%zu, expert=%zu)\n", h.tensors.size(),
                dense_tensors, expert_tensors);
    std::printf("MoE         : layers=%u experts/layer>=%u (groups=%zu)\n", layers,
                experts, h.groups.size());
    std::printf("checksums   : %d/%zu filled\n",
                static_cast<int>(h.tensors.size()) - zero_ck, h.tensors.size());
    if (info_only) return 0;

    // ---------- 全量校验(+可选回填) ----------
    int bad = 0, skipped = 0;
    std::vector<llmoc::lwc::TensorMeta> updated = h.tensors;
    for (size_t i = 0; i < h.tensors.size(); ++i) {
        try {
            auto buf = llmoc::lwc::ReadTensor(file, h, h.tensors[i].name);
            if (h.tensors[i].checksum == 0) {
                ++skipped;
                updated[i].checksum = llmoc::lwc::fnv1a64(buf.data(), buf.size());
            }
        } catch (const std::exception& e) {
            ++bad;
            std::fprintf(stderr, "BAD: %s: %s\n", h.tensors[i].name.c_str(), e.what());
        }
    }
    std::printf("verify      : %zu ok, %d bad, %d skipped(zero)\n",
                h.tensors.size() - bad, bad, skipped);

    if (do_update && skipped > 0 && bad == 0) {
        h.tensors = updated;
        llmoc::lwc::RewriteCatalog(file, h);
        std::printf("update      : catalog rewritten with %d checksums\n", skipped);
    }

    // ---------- config.json 交叉核对(MODEL_*.md 校准清单自动化) ----------
    int mismatches = 0;
    if (!cfg_path.empty()) {
        parse_config(cfg_path);
        std::printf("\n--config cross-check (%s)\n", cfg_path.string().c_str());

        // header 层数: MoE 从 groups 推导; 稠密从张量名 layers.N. 推导
        uint32_t hdr_layers = layers;
        if (h.groups.empty()) {
            std::regex re("layers\\.(\\d+)\\.");
            for (const auto& t : h.tensors) {
                std::smatch m;
                if (std::regex_search(t.name, m, re)) {
                    const uint32_t v = static_cast<uint32_t>(std::atoi(m[1].str().c_str())) + 1;
                    if (v > hdr_layers) hdr_layers = v;
                }
            }
        }
        const bool dense_model = h.groups.empty() && g_experts == -1;
        if (dense_model)
            std::printf("  [INFO] dense model (config 无 MoE 键, 跳过专家类核对)\n");

        auto check = [&](bool ok, const char* item, const std::string& detail) {
            std::printf("  [%s] %s%s\n", ok ? " OK " : "MISMATCH", item,
                        detail.empty() ? "" : ("  " + detail).c_str());
            if (!ok) ++mismatches;
        };
        if (g_layers > 0)
            check(static_cast<int>(hdr_layers) == g_layers, "num_hidden_layers",
                  "config=" + std::to_string(g_layers) + " header=" +
                      std::to_string(hdr_layers));

        if (dense_model) {
            // 跳过 experts/groups/per-expert 三项
        } else if (g_experts == -1 && !h.groups.empty()) {
            check(false, "n_routed_experts",
                  "config 缺 MoE 键, 但 LWC 含 " + std::to_string(h.groups.size()) +
                      " 个专家组");
        } else if (g_experts > 0) {
            check(static_cast<int>(experts) == g_experts, "n_routed_experts",
                  "config=" + std::to_string(g_experts) + " header=" +
                      std::to_string(experts));
            if (g_layers > 0) {
                const int expected_groups = (g_layers - g_fkdr) * g_experts;
                check(static_cast<int>(h.groups.size()) == expected_groups,
                      "expert groups",
                      "expected=" + std::to_string(expected_groups) +
                          " = (layers-fkdr) x experts, header=" +
                          std::to_string(h.groups.size()));
            }
        }

        int dtype_code = g_torch_dtype == "bfloat16" ? 1
                         : g_torch_dtype == "float16" ? 2
                         : g_torch_dtype == "float32" ? 3 : -1;
        if (dtype_code < 0) {
            std::printf("  [INFO] torch_dtype/dtype missing in config — skip dtype check "
                        "(header=%s)\n", dtype);
        } else {
            check(dtype_code == static_cast<int>(h.dtype), "torch_dtype",
                  "config=" + g_torch_dtype + " header=" + std::string(dtype));
        }

        // 每专家体积: config 预测(3 矩阵 × hidden×moe_inter × 2B) vs header 实测
        if (!dense_model && g_hidden > 0 && g_moe_inter > 0 && !h.groups.empty()) {
            const double predicted = 3.0 * g_hidden * g_moe_inter * 2.0;  // bf16/f16=2B
            const double actual =
                static_cast<double>(expert_bytes) / static_cast<double>(h.groups.size());
            const double drift = predicted > 0 ? std::fabs(actual - predicted) / predicted : 1.0;
            char buf[160];
            std::snprintf(buf, sizeof(buf), "predicted=%.1fKiB actual=%.1fKiB (drift %.1f%%)",
                          predicted / 1024.0, actual / 1024.0, drift * 100.0);
            check(drift < 0.01, "per-expert bytes", buf);
        }

        if (g_topk > 0)
            std::printf("  [INFO] num_experts_per_tok=%d (权重文件不含路由, 无法自动核对)\n",
                        g_topk);
        if (g_vocab > 0) {
            const auto* emb = h.find("embedding.weight");
            if (emb && !emb->shape.empty())
                check(static_cast<int>(emb->shape[0]) == g_vocab, "vocab_size(vs embed dim0)",
                      "config=" + std::to_string(g_vocab) + " header=" +
                          std::to_string(emb->shape[0]));
        }

        // ---------- 档A(全常驻)预算判定 ----------
        const double total_gib = static_cast<double>(total) / (1024.0 * 1024.0 * 1024.0);
        const double need = total_gib + kv_gb + 2.0 /*工作区*/ + 2.0 /*OS*/;
        std::printf("\nbudget(plan-A full-residency)\n");
        std::printf("  weights=%.1f GiB + kv=%.1f + ws/os=4.0 => need %.1f GiB / host %.1f GiB\n",
                    total_gib, kv_gb, need, host_gb);
        const bool fits = need <= host_gb - 1.0;  // 1G 安全边际
        check(fits, "plan-A fits host memory",
              "margin=" + std::to_string(host_gb - 1.0 - need) + " GiB");
        if (fits)
            std::printf("  => LRU budget should cover ALL experts "
                        "(%.1f GiB) -> zero page-miss engine\n",
                        static_cast<double>(expert_bytes) / (1024.0 * 1024.0 * 1024.0));
    }

    if (bad != 0) return 1;
    if (mismatches != 0) {
        std::printf("\nCONFIG CROSS-CHECK FAILED (%d mismatch) — 回填 docs/MODEL_*.md 校准清单\n",
                    mismatches);
        return 3;
    }
    return 0;
}
