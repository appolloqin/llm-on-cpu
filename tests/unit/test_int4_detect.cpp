// llm-on-cpu :: tests/unit/test_int4_detect.cpp
#include "test_main.h"

#include <filesystem>
#include <fstream>
#include <string>

#include "server/int4_detect.h"

namespace {

std::string write_tmp_config(const std::string& name, int hidden, int layers) {
  const std::string path = "models/_test_" + name + "_config.json";
  std::filesystem::create_directories("models");
  std::ofstream f(path);
  f << "{\n  \"text_config\": {\n"
    << "    \"hidden_size\": " << hidden << ",\n"
    << "    \"num_hidden_layers\": " << layers << "\n"
    << "  }\n}\n";
  return path;
}

}  // namespace

TINY_TEST(Int4Detect, Qwen35_9BNotClassifiedAs38) {
  using llmoc::server::int4_detect::hf_config_looks_qwen38;
  using llmoc::server::int4_detect::path_looks_qwen35;

  EXPECT_TRUE(path_looks_qwen35("models/Qwen3.5-9B-AWQ-hf/config.json"));
  EXPECT_TRUE(path_looks_qwen35("models/Qwen3.5-9B.int4.qlwc"));

  const std::string cfg9 = write_tmp_config("q35_9b", 4096, 32);
  // Path hint says 3.5 → never 3.8, even though H=4096.
  EXPECT_TRUE(!hf_config_looks_qwen38(cfg9, "models/Qwen3.5-9B.int4.qlwc", false));
  // Geometry alone: H=4096 L=32 is 9B, not 3.8-27B (H=5120 L=64).
  EXPECT_TRUE(!hf_config_looks_qwen38(cfg9, "models/unknown.int4.qlwc", false));

  const std::string cfg38 = write_tmp_config("q38_27b", 5120, 64);
  EXPECT_TRUE(hf_config_looks_qwen38(cfg38, "models/unknown.int4.qlwc", false));
  EXPECT_TRUE(hf_config_looks_qwen38(cfg38, "models/Qwen3.8-27B.int4.qlwc", false));
}
