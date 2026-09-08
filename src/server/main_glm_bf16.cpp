// llm-on-cpu :: src/server/main_glm_bf16.cpp — GLM MoE BF16
#include "server/glm_server_run.h"

int main(int argc, char** argv) {
  return llmoc::server::glm_run::run(argc, argv, llmoc::glm::QuantKind::kBf16,
                                     "llmoc_server_glm_bf16", "configs/engine_glm_bf16.yaml");
}
