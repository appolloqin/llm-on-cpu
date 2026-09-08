// llm-on-cpu :: src/server/main_glm_nvfp4.cpp — GLM MoE NVFP4
#include "server/glm_server_run.h"

int main(int argc, char** argv) {
  return llmoc::server::glm_run::run(argc, argv, llmoc::glm::QuantKind::kNvfp4,
                                     "llmoc_server_glm_nvfp4", "configs/engine_glm_nvfp4.yaml");
}
