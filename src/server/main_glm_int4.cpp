// llm-on-cpu :: src/server/main_glm_int4.cpp — GLM MoE AWQ INT4
#include "server/glm_server_run.h"

int main(int argc, char** argv) {
  return llmoc::server::glm_run::run(argc, argv, llmoc::glm::QuantKind::kAwqInt4,
                                     "llmoc_server_glm_int4", "configs/engine_glm_int4.yaml");
}
