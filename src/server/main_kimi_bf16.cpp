#include "server/kimi_server_run.h"
int main(int argc, char** argv) {
  return llmoc::server::kimi_run::run(argc, argv, llmoc::families::deepseek::ExpertQuant::kBf16,
                                      "llmoc_server_kimi_bf16", "configs/engine_kimi_bf16.yaml");
}
