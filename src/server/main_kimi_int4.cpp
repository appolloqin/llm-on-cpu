#include "server/kimi_server_run.h"
int main(int argc, char** argv) {
  return llmoc::server::kimi_run::run(argc, argv, llmoc::families::deepseek::ExpertQuant::kAwqInt4,
                                      "llmoc_server_kimi_int4", "configs/engine_kimi_int4.yaml");
}
