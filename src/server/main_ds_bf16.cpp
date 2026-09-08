#include "server/ds_server_run.h"
int main(int argc, char** argv) {
  return llmoc::server::ds_run::run(argc, argv, llmoc::families::deepseek::ExpertQuant::kBf16,
                                    "llmoc_server_ds_bf16", "configs/engine_ds_bf16.yaml");
}
