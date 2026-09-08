#include "server/ds_server_run.h"
int main(int argc, char** argv) {
  return llmoc::server::ds_run::run(argc, argv, llmoc::families::deepseek::ExpertQuant::kAwqInt4,
                                    "llmoc_server_ds_int4", "configs/engine_ds_int4.yaml");
}
