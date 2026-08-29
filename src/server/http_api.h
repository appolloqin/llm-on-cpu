#pragma once
// llm-on-cpu :: server/http_api.h
#include <memory>
#include <string>

#include "common/engine_config.h"
#include "sched/scheduler.h"

namespace llmoc::server {

class HttpApi {
 public:
  void bind(const EngineConfig& cfg, sched::Scheduler* sched);
  void listen();  // blocking

 private:
  EngineConfig cfg_;
  sched::Scheduler* sched_ = nullptr;
  std::string api_key_;
};

}  // namespace llmoc::server
