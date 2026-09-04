#pragma once
// llm-on-cpu :: sched/scheduler.h
// M4: 请求队列 + 单流执行(结构预留 continuous batching)。

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

#include "model/generate.h"

namespace llmoc::sched {

struct Metrics {
  uint64_t requests_total = 0;
  uint64_t tokens_generated = 0;
  uint64_t prompt_tokens = 0;
  double last_tps = 0.0;
  size_t queue_depth = 0;
  size_t radix_nodes = 0;
};

class Scheduler {
 public:
  using Job = std::function<void()>;

  void start(model::Generator* gen);
  void stop();

  model::GenerateResult enqueue_sync(const model::GenerateRequest& req,
                                     const model::TokenSink& on_token = {});

  Metrics metrics() const;

 private:
  model::Generator* gen_ = nullptr;
  mutable std::mutex mu_;
  Metrics met_;
};

}  // namespace llmoc::sched
