// llm-on-cpu :: sched/scheduler.cpp
#include "sched/scheduler.h"

#include <chrono>
#include <stdexcept>

namespace llmoc::sched {

void Scheduler::start(model::Generator* gen) { gen_ = gen; }
void Scheduler::stop() {}

model::GenerateResult Scheduler::enqueue_sync(const model::GenerateRequest& req,
                                              const model::TokenSink& on_token) {
  if (!gen_) throw std::runtime_error("scheduler not started");
  {
    std::lock_guard<std::mutex> g(mu_);
    met_.queue_depth = 1;
    ++met_.requests_total;
  }
  const auto t0 = std::chrono::steady_clock::now();
  auto result = gen_->generate(req, on_token);
  const auto t1 = std::chrono::steady_clock::now();
  const double sec = std::chrono::duration<double>(t1 - t0).count();
  std::lock_guard<std::mutex> g(mu_);
  met_.queue_depth = 0;
  met_.tokens_generated += result.completion_tokens;
  met_.prompt_tokens += result.prompt_tokens;
  met_.last_tps = sec > 0 ? result.completion_tokens / sec : 0.0;
  met_.radix_nodes = gen_->radix().nodes();
  return result;
}

Metrics Scheduler::metrics() const {
  std::lock_guard<std::mutex> g(mu_);
  return met_;
}

}  // namespace llmoc::sched
