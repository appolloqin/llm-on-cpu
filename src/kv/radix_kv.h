#pragma once
// llm-on-cpu :: kv/radix_kv.h
// C期 radix 前缀池 —— 实现位于 model/kv_cache.h (RadixKvPool)。

#include "model/kv_cache.h"

namespace llmoc::kv {
using RadixPool = model::RadixKvPool;
using SessionCache = model::SessionCache;
}  // namespace llmoc::kv
