#pragma once
// llm-on-cpu :: common/alloc.h
// Page-aligned memory helpers shared by io buffers, KV pool, benches.

#include <cstdint>
#include <cstdlib>
#include <new>

#if defined(_WIN32)
#  include <malloc.h>
#endif

namespace llmoc::mem {

inline void* alloc_aligned(size_t nbytes, size_t alignment = 4096) {
#if defined(_WIN32)
    void* p = _aligned_malloc(nbytes, alignment);
#else
    void* p = std::aligned_alloc(alignment, ((nbytes + alignment - 1) / alignment) * alignment);
#endif
    if (!p) throw std::bad_alloc();
    return p;
}

inline void free_aligned(void* p) {
#if defined(_WIN32)
    _aligned_free(p);
#else
    std::free(p);
#endif
}

struct AlignedDeleter { void operator()(void* p) const noexcept { free_aligned(p); } };

}  // namespace llmoc::mem
