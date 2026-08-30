// llm-on-cpu :: common/platform.cpp
#include "common/platform.h"

#include "common/log.h"

#if defined(LLMOC_OS_WINDOWS)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#elif defined(LLMOC_OS_LINUX) || defined(LLMOC_OS_MACOS)
#  include <sys/mman.h>
#endif

namespace llmoc::sys {

const char* os_name() {
#if defined(LLMOC_OS_WINDOWS)
    return "Windows";
#elif defined(LLMOC_OS_MACOS)
    return "macOS";
#else
    return "Linux";
#endif
}

bool lock_pages(const void* addr, size_t nbytes) {
#if defined(LLMOC_OS_WINDOWS)
    if (!VirtualLock(const_cast<void*>(addr), nbytes)) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            LOG_WARN("VirtualLock failed (GetLastError=%lu) — degraded, continuing (further "
                     "failures suppressed)",
                     static_cast<unsigned long>(GetLastError()));
        }
        return false;
    }
    return true;
#elif defined(LLMOC_OS_LINUX) || defined(LLMOC_OS_MACOS)
    if (::mlock(addr, nbytes) != 0) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            LOG_WARN("mlock failed — degraded, continuing (further failures suppressed)");
        }
        return false;
    }
    return true;
#endif
}

bool unlock_pages(const void* addr, size_t nbytes) {
#if defined(LLMOC_OS_WINDOWS)
    return VirtualUnlock(const_cast<void*>(addr), nbytes) != 0;
#elif defined(LLMOC_OS_LINUX) || defined(LLMOC_OS_MACOS)
    return ::munlock(addr, nbytes) == 0;
#endif
}

}  // namespace llmoc::sys
