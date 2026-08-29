#pragma once
// llm-on-cpu :: common/platform.h
// 三系统(Win/Linux/macOS)识别与薄封装。约定见 docs/PLATFORM.md §三端共同约定：
// 业务代码禁止直接写条件编译，一律经由本头文件的宏与函数。

// ---- OS 识别 ----
#if defined(_WIN32)
#  define LLMOC_OS_WINDOWS 1
#elif defined(__APPLE__) && defined(__MACH__)
#  define LLMOC_OS_MACOS 1
#elif defined(__linux__)
#  define LLMOC_OS_LINUX 1
#else
#  error "llm-on-cpu: unsupported target OS"
#endif

#include <cstddef>

namespace llmoc::sys {

const char* os_name();  // "Windows" / "Linux" / "macOS"

// 页锁定(best-effort): Win=VirtualLock, POSIX=mlock。
// 失败返回 false 且不影响主流程(权限不足等场景允许静默退化)。
bool lock_pages(const void* addr, size_t nbytes);
bool unlock_pages(const void* addr, size_t nbytes);

}  // namespace llmoc::sys
