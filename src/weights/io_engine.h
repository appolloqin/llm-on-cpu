#pragma once
// llm-on-cpu :: weights/io_engine.h
// 异步文件读取引擎 —— D2 双缓冲预取的地基。
//
// 三端策略(docs/PLATFORM.md):
//   * 当前交付 ThreadIoEngine: Win/Linux/macOS 通用的后台线程池 + 定位读
//   * Linux 专属 io_uring+O_DIRECT 版本为 M1 收尾项, 经 make_engine() 工厂无缝替换,
//     调用方语义不变(submit 回调 + drain 栅栏)

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <system_error>

namespace llmoc::io {

using Path = std::filesystem::path;
using ReadCallback = std::function<void(std::error_code ec)>;

class IoEngine {
   public:
    virtual ~IoEngine() = default;

    // 提交一次定位读: 读 [offset, offset+nbytes) 至调用方提供的缓冲 dst。
    // 约束: dst 在回调触发前必须保持有效(双缓冲槽的所有权归调用方)。
    virtual void submit(const Path& file, uint64_t offset, void* dst,
                        size_t nbytes, ReadCallback cb) = 0;

    // 栅栏: 等待所有在途请求完成(成功或失败)后才返回。前向线程每层边界调用。
    virtual void drain() = 0;

    virtual const char* backend_name() const = 0;
};

// 工厂: 唯一的引擎选择点。workers 是并发读线程数(0=自适应)。
std::unique_ptr<IoEngine> make_engine(unsigned workers = 0);

// Linux + liburing 编译时可用; make_engine 内部优先选用, 此处仅为测试暴露
#if defined(LLMOC_OS_LINUX) && defined(LLMOC_HAVE_LIBURING)
std::unique_ptr<IoEngine> make_uring_engine_if_available();
#endif

}  // namespace llmoc::io
