#pragma once
// llm-on-cpu :: tests/unit/test_main.h
// 零依赖微型测试框架(开发机离线可跑)。CI 接入网络后按 IMPLEMENTATION §8 可切换 gtest,
// 断言宏保持同名迁移成本极低。

#include <cstdio>
#include <exception>
#include <vector>

namespace tinytest {

using Fn = void (*)();
struct Case {
    const char* suite;
    const char* name;
    Fn fn;
};
inline std::vector<Case>& registry() {
    static std::vector<Case> r;
    return r;
}
inline int& failures() {
    static int f = 0;
    return f;
}
struct Registrar {
    Registrar(const char* s, const char* n, Fn f) { registry().push_back({s, n, f}); }
};

}  // namespace tinytest

#define TINY_TEST(suite, name)                                        \
    static void tiny_##suite##_##name();                              \
    static ::tinytest::Registrar reg_##suite##_##name(                \
        #suite, #name, tiny_##suite##_##name);                        \
    static void tiny_##suite##_##name()

#ifdef TINY_TEST_MAIN
inline int run_all_tests() {
    int pass = 0;
    for (auto& c : tinytest::registry()) {
        const int before = tinytest::failures();
        try {
            c.fn();
        } catch (const std::exception& e) {
            ++tinytest::failures();
            std::fprintf(stderr, "[FAIL] %s.%s threw: %s\n", c.suite, c.name, e.what());
        }
        if (tinytest::failures() == before) {
            ++pass;
            std::printf("[PASS] %s.%s\n", c.suite, c.name);
        }
    }
    std::printf("%d/%d passed\n", pass, static_cast<int>(tinytest::registry().size()));
    return tinytest::failures() == 0 ? 0 : 1;
}

int main() { return run_all_tests(); }
#endif

#define EXPECT_TRUE(cond)                                              \
    do {                                                               \
        if (!(cond)) {                                                 \
            ++tinytest::failures();                                    \
            std::fprintf(stderr, "EXPECT_TRUE failed @ %s:%d : %s\n",  \
                         __FILE__, __LINE__, #cond);                   \
        }                                                              \
    } while (0)

#define EXPECT_EQ(a, b)                                                \
    do {                                                               \
        if (!((a) == (b))) {                                           \
            ++tinytest::failures();                                    \
            std::fprintf(stderr, "EXPECT_EQ failed @ %s:%d : %s\n",    \
                         __FILE__, __LINE__, #a " == " #b);            \
        }                                                              \
    } while (0)
