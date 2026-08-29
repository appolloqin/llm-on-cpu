// llm-on-cpu :: tools/m3_spec_table/main.cpp
// M3 数字仿真: 理论 tokens/step(E[k, alpha]) × M2 实测原始吞吐 = 预测有效吞吐。
// 用于权重到位前锁定 D3 达标路径: 需要 alpha×k 组合达到多少倍乘法器。
//
// 用法: m3_spec_table [--base 51.67]   (base=未投机单流原始 tok/s, 默认取 M2 实测)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

namespace {

// iid 独立接受概率近似: E = 1 + a + a^2 + ... + a^k
double tokens_per_step(double alpha, int k) {
    double e = 1.0, p = 1.0;
    for (int i = 0; i < k; ++i) {
        p *= alpha;
        e += p;
    }
    return e;
}

}  // namespace

int main(int argc, char** argv) {
    double base = 51.67;
    for (int i = 1; i < argc; ++i)
        if (!std::strcmp(argv[i], "--base") && i + 1 < argc)
            base = atof(argv[++i]);

    std::printf("base(raw, no-spec) tok/s = %.2f  (M2 dev-machine measured)\n\n", base);
    std::printf("%8s", "alpha\\k");
    for (int k = 2; k <= 5; ++k) std::printf("%12dk", k);
    std::printf("\n");

    const double alphas[] = {0.5, 0.6, 0.7, 0.8, 0.894, 0.95};
    for (double a : alphas) {
        std::printf("%8.3f", a);
        for (int k = 2; k <= 5; ++k) {
            const double m = tokens_per_step(a, k);
            std::printf("%9.2fx %5.1f", m, base * m);
        }
        std::printf("\n");
    }

    // 达标线判定: 目标 30 tok/s, 需要乘法器 = 30/base
    const double need = 30.0 / base;
    std::printf("\ntarget 30 tok/s needs multiplier x%.2f over base\n", need);
    if (need <= 1.0)
        std::printf("=> base already meets target; MTP is headroom, not requirement\n");
    else
        std::printf("=> e.g. alpha>=0.7 with k>=3 suffices: %.2fx available\n",
                    tokens_per_step(0.7, 3));
    return 0;
}
