// tools/make_fake_kimi/main.cpp
#include <cstdio>
#include "families/kimi_k3/kimi_stub_model.h"

int main(int argc, char** argv) {
  const char* path = argc > 1 ? argv[1] : "models/fake_kimi.kimiq";
  try {
    llmoc::families::kimi::write_fake_kimiq(path);
    std::printf("wrote %s\n", path);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
