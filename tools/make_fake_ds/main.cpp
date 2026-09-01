// tools/make_fake_ds/main.cpp
#include <cstdio>
#include "families/deepseek_v4/ds_stub_model.h"

int main(int argc, char** argv) {
  const char* path = argc > 1 ? argv[1] : "models/fake_ds.dskq";
  try {
    llmoc::families::deepseek::write_fake_dskq(path);
    std::printf("wrote %s\n", path);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
