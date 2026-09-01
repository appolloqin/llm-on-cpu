# llm-on-cpu root build
cmake_minimum_required(VERSION 3.25)
project(llm-on-cpu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

if(MSVC)
  add_compile_options(/W4 /permissive- /EHsc /utf-8 /Zi)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
else()
  add_compile_options(-Wall -Wextra)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include(CheckCXXCompilerFlag)
include(CheckCXXSourceCompiles)
include(FetchContent)

# ---------- third_party ----------
FetchContent_Declare(
  json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
# 本引擎只提供明文 HTTP；关掉 OpenSSL 自动探测，避免 Windows 发布包依赖 libssl-*.dll
set(HTTPLIB_USE_OPENSSL_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
set(HTTPLIB_REQUIRE_OPENSSL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  httplib
  URL https://github.com/yhirose/cpp-httplib/archive/refs/tags/v0.18.3.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(json httplib)

# ---------- AMX 微内核探测(仅编译开关; 运行时仍需 SPR 硬件) ----------
add_library(llmoc_weights STATIC
  src/weights/lwc_format.cpp
  src/weights/qlwc_format.cpp
  src/weights/qlwc_store.cpp
  src/weights/io_engine.cpp
  src/weights/weight_manager.cpp
  src/weights/prefetch_pipeline.cpp
  src/weights/layer_stream.cpp
  src/common/platform.cpp
  src/common/engine_config.cpp
  src/common/log.cpp
  src/common/omp_tune.cpp
  src/hal/amx_gemm.cpp
  src/hal/cpu_ops.cpp
  src/hal/cuda_backend.cpp
  src/hal/int4_ops.cpp
  src/hal/quant_dequant.cpp
  src/sched/mode_controller.cpp
  src/sched/placement_planner.cpp
  src/contracts/device_mesh.cpp
  src/exec/factory.cpp
  src/exec/nccl_probe.cpp
  src/exec/mesh_nccl.cpp
  src/exec/cpu/exec_cpu.cpp
  src/exec/hybrid/exec_hybrid.cpp
  src/exec/gpu/exec_gpu.cpp
  src/exec/gpu/residency_gpu.cpp
  src/families/family_packs.cpp
)
target_include_directories(llmoc_weights PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

# 开发机性能路径：AVX2+FMA + OpenMP（SPR 上再叠 AMX；无则自动不开）
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(llmoc_weights PUBLIC OpenMP::OpenMP_CXX)
  message(STATUS "OpenMP: enabled")
else()
  message(STATUS "OpenMP: not found (serial GEMM)")
endif()
if(MSVC)
  target_compile_options(llmoc_weights PRIVATE /arch:AVX2)
  target_compile_definitions(llmoc_weights PUBLIC LLMOC_ENABLE_AVX2=1)
  message(STATUS "AVX2/FMA kernels: enabled (/arch:AVX2)")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-mavx2 -mfma" LLMOC_HAS_AVX2_FMA)
  if(LLMOC_HAS_AVX2_FMA AND CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    target_compile_options(llmoc_weights PRIVATE -mavx2 -mfma)
    target_compile_definitions(llmoc_weights PUBLIC LLMOC_ENABLE_AVX2=1)
    message(STATUS "AVX2/FMA kernels: enabled (-mavx2 -mfma)")
  else()
    message(STATUS "AVX2/FMA kernels: unavailable")
  endif()
endif()

if(UNIX AND NOT APPLE)
  check_cxx_source_compiles(
    "#include <liburing.h>\nint main(){io_uring r; io_uring_queue_init(8,&r,0); io_uring_queue_exit(&r); return 0;}"
    LLMOC_HAS_LIBURING)
  if(LLMOC_HAS_LIBURING)
    target_sources(llmoc_weights PRIVATE src/weights/io_engine_uring.cpp)
    target_compile_definitions(llmoc_weights PUBLIC LLMOC_HAVE_LIBURING=1)
    target_link_libraries(llmoc_weights PUBLIC uring)
    message(STATUS "io_uring backend: enabled")
  else()
    message(STATUS "io_uring backend: unavailable (apt install liburing-dev)")
  endif()
endif()

include(CheckCXXCompilerFlag)
include(CheckCXXSourceCompiles)
set(LLMOC_IS_X86 FALSE)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
  set(LLMOC_IS_X86 TRUE)
endif()
set(LLMOC_HAS_AMX FALSE)
if(MSVC AND LLMOC_IS_X86)
  set(CMAKE_REQUIRED_FLAGS "/arch:AMX")
  check_cxx_source_compiles("
    #include <immintrin.h>
    int main() { _tile_zero(0); return 0; }
  " LLMOC_HAS_AMX)
  set(LLMOC_AMX_FLAGS "/arch:AMX")
elseif(NOT MSVC AND LLMOC_IS_X86)
  set(CMAKE_REQUIRED_FLAGS "-mamx-tile -mamx-bf16")
  check_cxx_source_compiles("
    #include <immintrin.h>
    int main() { _tile_zero(0); return 0; }
  " LLMOC_HAS_AMX)
  set(LLMOC_AMX_FLAGS "-mamx-tile;-mamx-bf16")
endif()
unset(CMAKE_REQUIRED_FLAGS)
if(LLMOC_HAS_AMX)
  set_source_files_properties(src/hal/amx_gemm.cpp
      PROPERTIES COMPILE_OPTIONS "${LLMOC_AMX_FLAGS}")
  target_compile_definitions(llmoc_weights PUBLIC LLMOC_AMX_ENABLED=1)
  message(STATUS "AMX microkernel: enabled (${LLMOC_AMX_FLAGS})")
else()
  message(STATUS "AMX microkernel: stub (x86_64 + working AMX intrinsics required)")
endif()

# ---------- model + sched + server ----------
add_library(llmoc_model STATIC
  src/model/spec_decode.cpp
  src/model/tokenizer_hf.cpp
  src/model/kv_cache.cpp
  src/model/mtp_head.cpp
  src/model/qwen3_5_model.cpp
  src/model/moe_model.cpp
  src/model/generate.cpp
  src/model/vision/image_preprocess.cpp
  src/model/vision/qwen_vision_encoder.cpp
  src/model/graph/model_spec.cpp
  src/model/graph/graph_executor.cpp
  src/sched/scheduler.cpp
)
target_include_directories(llmoc_model PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)
target_link_libraries(llmoc_model PUBLIC llmoc_weights nlohmann_json::nlohmann_json)

add_library(llmoc_server_lib STATIC
  src/server/http_api.cpp
)
target_include_directories(llmoc_server_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(llmoc_server_lib PUBLIC llmoc_model httplib::httplib nlohmann_json::nlohmann_json)
if(WIN32)
  target_link_libraries(llmoc_server_lib PUBLIC ws2_32)
endif()

add_executable(llmoc_server src/server/main.cpp)
target_link_libraries(llmoc_server PRIVATE llmoc_server_lib)

# ---------- INT4/QLWC 路径（独立可执行文件，不改 BF16 llmoc_server） ----------
add_executable(llmoc_server_int4
  src/server/main_int4.cpp
  src/model/qwen3_5_int4_model.cpp
)
target_link_libraries(llmoc_server_int4 PRIVATE llmoc_server_lib)

# ---------- 单元测试 ----------
add_executable(llmoc_unit_tests
  tests/unit/test_main.cpp
  tests/unit/test_lwc.cpp
  tests/unit/test_weight_manager.cpp
  tests/unit/test_layer_stream.cpp
  tests/unit/test_prefetch_pipeline.cpp
  tests/unit/test_spec_decode.cpp
  tests/unit/test_tokenizer.cpp
  tests/unit/test_moe_forward.cpp
  tests/unit/test_int4_gemm.cpp
  tests/unit/test_kv_snapshot.cpp
  tests/unit/test_model_spec.cpp
  tests/unit/test_m5_mode.cpp
  tests/unit/test_exec_mesh.cpp
  tests/unit/test_gpu_quant_gemm.cpp
  tests/unit/test_ds_kimi_stub.cpp
)
target_link_libraries(llmoc_unit_tests PRIVATE llmoc_weights llmoc_model)

enable_testing()
add_test(NAME unit COMMAND llmoc_unit_tests
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(m3_spec_table tools/m3_spec_table/main.cpp)
target_link_libraries(m3_spec_table PRIVATE llmoc_model)

add_executable(bench_decode_tps tools/bench_decode_tps/main.cpp src/model/qwen3_5_int4_model.cpp)
target_link_libraries(bench_decode_tps PRIVATE llmoc_model llmoc_weights nlohmann_json::nlohmann_json)

add_executable(m2_pipeline_bench tools/m2_pipeline_bench/main.cpp)
target_link_libraries(m2_pipeline_bench PRIVATE llmoc_weights)

add_executable(m0_bandwidth tools/m0_bandwidth/main.cpp)
target_link_libraries(m0_bandwidth PRIVATE llmoc_weights)

add_executable(m0_isa tools/m0_amx_flops/main.cpp)
target_link_libraries(m0_isa PRIVATE llmoc_weights)

add_executable(lwc_verify tools/lwc_verify/main.cpp)
target_link_libraries(lwc_verify PRIVATE llmoc_weights)

add_executable(int4_gemm_bench tools/int4_gemm_bench/main.cpp)
target_link_libraries(int4_gemm_bench PRIVATE llmoc_weights)

add_executable(nan_probe
  tools/nan_probe/main.cpp
  src/model/qwen3_5_int4_model.cpp
)
target_link_libraries(nan_probe PRIVATE llmoc_model llmoc_weights nlohmann_json::nlohmann_json)

# ---------- GLM-5.3-Flash 独立模块（不修改上方 Qwen/DeepSeek target 源列表） ----------
add_library(llmoc_glm STATIC
  src/glm/glm_config.cpp
  src/glm/glm_flash_model.cpp
  src/glm/device/glm_device.cpp
  src/glm/hal/glm_awq_int4_ops.cpp
  src/glm/hal/glm_nvfp4_ops.cpp
  src/glm/hal/glm_cuda_ops.cpp
  src/glm/weights/glm_weight_store.cpp
  src/glm/weights/glm_expert_prefetch.cpp
  src/glm/ops/glm_ops.cpp
)
target_include_directories(llmoc_glm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(llmoc_glm PUBLIC llmoc_weights llmoc_model)
# 可选: 强制编译期 CUDA 宏（运行时已动态加载 cudart/cublas，通常无需打开）
option(LLMOC_GLM_CUDA "Define LLMOC_GLM_CUDA=1 (runtime CUDA is auto-detected)" OFF)
if(LLMOC_GLM_CUDA)
  target_compile_definitions(llmoc_glm PUBLIC LLMOC_GLM_CUDA=1)
  message(STATUS "GLM CUDA: compile flag ON + runtime dynamic cudart/cublas")
else()
  message(STATUS "GLM CUDA: runtime dynamic cudart/cublas (no nvcc required)")
endif()

add_executable(llmoc_server_glm src/server/main_glm.cpp)
target_link_libraries(llmoc_server_glm PRIVATE llmoc_glm llmoc_server_lib)

add_executable(make_fake_glmq tools/glm/make_fake_glmq.cpp)
target_link_libraries(make_fake_glmq PRIVATE llmoc_glm)

# ---------- DeepSeek / Kimi stubs ----------
add_library(llmoc_family_stubs STATIC
  src/families/deepseek_v4/ds_stub_model.cpp
  src/families/kimi_k3/kimi_stub_model.cpp
)
target_include_directories(llmoc_family_stubs PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(llmoc_family_stubs PUBLIC llmoc_weights llmoc_model llmoc_glm)

add_executable(llmoc_server_ds src/server/main_ds.cpp)
target_link_libraries(llmoc_server_ds PRIVATE llmoc_family_stubs llmoc_server_lib)

add_executable(llmoc_server_kimi src/server/main_kimi.cpp)
target_link_libraries(llmoc_server_kimi PRIVATE llmoc_family_stubs llmoc_server_lib)

add_executable(make_fake_ds tools/make_fake_ds/main.cpp)
target_link_libraries(make_fake_ds PRIVATE llmoc_family_stubs)

add_executable(make_fake_kimi tools/make_fake_kimi/main.cpp)
target_link_libraries(make_fake_kimi PRIVATE llmoc_family_stubs)

# GLM 单测编入同一 unit 可执行文件，但源码独立；不改现有 test_*.cpp
target_sources(llmoc_unit_tests PRIVATE
  tests/unit/test_glm_mode.cpp
  tests/unit/test_glm_forward.cpp
)
target_link_libraries(llmoc_unit_tests PRIVATE llmoc_glm llmoc_family_stubs)

if(MSVC)
  target_compile_options(m0_isa PRIVATE /arch:AVX2)
  target_compile_definitions(m0_isa PRIVATE LLMOC_ENABLE_AVX2=1)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-mavx2 -mfma" HAS_AVX2_FMA)
  if(HAS_AVX2_FMA AND CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    target_compile_options(m0_isa PRIVATE -mavx2 -mfma)
    target_compile_definitions(m0_isa PRIVATE LLMOC_ENABLE_AVX2=1)
  endif()
endif()
