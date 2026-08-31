#pragma once
// llm-on-cpu :: common/omp_tune.h — bandwidth-bound decode prefers fewer threads

namespace llmoc {

// If OMP_NUM_THREADS is unset and OpenMP would use too many logical CPUs,
// cap to physical-core estimate (logical/2, clamped). No-op without OpenMP.
// Returns the effective max threads after tuning.
int tune_openmp_for_decode();

}  // namespace llmoc
