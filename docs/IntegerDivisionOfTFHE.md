# IntegerDivisionOfTFHE artifact benchmarks

This repository contains the focused TFHEpp implementation needed to reproduce
the runtime tables in `IntegerDivisionOfTFHE`.  It uses the paper's parameter
set (`n=760`, `N=2048`, `l=2`, `Bgbit=17`) through
`USE_DIVISION_BENCH`; it does not use the newer block-binary
`USE_AXELL_INTEGER` parameters.

## Build

```bash
cmake -S . -B build-integer-division \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_TEST=ON \
  -DENABLE_AXELL=ON \
  -DUSE_DIVISION_BENCH=ON

cmake --build build-integer-division \
  --target integer_division_benchmarks -j
```

The division programs run ten randomized encrypted inputs, report inversion,
division-by-multiplication, and total latency. The local harnesses stop if a
benchmark reports a nonzero correctness error count, including in Release
builds where C++ assertions are disabled.  For example:

```bash
taskset -c 0 env OMP_NUM_THREADS=1 \
  ./build-integer-division/test/axell/integer-wise/intdivlutnr
```

## Reproduce the paper tables

Run the complete local harness from the repository root:

```bash
BUILD_JOBS=8 ./test/axell/paper_bench_local.bash
```

It produces `boot.tsv`, `misc.tsv`, `division.tsv`, and `bfloat16.csv` below
`build-paper-bench-local/logs/paper-bench-local`.  The harness pins the TFHEpp
single-core cases to CPU 0 and the multi-core cases to CPUs 0--7.  It also
clones the pinned official `tfhe-rs-1.5.4` release into the build directory,
applies the recorded benchmark-only 2^-40 parameter selection, and saves the
exact checkout metadata and patch beside the results.  Therefore it requires
Git, Cargo/Rust, CMake, Python 3, `taskset`, network access, and at least eight
available CPU cores.

For only the 16-bit single/multi-core division comparison, use:

```bash
BUILD_JOBS=8 ./test/axell/division_bench_16bit_local.bash
```

Timings depend on the CPU and FFT backend. A successful run must report
`count:0` for every correctness check; `Passed` alone is insufficient because
Release builds disable C++ assertions. Run one harness at a time to avoid
CPU contention affecting the timings.

## Check the Lean proofs

Build and run the Lean 4.34.1 + Mathlib container:

```bash
cd scripts/lean
apptainer build --fakeroot lean.sif lean.def
apptainer run lean.sif
```

The build and default run typecheck both `ReducedNR.lean` and `LUTNR.lean`.
`ProofAudit.lean` checks every declaration in both modules, including its
transitive dependencies. It rejects `sorryAx`, compiler-trust axioms, and
custom axioms, allowing only the standard logical axioms `propext`,
`Classical.choice`, and `Quot.sound`.

The proved results retain their stated input conditions. ReducedNR's
`Reciprocal.thm_all_k_complete` requires `k ≥ 3` and `1 ≤ y < 2^(4k)`.
LUTNR's general `LUTNR.thm_lutnr` requires the approximation bound `ApproxHyp`
for the supplied `hz`, with the corrected range `2^n ≤ hz < 2^(L+1)`.
The concrete `hzFloor n y = floor((2^(2L) - 1) / (hy * 2^n))` construction
has proved range and approximation bounds. `LUTNR.thm_lutnr_hzFloor` proves
its corrected output is valid using only `n > 0` and `1 ≤ y < 2^L`, with
`L = 2n`. This construction does not establish that the LUT/Newton–Raphson
implementation computes the same candidate or that the C++ encrypted
computations are correct.
