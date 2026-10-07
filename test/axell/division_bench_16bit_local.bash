#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-division-bench-local}"
BUILD_JOBS="${BUILD_JOBS:-8}"
LOG_DIR="${LOG_DIR:-${BUILD_DIR}/logs/division-bench-16bit-local}"
source "${SCRIPT_DIR}/../lib/tfhers_release_helpers.bash"

TFHERS_PARAM_ALIAS="${TFHERS_PARAM_ALIAS:-}"

mkdir -p "${LOG_DIR}"

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "Missing required command: $1" >&2
        exit 1
    }
}

extract_tfhepp_ms() {
    local label="$1"
    local log_file="$2"
    grep "^${label}:" "${log_file}" | tail -n1 | sed -E "s/^${label}:([0-9.]+)ms.*/\\1/"
}

extract_criterion_seconds() {
    local log_file="$1"
    python3 - "${log_file}" <<'PY'
import re
import sys

time_line = None
with open(sys.argv[1], encoding="utf-8", errors="ignore") as fh:
    for line in fh:
        if "time:" in line and "[" in line and "]" in line:
            time_line = line.strip()

if time_line is None:
    raise SystemExit("Failed to find Criterion time line")

match = re.search(
    r"time:\s*\[\s*([0-9.]+)\s*(ms|s)\s+([0-9.]+)\s*(ms|s)\s+([0-9.]+)\s*(ms|s)\s*\]",
    time_line,
)
if match is None:
    raise SystemExit(f"Failed to parse Criterion time line: {time_line}")

value = float(match.group(3))
unit = match.group(4)
if unit == "ms":
    value /= 1000.0
print(f"{value:.6f}")
PY
}

configure_and_build() {
    cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
        -DUSE_DIVISION_BENCH=ON \
        -DENABLE_TEST=ON \
        -DENABLE_AXELL=ON \
        -DCMAKE_BUILD_TYPE=Release
    cmake --build "${BUILD_DIR}" --target \
        intdivnr intdivnr_par \
        intdivreducednr intdivreducednr_par \
        intdivlutnr intdivlutnr_par \
        -j "${BUILD_JOBS}"
}

run_tfhepp_case() {
    local cpu_set="$1"
    local threads="$2"
    local binary="$3"
    local log_file="$4"

    echo "Running ${binary} on CPUs ${cpu_set} with OMP_NUM_THREADS=${threads}" | tee "${log_file}"
    taskset -c "${cpu_set}" env \
        OMP_NUM_THREADS="${threads}" \
        OMP_PROC_BIND=close \
        OMP_PLACES=cores \
        "${BUILD_DIR}/test/axell/integer-wise/${binary}" | tee -a "${log_file}"

    # Release builds disable assert(), so check reported errors explicitly.
    awk '
        /^count:/ {
            if ($0 !~ /^count:[[:space:]]*0[[:space:]]*$/) {
                print "Correctness check failed: " $0 > "/dev/stderr"
                failed = 1
            }
        }
        END { exit failed }
    ' "${log_file}" || {
        echo "Benchmark reported incorrect results; see ${log_file}" >&2
        return 1
    }
}

run_tfhers_case() {
    local cpu_set="$1"
    local threads="$2"
    local log_file="$3"

    echo "Running tfhe-rs div_rem_parallelized on CPUs ${cpu_set} with RAYON_NUM_THREADS=${threads}" | tee "${log_file}"
    taskset -c "${cpu_set}" env \
        CARGO_TARGET_DIR="${TFHERS_CARGO_TARGET_DIR}" \
        RAYON_NUM_THREADS="${threads}" \
        __TFHE_RS_BENCH_TYPE=latency \
        __TFHE_RS_BENCH_BIT_SIZES_SET=all \
        RUSTFLAGS="${TFHERS_RUSTFLAGS:-"-C target-cpu=native"}" \
        cargo "${CARGO_TOOLCHAIN}" bench \
        --manifest-path "${TFHERS_BENCHMARK_MANIFEST}" \
        --bench integer \
        -p tfhe-benchmark \
        --features=integer,internal-keycache,pbs-stats,avx512 \
        -- "${TFHERS_BENCH_ID}" --noplot | tee -a "${log_file}"
}

require_cmd cmake
require_cmd cargo
require_cmd taskset
require_cmd python3
require_cmd awk
require_cmd git

prepare_tfhers_release_checkout

if [[ -z "${TFHERS_PARAM_ALIAS}" ]]; then
    TFHERS_PARAM_ALIAS="${TFHERS_INTEGER_PARAM_ALIAS}"
fi

TFHERS_BENCH_ID="integer::div_rem_parallelized::${TFHERS_PARAM_ALIAS}::16_bits"

configure_and_build

run_tfhepp_case "0" "1" "intdivnr" "${LOG_DIR}/intdivnr_single.log"
run_tfhepp_case "0-7" "8" "intdivnr_par" "${LOG_DIR}/intdivnr_multi.log"
run_tfhepp_case "0" "1" "intdivreducednr" "${LOG_DIR}/intdivreducednr_single.log"
run_tfhepp_case "0-7" "8" "intdivreducednr_par" "${LOG_DIR}/intdivreducednr_multi.log"
run_tfhepp_case "0" "1" "intdivlutnr" "${LOG_DIR}/intdivlutnr_single.log"
run_tfhepp_case "0-7" "8" "intdivlutnr_par" "${LOG_DIR}/intdivlutnr_multi.log"
run_tfhers_case "0" "1" "${LOG_DIR}/tfhers_single.log"
run_tfhers_case "0-7" "8" "${LOG_DIR}/tfhers_multi.log"

summary_file="${LOG_DIR}/summary.tsv"
{
    echo -e "impl\tmode\tmethod\tinv_s\tdiv_s\ttotal_s"
    echo -e "TFHEpp\tsingle\tRodehefferNR\t$(extract_tfhepp_ms "Inv" "${LOG_DIR}/intdivnr_single.log" | awk '{printf "%.6f", $1/1000}')\t$(extract_tfhepp_ms "DivByMul" "${LOG_DIR}/intdivnr_single.log" | awk '{printf "%.6f", $1/1000}')\t$(extract_tfhepp_ms "Sum" "${LOG_DIR}/intdivnr_single.log" | awk '{printf "%.6f", $1/1000}')"
    echo -e "TFHEpp\tmulti\tRodehefferNR\t$(extract_tfhepp_ms "Inv" "${LOG_DIR}/intdivnr_multi.log" | awk '{printf "%.6f", $1/1000}')\t$(extract_tfhepp_ms "DivByMul" "${LOG_DIR}/intdivnr_multi.log" | awk '{printf "%.6f", $1/1000}')\t$(extract_tfhepp_ms "Sum" "${LOG_DIR}/intdivnr_multi.log" | awk '{printf "%.6f", $1/1000}')"
    echo -e "TFHEpp\tsingle\tReducedNR\t$(extract_tfhepp_ms "Inv" "${LOG_DIR}/intdivreducednr_single.log" | awk '{printf "%.6f", $1/1000}')\t$(extract_tfhepp_ms "DivByMul" "${LOG_DIR}/intdivreducednr_single.log" | awk '{printf "%.6f", $1/1000}')\t$(extract_tfhepp_ms "Sum" "${LOG_DIR}/intdivreducednr_single.log" | awk '{printf "%.6f", $1/1000}')"
    echo -e "TFHEpp\tmulti\tReducedNR\t$(extract_tfhepp_ms "Inv" "${LOG_DIR}/intdivreducednr_multi.log" | awk '{printf "%.6f", $1/1000}')\t$(extract_tfhepp_ms "DivByMul" "${LOG_DIR}/intdivreducednr_multi.log" | awk '{printf "%.6f", $1/1000}')\t$(extract_tfhepp_ms "Sum" "${LOG_DIR}/intdivreducednr_multi.log" | awk '{printf "%.6f", $1/1000}')"
    echo -e "TFHEpp\tsingle\tLUTNR\t$(extract_tfhepp_ms "Inv" "${LOG_DIR}/intdivlutnr_single.log" | awk '{printf "%.6f", $1/1000}')\t$(extract_tfhepp_ms "DivByMul" "${LOG_DIR}/intdivlutnr_single.log" | awk '{printf "%.6f", $1/1000}')\t$(extract_tfhepp_ms "Sum" "${LOG_DIR}/intdivlutnr_single.log" | awk '{printf "%.6f", $1/1000}')"
    echo -e "TFHEpp\tmulti\tLUTNR\t$(extract_tfhepp_ms "Inv" "${LOG_DIR}/intdivlutnr_multi.log" | awk '{printf "%.6f", $1/1000}')\t$(extract_tfhepp_ms "DivByMul" "${LOG_DIR}/intdivlutnr_multi.log" | awk '{printf "%.6f", $1/1000}')\t$(extract_tfhepp_ms "Sum" "${LOG_DIR}/intdivlutnr_multi.log" | awk '{printf "%.6f", $1/1000}')"
    echo -e "TFHE-rs\tsingle\tdiv_rem_parallelized\tNA\tNA\t$(extract_criterion_seconds "${LOG_DIR}/tfhers_single.log")"
    echo -e "TFHE-rs\tmulti\tdiv_rem_parallelized\tNA\tNA\t$(extract_criterion_seconds "${LOG_DIR}/tfhers_multi.log")"
} > "${summary_file}"

cat "${summary_file}"
echo
cat "${LOG_DIR}/tfhers_checkout.txt"
echo
echo "Logs written under ${LOG_DIR}"
