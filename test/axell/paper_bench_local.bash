#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-paper-bench-local}"
BUILD_JOBS="${BUILD_JOBS:-8}"
LOG_DIR="${LOG_DIR:-${BUILD_DIR}/logs/paper-bench-local}"
source "${SCRIPT_DIR}/../lib/tfhers_release_helpers.bash"

TFHERS_PARAM_ALIAS="${TFHERS_PARAM_ALIAS:-}"

mkdir -p "${LOG_DIR}"

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "Missing required command: $1" >&2
        exit 1
    }
}

extract_plain_ms() {
    local log_file="$1"
    python3 - "${log_file}" <<'PY'
import re
import sys

value = None
with open(sys.argv[1], encoding="utf-8", errors="ignore") as fh:
    for line in fh:
        line = line.strip()
        match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)ms", line)
        if match is not None:
            value = match.group(1)

if value is None:
    raise SystemExit(f"Failed to find trailing ms line in {sys.argv[1]}")

print(value)
PY
}

extract_tfhepp_div_ms() {
    local label="$1"
    local log_file="$2"
    grep "^${label}:" "${log_file}" | tail -n1 | sed -E "s/^${label}:([0-9.]+)ms.*/\\1/"
}

extract_bench_br_metric() {
    local section_name="$1"
    local label="$2"
    local log_file="$3"
    python3 - "${section_name}" "${label}" "${log_file}" <<'PY'
import re
import sys

section_name, label, log_file = sys.argv[1:4]
current = None
value = None

with open(log_file, encoding="utf-8", errors="ignore") as fh:
    for raw in fh:
        line = raw.strip()
        section_match = re.fullmatch(r"--- (.+) ---", line)
        if section_match is not None:
            current = section_match.group(1)
            continue
        if current != section_name:
            continue
        metric_match = re.fullmatch(
            rf"{re.escape(label)}:\s*([0-9]+(?:\.[0-9]+)?)ms(?:\s*\(.*\))?",
            line,
        )
        if metric_match is not None:
            value = metric_match.group(1)

if value is None:
    raise SystemExit(
        f"Failed to find metric {label!r} in section {section_name!r} from {log_file}"
    )

print(value)
PY
}

extract_labelled_ms() {
    local label="$1"
    local log_file="$2"
    python3 - "${label}" "${log_file}" <<'PY'
import re
import sys

label, log_file = sys.argv[1:3]
value = None

with open(log_file, encoding="utf-8", errors="ignore") as fh:
    for raw in fh:
        line = raw.strip()
        metric_match = re.fullmatch(
            rf"{re.escape(label)}:\s*([0-9]+(?:\.[0-9]+)?)ms(?:\s*\(.*\))?",
            line,
        )
        if metric_match is not None:
            value = metric_match.group(1)

if value is None:
    raise SystemExit(f"Failed to find metric {label!r} in {log_file}")

print(value)
PY
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
    raise SystemExit(f"Failed to find Criterion time line in {sys.argv[1]}")

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
        bench_br \
        hfp_bench \
        longmul longmull ulshift clz \
        intdivnr intdivnr32 intdivnr_par \
        intdivreducednr intdivreducednr32 intdivreducednr_par \
        intdivlutnr intdivlutnr32 intdivlutnr_par \
        -j "${BUILD_JOBS}"
}

run_tfhepp_case() {
    local cpu_set="$1"
    local threads="$2"
    local binary_path="$3"
    local log_file="$4"
    shift 4

    echo "Running $(basename "${binary_path}") on CPUs ${cpu_set} with OMP_NUM_THREADS=${threads}" | tee "${log_file}"
    taskset -c "${cpu_set}" env \
        OMP_NUM_THREADS="${threads}" \
        OMP_PROC_BIND=close \
        OMP_PLACES=cores \
        "${binary_path}" "$@" | tee -a "${log_file}"

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

run_tfhers_integer_case() {
    local cpu_set="$1"
    local threads="$2"
    local bench_id="$3"
    local log_file="$4"

    echo "Running tfhe-rs integer bench ${bench_id} on CPUs ${cpu_set} with RAYON_NUM_THREADS=${threads}" | tee "${log_file}"
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
        -- "${bench_id}" --noplot | tee -a "${log_file}"
}

run_tfhers_boot_case() {
    local cpu_set="$1"
    local log_file="$2"

    echo "Running tfhe-rs core_crypto-ks-pbs bench (2M40) on CPUs ${cpu_set}" | tee "${log_file}"
    taskset -c "${cpu_set}" env \
        CARGO_TARGET_DIR="${TFHERS_CARGO_TARGET_DIR}" \
        __TFHE_RS_BENCH_TYPE=latency \
        RUSTFLAGS="${TFHERS_RUSTFLAGS:-"-C target-cpu=native"}" \
        cargo "${CARGO_TOOLCHAIN}" bench \
        --manifest-path "${TFHERS_BENCHMARK_MANIFEST}" \
        --bench core_crypto-ks-pbs \
        -p tfhe-benchmark \
        --features=shortint,internal-keycache,avx512 \
        -- "TUNIFORM_2M40" --noplot | tee -a "${log_file}"
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

TFHERS_DIV16_BENCH_ID="integer::div_rem_parallelized::${TFHERS_PARAM_ALIAS}::16_bits"
TFHERS_DIV32_BENCH_ID="integer::div_rem_parallelized::${TFHERS_PARAM_ALIAS}::32_bits"
TFHERS_MUL16_BENCH_ID="integer::mul_parallelized::${TFHERS_PARAM_ALIAS}::16_bits"
TFHERS_LSHIFT16_BENCH_ID="integer::left_shift_parallelized::${TFHERS_PARAM_ALIAS}::16_bits"
TFHERS_CLZ16_BENCH_ID="integer::leading_zeros_parallelized::${TFHERS_PARAM_ALIAS}::16_bits"

configure_and_build

run_tfhepp_case "0" "1" "${BUILD_DIR}/test/axell/bench_br" "${LOG_DIR}/bench_br.log"
run_tfhepp_case "0" "1" "${BUILD_DIR}/test/axell/integer-wise/longmul" "${LOG_DIR}/longmul.log"
run_tfhepp_case "0" "1" "${BUILD_DIR}/test/axell/integer-wise/longmull" "${LOG_DIR}/longmull.log"
run_tfhepp_case "0" "1" "${BUILD_DIR}/test/axell/integer-wise/ulshift" "${LOG_DIR}/ulshift.log"
run_tfhepp_case "0" "1" "${BUILD_DIR}/test/axell/integer-wise/clz" "${LOG_DIR}/clz.log"

run_tfhepp_case "0" "1" "${BUILD_DIR}/test/axell/integer-wise/intdivnr" "${LOG_DIR}/intdivnr_single.log"
run_tfhepp_case "0" "1" "${BUILD_DIR}/test/axell/integer-wise/intdivreducednr" "${LOG_DIR}/intdivreducednr_single.log"
run_tfhepp_case "0" "1" "${BUILD_DIR}/test/axell/integer-wise/intdivlutnr" "${LOG_DIR}/intdivlutnr_single.log"
run_tfhepp_case "0" "1" "${BUILD_DIR}/test/axell/integer-wise/intdivnr32" "${LOG_DIR}/intdivnr32_single.log"
run_tfhepp_case "0" "1" "${BUILD_DIR}/test/axell/integer-wise/intdivreducednr32" "${LOG_DIR}/intdivreducednr32_single.log"
run_tfhepp_case "0" "1" "${BUILD_DIR}/test/axell/integer-wise/intdivlutnr32" "${LOG_DIR}/intdivlutnr32_single.log"
run_tfhepp_case "0-7" "8" "${BUILD_DIR}/test/axell/integer-wise/intdivnr_par" "${LOG_DIR}/intdivnr_multi.log"
run_tfhepp_case "0-7" "8" "${BUILD_DIR}/test/axell/integer-wise/intdivreducednr_par" "${LOG_DIR}/intdivreducednr_multi.log"
run_tfhepp_case "0-7" "8" "${BUILD_DIR}/test/axell/integer-wise/intdivlutnr_par" "${LOG_DIR}/intdivlutnr_multi.log"
run_tfhepp_case "0" "1" "${BUILD_DIR}/test/axell/integer-wise/hfp_bench" "${LOG_DIR}/hfp_bench.log" 10 bfloat16

run_tfhers_boot_case "0" "${LOG_DIR}/tfhers_boot.log"
run_tfhers_integer_case "0" "1" "${TFHERS_MUL16_BENCH_ID}" "${LOG_DIR}/tfhers_mul16.log"
run_tfhers_integer_case "0" "1" "${TFHERS_LSHIFT16_BENCH_ID}" "${LOG_DIR}/tfhers_lshift16.log"
run_tfhers_integer_case "0" "1" "${TFHERS_CLZ16_BENCH_ID}" "${LOG_DIR}/tfhers_clz16.log"
run_tfhers_integer_case "0" "1" "${TFHERS_DIV16_BENCH_ID}" "${LOG_DIR}/tfhers_div16_single.log"
run_tfhers_integer_case "0" "1" "${TFHERS_DIV32_BENCH_ID}" "${LOG_DIR}/tfhers_div32_single.log"
run_tfhers_integer_case "0-7" "8" "${TFHERS_DIV16_BENCH_ID}" "${LOG_DIR}/tfhers_div16_multi.log"

boot_summary="${LOG_DIR}/boot.tsv"
{
    echo -e "impl\tmetric\ttime_ms"
    echo -e "TFHEpp\tBlindRotate\t$(extract_bench_br_metric "lvlh2param (N=2048, division-bench)" "BR" "${LOG_DIR}/bench_br.log")"
    echo -e "TFHEpp\tIdentityKeySwitch\t$(extract_bench_br_metric "lvlh2param (N=2048, division-bench)" "IKS" "${LOG_DIR}/bench_br.log")"
    echo -e "TFHEpp\tKS-PBS\t$(awk 'BEGIN{printf "%.3f", '"$(extract_bench_br_metric "lvlh2param (N=2048, division-bench)" "BR" "${LOG_DIR}/bench_br.log")"' + '"$(extract_bench_br_metric "lvlh2param (N=2048, division-bench)" "IKS" "${LOG_DIR}/bench_br.log")"'}')"
    echo -e "TFHE-rs\tBlindRotate\tN/A"
    echo -e "TFHE-rs\tIdentityKeySwitch\tN/A"
    echo -e "TFHE-rs\tKS-PBS\t$(extract_criterion_seconds "${LOG_DIR}/tfhers_boot.log" | awk '{printf "%.3f", $1*1000}')"
} > "${boot_summary}"

misc_summary="${LOG_DIR}/misc.tsv"
{
    echo -e "impl\tfunction\ttime_ms"
    echo -e "TFHEpp\tLongUMul\t$(extract_plain_ms "${LOG_DIR}/longmul.log")"
    echo -e "TFHEpp\tLongUMulL\t$(extract_plain_ms "${LOG_DIR}/longmull.log")"
    echo -e "TFHEpp\tLongLeftShift\t$(extract_plain_ms "${LOG_DIR}/ulshift.log")"
    echo -e "TFHEpp\tHomCLZ\t$(extract_plain_ms "${LOG_DIR}/clz.log")"
    echo -e "TFHE-rs\tLongUMul\tN/A"
    echo -e "TFHE-rs\tLongUMulL\t$(extract_criterion_seconds "${LOG_DIR}/tfhers_mul16.log" | awk '{printf "%.3f", $1*1000}')"
    echo -e "TFHE-rs\tLongLeftShift\t$(extract_criterion_seconds "${LOG_DIR}/tfhers_lshift16.log" | awk '{printf "%.3f", $1*1000}')"
    echo -e "TFHE-rs\tHomCLZ\t$(extract_criterion_seconds "${LOG_DIR}/tfhers_clz16.log" | awk '{printf "%.3f", $1*1000}')"
} > "${misc_summary}"

div_summary="${LOG_DIR}/division.tsv"
{
    echo -e "impl\tmode\tsize_bits\tmethod\tinv_s\tdiv_s\ttotal_s"
    echo -e "TFHEpp\tsingle\t16\tRodehefferNR\t$(extract_tfhepp_div_ms "Inv" "${LOG_DIR}/intdivnr_single.log" | awk '{printf "%.6f", $1/1000}')\t$(extract_tfhepp_div_ms "DivByMul" "${LOG_DIR}/intdivnr_single.log" | awk '{printf "%.6f", $1/1000}')\t$(extract_tfhepp_div_ms "Sum" "${LOG_DIR}/intdivnr_single.log" | awk '{printf "%.6f", $1/1000}')"
    echo -e "TFHEpp\tsingle\t16\tReducedNR\t$(extract_tfhepp_div_ms "Inv" "${LOG_DIR}/intdivreducednr_single.log" | awk '{printf "%.6f", $1/1000}')\t$(extract_tfhepp_div_ms "DivByMul" "${LOG_DIR}/intdivreducednr_single.log" | awk '{printf "%.6f", $1/1000}')\t$(extract_tfhepp_div_ms "Sum" "${LOG_DIR}/intdivreducednr_single.log" | awk '{printf "%.6f", $1/1000}')"
    echo -e "TFHEpp\tsingle\t16\tLUTNR\t$(extract_tfhepp_div_ms "Inv" "${LOG_DIR}/intdivlutnr_single.log" | awk '{printf "%.6f", $1/1000}')\t$(extract_tfhepp_div_ms "DivByMul" "${LOG_DIR}/intdivlutnr_single.log" | awk '{printf "%.6f", $1/1000}')\t$(extract_tfhepp_div_ms "Sum" "${LOG_DIR}/intdivlutnr_single.log" | awk '{printf "%.6f", $1/1000}')"
    echo -e "TFHEpp\tsingle\t32\tRodehefferNR\t$(extract_tfhepp_div_ms "Inv" "${LOG_DIR}/intdivnr32_single.log" | awk '{printf "%.6f", $1/1000}')\t$(extract_tfhepp_div_ms "DivByMul" "${LOG_DIR}/intdivnr32_single.log" | awk '{printf "%.6f", $1/1000}')\t$(extract_tfhepp_div_ms "Sum" "${LOG_DIR}/intdivnr32_single.log" | awk '{printf "%.6f", $1/1000}')"
    echo -e "TFHEpp\tsingle\t32\tReducedNR\t$(extract_tfhepp_div_ms "Inv" "${LOG_DIR}/intdivreducednr32_single.log" | awk '{printf "%.6f", $1/1000}')\t$(extract_tfhepp_div_ms "DivByMul" "${LOG_DIR}/intdivreducednr32_single.log" | awk '{printf "%.6f", $1/1000}')\t$(extract_tfhepp_div_ms "Sum" "${LOG_DIR}/intdivreducednr32_single.log" | awk '{printf "%.6f", $1/1000}')"
    echo -e "TFHEpp\tsingle\t32\tLUTNR\t$(extract_tfhepp_div_ms "Inv" "${LOG_DIR}/intdivlutnr32_single.log" | awk '{printf "%.6f", $1/1000}')\t$(extract_tfhepp_div_ms "DivByMul" "${LOG_DIR}/intdivlutnr32_single.log" | awk '{printf "%.6f", $1/1000}')\t$(extract_tfhepp_div_ms "Sum" "${LOG_DIR}/intdivlutnr32_single.log" | awk '{printf "%.6f", $1/1000}')"
    echo -e "TFHEpp\tmulti\t16\tRodehefferNR\t$(extract_tfhepp_div_ms "Inv" "${LOG_DIR}/intdivnr_multi.log" | awk '{printf "%.6f", $1/1000}')\t$(extract_tfhepp_div_ms "DivByMul" "${LOG_DIR}/intdivnr_multi.log" | awk '{printf "%.6f", $1/1000}')\t$(extract_tfhepp_div_ms "Sum" "${LOG_DIR}/intdivnr_multi.log" | awk '{printf "%.6f", $1/1000}')"
    echo -e "TFHEpp\tmulti\t16\tReducedNR\t$(extract_tfhepp_div_ms "Inv" "${LOG_DIR}/intdivreducednr_multi.log" | awk '{printf "%.6f", $1/1000}')\t$(extract_tfhepp_div_ms "DivByMul" "${LOG_DIR}/intdivreducednr_multi.log" | awk '{printf "%.6f", $1/1000}')\t$(extract_tfhepp_div_ms "Sum" "${LOG_DIR}/intdivreducednr_multi.log" | awk '{printf "%.6f", $1/1000}')"
    echo -e "TFHEpp\tmulti\t16\tLUTNR\t$(extract_tfhepp_div_ms "Inv" "${LOG_DIR}/intdivlutnr_multi.log" | awk '{printf "%.6f", $1/1000}')\t$(extract_tfhepp_div_ms "DivByMul" "${LOG_DIR}/intdivlutnr_multi.log" | awk '{printf "%.6f", $1/1000}')\t$(extract_tfhepp_div_ms "Sum" "${LOG_DIR}/intdivlutnr_multi.log" | awk '{printf "%.6f", $1/1000}')"
    echo -e "TFHE-rs\tsingle\t16\tdiv_rem_parallelized\tNA\tNA\t$(extract_criterion_seconds "${LOG_DIR}/tfhers_div16_single.log")"
    echo -e "TFHE-rs\tsingle\t32\tdiv_rem_parallelized\tNA\tNA\t$(extract_criterion_seconds "${LOG_DIR}/tfhers_div32_single.log")"
    echo -e "TFHE-rs\tmulti\t16\tdiv_rem_parallelized\tNA\tNA\t$(extract_criterion_seconds "${LOG_DIR}/tfhers_div16_multi.log")"
} > "${div_summary}"

{
    echo "format,operation,min_ms,mean_ms,max_ms"
    grep '^BFloat16,div_' "${LOG_DIR}/hfp_bench.log"
} > "${LOG_DIR}/bfloat16.csv"

cat "${boot_summary}"
echo
cat "${misc_summary}"
echo
cat "${div_summary}"
echo
cat "${LOG_DIR}/bfloat16.csv"
echo
cat "${LOG_DIR}/tfhers_checkout.txt"
echo
echo "Logs written under ${LOG_DIR}"
