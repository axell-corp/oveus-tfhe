#!/usr/bin/env bash

# Shared helper for reproducible tfhe-rs benchmark checkouts.
#
# Default behavior:
# - clones a clean official tfhe-rs release checkout under BUILD_DIR
# - checks out the pinned release tag in detached HEAD mode
# - applies the small runtime patch needed to benchmark integer ops and
#   native shortint KS-PBS with 2M40
# - records the exact release, commit, and patch diff under LOG_DIR

TFHERS_REMOTE_URL_DEFAULT="https://github.com/zama-ai/tfhe-rs.git"
# Latest official release as of 2026-04-14.
TFHERS_RELEASE_TAG_DEFAULT="tfhe-rs-1.5.4"
TFHERS_INTEGER_PARAM_BASENAME_DEFAULT="PARAM_MESSAGE_2_CARRY_2_KS_PBS_TUNIFORM_2M40"

prepare_tfhers_release_checkout() {
    : "${ROOT_DIR:?ROOT_DIR must be set before sourcing tfhers_release_helpers.bash}"
    : "${BUILD_DIR:?BUILD_DIR must be set before sourcing tfhers_release_helpers.bash}"
    : "${LOG_DIR:?LOG_DIR must be set before sourcing tfhers_release_helpers.bash}"

    TFHERS_REMOTE_URL="${TFHERS_REMOTE_URL:-${TFHERS_REMOTE_URL_DEFAULT}}"
    TFHERS_RELEASE_TAG="${TFHERS_RELEASE_TAG:-${TFHERS_RELEASE_TAG_DEFAULT}}"
    TFHERS_INTEGER_PARAM_BASENAME="${TFHERS_INTEGER_PARAM_BASENAME:-${TFHERS_INTEGER_PARAM_BASENAME_DEFAULT}}"
    TFHERS_DIR="${TFHERS_DIR:-${BUILD_DIR}/extern/${TFHERS_RELEASE_TAG}}"
    TFHERS_WORKSPACE_DIR="${TFHERS_WORKSPACE_DIR:-${BUILD_DIR}/extern/tfhers-workspace}"
    TFHERS_CARGO_TARGET_DIR="${TFHERS_CARGO_TARGET_DIR:-${BUILD_DIR}/cargo-target/${TFHERS_RELEASE_TAG}}"
    TFHERS_BENCHMARK_MANIFEST="${TFHERS_BENCHMARK_MANIFEST:-${TFHERS_WORKSPACE_DIR}/tfhe-rs/tfhe-benchmark/Cargo.toml}"
    TFHERS_BOOLEAN_MANIFEST="${TFHERS_BOOLEAN_MANIFEST:-${TFHERS_WORKSPACE_DIR}/bench-tfhers/Cargo.toml}"
    TFHERS_FFT_BOOLEAN_MANIFEST="${TFHERS_FFT_BOOLEAN_MANIFEST:-${TFHERS_WORKSPACE_DIR}/bench-tfhers-fft/Cargo.toml}"

    require_cmd git

    if [[ -e "${TFHERS_DIR}" && ! -d "${TFHERS_DIR}/.git" ]]; then
        echo "Expected a git checkout at ${TFHERS_DIR}, but found a non-git path" >&2
        exit 1
    fi

    mkdir -p "$(dirname "${TFHERS_DIR}")"

    if [[ ! -d "${TFHERS_DIR}/.git" ]]; then
        git clone --branch "${TFHERS_RELEASE_TAG}" --depth 1 \
            "${TFHERS_REMOTE_URL}" "${TFHERS_DIR}"
    else
        git -C "${TFHERS_DIR}" remote set-url origin "${TFHERS_REMOTE_URL}"
        git -C "${TFHERS_DIR}" fetch --depth 1 origin \
            "refs/tags/${TFHERS_RELEASE_TAG}:refs/tags/${TFHERS_RELEASE_TAG}"
        git -C "${TFHERS_DIR}" checkout --detach -f "${TFHERS_RELEASE_TAG}"
        git -C "${TFHERS_DIR}" reset --hard
        git -C "${TFHERS_DIR}" clean -fdx
    fi

    git -C "${TFHERS_DIR}" checkout --detach -f "${TFHERS_RELEASE_TAG}"
    git -C "${TFHERS_DIR}" reset --hard
    git -C "${TFHERS_DIR}" clean -fdx

    patch_tfhers_bench_for_2m40
    resolve_tfhers_integer_param_alias
    ensure_tfhers_workspace_links
    resolve_tfhers_cargo_toolchain
    record_tfhers_checkout_metadata
}

patch_tfhers_bench_for_2m40() {
    python3 - "${TFHERS_DIR}" <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
aliases_path = root / "tfhe-benchmark" / "src" / "params_aliases.rs"
params_path = root / "tfhe-benchmark" / "src" / "params.rs"

aliases = aliases_path.read_text(encoding="utf-8")
alias_name = "BENCH_PARAM_MESSAGE_2_CARRY_2_KS_PBS_TUNIFORM_2M40"
if alias_name not in aliases:
    pattern = re.compile(
        r"(pub const BENCH_PARAM_MESSAGE_2_CARRY_2_KS_PBS_TUNIFORM_2M64: ClassicPBSParameters =\n"
        r"\s+)(V1_[0-9_]+_PARAM_MESSAGE_2_CARRY_2_KS_PBS_TUNIFORM_2M64);"
    )
    match = pattern.search(aliases)
    if match is None:
        raise SystemExit(f"Failed to locate 2M64 alias anchor in {aliases_path}")
    version_prefix = match.group(2).replace("2M64", "2M40")
    insertion = (
        f"{match.group(0)}\n"
        f"    pub const {alias_name}: ClassicPBSParameters =\n"
        f"        {version_prefix};"
    )
    aliases = aliases.replace(match.group(0), insertion, 1)
    aliases_path.write_text(aliases, encoding="utf-8")

text = params_path.read_text(encoding="utf-8")

shortint_old = """    pub const SHORTINT_BENCH_PARAMS_TUNIFORM: [ClassicPBSParameters; 4] = [
        BENCH_PARAM_MESSAGE_1_CARRY_1_KS_PBS_TUNIFORM_2M128,
        BENCH_PARAM_MESSAGE_2_CARRY_2_KS_PBS_TUNIFORM_2M128,
        BENCH_PARAM_MESSAGE_3_CARRY_3_KS_PBS_TUNIFORM_2M128,
        BENCH_PARAM_MESSAGE_4_CARRY_4_KS_PBS_TUNIFORM_2M128,
    ];"""
shortint_new = """    pub const SHORTINT_BENCH_PARAMS_TUNIFORM: [ClassicPBSParameters; 4] = [
        BENCH_PARAM_MESSAGE_1_CARRY_1_KS_PBS_TUNIFORM_2M128,
        BENCH_PARAM_MESSAGE_2_CARRY_2_KS_PBS_TUNIFORM_2M40,
        BENCH_PARAM_MESSAGE_3_CARRY_3_KS_PBS_TUNIFORM_2M128,
        BENCH_PARAM_MESSAGE_4_CARRY_4_KS_PBS_TUNIFORM_2M128,
    ];"""
if shortint_new not in text:
    if shortint_old not in text:
        raise SystemExit(f"Failed to find SHORTINT_BENCH_PARAMS_TUNIFORM in {params_path}")
    text = text.replace(shortint_old, shortint_new, 1)

integer_old = """                #[cfg(not(any(feature = "gpu", feature = "hpu")))]
                let params = vec![BENCH_PARAM_MESSAGE_2_CARRY_2_KS32_PBS.into()];"""
integer_new = """                #[cfg(not(any(feature = "gpu", feature = "hpu")))]
                let params = vec![BENCH_PARAM_MESSAGE_2_CARRY_2_KS_PBS_TUNIFORM_2M40.into()];"""

if integer_new not in text:
    if integer_old not in text:
        raise SystemExit(f"Failed to find integer benchmark parameter selection in {params_path}")
    text = text.replace(integer_old, integer_new, 1)

params_path.write_text(text, encoding="utf-8")
PY
}

resolve_tfhers_integer_param_alias() {
    if [[ -n "${TFHERS_INTEGER_PARAM_ALIAS:-}" ]]; then
        return
    fi

    TFHERS_INTEGER_PARAM_ALIAS="$(python3 - "${TFHERS_DIR}" "${TFHERS_INTEGER_PARAM_BASENAME}" <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
basename = sys.argv[2]
pattern = re.compile(rf"pub const (V1_[0-9_]+_{re.escape(basename)}):")
matches = []
for path in sorted((root / "tfhe" / "src" / "shortint" / "parameters").rglob("*.rs")):
    text = path.read_text(encoding="utf-8", errors="ignore")
    matches.extend(pattern.findall(text))

if not matches:
    raise SystemExit(f"Failed to resolve tfhe-rs parameter alias for {basename}")

print(sorted(set(matches))[-1])
PY
)"
}

ensure_tfhers_workspace_links() {
    mkdir -p "${TFHERS_WORKSPACE_DIR}"
    ln -sfn "${ROOT_DIR}/bench-tfhers" "${TFHERS_WORKSPACE_DIR}/bench-tfhers"
    ln -sfn "${ROOT_DIR}/bench-tfhers-fft" "${TFHERS_WORKSPACE_DIR}/bench-tfhers-fft"
    ln -sfn "${TFHERS_DIR}" "${TFHERS_WORKSPACE_DIR}/tfhe-rs"
}

resolve_tfhers_cargo_toolchain() {
    if [[ -n "${CARGO_TOOLCHAIN:-}" ]]; then
        return
    fi

    local channel
    channel="$(python3 - "${TFHERS_DIR}/rust-toolchain.toml" <<'PY'
import re
import sys

text = open(sys.argv[1], encoding="utf-8").read()
match = re.search(r'channel\s*=\s*"([^"]+)"', text)
if match is None:
    raise SystemExit("Failed to parse rust-toolchain.toml")
print(match.group(1))
PY
)"
    CARGO_TOOLCHAIN="+${channel}"
}

record_tfhers_checkout_metadata() {
    mkdir -p "${LOG_DIR}"

    {
        echo "remote_url=${TFHERS_REMOTE_URL}"
        echo "release_tag=${TFHERS_RELEASE_TAG}"
        echo "commit=$(git -C "${TFHERS_DIR}" rev-parse HEAD)"
        echo "describe=$(git -C "${TFHERS_DIR}" describe --tags --always --dirty)"
        echo "cargo_toolchain=${CARGO_TOOLCHAIN}"
        echo "integer_param_alias=${TFHERS_INTEGER_PARAM_ALIAS}"
        echo "runtime_patch=cpu_integer_bench_param:${TFHERS_INTEGER_PARAM_ALIAS}"
        echo "bench_bit_sizes_env=all"
    } > "${LOG_DIR}/tfhers_checkout.txt"

    git -C "${TFHERS_DIR}" diff -- \
        tfhe-benchmark/src/params.rs \
        tfhe-benchmark/src/params_aliases.rs \
        > "${LOG_DIR}/tfhers_runtime_patch.diff"
}
