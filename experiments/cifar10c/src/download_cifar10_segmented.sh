#!/usr/bin/env bash
set -euo pipefail

URL='https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
TOTAL_BYTES=170498071
CHUNK_BYTES=16777216
EXPECTED_MD5='c58f30108f718f92721af3b95e74349a'
OUT_DIR="${1:-round13_second_domain/data}"
OUT_FILE="${OUT_DIR}/cifar-10-python.tar.gz"
PART_DIR="${OUT_DIR}/cifar10_parts"

mkdir -p "${OUT_DIR}" "${PART_DIR}"
if [[ -f "${OUT_FILE}" ]]; then
  current_size="$(stat -c '%s' "${OUT_FILE}")"
  if [[ "${current_size}" -ne "${TOTAL_BYTES}" ]]; then
    mv "${OUT_FILE}" "${OUT_FILE}.slow.partial"
  fi
fi

download_part() {
  local part_index="$1"
  local start=$((part_index * CHUNK_BYTES))
  local end=$((start + CHUNK_BYTES - 1))
  if [[ "${end}" -ge $((TOTAL_BYTES - 1)) ]]; then end=$((TOTAL_BYTES - 1)); fi
  local expected=$((end - start + 1))
  local target
  target="$(printf '%s/part_%03d.bin' "${PART_DIR}" "${part_index}")"
  if [[ -f "${target}" ]] && [[ "$(stat -c '%s' "${target}")" -eq "${expected}" ]]; then return; fi
  curl -L --fail --retry 10 --retry-all-errors --silent --show-error --range "${start}-${end}" --output "${target}.tmp" "${URL}"
  if [[ "$(stat -c '%s' "${target}.tmp")" -ne "${expected}" ]]; then exit 1; fi
  mv "${target}.tmp" "${target}"
  echo "completed CIFAR-10 part ${part_index}"
}
export URL TOTAL_BYTES CHUNK_BYTES PART_DIR
export -f download_part

N_PARTS=$(((TOTAL_BYTES + CHUNK_BYTES - 1) / CHUNK_BYTES))
seq 0 $((N_PARTS - 1)) | xargs -P 3 -n 1 bash -c 'download_part "$0"'
truncate -s "${TOTAL_BYTES}" "${OUT_FILE}"
for part_index in $(seq 0 $((N_PARTS - 1))); do
  part_file="$(printf '%s/part_%03d.bin' "${PART_DIR}" "${part_index}")"
  dd if="${part_file}" of="${OUT_FILE}" bs=1M seek=$((part_index * 16)) conv=notrunc status=none
done
actual_md5="$(md5sum "${OUT_FILE}" | awk '{print $1}')"
if [[ "${actual_md5}" != "${EXPECTED_MD5}" ]]; then echo "MD5 mismatch: ${actual_md5}" >&2; exit 1; fi
echo "CIFAR-10 verified: ${actual_md5}"
