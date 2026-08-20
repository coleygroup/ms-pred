#!/usr/bin/env bash

set -u

SFTP_HOST="${SFTP_HOST:-iceberg-ms.mit.edu}"
SFTP_PORT="${SFTP_PORT:-2222}"
SFTP_USER="${SFTP_USER:-runzhong@mit.edu}"
SSH_USER="${SSH_USER:-coley-group}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/coley-group/atlas_nist/h_plus_out_mgf}"
VIRTUAL_ROOT="${VIRTUAL_ROOT:-/nist23/h_plus}"
TARGET_DIR="${TARGET_DIR:-C20}"
LOCAL_DIR="${LOCAL_DIR:-/tmp/iceberg_sftp_speed}"
REQUESTS="${REQUESTS:-64}"
BUFFER_SIZE="${BUFFER_SIZE:-65536}"

REMOTE_FIND_ROOT="${REMOTE_ROOT%/}/${TARGET_DIR#/}"
VIRTUAL_FIND_ROOT="${VIRTUAL_ROOT%/}/${TARGET_DIR#/}"
LIST_FILE="${LOCAL_DIR%/}/file_list.txt"
CMD_FILE="${LOCAL_DIR%/}/sftp_commands.txt"
START_FILE="${LOCAL_DIR%/}/start_time"
END_FILE="${LOCAL_DIR%/}/end_time"
DOWNLOAD_DIR="${LOCAL_DIR%/}/download"

echo "SFTP benchmark"
echo "  host:        ${SFTP_HOST}:${SFTP_PORT}"
echo "  user:        ${SFTP_USER}"
echo "  remote path: ${VIRTUAL_FIND_ROOT}"
echo "  local path:  ${DOWNLOAD_DIR}"
echo

rm -rf "$LOCAL_DIR"
mkdir -p "$DOWNLOAD_DIR"

echo "Fetching remote file list over SSH..."
ssh "${SSH_USER}@${SFTP_HOST}" \
  "cd '$REMOTE_FIND_ROOT' && find . -type f -name '*.mgf' -printf '%P\n' | sort" \
  > "$LIST_FILE"

file_count=$(wc -l < "$LIST_FILE" | tr -d ' ')
if [ "$file_count" = "0" ]; then
  echo "No .mgf files found under ${REMOTE_FIND_ROOT}" >&2
  exit 1
fi

echo "Preparing local directories for ${file_count} files..."
while IFS= read -r relpath; do
  dir=$(dirname "$relpath")
  mkdir -p "${DOWNLOAD_DIR}/${dir}"
done < "$LIST_FILE"

{
  echo "!python3 -c 'import time; print(time.time())' > '${START_FILE}'"
  while IFS= read -r relpath; do
    echo "get ${VIRTUAL_FIND_ROOT}/${relpath} ${DOWNLOAD_DIR}/${relpath}"
  done < "$LIST_FILE"
  echo "!python3 -c 'import time; print(time.time())' > '${END_FILE}'"
  echo "bye"
} > "$CMD_FILE"

echo
echo "Starting SFTP download. Enter your WebUI password when prompted."
sftp -R "$REQUESTS" -B "$BUFFER_SIZE" -P "$SFTP_PORT" -o "User=${SFTP_USER}" "$SFTP_HOST" < "$CMD_FILE"
status=$?
if [ ! -s "$START_FILE" ]; then
  python3 -c 'import time; print(time.time())' > "$START_FILE"
fi
if [ ! -s "$END_FILE" ]; then
  python3 -c 'import time; print(time.time())' > "$END_FILE"
fi

downloaded_files=$(find "$DOWNLOAD_DIR" -type f -name '*.mgf' | wc -l | tr -d ' ')
downloaded_bytes=$(find "$DOWNLOAD_DIR" -type f -name '*.mgf' -printf '%s\n' | awk '{s += $1} END {print s + 0}')

python3 - <<PY
from pathlib import Path

status = int("${status}")
expected_files = int("${file_count}")
downloaded_files = int("${downloaded_files}")
downloaded_bytes = int("${downloaded_bytes}")
start = float(Path("${START_FILE}").read_text().strip())
end = float(Path("${END_FILE}").read_text().strip())
elapsed = max(end - start, 0.001)

print()
print("Result")
print(f"  status:      {status}")
print(f"  files:       {downloaded_files} / {expected_files}")
print(f"  transferred: {downloaded_bytes / 1_000_000:.2f} MB")
print(f"  elapsed:     {elapsed:.2f} s")
print(f"  speed:       {downloaded_bytes / 1_000_000 / elapsed:.2f} MB/s")
print(f"  speed:       {downloaded_bytes / 1_048_576 / elapsed:.2f} MiB/s")
print(f"  output:      ${DOWNLOAD_DIR}")

if status != 0:
    raise SystemExit(status)
if downloaded_files != expected_files:
    raise SystemExit("Downloaded file count does not match the server file list")
PY
