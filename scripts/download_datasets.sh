#!/usr/bin/env bash
# Download Calabi–Yau datasets reproducibly.
#
# Usage:
#   ./scripts/download_datasets.sh [--data-dir dataset] [--skip-existing] [--with-checksum]
#
# Environment overrides (optional):
#   CICY_LIST_URL, CICY_4FOLDS_URL, REF_POLY_D3_URL
#   CICY_LIST_MD5, CICY_4FOLDS_MD5, REF_POLY_D3_MD5
#
# Notes:
# - Files already present are skipped by default if --skip-existing is set.
# - Checksums are validated if --with-checksum and corresponding *_MD5 is provided.
# - This script is idempotent.

set -euo pipefail

DATA_DIR="dataset"
SKIP_EXISTING=false
WITH_CHECKSUM=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir)
      DATA_DIR="$2"; shift 2;;
    --skip-existing)
      SKIP_EXISTING=true; shift;;
    --with-checksum)
      WITH_CHECKSUM=true; shift;;
    -h|--help)
      sed -n '1,60p' "$0"; exit 0;;
    *)
      echo "Unknown option: $1" >&2; exit 1;;
  esac
done

mkdir -p "$DATA_DIR"

# Default primary sources (provided by user)
# You can override or add mirrors via environment variables below.
DEFAULT_CICY_LIST_URLS="https://www-thphys.physics.ox.ac.uk/projects/CalabiYau/cicylist/cicylist.txt"
DEFAULT_CICY_4FOLDS_URLS="https://www-thphys.physics.ox.ac.uk/projects/CalabiYau/Cicy4folds/cicy4folds.txt.zip"
DEFAULT_REF_POLY_D3_URLS="http://hep.itp.tuwien.ac.at/%7Ekreuzer/pub/K3/RefPoly.d3"

# Provide URLs via environment or use defaults above.
# You may also pass multiple mirrors separated by spaces.
[[ -z "${CICY_LIST_URLS:-}" ]] && CICY_LIST_URLS="$DEFAULT_CICY_LIST_URLS"
[[ -z "${CICY_4FOLDS_URLS:-}" ]] && CICY_4FOLDS_URLS="$DEFAULT_CICY_4FOLDS_URLS"
[[ -z "${REF_POLY_D3_URLS:-}" ]] && REF_POLY_D3_URLS="$DEFAULT_REF_POLY_D3_URLS"

# Backward compatibility for single-URL env vars
[[ -n "${CICY_LIST_URL:-}" ]] && CICY_LIST_URLS="${CICY_LIST_URL}"
[[ -n "${CICY_4FOLDS_URL:-}" ]] && CICY_4FOLDS_URLS="${CICY_4FOLDS_URL}"
[[ -n "${REF_POLY_D3_URL:-}" ]] && REF_POLY_D3_URLS="${REF_POLY_D3_URL}"

CICY_LIST_MD5="${CICY_LIST_MD5:-}"
CICY_4FOLDS_MD5="${CICY_4FOLDS_MD5:-}"
REF_POLY_D3_MD5="${REF_POLY_D3_MD5:-}"

FILES=(
  "cicylist.txt|$CICY_LIST_URLS|$CICY_LIST_MD5"
  "cicy4folds.txt|$CICY_4FOLDS_URLS|$CICY_4FOLDS_MD5"
  "RefPoly.d3|$REF_POLY_D3_URLS|$REF_POLY_D3_MD5"
)

have_cmd() { command -v "$1" >/dev/null 2>&1; }

download() {
  local url="$1" dest="$2"
  echo "→ Downloading $url -> $dest"
  if have_cmd curl; then
    curl -L --fail --retry 3 --retry-delay 2 -o "$dest" "$url"
  elif have_cmd wget; then
    wget -O "$dest" "$url"
  else
    echo "Error: curl or wget required" >&2; exit 1
  fi
}

url_ok() {
  local url="$1"
  if have_cmd curl; then
    curl -sI "$url" | head -n 1 | grep -qE "200|302|301"
  elif have_cmd wget; then
    wget --spider -q "$url"
  else
    return 1
  fi
}

check_md5() {
  local file="$1" expected="$2"
  if [[ -z "$expected" ]]; then return 0; fi
  if ! have_cmd md5sum && ! have_cmd md5; then
    echo "Warning: no md5 tool found; skipping checksum for $file"; return 0
  fi
  local actual
  if have_cmd md5sum; then
    actual=$(md5sum "$file" | awk '{print $1}')
  else
    actual=$(md5 -q "$file")
  fi
  if [[ "$actual" != "$expected" ]]; then
    echo "Checksum mismatch for $file: expected $expected got $actual" >&2
    return 1
  fi
  echo "✔ MD5 OK for $file"
}

for entry in "${FILES[@]}"; do
  IFS='|' read -r name urls md5 <<< "$entry"
  dest="$DATA_DIR/$name"

  if [[ -f "$dest" && "$SKIP_EXISTING" == true ]]; then
    echo "✓ Exists, skipping: $dest"
  elif [[ -f "$dest" && -s "$dest" ]]; then
    echo "✓ Found local $dest, using it"
  else
    # Try each mirror until one works
    success=false
    for u in $urls; do
      if [[ -n "$u" ]] && url_ok "$u"; then
        if [[ "$u" == *.zip ]]; then
          tmp_zip="$dest.zip"
          download "$u" "$tmp_zip" || { rm -f "$tmp_zip"; continue; }
          echo "→ Extracting $tmp_zip to $dest"
          if have_cmd unzip; then
            unzip -p "$tmp_zip" > "$dest" || { rm -f "$tmp_zip"; continue; }
          else
            python - <<PYEOF
import sys, zipfile
zf = zipfile.ZipFile(r"$tmp_zip", 'r')
name = zf.namelist()[0]
with zf.open(name) as src, open(r"$dest", 'wb') as out:
    out.write(src.read())
PYEOF
          fi
          rm -f "$tmp_zip"
          success=true
          break
        else
          download "$u" "$dest" && success=true && break
        fi
      fi
    done
    if [[ "$success" != true ]]; then
      echo "Warning: No reachable URL for $name. Provide URLs via env (e.g. CICY_LIST_URLS)." >&2
      continue
    fi
  fi

  if [[ "$WITH_CHECKSUM" == true ]]; then
    check_md5 "$dest" "$md5" || { echo "Checksum failed for $name"; exit 1; }
  fi

  # Basic sanity check
  if [[ ! -s "$dest" ]]; then
    echo "Error: downloaded file is empty: $dest" >&2; exit 1
  fi
  echo "✓ Ready: $dest ($(du -h "$dest" | cut -f1))"
  echo
done

echo "All datasets are in: $DATA_DIR"
