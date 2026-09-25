#!/usr/bin/env bash
# Download and unpack an official OpenMC cross-section library.
#
# This is the step that stops most OpenMC installs: nothing runs until
# OPENMC_CROSS_SECTIONS points at a real cross_sections.xml.  The library is
# several GB, so it lives in a mounted data directory and never in the image
# or the repository.
#
# Usage:
#   scripts/fetch_data.sh [library] [destination]
#
#   library      endfb-vii.1 (default) | endfb-viii.0 | endfb-viii.1
#   destination  default /data
#
# Re-running is cheap: an already-unpacked library is detected and skipped.
set -euo pipefail

LIB="${1:-endfb-vii.1}"
DEST="${2:-/data}"

# Box "shared/static" ids from https://openmc.org/data.  They are opaque, so
# the directory each one unpacks into is recorded alongside it and verified
# after extraction -- a silently changed link fails loudly instead of leaving
# a half-populated data directory.
case "$LIB" in
  endfb-vii.1)
    ID=9igk353zpy8fn9ttvtrqgzvw1vtejoz6 ; DIR=endfb-vii.1-hdf5 ;;
  endfb-viii.0)
    ID=uhbxlrx7hvxqw27psymfbhi7bx7s6u6a ; DIR=endfb-viii.0-hdf5 ;;
  endfb-viii.1)
    ID=6qr7jezzihkj9p9esl5jn19qgpujyjyz ; DIR=endfb-viii.1-hdf5 ;;
  *)
    echo "unknown library '$LIB'" >&2 ; exit 2 ;;
esac

URL="https://anl.box.com/shared/static/${ID}.xz"
XML="${DEST}/${DIR}/cross_sections.xml"

if [[ -f "$XML" ]]; then
  echo "[fetch_data] already present: $XML"
  exit 0
fi

mkdir -p "$DEST"
ARCHIVE="${DEST}/${DIR}.tar.xz"

# Integrity-test first: a complete archive left behind by an earlier run must
# not be re-downloaded, and `curl -C -` on an already-complete file asks for a
# range past EOF and errors out rather than succeeding quietly.
if [[ -f "$ARCHIVE" ]] && xz -t "$ARCHIVE" 2>/dev/null; then
  echo "[fetch_data] reusing complete archive $ARCHIVE"
else
  echo "[fetch_data] downloading $LIB (several GB) -> $ARCHIVE"
  # -C - resumes a partial download, which matters on a link this large.
  curl -L --fail --retry 5 --retry-delay 5 --retry-all-errors \
       -C - -o "$ARCHIVE" "$URL"
  xz -t "$ARCHIVE"
fi

echo "[fetch_data] extracting into $DEST"
tar -xf "$ARCHIVE" -C "$DEST"

# The upstream tarball carries mode 0750 and the packager's uid, so a
# container running as a non-root user cannot read it.  Unpacking has to be
# done as root; reading must not be.
chmod -R a+rX "${DEST}/${DIR}"

if [[ ! -f "$XML" ]]; then
  echo "[fetch_data] expected $XML after extraction; the upstream link may have" >&2
  echo "             changed. Check https://openmc.org/data" >&2
  exit 1
fi

rm -f "$ARCHIVE"
echo "[fetch_data] done. Set:"
echo "    export OPENMC_CROSS_SECTIONS=$XML"
