#!/usr/bin/env bash
# Builds the API reference published at https://python.reference.langfuse.com.
#
# Use this instead of calling pdoc directly: it applies the template overrides
# in pdoc-templates/ and ships the 404 page, both of which the hosted site
# needs. Set PDOC_CANONICAL_BASE_URL to build for a different origin.
set -euo pipefail

OUT_DIR="${1:-docs}"

# Resolve a relative output path against the caller's working directory before
# cd'ing to the repo root, so `build_reference_docs.sh out` from elsewhere does
# not silently write to <repo>/out.
case "$OUT_DIR" in
  /*) ;;
  *) OUT_DIR="$PWD/$OUT_DIR" ;;
esac

cd "$(dirname "$0")/.."

uv run --group docs pdoc \
  -o "$OUT_DIR" \
  --docformat google \
  --logo "https://langfuse.com/langfuse_logo.svg" \
  --logo-link "https://langfuse.com" \
  --template-directory pdoc-templates \
  --edit-url "langfuse=https://github.com/langfuse/langfuse-python/blob/main/langfuse/" \
  --no-show-source \
  langfuse

# Cloudflare Pages serves the closest index.html with a 200 for any unmatched
# path unless the output contains a 404.html, which turns every stale or
# mistyped URL into an indexable duplicate of the landing page.
cp pdoc-templates/404.html "$OUT_DIR/404.html"

echo "Reference docs written to $OUT_DIR/"
