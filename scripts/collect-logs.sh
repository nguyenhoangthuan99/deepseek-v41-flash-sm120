#!/usr/bin/env bash
# Dump container + launcher logs to logs/<name>-<ts>.log and refresh the
# latest-*.log symlinks. The logs/ dir is served over HTTP so the files are
# readable in a browser:  python3 -m http.server 8899 --directory logs
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
NAME=${1:-dsv41-mxfp8test}
TS=$(date +%Y%m%dT%H%M%SZ)
OUT="$ROOT/logs/${NAME}-${TS}.log"
{
  echo "=== docker logs $NAME @ $TS ==="
  docker logs "$NAME" 2>&1 || echo "(container not found)"
} > "$OUT"
ln -sf "$(basename "$OUT")" "$ROOT/logs/latest-${NAME}.log"
echo "wrote $OUT ($(wc -l < "$OUT") lines)"
echo "serve:  cd $ROOT && python3 -m http.server 8899 --directory logs"
echo "browse: http://<host>:8899/$(basename "$OUT")"
