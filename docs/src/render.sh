#!/usr/bin/env bash
# Renders docs/banner.png and docs/recall.png from the HTML in docs/src with headless Chrome.
set -euo pipefail
cd "$(dirname "$0")"
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
"$CHROME" --headless=new --hide-scrollbars --force-device-scale-factor=2 --window-size=1200,600 \
  --screenshot="$PWD/../banner.png" "file://$PWD/banner.html" 2>/dev/null
"$CHROME" --headless=new --hide-scrollbars --force-device-scale-factor=2 --window-size=1200,640 \
  --screenshot="$PWD/../recall.png" "file://$PWD/recall.html" 2>/dev/null
