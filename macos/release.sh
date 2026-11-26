#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SWARM_DIR="$(cd "$HERE/.." && pwd)"
DIST_DIR="$SWARM_DIR/dist"
APP_BUNDLE="$DIST_DIR/Open Swarm.app"
ZIP_PATH="$DIST_DIR/OpenSwarm-macOS.zip"

bash "$HERE/build_app.sh"

if [ ! -d "$APP_BUNDLE" ]; then
  echo "ERROR: Expected app bundle not found at $APP_BUNDLE" >&2
  exit 1
fi

echo "Verifying code signature on $APP_BUNDLE..."
codesign --verify --deep --strict "$APP_BUNDLE"

rm -f "$ZIP_PATH"
echo "Creating release archive at $ZIP_PATH..."
ditto -c -k --sequesterRsrc --keepParent "$APP_BUNDLE" "$ZIP_PATH"

shasum -a 256 "$ZIP_PATH"
echo "RELEASE_OK"
