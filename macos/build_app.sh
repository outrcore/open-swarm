#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SWARM_DIR="$(cd "$HERE/.." && pwd)"
DIST_DIR="$SWARM_DIR/dist"
APP_BUNDLE="$DIST_DIR/Open Swarm.app"
CONTENTS="$APP_BUNDLE/Contents"
MACOS_DIR="$CONTENTS/MacOS"
RES_DIR="$CONTENTS/Resources"
BUNDLE_SWARM="$RES_DIR/open_swarm"

rm -rf "$APP_BUNDLE"
mkdir -p "$MACOS_DIR" "$BUNDLE_SWARM"

echo "Compiling OpenSwarmApp.swift..."
mkdir -p /tmp/swift_modcache
swiftc -O \
  -module-cache-path /tmp/swift_modcache \
  -framework AppKit \
  -framework WebKit \
  "$HERE/OpenSwarmApp.swift" \
  -o "$MACOS_DIR/OpenSwarm"

cp "$SWARM_DIR/desktop_server.py" "$BUNDLE_SWARM/"
cp "$SWARM_DIR/desktop_ui.html" "$BUNDLE_SWARM/"
cp "$SWARM_DIR/swarm_run.py" "$BUNDLE_SWARM/"
cp "$SWARM_DIR/metering.py" "$BUNDLE_SWARM/"
cp "$SWARM_DIR/continuity_store.py" "$BUNDLE_SWARM/"
cp "$SWARM_DIR/fleet_scanner.py" "$BUNDLE_SWARM/"
cp "$SWARM_DIR/orchestrator.py" "$BUNDLE_SWARM/"
cp "$SWARM_DIR/agent_tools.py" "$BUNDLE_SWARM/"
cp "$SWARM_DIR/agent_loop.py" "$BUNDLE_SWARM/"
cp "$SWARM_DIR/research_loop.py" "$BUNDLE_SWARM/"
cp "$SWARM_DIR/device_mesh.py" "$BUNDLE_SWARM/"
cp "$SWARM_DIR/swarm_node.py" "$BUNDLE_SWARM/"
chmod +x "$BUNDLE_SWARM/desktop_server.py"

cat > "$CONTENTS/Info.plist" << 'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key>
  <string>en</string>
  <key>CFBundleExecutable</key>
  <string>OpenSwarm</string>
  <key>CFBundleIdentifier</key>
  <string>com.outrcore.openswarm</string>
  <key>CFBundleInfoDictionaryVersion</key>
  <string>6.0</string>
  <key>CFBundleName</key>
  <string>Open Swarm</string>
  <key>CFBundleDisplayName</key>
  <string>Open Swarm</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleShortVersionString</key>
  <string>0.4.0</string>
  <key>CFBundleVersion</key>
  <string>4</string>
  <key>LSMinimumSystemVersion</key>
  <string>13.0</string>
  <key>NSHighResolutionCapable</key>
  <true/>
  <key>NSAppTransportSecurity</key>
  <dict>
    <key>NSAllowsLocalNetworking</key>
    <true/>
  </dict>
</dict>
</plist>
PLIST

if command -v codesign >/dev/null 2>&1; then
  echo "Ad-hoc signing Open Swarm.app..."
  codesign --force --deep --sign - "$APP_BUNDLE"
fi

echo "Built: $APP_BUNDLE"
