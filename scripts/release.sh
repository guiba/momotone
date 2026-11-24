#!/bin/sh
# Build an **unsigned** release DMG for momotone.
#
#   scripts/release.sh           # build target/release/Momotone.dmg
#
# Requires: macOS, cargo, hdiutil. Deliberately NOT signed/notarized — for a
# personal, internal build. Signed/notarized releases need a Developer ID
# certificate and an Apple ID; see the release workflow before publishing
# anything outside your own machine.
#
# The DMG bundles the ONNX models from assets/ (the app resolves default
# asset paths against the bundle when installed), so a binary-only release
# is the only other shape you would need with `--no-bundle`.
set -eu

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

VERSION=$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)
APP_NAME=Momotone
BIN_NAME=momotone
OUT=target/release/${APP_NAME}.dmg
APP=target/release/${APP_NAME}.app

echo "==> cargo build --release"
cargo build --release

echo "==> assembling ${APP_NAME}.app (v${VERSION})"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources/assets"
cp "$(dirname "$0")/../target/release/${BIN_NAME}" "$APP/Contents/MacOS/${BIN_NAME}"
chmod +x "$APP/Contents/MacOS/${BIN_NAME}"

# Model assets ship inside the bundle. Only the files that exist are copied,
# so a `--no-assets` shape is possible by moving assets aside.
for f in hand_detector.onnx hand_landmarks_detector.onnx anchors.csv yolo_hand_pose.onnx; do
    if [ -f "assets/$f" ]; then
        cp "assets/$f" "$APP/Contents/Resources/assets/$f"
    else
        echo "    ! assets/$f missing — the app's default path for this model is not bundled"
    fi
done

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>${APP_NAME}</string>
    <key>CFBundleDisplayName</key><string>${APP_NAME}</string>
    <key>CFBundleIdentifier</key><string>com.guiba.momotone</string>
    <key>CFBundleVersion</key><string>${VERSION}</string>
    <key>CFBundleShortVersionString</key><string>${VERSION}</string>
    <key>CFBundleExecutable</key><string>${BIN_NAME}</string>
    <key>CFBundleIcon</key><string>AppIcon</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
    <key>LSMinimumSystemVersion</key><string>12.0</string>
    <key>NSHighResolutionCapable</key><true/>
    <key>LSUIElement</key><false/>
</dict>
</plist>
PLIST

# Drag-to-Applications layout. Alias /Applications holds the target folder.
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"

echo "==> hdiutil create ${OUT}"
rm -f "$OUT"
hdiutil create \
    -volname "${APP_NAME}" \
    -srcfolder "$STAGE" \
    -ov \
    -format UDZO \
    "$OUT"

echo "==> done: ${OUT}"
echo "    unsigned — Gatekeeper will block first run; right-click -> Open, or 'xattr -d com.apple.quarantine ${APP_NAME}.app' after mounting"
