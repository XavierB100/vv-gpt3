#!/bin/bash
# ============================================================
# VV-GPT3 — macOS App Bundle Creator
# Run this once to create a clickable .app on your Desktop/Dock
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_NAME="VV-GPT3"
APP_PATH="$HOME/Desktop/${APP_NAME}.app"
ICON_SRC="$SCRIPT_DIR/homemade_gpt_icon.png"

echo ""
echo "=================================================="
echo "  🤖  VV-GPT3 — Mac App Builder"
echo "=================================================="
echo ""

# --- Create .app bundle structure ---
mkdir -p "${APP_PATH}/Contents/MacOS"
mkdir -p "${APP_PATH}/Contents/Resources"

# --- Write the Info.plist ---
cat > "${APP_PATH}/Contents/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>launch</string>
    <key>CFBundleIdentifier</key>
    <string>com.xavierblake.vv-gpt3</string>
    <key>CFBundleName</key>
    <string>VV-GPT3</string>
    <key>CFBundleDisplayName</key>
    <string>VV-GPT3</string>
    <key>CFBundleVersion</key>
    <string>3.0</string>
    <key>CFBundleShortVersionString</key>
    <string>3.0</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>LSMinimumSystemVersion</key>
    <string>11.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
EOF

# --- Write the launch executable ---
cat > "${APP_PATH}/Contents/MacOS/launch" << LAUNCHER
#!/bin/bash
cd "${SCRIPT_DIR}"
open -a Terminal "${SCRIPT_DIR}/VV_GPT3.command"
LAUNCHER

chmod +x "${APP_PATH}/Contents/MacOS/launch"

# --- Convert PNG icon to .icns if possible ---
if [ -f "$ICON_SRC" ]; then
    ICONSET_DIR="/tmp/HomeMadeGPT.iconset"
    mkdir -p "$ICONSET_DIR"
    # Generate all required icon sizes
    for SIZE in 16 32 64 128 256 512; do
        sips -z $SIZE $SIZE "$ICON_SRC" --out "$ICONSET_DIR/icon_${SIZE}x${SIZE}.png" &>/dev/null
        DOUBLE=$((SIZE * 2))
        sips -z $DOUBLE $DOUBLE "$ICON_SRC" --out "$ICONSET_DIR/icon_${SIZE}x${SIZE}@2x.png" &>/dev/null
    done
    iconutil -c icns "$ICONSET_DIR" -o "${APP_PATH}/Contents/Resources/AppIcon.icns" 2>/dev/null
    rm -rf "$ICONSET_DIR"
    echo "🎨  Custom AI icon applied!"
else
    echo "ℹ️   No custom icon found, using default."
fi

# --- Touch the app to refresh macOS icon cache ---
touch "${APP_PATH}"

echo ""
echo "✅  VV-GPT3.app created on your Desktop!"
echo ""
echo "👉  To add it to your Dock:"
echo "    1. Find 'VV-GPT3' on your Desktop"
echo "    2. Drag it to your Dock"
echo "    3. That's it — click it any time to launch!"
echo ""
echo "⚠️   First launch: macOS may ask you to confirm opening."
echo "     Right-click → Open → Open (to bypass Gatekeeper once)"
echo ""
