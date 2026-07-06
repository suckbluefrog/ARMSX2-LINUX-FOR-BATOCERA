#!/usr/bin/env bash
# Full Linux ARM64 release pipeline: deps (optional) -> Release build -> AppImage.
set -euo pipefail

SCRIPTDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOTDIR="$(cd "$SCRIPTDIR/.." && pwd)"
DEPSDIR="${DEPSDIR:-$ROOTDIR/deps}"
BUILDDIR="${BUILDDIR:-$ROOTDIR/build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
BUILD_DEPS="${BUILD_DEPS:-auto}"
APPIMAGE_NAME="${APPIMAGE_NAME:-ARMSX2-linux-Qt-arm64}"
JOBS="${JOBS:-$(nproc)}"

usage() {
	cat <<EOF
Usage: $0 [options]

  DEPSDIR=path       Prefix for bundled deps (default: \$ROOT/deps)
  BUILDDIR=path      CMake build directory (default: \$ROOT/build)
  BUILD_TYPE=Release|Devel
  BUILD_DEPS=auto|1|0   auto = build deps if Qt6Core.pc missing
  SKIP_APPIMAGE=1    Only compile the binary, skip AppImage packaging
  BUILD_QTAPNG=0     Passed to build-dependencies-qt.sh (animated PNG plugin)

Example:
  $0
  BUILD_DEPS=1 $0
  SKIP_APPIMAGE=1 BUILD_TYPE=Devel $0
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
	usage
	exit 0
fi

echo "=== ARMSX2 Linux ARM64 release ==="
echo "Root:     $ROOTDIR"
echo "Deps:     $DEPSDIR"
echo "Build:    $BUILDDIR"
echo "Type:     $BUILD_TYPE"
echo "AppImage: ${SKIP_APPIMAGE:-0}"

if ! command -v clang++ >/dev/null; then
	echo "Error: clang++ is required."
	exit 1
fi

need_deps=0
if [[ "$BUILD_DEPS" == "1" ]]; then
	need_deps=1
elif [[ "$BUILD_DEPS" == "auto" && ! -f "$DEPSDIR/lib/pkgconfig/shaderc.pc" ]]; then
	need_deps=1
fi

if [[ "$need_deps" -eq 1 ]]; then
	echo ""
	echo ">>> Building dependencies into $DEPSDIR (this takes a long time)..."
	export PKG_CONFIG_PATH="$DEPSDIR/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
	BUILD_FFMPEG="${BUILD_FFMPEG:-1}" \
	BUILD_QTAPNG="${BUILD_QTAPNG:-0}" \
		"$ROOTDIR/.github/workflows/scripts/linux/build-dependencies-qt.sh" "$DEPSDIR"
else
	echo ">>> Using existing deps in $DEPSDIR"
fi

export PKG_CONFIG_PATH="$DEPSDIR/lib/pkgconfig:${PKG_CONFIG_PATH:-}"

echo ""
echo ">>> Configuring CMake ($BUILD_TYPE)..."
cmake -B "$BUILDDIR" -G Ninja \
	-DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
	-DCMAKE_C_COMPILER=clang \
	-DCMAKE_CXX_COMPILER=clang++ \
	-DCMAKE_PREFIX_PATH="$DEPSDIR" \
	-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON \
	-DENABLE_TESTS=OFF \
	-DLTO_PCSX2_CORE=ON \
	-DUSE_LINKED_FFMPEG=ON \
	"$ROOTDIR"

echo ""
echo ">>> Compiling..."
cmake --build "$BUILDDIR" --parallel "$JOBS"

if [[ -f "$ROOTDIR/bin/resources/patches.zip" ]]; then
	cp -a "$ROOTDIR/bin/resources/patches.zip" "$BUILDDIR/bin/resources/" 2>/dev/null || true
fi

echo ""
echo "Binary: $BUILDDIR/bin/pcsx2-qt"

if [[ "${SKIP_APPIMAGE:-0}" == "1" ]]; then
	echo "SKIP_APPIMAGE=1 — done."
	exit 0
fi

if ! command -v wget >/dev/null && ! command -v curl >/dev/null; then
	echo "Error: wget or curl required for AppImage tooling."
	exit 1
fi

echo ""
echo ">>> Packaging AppImage..."
PACKDIR="$(mktemp -d)"
trap 'rm -rf "$PACKDIR"' EXIT
cd "$PACKDIR"
"$ROOTDIR/.github/workflows/scripts/linux/appimage-qt.sh" \
	"$(realpath "$ROOTDIR")" \
	"$(realpath "$BUILDDIR")" \
	"$(realpath "$DEPSDIR")" \
	"$APPIMAGE_NAME"
mv "${APPIMAGE_NAME}.AppImage" "$ROOTDIR/"
echo ""
echo "=== Done ==="
echo "AppImage: $ROOTDIR/${APPIMAGE_NAME}.AppImage"
echo "Run: ./${APPIMAGE_NAME}.AppImage"
