#!/usr/bin/env bash
# Build ARMSX2 for Linux ARM64 with native JIT (commit 09109bb / 2.3.5.1)
set -euo pipefail

SCRIPTDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOTDIR="$(cd "$SCRIPTDIR/.." && pwd)"
BUILDDIR="${BUILDDIR:-$ROOTDIR/build-arm64}"
BUILD_TYPE="${BUILD_TYPE:-Devel}"

echo "=== ARMSX2 Linux ARM64 build ==="
echo "Source: $ROOTDIR"
echo "Build:  $BUILDDIR"
echo "Type:   $BUILD_TYPE"

if ! command -v clang++ >/dev/null; then
	echo "Error: clang++ is required. PCSX2 officially supports Clang on Linux."
	exit 1
fi

# Core dependencies for Debian/Ubuntu/Armbian (adjust for your distro if needed)
MISSING_PKGS=()
for pkg in \
	liblz4-dev libwebp-dev libsdl3-dev libfreetype-dev \
	libplutovg-dev libplutosvg-dev libryml-dev \
	libcurl4-openssl-dev libfontconfig-dev libdbus-1-dev \
	libbacktrace-dev libegl1-mesa-dev libwayland-dev \
	libshaderc-dev spirv-tools libvulkan-dev \
	qt6-base-dev qt6-base-private-dev libqt6svg6-dev \
	ninja-build lld clang; do
	if ! dpkg -s "$pkg" >/dev/null 2>&1; then
		MISSING_PKGS+=("$pkg")
	fi
done

if ((${#MISSING_PKGS[@]} > 0)); then
	echo ""
	echo "Missing packages (install with sudo apt install ...):"
	printf '  %s\n' "${MISSING_PKGS[@]}"
	echo ""
	echo "Example:"
	echo "  sudo apt install ${MISSING_PKGS[*]}"
	exit 1
fi

DEPSDIR="${DEPSDIR:-$ROOTDIR/deps}"

cmake -B "$BUILDDIR" -G Ninja \
	-DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
	-DCMAKE_C_COMPILER=clang \
	-DCMAKE_CXX_COMPILER=clang++ \
	-DCMAKE_PREFIX_PATH="$DEPSDIR" \
	-DENABLE_TESTS=OFF \
	-DLTO_PCSX2_CORE=ON \
	"$ROOTDIR"

cmake --build "$BUILDDIR" --parallel "$(nproc)"

echo ""
echo "Build complete. Binary:"
echo "  $BUILDDIR/bin/pcsx2-qt"
echo ""
echo "The native ARM64 JIT (pcsx2_macrec backend) is enabled by default on aarch64."
echo "Ensure EE/IOP/VU recompilers are enabled in Settings > Emulation Settings > CPU."
