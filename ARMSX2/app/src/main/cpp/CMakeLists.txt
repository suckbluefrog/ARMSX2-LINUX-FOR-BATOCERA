# Setting it to a range tells it that it supports the features on the newer
# versions as well, avoiding setting policies.
cmake_minimum_required(VERSION 3.16...3.25)

# Enabling this cmake policy gets rid of warnings regarding LTO.
cmake_policy(SET CMP0069 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
	message(FATAL_ERROR "PCSX2 does not support in-tree builds.  Please make a build directory that is not the PCSX2 source directory and generate your CMake project there using either `cmake -B build_directory` or by running cmake from the build directory.")
endif()

# Project Name
project(Pcsx2 C CXX)

# Variable to check that people use the good file
set(TOP_CMAKE_WAS_SOURCED TRUE)

# set module path
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# Write binaries to the bin directory under the build directory.
# This makes it simple to run development builds directly out of the build directory.
if (UNIX AND NOT APPLE)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")
	set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")
endif()

# include some generic functions to ensure correctness of the env
include(Pcsx2Utils)

check_no_parenthesis_in_path()
detect_operating_system()
detect_compiler()

# Disable desktop-only features on Android
if(ANDROID)
	set(ENABLE_QT_UI OFF CACHE BOOL "" FORCE)
endif()

#-------------------------------------------------------------------------------
# Include specific module
include(BuildParameters)
include(SearchForStuff)

# Must be done after SearchForStuff
get_git_version_info()
write_svnrev_h()

# make common
add_subdirectory(common)

# make pcsx2
add_subdirectory(pcsx2)
#add_subdirectory(pcsx2-qt)

# Android JNI shared library
if(ANDROID)
	set(ARMSX2_EMUCORE_LIBRARY_NAME "emucore_4k" CACHE STRING "Android emucore shared library target/output name")
	add_subdirectory(3rdparty/googletest EXCLUDE_FROM_ALL)
	# libadrenotools: custom Vulkan ICD loading for Adreno devices. Provides
	# the static `adrenotools` library (linked into emucore for the loader
	# splice) plus four runtime hook .so's (hook_impl/main_hook/
	# file_redirect_hook/gsl_alloc_hook) that ship in the apk's
	# lib/<abi>/ and get dlopen'd by the loaded driver.
	add_subdirectory(3rdparty/adrenotools)
	add_library(${ARMSX2_EMUCORE_LIBRARY_NAME} SHARED
		native-lib.cpp
		tests/arm64/codegen_tests.cpp
		tests/arm64/codegen_tests_main.cpp
		tests/arm64/run_tests.cpp
		tests/core/run_patch_tests.cpp
		tests/ee/run_ee_seq_tests.cpp
		tests/ee/run_ee_tests.cpp
		tests/mvu/run_mvu_tests.cpp
		tests/vif/run_vif_tests.cpp
	)
	target_link_libraries(${ARMSX2_EMUCORE_LIBRARY_NAME} PRIVATE
		PCSX2
		PCSX2_FLAGS
		common
		imgui
		gtest
		libzip::zip
		SDL3::SDL3
		shaderc
		android
		log
		vixl
		adrenotools
	)
	target_include_directories(${ARMSX2_EMUCORE_LIBRARY_NAME} PRIVATE
		"${CMAKE_SOURCE_DIR}"
		"${CMAKE_SOURCE_DIR}/pcsx2"
	)
endif()

# Desktop Linux executable
if(NOT ANDROID AND UNIX AND NOT APPLE)
	add_executable(armsx2
		desktop/main.cpp
	)
	target_link_libraries(armsx2 PRIVATE
		PCSX2
		common
		imgui
		libzip::zip
		SDL3::SDL3
		simpleini
		vixl
	)
	target_include_directories(armsx2 PRIVATE
		"${CMAKE_SOURCE_DIR}"
		"${CMAKE_SOURCE_DIR}/pcsx2"
	)
	target_compile_definitions(armsx2 PRIVATE
		ARMSX2_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
	)

	if(USE_OPENGL)
		# Force libEGL into DT_NEEDED. The OpenGL renderer otherwise dlopens
		# libEGL lazily, which trips LD_PRELOAD-based GL interposers (MangoHud)
		# that index the EGL symbol table at process start.
		target_link_libraries(armsx2 PRIVATE -Wl,--no-as-needed EGL -Wl,--as-needed)
	endif()
endif()

# Updater is Windows only for now.
if (WIN32)
	add_subdirectory(updater)
endif()

## tests
#if(ENABLE_TESTS)
#	add_subdirectory(3rdparty/googletest)
#	add_subdirectory(tests/ctest)
#endif()

# gsrunner
if(ENABLE_GSRUNNER)
	if (NOT WIN32)
		message(WARNING "GSRunner is only supported on Windows and may not build on your system")
	endif()
	add_subdirectory(pcsx2-gsrunner)
endif()

#-------------------------------------------------------------------------------
if(NOT IS_SUPPORTED_COMPILER)
	message(WARNING "
*************** UNSUPPORTED CONFIGURATION ***************
You are not compiling PCSX2 with a supported compiler.
It may not even build successfully.
PCSX2 only supports the Clang and MSVC compilers.
No support will be provided, continue at your own risk.
*********************************************************")
endif()

#if(_M_ARM64)
#	message(WARNING "
#*************** UNSUPPORTED CONFIGURATION ***************
#
#Apple Silicon support in PCSX2 is INCOMPLETE. There are
#currently no EE/VU/IOP recompilers, and games will run
#VERY slow. There is no date for completion yet, you
#should set -DCMAKE_OSX_ARCHITECTURES=x86_64 for now,
#unless you want to work on the recompilers.
#
#We also ask that you read https://dont-ship.it/.
#
#*********************************************************")
#endif()
