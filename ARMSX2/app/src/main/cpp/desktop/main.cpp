// SPDX-FileCopyrightText: 2002-2026 PCSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0+

// Desktop Linux host layer for ARMSX2.
// Provides SDL3 window, VM lifecycle, and Host interface stubs.

#include "PrecompiledHeader.h"

#include <cstring>
#include <deque>
#include <functional>
#include <mutex>
#include <string>

#include "SDL3/SDL.h"

#include "common/FileSystem.h"
#include "common/Path.h"
#include "common/StringUtil.h"

#include "pcsx2/Achievements.h"
#include "pcsx2/GS.h"
#include "pcsx2/GSDumpReplayer.h"
#include "pcsx2/Host.h"
#include "pcsx2/ImGui/FullscreenUI.h"
#include "pcsx2/ImGui/ImGuiFullscreen.h"
#include "pcsx2/ImGui/ImGuiManager.h"
#include "pcsx2/Input/InputManager.h"
#include "pcsx2/MTGS.h"

#include "pcsx2/INISettingsInterface.h"
#include "pcsx2/SIO/Pad/Pad.h"
#include "pcsx2/VMManager.h"
#include "pcsx2/GS/GSPerfMon.h"
#include "pcsx2/PerformanceMetrics.h"

static SDL_Window* s_window = nullptr;
static int s_window_width = 640;
static int s_window_height = 448;
static bool s_running = true;
static std::unique_ptr<INISettingsInterface> s_settings_interface;

// RunOnCPUThread queue
static std::mutex s_cpu_thread_mutex;
static std::deque<std::function<void()>> s_cpu_thread_queue;

// ---------------------------------------------------------------------------
// Host interface implementation
// ---------------------------------------------------------------------------

std::optional<WindowInfo> Host::AcquireRenderWindow(bool recreate_window)
{
	if (!s_window)
	{
		WindowInfo wi = {};
		wi.type = WindowInfo::Type::Surfaceless;
		wi.surface_width = 640;
		wi.surface_height = 448;
		wi.surface_scale = 1.0f;
		return wi;
	}

	WindowInfo wi = {};
	wi.surface_width = s_window_width;
	wi.surface_height = s_window_height;
	wi.surface_scale = 1.0f;

	SDL_PropertiesID props = SDL_GetWindowProperties(s_window);
	void* x11_display = SDL_GetPointerProperty(props, "SDL.window.x11.display", nullptr);
	if (x11_display)
	{
		wi.type = WindowInfo::Type::X11;
		wi.display_connection = x11_display;
		wi.window_handle = (void*)(uintptr_t)SDL_GetNumberProperty(props, "SDL.window.x11.window", 0);
	}
	else
	{
		wi.type = WindowInfo::Type::Wayland;
		wi.display_connection = SDL_GetPointerProperty(props, "SDL.window.wayland.display", nullptr);
		wi.window_handle = SDL_GetPointerProperty(props, "SDL.window.wayland.surface", nullptr);
	}

	return wi;
}

void Host::ReleaseRenderWindow()
{
}

void Host::PumpMessagesOnCPUThread()
{
	// Drain RunOnCPUThread queue
	std::deque<std::function<void()>> queue;
	{
		std::lock_guard lock(s_cpu_thread_mutex);
		queue.swap(s_cpu_thread_queue);
	}
	for (auto& fn : queue)
		fn();

	// Drain OS events into SDL's queue ourselves; SDL_PollEvent would do this
	// implicitly, but we use SDL_PeepEvents below to leave gamepad/joystick
	// events in the queue for SDLInputSource::PollEvents (called from
	// VMManager::Internal::PollInputOnCPUThread immediately after this).
	SDL_PumpEvents();

	SDL_Event ev;
	while (SDL_PeepEvents(&ev, 1, SDL_GETEVENT, SDL_EVENT_QUIT, SDL_EVENT_QUIT) > 0)
	{
		VMManager::SetState(VMState::Stopping);
		s_running = false;
	}
	while (SDL_PeepEvents(&ev, 1, SDL_GETEVENT, SDL_EVENT_KEY_DOWN, SDL_EVENT_KEY_UP) > 0)
	{
		if (ev.key.repeat)
			continue;
		const auto key = InputManager::MakeHostKeyboardKey(static_cast<u32>(ev.key.scancode));
		const float value = (ev.type == SDL_EVENT_KEY_DOWN) ? 1.0f : 0.0f;
		InputManager::InvokeEvents(key, value);
		// Always forward releases to ImGui - an input hook may have
		// eaten the event, leaving ImGui with a stuck key.
		if (ev.type == SDL_EVENT_KEY_UP)
			ImGuiManager::ProcessHostKeyEvent(key, 0.0f);
	}
	while (SDL_PeepEvents(&ev, 1, SDL_GETEVENT, SDL_EVENT_WINDOW_FIRST, SDL_EVENT_WINDOW_LAST) > 0)
	{
		if (ev.type == SDL_EVENT_WINDOW_RESIZED)
		{
			s_window_width = ev.window.data1;
			s_window_height = ev.window.data2;
			if (MTGS::IsOpen())
				MTGS::UpdateDisplayWindow();
		}
	}
}

void Host::RunOnCPUThread(std::function<void()> function, bool block)
{
	if (block)
	{
		std::mutex mtx;
		std::condition_variable cv;
		bool done = false;
		{
			std::lock_guard lock(s_cpu_thread_mutex);
			s_cpu_thread_queue.push_back([&]() {
				function();
				std::lock_guard lk(mtx);
				done = true;
				cv.notify_one();
			});
		}
		std::unique_lock lk(mtx);
		cv.wait(lk, [&] { return done; });
	}
	else
	{
		std::lock_guard lock(s_cpu_thread_mutex);
		s_cpu_thread_queue.push_back(std::move(function));
	}
}

void Host::RequestVMShutdown(bool allow_confirm, bool allow_save_state, bool default_save_state)
{
	VMManager::SetState(VMState::Stopping);
}

// --- BeginPresentFrame (GS stats, same as native-lib.cpp) ---

static s32 s_loop_count = 1;
static u32 s_dump_frame_number = 0;
static u32 s_loop_number = s_loop_count;
static double s_last_internal_draws = 0;
static double s_last_draws = 0;
static double s_last_render_passes = 0;
static double s_last_barriers = 0;
static double s_last_copies = 0;
static double s_last_uploads = 0;
static double s_last_readbacks = 0;
static u64 s_total_internal_draws = 0;
static u64 s_total_draws = 0;
static u64 s_total_render_passes = 0;
static u64 s_total_barriers = 0;
static u64 s_total_copies = 0;
static u64 s_total_uploads = 0;
static u64 s_total_readbacks = 0;
static u32 s_total_frames = 0;
static u32 s_total_drawn_frames = 0;

void Host::BeginPresentFrame()
{
	if (GSIsHardwareRenderer())
	{
		const u32 last_draws = s_total_internal_draws;
		const u32 last_uploads = s_total_uploads;

		static constexpr auto update_stat = [](GSPerfMon::counter_t counter, u64& dst, double& last) {
			const double val = g_perfmon.GetCounter(counter);
			dst += static_cast<u64>((val < last) ? val : (val - last));
			last = val;
		};

		update_stat(GSPerfMon::Draw, s_total_internal_draws, s_last_internal_draws);
		update_stat(GSPerfMon::DrawCalls, s_total_draws, s_last_draws);
		update_stat(GSPerfMon::RenderPasses, s_total_render_passes, s_last_render_passes);
		update_stat(GSPerfMon::Barriers, s_total_barriers, s_last_barriers);
		update_stat(GSPerfMon::TextureCopies, s_total_copies, s_last_copies);
		update_stat(GSPerfMon::TextureUploads, s_total_uploads, s_last_uploads);
		update_stat(GSPerfMon::Readbacks, s_total_readbacks, s_last_readbacks);

		const bool idle_frame = s_total_frames && (last_draws == s_total_internal_draws && last_uploads == s_total_uploads);

		if (!idle_frame)
			s_total_drawn_frames++;

		s_total_frames++;

		std::atomic_thread_fence(std::memory_order_release);
	}
}

// --- Trivial stubs (copied from native-lib.cpp) ---

void Host::OnGameChanged(const std::string& title, const std::string& elf_override, const std::string& disc_path,
	const std::string& disc_serial, u32 disc_crc, u32 current_crc)
{
}

void Host::CommitBaseSettingChanges()
{
	if (s_settings_interface)
		s_settings_interface->Save();
}

void Host::LoadSettings(SettingsInterface& si, std::unique_lock<std::mutex>& lock)
{
}

void Host::CheckForSettingsChanges(const Pcsx2Config& old_config)
{
}

bool Host::RequestResetSettings(bool folders, bool core, bool controllers, bool hotkeys, bool ui)
{
	return false;
}

void Host::SetDefaultUISettings(SettingsInterface& si)
{
}

std::unique_ptr<ProgressCallback> Host::CreateHostProgressCallback()
{
	return nullptr;
}

void Host::ReportErrorAsync(const std::string_view title, const std::string_view message)
{
	if (!title.empty() && !message.empty())
		ERROR_LOG("ReportErrorAsync: {}: {}", title, message);
	else if (!message.empty())
		ERROR_LOG("ReportErrorAsync: {}", message);
}

void Host::OpenURL(const std::string_view url)
{
}

bool Host::CopyTextToClipboard(const std::string_view text)
{
	return false;
}

std::string Host::GetTextFromClipboard()
{
	return {};
}

void Host::BeginTextInput()
{
}

void Host::EndTextInput()
{
}

std::optional<WindowInfo> Host::GetTopLevelWindowInfo()
{
	return std::nullopt;
}

void Host::OnInputDeviceConnected(const std::string_view identifier, const std::string_view device_name)
{
}

void Host::OnInputDeviceDisconnected(const InputBindingKey key, const std::string_view identifier)
{
}

void Host::SetMouseMode(bool relative_mode, bool hide_cursor)
{
}

void Host::RequestResizeHostDisplay(s32 width, s32 height)
{
}

void Host::OnVMStarting()
{
}

void Host::OnVMStarted()
{
}

void Host::OnVMDestroyed()
{
}

void Host::OnVMPaused()
{
}

void Host::OnVMResumed()
{
}

void Host::OnPerformanceMetricsUpdated()
{
}

void Host::OnSaveStateLoading(const std::string_view filename)
{
}

void Host::OnSaveStateLoaded(const std::string_view filename, bool was_successful)
{
}

void Host::OnSaveStateSaved(const std::string_view filename)
{
}

void Host::RefreshGameListAsync(bool invalidate_cache)
{
}

void Host::CancelGameListRefresh()
{
}

bool Host::IsFullscreen()
{
	return s_window && (SDL_GetWindowFlags(s_window) & SDL_WINDOW_FULLSCREEN);
}

void Host::SetFullscreen(bool enabled)
{
	if (s_window)
		SDL_SetWindowFullscreen(s_window, enabled);
	if (s_settings_interface)
	{
		s_settings_interface->SetBoolValue("UI", "Fullscreen", enabled);
		s_settings_interface->Save();
	}
}

void Host::OnCaptureStarted(const std::string& filename)
{
}

void Host::OnCaptureStopped()
{
}

void Host::RequestExitApplication(bool allow_confirm)
{
	s_running = false;
}

void Host::RequestExitBigPicture()
{
}

void Host::OnAchievementsLoginSuccess(const char* username, u32 points, u32 sc_points, u32 unread_messages)
{
}

void Host::OnAchievementsLoginRequested(Achievements::LoginRequestReason reason)
{
}

void Host::OnAchievementsHardcoreModeChanged(bool enabled)
{
}

void Host::OnAchievementsRefreshed()
{
}

void Host::OnCoverDownloaderOpenRequested()
{
}

void Host::OnCreateMemoryCardOpenRequested()
{
}

bool Host::ShouldPreferHostFileSelector()
{
	return false;
}

void Host::OpenHostFileSelectorAsync(std::string_view title, bool select_directory, FileSelectorCallback callback,
	FileSelectorFilters filters, std::string_view initial_directory)
{
	callback(std::string());
}

void Host::ReportInfoAsync(const std::string_view title, const std::string_view message)
{
}

bool Host::LocaleCircleConfirm()
{
	return false;
}

bool Host::InNoGUIMode()
{
	return false;
}

int Host::LocaleSensitiveCompare(std::string_view lhs, std::string_view rhs)
{
	return lhs.compare(rhs);
}

// --- Stubs from AndroidStubs.cpp ---

BEGIN_HOTKEY_LIST(g_host_hotkeys)
END_HOTKEY_LIST()

void Host::SetMouseLock(bool state)
{
}

// --- InputManager: SDL scancode <-> name mapping ---

struct SDLKeyName { SDL_Scancode sc; const char* name; };
static constexpr SDLKeyName s_sdl_key_names[] = {
	{SDL_SCANCODE_A, "A"}, {SDL_SCANCODE_B, "B"}, {SDL_SCANCODE_C, "C"}, {SDL_SCANCODE_D, "D"},
	{SDL_SCANCODE_E, "E"}, {SDL_SCANCODE_F, "F"}, {SDL_SCANCODE_G, "G"}, {SDL_SCANCODE_H, "H"},
	{SDL_SCANCODE_I, "I"}, {SDL_SCANCODE_J, "J"}, {SDL_SCANCODE_K, "K"}, {SDL_SCANCODE_L, "L"},
	{SDL_SCANCODE_M, "M"}, {SDL_SCANCODE_N, "N"}, {SDL_SCANCODE_O, "O"}, {SDL_SCANCODE_P, "P"},
	{SDL_SCANCODE_Q, "Q"}, {SDL_SCANCODE_R, "R"}, {SDL_SCANCODE_S, "S"}, {SDL_SCANCODE_T, "T"},
	{SDL_SCANCODE_U, "U"}, {SDL_SCANCODE_V, "V"}, {SDL_SCANCODE_W, "W"}, {SDL_SCANCODE_X, "X"},
	{SDL_SCANCODE_Y, "Y"}, {SDL_SCANCODE_Z, "Z"},
	{SDL_SCANCODE_1, "1"}, {SDL_SCANCODE_2, "2"}, {SDL_SCANCODE_3, "3"}, {SDL_SCANCODE_4, "4"},
	{SDL_SCANCODE_5, "5"}, {SDL_SCANCODE_6, "6"}, {SDL_SCANCODE_7, "7"}, {SDL_SCANCODE_8, "8"},
	{SDL_SCANCODE_9, "9"}, {SDL_SCANCODE_0, "0"},
	{SDL_SCANCODE_F1, "F1"}, {SDL_SCANCODE_F2, "F2"}, {SDL_SCANCODE_F3, "F3"}, {SDL_SCANCODE_F4, "F4"},
	{SDL_SCANCODE_F5, "F5"}, {SDL_SCANCODE_F6, "F6"}, {SDL_SCANCODE_F7, "F7"}, {SDL_SCANCODE_F8, "F8"},
	{SDL_SCANCODE_F9, "F9"}, {SDL_SCANCODE_F10, "F10"}, {SDL_SCANCODE_F11, "F11"}, {SDL_SCANCODE_F12, "F12"},
	{SDL_SCANCODE_RETURN, "Return"}, {SDL_SCANCODE_ESCAPE, "Escape"},
	{SDL_SCANCODE_BACKSPACE, "Backspace"}, {SDL_SCANCODE_TAB, "Tab"}, {SDL_SCANCODE_TAB, "Backtab"},
	{SDL_SCANCODE_SPACE, "Space"},
	{SDL_SCANCODE_LEFT, "Left"}, {SDL_SCANCODE_RIGHT, "Right"},
	{SDL_SCANCODE_UP, "Up"}, {SDL_SCANCODE_DOWN, "Down"},
	{SDL_SCANCODE_HOME, "Home"}, {SDL_SCANCODE_END, "End"},
	{SDL_SCANCODE_PAGEUP, "PageUp"}, {SDL_SCANCODE_PAGEDOWN, "PageDown"},
	{SDL_SCANCODE_INSERT, "Insert"}, {SDL_SCANCODE_DELETE, "Delete"},
	{SDL_SCANCODE_LCTRL, "LeftCtrl"}, {SDL_SCANCODE_LCTRL, "Ctrl"}, {SDL_SCANCODE_LCTRL, "Control"},
	{SDL_SCANCODE_RCTRL, "RightCtrl"},
	{SDL_SCANCODE_LSHIFT, "LeftShift"}, {SDL_SCANCODE_LSHIFT, "Shift"},
	{SDL_SCANCODE_RSHIFT, "RightShift"},
	{SDL_SCANCODE_LALT, "LeftAlt"}, {SDL_SCANCODE_LALT, "Alt"},
	{SDL_SCANCODE_RALT, "RightAlt"},
	{SDL_SCANCODE_LGUI, "LeftSuper"}, {SDL_SCANCODE_LGUI, "Super"},
	{SDL_SCANCODE_RGUI, "RightSuper"},
	{SDL_SCANCODE_MENU, "Menu"},
	{SDL_SCANCODE_APOSTROPHE, "Apostrophe"}, {SDL_SCANCODE_COMMA, "Comma"},
	{SDL_SCANCODE_MINUS, "Minus"}, {SDL_SCANCODE_PERIOD, "Period"},
	{SDL_SCANCODE_SLASH, "Slash"}, {SDL_SCANCODE_SEMICOLON, "Semicolon"},
	{SDL_SCANCODE_EQUALS, "Equal"}, {SDL_SCANCODE_LEFTBRACKET, "BracketLeft"},
	{SDL_SCANCODE_BACKSLASH, "Backslash"}, {SDL_SCANCODE_RIGHTBRACKET, "BracketRight"},
	{SDL_SCANCODE_GRAVE, "QuoteLeft"},
	{SDL_SCANCODE_CAPSLOCK, "CapsLock"}, {SDL_SCANCODE_SCROLLLOCK, "ScrollLock"},
	{SDL_SCANCODE_NUMLOCKCLEAR, "NumLock"},
	{SDL_SCANCODE_PRINTSCREEN, "PrintScreen"}, {SDL_SCANCODE_PAUSE, "Pause"},
	{SDL_SCANCODE_KP_0, "Keypad0"}, {SDL_SCANCODE_KP_1, "Keypad1"},
	{SDL_SCANCODE_KP_2, "Keypad2"}, {SDL_SCANCODE_KP_3, "Keypad3"},
	{SDL_SCANCODE_KP_4, "Keypad4"}, {SDL_SCANCODE_KP_5, "Keypad5"},
	{SDL_SCANCODE_KP_6, "Keypad6"}, {SDL_SCANCODE_KP_7, "Keypad7"},
	{SDL_SCANCODE_KP_8, "Keypad8"}, {SDL_SCANCODE_KP_9, "Keypad9"},
	{SDL_SCANCODE_KP_PERIOD, "KeypadPeriod"}, {SDL_SCANCODE_KP_DIVIDE, "KeypadDivide"},
	{SDL_SCANCODE_KP_MULTIPLY, "KeypadMultiply"}, {SDL_SCANCODE_KP_MINUS, "KeypadMinus"},
	{SDL_SCANCODE_KP_PLUS, "KeypadPlus"}, {SDL_SCANCODE_KP_ENTER, "KeypadReturn"},
	{SDL_SCANCODE_KP_EQUALS, "KeypadEqual"},
};

std::optional<u32> InputManager::ConvertHostKeyboardStringToCode(const std::string_view str)
{
	for (const auto& kn : s_sdl_key_names)
	{
		if (str == kn.name)
			return static_cast<u32>(kn.sc);
	}
	return std::nullopt;
}

std::optional<std::string> InputManager::ConvertHostKeyboardCodeToString(u32 code)
{
	for (const auto& kn : s_sdl_key_names)
	{
		if (static_cast<u32>(kn.sc) == code)
			return std::string(kn.name);
	}
	return std::nullopt;
}

const char* InputManager::ConvertHostKeyboardCodeToIcon(u32 code)
{
	return nullptr;
}

// --- Translation stubs ---

s32 Host::Internal::GetTranslatedStringImpl(
	const std::string_view context, const std::string_view msg, char* tbuf, size_t tbuf_space)
{
	if (msg.size() > tbuf_space)
		return -1;
	else if (msg.empty())
		return 0;

	std::memcpy(tbuf, msg.data(), msg.size());
	return static_cast<s32>(msg.size());
}

std::string Host::TranslatePluralToString(const char* context, const char* msg, const char* disambiguation, int count)
{
	TinyString count_str = TinyString::from_format("{}", count);

	std::string ret(msg);
	for (;;)
	{
		std::string::size_type pos = ret.find("%n");
		if (pos == std::string::npos)
			break;

		ret.replace(pos, pos + 2, count_str.view());
	}

	return ret;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

static void print_usage(const char* argv0)
{
	std::fprintf(stderr,
		"Usage: %s [options] [iso-path]\n"
		"  --app-root DIR   Assets directory\n"
		"  --data-root DIR  Config/data directory (also used as cfgpath)\n"
		"  --elf FILE       Boot an ELF directly (no disc required)\n"
		"  --logfile FILE   Force log output to FILE\n"
		"  --headless       No window (for automated testing)\n"
		"  --interp         Force all interpreters (EE/IOP/VU)\n",
		argv0);
}

int main(int argc, char* argv[])
{
	std::string iso_path;
	std::string app_root;
	std::string data_root;
	std::string elf_path;
	std::string logfile;
	bool headless = false;
	bool use_interp = false;

	// Parse arguments
	for (int i = 1; i < argc; i++)
	{
		if (std::strcmp(argv[i], "--app-root") == 0 && i + 1 < argc)
			app_root = argv[++i];
		else if (std::strcmp(argv[i], "--data-root") == 0 && i + 1 < argc)
			data_root = argv[++i];
		else if (std::strcmp(argv[i], "--elf") == 0 && i + 1 < argc)
			elf_path = argv[++i];
		else if (std::strcmp(argv[i], "--logfile") == 0 && i + 1 < argc)
			logfile = argv[++i];
		else if (std::strcmp(argv[i], "--headless") == 0)
			headless = true;
		else if (std::strcmp(argv[i], "--interp") == 0)
			use_interp = true;
		else if (argv[i][0] != '-')
			iso_path = argv[i];
		else
		{
			print_usage(argv[0]);
			return 1;
		}
	}

	if (iso_path.empty() && elf_path.empty())
	{
		print_usage(argv[0]);
		return 1;
	}

	// Default AppRoot: <source-tree>/app/src/main/assets
	// CMAKE_SOURCE_DIR is baked in at compile time.
	if (app_root.empty())
	{
#ifdef ARMSX2_SOURCE_DIR
		app_root = Path::Combine(ARMSX2_SOURCE_DIR, "../assets");
#else
		app_root = "../assets";
#endif
		if (!FileSystem::DirectoryExists(app_root.c_str()))
		{
			std::fprintf(stderr, "Cannot find assets dir at %s\n", app_root.c_str());
			std::fprintf(stderr, "Use --app-root to specify the assets directory.\n");
			return 1;
		}
	}

	// Default DataRoot: ~/.config/armsx2
	if (data_root.empty())
	{
		const char* xdg = std::getenv("XDG_CONFIG_HOME");
		if (xdg && xdg[0])
			data_root = Path::Combine(xdg, "armsx2");
		else
		{
			const char* home = std::getenv("HOME");
			if (home)
				data_root = Path::Combine(Path::Combine(home, ".config"), "armsx2");
			else
				data_root = "/tmp/armsx2";
		}
	}
	FileSystem::EnsureDirectoryExists(data_root.c_str(), false);

	EmuFolders::AppRoot = app_root;
	EmuFolders::DataRoot = data_root;
	EmuFolders::SetResourcesDirectory();

	Log::SetConsoleOutputLevel(LOGLEVEL_DEBUG);

	// Settings - backed by INI file in DataRoot
	std::string settings_path = Path::Combine(data_root, "ARMSX2.ini");
	s_settings_interface = std::make_unique<INISettingsInterface>(std::move(settings_path));
	Host::Internal::SetBaseSettingsLayer(s_settings_interface.get());

	if (!FileSystem::FileExists(s_settings_interface->GetFileName().c_str()))
	{
		// First run - write defaults
		INISettingsInterface& si = *s_settings_interface;
		VMManager::SetDefaultSettings(si, true, true, true, true, true);

		si.SetBoolValue("EmuCore/GS", "FrameLimitEnable", false);
		si.SetIntValue("EmuCore/GS", "VsyncEnable", false);
		si.SetBoolValue("EmuCore", "EnableThreadPinning", true);
		si.SetBoolValue("EmuCore/CPU/Recompiler", "EnableFastmem", true);

		si.SetBoolValue("InputSources", "SDL", true);
		si.SetBoolValue("InputSources", "XInput", false);

		si.SetStringValue("SPU2/Output", "Backend", "SDL");
		si.SetBoolValue("EmuCore", "EnableFastBoot", false);
		si.SetIntValue("EmuCore/GS", "Renderer", static_cast<int>(GSRendererType::SW));

		Pad::ClearPortBindings(si, 0);

		si.SetBoolValue("Logging", "EnableSystemConsole", true);
		si.SetBoolValue("Logging", "EnableTimestamps", true);
		si.SetBoolValue("Logging", "EnableVerbose", true);

		si.SetBoolValue("EmuCore/GS", "OsdShowFPS", true);
		si.SetBoolValue("EmuCore/GS", "OsdShowSpeed", true);
		si.SetBoolValue("EmuCore/GS", "OsdShowResolution", true);
		si.SetBoolValue("EmuCore/GS", "OsdShowCPU", true);
		si.SetBoolValue("EmuCore/GS", "OsdShowGPU", true);
		si.SetBoolValue("EmuCore/GS", "OsdShowGSStats", true);
		si.SetBoolValue("EmuCore/GS", "OsdShowFrameTimes", true);
		si.SetBoolValue("EmuCore/GS", "OsdShowHardwareInfo", true);
		si.SetBoolValue("EmuCore/GS", "OsdShowVersion", true);
		si.SetBoolValue("EmuCore/GS", "OsdShowSettings", true);
		si.SetBoolValue("EmuCore/GS", "OsdShowInputs", true);

		si.Save();
	}
	else
	{
		s_settings_interface->Load();
	}

	if (!logfile.empty())
		VMManager::Internal::SetFileLogPath(logfile);

	if (headless || !elf_path.empty())
	{
		// Enable EE/IOP console output for test harnesses
		s_settings_interface->SetBoolValue("Logging", "EnableEEConsole", true);
		s_settings_interface->SetBoolValue("Logging", "EnableIOPConsole", true);
		s_settings_interface->SetBoolValue("Logging", "EnableFileLogging", true);
	}

	if (use_interp)
	{
		s_settings_interface->SetBoolValue("EmuCore/CPU/Recompiler", "EnableEE", false);
		s_settings_interface->SetBoolValue("EmuCore/CPU/Recompiler", "EnableIOP", false);
		s_settings_interface->SetBoolValue("EmuCore/CPU/Recompiler", "EnableVU0", false);
		s_settings_interface->SetBoolValue("EmuCore/CPU/Recompiler", "EnableVU1", false);
	}

	VMManager::Internal::LoadStartupSettings();

	// SDL init
	if (!SDL_Init(headless ? SDL_INIT_AUDIO : (SDL_INIT_VIDEO | SDL_INIT_AUDIO | SDL_INIT_GAMEPAD)))
	{
		std::fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
		return 1;
	}

	if (!headless)
	{
		s_window = SDL_CreateWindow("ARMSX2", s_window_width, s_window_height, SDL_WINDOW_RESIZABLE);
		if (!s_window)
		{
			std::fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
			SDL_Quit();
			return 1;
		}

		if (s_settings_interface->GetBoolValue("UI", "Fullscreen", false))
			SDL_SetWindowFullscreen(s_window, true);
	}

	// CPU thread init
	if (!VMManager::Internal::CPUThreadInitialize())
	{
		std::fprintf(stderr, "CPUThreadInitialize failed\n");
		VMManager::Internal::CPUThreadShutdown();
		SDL_DestroyWindow(s_window);
		SDL_Quit();
		return 1;
	}

	VMManager::ApplySettings();
	GSDumpReplayer::SetIsDumpRunner(false);

	VMBootParameters boot_params;
	if (!elf_path.empty())
	{
		boot_params.filename = elf_path;
		// Must go through BIOS to initialize Deci2 handler for EE console output
		boot_params.fast_boot = false;
	}
	else
	{
		boot_params.filename = iso_path;
		boot_params.fast_boot = !iso_path.empty();
	}

	Console.Error("Loading %s", boot_params.filename.c_str());

	if (VMManager::Initialize(boot_params, nullptr) == VMBootResult::StartupSuccess)
	{
		Console.Error("VM INIT");
		VMManager::SetState(VMState::Running);

		while (s_running)
		{
			VMState state = VMManager::GetState();
			if (state == VMState::Stopping || state == VMState::Shutdown)
				break;
			else if (state == VMState::Running)
				VMManager::Execute();
			else if (state == VMState::Paused)
			{
				// IdlePollUpdate drains the SDL gamepad queue (via
				// InputManager::PollSources) which is otherwise only drained
				// from Counters' EE vsync handler. Without this, FullscreenUI
				// receives keyboard nav (drained by PumpMessages) but no
				// gamepad nav while PauseOnMenu is in effect.
				VMManager::IdlePollUpdate();
				Host::PumpMessagesOnCPUThread();
				usleep(16000); // ~60fps pump while paused
			}
			else
				usleep(250000);
		}

		VMManager::Shutdown(false);
	}
	else
	{
		std::fprintf(stderr, "VMManager::Initialize failed\n");
	}

	VMManager::Internal::CPUThreadShutdown();

	if (s_window)
	{
		SDL_DestroyWindow(s_window);
		s_window = nullptr;
	}
	SDL_Quit();

	return 0;
}
