package com.armsx2.input

import android.view.InputDevice

/**
 * Auto-assigns physical controllers to PS2 ports for local co-op (2 players).
 *
 *  - First distinct gamepad to send in-game input this session = Player 1 (port 0).
 *  - Next distinct gamepad = Player 2 (port 1); enabling Pad2 natively (once) the
 *    instant it joins, via [onPlayer2Joined].
 *  - Any further devices fold into Player 1 (no multitap in this build).
 *
 * The on-screen touch controls and all menu navigation always use port 0 — they
 * never go through here. Reset on VM start AND stop ([reset]) so each session
 * re-pairs deterministically (avoids P1/P2 swapping and the builtin-vs-external
 * pad race at boot).
 *
 * Called only from the in-game input dispatch where a real `event.deviceId` is
 * live (see Main.dispatchKeyEvent / dispatchGenericMotionEvent).
 */
object PadRouter {
    @Volatile private var p1DeviceId = -1
    @Volatile private var p2DeviceId = -1
    @Volatile private var pad2Enabled = false

    /** Fired exactly once, the first time a second controller is assigned, so the
     *  app can enable the native Pad2 slot before any P2 input is sent. */
    @Volatile var onPlayer2Joined: (() -> Unit)? = null

    fun reset() {
        p1DeviceId = -1
        p2DeviceId = -1
        pad2Enabled = false
    }

    /** True once a second controller has joined this session (P2 is live). */
    fun coopActive(): Boolean = p2DeviceId != -1

    /** Android InputDevice id assigned to a PS2 port (0 = P1, 1 = P2), or -1 if that
     *  port hasn't been claimed yet. Lets per-port PS2 rumble buzz the right pad. */
    fun deviceIdForPort(port: Int): Int = when (port) {
        0 -> p1DeviceId
        1 -> p2DeviceId
        else -> -1
    }

    /**
     * Map a physical input device to a PS2 port (0 = P1, 1 = P2), auto-assigning
     * the first two distinct gamepads. Synthetic / virtual events (deviceId < 0)
     * never claim a port — they're treated as P1.
     */
    fun portForDevice(deviceId: Int): Int {
        if (deviceId < 0) return 0
        // Fast path: already-claimed nodes (no InputDevice lookup).
        if (deviceId == p1DeviceId) return 0
        if (deviceId == p2DeviceId) return 1
        // Only a real GAMEPAD/JOYSTICK node may claim a player slot. One physical
        // controller (notably a DualSense over Bluetooth) enumerates as SEVERAL
        // InputDevices — a gamepad node PLUS a touchpad/mouse (and motion) node, each
        // with its own deviceId. If a non-gamepad node grabs P2, the pad's real
        // gamepad node becomes the "3rd device" and folds to P1 — so BOTH pads end up
        // driving Player 1. Gate claims to gamepad sources so secondary nodes can't
        // eat a slot. (Keyed by raw deviceId, NOT descriptor, so two IDENTICAL pads
        // still get separate slots.)
        val src = InputDevice.getDevice(deviceId)?.sources ?: 0
        val isPad = (src and InputDevice.SOURCE_GAMEPAD) == InputDevice.SOURCE_GAMEPAD ||
            (src and InputDevice.SOURCE_JOYSTICK) == InputDevice.SOURCE_JOYSTICK
        if (!isPad) return 0 // touchpad / mouse / keyboard node → treat as P1, don't claim
        if (p1DeviceId == -1) { p1DeviceId = deviceId; return 0 }
        if (p2DeviceId == -1) {
            p2DeviceId = deviceId
            if (!pad2Enabled) { pad2Enabled = true; onPlayer2Joined?.invoke() }
            return 1
        }
        return 0 // 3rd+ gamepad, no multitap: fold into Player 1
    }
}
