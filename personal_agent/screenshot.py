"""On-demand Windows desktop screenshot capture using Win32 GDI."""
from __future__ import annotations

import ctypes
import os
import struct
import sys
import time
import zlib
from ctypes import wintypes
from pathlib import Path

from .security import owned_path


SRCCOPY = 0x00CC0020
CAPTUREBLT = 0x40000000
DIB_RGB_COLORS = 0
BI_RGB = 0
SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79


class _BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


class _RGBQUAD(ctypes.Structure):
    _fields_ = [
        ("rgbBlue", wintypes.BYTE),
        ("rgbGreen", wintypes.BYTE),
        ("rgbRed", wintypes.BYTE),
        ("rgbReserved", wintypes.BYTE),
    ]


class _BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", _BITMAPINFOHEADER), ("bmiColors", _RGBQUAD * 1)]


def _win_error(label: str) -> OSError:
    code = ctypes.get_last_error()
    detail = ctypes.FormatError(code).strip() if code else "desktop capture unavailable"
    return OSError(f"{label} failed: {detail}")


def _capture_desktop_bgrx() -> tuple[int, int, bytes]:
    if sys.platform != "win32":
        raise OSError("Windows desktop capture is unavailable on this platform")
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    set_dpi = getattr(user32, "SetThreadDpiAwarenessContext", None)
    previous = None
    if set_dpi:
        set_dpi.argtypes = [ctypes.c_void_p]
        set_dpi.restype = ctypes.c_void_p
        previous = set_dpi(ctypes.c_void_p(-4))  # PER_MONITOR_AWARE_V2, thread-local only.
    try:
        return _capture_desktop_bgrx_native()
    finally:
        if previous:
            set_dpi(previous)


def _capture_desktop_bgrx_native() -> tuple[int, int, bytes]:
    if sys.platform != "win32":
        raise OSError("Windows GDI screenshot capture is only available on Windows")

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)

    user32.GetDC.argtypes = [wintypes.HWND]
    user32.GetDC.restype = wintypes.HDC
    user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
    user32.ReleaseDC.restype = ctypes.c_int
    user32.GetSystemMetrics.argtypes = [ctypes.c_int]
    user32.GetSystemMetrics.restype = ctypes.c_int

    gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
    gdi32.CreateCompatibleDC.restype = wintypes.HDC
    gdi32.DeleteDC.argtypes = [wintypes.HDC]
    gdi32.DeleteDC.restype = wintypes.BOOL
    gdi32.CreateCompatibleBitmap.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int]
    gdi32.CreateCompatibleBitmap.restype = wintypes.HBITMAP
    gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
    gdi32.SelectObject.restype = wintypes.HGDIOBJ
    gdi32.BitBlt.argtypes = [
        wintypes.HDC,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        wintypes.HDC,
        ctypes.c_int,
        ctypes.c_int,
        wintypes.DWORD,
    ]
    gdi32.BitBlt.restype = wintypes.BOOL
    gdi32.GetDIBits.argtypes = [
        wintypes.HDC,
        wintypes.HBITMAP,
        wintypes.UINT,
        wintypes.UINT,
        wintypes.LPVOID,
        ctypes.POINTER(_BITMAPINFO),
        wintypes.UINT,
    ]
    gdi32.GetDIBits.restype = ctypes.c_int
    gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
    gdi32.DeleteObject.restype = wintypes.BOOL

    x = user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
    y = user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
    width = user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)
    height = user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)
    if width <= 0 or height <= 0:
        raise OSError("Desktop capture unavailable (desktop may be locked or disconnected)")
    if width * height > 16_777_216:
        raise OSError("Desktop exceeds the 16-megapixel capture limit")

    screen_dc = user32.GetDC(None)
    if not screen_dc:
        raise _win_error("GetDC")
    memory_dc = None
    bitmap = None
    old_object = None
    try:
        memory_dc = gdi32.CreateCompatibleDC(screen_dc)
        if not memory_dc:
            raise _win_error("CreateCompatibleDC")
        bitmap = gdi32.CreateCompatibleBitmap(screen_dc, width, height)
        if not bitmap:
            raise _win_error("CreateCompatibleBitmap")
        old_object = gdi32.SelectObject(memory_dc, bitmap)
        if not old_object:
            raise _win_error("SelectObject")
        if not gdi32.BitBlt(memory_dc, 0, 0, width, height, screen_dc, x, y, SRCCOPY | CAPTUREBLT):
            raise _win_error("BitBlt")

        # Microsoft documents that GetDIBits must not receive a bitmap while it
        # is selected into a device context, so restore the previous object first.
        if not gdi32.SelectObject(memory_dc, old_object):
            raise _win_error("SelectObject restore")
        old_object = None

        info = _BITMAPINFO()
        info.bmiHeader.biSize = ctypes.sizeof(_BITMAPINFOHEADER)
        info.bmiHeader.biWidth = width
        info.bmiHeader.biHeight = -height  # top-down DIB
        info.bmiHeader.biPlanes = 1
        info.bmiHeader.biBitCount = 32
        info.bmiHeader.biCompression = BI_RGB
        buffer = (ctypes.c_ubyte * (width * height * 4))()
        lines = gdi32.GetDIBits(
            screen_dc,
            bitmap,
            0,
            height,
            ctypes.byref(buffer),
            ctypes.byref(info),
            DIB_RGB_COLORS,
        )
        if lines != height:
            raise _win_error("GetDIBits")
        return width, height, bytes(buffer)
    finally:
        if old_object and memory_dc:
            gdi32.SelectObject(memory_dc, old_object)
        if bitmap:
            gdi32.DeleteObject(bitmap)
        if memory_dc:
            gdi32.DeleteDC(memory_dc)
        user32.ReleaseDC(None, screen_dc)


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)


def _encode_png(width: int, height: int, bgrx: bytes) -> bytes:
    if width <= 0 or height <= 0 or len(bgrx) != width * height * 4:
        raise ValueError("Invalid screenshot pixel buffer")
    rgb = bytearray(width * height * 3)
    rgb[0::3], rgb[1::3], rgb[2::3] = bgrx[2::4], bgrx[1::4], bgrx[0::4]
    rows = bytearray()
    stride = width * 3
    for row_index in range(height):
        rows.append(0)  # PNG filter type 0
        rows.extend(rgb[row_index * stride : (row_index + 1) * stride])
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", header) + _png_chunk(b"IDAT", zlib.compress(bytes(rows), 6)) + _png_chunk(b"IEND", b"")


def capture_screenshot(root: Path) -> Path:
    """Capture the desktop once and save the PNG under root/memory/screenshots."""
    root = Path(root)
    output_dir = owned_path(root, "memory", "screenshots")
    output_dir.mkdir(parents=True, exist_ok=True)
    width, height, pixels = _capture_desktop_bgrx()
    name = f"screenshot-{time.time_ns()}.png"
    target = owned_path(root, "memory", "screenshots", name)
    target.write_bytes(_encode_png(width, height, pixels))
    return target


def screenshot_health() -> dict[str, object]:
    return {
        "available": sys.platform == "win32",
        "platform": sys.platform,
        "method": "win32_gdi",
        "capture_mode": "explicit_request_once",
    }
