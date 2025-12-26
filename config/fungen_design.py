"""
FunGen Design System
====================

Professional UI design system for FunGen using Dear ImGui.
Spotify-inspired theme with clean vector icons.

All icons are drawn programmatically using ImGui draw primitives:
- Consistent 2px stroke weight
- Monochromatic (single color)
- Scalable to any size
- Clean geometric shapes

Usage:
    from config.fungen_design import FunGenTheme, Icons, UI, Colors

    # Apply theme at startup
    FunGenTheme.apply()

    # Render icon button
    if UI.icon_button("save", Icons.save, tooltip="Save Project"):
        save_project()
"""

import imgui
import math
from typing import Tuple, Callable, Optional


# ============================================================================
# COLORS - Spotify-Inspired Palette
# ============================================================================

class Colors:
    """Spotify-inspired color palette for FunGen."""

    # Primary accent - Spotify Green
    GREEN = (0.114, 0.725, 0.329, 1.0)          # #1db954
    GREEN_HOVER = (0.141, 0.800, 0.380, 1.0)    # #24cc5f
    GREEN_DARK = (0.090, 0.627, 0.263, 1.0)     # #17a043
    GREEN_DIM = (0.114, 0.725, 0.329, 0.2)      # Green at 20% opacity

    # Backgrounds (darkest to lightest)
    BG_BASE = (0.071, 0.071, 0.071, 1.0)        # #121212 - App background
    BG_SURFACE = (0.094, 0.094, 0.094, 1.0)     # #181818 - Panels
    BG_ELEVATED = (0.157, 0.157, 0.157, 1.0)    # #282828 - Cards, buttons
    BG_HIGHLIGHT = (0.243, 0.243, 0.243, 1.0)   # #3e3e3e - Hover states
    BG_OVERLAY = (0.039, 0.039, 0.039, 1.0)     # #0a0a0a - Video area

    # Text (brightest to dimmest)
    TEXT_PRIMARY = (1.0, 1.0, 1.0, 1.0)         # #ffffff
    TEXT_SECONDARY = (0.702, 0.702, 0.702, 1.0) # #b3b3b3
    TEXT_MUTED = (0.447, 0.447, 0.447, 1.0)     # #727272
    TEXT_DISABLED = (0.325, 0.325, 0.325, 1.0)  # #535353

    # Status colors
    SUCCESS = GREEN
    WARNING = (0.961, 0.651, 0.137, 1.0)        # #f5a623
    ERROR = (0.910, 0.298, 0.235, 1.0)          # #e84c3c
    INFO = (0.204, 0.596, 0.859, 1.0)           # #3498db

    # Utility
    TRANSPARENT = (0.0, 0.0, 0.0, 0.0)
    BORDER = (0.157, 0.157, 0.157, 1.0)         # #282828
    BORDER_LIGHT = (0.243, 0.243, 0.243, 1.0)   # #3e3e3e

    # Legacy color aliases for compatibility
    RED = (0.910, 0.298, 0.235, 1.0)
    BLUE = (0.204, 0.596, 0.859, 1.0)
    YELLOW = (0.961, 0.651, 0.137, 1.0)
    CYAN = (0.102, 0.737, 0.612, 1.0)           # #1abc9c
    ORANGE = (0.902, 0.494, 0.133, 1.0)         # #e67e22

    @staticmethod
    def u32(color: Tuple[float, float, float, float]) -> int:
        """Convert RGBA tuple to ImGui u32 color."""
        return imgui.get_color_u32_rgba(*color)

    @staticmethod
    def alpha(color: Tuple[float, float, float, float], a: float) -> Tuple[float, float, float, float]:
        """Return color with modified alpha."""
        return (color[0], color[1], color[2], a)

    @staticmethod
    def rgb_to_tuple(r: int, g: int, b: int, a: int = 255) -> Tuple[float, float, float, float]:
        """Convert RGB 0-255 values to normalized tuple."""
        return (r / 255.0, g / 255.0, b / 255.0, a / 255.0)

    @staticmethod
    def hex_to_tuple(hex_color: str) -> Tuple[float, float, float, float]:
        """Convert hex color string to normalized tuple."""
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        a = int(hex_color[6:8], 16) / 255.0 if len(hex_color) == 8 else 1.0
        return (r, g, b, a)


# ============================================================================
# VECTOR ICONS
# ============================================================================

class Icons:
    """
    Vector icon drawing functions.

    All functions have signature:
        draw(draw_list, x, y, size, color) -> None

    Icons are designed on a 20x20 grid with 2px stroke weight.
    """

    STROKE = 2.0

    # ------------------------------------------------------------------------
    # MODE TOGGLE
    # ------------------------------------------------------------------------

    @staticmethod
    def simple_mode(dl, x: float, y: float, size: float, color: Tuple):
        """4-pointed star - represents simple/magic mode"""
        c = Colors.u32(color)
        cx, cy = x + size/2, y + size/2
        r_out = size * 0.38
        r_in = size * 0.15

        pts = []
        for i in range(8):
            angle = (i * math.pi / 4) - math.pi/2
            r = r_out if i % 2 == 0 else r_in
            pts.append((cx + math.cos(angle) * r, cy + math.sin(angle) * r))

        for i in range(8):
            dl.add_line(pts[i][0], pts[i][1], pts[(i+1) % 8][0], pts[(i+1) % 8][1], c, Icons.STROKE)

    @staticmethod
    def expert_mode(dl, x: float, y: float, size: float, color: Tuple):
        """Gear/cog - represents settings/expert mode"""
        c = Colors.u32(color)
        cx, cy = x + size/2, y + size/2
        r_inner = size * 0.12
        r_outer = size * 0.38
        teeth = 8

        # Center hole
        dl.add_circle(cx, cy, r_inner, c, 12, Icons.STROKE)

        # Outer ring with teeth
        for i in range(teeth):
            angle = (i * 2 * math.pi / teeth)
            a1 = angle - 0.2
            a2 = angle + 0.2

            # Tooth
            x1, y1 = cx + math.cos(a1) * r_outer * 0.7, cy + math.sin(a1) * r_outer * 0.7
            x2, y2 = cx + math.cos(a1) * r_outer, cy + math.sin(a1) * r_outer
            x3, y3 = cx + math.cos(a2) * r_outer, cy + math.sin(a2) * r_outer
            x4, y4 = cx + math.cos(a2) * r_outer * 0.7, cy + math.sin(a2) * r_outer * 0.7

            dl.add_line(x1, y1, x2, y2, c, Icons.STROKE)
            dl.add_line(x2, y2, x3, y3, c, Icons.STROKE)
            dl.add_line(x3, y3, x4, y4, c, Icons.STROKE)

        # Connect teeth with arc segments
        dl.add_circle(cx, cy, r_outer * 0.7, c, teeth * 4, Icons.STROKE)

    # ------------------------------------------------------------------------
    # FILE OPERATIONS
    # ------------------------------------------------------------------------

    @staticmethod
    def new_file(dl, x: float, y: float, size: float, color: Tuple):
        """Document with plus sign"""
        c = Colors.u32(color)
        m = size * 0.18  # margin

        # Document outline with folded corner
        dl.add_line(x + m, y + m, x + size - m - size*0.15, y + m, c, Icons.STROKE)
        dl.add_line(x + size - m - size*0.15, y + m, x + size - m, y + m + size*0.15, c, Icons.STROKE)
        dl.add_line(x + size - m, y + m + size*0.15, x + size - m, y + size - m, c, Icons.STROKE)
        dl.add_line(x + size - m, y + size - m, x + m, y + size - m, c, Icons.STROKE)
        dl.add_line(x + m, y + size - m, x + m, y + m, c, Icons.STROKE)

        # Fold line
        dl.add_line(x + size - m - size*0.15, y + m, x + size - m - size*0.15, y + m + size*0.15, c, Icons.STROKE)
        dl.add_line(x + size - m - size*0.15, y + m + size*0.15, x + size - m, y + m + size*0.15, c, Icons.STROKE)

        # Plus sign
        cx, cy = x + size/2, y + size/2 + size*0.05
        ps = size * 0.15
        dl.add_line(cx - ps, cy, cx + ps, cy, c, Icons.STROKE)
        dl.add_line(cx, cy - ps, cx, cy + ps, c, Icons.STROKE)

    @staticmethod
    def folder(dl, x: float, y: float, size: float, color: Tuple):
        """Folder icon"""
        c = Colors.u32(color)
        m = size * 0.18

        # Folder body
        dl.add_line(x + m, y + size * 0.35, x + m, y + size - m, c, Icons.STROKE)
        dl.add_line(x + m, y + size - m, x + size - m, y + size - m, c, Icons.STROKE)
        dl.add_line(x + size - m, y + size - m, x + size - m, y + size * 0.35, c, Icons.STROKE)

        # Folder tab
        dl.add_line(x + m, y + size * 0.35, x + m, y + m, c, Icons.STROKE)
        dl.add_line(x + m, y + m, x + size * 0.4, y + m, c, Icons.STROKE)
        dl.add_line(x + size * 0.4, y + m, x + size * 0.5, y + size * 0.35, c, Icons.STROKE)
        dl.add_line(x + size * 0.5, y + size * 0.35, x + size - m, y + size * 0.35, c, Icons.STROKE)

    @staticmethod
    def save(dl, x: float, y: float, size: float, color: Tuple):
        """Floppy disk / save icon"""
        c = Colors.u32(color)
        m = size * 0.18

        # Outer rectangle
        dl.add_rect(x + m, y + m, x + size - m, y + size - m, c, 0, 0, Icons.STROKE)

        # Top slot
        slot_l = x + size * 0.32
        slot_r = x + size * 0.68
        slot_b = y + size * 0.38
        dl.add_line(slot_l, y + m, slot_l, slot_b, c, Icons.STROKE)
        dl.add_line(slot_l, slot_b, slot_r, slot_b, c, Icons.STROKE)
        dl.add_line(slot_r, slot_b, slot_r, y + m, c, Icons.STROKE)

        # Label rectangle at bottom
        lab_t = y + size * 0.52
        lab_l = x + size * 0.28
        lab_r = x + size * 0.72
        dl.add_rect(lab_l, lab_t, lab_r, y + size - m, c, 0, 0, Icons.STROKE)

    @staticmethod
    def export(dl, x: float, y: float, size: float, color: Tuple):
        """Upload/export arrow"""
        c = Colors.u32(color)
        cx = x + size/2

        # Vertical arrow shaft
        dl.add_line(cx, y + size * 0.22, cx, y + size * 0.62, c, Icons.STROKE)

        # Arrow head
        aw = size * 0.18
        dl.add_line(cx, y + size * 0.22, cx - aw, y + size * 0.38, c, Icons.STROKE)
        dl.add_line(cx, y + size * 0.22, cx + aw, y + size * 0.38, c, Icons.STROKE)

        # Base tray
        tray_y = y + size * 0.78
        dl.add_line(x + size * 0.22, tray_y, x + size * 0.22, y + size * 0.62, c, Icons.STROKE)
        dl.add_line(x + size * 0.22, tray_y, x + size * 0.78, tray_y, c, Icons.STROKE)
        dl.add_line(x + size * 0.78, tray_y, x + size * 0.78, y + size * 0.62, c, Icons.STROKE)

    # ------------------------------------------------------------------------
    # TRACKING
    # ------------------------------------------------------------------------

    @staticmethod
    def crosshair(dl, x: float, y: float, size: float, color: Tuple):
        """Crosshair/target for tracking"""
        c = Colors.u32(color)
        cx, cy = x + size/2, y + size/2
        r_out = size * 0.38
        r_in = size * 0.15
        gap = size * 0.06

        # Outer circle
        dl.add_circle(cx, cy, r_out, c, 20, Icons.STROKE)

        # Inner circle
        dl.add_circle(cx, cy, r_in, c, 16, Icons.STROKE)

        # Cross lines (with gap for inner circle)
        dl.add_line(cx, y + size * 0.12, cx, cy - r_in - gap, c, Icons.STROKE)
        dl.add_line(cx, cy + r_in + gap, cx, y + size * 0.88, c, Icons.STROKE)
        dl.add_line(x + size * 0.12, cy, cx - r_in - gap, cy, c, Icons.STROKE)
        dl.add_line(cx + r_in + gap, cy, x + size * 0.88, cy, c, Icons.STROKE)

    @staticmethod
    def simplify(dl, x: float, y: float, size: float, color: Tuple):
        """Wavy line becoming straight - simplification"""
        c = Colors.u32(color)

        # Complex wave on top
        wave_y = y + size * 0.32
        amp = size * 0.08
        pts = []
        for i in range(7):
            px = x + size * 0.15 + (i * size * 0.12)
            py = wave_y + math.sin(i * 1.5) * amp
            pts.append((px, py))
        for i in range(len(pts) - 1):
            dl.add_line(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1], c, Icons.STROKE)

        # Down arrow
        cx = x + size/2
        arr_t = y + size * 0.45
        arr_b = y + size * 0.55
        dl.add_line(cx, arr_t, cx, arr_b, c, Icons.STROKE)
        dl.add_line(cx - size*0.06, arr_b - size*0.05, cx, arr_b, c, Icons.STROKE)
        dl.add_line(cx + size*0.06, arr_b - size*0.05, cx, arr_b, c, Icons.STROKE)

        # Straight line on bottom
        line_y = y + size * 0.68
        dl.add_line(x + size * 0.15, line_y, x + size * 0.85, line_y, c, Icons.STROKE)

    @staticmethod
    def robot(dl, x: float, y: float, size: float, color: Tuple):
        """Robot head - AI/tracking"""
        c = Colors.u32(color)
        m = size * 0.2

        # Head rectangle
        dl.add_rect(x + m, y + m + size * 0.1, x + size - m, y + size - m, c, 2, 0, Icons.STROKE)

        # Antenna
        cx = x + size/2
        dl.add_line(cx, y + m + size * 0.1, cx, y + m - size * 0.05, c, Icons.STROKE)
        dl.add_circle_filled(cx, y + m - size * 0.05, size * 0.05, c)

        # Eyes
        eye_y = y + size * 0.4
        eye_r = size * 0.08
        dl.add_circle_filled(x + size * 0.35, eye_y, eye_r, c)
        dl.add_circle_filled(x + size * 0.65, eye_y, eye_r, c)

        # Mouth
        mouth_y = y + size * 0.65
        dl.add_line(x + size * 0.32, mouth_y, x + size * 0.68, mouth_y, c, Icons.STROKE)

    # ------------------------------------------------------------------------
    # VIDEO
    # ------------------------------------------------------------------------

    @staticmethod
    def monitor(dl, x: float, y: float, size: float, color: Tuple):
        """Monitor/display icon"""
        c = Colors.u32(color)
        m = size * 0.15

        # Screen
        dl.add_rect(x + m, y + m, x + size - m, y + size * 0.68, c, 2, 0, Icons.STROKE)

        # Stand neck
        cx = x + size/2
        dl.add_line(cx, y + size * 0.68, cx, y + size * 0.78, c, Icons.STROKE)

        # Stand base
        dl.add_line(x + size * 0.3, y + size * 0.78, x + size * 0.7, y + size * 0.78, c, Icons.STROKE)

    @staticmethod
    def monitor_off(dl, x: float, y: float, size: float, color: Tuple):
        """Monitor with slash - hidden video"""
        Icons.monitor(dl, x, y, size, color)
        c = Colors.u32(color)
        # Diagonal slash
        dl.add_line(x + size * 0.18, y + size * 0.78, x + size * 0.82, y + size * 0.18, c, Icons.STROKE)

    # ------------------------------------------------------------------------
    # SPEED (Gauge Style)
    # ------------------------------------------------------------------------

    @staticmethod
    def gauge_slow(dl, x: float, y: float, size: float, color: Tuple):
        """Speedometer at low position"""
        c = Colors.u32(color)
        cx, cy = x + size/2, y + size * 0.58
        r = size * 0.35

        # Arc (semi-circle)
        segments = 16
        for i in range(segments):
            a1 = math.pi + (i * math.pi / segments)
            a2 = math.pi + ((i + 1) * math.pi / segments)
            dl.add_line(
                cx + math.cos(a1) * r, cy + math.sin(a1) * r,
                cx + math.cos(a2) * r, cy + math.sin(a2) * r,
                c, Icons.STROKE
            )

        # Needle pointing to slow (left side)
        needle_angle = math.pi * 0.8
        needle_len = size * 0.22
        dl.add_line(cx, cy, cx + math.cos(needle_angle) * needle_len, cy + math.sin(needle_angle) * needle_len, c, Icons.STROKE)
        dl.add_circle_filled(cx, cy, 3, c)

    @staticmethod
    def gauge_normal(dl, x: float, y: float, size: float, color: Tuple):
        """Speedometer at middle position"""
        c = Colors.u32(color)
        cx, cy = x + size/2, y + size * 0.58
        r = size * 0.35

        # Arc
        segments = 16
        for i in range(segments):
            a1 = math.pi + (i * math.pi / segments)
            a2 = math.pi + ((i + 1) * math.pi / segments)
            dl.add_line(
                cx + math.cos(a1) * r, cy + math.sin(a1) * r,
                cx + math.cos(a2) * r, cy + math.sin(a2) * r,
                c, Icons.STROKE
            )

        # Needle pointing up (center)
        needle_len = size * 0.22
        dl.add_line(cx, cy, cx, cy - needle_len, c, Icons.STROKE)
        dl.add_circle_filled(cx, cy, 3, c)

    @staticmethod
    def gauge_fast(dl, x: float, y: float, size: float, color: Tuple):
        """Speedometer at high position"""
        c = Colors.u32(color)
        cx, cy = x + size/2, y + size * 0.58
        r = size * 0.35

        # Arc
        segments = 16
        for i in range(segments):
            a1 = math.pi + (i * math.pi / segments)
            a2 = math.pi + ((i + 1) * math.pi / segments)
            dl.add_line(
                cx + math.cos(a1) * r, cy + math.sin(a1) * r,
                cx + math.cos(a2) * r, cy + math.sin(a2) * r,
                c, Icons.STROKE
            )

        # Needle pointing to fast (right side)
        needle_angle = math.pi * 0.2
        needle_len = size * 0.22
        dl.add_line(cx, cy, cx + math.cos(needle_angle) * needle_len, cy + math.sin(needle_angle) * needle_len, c, Icons.STROKE)
        dl.add_circle_filled(cx, cy, 3, c)

    # ------------------------------------------------------------------------
    # EDIT
    # ------------------------------------------------------------------------

    @staticmethod
    def undo(dl, x: float, y: float, size: float, color: Tuple):
        """Curved arrow pointing left"""
        c = Colors.u32(color)

        # Curved path using line segments
        segments = 8
        start_angle = -0.3
        end_angle = math.pi + 0.3
        r = size * 0.25
        cx, cy = x + size * 0.55, y + size * 0.45

        pts = []
        for i in range(segments + 1):
            t = i / segments
            angle = start_angle + t * (end_angle - start_angle)
            pts.append((cx + math.cos(angle) * r, cy + math.sin(angle) * r))

        for i in range(len(pts) - 1):
            dl.add_line(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1], c, Icons.STROKE)

        # Arrow head at end
        end_x, end_y = pts[-1]
        dl.add_line(end_x, end_y, end_x + size * 0.1, end_y - size * 0.08, c, Icons.STROKE)
        dl.add_line(end_x, end_y, end_x - size * 0.02, end_y - size * 0.12, c, Icons.STROKE)

    @staticmethod
    def redo(dl, x: float, y: float, size: float, color: Tuple):
        """Curved arrow pointing right"""
        c = Colors.u32(color)

        # Curved path (mirrored)
        segments = 8
        start_angle = math.pi + 0.3
        end_angle = -0.3
        r = size * 0.25
        cx, cy = x + size * 0.45, y + size * 0.45

        pts = []
        for i in range(segments + 1):
            t = i / segments
            angle = start_angle + t * (end_angle - start_angle)
            pts.append((cx + math.cos(angle) * r, cy + math.sin(angle) * r))

        for i in range(len(pts) - 1):
            dl.add_line(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1], c, Icons.STROKE)

        # Arrow head
        end_x, end_y = pts[-1]
        dl.add_line(end_x, end_y, end_x - size * 0.1, end_y - size * 0.08, c, Icons.STROKE)
        dl.add_line(end_x, end_y, end_x + size * 0.02, end_y - size * 0.12, c, Icons.STROKE)

    # ------------------------------------------------------------------------
    # TIMELINE
    # ------------------------------------------------------------------------

    @staticmethod
    def timeline(dl, x: float, y: float, size: float, color: Tuple):
        """Timeline / waveform bars"""
        c = Colors.u32(color)
        cx = x + size/2

        # Vertical bars of varying heights (like audio waveform)
        bar_w = size * 0.06
        heights = [0.3, 0.5, 0.8, 0.6, 0.4, 0.7, 0.35]
        total_w = len(heights) * bar_w + (len(heights) - 1) * bar_w * 0.8
        start_x = cx - total_w / 2

        for i, h in enumerate(heights):
            bx = start_x + i * bar_w * 1.8
            bar_h = size * h * 0.5
            by = y + size/2 - bar_h/2
            dl.add_rect_filled(bx, by, bx + bar_w, by + bar_h, c, 1)

    @staticmethod
    def star(dl, x: float, y: float, size: float, color: Tuple):
        """5-pointed star - ultimate/premium feature"""
        c = Colors.u32(color)
        cx, cy = x + size/2, y + size/2
        r_out = size * 0.4
        r_in = size * 0.16

        pts = []
        for i in range(10):
            angle = (i * math.pi / 5) - math.pi/2
            r = r_out if i % 2 == 0 else r_in
            pts.append((cx + math.cos(angle) * r, cy + math.sin(angle) * r))

        for i in range(10):
            dl.add_line(pts[i][0], pts[i][1], pts[(i+1) % 10][0], pts[(i+1) % 10][1], c, Icons.STROKE)

    @staticmethod
    def magic_wand(dl, x: float, y: float, size: float, color: Tuple):
        """Magic wand - autotune/magic feature"""
        c = Colors.u32(color)

        # Wand body (diagonal line)
        dl.add_line(x + size * 0.25, y + size * 0.75, x + size * 0.7, y + size * 0.3, c, Icons.STROKE + 1)

        # Sparkles around wand tip
        tip_x, tip_y = x + size * 0.7, y + size * 0.3
        spark_size = size * 0.08

        # Cross sparkle
        dl.add_line(tip_x - spark_size, tip_y, tip_x + spark_size, tip_y, c, Icons.STROKE)
        dl.add_line(tip_x, tip_y - spark_size, tip_x, tip_y + spark_size, c, Icons.STROKE)

        # Small dots
        dl.add_circle_filled(tip_x + size * 0.1, tip_y - size * 0.1, 2, c)
        dl.add_circle_filled(tip_x - size * 0.05, tip_y - size * 0.15, 2, c)

    # ------------------------------------------------------------------------
    # EXTRAS
    # ------------------------------------------------------------------------

    @staticmethod
    def list_view(dl, x: float, y: float, size: float, color: Tuple):
        """List/chapters icon"""
        c = Colors.u32(color)
        m = size * 0.22

        # Three rows with bullets
        for i in range(3):
            row_y = y + m + i * size * 0.22
            # Bullet
            dl.add_circle_filled(x + m + size * 0.04, row_y + size * 0.03, size * 0.035, c)
            # Line
            dl.add_line(x + m + size * 0.15, row_y + size * 0.03, x + size - m, row_y + size * 0.03, c, Icons.STROKE)

    @staticmethod
    def cube(dl, x: float, y: float, size: float, color: Tuple):
        """3D cube - simulator"""
        c = Colors.u32(color)
        cx, cy = x + size/2, y + size/2
        s = size * 0.22

        # Front face
        f = [(cx - s, cy), (cx - s, cy + s), (cx + s * 0.3, cy + s), (cx + s * 0.3, cy)]
        for i in range(4):
            dl.add_line(f[i][0], f[i][1], f[(i+1) % 4][0], f[(i+1) % 4][1], c, Icons.STROKE)

        # Top face
        dl.add_line(cx - s, cy, cx - s * 0.3, cy - s * 0.6, c, Icons.STROKE)
        dl.add_line(cx - s * 0.3, cy - s * 0.6, cx + s, cy - s * 0.6, c, Icons.STROKE)
        dl.add_line(cx + s, cy - s * 0.6, cx + s * 0.3, cy, c, Icons.STROKE)

        # Right face
        dl.add_line(cx + s * 0.3, cy, cx + s, cy - s * 0.6, c, Icons.STROKE)
        dl.add_line(cx + s, cy - s * 0.6, cx + s, cy + s * 0.4, c, Icons.STROKE)
        dl.add_line(cx + s, cy + s * 0.4, cx + s * 0.3, cy + s, c, Icons.STROKE)

    @staticmethod
    def wrench(dl, x: float, y: float, size: float, color: Tuple):
        """Wrench - tools/settings"""
        c = Colors.u32(color)

        # Handle
        dl.add_line(x + size * 0.25, y + size * 0.75, x + size * 0.55, y + size * 0.45, c, Icons.STROKE + 1)

        # Head (circle with gap)
        cx, cy = x + size * 0.65, y + size * 0.35
        r = size * 0.18
        segments = 12
        for i in range(segments):
            if i >= 2 and i <= 4:  # Gap for jaw
                continue
            a1 = (i * 2 * math.pi / segments) - math.pi/4
            a2 = ((i + 1) * 2 * math.pi / segments) - math.pi/4
            dl.add_line(
                cx + math.cos(a1) * r, cy + math.sin(a1) * r,
                cx + math.cos(a2) * r, cy + math.sin(a2) * r,
                c, Icons.STROKE
            )

    @staticmethod
    def sparkles(dl, x: float, y: float, size: float, color: Tuple):
        """Sparkles - auto-processing"""
        c = Colors.u32(color)

        # Three 4-pointed stars of different sizes
        def draw_sparkle(cx, cy, r):
            pts = []
            for i in range(8):
                angle = (i * math.pi / 4) - math.pi/2
                rad = r if i % 2 == 0 else r * 0.4
                pts.append((cx + math.cos(angle) * rad, cy + math.sin(angle) * rad))
            for i in range(8):
                dl.add_line(pts[i][0], pts[i][1], pts[(i+1) % 8][0], pts[(i+1) % 8][1], c, Icons.STROKE)

        draw_sparkle(x + size * 0.3, y + size * 0.35, size * 0.18)
        draw_sparkle(x + size * 0.7, y + size * 0.3, size * 0.12)
        draw_sparkle(x + size * 0.5, y + size * 0.7, size * 0.15)

    # ------------------------------------------------------------------------
    # PLAYBACK
    # ------------------------------------------------------------------------

    @staticmethod
    def play(dl, x: float, y: float, size: float, color: Tuple):
        """Play triangle"""
        c = Colors.u32(color)
        m = size * 0.28
        dl.add_triangle_filled(
            x + m, y + m,
            x + m, y + size - m,
            x + size - m + size * 0.05, y + size/2,
            c
        )

    @staticmethod
    def pause(dl, x: float, y: float, size: float, color: Tuple):
        """Pause bars"""
        c = Colors.u32(color)
        m = size * 0.28
        bar_w = size * 0.14
        gap = size * 0.08
        cx = x + size/2

        dl.add_rect_filled(cx - gap - bar_w, y + m, cx - gap, y + size - m, c, 1)
        dl.add_rect_filled(cx + gap, y + m, cx + gap + bar_w, y + size - m, c, 1)

    @staticmethod
    def stop(dl, x: float, y: float, size: float, color: Tuple):
        """Stop square"""
        c = Colors.u32(color)
        m = size * 0.3
        dl.add_rect_filled(x + m, y + m, x + size - m, y + size - m, c, 2)

    @staticmethod
    def skip_start(dl, x: float, y: float, size: float, color: Tuple):
        """Skip to start - bar + triangle"""
        c = Colors.u32(color)
        m = size * 0.25

        # Bar
        dl.add_rect_filled(x + m, y + m, x + m + size * 0.1, y + size - m, c, 0)

        # Triangle pointing left
        dl.add_triangle_filled(
            x + size - m, y + m,
            x + size - m, y + size - m,
            x + m + size * 0.15, y + size/2,
            c
        )

    @staticmethod
    def skip_end(dl, x: float, y: float, size: float, color: Tuple):
        """Skip to end - triangle + bar"""
        c = Colors.u32(color)
        m = size * 0.25

        # Triangle pointing right
        dl.add_triangle_filled(
            x + m, y + m,
            x + m, y + size - m,
            x + size - m - size * 0.15, y + size/2,
            c
        )

        # Bar
        dl.add_rect_filled(x + size - m - size * 0.1, y + m, x + size - m, y + size - m, c, 0)

    @staticmethod
    def step_back(dl, x: float, y: float, size: float, color: Tuple):
        """Step backward - single triangle left"""
        c = Colors.u32(color)
        m = size * 0.3
        dl.add_triangle_filled(
            x + size - m, y + m,
            x + size - m, y + size - m,
            x + m, y + size/2,
            c
        )

    @staticmethod
    def step_forward(dl, x: float, y: float, size: float, color: Tuple):
        """Step forward - single triangle right"""
        c = Colors.u32(color)
        m = size * 0.3
        dl.add_triangle_filled(
            x + m, y + m,
            x + m, y + size - m,
            x + size - m, y + size/2,
            c
        )

    # ------------------------------------------------------------------------
    # STATUS & NAVIGATION
    # ------------------------------------------------------------------------

    @staticmethod
    def dot(dl, x: float, y: float, size: float, color: Tuple, filled: bool = True):
        """Status dot"""
        c = Colors.u32(color)
        cx, cy = x + size/2, y + size/2
        r = size * 0.2
        if filled:
            dl.add_circle_filled(cx, cy, r, c)
        else:
            dl.add_circle(cx, cy, r, c, 12, Icons.STROKE)

    @staticmethod
    def check(dl, x: float, y: float, size: float, color: Tuple):
        """Checkmark"""
        c = Colors.u32(color)
        dl.add_line(x + size * 0.2, y + size * 0.5, x + size * 0.4, y + size * 0.7, c, Icons.STROKE)
        dl.add_line(x + size * 0.4, y + size * 0.7, x + size * 0.8, y + size * 0.3, c, Icons.STROKE)

    @staticmethod
    def menu(dl, x: float, y: float, size: float, color: Tuple):
        """Hamburger menu"""
        c = Colors.u32(color)
        m = size * 0.25
        for i in range(3):
            ly = y + m + i * size * 0.2
            dl.add_line(x + m, ly, x + size - m, ly, c, Icons.STROKE)

    @staticmethod
    def close(dl, x: float, y: float, size: float, color: Tuple):
        """Close X"""
        c = Colors.u32(color)
        m = size * 0.28
        dl.add_line(x + m, y + m, x + size - m, y + size - m, c, Icons.STROKE)
        dl.add_line(x + size - m, y + m, x + m, y + size - m, c, Icons.STROKE)

    @staticmethod
    def chevron_down(dl, x: float, y: float, size: float, color: Tuple):
        """Chevron pointing down"""
        c = Colors.u32(color)
        cx, cy = x + size/2, y + size/2
        w = size * 0.25
        h = size * 0.15
        dl.add_line(cx - w, cy - h, cx, cy + h, c, Icons.STROKE)
        dl.add_line(cx, cy + h, cx + w, cy - h, c, Icons.STROKE)

    @staticmethod
    def chevron_right(dl, x: float, y: float, size: float, color: Tuple):
        """Chevron pointing right"""
        c = Colors.u32(color)
        cx, cy = x + size/2, y + size/2
        w = size * 0.15
        h = size * 0.25
        dl.add_line(cx - w, cy - h, cx + w, cy, c, Icons.STROKE)
        dl.add_line(cx + w, cy, cx - w, cy + h, c, Icons.STROKE)

    @staticmethod
    def satellite(dl, x: float, y: float, size: float, color: Tuple):
        """Satellite dish - streaming"""
        c = Colors.u32(color)
        cx, cy = x + size/2, y + size/2

        # Dish base arc
        r = size * 0.35
        for i in range(8):
            a1 = math.pi * 0.75 + (i * math.pi * 0.5 / 8)
            a2 = math.pi * 0.75 + ((i + 1) * math.pi * 0.5 / 8)
            dl.add_line(
                cx + math.cos(a1) * r, cy + math.sin(a1) * r,
                cx + math.cos(a2) * r, cy + math.sin(a2) * r,
                c, Icons.STROKE
            )

        # Signal waves
        for wave_r in [size * 0.15, size * 0.25]:
            for i in range(4):
                a1 = -math.pi * 0.25 + (i * math.pi * 0.5 / 4)
                a2 = -math.pi * 0.25 + ((i + 1) * math.pi * 0.5 / 4)
                dl.add_line(
                    cx + math.cos(a1) * wave_r, cy + math.sin(a1) * wave_r,
                    cx + math.cos(a2) * wave_r, cy + math.sin(a2) * wave_r,
                    c, Icons.STROKE
                )

        # Center dot
        dl.add_circle_filled(cx, cy, 3, c)

    @staticmethod
    def device(dl, x: float, y: float, size: float, color: Tuple):
        """Device/controller icon"""
        c = Colors.u32(color)
        m = size * 0.2

        # Body
        dl.add_rect(x + m, y + size * 0.35, x + size - m, y + size * 0.75, c, 3, 0, Icons.STROKE)

        # D-pad (left side)
        pad_cx = x + size * 0.35
        pad_cy = y + size * 0.55
        pad_s = size * 0.08
        dl.add_line(pad_cx - pad_s, pad_cy, pad_cx + pad_s, pad_cy, c, Icons.STROKE)
        dl.add_line(pad_cx, pad_cy - pad_s, pad_cx, pad_cy + pad_s, c, Icons.STROKE)

        # Buttons (right side)
        btn_cx = x + size * 0.65
        btn_cy = y + size * 0.55
        dl.add_circle_filled(btn_cx, btn_cy, size * 0.04, c)
        dl.add_circle_filled(btn_cx + size * 0.1, btn_cy, size * 0.04, c)

    # ------------------------------------------------------------------------
    # NUMBERS & SPECIAL
    # ------------------------------------------------------------------------

    @staticmethod
    def number_1(dl, x: float, y: float, size: float, color: Tuple):
        """Number 1 in a rounded square"""
        c = Colors.u32(color)
        m = size * 0.18

        # Rounded square background (outline)
        dl.add_rect(x + m, y + m, x + size - m, y + size - m, c, 4, 0, Icons.STROKE)

        # Number 1
        cx = x + size / 2
        dl.add_line(cx - size * 0.05, y + size * 0.32, cx + size * 0.05, y + size * 0.28, c, Icons.STROKE)
        dl.add_line(cx + size * 0.05, y + size * 0.28, cx + size * 0.05, y + size * 0.72, c, Icons.STROKE)
        dl.add_line(cx - size * 0.12, y + size * 0.72, cx + size * 0.12, y + size * 0.72, c, Icons.STROKE)

    @staticmethod
    def number_2(dl, x: float, y: float, size: float, color: Tuple):
        """Number 2 in a rounded square"""
        c = Colors.u32(color)
        m = size * 0.18

        # Rounded square background (outline)
        dl.add_rect(x + m, y + m, x + size - m, y + size - m, c, 4, 0, Icons.STROKE)

        # Number 2 - simplified curved shape
        cx = x + size / 2
        # Top arc approximation
        dl.add_line(cx - size * 0.12, y + size * 0.38, cx - size * 0.08, y + size * 0.28, c, Icons.STROKE)
        dl.add_line(cx - size * 0.08, y + size * 0.28, cx + size * 0.08, y + size * 0.28, c, Icons.STROKE)
        dl.add_line(cx + size * 0.08, y + size * 0.28, cx + size * 0.12, y + size * 0.38, c, Icons.STROKE)
        # Diagonal
        dl.add_line(cx + size * 0.12, y + size * 0.38, cx - size * 0.12, y + size * 0.72, c, Icons.STROKE)
        # Bottom line
        dl.add_line(cx - size * 0.12, y + size * 0.72, cx + size * 0.12, y + size * 0.72, c, Icons.STROKE)

    @staticmethod
    def document(dl, x: float, y: float, size: float, color: Tuple):
        """Document/page icon"""
        c = Colors.u32(color)
        m = size * 0.2

        # Document outline with folded corner
        fold = size * 0.15
        dl.add_line(x + m, y + m, x + size - m - fold, y + m, c, Icons.STROKE)
        dl.add_line(x + size - m - fold, y + m, x + size - m, y + m + fold, c, Icons.STROKE)
        dl.add_line(x + size - m, y + m + fold, x + size - m, y + size - m, c, Icons.STROKE)
        dl.add_line(x + size - m, y + size - m, x + m, y + size - m, c, Icons.STROKE)
        dl.add_line(x + m, y + size - m, x + m, y + m, c, Icons.STROKE)

        # Fold corner
        dl.add_line(x + size - m - fold, y + m, x + size - m - fold, y + m + fold, c, Icons.STROKE)
        dl.add_line(x + size - m - fold, y + m + fold, x + size - m, y + m + fold, c, Icons.STROKE)

        # Text lines
        line_l = x + m + size * 0.1
        line_r = x + size - m - size * 0.1
        for i in range(3):
            ly = y + size * 0.45 + i * size * 0.12
            dl.add_line(line_l, ly, line_r - (i * size * 0.1), ly, c, Icons.STROKE)

    @staticmethod
    def sync(dl, x: float, y: float, size: float, color: Tuple):
        """Sync/refresh circular arrows"""
        c = Colors.u32(color)
        cx, cy = x + size/2, y + size/2
        r = size * 0.3

        # Draw two curved arrows
        segments = 6

        # Top arc (right to left)
        for i in range(segments):
            a1 = -0.3 + (i * math.pi * 0.8 / segments)
            a2 = -0.3 + ((i + 1) * math.pi * 0.8 / segments)
            dl.add_line(
                cx + math.cos(a1) * r, cy + math.sin(a1) * r,
                cx + math.cos(a2) * r, cy + math.sin(a2) * r,
                c, Icons.STROKE
            )

        # Top arrow head
        ah_x = cx + math.cos(-0.3) * r
        ah_y = cy + math.sin(-0.3) * r
        dl.add_line(ah_x, ah_y, ah_x - size * 0.08, ah_y + size * 0.08, c, Icons.STROKE)
        dl.add_line(ah_x, ah_y, ah_x + size * 0.05, ah_y + size * 0.1, c, Icons.STROKE)

        # Bottom arc (left to right)
        for i in range(segments):
            a1 = math.pi - 0.3 + (i * math.pi * 0.8 / segments)
            a2 = math.pi - 0.3 + ((i + 1) * math.pi * 0.8 / segments)
            dl.add_line(
                cx + math.cos(a1) * r, cy + math.sin(a1) * r,
                cx + math.cos(a2) * r, cy + math.sin(a2) * r,
                c, Icons.STROKE
            )

        # Bottom arrow head
        ah_x2 = cx + math.cos(math.pi - 0.3) * r
        ah_y2 = cy + math.sin(math.pi - 0.3) * r
        dl.add_line(ah_x2, ah_y2, ah_x2 + size * 0.08, ah_y2 - size * 0.08, c, Icons.STROKE)
        dl.add_line(ah_x2, ah_y2, ah_x2 - size * 0.05, ah_y2 - size * 0.1, c, Icons.STROKE)

    @staticmethod
    def flashlight(dl, x: float, y: float, size: float, color: Tuple):
        """Flashlight/torch icon for device control"""
        c = Colors.u32(color)
        cx = x + size/2

        # Handle
        h_top = y + size * 0.55
        h_bot = y + size * 0.85
        h_w = size * 0.12
        dl.add_rect(cx - h_w, h_top, cx + h_w, h_bot, c, 2, 0, Icons.STROKE)

        # Head (wider top)
        head_top = y + size * 0.2
        head_w = size * 0.22
        dl.add_line(cx - head_w, h_top, cx - head_w, head_top + size * 0.1, c, Icons.STROKE)
        dl.add_line(cx - head_w, head_top + size * 0.1, cx - h_w, head_top, c, Icons.STROKE)
        dl.add_line(cx - h_w, head_top, cx + h_w, head_top, c, Icons.STROKE)
        dl.add_line(cx + h_w, head_top, cx + head_w, head_top + size * 0.1, c, Icons.STROKE)
        dl.add_line(cx + head_w, head_top + size * 0.1, cx + head_w, h_top, c, Icons.STROKE)

        # Light rays
        ray_y = y + size * 0.12
        dl.add_line(cx, ray_y, cx, ray_y - size * 0.08, c, Icons.STROKE)
        dl.add_line(cx - size * 0.12, ray_y + size * 0.02, cx - size * 0.18, ray_y - size * 0.04, c, Icons.STROKE)
        dl.add_line(cx + size * 0.12, ray_y + size * 0.02, cx + size * 0.18, ray_y - size * 0.04, c, Icons.STROKE)


# ============================================================================
# UI COMPONENTS
# ============================================================================

class UI:
    """UI component helpers for FunGen design system."""

    @staticmethod
    def icon_button(id_str: str, icon_func: Callable,
                    size: float = 20, active: bool = False,
                    tooltip: str = "", disabled: bool = False) -> bool:
        """
        Render an icon button with vector graphics.

        Args:
            id_str: Unique identifier (can start with ## to hide label)
            icon_func: Drawing function from Icons class
            size: Icon size in pixels
            active: Whether button shows active state
            tooltip: Hover tooltip
            disabled: Whether button is disabled

        Returns:
            True if clicked
        """
        btn_size = size + 16

        if disabled:
            imgui.push_style_var(imgui.STYLE_ALPHA, 0.4)

        pos = imgui.get_cursor_screen_pos()
        clicked = imgui.invisible_button(f"##{id_str}", btn_size, btn_size)

        if disabled:
            clicked = False
            imgui.pop_style_var()

        dl = imgui.get_window_draw_list()
        hovered = imgui.is_item_hovered() and not disabled
        pressed = imgui.is_item_active()

        # Background
        if pressed:
            bg = Colors.GREEN if active else Colors.BG_HIGHLIGHT
        elif hovered:
            bg = Colors.BG_HIGHLIGHT
        elif active:
            bg = Colors.GREEN_DIM
        else:
            bg = Colors.BG_ELEVATED

        dl.add_rect_filled(pos[0], pos[1], pos[0] + btn_size, pos[1] + btn_size,
                          Colors.u32(bg), rounding=6)

        # Icon
        icon_x = pos[0] + (btn_size - size) / 2
        icon_y = pos[1] + (btn_size - size) / 2

        if disabled:
            icon_color = Colors.TEXT_DISABLED
        elif active:
            icon_color = Colors.GREEN
        elif hovered:
            icon_color = Colors.TEXT_PRIMARY
        else:
            icon_color = Colors.TEXT_SECONDARY

        icon_func(dl, icon_x, icon_y, size, icon_color)

        if tooltip and hovered:
            imgui.set_tooltip(tooltip)

        return clicked

    @staticmethod
    def playback_button(icon_func: Callable, primary: bool = False,
                        tooltip: str = "", active: bool = False) -> bool:
        """
        Circular playback button.

        Args:
            icon_func: Drawing function from Icons class
            primary: True for main play/pause button (larger, green)
            tooltip: Hover tooltip
            active: Whether button is in active/playing state

        Returns:
            True if clicked
        """
        btn_size = 48 if primary else 36
        icon_size = 20 if primary else 14

        pos = imgui.get_cursor_screen_pos()
        clicked = imgui.invisible_button(f"##pb{id(icon_func)}{primary}", btn_size, btn_size)

        dl = imgui.get_window_draw_list()
        hovered = imgui.is_item_hovered()
        cx, cy = pos[0] + btn_size/2, pos[1] + btn_size/2

        # Circle background
        if primary:
            if active:
                bg = Colors.GREEN_HOVER if hovered else Colors.GREEN
            else:
                bg = Colors.GREEN_HOVER if hovered else Colors.GREEN
            icon_color = Colors.BG_BASE
        else:
            bg = Colors.BG_HIGHLIGHT if hovered else Colors.BG_ELEVATED
            icon_color = Colors.TEXT_PRIMARY if hovered else Colors.TEXT_SECONDARY

        dl.add_circle_filled(cx, cy, btn_size/2, Colors.u32(bg))

        # Icon
        icon_x = pos[0] + (btn_size - icon_size) / 2
        icon_y = pos[1] + (btn_size - icon_size) / 2
        icon_func(dl, icon_x, icon_y, icon_size, icon_color)

        if tooltip and hovered:
            imgui.set_tooltip(tooltip)

        return clicked

    @staticmethod
    def status_pill(label: str, is_on: bool) -> bool:
        """
        Status indicator pill (e.g., Device ON/OFF).

        Args:
            label: Status label text
            is_on: Whether status is active

        Returns:
            True if clicked
        """
        text_w, text_h = imgui.calc_text_size(label)
        pill_w = text_w + 26
        pill_h = 22

        pos = imgui.get_cursor_screen_pos()
        clicked = imgui.invisible_button(f"##pill_{label}", pill_w, pill_h)

        dl = imgui.get_window_draw_list()
        hovered = imgui.is_item_hovered()

        # Background
        if is_on:
            bg = Colors.GREEN_DIM
        elif hovered:
            bg = Colors.BG_HIGHLIGHT
        else:
            bg = Colors.TRANSPARENT

        dl.add_rect_filled(pos[0], pos[1], pos[0] + pill_w, pos[1] + pill_h,
                          Colors.u32(bg), rounding=pill_h/2)

        # Status dot
        dot_x = pos[0] + 10
        dot_y = pos[1] + pill_h/2
        dot_color = Colors.GREEN if is_on else Colors.TEXT_DISABLED
        dl.add_circle_filled(dot_x, dot_y, 4, Colors.u32(dot_color))

        # Glow when on
        if is_on:
            dl.add_circle(dot_x, dot_y, 6, Colors.u32(Colors.alpha(Colors.GREEN, 0.3)), 12, 1)

        # Label
        text_color = Colors.TEXT_PRIMARY if is_on else Colors.TEXT_MUTED
        dl.add_text(pos[0] + 20, pos[1] + (pill_h - text_h)/2, Colors.u32(text_color), label)

        return clicked

    @staticmethod
    def section_header(label: str, collapsible: bool = False, default_open: bool = True) -> bool:
        """
        Render a section header with optional collapse toggle.

        Args:
            label: Section title
            collapsible: Whether section can be collapsed
            default_open: Initial state if collapsible

        Returns:
            True if section is open/expanded
        """
        if collapsible:
            imgui.push_style_color(imgui.COLOR_HEADER, *Colors.BG_SURFACE)
            imgui.push_style_color(imgui.COLOR_HEADER_HOVERED, *Colors.BG_ELEVATED)
            flags = imgui.TREE_NODE_DEFAULT_OPEN if default_open else 0
            is_open = imgui.collapsing_header(label, flags=flags)[0]
            imgui.pop_style_color(2)
            return is_open
        else:
            imgui.push_style_color(imgui.COLOR_TEXT, *Colors.TEXT_SECONDARY)
            imgui.text(label.upper())
            imgui.pop_style_color()
            imgui.separator()
            return True

    @staticmethod
    def toolbar_separator():
        """Render a vertical separator for toolbar sections."""
        draw_list = imgui.get_window_draw_list()
        pos = imgui.get_cursor_screen_pos()
        height = 32

        dl = imgui.get_window_draw_list()
        dl.add_line(pos[0], pos[1], pos[0], pos[1] + height,
                   Colors.u32(Colors.BORDER_LIGHT), 1.0)

        imgui.dummy(8, height)


# ============================================================================
# THEME
# ============================================================================

class FunGenTheme:
    """ImGui theme configuration for FunGen."""

    # Layout constants
    HEADER_H = 40
    TOOLBAR_H = 52
    SIDEBAR_W = 280
    STATUS_W = 200
    TIMELINE_H = 180

    # Spacing constants
    SPACING_XS = 4
    SPACING_SM = 8
    SPACING_MD = 12
    SPACING_LG = 16
    SPACING_XL = 24

    # Rounding constants
    ROUNDING_SM = 4
    ROUNDING_MD = 6
    ROUNDING_LG = 8
    ROUNDING_FULL = 999  # For pills/circles

    @staticmethod
    def apply():
        """Apply FunGen Spotify-inspired theme to ImGui context."""
        style = imgui.get_style()

        # Sizing
        style.window_padding = (12, 12)
        style.frame_padding = (8, 6)
        style.item_spacing = (8, 6)
        style.item_inner_spacing = (6, 4)
        style.scrollbar_size = 10
        style.grab_min_size = 10

        # Rounding
        style.window_rounding = 0
        style.child_rounding = FunGenTheme.ROUNDING_SM
        style.frame_rounding = FunGenTheme.ROUNDING_SM
        style.popup_rounding = FunGenTheme.ROUNDING_LG  # More rounded modals/popups
        style.scrollbar_rounding = FunGenTheme.ROUNDING_SM
        style.grab_rounding = FunGenTheme.ROUNDING_SM
        style.tab_rounding = FunGenTheme.ROUNDING_SM

        # Borders
        style.window_border_size = 0
        style.child_border_size = 1
        style.frame_border_size = 0
        style.popup_border_size = 1

        # Apply colors
        c = style.colors

        # Backgrounds
        c[imgui.COLOR_WINDOW_BACKGROUND] = Colors.BG_SURFACE
        c[imgui.COLOR_CHILD_BACKGROUND] = Colors.TRANSPARENT
        c[imgui.COLOR_POPUP_BACKGROUND] = Colors.BG_ELEVATED
        c[imgui.COLOR_BORDER] = Colors.BORDER
        c[imgui.COLOR_BORDER_SHADOW] = Colors.TRANSPARENT

        # Text
        c[imgui.COLOR_TEXT] = Colors.TEXT_PRIMARY
        c[imgui.COLOR_TEXT_DISABLED] = Colors.TEXT_DISABLED

        # Frame elements (inputs, etc.)
        c[imgui.COLOR_FRAME_BACKGROUND] = Colors.BG_ELEVATED
        c[imgui.COLOR_FRAME_BACKGROUND_HOVERED] = Colors.BG_HIGHLIGHT
        c[imgui.COLOR_FRAME_BACKGROUND_ACTIVE] = Colors.BG_HIGHLIGHT

        # Buttons
        c[imgui.COLOR_BUTTON] = Colors.BG_ELEVATED
        c[imgui.COLOR_BUTTON_HOVERED] = Colors.BG_HIGHLIGHT
        c[imgui.COLOR_BUTTON_ACTIVE] = Colors.GREEN

        # Headers
        c[imgui.COLOR_HEADER] = Colors.BG_ELEVATED
        c[imgui.COLOR_HEADER_HOVERED] = Colors.BG_HIGHLIGHT
        c[imgui.COLOR_HEADER_ACTIVE] = Colors.GREEN_DIM

        # Tabs
        c[imgui.COLOR_TAB] = Colors.BG_SURFACE
        c[imgui.COLOR_TAB_HOVERED] = Colors.BG_HIGHLIGHT
        c[imgui.COLOR_TAB_ACTIVE] = Colors.BG_ELEVATED
        c[imgui.COLOR_TAB_UNFOCUSED] = Colors.BG_SURFACE
        c[imgui.COLOR_TAB_UNFOCUSED_ACTIVE] = Colors.BG_ELEVATED

        # Title bar
        c[imgui.COLOR_TITLE_BACKGROUND] = Colors.BG_BASE
        c[imgui.COLOR_TITLE_BACKGROUND_ACTIVE] = Colors.BG_SURFACE
        c[imgui.COLOR_TITLE_BACKGROUND_COLLAPSED] = Colors.BG_BASE

        # Scrollbar
        c[imgui.COLOR_SCROLLBAR_BACKGROUND] = Colors.BG_BASE
        c[imgui.COLOR_SCROLLBAR_GRAB] = Colors.TEXT_DISABLED
        c[imgui.COLOR_SCROLLBAR_GRAB_HOVERED] = Colors.TEXT_MUTED
        c[imgui.COLOR_SCROLLBAR_GRAB_ACTIVE] = Colors.TEXT_SECONDARY

        # Slider
        c[imgui.COLOR_SLIDER_GRAB] = Colors.GREEN
        c[imgui.COLOR_SLIDER_GRAB_ACTIVE] = Colors.GREEN_HOVER

        # Checkbox / Radio
        c[imgui.COLOR_CHECK_MARK] = Colors.GREEN

        # Separators
        c[imgui.COLOR_SEPARATOR] = Colors.BORDER
        c[imgui.COLOR_SEPARATOR_HOVERED] = Colors.TEXT_MUTED
        c[imgui.COLOR_SEPARATOR_ACTIVE] = Colors.GREEN

        # Resize grip
        c[imgui.COLOR_RESIZE_GRIP] = Colors.TRANSPARENT
        c[imgui.COLOR_RESIZE_GRIP_HOVERED] = Colors.GREEN_DIM
        c[imgui.COLOR_RESIZE_GRIP_ACTIVE] = Colors.GREEN

        # Menu bar
        c[imgui.COLOR_MENUBAR_BACKGROUND] = Colors.BG_BASE

        # Plot (for any graphs/charts)
        c[imgui.COLOR_PLOT_LINES] = Colors.GREEN
        c[imgui.COLOR_PLOT_LINES_HOVERED] = Colors.GREEN_HOVER
        c[imgui.COLOR_PLOT_HISTOGRAM] = Colors.GREEN
        c[imgui.COLOR_PLOT_HISTOGRAM_HOVERED] = Colors.GREEN_HOVER

        # Drag/drop
        c[imgui.COLOR_DRAG_DROP_TARGET] = Colors.GREEN

        # Navigation highlight
        c[imgui.COLOR_NAV_HIGHLIGHT] = Colors.GREEN
        c[imgui.COLOR_NAV_WINDOWING_HIGHLIGHT] = Colors.alpha(Colors.TEXT_PRIMARY, 0.7)
        c[imgui.COLOR_NAV_WINDOWING_DIM_BACKGROUND] = Colors.alpha(Colors.BG_BASE, 0.2)

        # Modal dimming
        c[imgui.COLOR_MODAL_WINDOW_DIM_BACKGROUND] = Colors.alpha(Colors.BG_BASE, 0.6)

    @staticmethod
    def push_primary_button_style():
        """Push style for primary (green) buttons."""
        imgui.push_style_color(imgui.COLOR_BUTTON, *Colors.GREEN)
        imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, *Colors.GREEN_HOVER)
        imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, *Colors.GREEN_DARK)
        imgui.push_style_color(imgui.COLOR_TEXT, *Colors.BG_BASE)

    @staticmethod
    def pop_primary_button_style():
        """Pop primary button style."""
        imgui.pop_style_color(4)

    @staticmethod
    def push_danger_button_style():
        """Push style for danger/destructive buttons."""
        imgui.push_style_color(imgui.COLOR_BUTTON, *Colors.ERROR)
        imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, *Colors.alpha(Colors.ERROR, 0.8))
        imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, *Colors.alpha(Colors.ERROR, 0.6))
        imgui.push_style_color(imgui.COLOR_TEXT, *Colors.TEXT_PRIMARY)

    @staticmethod
    def pop_danger_button_style():
        """Pop danger button style."""
        imgui.pop_style_color(4)

    @staticmethod
    def push_modal_style():
        """
        Push Spotify-inspired modal dialog styling.

        Use before imgui.begin_popup_modal() or imgui.begin().
        Call pop_modal_style() after imgui.end().
        """
        imgui.push_style_color(imgui.COLOR_POPUP_BACKGROUND, *Colors.BG_SURFACE)
        imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND, *Colors.BG_ELEVATED)
        imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND_ACTIVE, *Colors.GREEN_DIM)
        imgui.push_style_color(imgui.COLOR_BORDER, *Colors.BORDER_LIGHT)
        imgui.push_style_var(imgui.STYLE_WINDOW_ROUNDING, FunGenTheme.ROUNDING_LG)
        imgui.push_style_var(imgui.STYLE_WINDOW_BORDER_SIZE, 1.0)
        imgui.push_style_var(imgui.STYLE_WINDOW_PADDING, (16, 16))

    @staticmethod
    def pop_modal_style():
        """Pop modal dialog styling."""
        imgui.pop_style_var(3)
        imgui.pop_style_color(4)


# ============================================================================
# THEME INTEGRATION FOR EXISTING COLOR SYSTEM
# ============================================================================

class SpotifyTheme:
    """
    Spotify-inspired theme that can be used with the existing ThemeManager.
    Maps all the original DarkTheme colors to the new Spotify palette.
    """

    # Primary accent - Use Spotify Green
    GREEN = Colors.GREEN
    GREEN_LIGHT = Colors.GREEN_HOVER
    GREEN_DARK = Colors.GREEN_DARK

    # Basic Colors - Keep similar but adjusted
    RED = Colors.ERROR
    RED_LIGHT = Colors.alpha(Colors.ERROR, 0.8)
    RED_DARK = (0.7, 0.2, 0.2, 1.0)

    BLUE = Colors.INFO
    BLUE_LIGHT = (0.7, 0.9, 1.0, 1.0)
    BLUE_DARK = (0.2, 0.35, 0.6, 0.6)

    YELLOW = Colors.WARNING
    YELLOW_LIGHT = Colors.alpha(Colors.WARNING, 0.8)
    YELLOW_DARK = (0.8, 0.8, 0.2, 1.0)

    CYAN = Colors.CYAN
    MAGENTA = (0.9, 0.2, 0.9, 1.0)
    ORANGE = Colors.ORANGE
    ORANGE_LIGHT = Colors.alpha(Colors.ORANGE, 0.8)
    ORANGE_DARK = (0.9, 0.6, 0.0, 1.0)

    # Grays - Use Spotify palette
    WHITE = Colors.TEXT_PRIMARY
    WHITE_DARK = (0.9, 0.9, 0.9, 1.0)
    BLACK = (0.0, 0.0, 0.0, 1.0)
    GRAY = Colors.TEXT_SECONDARY
    GRAY_LIGHT = (0.8, 0.8, 0.8, 1.0)
    GRAY_DARK = Colors.TEXT_MUTED
    GRAY_MEDIUM = (0.5, 0.5, 0.5, 1.0)

    BROWN = (0.9, 0.6, 0.2, 0.8)
    PURPLE = (0.7, 0.7, 0.9, 0.8)
    PINK = (0.9, 0.3, 0.6, 0.8)

    TRANSPARENT = Colors.TRANSPARENT
    SEMI_TRANSPARENT = (0, 0, 0, 0.4)

    # App GUI Colors - Use Spotify backgrounds
    APP_MARKER = (*RED[:3], 0.85)
    FLOATING_WIDGET_BG = Colors.BG_SURFACE
    FLOATING_WIDGET_BORDER = Colors.BORDER
    FLOATING_WIDGET_TEXT = Colors.TEXT_PRIMARY
    ENERGY_SAVER_INDICATOR = GREEN
    VERSION_CURRENT_HIGHLIGHT = GREEN
    VERSION_CHANGELOG_TEXT = GRAY
    VIDEO_STATUS_FUNGEN = GREEN
    VIDEO_STATUS_OTHER = YELLOW
    BACKGROUND_CLEAR = Colors.BG_BASE

    # Menu Colors
    FRAME_OFFSET = YELLOW

    # Gauge-specific colors
    GAUGE_BG = Colors.BG_SURFACE
    GAUGE_BORDER = Colors.BORDER
    GAUGE_BAR_RED = RED_DARK
    GAUGE_BAR_GREEN = GREEN_DARK
    GAUGE_TEXT = Colors.TEXT_PRIMARY
    GAUGE_VALUE_TEXT = YELLOW

    # Timeline Colors - Updated for Spotify feel
    TIMELINE_CANVAS_BG = Colors.BG_OVERLAY
    TIMELINE_MARKER = (*RED[:3], 0.9)
    TIMELINE_TIME_TEXT = (*WHITE[:3], 0.7)
    TIMELINE_GRID = (*GRAY_DARK[:3], 0.8)
    TIMELINE_GRID_MAJOR = (*GRAY_DARK[:3], 0.9)
    TIMELINE_GRID_LABELS = GRAY
    TIMELINE_WAVEFORM = BLUE_DARK
    TIMELINE_SELECTED_BORDER = (0.6, 0.0, 0.0, 1.0)
    TIMELINE_PREVIEW_LINE = (*ORANGE[:3], 0.9)
    TIMELINE_PREVIEW_POINT = ORANGE
    TIMELINE_MARQUEE_FILL = (0.5, 0.5, 1.0, 0.3)
    TIMELINE_MARQUEE_BORDER = (0.8, 0.8, 1.0, 0.7)
    TIMELINE_POINT_DEFAULT = GREEN
    TIMELINE_POINT_DRAGGING = RED
    TIMELINE_POINT_SELECTED = (1.0, 0.0, 0.0, 1.0)
    TIMELINE_POINT_HOVER = (0.5, 1.0, 0.5, 1.0)
    ULTIMATE_AUTOTUNE_PREVIEW = (0.2, 1.0, 0.5, 0.5)

    # Control Panel - Spotify styled
    CONTROL_PANEL_ACTIVE_PROGRESS = Colors.INFO
    CONTROL_PANEL_COMPLETED_PROGRESS = GREEN
    CONTROL_PANEL_SUB_PROGRESS = Colors.CYAN
    CONTROL_PANEL_STATUS_READY = GREEN
    CONTROL_PANEL_STATUS_WARNING = ORANGE
    CONTROL_PANEL_STATUS_ERROR = RED
    CONTROL_PANEL_STATUS_INFO = BLUE_LIGHT
    CONTROL_PANEL_SECTION_HEADER = WHITE_DARK

    # Button Palette - Spotify styled
    BUTTON_PRIMARY = Colors.GREEN
    BUTTON_PRIMARY_HOVERED = Colors.GREEN_HOVER
    BUTTON_PRIMARY_ACTIVE = Colors.GREEN_DARK
    BUTTON_DESTRUCTIVE = Colors.ERROR
    BUTTON_DESTRUCTIVE_HOVERED = Colors.alpha(Colors.ERROR, 0.85)
    BUTTON_DESTRUCTIVE_ACTIVE = Colors.alpha(Colors.ERROR, 0.7)


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("""
FunGen Design System - Spotify-Inspired Theme
==============================================

Professional UI design system with vector icons and Spotify color palette.

Quick Start:
------------
    from config.fungen_design import FunGenTheme, Icons, UI, Colors

    # Apply theme once at startup (after ImGui context creation)
    FunGenTheme.apply()

    # In your render loop:

    # Icon button with vector graphics
    if UI.icon_button("save", Icons.save, tooltip="Save Project"):
        save_project()

    imgui.same_line()

    # Circular playback button
    if UI.playback_button(Icons.play, primary=True, tooltip="Play"):
        toggle_playback()

    # Status indicator
    UI.status_pill("Connected", is_on=True)

    # Section header
    if UI.section_header("Settings", collapsible=True):
        # Section content here
        pass

Available Icons:
----------------
Mode:       simple_mode, expert_mode
File:       new_file, folder, save, export
Tracking:   crosshair, simplify, robot
Video:      monitor, monitor_off
Speed:      gauge_slow, gauge_normal, gauge_fast
Edit:       undo, redo
Timeline:   timeline, star, magic_wand
Extras:     list_view, cube, wrench, sparkles, satellite, device
Playback:   play, pause, stop, skip_start, skip_end, step_back, step_forward
Status:     dot, check, menu, close, chevron_down, chevron_right

Color Palette:
--------------
Primary:    Colors.GREEN, Colors.GREEN_HOVER, Colors.GREEN_DARK
Background: Colors.BG_BASE, Colors.BG_SURFACE, Colors.BG_ELEVATED, Colors.BG_HIGHLIGHT
Text:       Colors.TEXT_PRIMARY, Colors.TEXT_SECONDARY, Colors.TEXT_MUTED, Colors.TEXT_DISABLED
Status:     Colors.SUCCESS, Colors.WARNING, Colors.ERROR, Colors.INFO
""")
