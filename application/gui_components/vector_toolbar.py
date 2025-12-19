"""
Vector Toolbar Components
=========================

Provides vector-based toolbar buttons using the FunGen design system.
Replaces PNG emoji icons with programmatically drawn vector graphics.

Usage:
    from application.gui_components.vector_toolbar import VectorToolbar

    # In toolbar render loop:
    if VectorToolbar.button("save", VectorToolbar.SAVE, tooltip="Save"):
        save_project()
"""

import imgui
from typing import Callable, Tuple, Optional
from config.fungen_design import Icons, Colors, UI


class VectorToolbar:
    """
    Vector-based toolbar button renderer.

    Provides a clean API for rendering toolbar buttons with vector icons
    from the FunGen design system.
    """

    # Icon size for toolbar buttons
    ICON_SIZE = 20
    BUTTON_PADDING = 4

    # ========================================================================
    # ICON MAPPING - Maps semantic names to icon functions
    # ========================================================================

    # Mode toggles
    SIMPLE_MODE = Icons.simple_mode
    EXPERT_MODE = Icons.expert_mode

    # File operations
    NEW_FILE = Icons.new_file
    FOLDER = Icons.folder
    SAVE = Icons.save
    EXPORT = Icons.export

    # Edit operations
    UNDO = Icons.undo
    REDO = Icons.redo

    # Playback
    PLAY = Icons.play
    PAUSE = Icons.pause
    STOP = Icons.stop
    SKIP_START = Icons.skip_start
    SKIP_END = Icons.skip_end
    STEP_BACK = Icons.step_back
    STEP_FORWARD = Icons.step_forward

    # Speed/gauge
    GAUGE_SLOW = Icons.gauge_slow
    GAUGE_NORMAL = Icons.gauge_normal
    GAUGE_FAST = Icons.gauge_fast

    # Tracking
    CROSSHAIR = Icons.crosshair
    ROBOT = Icons.robot
    SIMPLIFY = Icons.simplify

    # Video
    MONITOR = Icons.monitor
    MONITOR_OFF = Icons.monitor_off

    # Tools
    MAGIC_WAND = Icons.magic_wand
    WRENCH = Icons.wrench
    SPARKLES = Icons.sparkles
    STAR = Icons.star

    # Timeline
    TIMELINE = Icons.timeline
    NUMBER_1 = Icons.number_1
    NUMBER_2 = Icons.number_2

    # Extras
    LIST_VIEW = Icons.list_view
    CUBE = Icons.cube
    SATELLITE = Icons.satellite
    DEVICE = Icons.device
    FLASHLIGHT = Icons.flashlight
    DOCUMENT = Icons.document
    SYNC = Icons.sync

    # Navigation
    CHEVRON_DOWN = Icons.chevron_down
    CHEVRON_RIGHT = Icons.chevron_right
    MENU = Icons.menu
    CLOSE = Icons.close
    CHECK = Icons.check

    @staticmethod
    def button(id_str: str, icon_func: Callable,
               size: float = None, tooltip: str = "",
               active: bool = False, disabled: bool = False,
               color_override: Tuple = None) -> bool:
        """
        Render a vector icon button.

        Args:
            id_str: Unique button identifier
            icon_func: Icon drawing function from Icons class
            size: Icon size (default: ICON_SIZE)
            tooltip: Hover tooltip text
            active: Show active/pressed state
            disabled: Disable the button
            color_override: Override icon color (R, G, B, A tuple)

        Returns:
            True if button was clicked
        """
        if size is None:
            size = VectorToolbar.ICON_SIZE

        btn_size = size + VectorToolbar.BUTTON_PADDING * 2 + 8

        if disabled:
            imgui.push_style_var(imgui.STYLE_ALPHA, 0.35)

        pos = imgui.get_cursor_screen_pos()
        clicked = imgui.invisible_button(f"##{id_str}", btn_size, btn_size)

        if disabled:
            clicked = False
            imgui.pop_style_var()

        dl = imgui.get_window_draw_list()
        hovered = imgui.is_item_hovered() and not disabled
        pressed = imgui.is_item_active()

        # Determine background color
        if pressed and not disabled:
            bg = Colors.GREEN if active else Colors.BG_HIGHLIGHT
        elif hovered:
            bg = Colors.BG_HIGHLIGHT
        elif active:
            bg = Colors.GREEN_DIM
        else:
            bg = Colors.BG_ELEVATED

        # Draw background
        dl.add_rect_filled(
            pos[0], pos[1],
            pos[0] + btn_size, pos[1] + btn_size,
            Colors.u32(bg), rounding=6
        )

        # Determine icon color
        if color_override:
            icon_color = color_override
        elif disabled:
            icon_color = Colors.TEXT_DISABLED
        elif active:
            icon_color = Colors.GREEN
        elif hovered:
            icon_color = Colors.TEXT_PRIMARY
        else:
            icon_color = Colors.TEXT_SECONDARY

        # Draw icon centered in button
        icon_x = pos[0] + (btn_size - size) / 2
        icon_y = pos[1] + (btn_size - size) / 2
        icon_func(dl, icon_x, icon_y, size, icon_color)

        # Tooltip
        if tooltip and hovered:
            imgui.set_tooltip(tooltip)

        return clicked

    @staticmethod
    def toggle_button(id_str: str, icon_func: Callable,
                      is_active: bool, size: float = None,
                      tooltip: str = "", disabled: bool = False) -> bool:
        """
        Render a toggle button that shows active state.

        Args:
            id_str: Unique button identifier
            icon_func: Icon drawing function
            is_active: Current toggle state
            size: Icon size
            tooltip: Base tooltip (will append state)
            disabled: Disable the button

        Returns:
            True if button was clicked
        """
        full_tooltip = f"{tooltip} ({'On' if is_active else 'Off'})" if tooltip else ""
        return VectorToolbar.button(
            id_str, icon_func, size, full_tooltip,
            active=is_active, disabled=disabled
        )

    @staticmethod
    def state_button(id_str: str, icon_func: Callable,
                     state: str, size: float = None,
                     tooltip: str = "", disabled: bool = False) -> bool:
        """
        Render a button with state-based coloring.

        Args:
            id_str: Unique button identifier
            icon_func: Icon drawing function
            state: One of 'default', 'active', 'success', 'warning', 'error'
            size: Icon size
            tooltip: Hover tooltip
            disabled: Disable the button

        Returns:
            True if button was clicked
        """
        color_map = {
            'default': None,
            'active': Colors.GREEN,
            'success': Colors.GREEN,
            'warning': Colors.WARNING,
            'error': Colors.ERROR,
            'info': Colors.INFO,
        }

        color = color_map.get(state, None)
        is_active = state in ('active', 'success')

        return VectorToolbar.button(
            id_str, icon_func, size, tooltip,
            active=is_active, disabled=disabled,
            color_override=color
        )

    @staticmethod
    def playback_button(icon_func: Callable, primary: bool = False,
                        tooltip: str = "", active: bool = False) -> bool:
        """
        Render a circular playback control button.

        Args:
            icon_func: Icon drawing function (play, pause, stop, etc.)
            primary: True for main play/pause button (larger, green)
            tooltip: Hover tooltip
            active: Whether currently playing/active

        Returns:
            True if button was clicked
        """
        btn_size = 44 if primary else 32
        icon_size = 18 if primary else 12

        pos = imgui.get_cursor_screen_pos()
        clicked = imgui.invisible_button(f"##pb_{id(icon_func)}_{primary}", btn_size, btn_size)

        dl = imgui.get_window_draw_list()
        hovered = imgui.is_item_hovered()
        cx, cy = pos[0] + btn_size/2, pos[1] + btn_size/2

        # Background circle
        if primary:
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
    def separator():
        """Render a vertical separator between button groups."""
        draw_list = imgui.get_window_draw_list()
        pos = imgui.get_cursor_screen_pos()
        height = VectorToolbar.ICON_SIZE + VectorToolbar.BUTTON_PADDING * 2 + 8

        draw_list.add_line(
            pos[0], pos[1],
            pos[0], pos[1] + height,
            Colors.u32(Colors.BORDER), 1.0
        )

        imgui.dummy(8, height)

    @staticmethod
    def section_label(text: str):
        """Render a section label above buttons."""
        draw_list = imgui.get_window_draw_list()
        pos = imgui.get_cursor_screen_pos()

        draw_list.add_text(
            pos[0], pos[1],
            Colors.u32(Colors.TEXT_MUTED),
            text.upper()
        )

        imgui.dummy(0, 14)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def vector_icon_button(id_str: str, icon_func: Callable,
                       tooltip: str = "", active: bool = False,
                       disabled: bool = False) -> bool:
    """Shorthand for VectorToolbar.button with default size."""
    return VectorToolbar.button(id_str, icon_func, tooltip=tooltip,
                                active=active, disabled=disabled)


def get_mode_icon(is_expert: bool) -> Callable:
    """Get the appropriate mode icon based on current mode."""
    return VectorToolbar.EXPERT_MODE if is_expert else VectorToolbar.SIMPLE_MODE


def get_playback_icon(is_playing: bool) -> Callable:
    """Get play or pause icon based on playback state."""
    return VectorToolbar.PAUSE if is_playing else VectorToolbar.PLAY


def get_video_icon(is_visible: bool) -> Callable:
    """Get monitor or monitor_off icon based on visibility."""
    return VectorToolbar.MONITOR if is_visible else VectorToolbar.MONITOR_OFF
