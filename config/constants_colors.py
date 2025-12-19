"""
Color constants for GUI components with theme support.
Centralizes all color definitions to replace hardcoded RGBA values throughout the codebase.
All constants are named by their actual color, not their purpose.
Naming scheme: COLOR_SHADE (e.g., BLUE_LIGHT, GRAY_DARK)
"""

# TODO: Structure colors. Global base colors (full opaque), class themes with modified base colors.

# Dark Theme Colors (RGBA format: red, green, blue, alpha) - values from 0.0 to 1.0
class DarkTheme:
    # Basic Colors
    RED = (0.9, 0.2, 0.2, 1.0)
    RED_LIGHT = (0.9, 0.4, 0.4, 0.8)
    RED_DARK = (0.7, 0.2, 0.2, 1.0)

    GREEN = (0.2, 0.8, 0.2, 1.0)
    GREEN_LIGHT = (0.4, 0.9, 0.4, 0.8)
    GREEN_DARK = (0.2, 0.7, 0.2, 1.0)

    BLUE = (0.2, 0.2, 0.9, 1.0)
    BLUE_LIGHT = (0.7, 0.9, 1.0, 1.0)
    BLUE_DARK = (0.2, 0.35, 0.6, 0.6)

    YELLOW = (0.9, 0.9, 0.2, 1.0)
    YELLOW_LIGHT = (0.9, 0.9, 0.3, 0.8)
    YELLOW_DARK = (0.8, 0.8, 0.2, 1.0)

    CYAN = (0.1, 0.9, 0.9, 1.0)
    MAGENTA = (0.9, 0.2, 0.9, 1.0)

    WHITE = (1.0, 1.0, 1.0, 1.0)
    WHITE_DARK = (0.9, 0.9, 0.9, 1.0)
    BLACK = (0.0, 0.0, 0.0, 1.0)
    GRAY = (0.7, 0.7, 0.7, 1.0)
    GRAY_LIGHT = (0.8, 0.8, 0.8, 1.0)
    GRAY_DARK = (0.3, 0.3, 0.3, 1.0)
    GRAY_MEDIUM = (0.5, 0.5, 0.5, 1.0)

    ORANGE = (1.0, 0.6, 0.0, 1.0)
    ORANGE_LIGHT = (1.0, 0.8, 0.4, 0.8)
    ORANGE_DARK = (0.9, 0.6, 0.0, 1.0)
    
    BROWN = (0.9, 0.6, 0.2, 0.8)
    PURPLE = (0.7, 0.7, 0.9, 0.8)
    PINK = (0.9, 0.3, 0.6, 0.8)

    TRANSPARENT = (0, 0, 0, 0)
    SEMI_TRANSPARENT = (0, 0, 0, 0.4)
    
    # App GUI Colors - Spotify-inspired dark surfaces
    APP_MARKER = (*RED[:3], 0.85)  # Red marker for app GUI
    FLOATING_WIDGET_BG = (0.094, 0.094, 0.094, 1.0)  # #181818 Spotify surface
    FLOATING_WIDGET_BORDER = (0.157, 0.157, 0.157, 1.0)  # #282828 Spotify border
    FLOATING_WIDGET_TEXT = WHITE_DARK
    ENERGY_SAVER_INDICATOR = (0.114, 0.725, 0.329, 1.0)  # Spotify green
    VERSION_CURRENT_HIGHLIGHT = (0.114, 0.725, 0.329, 1.0)  # Spotify green
    VERSION_CHANGELOG_TEXT = GRAY
    VIDEO_STATUS_FUNGEN = (0.114, 0.725, 0.329, 1.0)  # Spotify green
    VIDEO_STATUS_OTHER = YELLOW
    BACKGROUND_CLEAR = (0.071, 0.071, 0.071, 1.0)  # #121212 Spotify base

    # Menu Colors
    FRAME_OFFSET = YELLOW

    # Gauge-specific colors - Spotify styled
    GAUGE_BG = (0.094, 0.094, 0.094, 1.0)  # #181818
    GAUGE_BORDER = (0.157, 0.157, 0.157, 1.0)  # #282828
    GAUGE_BAR_RED = (0.85, 0.25, 0.2, 1.0)  # Muted red
    GAUGE_BAR_GREEN = (0.114, 0.725, 0.329, 1.0)  # Spotify green
    GAUGE_TEXT = (0.702, 0.702, 0.702, 1.0)  # #b3b3b3 secondary text
    GAUGE_VALUE_TEXT = (1.0, 1.0, 1.0, 1.0)  # White for emphasis

    # LR Dial Colors
    LR_DIAL_BACKGROUND = FLOATING_WIDGET_BG
    LR_DIAL_BORDER = FLOATING_WIDGET_BORDER
    LR_DIAL_LEFT_LABEL = RED
    LR_DIAL_RIGHT_LABEL = BLUE
    LR_DIAL_INDICATOR_LINE = CYAN
    LR_DIAL_INDICATOR_TIP = YELLOW
    LR_DIAL_VALUE_TEXT = YELLOW

    # Timeline Colors - Spotify styled
    TIMELINE_CANVAS_BG = (0.039, 0.039, 0.039, 1.0)  # #0a0a0a darker for contrast
    TIMELINE_MARKER = (0.85, 0.25, 0.2, 0.9)  # Muted red
    TIMELINE_TIME_TEXT = (0.702, 0.702, 0.702, 0.9)  # #b3b3b3
    TIMELINE_GRID = (0.157, 0.157, 0.157, 0.6)  # #282828
    TIMELINE_GRID_MAJOR = (0.243, 0.243, 0.243, 0.8)  # #3e3e3e
    TIMELINE_GRID_LABELS = (0.447, 0.447, 0.447, 1.0)  # #727272 muted
    TIMELINE_WAVEFORM = (0.2, 0.45, 0.7, 0.6)  # Subtle blue
    TIMELINE_SELECTED_BORDER = (0.85, 0.25, 0.2, 1.0)  # Muted red
    TIMELINE_PREVIEW_LINE = (0.9, 0.5, 0.1, 0.9)  # Orange
    TIMELINE_PREVIEW_POINT = (0.9, 0.5, 0.1, 1.0)  # Orange
    TIMELINE_MARQUEE_FILL = (0.114, 0.725, 0.329, 0.2)  # Green tint
    TIMELINE_MARQUEE_BORDER = (0.114, 0.725, 0.329, 0.7)  # Green border
    TIMELINE_POINT_DEFAULT = (0.114, 0.725, 0.329, 1.0)  # Spotify green
    TIMELINE_POINT_DRAGGING = (0.85, 0.25, 0.2, 1.0)  # Red
    TIMELINE_POINT_SELECTED = (1.0, 1.0, 1.0, 1.0)  # White when selected
    TIMELINE_POINT_HOVER = (0.141, 0.800, 0.380, 1.0)  # Lighter green
    ULTIMATE_AUTOTUNE_PREVIEW = (0.114, 0.725, 0.329, 0.4)  # Green preview

    # Compiler Colors
    COMPILER_SUCCESS = GREEN
    COMPILER_ERROR = RED
    COMPILER_INFO = GRAY
    COMPILER_WARNING = YELLOW
    COMPILER_STARTING = ORANGE
    COMPILER_STOPPING = YELLOW
    COMPILER_OUTPUT_TEXT = BLUE_LIGHT

    # Video Display Colors
    VIDEO_DOMINANT_LIMB = (*GREEN[:3], 0.95)
    VIDEO_DOMINANT_KEYPOINT = (1.0, 0.5, 0.1, 1.0) # orange
    VIDEO_MUTED_LIMB = (0.2, 0.6, 1.0, 0.4) # muted cyan
    VIDEO_MUTED_KEYPOINT = (*RED[:3], 0.5)
    VIDEO_MOTION_UNDETERMINED = (*YELLOW[:3], 0.9)
    VIDEO_MOTION_THRUSTING = (*GREEN[:3], 0.95)
    VIDEO_MOTION_RIDING = (*MAGENTA[:3], 0.95)
    VIDEO_ROI_DRAWING = (*YELLOW[:3], 0.7)
    VIDEO_ROI_BORDER = (*CYAN[:3], 0.7)
    VIDEO_TRACKING_POINT = (*GREEN[:3], 0.9)
    VIDEO_FLOW_VECTOR = (*RED[:3], 0.9)
    VIDEO_BOX_LABEL = WHITE
    VIDEO_OCCLUSION_WARNING = (*ORANGE[:3], 0.95)
    VIDEO_PERSISTENT_REFINED_TRACK = CYAN
    VIDEO_ACTIVE_INTERACTOR = YELLOW
    VIDEO_LOCKED_PENIS = GREEN_DARK
    VIDEO_FILL_COLOR = (*GREEN[:3], 0.4)
    VIDEO_ALIGNED_FALLBACK = (*ORANGE[:3], 0.9)
    VIDEO_INFERRED_BOX = (*PURPLE[:3], 0.85)

    # Navigation Colors - Spotify styled
    NAV_BACKGROUND = (0.094, 0.094, 0.094, 1.0)  # #181818
    NAV_ICON = (0.702, 0.702, 0.702, 0.9)  # #b3b3b3 secondary
    NAV_SCRIPTING_BORDER = (0.114, 0.725, 0.329, 0.6)  # Green
    NAV_SELECTION_PRIMARY = (0.114, 0.725, 0.329, 0.95)  # Spotify green
    NAV_SELECTION_SECONDARY = (0.2, 0.45, 0.7, 0.95)  # Blue
    NAV_TEXT_BLACK = BLACK
    NAV_TEXT_WHITE = WHITE
    NAV_MARKER = (1.0, 1.0, 1.0, 0.7)  # White

    # Box Style Colors
    BOX_GENERAL = (*GRAY_LIGHT[:3], 0.7)
    BOX_PREF_PENIS = (*GREEN[:3], 0.9)
    BOX_LOCKED_PENIS = (*CYAN[:3], 0.8)
    BOX_PUSSY = (1.0, 0.5, 0.8, 0.8) # pink
    BOX_BUTT = (*BROWN[:3], 0.8)
    BOX_TRACKED = (*YELLOW[:3], 0.8)
    BOX_TRACKED_ALT = (*GRAY[:3], 0.7)
    BOX_GENERAL_DETECTION = (0.2, 0.5, 1.0, 0.6) # blue
    BOX_EXCLUDED = (0.5, 0.1, 0.1, 0.5) # dark red

    # Segment Colors
    SEGMENT_BJ = (*GREEN_LIGHT[:3], 0.8)
    SEGMENT_HJ = (*GREEN_LIGHT[:3], 0.8)
    SEGMENT_NR = (*GRAY[:3], 0.7)
    SEGMENT_CG_MISS = (0.4, 0.4, 0.9, 0.8) # blue
    SEGMENT_REV_CG_DOG = (*BROWN[:3], 0.8)
    SEGMENT_CG = (0.3, 0.5, 0.9, 0.8)  # lighter blue for Cowgirl
    SEGMENT_MISS = (0.5, 0.3, 0.9, 0.8)  # purple-blue for Missionary
    SEGMENT_REV_CG = (0.7, 0.5, 0.3, 0.8)  # lighter brown for Reverse Cowgirl
    SEGMENT_DOG = (0.5, 0.4, 0.2, 0.8)  # darker brown for Doggy
    SEGMENT_FOOTJ = (*YELLOW_LIGHT[:3], 0.8)
    SEGMENT_BOOBJ = (*PINK[:3], 0.8)
    SEGMENT_CLOSEUP = (*PURPLE[:3], 0.8)
    SEGMENT_INTRO = (0.5, 0.6, 0.7, 0.7)  # light gray-blue
    SEGMENT_OUTRO = (0.6, 0.5, 0.6, 0.7)  # light gray-purple
    SEGMENT_TRANSITION = (0.6, 0.6, 0.5, 0.7)  # light gray-yellow
    SEGMENT_DEFAULT = TRANSPARENT

    # Control Panel Colors - Spotify styled
    CONTROL_PANEL_ACTIVE_PROGRESS = (0.114, 0.725, 0.329, 1.0)  # Spotify green
    CONTROL_PANEL_COMPLETED_PROGRESS = (0.114, 0.725, 0.329, 1.0)  # Spotify green
    CONTROL_PANEL_SUB_PROGRESS = (0.141, 0.800, 0.380, 1.0)  # Lighter green

    # Control Panel Status Indicator Colors
    CONTROL_PANEL_STATUS_READY = (0.114, 0.725, 0.329, 1.0)  # Spotify green
    CONTROL_PANEL_STATUS_WARNING = (0.961, 0.651, 0.137, 1.0)  # Warning yellow
    CONTROL_PANEL_STATUS_ERROR = (0.85, 0.25, 0.2, 1.0)  # Muted red
    CONTROL_PANEL_STATUS_INFO = (0.702, 0.702, 0.702, 1.0)  # Secondary text

    # Control Panel Section Header Colors
    CONTROL_PANEL_SECTION_HEADER = (1.0, 1.0, 1.0, 1.0)  # White

    # Update Settings Dialog Colors
    UPDATE_TOKEN_VALID = GREEN
    UPDATE_TOKEN_INVALID = RED
    UPDATE_TOKEN_WARNING = ORANGE
    UPDATE_TOKEN_SET = GREEN
    UPDATE_TOKEN_NOT_SET = ORANGE
    UPDATE_DIALOG_TEXT = WHITE
    UPDATE_DIALOG_GRAY_TEXT = GRAY

    # Button Palette Colors (for visual hierarchy) - Spotify-inspired
    # PRIMARY buttons (positive/affirmative actions: Start, Create, Save, etc.)
    # Using Spotify Green (#1db954)
    BUTTON_PRIMARY = (0.114, 0.725, 0.329, 1.0)  # Spotify Green
    BUTTON_PRIMARY_HOVERED = (0.141, 0.800, 0.380, 1.0)  # Lighter green
    BUTTON_PRIMARY_ACTIVE = (0.090, 0.627, 0.263, 1.0)  # Darker green

    # DESTRUCTIVE buttons (dangerous/irreversible actions: Delete, Clear, Abort, etc.)
    # Slightly muted red for Spotify aesthetic
    BUTTON_DESTRUCTIVE = (0.85, 0.25, 0.2, 1.0)  # Muted red
    BUTTON_DESTRUCTIVE_HOVERED = (0.95, 0.35, 0.3, 1.0)  # Lighter red
    BUTTON_DESTRUCTIVE_ACTIVE = (0.75, 0.2, 0.15, 1.0)  # Darker red

    # SECONDARY buttons (default/neutral actions: Browse, Edit, Cancel, etc.)
    # These use ImGui's default button colors (no custom styling needed)


# Current theme (default to dark)
CurrentTheme = DarkTheme

class RGBColors:
    # RGB Colors (for legacy compatibility - 0-255 range)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GREY = (128, 128, 128)
    GREY_LIGHT = (180, 180, 180)
    GREY_DARK = (64, 64, 64)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    GREEN_LIGHT = (128, 255, 128)
    GREEN_DARK = (0, 128, 0)
    BLUE = (0, 0, 255)
    BLUE_DARK = (0, 0, 128)
    YELLOW = (255, 255, 0)
    TEAL = (0, 220, 220)
    CYAN = (0, 255, 255)
    ORANGE = (255, 165, 0)
    ORANGE_LIGHT = (255, 180, 0)
    MAGENTA = (255, 0, 255)
    PURPLE = (128, 0, 128)


    TIMELINE_HEATMAP = [
        (30, 144, 255),   # Dodger Blue
        (34, 139, 34),    # Lime Green
        (255, 215, 0),    # Gold
        (220, 20, 60),    # Crimson
        (147, 112, 219),  # Medium Purple
        (37, 22, 122)     # Dark Blue (Cap color)
    ]
    TIMELINE_COLOR_SPEED_STEP = 250.0  # Speed (pixels/sec) used to step through the color map.
    TIMELINE_HEATMAP_BACKGROUND = (20, 20, 25, 255)
    TIMELINE_COLOR_ALPHA = 0.9 # alpha value for the timeline heatmap

    # General Slider Colors
    FPS_TARGET_MARKER = YELLOW
    FPS_TRACKER_MARKER = GREEN
    FPS_PROCESSOR_MARKER = RED

    # OBJECT DETECTION CLASS COLORS
    CLASS_COLORS = {
        "penis": RED,
        "glans": GREEN_DARK,
        "pussy": BLUE,
        "butt": ORANGE_LIGHT,
        "anus": PURPLE,
        "breast": ORANGE,
        "navel": CYAN,
        "hand": MAGENTA,
        "face": GREEN,
        "foot": (165, 42, 42), # Brown
        "hips center": BLACK,
        "locked_penis": CYAN,
    }
