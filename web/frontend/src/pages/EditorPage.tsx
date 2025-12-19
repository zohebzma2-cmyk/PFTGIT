import { useState } from 'react'
import {
  Upload,
  Play,
  Pause,
  SkipBack,
  SkipForward,
  Settings,
  Wand2,
  Save,
  Download,
  Bluetooth,
  Zap,
  ChevronLeft,
  ChevronRight,
  ZoomIn,
  ZoomOut,
  Maximize
} from 'lucide-react'
import { clsx } from 'clsx'

// Toolbar button component
function ToolbarButton({
  icon: Icon,
  label,
  active = false,
  onClick,
}: {
  icon: React.ElementType
  label: string
  active?: boolean
  onClick?: () => void
}) {
  return (
    <button
      onClick={onClick}
      className={clsx('toolbar-btn', active && 'active')}
      title={label}
    >
      <Icon className="w-5 h-5" />
    </button>
  )
}

export default function EditorPage() {
  const [isPlaying, setIsPlaying] = useState(false)
  const [hasVideo, setHasVideo] = useState(false)

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* Toolbar */}
      <div className="h-12 bg-bg-surface border-b border-border flex items-center px-2 gap-1">
        {/* File actions */}
        <div className="flex items-center gap-1 pr-2 border-r border-border">
          <ToolbarButton icon={Upload} label="Open Video" />
          <ToolbarButton icon={Save} label="Save Project" />
          <ToolbarButton icon={Download} label="Export Funscript" />
        </div>

        {/* Playback controls */}
        <div className="flex items-center gap-1 px-2 border-r border-border">
          <ToolbarButton icon={SkipBack} label="Previous Frame" />
          <ToolbarButton
            icon={isPlaying ? Pause : Play}
            label={isPlaying ? 'Pause' : 'Play'}
            active={isPlaying}
            onClick={() => setIsPlaying(!isPlaying)}
          />
          <ToolbarButton icon={SkipForward} label="Next Frame" />
        </div>

        {/* View controls */}
        <div className="flex items-center gap-1 px-2 border-r border-border">
          <ToolbarButton icon={ZoomOut} label="Zoom Out" />
          <ToolbarButton icon={ZoomIn} label="Zoom In" />
          <ToolbarButton icon={Maximize} label="Fit View" />
        </div>

        {/* AI Generation */}
        <div className="flex items-center gap-1 px-2 border-r border-border">
          <button className="flex items-center gap-2 px-3 py-1.5 rounded-md bg-primary text-bg-base hover:bg-primary-hover transition-colors text-sm font-medium">
            <Wand2 className="w-4 h-4" />
            Generate
          </button>
        </div>

        {/* Device connection */}
        <div className="flex items-center gap-1 px-2">
          <ToolbarButton icon={Bluetooth} label="Connect Device" />
          <ToolbarButton icon={Settings} label="Settings" />
        </div>

        {/* Spacer */}
        <div className="flex-1" />

        {/* Mode toggle */}
        <div className="flex items-center gap-2 text-sm text-text-secondary">
          <span>Simple</span>
          <button className="w-10 h-5 bg-bg-elevated rounded-full relative">
            <div className="absolute left-0.5 top-0.5 w-4 h-4 bg-text-secondary rounded-full transition-transform" />
          </button>
          <span className="text-text-muted">Expert</span>
        </div>
      </div>

      {/* Main editor area */}
      <div className="flex-1 flex overflow-hidden">
        {/* Video area */}
        <div className="flex-1 flex flex-col">
          {/* Video player */}
          <div className="flex-1 bg-bg-overlay flex items-center justify-center">
            {hasVideo ? (
              <video className="max-w-full max-h-full" controls />
            ) : (
              <div className="text-center">
                <div className="w-20 h-20 rounded-full bg-bg-elevated flex items-center justify-center mx-auto mb-4">
                  <Upload className="w-10 h-10 text-text-muted" />
                </div>
                <h3 className="text-lg font-semibold text-text-primary mb-2">
                  No video loaded
                </h3>
                <p className="text-text-secondary mb-4">
                  Upload a video to get started
                </p>
                <button
                  onClick={() => setHasVideo(true)}
                  className="btn-primary"
                >
                  <Upload className="w-5 h-5" />
                  Upload Video
                </button>
              </div>
            )}
          </div>

          {/* Timeline */}
          <div className="h-48 bg-bg-surface border-t border-border p-4">
            <div className="h-full bg-bg-elevated rounded-lg flex items-center justify-center">
              <p className="text-text-muted">Timeline - Load a video to see funscript points</p>
            </div>
          </div>
        </div>

        {/* Right sidebar - Control Panel */}
        <div className="w-72 bg-bg-surface border-l border-border overflow-y-auto">
          <div className="p-4">
            <h2 className="font-semibold text-text-primary mb-4 flex items-center gap-2">
              <Settings className="w-5 h-5" />
              Control Panel
            </h2>

            {/* Generation Settings */}
            <div className="space-y-4">
              <div className="card">
                <h3 className="text-sm font-medium text-text-primary mb-3">
                  AI Settings
                </h3>
                <div className="space-y-3">
                  <div>
                    <label className="text-xs text-text-secondary block mb-1">
                      Confidence Threshold
                    </label>
                    <input
                      type="range"
                      min="0"
                      max="100"
                      defaultValue="50"
                      className="w-full accent-primary"
                    />
                  </div>
                  <div>
                    <label className="text-xs text-text-secondary block mb-1">
                      Smoothing
                    </label>
                    <input
                      type="range"
                      min="0"
                      max="100"
                      defaultValue="30"
                      className="w-full accent-primary"
                    />
                  </div>
                </div>
              </div>

              {/* Device Status */}
              <div className="card">
                <h3 className="text-sm font-medium text-text-primary mb-3">
                  Device
                </h3>
                <div className="flex items-center gap-3">
                  <div className="w-3 h-3 rounded-full bg-text-disabled" />
                  <span className="text-sm text-text-secondary">
                    No device connected
                  </span>
                </div>
                <button className="w-full btn-secondary mt-3 text-sm">
                  <Bluetooth className="w-4 h-4" />
                  Connect
                </button>
              </div>

              {/* Stats */}
              <div className="card">
                <h3 className="text-sm font-medium text-text-primary mb-3">
                  Statistics
                </h3>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-text-secondary">Points</span>
                    <span className="text-text-primary font-medium">0</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-text-secondary">Duration</span>
                    <span className="text-text-primary font-medium">00:00</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-text-secondary">Avg Speed</span>
                    <span className="text-text-primary font-medium">0 u/s</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
