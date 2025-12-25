import { useState, useRef } from 'react'
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
  ZoomIn,
  ZoomOut,
  Maximize,
  Sliders,
  Activity,
  Loader2
} from 'lucide-react'
import { clsx } from 'clsx'
import { useModeStore } from '@/store/modeStore'
import { videosApi, VideoMetadata } from '@/api/client'

// Helper to format duration in ms to MM:SS
function formatDuration(ms: number): string {
  const totalSeconds = Math.floor(ms / 1000)
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`
}

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
  const [video, setVideo] = useState<VideoMetadata | null>(null)
  const [isUploading, setIsUploading] = useState(false)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const { mode, toggleMode } = useModeStore()
  const isExpert = mode === 'expert'

  const handleFileSelect = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    setIsUploading(true)
    setUploadError(null)

    try {
      const metadata = await videosApi.upload(file)
      setVideo(metadata)
    } catch (error) {
      console.error('Upload failed:', error)
      setUploadError(error instanceof Error ? error.message : 'Upload failed')
    } finally {
      setIsUploading(false)
      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = ''
      }
    }
  }

  const openFileDialog = () => {
    fileInputRef.current?.click()
  }

  // Get video stream URL from API
  const videoStreamUrl = video ? videosApi.getStreamUrl(video.id) : null

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept="video/*,.mp4,.mkv,.avi,.mov,.webm"
        className="hidden"
        onChange={handleFileSelect}
      />

      {/* Toolbar */}
      <div className="h-12 bg-bg-surface border-b border-border flex items-center px-2 gap-1">
        {/* File actions */}
        <div className="flex items-center gap-1 pr-2 border-r border-border">
          <ToolbarButton icon={Upload} label="Open Video" onClick={openFileDialog} />
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
        <div className="flex items-center gap-2 text-sm">
          <span className={clsx(
            'transition-colors',
            !isExpert ? 'text-primary font-medium' : 'text-text-muted'
          )}>
            Simple
          </span>
          <button
            onClick={toggleMode}
            className={clsx(
              'w-10 h-5 rounded-full relative transition-colors',
              isExpert ? 'bg-primary' : 'bg-bg-elevated'
            )}
          >
            <div
              className={clsx(
                'absolute top-0.5 w-4 h-4 rounded-full transition-all duration-200',
                isExpert
                  ? 'left-[22px] bg-bg-base'
                  : 'left-0.5 bg-text-secondary'
              )}
            />
          </button>
          <span className={clsx(
            'transition-colors',
            isExpert ? 'text-primary font-medium' : 'text-text-muted'
          )}>
            Expert
          </span>
        </div>
      </div>

      {/* Main editor area */}
      <div className="flex-1 flex overflow-hidden">
        {/* Video area */}
        <div className="flex-1 flex flex-col">
          {/* Video player */}
          <div className="flex-1 bg-bg-overlay flex items-center justify-center">
            {isUploading ? (
              <div className="text-center">
                <div className="w-20 h-20 rounded-full bg-bg-elevated flex items-center justify-center mx-auto mb-4">
                  <Loader2 className="w-10 h-10 text-primary animate-spin" />
                </div>
                <h3 className="text-lg font-semibold text-text-primary mb-2">
                  Uploading video...
                </h3>
                <p className="text-text-secondary">
                  Please wait while we process your video
                </p>
              </div>
            ) : video && videoStreamUrl ? (
              <video
                className="max-w-full max-h-full"
                controls
                src={videoStreamUrl}
                onPlay={() => setIsPlaying(true)}
                onPause={() => setIsPlaying(false)}
              />
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
                {uploadError && (
                  <p className="text-error text-sm mb-4">
                    Error: {uploadError}
                  </p>
                )}
                <button
                  onClick={openFileDialog}
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
              {isExpert ? <Sliders className="w-5 h-5" /> : <Settings className="w-5 h-5" />}
              {isExpert ? 'Expert Controls' : 'Control Panel'}
            </h2>

            <div className="space-y-4">
              {/* Simple Mode: Quick Actions */}
              {!isExpert && (
                <div className="card">
                  <h3 className="text-sm font-medium text-text-primary mb-3">
                    Quick Actions
                  </h3>
                  <p className="text-xs text-text-secondary mb-3">
                    Load a video and click Generate to create a funscript automatically.
                  </p>
                  <button className="w-full btn-primary text-sm">
                    <Wand2 className="w-4 h-4" />
                    Auto Generate
                  </button>
                </div>
              )}

              {/* Expert Mode: AI Settings */}
              {isExpert && (
                <div className="card">
                  <h3 className="text-sm font-medium text-text-primary mb-3 flex items-center gap-2">
                    <Activity className="w-4 h-4" />
                    AI Settings
                  </h3>
                  <div className="space-y-3">
                    <div>
                      <div className="flex justify-between items-center mb-1">
                        <label className="text-xs text-text-secondary">
                          Confidence Threshold
                        </label>
                        <span className="text-xs text-primary font-medium">50%</span>
                      </div>
                      <input
                        type="range"
                        min="0"
                        max="100"
                        defaultValue="50"
                        className="w-full accent-primary"
                      />
                    </div>
                    <div>
                      <div className="flex justify-between items-center mb-1">
                        <label className="text-xs text-text-secondary">
                          Smoothing
                        </label>
                        <span className="text-xs text-primary font-medium">30%</span>
                      </div>
                      <input
                        type="range"
                        min="0"
                        max="100"
                        defaultValue="30"
                        className="w-full accent-primary"
                      />
                    </div>
                    <div>
                      <div className="flex justify-between items-center mb-1">
                        <label className="text-xs text-text-secondary">
                          Min Point Distance
                        </label>
                        <span className="text-xs text-primary font-medium">100ms</span>
                      </div>
                      <input
                        type="range"
                        min="50"
                        max="500"
                        defaultValue="100"
                        className="w-full accent-primary"
                      />
                    </div>
                    <div>
                      <div className="flex justify-between items-center mb-1">
                        <label className="text-xs text-text-secondary">
                          Motion Sensitivity
                        </label>
                        <span className="text-xs text-primary font-medium">70%</span>
                      </div>
                      <input
                        type="range"
                        min="0"
                        max="100"
                        defaultValue="70"
                        className="w-full accent-primary"
                      />
                    </div>
                  </div>
                </div>
              )}

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
                    <span className="text-text-primary font-medium">
                      {video?.duration_ms ? formatDuration(video.duration_ms) : '00:00'}
                    </span>
                  </div>
                  {video && (
                    <>
                      {video.width && video.height && (
                        <div className="flex justify-between">
                          <span className="text-text-secondary">Resolution</span>
                          <span className="text-text-primary font-medium">
                            {video.width}x{video.height}
                          </span>
                        </div>
                      )}
                      {video.fps && (
                        <div className="flex justify-between">
                          <span className="text-text-secondary">FPS</span>
                          <span className="text-text-primary font-medium">
                            {video.fps.toFixed(1)}
                          </span>
                        </div>
                      )}
                    </>
                  )}
                  {isExpert && (
                    <>
                      <div className="flex justify-between">
                        <span className="text-text-secondary">Avg Speed</span>
                        <span className="text-text-primary font-medium">0 u/s</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-text-secondary">Peak Speed</span>
                        <span className="text-text-primary font-medium">0 u/s</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-text-secondary">Intensity</span>
                        <span className="text-text-primary font-medium">0%</span>
                      </div>
                    </>
                  )}
                </div>
              </div>

              {/* Expert Mode: Advanced Options */}
              {isExpert && (
                <div className="card">
                  <h3 className="text-sm font-medium text-text-primary mb-3">
                    Advanced Options
                  </h3>
                  <div className="space-y-2">
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input type="checkbox" className="accent-primary" defaultChecked />
                      <span className="text-sm text-text-secondary">Auto-smooth peaks</span>
                    </label>
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input type="checkbox" className="accent-primary" />
                      <span className="text-sm text-text-secondary">Invert output</span>
                    </label>
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input type="checkbox" className="accent-primary" defaultChecked />
                      <span className="text-sm text-text-secondary">Limit range (0-100)</span>
                    </label>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
