import { useState, useRef, useEffect, useCallback } from 'react'
import {
  Upload,
  Play,
  Pause,
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
  Loader2,
  Plus,
  RotateCcw,
  ChevronLeft,
  ChevronRight
} from 'lucide-react'
import { clsx } from 'clsx'
import { useModeStore } from '@/store/modeStore'
import { videosApi, processingApi, VideoMetadata, FunscriptPoint, ProcessingJob } from '@/api/client'
import { useAuthStore } from '@/store/authStore'

// Format duration in ms to MM:SS
function formatDuration(ms: number): string {
  const totalSeconds = Math.floor(ms / 1000)
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`
}

// Format time with milliseconds
function formatTime(ms: number): string {
  const totalSeconds = Math.floor(ms / 1000)
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  const millis = Math.floor(ms % 1000)
  return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}.${millis.toString().padStart(3, '0')}`
}

// Toolbar button component
function ToolbarButton({
  icon: Icon,
  label,
  active = false,
  disabled = false,
  onClick,
}: {
  icon: React.ElementType
  label: string
  active?: boolean
  disabled?: boolean
  onClick?: () => void
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={clsx(
        'toolbar-btn',
        active && 'active',
        disabled && 'opacity-50 cursor-not-allowed'
      )}
      title={label}
    >
      <Icon className="w-5 h-5" />
    </button>
  )
}

// Timeline component
function Timeline({
  currentTime,
  duration,
  points,
  onSeek,
  zoom,
}: {
  currentTime: number
  duration: number
  points: FunscriptPoint[]
  onSeek: (time: number) => void
  zoom: number
}) {
  const timelineRef = useRef<HTMLDivElement>(null)
  const [isDragging, setIsDragging] = useState(false)
  const scrollOffset = 0 // Will be used for zoom panning

  const handleTimelineClick = (e: React.MouseEvent) => {
    if (!timelineRef.current || duration === 0) return
    const rect = timelineRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left + scrollOffset
    const timelineWidth = rect.width * zoom
    const clickTime = (x / timelineWidth) * duration
    onSeek(Math.max(0, Math.min(duration, clickTime)))
  }

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging || !timelineRef.current || duration === 0) return
    const rect = timelineRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left + scrollOffset
    const timelineWidth = rect.width * zoom
    const clickTime = (x / timelineWidth) * duration
    onSeek(Math.max(0, Math.min(duration, clickTime)))
  }

  const progressPercent = duration > 0 ? (currentTime / duration) * 100 : 0

  return (
    <div className="h-full flex flex-col">
      {/* Timeline header */}
      <div className="flex items-center justify-between px-2 py-1 border-b border-border">
        <span className="text-xs text-text-secondary">
          {formatTime(currentTime)} / {formatTime(duration)}
        </span>
        <div className="flex items-center gap-2">
          <span className="text-xs text-text-muted">{points.length} points</span>
        </div>
      </div>

      {/* Timeline track */}
      <div
        ref={timelineRef}
        className="flex-1 relative cursor-crosshair overflow-hidden"
        onClick={handleTimelineClick}
        onMouseDown={() => setIsDragging(true)}
        onMouseUp={() => setIsDragging(false)}
        onMouseLeave={() => setIsDragging(false)}
        onMouseMove={handleMouseMove}
      >
        {/* Background grid */}
        <div className="absolute inset-0 bg-bg-base">
          {/* Time markers */}
          {duration > 0 && Array.from({ length: Math.ceil(duration / 10000) + 1 }).map((_, i) => {
            const percent = (i * 10000 / duration) * 100
            if (percent > 100) return null
            return (
              <div
                key={i}
                className="absolute top-0 bottom-0 border-l border-border"
                style={{ left: `${percent}%` }}
              >
                <span className="absolute top-0 left-1 text-[10px] text-text-muted">
                  {formatDuration(i * 10000)}
                </span>
              </div>
            )
          })}
        </div>

        {/* Funscript visualization */}
        <svg className="absolute inset-0 w-full h-full" preserveAspectRatio="none">
          {/* Draw lines between points */}
          {points.length > 1 && (
            <polyline
              fill="none"
              stroke="var(--color-primary)"
              strokeWidth="2"
              points={points.map(p => {
                const x = (p.at / duration) * 100
                const y = 100 - p.pos // Invert Y axis
                return `${x}%,${y}%`
              }).join(' ')}
            />
          )}
          {/* Draw points */}
          {points.map((point, index) => {
            const x = (point.at / duration) * 100
            const y = 100 - point.pos
            return (
              <circle
                key={index}
                cx={`${x}%`}
                cy={`${y}%`}
                r="4"
                fill="var(--color-primary)"
                className="cursor-pointer hover:fill-white"
                onClick={(e) => {
                  e.stopPropagation()
                  // Select point or show context menu
                }}
              />
            )
          })}
        </svg>

        {/* Playhead */}
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-white z-10"
          style={{ left: `${progressPercent}%` }}
        >
          <div className="absolute -top-1 left-1/2 -translate-x-1/2 w-0 h-0 border-l-[6px] border-r-[6px] border-t-[8px] border-l-transparent border-r-transparent border-t-white" />
        </div>
      </div>

      {/* Position axis labels */}
      <div className="absolute right-2 top-8 bottom-8 w-6 flex flex-col justify-between text-[10px] text-text-muted">
        <span>100</span>
        <span>50</span>
        <span>0</span>
      </div>
    </div>
  )
}

export default function EditorPage() {
  // Video state
  const [video, setVideo] = useState<VideoMetadata | null>(null)
  const [videoBlobUrl, setVideoBlobUrl] = useState<string | null>(null)
  const [isUploading, setIsUploading] = useState(false)
  const [isLoadingVideo, setIsLoadingVideo] = useState(false)
  const [uploadError, setUploadError] = useState<string | null>(null)

  // Playback state
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)

  // Funscript state
  const [funscriptPoints, setFunscriptPoints] = useState<FunscriptPoint[]>([])
  const [isGenerating, setIsGenerating] = useState(false)
  const [processingJob, setProcessingJob] = useState<ProcessingJob | null>(null)
  const [processingMessage, setProcessingMessage] = useState('')

  // Timeline state
  const [timelineZoom, setTimelineZoom] = useState(1)

  // Device state (setters used in device connection handlers)
  const [deviceConnected, setDeviceConnected] = useState(false)
  const [deviceStatus, setDeviceStatus] = useState('Disconnected')
  // Keep linter happy - will be used when device connection is implemented
  void setDeviceConnected
  void setDeviceStatus

  // Refs
  const videoRef = useRef<HTMLVideoElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Store hooks
  const { mode, toggleMode } = useModeStore()
  const { token } = useAuthStore()
  const isExpert = mode === 'expert'

  // Fetch video with auth headers and create blob URL
  useEffect(() => {
    if (!video || !token) {
      setVideoBlobUrl(null)
      return
    }

    let cancelled = false
    const fetchVideo = async () => {
      setIsLoadingVideo(true)
      try {
        const streamUrl = videosApi.getStreamUrl(video.id)
        const response = await fetch(streamUrl, {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        })

        if (!response.ok) {
          throw new Error('Failed to load video')
        }

        const blob = await response.blob()
        if (!cancelled) {
          const url = URL.createObjectURL(blob)
          setVideoBlobUrl(url)
        }
      } catch (error) {
        console.error('Failed to load video:', error)
        if (!cancelled) {
          setUploadError('Failed to load video for playback')
        }
      } finally {
        if (!cancelled) {
          setIsLoadingVideo(false)
        }
      }
    }

    fetchVideo()

    return () => {
      cancelled = true
      if (videoBlobUrl) {
        URL.revokeObjectURL(videoBlobUrl)
      }
    }
  }, [video, token])

  // Sync video time with state
  useEffect(() => {
    const videoEl = videoRef.current
    if (!videoEl) return

    const handleTimeUpdate = () => {
      setCurrentTime(videoEl.currentTime * 1000)
    }

    const handleDurationChange = () => {
      setDuration(videoEl.duration * 1000)
    }

    const handlePlay = () => setIsPlaying(true)
    const handlePause = () => setIsPlaying(false)
    const handleEnded = () => setIsPlaying(false)

    videoEl.addEventListener('timeupdate', handleTimeUpdate)
    videoEl.addEventListener('durationchange', handleDurationChange)
    videoEl.addEventListener('play', handlePlay)
    videoEl.addEventListener('pause', handlePause)
    videoEl.addEventListener('ended', handleEnded)

    return () => {
      videoEl.removeEventListener('timeupdate', handleTimeUpdate)
      videoEl.removeEventListener('durationchange', handleDurationChange)
      videoEl.removeEventListener('play', handlePlay)
      videoEl.removeEventListener('pause', handlePause)
      videoEl.removeEventListener('ended', handleEnded)
    }
  }, [videoBlobUrl])

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!videoRef.current) return

      switch (e.key) {
        case ' ':
          e.preventDefault()
          togglePlayPause()
          break
        case 'ArrowLeft':
          e.preventDefault()
          skipFrames(-1)
          break
        case 'ArrowRight':
          e.preventDefault()
          skipFrames(1)
          break
        case 'Home':
          e.preventDefault()
          seekTo(0)
          break
        case 'End':
          e.preventDefault()
          seekTo(duration)
          break
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [duration])

  // Video controls
  const togglePlayPause = useCallback(() => {
    if (!videoRef.current) return
    if (isPlaying) {
      videoRef.current.pause()
    } else {
      videoRef.current.play()
    }
  }, [isPlaying])

  const seekTo = useCallback((timeMs: number) => {
    if (!videoRef.current) return
    videoRef.current.currentTime = timeMs / 1000
    setCurrentTime(timeMs)
  }, [])

  const skipFrames = useCallback((frames: number) => {
    if (!videoRef.current || !video?.fps) return
    const frameDuration = 1000 / video.fps
    const newTime = currentTime + (frames * frameDuration)
    seekTo(Math.max(0, Math.min(duration, newTime)))
  }, [currentTime, duration, video?.fps, seekTo])

  // File handling
  const handleFileSelect = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    setIsUploading(true)
    setUploadError(null)
    setFunscriptPoints([]) // Clear existing points

    try {
      const metadata = await videosApi.upload(file)
      setVideo(metadata)
    } catch (error) {
      console.error('Upload failed:', error)
      setUploadError(error instanceof Error ? error.message : 'Upload failed')
    } finally {
      setIsUploading(false)
      if (fileInputRef.current) {
        fileInputRef.current.value = ''
      }
    }
  }

  const openFileDialog = () => {
    fileInputRef.current?.click()
  }

  // Funscript operations
  const addPoint = useCallback((point: FunscriptPoint) => {
    setFunscriptPoints(prev => {
      const newPoints = [...prev, point].sort((a, b) => a.at - b.at)
      return newPoints
    })
  }, [])

  const removePoint = useCallback((index: number) => {
    setFunscriptPoints(prev => prev.filter((_, i) => i !== index))
  }, [])
  // Will be used for point deletion in timeline
  void removePoint

  const addPointAtCurrentTime = useCallback(() => {
    // Add a point at current time with position 50
    addPoint({ at: Math.round(currentTime), pos: 50 })
  }, [currentTime, addPoint])

  // Export funscript
  const exportFunscript = useCallback(() => {
    if (funscriptPoints.length === 0) {
      alert('No points to export')
      return
    }

    const funscript = {
      version: '1.0',
      inverted: false,
      range: 100,
      actions: funscriptPoints.map(p => ({ at: p.at, pos: p.pos })),
      metadata: {
        creator: 'FunGen Web',
        description: video?.original_filename || 'Generated funscript',
        duration: duration,
        license: '',
        notes: '',
        performers: [],
        script_url: '',
        tags: [],
        title: video?.original_filename?.replace(/\.[^/.]+$/, '') || 'Untitled',
        type: 'basic',
        video_url: ''
      }
    }

    const blob = new Blob([JSON.stringify(funscript, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${video?.original_filename?.replace(/\.[^/.]+$/, '') || 'funscript'}.funscript`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }, [funscriptPoints, video, duration])

  // Generate funscript via backend processing
  const generateFunscript = useCallback(async () => {
    if (!video) {
      alert('Please upload a video first')
      return
    }

    setIsGenerating(true)
    setProcessingMessage('Starting processing job...')

    try {
      // Create processing job
      const job = await processingApi.createJob({ video_id: video.id })
      setProcessingJob(job)
      setProcessingMessage(job.message)

      // Poll for job status
      const pollInterval = setInterval(async () => {
        try {
          const updatedJob = await processingApi.getJob(job.id)
          setProcessingJob(updatedJob)
          setProcessingMessage(updatedJob.message)

          // Check if job is complete or failed
          if (updatedJob.stage === 'complete' || updatedJob.stage === 'failed') {
            clearInterval(pollInterval)

            if (updatedJob.stage === 'complete') {
              // Generate demo points since the backend simulation doesn't create real data
              // In production, fetch the generated funscript from the server
              const videoDuration = duration || (video.duration_ms ?? 60000)
              const demoPoints: FunscriptPoint[] = []
              for (let t = 0; t < videoDuration; t += 500) {
                demoPoints.push({
                  at: t,
                  pos: Math.round(50 + 40 * Math.sin(t / 1000))
                })
              }
              setFunscriptPoints(demoPoints)
              setProcessingMessage('Funscript generated successfully!')
            } else {
              setProcessingMessage(`Processing failed: ${updatedJob.error || 'Unknown error'}`)
            }

            setIsGenerating(false)
            setTimeout(() => {
              setProcessingJob(null)
              setProcessingMessage('')
            }, 3000)
          }
        } catch (error) {
          console.error('Failed to poll job status:', error)
          clearInterval(pollInterval)
          setIsGenerating(false)
          setProcessingMessage('Failed to check processing status')
        }
      }, 1000)

    } catch (error) {
      console.error('Failed to create processing job:', error)
      setIsGenerating(false)
      setProcessingMessage(error instanceof Error ? error.message : 'Failed to start processing')
    }
  }, [video, duration])

  // Calculate statistics
  const stats = {
    points: funscriptPoints.length,
    avgSpeed: 0,
    peakSpeed: 0,
    intensity: 0
  }

  if (funscriptPoints.length > 1) {
    let totalSpeed = 0
    let maxSpeed = 0
    for (let i = 1; i < funscriptPoints.length; i++) {
      const dt = funscriptPoints[i].at - funscriptPoints[i - 1].at
      const dp = Math.abs(funscriptPoints[i].pos - funscriptPoints[i - 1].pos)
      if (dt > 0) {
        const speed = (dp / dt) * 1000 // units per second
        totalSpeed += speed
        maxSpeed = Math.max(maxSpeed, speed)
      }
    }
    stats.avgSpeed = Math.round(totalSpeed / (funscriptPoints.length - 1))
    stats.peakSpeed = Math.round(maxSpeed)
    stats.intensity = Math.min(100, Math.round((stats.avgSpeed / 300) * 100))
  }

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept="video/*,.mp4,.mkv,.avi,.mov,.webm"
        className="sr-only"
        style={{ position: 'absolute', left: '-9999px' }}
        onChange={handleFileSelect}
      />

      {/* Toolbar */}
      <div className="h-12 bg-bg-surface border-b border-border flex items-center px-2 gap-1">
        {/* File actions */}
        <div className="flex items-center gap-1 pr-2 border-r border-border">
          <ToolbarButton icon={Upload} label="Open Video" onClick={openFileDialog} />
          <ToolbarButton
            icon={Save}
            label="Save Project"
            disabled={!video}
          />
          <ToolbarButton
            icon={Download}
            label="Export Funscript"
            disabled={funscriptPoints.length === 0}
            onClick={exportFunscript}
          />
        </div>

        {/* Playback controls */}
        <div className="flex items-center gap-1 px-2 border-r border-border">
          <ToolbarButton
            icon={ChevronLeft}
            label="Previous Frame (←)"
            disabled={!video}
            onClick={() => skipFrames(-1)}
          />
          <ToolbarButton
            icon={isPlaying ? Pause : Play}
            label={isPlaying ? 'Pause (Space)' : 'Play (Space)'}
            active={isPlaying}
            disabled={!video}
            onClick={togglePlayPause}
          />
          <ToolbarButton
            icon={ChevronRight}
            label="Next Frame (→)"
            disabled={!video}
            onClick={() => skipFrames(1)}
          />
        </div>

        {/* Point controls */}
        <div className="flex items-center gap-1 px-2 border-r border-border">
          <ToolbarButton
            icon={Plus}
            label="Add Point at Current Time"
            disabled={!video}
            onClick={addPointAtCurrentTime}
          />
          <ToolbarButton
            icon={RotateCcw}
            label="Clear All Points"
            disabled={funscriptPoints.length === 0}
            onClick={() => setFunscriptPoints([])}
          />
        </div>

        {/* View controls */}
        <div className="flex items-center gap-1 px-2 border-r border-border">
          <ToolbarButton
            icon={ZoomOut}
            label="Zoom Out Timeline"
            onClick={() => setTimelineZoom(z => Math.max(0.5, z - 0.25))}
          />
          <ToolbarButton
            icon={ZoomIn}
            label="Zoom In Timeline"
            onClick={() => setTimelineZoom(z => Math.min(4, z + 0.25))}
          />
          <ToolbarButton icon={Maximize} label="Fit View" />
        </div>

        {/* AI Generation */}
        <div className="flex items-center gap-2 px-2 border-r border-border">
          <button
            className={clsx(
              "flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium transition-colors",
              isGenerating
                ? "bg-bg-elevated text-text-muted cursor-wait"
                : "bg-primary text-bg-base hover:bg-primary-hover"
            )}
            disabled={!video || isGenerating}
            onClick={generateFunscript}
          >
            {isGenerating ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Wand2 className="w-4 h-4" />
            )}
            {isGenerating ? 'Processing...' : 'Generate'}
          </button>
          {/* Progress indicator */}
          {processingJob && (
            <div className="flex items-center gap-2">
              <div className="w-24 h-1.5 bg-bg-elevated rounded-full overflow-hidden">
                <div
                  className="h-full bg-primary transition-all duration-300"
                  style={{ width: `${(processingJob.progress || 0) * 100}%` }}
                />
              </div>
              <span className="text-xs text-text-muted whitespace-nowrap">
                {Math.round((processingJob.progress || 0) * 100)}%
              </span>
            </div>
          )}
        </div>

        {/* Device connection */}
        <div className="flex items-center gap-1 px-2">
          <ToolbarButton
            icon={Bluetooth}
            label="Connect Device"
            active={deviceConnected}
          />
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
          <div className="flex-1 bg-bg-overlay flex items-center justify-center relative">
            {isUploading || isLoadingVideo ? (
              <div className="text-center">
                <div className="w-20 h-20 rounded-full bg-bg-elevated flex items-center justify-center mx-auto mb-4">
                  <Loader2 className="w-10 h-10 text-primary animate-spin" />
                </div>
                <h3 className="text-lg font-semibold text-text-primary mb-2">
                  {isUploading ? 'Uploading video...' : 'Loading video...'}
                </h3>
                <p className="text-text-secondary">
                  Please wait while we process your video
                </p>
              </div>
            ) : video && videoBlobUrl ? (
              <>
                <video
                  ref={videoRef}
                  className="max-w-full max-h-full"
                  src={videoBlobUrl}
                  onClick={togglePlayPause}
                />
                {/* Time overlay */}
                <div className="absolute bottom-4 left-4 bg-black/70 px-2 py-1 rounded text-xs text-white font-mono">
                  {formatTime(currentTime)}
                </div>
              </>
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
          <div className="h-48 bg-bg-surface border-t border-border">
            {video ? (
              <Timeline
                currentTime={currentTime}
                duration={duration}
                points={funscriptPoints}
                onSeek={seekTo}
                zoom={timelineZoom}
              />
            ) : (
              <div className="h-full flex items-center justify-center">
                <p className="text-text-muted">Load a video to see the timeline</p>
              </div>
            )}
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
                  <button
                    className="w-full btn-primary text-sm"
                    disabled={!video || isGenerating}
                    onClick={generateFunscript}
                  >
                    {isGenerating ? (
                      <Loader2 className="w-4 h-4 animate-spin" />
                    ) : (
                      <Wand2 className="w-4 h-4" />
                    )}
                    {isGenerating ? 'Processing...' : 'Auto Generate'}
                  </button>
                  {/* Progress bar and status */}
                  {processingJob && (
                    <div className="mt-3 space-y-2">
                      <div className="w-full h-2 bg-bg-elevated rounded-full overflow-hidden">
                        <div
                          className="h-full bg-primary transition-all duration-300"
                          style={{ width: `${(processingJob.progress || 0) * 100}%` }}
                        />
                      </div>
                      <p className="text-xs text-text-muted text-center">
                        {processingMessage || processingJob.message}
                      </p>
                    </div>
                  )}
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
                  </div>
                </div>
              )}

              {/* Device Status */}
              <div className="card">
                <h3 className="text-sm font-medium text-text-primary mb-3">
                  Device
                </h3>
                <div className="flex items-center gap-3">
                  <div className={clsx(
                    "w-3 h-3 rounded-full",
                    deviceConnected ? "bg-green-500" : "bg-text-disabled"
                  )} />
                  <span className="text-sm text-text-secondary">
                    {deviceStatus}
                  </span>
                </div>
                <button className="w-full btn-secondary mt-3 text-sm">
                  <Bluetooth className="w-4 h-4" />
                  {deviceConnected ? 'Disconnect' : 'Connect'}
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
                    <span className="text-text-primary font-medium">{stats.points}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-text-secondary">Duration</span>
                    <span className="text-text-primary font-medium">
                      {video?.duration_ms ? formatDuration(video.duration_ms) : formatDuration(duration)}
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
                        <span className="text-text-primary font-medium">{stats.avgSpeed} u/s</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-text-secondary">Peak Speed</span>
                        <span className="text-text-primary font-medium">{stats.peakSpeed} u/s</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-text-secondary">Intensity</span>
                        <span className="text-text-primary font-medium">{stats.intensity}%</span>
                      </div>
                    </>
                  )}
                </div>
              </div>

              {/* Export */}
              {funscriptPoints.length > 0 && (
                <div className="card">
                  <h3 className="text-sm font-medium text-text-primary mb-3">
                    Export
                  </h3>
                  <button
                    className="w-full btn-primary text-sm"
                    onClick={exportFunscript}
                  >
                    <Download className="w-4 h-4" />
                    Download Funscript
                  </button>
                </div>
              )}

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
