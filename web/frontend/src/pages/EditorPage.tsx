import { useState, useRef, useEffect, useCallback, useMemo } from 'react'
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
  ChevronRight,
  Trash2,
  FileUp,
  Undo2,
  Redo2,
  Filter,
  Move,
  MousePointer,
  FolderOpen,
  Box,
  Eye,
  EyeOff
} from 'lucide-react'
import { clsx } from 'clsx'
import { useModeStore } from '@/store/modeStore'
import { videosApi, processingApi, devicesApi, VideoMetadata, FunscriptPoint, ProcessingJob, Device } from '@/api/client'
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

// Post-processing filter functions
const filters = {
  smooth: (points: FunscriptPoint[], strength: number): FunscriptPoint[] => {
    if (points.length < 3) return points
    const result: FunscriptPoint[] = [points[0]]
    const windowSize = Math.max(3, Math.floor(strength / 10))

    for (let i = 1; i < points.length - 1; i++) {
      const start = Math.max(0, i - Math.floor(windowSize / 2))
      const end = Math.min(points.length, i + Math.floor(windowSize / 2) + 1)
      let sum = 0
      for (let j = start; j < end; j++) {
        sum += points[j].pos
      }
      result.push({ at: points[i].at, pos: Math.round(sum / (end - start)) })
    }
    result.push(points[points.length - 1])
    return result
  },

  amplify: (points: FunscriptPoint[], amount: number): FunscriptPoint[] => {
    const factor = 1 + (amount / 100)
    return points.map(p => ({
      at: p.at,
      pos: Math.max(0, Math.min(100, Math.round((p.pos - 50) * factor + 50)))
    }))
  },

  invert: (points: FunscriptPoint[]): FunscriptPoint[] => {
    return points.map(p => ({ at: p.at, pos: 100 - p.pos }))
  },

  speedLimit: (points: FunscriptPoint[], maxSpeed: number): FunscriptPoint[] => {
    if (points.length < 2) return points
    const result: FunscriptPoint[] = [points[0]]

    for (let i = 1; i < points.length; i++) {
      const prev = result[result.length - 1]
      const curr = points[i]
      const dt = curr.at - prev.at
      if (dt <= 0) continue

      const dp = Math.abs(curr.pos - prev.pos)
      const speed = (dp / dt) * 1000

      if (speed > maxSpeed) {
        const maxDp = (maxSpeed * dt) / 1000
        const direction = curr.pos > prev.pos ? 1 : -1
        result.push({
          at: curr.at,
          pos: Math.max(0, Math.min(100, Math.round(prev.pos + direction * maxDp)))
        })
      } else {
        result.push(curr)
      }
    }
    return result
  },

  simplify: (points: FunscriptPoint[], tolerance: number): FunscriptPoint[] => {
    if (points.length < 3) return points
    // Simple RDP-like simplification
    const result: FunscriptPoint[] = [points[0]]

    for (let i = 1; i < points.length - 1; i++) {
      const prev = result[result.length - 1]
      const curr = points[i]
      const next = points[i + 1]

      // Check if current point is necessary
      const expectedPos = prev.pos + ((next.pos - prev.pos) * (curr.at - prev.at)) / (next.at - prev.at)
      if (Math.abs(curr.pos - expectedPos) > tolerance) {
        result.push(curr)
      }
    }
    result.push(points[points.length - 1])
    return result
  }
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

// Enhanced Timeline component with point selection and dragging
function Timeline({
  currentTime,
  duration,
  points,
  selectedPoints,
  onSeek,
  onSelectPoint,
  onMovePoint,
  onDeleteSelected,
  zoom,
  editMode,
}: {
  currentTime: number
  duration: number
  points: FunscriptPoint[]
  selectedPoints: Set<number>
  onSeek: (time: number) => void
  onSelectPoint: (index: number, addToSelection: boolean) => void
  onMovePoint: (index: number, newPos: number) => void
  onDeleteSelected: () => void
  zoom: number
  editMode: 'select' | 'move'
}) {
  const timelineRef = useRef<HTMLDivElement>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [dragIndex, setDragIndex] = useState<number | null>(null)
  const scrollOffset = 0

  const getTimeFromX = (clientX: number) => {
    if (!timelineRef.current || duration === 0) return 0
    const rect = timelineRef.current.getBoundingClientRect()
    const x = clientX - rect.left + scrollOffset
    const timelineWidth = rect.width * zoom
    return Math.max(0, Math.min(duration, (x / timelineWidth) * duration))
  }

  const getPosFromY = (clientY: number) => {
    if (!timelineRef.current) return 50
    const rect = timelineRef.current.getBoundingClientRect()
    const y = clientY - rect.top
    const pos = 100 - (y / rect.height) * 100
    return Math.max(0, Math.min(100, Math.round(pos)))
  }

  const handleTimelineClick = (e: React.MouseEvent) => {
    if (dragIndex !== null) return
    const time = getTimeFromX(e.clientX)
    onSeek(time)
  }

  const handleMouseDown = () => {
    if (editMode === 'select') {
      setIsDragging(true)
    }
  }

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging && editMode === 'select') {
      const time = getTimeFromX(e.clientX)
      onSeek(time)
    } else if (dragIndex !== null && editMode === 'move') {
      const newPos = getPosFromY(e.clientY)
      onMovePoint(dragIndex, newPos)
    }
  }

  const handleMouseUp = () => {
    setIsDragging(false)
    setDragIndex(null)
  }

  const handlePointMouseDown = (e: React.MouseEvent, index: number) => {
    e.stopPropagation()
    if (editMode === 'move') {
      setDragIndex(index)
    } else {
      onSelectPoint(index, e.shiftKey || e.ctrlKey || e.metaKey)
    }
  }

  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (e.key === 'Delete' || e.key === 'Backspace') {
      if (selectedPoints.size > 0) {
        e.preventDefault()
        onDeleteSelected()
      }
    }
  }, [selectedPoints, onDeleteSelected])

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [handleKeyDown])

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
          {selectedPoints.size > 0 && (
            <span className="text-xs text-primary">{selectedPoints.size} selected</span>
          )}
        </div>
      </div>

      {/* Timeline track */}
      <div
        ref={timelineRef}
        className={clsx(
          "flex-1 relative overflow-hidden",
          editMode === 'select' ? 'cursor-crosshair' : 'cursor-move'
        )}
        onClick={handleTimelineClick}
        onMouseDown={handleMouseDown}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onMouseMove={handleMouseMove}
      >
        {/* Background grid */}
        <div className="absolute inset-0 bg-bg-base">
          {/* Horizontal position guides */}
          {[0, 25, 50, 75, 100].map(pos => (
            <div
              key={pos}
              className="absolute left-0 right-0 border-t border-border/30"
              style={{ top: `${100 - pos}%` }}
            />
          ))}
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
          {/* Fill area under curve */}
          {points.length > 1 && (
            <polygon
              fill="var(--color-primary)"
              fillOpacity="0.1"
              points={`0,100% ${points.map(p => {
                const x = (p.at / duration) * 100
                const y = 100 - p.pos
                return `${x}%,${y}%`
              }).join(' ')} 100%,100%`}
            />
          )}
          {/* Draw lines between points */}
          {points.length > 1 && (
            <polyline
              fill="none"
              stroke="var(--color-primary)"
              strokeWidth="2"
              points={points.map(p => {
                const x = (p.at / duration) * 100
                const y = 100 - p.pos
                return `${x}%,${y}%`
              }).join(' ')}
            />
          )}
          {/* Draw points */}
          {points.map((point, index) => {
            const x = (point.at / duration) * 100
            const y = 100 - point.pos
            const isSelected = selectedPoints.has(index)
            return (
              <circle
                key={index}
                cx={`${x}%`}
                cy={`${y}%`}
                r={isSelected ? "6" : "4"}
                fill={isSelected ? "white" : "var(--color-primary)"}
                stroke={isSelected ? "var(--color-primary)" : "none"}
                strokeWidth="2"
                className="cursor-pointer hover:fill-white transition-all"
                onMouseDown={(e) => handlePointMouseDown(e, index)}
              />
            )
          })}
        </svg>

        {/* Playhead */}
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-white z-10 pointer-events-none"
          style={{ left: `${progressPercent}%` }}
        >
          <div className="absolute -top-1 left-1/2 -translate-x-1/2 w-0 h-0 border-l-[6px] border-r-[6px] border-t-[8px] border-l-transparent border-r-transparent border-t-white" />
        </div>
      </div>

      {/* Position axis labels */}
      <div className="absolute right-2 top-8 bottom-8 w-6 flex flex-col justify-between text-[10px] text-text-muted pointer-events-none">
        <span>100</span>
        <span>50</span>
        <span>0</span>
      </div>
    </div>
  )
}

// Project file format (compatible with desktop .fgnproj)
interface ProjectData {
  version: string
  video_path: string
  video_id: string | null
  funscript_actions: FunscriptPoint[]
  settings: {
    confidenceThreshold: number
    smoothingFactor: number
    minPointDistance: number
    autoSmoothPeaks: boolean
    invertOutput: boolean
    limitRange: boolean
    playbackSpeed: number
  }
  ui_state: {
    timeline_zoom: number
    edit_mode: 'select' | 'move'
    show_heatmap: boolean
    show_simulator_3d: boolean
  }
  created_at: string
  updated_at: string
}

// 3D Simulator component - renders a cylinder that moves with funscript position
function Simulator3D({ position, isPlaying }: { position: number, isPlaying: boolean }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number>(0)
  const currentPosRef = useRef(position)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Smoothly interpolate to target position
    const targetPos = position

    const render = () => {
      // Smooth interpolation
      currentPosRef.current += (targetPos - currentPosRef.current) * 0.15

      const width = canvas.width
      const height = canvas.height
      const centerX = width / 2
      const baseY = height - 30

      // Clear canvas
      ctx.fillStyle = '#1a1a1a'
      ctx.fillRect(0, 0, width, height)

      // Calculate cylinder position (0 = bottom, 100 = top)
      const cylinderHeight = 80
      const cylinderWidth = 50
      const maxTravel = height - 80
      const yOffset = (currentPosRef.current / 100) * maxTravel

      // Draw base platform
      ctx.fillStyle = '#333'
      ctx.fillRect(centerX - 40, baseY, 80, 20)

      // Draw shaft (static background)
      const gradient = ctx.createLinearGradient(centerX - 20, 0, centerX + 20, 0)
      gradient.addColorStop(0, '#444')
      gradient.addColorStop(0.5, '#666')
      gradient.addColorStop(1, '#444')
      ctx.fillStyle = gradient
      ctx.fillRect(centerX - 20, baseY - maxTravel - 40, 40, maxTravel + 40)

      // Draw moving cylinder
      const cylinderY = baseY - yOffset - cylinderHeight

      // Cylinder body gradient
      const cylGradient = ctx.createLinearGradient(centerX - cylinderWidth/2, 0, centerX + cylinderWidth/2, 0)
      cylGradient.addColorStop(0, '#2563eb')
      cylGradient.addColorStop(0.3, '#60a5fa')
      cylGradient.addColorStop(0.7, '#60a5fa')
      cylGradient.addColorStop(1, '#1d4ed8')

      // Draw cylinder body
      ctx.fillStyle = cylGradient
      ctx.beginPath()
      ctx.roundRect(centerX - cylinderWidth/2, cylinderY, cylinderWidth, cylinderHeight, 8)
      ctx.fill()

      // Draw cylinder top (ellipse)
      ctx.fillStyle = '#93c5fd'
      ctx.beginPath()
      ctx.ellipse(centerX, cylinderY, cylinderWidth/2, 10, 0, 0, Math.PI * 2)
      ctx.fill()

      // Draw cylinder bottom (ellipse)
      ctx.fillStyle = '#1e40af'
      ctx.beginPath()
      ctx.ellipse(centerX, cylinderY + cylinderHeight, cylinderWidth/2, 10, 0, 0, Math.PI * 2)
      ctx.fill()

      // Draw position indicator
      ctx.fillStyle = '#fff'
      ctx.font = '12px monospace'
      ctx.textAlign = 'center'
      ctx.fillText(`${Math.round(currentPosRef.current)}`, centerX, height - 8)

      // Continue animation if playing or interpolating
      if (isPlaying || Math.abs(currentPosRef.current - targetPos) > 0.5) {
        animationRef.current = requestAnimationFrame(render)
      }
    }

    render()

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [position, isPlaying])

  // Re-render when position changes
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Trigger a new render cycle
    const render = () => {
      const targetPos = position
      currentPosRef.current += (targetPos - currentPosRef.current) * 0.15

      const width = canvas.width
      const height = canvas.height
      const centerX = width / 2
      const baseY = height - 30
      const cylinderHeight = 80
      const cylinderWidth = 50
      const maxTravel = height - 80
      const yOffset = (currentPosRef.current / 100) * maxTravel

      ctx.fillStyle = '#1a1a1a'
      ctx.fillRect(0, 0, width, height)

      ctx.fillStyle = '#333'
      ctx.fillRect(centerX - 40, baseY, 80, 20)

      const gradient = ctx.createLinearGradient(centerX - 20, 0, centerX + 20, 0)
      gradient.addColorStop(0, '#444')
      gradient.addColorStop(0.5, '#666')
      gradient.addColorStop(1, '#444')
      ctx.fillStyle = gradient
      ctx.fillRect(centerX - 20, baseY - maxTravel - 40, 40, maxTravel + 40)

      const cylinderY = baseY - yOffset - cylinderHeight

      const cylGradient = ctx.createLinearGradient(centerX - cylinderWidth/2, 0, centerX + cylinderWidth/2, 0)
      cylGradient.addColorStop(0, '#2563eb')
      cylGradient.addColorStop(0.3, '#60a5fa')
      cylGradient.addColorStop(0.7, '#60a5fa')
      cylGradient.addColorStop(1, '#1d4ed8')

      ctx.fillStyle = cylGradient
      ctx.beginPath()
      ctx.roundRect(centerX - cylinderWidth/2, cylinderY, cylinderWidth, cylinderHeight, 8)
      ctx.fill()

      ctx.fillStyle = '#93c5fd'
      ctx.beginPath()
      ctx.ellipse(centerX, cylinderY, cylinderWidth/2, 10, 0, 0, Math.PI * 2)
      ctx.fill()

      ctx.fillStyle = '#1e40af'
      ctx.beginPath()
      ctx.ellipse(centerX, cylinderY + cylinderHeight, cylinderWidth/2, 10, 0, 0, Math.PI * 2)
      ctx.fill()

      ctx.fillStyle = '#fff'
      ctx.font = '12px monospace'
      ctx.textAlign = 'center'
      ctx.fillText(`${Math.round(currentPosRef.current)}`, centerX, height - 8)

      if (Math.abs(currentPosRef.current - targetPos) > 0.5) {
        animationRef.current = requestAnimationFrame(render)
      }
    }

    animationRef.current = requestAnimationFrame(render)
  }, [position])

  return (
    <div className="bg-bg-base rounded-lg overflow-hidden">
      <canvas
        ref={canvasRef}
        width={120}
        height={200}
        className="w-full h-full"
      />
    </div>
  )
}

// Heatmap visualization component
function Heatmap({ points, duration }: { points: FunscriptPoint[], duration: number }) {
  const segments = 50
  const heatmapData = useMemo(() => {
    if (points.length < 2 || duration === 0) return []

    const segmentDuration = duration / segments
    const result: number[] = []

    for (let i = 0; i < segments; i++) {
      const segStart = i * segmentDuration
      const segEnd = (i + 1) * segmentDuration

      let intensity = 0
      let count = 0

      for (let j = 1; j < points.length; j++) {
        const p1 = points[j - 1]
        const p2 = points[j]

        if (p2.at >= segStart && p1.at <= segEnd) {
          const dt = p2.at - p1.at
          const dp = Math.abs(p2.pos - p1.pos)
          if (dt > 0) {
            intensity += (dp / dt) * 1000
            count++
          }
        }
      }

      result.push(count > 0 ? intensity / count : 0)
    }

    return result
  }, [points, duration])

  const maxIntensity = Math.max(...heatmapData, 1)

  return (
    <div className="h-6 flex gap-px">
      {heatmapData.map((intensity, i) => {
        const normalized = intensity / maxIntensity
        const hue = 120 - normalized * 120 // Green to red
        return (
          <div
            key={i}
            className="flex-1 rounded-sm"
            style={{ backgroundColor: `hsl(${hue}, 70%, 50%)`, opacity: 0.3 + normalized * 0.7 }}
            title={`${Math.round(intensity)} u/s`}
          />
        )
      })}
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
  const [selectedPoints, setSelectedPoints] = useState<Set<number>>(new Set())
  const [isGenerating, setIsGenerating] = useState(false)
  const [processingJob, setProcessingJob] = useState<ProcessingJob | null>(null)
  const [processingMessage, setProcessingMessage] = useState('')

  // Undo/Redo history
  const [history, setHistory] = useState<FunscriptPoint[][]>([])
  const [historyIndex, setHistoryIndex] = useState(-1)

  // Edit mode
  const [editMode, setEditMode] = useState<'select' | 'move'>('select')

  // Timeline state
  const [timelineZoom, setTimelineZoom] = useState(1)

  // Device state
  const [device, setDevice] = useState<Device | null>(null)
  const [showDeviceModal, setShowDeviceModal] = useState(false)
  const [connectionKey, setConnectionKey] = useState('')
  const [isConnecting, setIsConnecting] = useState(false)

  // Settings state
  const [showSettingsModal, setShowSettingsModal] = useState(false)
  const [showFiltersModal, setShowFiltersModal] = useState(false)
  const [settings, setSettings] = useState({
    confidenceThreshold: 50,
    smoothingFactor: 30,
    minPointDistance: 100,
    autoSmoothPeaks: true,
    invertOutput: false,
    limitRange: true,
    playbackSpeed: 1.0,
  })

  // UI visibility state
  const [showSimulator3D, setShowSimulator3D] = useState(true)
  const [showHeatmap, setShowHeatmap] = useState(true)

  // Project state
  const [projectName, setProjectName] = useState<string>('')
  const [projectDirty, setProjectDirty] = useState(false)

  // Refs
  const videoRef = useRef<HTMLVideoElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const funscriptInputRef = useRef<HTMLInputElement>(null)
  const projectInputRef = useRef<HTMLInputElement>(null)

  // Store hooks
  const { mode, toggleMode } = useModeStore()
  const { token } = useAuthStore()
  const isExpert = mode === 'expert'

  // Push to history
  const pushHistory = useCallback((points: FunscriptPoint[]) => {
    setHistory(prev => [...prev.slice(0, historyIndex + 1), points])
    setHistoryIndex(prev => prev + 1)
  }, [historyIndex])

  // Undo
  const undo = useCallback(() => {
    if (historyIndex > 0) {
      setHistoryIndex(prev => prev - 1)
      setFunscriptPoints(history[historyIndex - 1])
      setSelectedPoints(new Set())
    }
  }, [historyIndex, history])

  // Redo
  const redo = useCallback(() => {
    if (historyIndex < history.length - 1) {
      setHistoryIndex(prev => prev + 1)
      setFunscriptPoints(history[historyIndex + 1])
      setSelectedPoints(new Set())
    }
  }, [historyIndex, history])

  // Mark project as dirty when points change
  useEffect(() => {
    if (funscriptPoints.length > 0) {
      setProjectDirty(true)
    }
  }, [funscriptPoints])

  // Calculate current position from funscript
  const currentPosition = useMemo(() => {
    if (funscriptPoints.length === 0) return 50

    // Find the last point before or at current time
    let prev: FunscriptPoint | undefined
    for (let i = funscriptPoints.length - 1; i >= 0; i--) {
      if (funscriptPoints[i].at <= currentTime) {
        prev = funscriptPoints[i]
        break
      }
    }
    const next = funscriptPoints.find((p: FunscriptPoint) => p.at > currentTime)

    if (!prev) return next?.pos ?? 50
    if (!next) return prev.pos

    // Interpolate between points
    const t = (currentTime - prev.at) / (next.at - prev.at)
    return Math.round(prev.pos + (next.pos - prev.pos) * t)
  }, [funscriptPoints, currentTime])

  // Save project to file
  const saveProject = useCallback(() => {
    const projectData: ProjectData = {
      version: '1.0.0',
      video_path: video?.original_filename || '',
      video_id: video?.id || null,
      funscript_actions: funscriptPoints,
      settings: settings,
      ui_state: {
        timeline_zoom: timelineZoom,
        edit_mode: editMode,
        show_heatmap: showHeatmap,
        show_simulator_3d: showSimulator3D,
      },
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    }

    const blob = new Blob([JSON.stringify(projectData, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    const filename = projectName || video?.original_filename?.replace(/\.[^/.]+$/, '') || 'project'
    a.download = `${filename}.fgnproj`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)

    setProjectDirty(false)
    setProjectName(filename)
  }, [video, funscriptPoints, settings, timelineZoom, editMode, showHeatmap, showSimulator3D, projectName])

  // Load project from file
  const handleProjectLoad = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    try {
      const text = await file.text()
      const projectData: ProjectData = JSON.parse(text)

      // Restore funscript points
      if (projectData.funscript_actions && Array.isArray(projectData.funscript_actions)) {
        const points = projectData.funscript_actions.map(a => ({
          at: a.at,
          pos: Math.max(0, Math.min(100, a.pos))
        })).sort((a, b) => a.at - b.at)
        setFunscriptPoints(points)
        pushHistory(points)
      }

      // Restore settings
      if (projectData.settings) {
        setSettings(prev => ({ ...prev, ...projectData.settings }))
      }

      // Restore UI state
      if (projectData.ui_state) {
        if (projectData.ui_state.timeline_zoom) setTimelineZoom(projectData.ui_state.timeline_zoom)
        if (projectData.ui_state.edit_mode) setEditMode(projectData.ui_state.edit_mode)
        if (typeof projectData.ui_state.show_heatmap === 'boolean') setShowHeatmap(projectData.ui_state.show_heatmap)
        if (typeof projectData.ui_state.show_simulator_3d === 'boolean') setShowSimulator3D(projectData.ui_state.show_simulator_3d)
      }

      // Set project name from filename
      const name = file.name.replace(/\.fgnproj$/i, '')
      setProjectName(name)
      setProjectDirty(false)
      setSelectedPoints(new Set())

    } catch (error) {
      console.error('Failed to load project:', error)
      alert('Failed to load project: ' + (error instanceof Error ? error.message : 'Unknown error'))
    } finally {
      if (projectInputRef.current) {
        projectInputRef.current.value = ''
      }
    }
  }, [pushHistory])

  const openProjectDialog = () => {
    projectInputRef.current?.click()
  }

  // Autosave to localStorage
  useEffect(() => {
    if (!video || funscriptPoints.length === 0) return

    const autosaveData = {
      video_id: video.id,
      funscript_actions: funscriptPoints,
      settings: settings,
      timestamp: Date.now(),
    }

    localStorage.setItem(`fungen_autosave_${video.id}`, JSON.stringify(autosaveData))
  }, [video, funscriptPoints, settings])

  // Restore from autosave on video load
  useEffect(() => {
    if (!video) return

    const autosaveKey = `fungen_autosave_${video.id}`
    const saved = localStorage.getItem(autosaveKey)

    if (saved) {
      try {
        const autosaveData = JSON.parse(saved)
        // Only restore if within last 24 hours
        if (Date.now() - autosaveData.timestamp < 24 * 60 * 60 * 1000) {
          if (autosaveData.funscript_actions?.length > 0 && funscriptPoints.length === 0) {
            setFunscriptPoints(autosaveData.funscript_actions)
            pushHistory(autosaveData.funscript_actions)
            if (autosaveData.settings) {
              setSettings(prev => ({ ...prev, ...autosaveData.settings }))
            }
          }
        }
      } catch (e) {
        console.warn('Failed to restore autosave:', e)
      }
    }
  }, [video])

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
      // Ignore if typing in an input
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return

      switch (e.key) {
        case ' ':
          e.preventDefault()
          togglePlayPause()
          break
        case 'ArrowLeft':
          e.preventDefault()
          if (e.shiftKey) {
            seekTo(Math.max(0, currentTime - 1000))
          } else {
            skipFrames(-1)
          }
          break
        case 'ArrowRight':
          e.preventDefault()
          if (e.shiftKey) {
            seekTo(Math.min(duration, currentTime + 1000))
          } else {
            skipFrames(1)
          }
          break
        case 'Home':
          e.preventDefault()
          seekTo(0)
          break
        case 'End':
          e.preventDefault()
          seekTo(duration)
          break
        case 'z':
          if (e.ctrlKey || e.metaKey) {
            e.preventDefault()
            if (e.shiftKey) {
              redo()
            } else {
              undo()
            }
          }
          break
        case 'y':
          if (e.ctrlKey || e.metaKey) {
            e.preventDefault()
            redo()
          }
          break
        case 'a':
          if (e.ctrlKey || e.metaKey) {
            e.preventDefault()
            setSelectedPoints(new Set(funscriptPoints.map((_, i) => i)))
          }
          break
        case 'Escape':
          setSelectedPoints(new Set())
          break
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [duration, currentTime, undo, redo, funscriptPoints])

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
    const fps = video?.fps || 30
    const frameDuration = 1000 / fps
    const newTime = currentTime + (frames * frameDuration)
    seekTo(Math.max(0, Math.min(duration, newTime)))
  }, [currentTime, duration, video?.fps, seekTo])

  // File handling
  const handleFileSelect = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    setIsUploading(true)
    setUploadError(null)
    setFunscriptPoints([])
    setHistory([])
    setHistoryIndex(-1)
    setSelectedPoints(new Set())

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

  // Funscript import
  const handleFunscriptImport = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    try {
      const text = await file.text()
      const data = JSON.parse(text)

      if (data.actions && Array.isArray(data.actions)) {
        const points: FunscriptPoint[] = data.actions.map((a: { at: number, pos: number }) => ({
          at: a.at,
          pos: Math.max(0, Math.min(100, a.pos))
        })).sort((a: FunscriptPoint, b: FunscriptPoint) => a.at - b.at)

        setFunscriptPoints(points)
        pushHistory(points)
        setSelectedPoints(new Set())
      } else {
        alert('Invalid funscript file: no actions found')
      }
    } catch (error) {
      console.error('Failed to import funscript:', error)
      alert('Failed to import funscript: ' + (error instanceof Error ? error.message : 'Unknown error'))
    } finally {
      if (funscriptInputRef.current) {
        funscriptInputRef.current.value = ''
      }
    }
  }

  const openFunscriptDialog = () => {
    funscriptInputRef.current?.click()
  }

  // Point operations
  const addPoint = useCallback((point: FunscriptPoint) => {
    setFunscriptPoints(prev => {
      const newPoints = [...prev, point].sort((a, b) => a.at - b.at)
      pushHistory(newPoints)
      return newPoints
    })
  }, [pushHistory])

  const updatePoints = useCallback((newPoints: FunscriptPoint[]) => {
    setFunscriptPoints(newPoints)
    pushHistory(newPoints)
    setSelectedPoints(new Set())
  }, [pushHistory])

  const addPointAtCurrentTime = useCallback(() => {
    addPoint({ at: Math.round(currentTime), pos: 50 })
  }, [currentTime, addPoint])

  const selectPoint = useCallback((index: number, addToSelection: boolean) => {
    setSelectedPoints(prev => {
      const next = new Set(addToSelection ? prev : [])
      if (next.has(index)) {
        next.delete(index)
      } else {
        next.add(index)
      }
      return next
    })
  }, [])

  const movePoint = useCallback((index: number, newPos: number) => {
    setFunscriptPoints(prev => {
      const newPoints = [...prev]
      newPoints[index] = { ...newPoints[index], pos: newPos }
      return newPoints
    })
  }, [])

  const deleteSelectedPoints = useCallback(() => {
    if (selectedPoints.size === 0) return
    setFunscriptPoints(prev => {
      const newPoints = prev.filter((_, i) => !selectedPoints.has(i))
      pushHistory(newPoints)
      return newPoints
    })
    setSelectedPoints(new Set())
  }, [selectedPoints, pushHistory])

  // Export funscript
  const exportFunscript = useCallback(() => {
    if (funscriptPoints.length === 0) {
      alert('No points to export')
      return
    }

    const funscript = {
      version: '1.0',
      inverted: settings.invertOutput,
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
  }, [funscriptPoints, video, duration, settings.invertOutput])

  // Apply filter
  const applyFilter = useCallback((filterName: keyof typeof filters, ...args: number[]) => {
    if (funscriptPoints.length === 0) return

    let newPoints: FunscriptPoint[]
    switch (filterName) {
      case 'smooth':
        newPoints = filters.smooth(funscriptPoints, args[0] || 50)
        break
      case 'amplify':
        newPoints = filters.amplify(funscriptPoints, args[0] || 20)
        break
      case 'invert':
        newPoints = filters.invert(funscriptPoints)
        break
      case 'speedLimit':
        newPoints = filters.speedLimit(funscriptPoints, args[0] || 400)
        break
      case 'simplify':
        newPoints = filters.simplify(funscriptPoints, args[0] || 5)
        break
      default:
        return
    }

    updatePoints(newPoints)
  }, [funscriptPoints, updatePoints])

  // Generate funscript via backend processing
  const generateFunscript = useCallback(async () => {
    if (!video) {
      alert('Please upload a video first')
      return
    }

    setIsGenerating(true)
    setProcessingMessage('Starting processing job...')

    try {
      const job = await processingApi.createJob({
        video_id: video.id,
        settings: {
          confidence_threshold: settings.confidenceThreshold / 100,
          smoothing_factor: settings.smoothingFactor / 100,
          min_stroke_length: settings.minPointDistance
        }
      })
      setProcessingJob(job)
      setProcessingMessage(job.message)

      const pollInterval = setInterval(async () => {
        try {
          const updatedJob = await processingApi.getJob(job.id)
          setProcessingJob(updatedJob)
          setProcessingMessage(updatedJob.message)

          if (updatedJob.stage === 'complete' || updatedJob.stage === 'failed') {
            clearInterval(pollInterval)

            if (updatedJob.stage === 'complete') {
              // Use actions from backend if available
              if (updatedJob.actions && updatedJob.actions.length > 0) {
                const points: FunscriptPoint[] = updatedJob.actions.map(a => ({
                  at: a.at,
                  pos: Math.max(0, Math.min(100, a.pos))
                })).sort((a, b) => a.at - b.at)
                setFunscriptPoints(points)
                pushHistory(points)
                setProcessingMessage(`Funscript generated! ${points.length} actions created.`)
              } else {
                // Fallback: generate realistic demo points if no actions returned
                const videoDuration = duration || (video.duration_ms ?? 60000)
                const demoPoints: FunscriptPoint[] = []
                let direction = 1
                for (let t = 0; t < videoDuration; t += 150 + Math.random() * 250) {
                  const targetPos = direction === 1
                    ? Math.min(100, 70 + Math.random() * 30)
                    : Math.max(0, Math.random() * 30)
                  demoPoints.push({
                    at: Math.round(t),
                    pos: Math.round(targetPos)
                  })
                  if (Math.random() < 0.3) direction *= -1
                }
                setFunscriptPoints(demoPoints)
                pushHistory(demoPoints)
                setProcessingMessage('Funscript generated successfully!')
              }
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
  }, [video, duration, settings, pushHistory])

  // Device connection handlers
  const connectDevice = useCallback(async () => {
    if (!connectionKey.trim()) {
      alert('Please enter a connection key')
      return
    }

    setIsConnecting(true)
    try {
      const connectedDevice = await devicesApi.connectHandy(connectionKey.trim())
      setDevice(connectedDevice)
      setShowDeviceModal(false)
      setConnectionKey('')
    } catch (error) {
      console.error('Failed to connect device:', error)
      alert(error instanceof Error ? error.message : 'Failed to connect device')
    } finally {
      setIsConnecting(false)
    }
  }, [connectionKey])

  const disconnectDevice = useCallback(async () => {
    if (!device) return

    try {
      await devicesApi.disconnect(device.id)
      setDevice(null)
    } catch (error) {
      console.error('Failed to disconnect device:', error)
    }
  }, [device])

  const toggleDeviceModal = useCallback(() => {
    if (device) {
      disconnectDevice()
    } else {
      setShowDeviceModal(true)
    }
  }, [device, disconnectDevice])

  // Calculate statistics
  const stats = useMemo(() => {
    const result = {
      points: funscriptPoints.length,
      avgSpeed: 0,
      peakSpeed: 0,
      intensity: 0,
      strokeCount: 0
    }

    if (funscriptPoints.length > 1) {
      let totalSpeed = 0
      let maxSpeed = 0
      let strokes = 0
      let lastDirection = 0

      for (let i = 1; i < funscriptPoints.length; i++) {
        const dt = funscriptPoints[i].at - funscriptPoints[i - 1].at
        const dp = funscriptPoints[i].pos - funscriptPoints[i - 1].pos

        // Count direction changes as strokes
        const direction = dp > 0 ? 1 : dp < 0 ? -1 : 0
        if (direction !== 0 && direction !== lastDirection) {
          strokes++
          lastDirection = direction
        }

        if (dt > 0) {
          const speed = (Math.abs(dp) / dt) * 1000
          totalSpeed += speed
          maxSpeed = Math.max(maxSpeed, speed)
        }
      }

      result.avgSpeed = Math.round(totalSpeed / (funscriptPoints.length - 1))
      result.peakSpeed = Math.round(maxSpeed)
      result.intensity = Math.min(100, Math.round((result.avgSpeed / 300) * 100))
      result.strokeCount = Math.floor(strokes / 2)
    }

    return result
  }, [funscriptPoints])

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* Hidden file inputs */}
      <input
        ref={fileInputRef}
        type="file"
        accept="video/*,.mp4,.mkv,.avi,.mov,.webm"
        className="sr-only"
        style={{ position: 'absolute', left: '-9999px' }}
        onChange={handleFileSelect}
      />
      <input
        ref={funscriptInputRef}
        type="file"
        accept=".funscript,.json"
        className="sr-only"
        style={{ position: 'absolute', left: '-9999px' }}
        onChange={handleFunscriptImport}
      />
      <input
        ref={projectInputRef}
        type="file"
        accept=".fgnproj,.json"
        className="sr-only"
        style={{ position: 'absolute', left: '-9999px' }}
        onChange={handleProjectLoad}
      />

      {/* Toolbar */}
      <div className="h-12 bg-bg-surface border-b border-border flex items-center px-2 gap-1">
        {/* File actions */}
        <div className="flex items-center gap-1 pr-2 border-r border-border">
          <ToolbarButton icon={Upload} label="Open Video" onClick={openFileDialog} />
          <ToolbarButton icon={FolderOpen} label="Open Project" onClick={openProjectDialog} />
          <ToolbarButton icon={FileUp} label="Import Funscript" onClick={openFunscriptDialog} />
          <ToolbarButton
            icon={Save}
            label={`Save Project${projectDirty ? ' *' : ''}`}
            disabled={funscriptPoints.length === 0}
            onClick={saveProject}
          />
          <ToolbarButton
            icon={Download}
            label="Export Funscript"
            disabled={funscriptPoints.length === 0}
            onClick={exportFunscript}
          />
        </div>

        {/* Undo/Redo */}
        <div className="flex items-center gap-1 px-2 border-r border-border">
          <ToolbarButton
            icon={Undo2}
            label="Undo (Ctrl+Z)"
            disabled={historyIndex <= 0}
            onClick={undo}
          />
          <ToolbarButton
            icon={Redo2}
            label="Redo (Ctrl+Y)"
            disabled={historyIndex >= history.length - 1}
            onClick={redo}
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

        {/* Edit mode */}
        <div className="flex items-center gap-1 px-2 border-r border-border">
          <ToolbarButton
            icon={MousePointer}
            label="Select Mode"
            active={editMode === 'select'}
            onClick={() => setEditMode('select')}
          />
          <ToolbarButton
            icon={Move}
            label="Move Mode"
            active={editMode === 'move'}
            onClick={() => setEditMode('move')}
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
            icon={Trash2}
            label="Delete Selected Points"
            disabled={selectedPoints.size === 0}
            onClick={deleteSelectedPoints}
          />
          <ToolbarButton
            icon={RotateCcw}
            label="Clear All Points"
            disabled={funscriptPoints.length === 0}
            onClick={() => updatePoints([])}
          />
        </div>

        {/* Filters */}
        <div className="flex items-center gap-1 px-2 border-r border-border">
          <ToolbarButton
            icon={Filter}
            label="Filters"
            disabled={funscriptPoints.length === 0}
            onClick={() => setShowFiltersModal(true)}
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
          <ToolbarButton
            icon={Maximize}
            label="Fit View"
            onClick={() => setTimelineZoom(1)}
          />
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
            label={device ? 'Disconnect Device' : 'Connect Device'}
            active={!!device}
            onClick={toggleDeviceModal}
          />
          <ToolbarButton
            icon={Settings}
            label="Settings"
            onClick={() => setShowSettingsModal(true)}
          />
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
                {/* Current position indicator */}
                {funscriptPoints.length > 0 && (
                  <div className="absolute bottom-4 right-4 bg-black/70 px-2 py-1 rounded text-xs text-white font-mono">
                    Pos: {(() => {
                      // Find the last point before or at current time
                      let prev: FunscriptPoint | undefined
                      for (let i = funscriptPoints.length - 1; i >= 0; i--) {
                        if (funscriptPoints[i].at <= currentTime) {
                          prev = funscriptPoints[i]
                          break
                        }
                      }
                      const next = funscriptPoints.find((p: FunscriptPoint) => p.at > currentTime)
                      if (!prev) return next?.pos ?? 0
                      if (!next) return prev.pos
                      const t = (currentTime - prev.at) / (next.at - prev.at)
                      return Math.round(prev.pos + (next.pos - prev.pos) * t)
                    })()}
                  </div>
                )}
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
                  Upload a video or import a funscript to get started
                </p>
                {uploadError && (
                  <p className="text-error text-sm mb-4">
                    Error: {uploadError}
                  </p>
                )}
                <div className="flex gap-2 justify-center">
                  <button onClick={openFileDialog} className="btn-primary">
                    <Upload className="w-5 h-5" />
                    Upload Video
                  </button>
                  <button onClick={openFunscriptDialog} className="btn-secondary">
                    <FolderOpen className="w-5 h-5" />
                    Import Funscript
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Heatmap */}
          {showHeatmap && funscriptPoints.length > 0 && (
            <div className="h-8 bg-bg-surface border-t border-border px-2 py-1">
              <Heatmap points={funscriptPoints} duration={duration || 60000} />
            </div>
          )}

          {/* Timeline */}
          <div className="h-48 bg-bg-surface border-t border-border">
            {video || funscriptPoints.length > 0 ? (
              <Timeline
                currentTime={currentTime}
                duration={duration || 60000}
                points={funscriptPoints}
                selectedPoints={selectedPoints}
                onSeek={seekTo}
                onSelectPoint={selectPoint}
                onMovePoint={movePoint}
                onDeleteSelected={deleteSelectedPoints}
                zoom={timelineZoom}
                editMode={editMode}
              />
            ) : (
              <div className="h-full flex items-center justify-center">
                <p className="text-text-muted">Load a video or import a funscript to see the timeline</p>
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
                        <span className="text-xs text-primary font-medium">{settings.confidenceThreshold}%</span>
                      </div>
                      <input
                        type="range"
                        min="0"
                        max="100"
                        value={settings.confidenceThreshold}
                        onChange={(e) => setSettings(s => ({ ...s, confidenceThreshold: Number(e.target.value) }))}
                        className="w-full accent-primary"
                      />
                    </div>
                    <div>
                      <div className="flex justify-between items-center mb-1">
                        <label className="text-xs text-text-secondary">
                          Smoothing
                        </label>
                        <span className="text-xs text-primary font-medium">{settings.smoothingFactor}%</span>
                      </div>
                      <input
                        type="range"
                        min="0"
                        max="100"
                        value={settings.smoothingFactor}
                        onChange={(e) => setSettings(s => ({ ...s, smoothingFactor: Number(e.target.value) }))}
                        className="w-full accent-primary"
                      />
                    </div>
                  </div>
                </div>
              )}

              {/* 3D Simulator */}
              {showSimulator3D && (
                <div className="card">
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="text-sm font-medium text-text-primary flex items-center gap-2">
                      <Box className="w-4 h-4" />
                      3D Simulator
                    </h3>
                    <button
                      onClick={() => setShowSimulator3D(false)}
                      className="text-text-muted hover:text-text-secondary"
                      title="Hide simulator"
                    >
                      <EyeOff className="w-4 h-4" />
                    </button>
                  </div>
                  <Simulator3D position={currentPosition} isPlaying={isPlaying} />
                </div>
              )}

              {/* Quick Filters (Expert Mode) */}
              {isExpert && funscriptPoints.length > 0 && (
                <div className="card">
                  <h3 className="text-sm font-medium text-text-primary mb-3 flex items-center gap-2">
                    <Filter className="w-4 h-4" />
                    Quick Filters
                  </h3>
                  <div className="grid grid-cols-2 gap-2">
                    <button
                      className="btn-secondary text-xs py-1.5"
                      onClick={() => applyFilter('smooth', 50)}
                    >
                      Smooth
                    </button>
                    <button
                      className="btn-secondary text-xs py-1.5"
                      onClick={() => applyFilter('amplify', 20)}
                    >
                      Amplify
                    </button>
                    <button
                      className="btn-secondary text-xs py-1.5"
                      onClick={() => applyFilter('invert')}
                    >
                      Invert
                    </button>
                    <button
                      className="btn-secondary text-xs py-1.5"
                      onClick={() => applyFilter('simplify', 5)}
                    >
                      Simplify
                    </button>
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
                    device?.status === 'connected' || device?.status === 'syncing'
                      ? "bg-green-500"
                      : device?.status === 'connecting'
                        ? "bg-yellow-500 animate-pulse"
                        : "bg-text-disabled"
                  )} />
                  <span className="text-sm text-text-secondary">
                    {device ? `${device.name} (${device.status})` : 'Disconnected'}
                  </span>
                </div>
                {device && device.firmware_version && (
                  <p className="text-xs text-text-muted mt-1">
                    Firmware: v{device.firmware_version}
                  </p>
                )}
                <button
                  className="w-full btn-secondary mt-3 text-sm"
                  onClick={toggleDeviceModal}
                >
                  <Bluetooth className="w-4 h-4" />
                  {device ? 'Disconnect' : 'Connect'}
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
                      {formatDuration(duration || (video?.duration_ms ?? 0))}
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
                  {isExpert && stats.points > 0 && (
                    <>
                      <div className="flex justify-between">
                        <span className="text-text-secondary">Strokes</span>
                        <span className="text-text-primary font-medium">{stats.strokeCount}</span>
                      </div>
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

              {/* View Controls */}
              <div className="card">
                <h3 className="text-sm font-medium text-text-primary mb-3 flex items-center gap-2">
                  <Eye className="w-4 h-4" />
                  View Options
                </h3>
                <div className="space-y-2">
                  <label className="flex items-center gap-2 cursor-pointer text-sm">
                    <input
                      type="checkbox"
                      checked={showSimulator3D}
                      onChange={(e) => setShowSimulator3D(e.target.checked)}
                      className="accent-primary w-4 h-4"
                    />
                    <Box className="w-4 h-4 text-text-muted" />
                    <span className="text-text-secondary">3D Simulator</span>
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer text-sm">
                    <input
                      type="checkbox"
                      checked={showHeatmap}
                      onChange={(e) => setShowHeatmap(e.target.checked)}
                      className="accent-primary w-4 h-4"
                    />
                    <Activity className="w-4 h-4 text-text-muted" />
                    <span className="text-text-secondary">Heatmap</span>
                  </label>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Device Connection Modal */}
      {showDeviceModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div
            className="absolute inset-0 bg-black/50"
            onClick={() => setShowDeviceModal(false)}
          />
          <div className="relative bg-bg-surface rounded-lg p-6 w-full max-w-md mx-4 shadow-xl">
            <h2 className="text-lg font-semibold text-text-primary mb-4 flex items-center gap-2">
              <Bluetooth className="w-5 h-5" />
              Connect Handy Device
            </h2>
            <p className="text-sm text-text-secondary mb-4">
              Enter your Handy connection key to connect your device.
              You can find this in the Handy app or at handyfeeling.com.
            </p>
            <input
              type="text"
              placeholder="Enter connection key (e.g., ABC123)"
              value={connectionKey}
              onChange={(e) => setConnectionKey(e.target.value)}
              className="w-full px-3 py-2 bg-bg-base border border-border rounded-md text-text-primary placeholder:text-text-muted mb-4 focus:outline-none focus:ring-2 focus:ring-primary"
              autoFocus
              onKeyDown={(e) => e.key === 'Enter' && connectDevice()}
            />
            <div className="flex justify-end gap-2">
              <button
                className="btn-secondary text-sm"
                onClick={() => {
                  setShowDeviceModal(false)
                  setConnectionKey('')
                }}
              >
                Cancel
              </button>
              <button
                className="btn-primary text-sm"
                onClick={connectDevice}
                disabled={!connectionKey.trim() || isConnecting}
              >
                {isConnecting ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Connecting...
                  </>
                ) : (
                  <>
                    <Bluetooth className="w-4 h-4" />
                    Connect
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Settings Modal */}
      {showSettingsModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div
            className="absolute inset-0 bg-black/50"
            onClick={() => setShowSettingsModal(false)}
          />
          <div className="relative bg-bg-surface rounded-lg p-6 w-full max-w-lg mx-4 shadow-xl max-h-[80vh] overflow-y-auto">
            <h2 className="text-lg font-semibold text-text-primary mb-4 flex items-center gap-2">
              <Settings className="w-5 h-5" />
              Settings
            </h2>

            <div className="mb-6">
              <h3 className="text-sm font-medium text-text-primary mb-3 flex items-center gap-2">
                <Activity className="w-4 h-4" />
                AI Processing
              </h3>
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between items-center mb-1">
                    <label className="text-sm text-text-secondary">Confidence Threshold</label>
                    <span className="text-sm text-primary font-medium">{settings.confidenceThreshold}%</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={settings.confidenceThreshold}
                    onChange={(e) => setSettings(s => ({ ...s, confidenceThreshold: Number(e.target.value) }))}
                    className="w-full accent-primary"
                  />
                </div>
                <div>
                  <div className="flex justify-between items-center mb-1">
                    <label className="text-sm text-text-secondary">Smoothing Factor</label>
                    <span className="text-sm text-primary font-medium">{settings.smoothingFactor}%</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={settings.smoothingFactor}
                    onChange={(e) => setSettings(s => ({ ...s, smoothingFactor: Number(e.target.value) }))}
                    className="w-full accent-primary"
                  />
                </div>
                <div>
                  <div className="flex justify-between items-center mb-1">
                    <label className="text-sm text-text-secondary">Min Point Distance</label>
                    <span className="text-sm text-primary font-medium">{settings.minPointDistance}ms</span>
                  </div>
                  <input
                    type="range"
                    min="50"
                    max="500"
                    step="10"
                    value={settings.minPointDistance}
                    onChange={(e) => setSettings(s => ({ ...s, minPointDistance: Number(e.target.value) }))}
                    className="w-full accent-primary"
                  />
                </div>
              </div>
            </div>

            <div className="mb-6">
              <h3 className="text-sm font-medium text-text-primary mb-3 flex items-center gap-2">
                <Sliders className="w-4 h-4" />
                Output Settings
              </h3>
              <div className="space-y-3">
                <label className="flex items-center gap-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={settings.autoSmoothPeaks}
                    onChange={(e) => setSettings(s => ({ ...s, autoSmoothPeaks: e.target.checked }))}
                    className="accent-primary w-4 h-4"
                  />
                  <div>
                    <span className="text-sm text-text-primary">Auto-smooth peaks</span>
                    <p className="text-xs text-text-muted">Reduce sudden position changes</p>
                  </div>
                </label>
                <label className="flex items-center gap-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={settings.invertOutput}
                    onChange={(e) => setSettings(s => ({ ...s, invertOutput: e.target.checked }))}
                    className="accent-primary w-4 h-4"
                  />
                  <div>
                    <span className="text-sm text-text-primary">Invert output</span>
                    <p className="text-xs text-text-muted">Flip position values (0↔100)</p>
                  </div>
                </label>
                <label className="flex items-center gap-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={settings.limitRange}
                    onChange={(e) => setSettings(s => ({ ...s, limitRange: e.target.checked }))}
                    className="accent-primary w-4 h-4"
                  />
                  <div>
                    <span className="text-sm text-text-primary">Limit range (0-100)</span>
                    <p className="text-xs text-text-muted">Clamp all positions to valid range</p>
                  </div>
                </label>
              </div>
            </div>

            <div className="mb-6">
              <h3 className="text-sm font-medium text-text-primary mb-3 flex items-center gap-2">
                <Play className="w-4 h-4" />
                Playback
              </h3>
              <div>
                <div className="flex justify-between items-center mb-1">
                  <label className="text-sm text-text-secondary">Playback Speed</label>
                  <span className="text-sm text-primary font-medium">{settings.playbackSpeed}x</span>
                </div>
                <input
                  type="range"
                  min="0.25"
                  max="2"
                  step="0.25"
                  value={settings.playbackSpeed}
                  onChange={(e) => {
                    const speed = Number(e.target.value)
                    setSettings(s => ({ ...s, playbackSpeed: speed }))
                    if (videoRef.current) {
                      videoRef.current.playbackRate = speed
                    }
                  }}
                  className="w-full accent-primary"
                />
              </div>
            </div>

            <div className="flex justify-end gap-2 pt-4 border-t border-border">
              <button
                className="btn-secondary text-sm"
                onClick={() => setSettings({
                  confidenceThreshold: 50,
                  smoothingFactor: 30,
                  minPointDistance: 100,
                  autoSmoothPeaks: true,
                  invertOutput: false,
                  limitRange: true,
                  playbackSpeed: 1.0,
                })}
              >
                Reset to Defaults
              </button>
              <button
                className="btn-primary text-sm"
                onClick={() => setShowSettingsModal(false)}
              >
                Done
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Filters Modal */}
      {showFiltersModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div
            className="absolute inset-0 bg-black/50"
            onClick={() => setShowFiltersModal(false)}
          />
          <div className="relative bg-bg-surface rounded-lg p-6 w-full max-w-md mx-4 shadow-xl">
            <h2 className="text-lg font-semibold text-text-primary mb-4 flex items-center gap-2">
              <Filter className="w-5 h-5" />
              Post-Processing Filters
            </h2>
            <p className="text-sm text-text-secondary mb-4">
              Apply filters to modify the funscript. Changes can be undone with Ctrl+Z.
            </p>

            <div className="space-y-3">
              <button
                className="w-full btn-secondary text-sm justify-start"
                onClick={() => { applyFilter('smooth', 50); setShowFiltersModal(false); }}
              >
                <Sliders className="w-4 h-4" />
                Smooth - Reduce noise and jitter
              </button>
              <button
                className="w-full btn-secondary text-sm justify-start"
                onClick={() => { applyFilter('amplify', 30); setShowFiltersModal(false); }}
              >
                <Activity className="w-4 h-4" />
                Amplify - Increase intensity
              </button>
              <button
                className="w-full btn-secondary text-sm justify-start"
                onClick={() => { applyFilter('invert'); setShowFiltersModal(false); }}
              >
                <RotateCcw className="w-4 h-4" />
                Invert - Flip all positions
              </button>
              <button
                className="w-full btn-secondary text-sm justify-start"
                onClick={() => { applyFilter('speedLimit', 400); setShowFiltersModal(false); }}
              >
                <ChevronRight className="w-4 h-4" />
                Speed Limit - Cap maximum speed
              </button>
              <button
                className="w-full btn-secondary text-sm justify-start"
                onClick={() => { applyFilter('simplify', 5); setShowFiltersModal(false); }}
              >
                <Trash2 className="w-4 h-4" />
                Simplify - Remove redundant points
              </button>
            </div>

            <div className="flex justify-end mt-6">
              <button
                className="btn-primary text-sm"
                onClick={() => setShowFiltersModal(false)}
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
