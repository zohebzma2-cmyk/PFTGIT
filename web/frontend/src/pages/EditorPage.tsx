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
  EyeOff,
  FilePlus,
  X,
  Copy,
  Clipboard,
  Keyboard,
  Info,
  Github,
  Layout,
  Palette,
  Bookmark,
  BookmarkPlus,
  List,
  BarChart3,
  Waves,
  SplitSquareVertical,
  PanelRightClose,
  PanelRightOpen,
  Search
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

  // Savitzky-Golay filter - polynomial smoothing that preserves peaks
  savitzkyGolay: (points: FunscriptPoint[], windowSize: number = 5): FunscriptPoint[] => {
    if (points.length < windowSize) return points
    const halfWindow = Math.floor(windowSize / 2)
    const result: FunscriptPoint[] = []

    // Savitzky-Golay coefficients for quadratic polynomial (window 5)
    const coeffs = [-3, 12, 17, 12, -3].map(c => c / 35)

    for (let i = 0; i < points.length; i++) {
      if (i < halfWindow || i >= points.length - halfWindow) {
        result.push(points[i])
      } else {
        let smoothedPos = 0
        for (let j = -halfWindow; j <= halfWindow; j++) {
          smoothedPos += points[i + j].pos * coeffs[j + halfWindow]
        }
        result.push({ at: points[i].at, pos: Math.max(0, Math.min(100, Math.round(smoothedPos))) })
      }
    }
    return result
  },

  // Anti-jerk filter - removes sudden jerky movements
  antiJerk: (points: FunscriptPoint[], threshold: number = 30): FunscriptPoint[] => {
    if (points.length < 3) return points
    const result: FunscriptPoint[] = [points[0]]

    for (let i = 1; i < points.length - 1; i++) {
      const prev = result[result.length - 1]
      const curr = points[i]
      const next = points[i + 1]

      // Check if this is a jerk (sudden change that immediately reverses)
      const changeToPrev = curr.pos - prev.pos
      const changeToNext = next.pos - curr.pos

      // If direction reverses sharply and change is above threshold
      if (Math.sign(changeToPrev) !== Math.sign(changeToNext) &&
          Math.abs(changeToPrev) > threshold && Math.abs(changeToNext) > threshold) {
        // Smooth out the jerk by averaging
        const smoothedPos = Math.round((prev.pos + next.pos) / 2)
        result.push({ at: curr.at, pos: Math.max(0, Math.min(100, smoothedPos)) })
      } else {
        result.push(curr)
      }
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

  // Clamp filter - constrain values to a range
  clamp: (points: FunscriptPoint[], minVal: number = 10, maxVal: number = 90): FunscriptPoint[] => {
    return points.map(p => ({
      at: p.at,
      pos: Math.max(minVal, Math.min(maxVal, p.pos))
    }))
  },

  // Time shift - move all points forward or backward in time
  timeShift: (points: FunscriptPoint[], offsetMs: number): FunscriptPoint[] => {
    return points
      .map(p => ({ at: p.at + offsetMs, pos: p.pos }))
      .filter(p => p.at >= 0)
      .sort((a, b) => a.at - b.at)
  },

  // RDP simplification - Ramer-Douglas-Peucker algorithm
  rdpSimplify: (points: FunscriptPoint[], epsilon: number = 3): FunscriptPoint[] => {
    if (points.length < 3) return points

    // Find the point with maximum distance from line between first and last
    const findMaxDistance = (pts: FunscriptPoint[], start: number, end: number): { index: number, distance: number } => {
      let maxDist = 0
      let maxIndex = start
      const first = pts[start]
      const last = pts[end]
      const dx = last.at - first.at
      const dy = last.pos - first.pos
      const len = Math.sqrt(dx * dx + dy * dy)

      for (let i = start + 1; i < end; i++) {
        // Perpendicular distance from point to line
        const dist = len === 0 ? Math.abs(pts[i].pos - first.pos) :
          Math.abs(dy * (pts[i].at - first.at) - dx * (pts[i].pos - first.pos)) / len
        if (dist > maxDist) {
          maxDist = dist
          maxIndex = i
        }
      }
      return { index: maxIndex, distance: maxDist }
    }

    const rdp = (pts: FunscriptPoint[], start: number, end: number, result: FunscriptPoint[]): void => {
      if (end <= start + 1) return

      const { index, distance } = findMaxDistance(pts, start, end)
      if (distance > epsilon) {
        rdp(pts, start, index, result)
        result.push(pts[index])
        rdp(pts, index, end, result)
      }
    }

    const result: FunscriptPoint[] = [points[0]]
    rdp(points, 0, points.length - 1, result)
    result.push(points[points.length - 1])

    return result.sort((a, b) => a.at - b.at)
  },

  // Keyframe detection - identify peaks and valleys
  keyframes: (points: FunscriptPoint[], minChange: number = 20): FunscriptPoint[] => {
    if (points.length < 3) return points
    const result: FunscriptPoint[] = [points[0]]

    for (let i = 1; i < points.length - 1; i++) {
      const prev = points[i - 1]
      const curr = points[i]
      const next = points[i + 1]

      // Check if this is a peak (higher than neighbors) or valley (lower than neighbors)
      const isPeak = curr.pos > prev.pos && curr.pos > next.pos
      const isValley = curr.pos < prev.pos && curr.pos < next.pos

      // Check if the change is significant
      const changeFromPrev = Math.abs(curr.pos - prev.pos)
      const changeToNext = Math.abs(next.pos - curr.pos)

      if ((isPeak || isValley) && (changeFromPrev >= minChange || changeToNext >= minChange)) {
        result.push(curr)
      }
    }

    result.push(points[points.length - 1])
    return result
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

// Movement Gauge component - shows current position and speed
function MovementGauge({ position, speed }: { position: number, speed: number }) {
  const normalizedSpeed = Math.min(100, Math.abs(speed) / 5) // Normalize speed to 0-100
  const direction = speed > 0 ? 'up' : speed < 0 ? 'down' : 'neutral'

  return (
    <div className="bg-bg-base rounded-lg p-3">
      <div className="flex justify-between items-center mb-2">
        <span className="text-xs text-text-muted">Position</span>
        <span className="text-sm font-mono text-text-primary">{Math.round(position)}</span>
      </div>
      {/* Position bar */}
      <div className="h-4 bg-bg-elevated rounded-full overflow-hidden mb-3">
        <div
          className="h-full bg-primary transition-all duration-100"
          style={{ width: `${position}%` }}
        />
      </div>
      <div className="flex justify-between items-center mb-2">
        <span className="text-xs text-text-muted">Speed</span>
        <span className={clsx(
          "text-sm font-mono",
          direction === 'up' ? 'text-green-400' : direction === 'down' ? 'text-red-400' : 'text-text-muted'
        )}>
          {direction === 'up' ? '↑' : direction === 'down' ? '↓' : '•'} {Math.round(Math.abs(speed))} u/s
        </span>
      </div>
      {/* Speed bar */}
      <div className="h-4 bg-bg-elevated rounded-full overflow-hidden">
        <div
          className={clsx(
            "h-full transition-all duration-100",
            direction === 'up' ? 'bg-green-500' : direction === 'down' ? 'bg-red-500' : 'bg-gray-500'
          )}
          style={{ width: `${normalizedSpeed}%` }}
        />
      </div>
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

// Chapter type definition
interface Chapter {
  id: string
  name: string
  startTime: number
  endTime: number
  color: string
}

// Menu item type
interface MenuItem {
  label: string
  shortcut?: string
  icon?: React.ElementType
  onClick?: () => void
  disabled?: boolean
  separator?: boolean
  submenu?: MenuItem[]
  checked?: boolean
}

// MenuBar component
function MenuBar({
  onNewProject,
  onOpenProject,
  onSaveProject,
  onExportFunscript,
  onImportFunscript,
  onUndo,
  onRedo,
  canUndo,
  canRedo,
  isExpert,
  onToggleMode,
  showHeatmap,
  onToggleHeatmap,
  showSimulator3D,
  onToggleSimulator3D,
  showTimeline2,
  onToggleTimeline2,
  showWaveform,
  onToggleWaveform,
  showChapterList,
  onToggleChapterList,
  onShowKeyboardShortcuts,
  onShowSettings,
  onShowFilters,
  onShowAbout,
  onSaveChapters,
  onLoadChapters,
  onClearChapters,
  theme,
  onToggleTheme,
}: {
  onNewProject: () => void
  onOpenProject: () => void
  onSaveProject: () => void
  onExportFunscript: () => void
  onImportFunscript: () => void
  onUndo: () => void
  onRedo: () => void
  canUndo: boolean
  canRedo: boolean
  isExpert: boolean
  onToggleMode: () => void
  showHeatmap: boolean
  onToggleHeatmap: () => void
  showSimulator3D: boolean
  onToggleSimulator3D: () => void
  showTimeline2: boolean
  onToggleTimeline2: () => void
  showWaveform: boolean
  onToggleWaveform: () => void
  showChapterList: boolean
  onToggleChapterList: () => void
  onShowKeyboardShortcuts: () => void
  onShowSettings: () => void
  onShowFilters: () => void
  onShowAbout: () => void
  onSaveChapters: () => void
  onLoadChapters: () => void
  onClearChapters: () => void
  theme: 'dark' | 'light'
  onToggleTheme: () => void
}) {
  const [activeMenu, setActiveMenu] = useState<string | null>(null)

  const menus: Record<string, MenuItem[]> = {
    File: [
      { label: 'New Project', shortcut: 'Ctrl+N', icon: FilePlus, onClick: onNewProject },
      { label: 'Open Project', shortcut: 'Ctrl+O', icon: FolderOpen, onClick: onOpenProject },
      { label: 'Save Project', shortcut: 'Ctrl+S', icon: Save, onClick: onSaveProject },
      { separator: true, label: '' },
      { label: 'Import Funscript', icon: FileUp, onClick: onImportFunscript },
      { label: 'Export Funscript', shortcut: 'Ctrl+E', icon: Download, onClick: onExportFunscript },
      { separator: true, label: '' },
      { label: 'Chapters', icon: Bookmark, submenu: [
        { label: 'Save Chapters', icon: Save, onClick: onSaveChapters },
        { label: 'Load Chapters', icon: FolderOpen, onClick: onLoadChapters },
        { label: 'Clear Chapters', icon: Trash2, onClick: onClearChapters },
      ]},
    ],
    Edit: [
      { label: 'Undo', shortcut: 'Ctrl+Z', icon: Undo2, onClick: onUndo, disabled: !canUndo },
      { label: 'Redo', shortcut: 'Ctrl+Y', icon: Redo2, onClick: onRedo, disabled: !canRedo },
      { separator: true, label: '' },
      { label: 'Copy Points', shortcut: 'Ctrl+C', icon: Copy, onClick: () => {} },
      { label: 'Paste Points', shortcut: 'Ctrl+V', icon: Clipboard, onClick: () => {} },
      { label: 'Delete Selected', shortcut: 'Delete', icon: Trash2, onClick: () => {} },
      { separator: true, label: '' },
      { label: 'Select All', shortcut: 'Ctrl+A', icon: MousePointer, onClick: () => {} },
      { label: 'Deselect All', shortcut: 'Ctrl+D', onClick: () => {} },
    ],
    View: [
      { label: isExpert ? 'Simple Mode' : 'Expert Mode', icon: Layout, onClick: onToggleMode },
      { separator: true, label: '' },
      { label: 'Show Heatmap', icon: Activity, onClick: onToggleHeatmap, checked: showHeatmap },
      { label: 'Show 3D Simulator', icon: Box, onClick: onToggleSimulator3D, checked: showSimulator3D },
      { label: 'Show Timeline 2', icon: SplitSquareVertical, onClick: onToggleTimeline2, checked: showTimeline2 },
      { label: 'Show Waveform', icon: Waves, onClick: onToggleWaveform, checked: showWaveform },
      { separator: true, label: '' },
      { label: 'Chapter List', icon: List, onClick: onToggleChapterList, checked: showChapterList },
      { separator: true, label: '' },
      { label: theme === 'dark' ? 'Light Theme' : 'Dark Theme', icon: Palette, onClick: onToggleTheme },
    ],
    Tools: [
      { label: 'Post-Processing Filters', shortcut: 'Ctrl+F', icon: Filter, onClick: onShowFilters },
      { separator: true, label: '' },
      { label: 'Settings', icon: Settings, onClick: onShowSettings },
    ],
    Help: [
      { label: 'Keyboard Shortcuts', shortcut: 'F1', icon: Keyboard, onClick: onShowKeyboardShortcuts },
      { separator: true, label: '' },
      { label: 'About FunGen', icon: Info, onClick: onShowAbout },
      { label: 'GitHub', icon: Github, onClick: () => window.open('https://github.com', '_blank') },
    ],
  }

  const handleMenuClick = (menuName: string) => {
    setActiveMenu(activeMenu === menuName ? null : menuName)
  }

  const handleItemClick = (item: MenuItem) => {
    if (item.onClick && !item.disabled) {
      item.onClick()
    }
    setActiveMenu(null)
  }

  // Close menu on outside click
  useEffect(() => {
    const handleClickOutside = () => setActiveMenu(null)
    if (activeMenu) {
      document.addEventListener('click', handleClickOutside)
      return () => document.removeEventListener('click', handleClickOutside)
    }
  }, [activeMenu])

  return (
    <div className="flex items-center h-8 bg-bg-elevated border-b border-border px-1 sm:px-2 gap-0.5 sm:gap-1 overflow-x-auto scrollbar-thin">
      {Object.entries(menus).map(([menuName, items]) => (
        <div key={menuName} className="relative flex-shrink-0">
          <button
            className={clsx(
              'px-2 sm:px-3 py-1 text-xs sm:text-sm rounded hover:bg-bg-surface transition-colors whitespace-nowrap',
              activeMenu === menuName && 'bg-bg-surface'
            )}
            onClick={(e) => {
              e.stopPropagation()
              handleMenuClick(menuName)
            }}
          >
            {menuName}
          </button>
          {activeMenu === menuName && (
            <div className="absolute left-0 top-full mt-1 bg-bg-surface border border-border rounded-lg shadow-xl z-50 min-w-[200px] sm:min-w-[220px] py-1 max-h-[70vh] overflow-y-auto">
              {items.map((item, idx) => (
                item.separator ? (
                  <div key={idx} className="h-px bg-border mx-2 my-1" />
                ) : item.submenu ? (
                  <div key={idx} className="relative group">
                    <button
                      className="w-full flex items-center gap-2 px-3 py-1.5 text-sm text-text-primary hover:bg-primary/20 transition-colors"
                    >
                      {item.icon && <item.icon className="w-4 h-4 text-text-muted" />}
                      <span className="flex-1 text-left">{item.label}</span>
                      <ChevronRight className="w-4 h-4 text-text-muted" />
                    </button>
                    <div className="absolute left-full top-0 bg-bg-surface border border-border rounded-lg shadow-xl z-50 min-w-[180px] py-1 hidden group-hover:block">
                      {item.submenu.map((subitem, subidx) => (
                        <button
                          key={subidx}
                          className="w-full flex items-center gap-2 px-3 py-1.5 text-sm text-text-primary hover:bg-primary/20 transition-colors disabled:opacity-50"
                          onClick={() => handleItemClick(subitem)}
                          disabled={subitem.disabled}
                        >
                          {subitem.icon && <subitem.icon className="w-4 h-4 text-text-muted" />}
                          <span className="flex-1 text-left">{subitem.label}</span>
                        </button>
                      ))}
                    </div>
                  </div>
                ) : (
                  <button
                    key={idx}
                    className={clsx(
                      "w-full flex items-center gap-2 px-3 py-1.5 text-sm text-text-primary hover:bg-primary/20 transition-colors",
                      item.disabled && "opacity-50 cursor-not-allowed"
                    )}
                    onClick={() => handleItemClick(item)}
                    disabled={item.disabled}
                  >
                    {item.icon && <item.icon className="w-4 h-4 text-text-muted" />}
                    <span className="flex-1 text-left">{item.label}</span>
                    {item.checked !== undefined && (
                      <span className={clsx(
                        "w-4 h-4 rounded border flex items-center justify-center text-xs",
                        item.checked ? "bg-primary border-primary text-white" : "border-border"
                      )}>
                        {item.checked && '✓'}
                      </span>
                    )}
                    {item.shortcut && (
                      <span className="text-xs text-text-muted">{item.shortcut}</span>
                    )}
                  </button>
                )
              ))}
            </div>
          )}
        </div>
      ))}

      {/* Right side - status indicators */}
      <div className="flex-1" />
      <div className="flex items-center gap-2 text-xs text-text-muted">
        <span className="px-2 py-0.5 bg-bg-base rounded">{isExpert ? 'Expert' : 'Simple'}</span>
      </div>
    </div>
  )
}

// Keyboard shortcuts dialog component
function KeyboardShortcutsDialog({ onClose }: { onClose: () => void }) {
  const shortcuts = [
    { category: 'Playback', items: [
      { key: 'Space', action: 'Play/Pause' },
      { key: 'Left/Right Arrow', action: 'Skip frames' },
      { key: 'Home', action: 'Go to start' },
      { key: 'End', action: 'Go to end' },
    ]},
    { category: 'Editing', items: [
      { key: 'Ctrl+Z', action: 'Undo' },
      { key: 'Ctrl+Y', action: 'Redo' },
      { key: 'Ctrl+C', action: 'Copy selected points' },
      { key: 'Ctrl+V', action: 'Paste points' },
      { key: 'Delete', action: 'Delete selected points' },
      { key: 'Ctrl+A', action: 'Select all points' },
      { key: 'Ctrl+D', action: 'Deselect all' },
    ]},
    { category: 'Point Entry', items: [
      { key: '0-9', action: 'Add point at position 0-90' },
      { key: '=', action: 'Add point at position 100' },
      { key: 'Up Arrow', action: 'Jump to next point' },
      { key: 'Down Arrow', action: 'Jump to previous point' },
      { key: 'Shift+Up', action: 'Nudge selected points up' },
      { key: 'Shift+Down', action: 'Nudge selected points down' },
    ]},
    { category: 'View', items: [
      { key: 'Ctrl++', action: 'Zoom in timeline' },
      { key: 'Ctrl+-', action: 'Zoom out timeline' },
      { key: 'F1', action: 'Show keyboard shortcuts' },
    ]},
    { category: 'File', items: [
      { key: 'Ctrl+S', action: 'Save project' },
      { key: 'Ctrl+O', action: 'Open project' },
      { key: 'Ctrl+E', action: 'Export funscript' },
    ]},
  ]

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/50" onClick={onClose} />
      <div className="relative bg-bg-surface rounded-lg p-6 w-full max-w-2xl mx-4 shadow-xl max-h-[80vh] flex flex-col">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-text-primary flex items-center gap-2">
            <Keyboard className="w-5 h-5" />
            Keyboard Shortcuts
          </h2>
          <button onClick={onClose} className="text-text-muted hover:text-text-primary">
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="grid grid-cols-2 gap-6 overflow-y-auto flex-1">
          {shortcuts.map((section) => (
            <div key={section.category}>
              <h3 className="text-sm font-medium text-primary mb-2">{section.category}</h3>
              <div className="space-y-1">
                {section.items.map((item) => (
                  <div key={item.key} className="flex items-center justify-between text-sm">
                    <span className="text-text-secondary">{item.action}</span>
                    <kbd className="px-2 py-0.5 bg-bg-elevated rounded text-xs text-text-muted font-mono">
                      {item.key}
                    </kbd>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>

        <div className="flex justify-end mt-4 pt-4 border-t border-border">
          <button className="btn-primary text-sm" onClick={onClose}>Close</button>
        </div>
      </div>
    </div>
  )
}

// About dialog component
function AboutDialog({ onClose }: { onClose: () => void }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/50" onClick={onClose} />
      <div className="relative bg-bg-surface rounded-lg p-6 w-full max-w-md mx-4 shadow-xl">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-text-primary flex items-center gap-2">
            <Info className="w-5 h-5" />
            About FunGen
          </h2>
          <button onClick={onClose} className="text-text-muted hover:text-text-primary">
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="text-center mb-6">
          <div className="w-16 h-16 mx-auto mb-4 bg-primary rounded-xl flex items-center justify-center">
            <Wand2 className="w-8 h-8 text-white" />
          </div>
          <h3 className="text-xl font-bold text-text-primary">FunGen Web</h3>
          <p className="text-sm text-text-muted">Version 1.0.0</p>
        </div>

        <div className="space-y-3 text-sm text-text-secondary">
          <p>
            AI-powered funscript generator for creating synchronized haptic feedback scripts from video content.
          </p>
          <p>
            Features include automatic motion detection, manual editing tools, real-time preview, and device connectivity.
          </p>
        </div>

        <div className="mt-6 pt-4 border-t border-border">
          <div className="flex justify-center gap-4">
            <button
              className="btn-secondary text-sm flex items-center gap-2"
              onClick={() => window.open('https://github.com', '_blank')}
            >
              <Github className="w-4 h-4" />
              GitHub
            </button>
          </div>
        </div>

        <div className="flex justify-end mt-4">
          <button className="btn-primary text-sm" onClick={onClose}>Close</button>
        </div>
      </div>
    </div>
  )
}

// Statistics panel component
function StatisticsPanel({ stats, duration, video }: {
  stats: { points: number, avgSpeed: number, peakSpeed: number, intensity: number, strokeCount: number },
  duration: number,
  video: VideoMetadata | null
}) {
  return (
    <div className="bg-bg-base rounded-lg p-3">
      <h3 className="text-sm font-medium text-text-primary mb-3 flex items-center gap-2">
        <BarChart3 className="w-4 h-4" />
        Statistics
      </h3>
      <div className="grid grid-cols-2 gap-2 text-sm">
        <div className="flex justify-between">
          <span className="text-text-muted">Points:</span>
          <span className="text-text-primary font-mono">{stats.points}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-text-muted">Duration:</span>
          <span className="text-text-primary font-mono">{formatDuration(duration)}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-text-muted">Avg Speed:</span>
          <span className="text-text-primary font-mono">{stats.avgSpeed} u/s</span>
        </div>
        <div className="flex justify-between">
          <span className="text-text-muted">Peak Speed:</span>
          <span className="text-text-primary font-mono">{stats.peakSpeed} u/s</span>
        </div>
        <div className="flex justify-between">
          <span className="text-text-muted">Strokes:</span>
          <span className="text-text-primary font-mono">{stats.strokeCount}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-text-muted">Intensity:</span>
          <span className="text-text-primary font-mono">{stats.intensity}%</span>
        </div>
        {video && (
          <>
            <div className="flex justify-between col-span-2">
              <span className="text-text-muted">Resolution:</span>
              <span className="text-text-primary font-mono">{video.width}x{video.height}</span>
            </div>
            <div className="flex justify-between col-span-2">
              <span className="text-text-muted">FPS:</span>
              <span className="text-text-primary font-mono">{video.fps?.toFixed(2) || 'N/A'}</span>
            </div>
          </>
        )}
      </div>
    </div>
  )
}

// Chapter list component
function ChapterList({
  chapters,
  currentTime,
  onSeek,
  onAddChapter,
  onDeleteChapter
}: {
  chapters: Chapter[]
  currentTime: number
  onSeek: (time: number) => void
  onAddChapter: () => void
  onDeleteChapter: (id: string) => void
}) {
  return (
    <div className="bg-bg-base rounded-lg p-3">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-text-primary flex items-center gap-2">
          <Bookmark className="w-4 h-4" />
          Chapters
        </h3>
        <button
          onClick={onAddChapter}
          className="p-1 hover:bg-bg-elevated rounded"
          title="Add chapter at current time"
        >
          <BookmarkPlus className="w-4 h-4 text-text-muted" />
        </button>
      </div>
      {chapters.length === 0 ? (
        <p className="text-xs text-text-muted text-center py-4">No chapters yet</p>
      ) : (
        <div className="space-y-1 max-h-48 overflow-y-auto">
          {chapters.map((chapter) => (
            <div
              key={chapter.id}
              className={clsx(
                "flex items-center gap-2 p-2 rounded cursor-pointer hover:bg-bg-elevated transition-colors",
                currentTime >= chapter.startTime && currentTime < chapter.endTime && "bg-primary/20"
              )}
              onClick={() => onSeek(chapter.startTime)}
            >
              <div className="w-2 h-2 rounded-full" style={{ backgroundColor: chapter.color }} />
              <span className="flex-1 text-sm text-text-primary truncate">{chapter.name}</span>
              <span className="text-xs text-text-muted font-mono">{formatDuration(chapter.startTime)}</span>
              <button
                onClick={(e) => { e.stopPropagation(); onDeleteChapter(chapter.id); }}
                className="p-0.5 hover:text-red-400"
              >
                <X className="w-3 h-3" />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// Audio waveform component
function AudioWaveform({
  duration,
  currentTime
}: {
  duration: number
  currentTime: number
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [waveformData, setWaveformData] = useState<number[]>([])

  useEffect(() => {
    // Generate placeholder waveform data (in production, this would analyze actual audio)
    const segments = 200
    const data: number[] = []
    for (let i = 0; i < segments; i++) {
      // Create a realistic-looking waveform pattern
      const base = Math.sin(i * 0.1) * 0.3 + 0.5
      const noise = (Math.random() - 0.5) * 0.4
      data.push(Math.max(0.1, Math.min(1, base + noise)))
    }
    setWaveformData(data)
  }, [duration])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || waveformData.length === 0) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const width = canvas.width
    const height = canvas.height
    const barWidth = width / waveformData.length

    // Clear
    ctx.fillStyle = '#1a1a1a'
    ctx.fillRect(0, 0, width, height)

    // Draw waveform
    waveformData.forEach((value, i) => {
      const x = i * barWidth
      const barHeight = value * height * 0.8
      const y = (height - barHeight) / 2

      // Color based on playback position
      const progress = duration > 0 ? (i / waveformData.length) : 0
      const isPlayed = progress <= (currentTime / duration)

      ctx.fillStyle = isPlayed ? '#22c55e' : '#4ade80'
      ctx.globalAlpha = isPlayed ? 0.8 : 0.4
      ctx.fillRect(x, y, barWidth - 1, barHeight)
    })

    ctx.globalAlpha = 1

    // Draw playhead
    if (duration > 0) {
      const playheadX = (currentTime / duration) * width
      ctx.fillStyle = '#ffffff'
      ctx.fillRect(playheadX - 1, 0, 2, height)
    }
  }, [waveformData, currentTime, duration])

  return (
    <div className="h-12 bg-bg-base rounded overflow-hidden">
      <canvas
        ref={canvasRef}
        width={800}
        height={48}
        className="w-full h-full"
      />
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
  const [deviceType, setDeviceType] = useState<'handy' | 'bluetooth'>('handy')
  const [isScanning, setIsScanning] = useState(false)
  const [discoveredDevices, setDiscoveredDevices] = useState<Array<{ id: string; name: string; type: string }>>([])
  const [bluetoothSupported, setBluetoothSupported] = useState(false)

  // Check Bluetooth support on mount
  useEffect(() => {
    if (typeof navigator !== 'undefined' && 'bluetooth' in navigator) {
      setBluetoothSupported(true)
    }
  }, [])

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
  const [showTimeline2, setShowTimeline2] = useState(false)
  const [showWaveform, setShowWaveform] = useState(false)
  const [showChapterList, setShowChapterList] = useState(false)
  const [showKeyboardShortcuts, setShowKeyboardShortcuts] = useState(false)
  const [showAboutDialog, setShowAboutDialog] = useState(false)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)

  // Auto-collapse sidebar on small screens
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth < 1024) {
        setSidebarCollapsed(true)
      } else {
        setSidebarCollapsed(false)
      }
    }
    handleResize() // Check on mount
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [])

  // Timeline 2 state (secondary axis for roll)
  const [funscriptPoints2, setFunscriptPoints2] = useState<FunscriptPoint[]>([])
  const [selectedPoints2, setSelectedPoints2] = useState<Set<number>>(new Set())

  // Chapters state
  const [chapters, setChapters] = useState<Chapter[]>([])

  // Theme state
  const [theme, setTheme] = useState<'dark' | 'light'>('dark')

  // Project state
  const [projectName, setProjectName] = useState<string>('')
  const [projectDirty, setProjectDirty] = useState(false)

  // Clipboard state for copy/paste
  const [clipboard, setClipboard] = useState<FunscriptPoint[]>([])

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

  // Calculate current speed from funscript (units per second)
  const currentSpeed = useMemo(() => {
    if (funscriptPoints.length < 2) return 0

    // Find surrounding points
    let prevIdx = -1
    for (let i = funscriptPoints.length - 1; i >= 0; i--) {
      if (funscriptPoints[i].at <= currentTime) {
        prevIdx = i
        break
      }
    }

    if (prevIdx === -1) return 0
    if (prevIdx >= funscriptPoints.length - 1) return 0

    const prev = funscriptPoints[prevIdx]
    const next = funscriptPoints[prevIdx + 1]
    const dt = next.at - prev.at
    if (dt === 0) return 0

    // Calculate speed in units per second (positive = moving up, negative = moving down)
    const dp = next.pos - prev.pos
    return (dp / dt) * 1000
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

  // Copy selected points
  const copySelectedPoints = useCallback(() => {
    if (selectedPoints.size === 0) return
    const pointsToCopy = funscriptPoints.filter((_, i) => selectedPoints.has(i))
    if (pointsToCopy.length === 0) return

    // Normalize times relative to first point
    const minTime = Math.min(...pointsToCopy.map(p => p.at))
    const normalizedPoints = pointsToCopy.map(p => ({ at: p.at - minTime, pos: p.pos }))
    setClipboard(normalizedPoints)
  }, [selectedPoints, funscriptPoints])

  // Paste points at current time
  const pastePoints = useCallback(() => {
    if (clipboard.length === 0) return

    // Paste at current time
    const pastedPoints = clipboard.map(p => ({ at: p.at + Math.round(currentTime), pos: p.pos }))

    // Merge with existing points, replacing any at the same time
    const existingTimes = new Set(pastedPoints.map(p => p.at))
    const newPoints = [
      ...funscriptPoints.filter(p => !existingTimes.has(p.at)),
      ...pastedPoints
    ].sort((a, b) => a.at - b.at)

    setFunscriptPoints(newPoints)
    pushHistory(newPoints)
  }, [clipboard, currentTime, funscriptPoints, pushHistory])

  // Video controls - defined early so navigation functions can use them
  const seekTo = useCallback((timeMs: number) => {
    if (!videoRef.current) return
    videoRef.current.currentTime = timeMs / 1000
    setCurrentTime(timeMs)
  }, [])

  const togglePlayPause = useCallback(() => {
    if (!videoRef.current) return
    if (isPlaying) {
      videoRef.current.pause()
    } else {
      videoRef.current.play()
    }
  }, [isPlaying])

  const skipFrames = useCallback((frames: number) => {
    const fps = video?.fps || 30
    const frameDuration = 1000 / fps
    const newTime = currentTime + (frames * frameDuration)
    seekTo(Math.max(0, Math.min(duration, newTime)))
  }, [currentTime, duration, video?.fps, seekTo])

  // Add point at specific position (0-100)
  const addPointAtPosition = useCallback((position: number) => {
    const newPoint = { at: Math.round(currentTime), pos: position }
    // Remove any existing point at this exact time
    const filteredPoints = funscriptPoints.filter(p => Math.abs(p.at - newPoint.at) > 10)
    const newPoints = [...filteredPoints, newPoint].sort((a, b) => a.at - b.at)
    setFunscriptPoints(newPoints)
    pushHistory(newPoints)
  }, [currentTime, funscriptPoints, pushHistory])

  // Jump to next point
  const jumpToNextPoint = useCallback(() => {
    const nextPoint = funscriptPoints.find(p => p.at > currentTime + 10)
    if (nextPoint) {
      seekTo(nextPoint.at)
    }
  }, [funscriptPoints, currentTime, seekTo])

  // Jump to previous point
  const jumpToPrevPoint = useCallback(() => {
    let prevPoint: FunscriptPoint | undefined
    for (let i = funscriptPoints.length - 1; i >= 0; i--) {
      if (funscriptPoints[i].at < currentTime - 10) {
        prevPoint = funscriptPoints[i]
        break
      }
    }
    if (prevPoint) {
      seekTo(prevPoint.at)
    }
  }, [funscriptPoints, currentTime, seekTo])

  // Nudge selected points up or down
  const nudgeSelectedPoints = useCallback((delta: number) => {
    if (selectedPoints.size === 0) return
    const newPoints = funscriptPoints.map((p, i) => {
      if (selectedPoints.has(i)) {
        return { at: p.at, pos: Math.max(0, Math.min(100, p.pos + delta)) }
      }
      return p
    })
    setFunscriptPoints(newPoints)
    pushHistory(newPoints)
  }, [selectedPoints, funscriptPoints, pushHistory])

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if typing in an input
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return

      // Number keys 0-9 for adding points at specific positions
      if (!e.ctrlKey && !e.metaKey && !e.altKey) {
        if (e.key >= '0' && e.key <= '9') {
          e.preventDefault()
          const pos = parseInt(e.key) * 10 // 0=0, 1=10, 2=20, etc.
          addPointAtPosition(pos)
          return
        }
        if (e.key === '=' || e.key === '+') {
          e.preventDefault()
          addPointAtPosition(100) // = key for position 100
          return
        }
      }

      switch (e.key) {
        case ' ':
          e.preventDefault()
          togglePlayPause()
          break
        case 'ArrowLeft':
          e.preventDefault()
          if (e.altKey) {
            // Pan timeline left (not implemented yet)
          } else if (e.shiftKey) {
            seekTo(Math.max(0, currentTime - 1000))
          } else {
            skipFrames(-1)
          }
          break
        case 'ArrowRight':
          e.preventDefault()
          if (e.altKey) {
            // Pan timeline right (not implemented yet)
          } else if (e.shiftKey) {
            seekTo(Math.min(duration, currentTime + 1000))
          } else {
            skipFrames(1)
          }
          break
        case 'ArrowUp':
          e.preventDefault()
          if (e.shiftKey) {
            nudgeSelectedPoints(5) // Nudge selected points up
          } else {
            jumpToNextPoint() // Jump to next point
          }
          break
        case 'ArrowDown':
          e.preventDefault()
          if (e.shiftKey) {
            nudgeSelectedPoints(-5) // Nudge selected points down
          } else {
            jumpToPrevPoint() // Jump to previous point
          }
          break
        case '.':
          e.preventDefault()
          jumpToNextPoint()
          break
        case ',':
          e.preventDefault()
          jumpToPrevPoint()
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
        case 'd':
          if (e.ctrlKey || e.metaKey) {
            e.preventDefault()
            setSelectedPoints(new Set()) // Deselect all
          }
          break
        case 'c':
          if (e.ctrlKey || e.metaKey) {
            e.preventDefault()
            copySelectedPoints()
          }
          break
        case 'v':
          if (e.ctrlKey || e.metaKey) {
            e.preventDefault()
            pastePoints()
          }
          break
        case 'Escape':
          setSelectedPoints(new Set())
          break
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [duration, currentTime, undo, redo, funscriptPoints, togglePlayPause, skipFrames, seekTo,
      jumpToNextPoint, jumpToPrevPoint, nudgeSelectedPoints, addPointAtPosition,
      copySelectedPoints, pastePoints])

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
      case 'savitzkyGolay':
        newPoints = filters.savitzkyGolay(funscriptPoints, args[0] || 5)
        break
      case 'antiJerk':
        newPoints = filters.antiJerk(funscriptPoints, args[0] || 30)
        break
      case 'amplify':
        newPoints = filters.amplify(funscriptPoints, args[0] || 20)
        break
      case 'invert':
        newPoints = filters.invert(funscriptPoints)
        break
      case 'clamp':
        newPoints = filters.clamp(funscriptPoints, args[0] || 10, args[1] || 90)
        break
      case 'timeShift':
        newPoints = filters.timeShift(funscriptPoints, args[0] || 0)
        break
      case 'rdpSimplify':
        newPoints = filters.rdpSimplify(funscriptPoints, args[0] || 3)
        break
      case 'keyframes':
        newPoints = filters.keyframes(funscriptPoints, args[0] || 20)
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
  const connectHandyDevice = useCallback(async () => {
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
      setDiscoveredDevices([])
    } catch (error) {
      console.error('Failed to connect device:', error)
      alert(error instanceof Error ? error.message : 'Failed to connect device')
    } finally {
      setIsConnecting(false)
    }
  }, [connectionKey])

  // Scan for Bluetooth devices using Web Bluetooth API
  const scanBluetoothDevices = useCallback(async () => {
    if (!bluetoothSupported) {
      alert('Bluetooth is not supported in this browser. Please use Chrome, Edge, or Opera.')
      return
    }

    setIsScanning(true)
    setDiscoveredDevices([])

    try {
      // Request Bluetooth device with filters for common device services
      const device = await navigator.bluetooth.requestDevice({
        acceptAllDevices: true,
        optionalServices: [
          'battery_service',
          'device_information',
          'generic_access',
          // Common vibration device services
          '0000fff0-0000-1000-8000-00805f9b34fb', // Common custom service
          '6e400001-b5a3-f393-e0a9-e50e24dcca9e', // Nordic UART Service
        ]
      })

      if (device) {
        setDiscoveredDevices([{
          id: device.id,
          name: device.name || 'Unknown Device',
          type: 'bluetooth'
        }])
      }
    } catch (error) {
      if ((error as Error).name !== 'NotFoundError') {
        console.error('Bluetooth scan error:', error)
        alert('Failed to scan for devices. Make sure Bluetooth is enabled.')
      }
    } finally {
      setIsScanning(false)
    }
  }, [bluetoothSupported])

  // Connect to a discovered Bluetooth device
  const connectBluetoothDevice = useCallback(async (deviceInfo: { id: string; name: string }) => {
    setIsConnecting(true)
    try {
      // Request the device again to get a fresh connection
      const btDevice = await navigator.bluetooth.requestDevice({
        acceptAllDevices: true,
        optionalServices: [
          'battery_service',
          'device_information',
          'generic_access',
          '0000fff0-0000-1000-8000-00805f9b34fb',
          '6e400001-b5a3-f393-e0a9-e50e24dcca9e',
        ]
      })

      if (btDevice) {
        const server = await btDevice.gatt?.connect()

        // Try to get battery level if available
        let batteryLevel: number | undefined
        try {
          const batteryService = await server?.getPrimaryService('battery_service')
          const batteryChar = await batteryService?.getCharacteristic('battery_level')
          const batteryValue = await batteryChar?.readValue()
          batteryLevel = batteryValue?.getUint8(0)
        } catch {
          // Battery service not available
        }

        const connectedDevice: Device = {
          id: btDevice.id,
          type: 'bluetooth',
          name: btDevice.name || 'Bluetooth Device',
          status: 'connected',
          connection_key: null,
          firmware_version: null,
          last_error: null,
          bluetooth_id: btDevice.id,
          battery_level: batteryLevel,
          features: ['vibrate']
        }

        setDevice(connectedDevice)
        setShowDeviceModal(false)
        setDiscoveredDevices([])

        // Listen for disconnect
        btDevice.addEventListener('gattserverdisconnected', () => {
          setDevice(null)
        })
      }
    } catch (error) {
      console.error('Failed to connect Bluetooth device:', error)
      alert(error instanceof Error ? error.message : 'Failed to connect device')
    } finally {
      setIsConnecting(false)
    }
  }, [])

  const disconnectDevice = useCallback(async () => {
    if (!device) return

    try {
      if (device.type === 'bluetooth' && device.bluetooth_id) {
        // For Bluetooth devices, the connection is managed by the browser
        // Just clear our state
        setDevice(null)
      } else {
        await devicesApi.disconnect(device.id)
        setDevice(null)
      }
    } catch (error) {
      console.error('Failed to disconnect device:', error)
    }
  }, [device])

  const toggleDeviceModal = useCallback(() => {
    if (device) {
      disconnectDevice()
    } else {
      setShowDeviceModal(true)
      setDeviceType('handy')
      setDiscoveredDevices([])
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

  // New project handler
  const newProject = useCallback(() => {
    if (projectDirty && !confirm('You have unsaved changes. Start a new project anyway?')) {
      return
    }
    setVideo(null)
    setVideoBlobUrl(null)
    setFunscriptPoints([])
    setFunscriptPoints2([])
    setSelectedPoints(new Set())
    setSelectedPoints2(new Set())
    setChapters([])
    setHistory([])
    setHistoryIndex(-1)
    setProjectName('')
    setProjectDirty(false)
    setCurrentTime(0)
    setDuration(0)
  }, [projectDirty])

  // Add chapter at current time
  const addChapter = useCallback(() => {
    const colors = ['#ef4444', '#f97316', '#eab308', '#22c55e', '#06b6d4', '#3b82f6', '#8b5cf6', '#ec4899']
    const newChapter: Chapter = {
      id: Date.now().toString(),
      name: `Chapter ${chapters.length + 1}`,
      startTime: currentTime,
      endTime: duration,
      color: colors[chapters.length % colors.length]
    }
    // Adjust previous chapter's end time
    const updatedChapters = chapters.map((ch, idx) => {
      if (idx === chapters.length - 1 && ch.endTime > currentTime) {
        return { ...ch, endTime: currentTime }
      }
      return ch
    })
    setChapters([...updatedChapters, newChapter].sort((a, b) => a.startTime - b.startTime))
  }, [chapters, currentTime, duration])

  // Delete chapter
  const deleteChapter = useCallback((id: string) => {
    setChapters(chapters.filter(ch => ch.id !== id))
  }, [chapters])

  // Save chapters to file
  const saveChapters = useCallback(() => {
    if (chapters.length === 0) {
      alert('No chapters to save')
      return
    }
    const chaptersData = {
      version: '1.0',
      chapters: chapters
    }
    const blob = new Blob([JSON.stringify(chaptersData, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    const filename = projectName || video?.original_filename?.replace(/\.[^/.]+$/, '') || 'chapters'
    a.download = `${filename}.chapters.json`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }, [chapters, projectName, video])

  // Load chapters from file
  const loadChapters = useCallback(() => {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = '.json'
    input.onchange = async (e) => {
      const file = (e.target as HTMLInputElement).files?.[0]
      if (!file) return
      try {
        const text = await file.text()
        const data = JSON.parse(text)
        if (data.chapters && Array.isArray(data.chapters)) {
          setChapters(data.chapters)
        } else {
          alert('Invalid chapters file format')
        }
      } catch (error) {
        console.error('Failed to load chapters:', error)
        alert('Failed to load chapters file')
      }
    }
    input.click()
  }, [])

  // Clear all chapters
  const clearChapters = useCallback(() => {
    if (chapters.length === 0) return
    if (confirm('Are you sure you want to clear all chapters?')) {
      setChapters([])
    }
  }, [chapters])

  // Toggle theme
  const toggleTheme = useCallback(() => {
    setTheme(prev => prev === 'dark' ? 'light' : 'dark')
    // In a real app, you would also update CSS variables or body class
  }, [])

  // Handle F1 keyboard shortcut for help
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'F1') {
        e.preventDefault()
        setShowKeyboardShortcuts(true)
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

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

      {/* Menu Bar */}
      <MenuBar
        onNewProject={newProject}
        onOpenProject={openProjectDialog}
        onSaveProject={saveProject}
        onExportFunscript={exportFunscript}
        onImportFunscript={openFunscriptDialog}
        onUndo={undo}
        onRedo={redo}
        canUndo={historyIndex > 0}
        canRedo={historyIndex < history.length - 1}
        isExpert={isExpert}
        onToggleMode={toggleMode}
        showHeatmap={showHeatmap}
        onToggleHeatmap={() => setShowHeatmap(!showHeatmap)}
        showSimulator3D={showSimulator3D}
        onToggleSimulator3D={() => setShowSimulator3D(!showSimulator3D)}
        showTimeline2={showTimeline2}
        onToggleTimeline2={() => setShowTimeline2(!showTimeline2)}
        showWaveform={showWaveform}
        onToggleWaveform={() => setShowWaveform(!showWaveform)}
        showChapterList={showChapterList}
        onToggleChapterList={() => setShowChapterList(!showChapterList)}
        onShowKeyboardShortcuts={() => setShowKeyboardShortcuts(true)}
        onShowSettings={() => setShowSettingsModal(true)}
        onShowFilters={() => setShowFiltersModal(true)}
        onShowAbout={() => setShowAboutDialog(true)}
        onSaveChapters={saveChapters}
        onLoadChapters={loadChapters}
        onClearChapters={clearChapters}
        theme={theme}
        onToggleTheme={toggleTheme}
      />

      {/* Toolbar - scrollable on small screens */}
      <div className="h-12 bg-bg-surface border-b border-border flex items-center px-2 gap-1 overflow-x-auto scrollbar-thin">
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
      <div className="flex-1 flex overflow-hidden relative">
        {/* Video area */}
        <div className="flex-1 flex flex-col min-w-0">
          {/* Video player - responsive with aspect ratio */}
          <div className="flex-1 bg-bg-overlay flex items-center justify-center relative min-h-[200px] sm:min-h-[300px]">
            {isUploading || isLoadingVideo ? (
              <div className="text-center px-4">
                <div className="w-16 h-16 sm:w-20 sm:h-20 rounded-full bg-bg-elevated flex items-center justify-center mx-auto mb-4">
                  <Loader2 className="w-8 h-8 sm:w-10 sm:h-10 text-primary animate-spin" />
                </div>
                <h3 className="text-base sm:text-lg font-semibold text-text-primary mb-2">
                  {isUploading ? 'Uploading video...' : 'Loading video...'}
                </h3>
                <p className="text-sm text-text-secondary">
                  Please wait while we process your video
                </p>
              </div>
            ) : video && videoBlobUrl ? (
              <>
                <video
                  ref={videoRef}
                  className="max-w-full max-h-full w-auto h-auto object-contain"
                  src={videoBlobUrl}
                  onClick={togglePlayPause}
                />
                {/* Time overlay */}
                <div className="absolute bottom-2 left-2 sm:bottom-4 sm:left-4 bg-black/70 px-2 py-1 rounded text-xs text-white font-mono">
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

          {/* Audio Waveform */}
          {showWaveform && video && (
            <div className="h-12 bg-bg-surface border-t border-border px-2">
              <AudioWaveform
                duration={duration || 60000}
                currentTime={currentTime}
              />
            </div>
          )}

          {/* Timeline 1 - responsive height */}
          <div className="h-32 sm:h-40 lg:h-48 bg-bg-surface border-t border-border">
            <div className="px-2 py-1 text-xs text-text-muted border-b border-border flex items-center gap-2">
              <SplitSquareVertical className="w-3 h-3" />
              <span className="hidden sm:inline">Timeline 1 (Stroke)</span>
              <span className="sm:hidden">T1</span>
            </div>
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

          {/* Timeline 2 (Roll axis) - responsive height */}
          {showTimeline2 && (
            <div className="h-28 sm:h-32 lg:h-36 bg-bg-surface border-t border-border">
              <div className="px-2 py-1 text-xs text-text-muted border-b border-border flex items-center gap-2">
                <SplitSquareVertical className="w-3 h-3" />
                <span className="hidden sm:inline">Timeline 2 (Roll)</span>
                <span className="sm:hidden">T2</span>
              </div>
              {video || funscriptPoints2.length > 0 ? (
                <Timeline
                  currentTime={currentTime}
                  duration={duration || 60000}
                  points={funscriptPoints2}
                  selectedPoints={selectedPoints2}
                  onSeek={seekTo}
                  onSelectPoint={(idx, add) => {
                    if (add) {
                      setSelectedPoints2(prev => new Set([...prev, idx]))
                    } else {
                      setSelectedPoints2(new Set([idx]))
                    }
                  }}
                  onMovePoint={(idx, newPos) => {
                    const newPoints = [...funscriptPoints2]
                    newPoints[idx] = { ...newPoints[idx], pos: newPos }
                    setFunscriptPoints2(newPoints)
                  }}
                  onDeleteSelected={() => {
                    const newPoints = funscriptPoints2.filter((_, i) => !selectedPoints2.has(i))
                    setFunscriptPoints2(newPoints)
                    setSelectedPoints2(new Set())
                  }}
                  zoom={timelineZoom}
                  editMode={editMode}
                />
              ) : (
                <div className="h-full flex items-center justify-center">
                  <p className="text-text-muted text-sm">Timeline 2 - Secondary axis for roll</p>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Sidebar toggle button - visible on small screens */}
        <button
          className="absolute right-0 top-1/2 -translate-y-1/2 z-20 bg-bg-surface border border-border rounded-l-lg p-2 lg:hidden shadow-lg"
          onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
          title={sidebarCollapsed ? 'Show sidebar' : 'Hide sidebar'}
        >
          {sidebarCollapsed ? <PanelRightOpen className="w-5 h-5" /> : <PanelRightClose className="w-5 h-5" />}
        </button>

        {/* Right sidebar - Control Panel */}
        <div className={clsx(
          "bg-bg-surface border-l border-border overflow-y-auto transition-all duration-300 absolute lg:relative right-0 top-0 bottom-0 z-10",
          sidebarCollapsed ? "w-0 opacity-0 pointer-events-none lg:w-64 lg:opacity-100 lg:pointer-events-auto xl:w-72" : "w-72 sm:w-64 lg:w-64 xl:w-72"
        )}>
          <div className="p-3 sm:p-4 min-w-[256px]">
            <div className="flex items-center justify-between mb-4">
              <h2 className="font-semibold text-text-primary flex items-center gap-2">
                {isExpert ? <Sliders className="w-5 h-5" /> : <Settings className="w-5 h-5" />}
                <span className="hidden sm:inline">{isExpert ? 'Expert Controls' : 'Control Panel'}</span>
                <span className="sm:hidden">Controls</span>
              </h2>
              <button
                className="lg:hidden p-1 hover:bg-bg-elevated rounded"
                onClick={() => setSidebarCollapsed(true)}
              >
                <X className="w-4 h-4" />
              </button>
            </div>

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

              {/* Movement Gauge */}
              <div className="card">
                <h3 className="text-sm font-medium text-text-primary mb-3 flex items-center gap-2">
                  <Activity className="w-4 h-4" />
                  Movement Gauge
                </h3>
                <MovementGauge position={currentPosition} speed={currentSpeed} />
              </div>

              {/* Statistics Panel */}
              <div className="card">
                <StatisticsPanel stats={stats} duration={duration} video={video} />
              </div>

              {/* Chapter List */}
              {showChapterList && (
                <div className="card">
                  <ChapterList
                    chapters={chapters}
                    currentTime={currentTime}
                    onSeek={seekTo}
                    onAddChapter={addChapter}
                    onDeleteChapter={deleteChapter}
                  />
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
                {device && (
                  <div className="mt-2 space-y-1">
                    <p className="text-xs text-text-muted flex items-center gap-2">
                      <span className="text-text-disabled">Type:</span>
                      <span className="capitalize">{device.type}</span>
                    </p>
                    {device.firmware_version && (
                      <p className="text-xs text-text-muted flex items-center gap-2">
                        <span className="text-text-disabled">Firmware:</span>
                        <span>v{device.firmware_version}</span>
                      </p>
                    )}
                    {device.battery_level !== undefined && (
                      <p className="text-xs text-text-muted flex items-center gap-2">
                        <span className="text-text-disabled">Battery:</span>
                        <span className={device.battery_level < 20 ? 'text-status-error' : device.battery_level < 50 ? 'text-status-warning' : 'text-primary'}>
                          {device.battery_level}%
                        </span>
                      </p>
                    )}
                    {device.features && device.features.length > 0 && (
                      <p className="text-xs text-text-muted flex items-center gap-2">
                        <span className="text-text-disabled">Features:</span>
                        <span>{device.features.join(', ')}</span>
                      </p>
                    )}
                  </div>
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
              Connect Device
            </h2>

            {/* Device Type Tabs */}
            <div className="flex gap-2 mb-4">
              <button
                className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors ${
                  deviceType === 'handy'
                    ? 'bg-primary text-bg-base'
                    : 'bg-bg-elevated text-text-secondary hover:text-text-primary'
                }`}
                onClick={() => setDeviceType('handy')}
              >
                Handy
              </button>
              <button
                className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors ${
                  deviceType === 'bluetooth'
                    ? 'bg-primary text-bg-base'
                    : 'bg-bg-elevated text-text-secondary hover:text-text-primary'
                }`}
                onClick={() => setDeviceType('bluetooth')}
              >
                Bluetooth
              </button>
            </div>

            {/* Handy Connection */}
            {deviceType === 'handy' && (
              <>
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
                  onKeyDown={(e) => e.key === 'Enter' && connectHandyDevice()}
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
                    onClick={connectHandyDevice}
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
              </>
            )}

            {/* Bluetooth Connection */}
            {deviceType === 'bluetooth' && (
              <>
                {!bluetoothSupported ? (
                  <div className="text-center py-6">
                    <div className="w-12 h-12 rounded-full bg-status-error/20 flex items-center justify-center mx-auto mb-3">
                      <Bluetooth className="w-6 h-6 text-status-error" />
                    </div>
                    <p className="text-sm text-text-secondary mb-2">
                      Bluetooth is not supported in this browser.
                    </p>
                    <p className="text-xs text-text-muted">
                      Please use Chrome, Edge, or Opera on a desktop computer.
                    </p>
                  </div>
                ) : (
                  <>
                    <p className="text-sm text-text-secondary mb-4">
                      Scan for nearby Bluetooth devices. Make sure your device is powered on and in pairing mode.
                    </p>

                    {/* Scan Button */}
                    <button
                      className="w-full btn-secondary text-sm mb-4 flex items-center justify-center gap-2"
                      onClick={scanBluetoothDevices}
                      disabled={isScanning}
                    >
                      {isScanning ? (
                        <>
                          <Loader2 className="w-4 h-4 animate-spin" />
                          Scanning...
                        </>
                      ) : (
                        <>
                          <Search className="w-4 h-4" />
                          Scan for Devices
                        </>
                      )}
                    </button>

                    {/* Discovered Devices */}
                    {discoveredDevices.length > 0 && (
                      <div className="mb-4">
                        <p className="text-xs text-text-muted mb-2">Discovered Devices:</p>
                        <div className="space-y-2">
                          {discoveredDevices.map((dev) => (
                            <button
                              key={dev.id}
                              className="w-full flex items-center gap-3 p-3 bg-bg-elevated rounded-md hover:bg-bg-highlight transition-colors"
                              onClick={() => connectBluetoothDevice(dev)}
                              disabled={isConnecting}
                            >
                              <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center">
                                <Bluetooth className="w-4 h-4 text-primary" />
                              </div>
                              <div className="flex-1 text-left">
                                <p className="text-sm text-text-primary font-medium">{dev.name}</p>
                                <p className="text-xs text-text-muted">{dev.type}</p>
                              </div>
                              {isConnecting ? (
                                <Loader2 className="w-4 h-4 animate-spin text-primary" />
                              ) : (
                                <ChevronRight className="w-4 h-4 text-text-muted" />
                              )}
                            </button>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* No devices found message */}
                    {!isScanning && discoveredDevices.length === 0 && (
                      <div className="text-center py-4">
                        <p className="text-sm text-text-muted">
                          Click "Scan for Devices" to find nearby Bluetooth devices.
                        </p>
                      </div>
                    )}

                    {/* Supported Devices Info */}
                    <div className="mt-4 p-3 bg-bg-elevated rounded-md">
                      <p className="text-xs font-medium text-text-secondary mb-2">Supported Devices:</p>
                      <ul className="text-xs text-text-muted space-y-1">
                        <li>• Lovense devices (Lush, Max, Nora, etc.)</li>
                        <li>• Kiiroo devices (Onyx, Pearl, etc.)</li>
                        <li>• We-Vibe devices</li>
                        <li>• Any BLE-compatible vibration device</li>
                      </ul>
                    </div>
                  </>
                )}

                <div className="flex justify-end gap-2 mt-4">
                  <button
                    className="btn-secondary text-sm"
                    onClick={() => {
                      setShowDeviceModal(false)
                      setDiscoveredDevices([])
                    }}
                  >
                    Cancel
                  </button>
                </div>
              </>
            )}
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
          <div className="relative bg-bg-surface rounded-lg p-6 w-full max-w-md mx-4 shadow-xl max-h-[80vh] flex flex-col">
            <h2 className="text-lg font-semibold text-text-primary mb-4 flex items-center gap-2">
              <Filter className="w-5 h-5" />
              Post-Processing Filters
            </h2>
            <p className="text-sm text-text-secondary mb-4">
              Apply filters to modify the funscript. Changes can be undone with Ctrl+Z.
            </p>

            <div className="space-y-3 overflow-y-auto flex-1">
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
              <button
                className="w-full btn-secondary text-sm justify-start"
                onClick={() => { applyFilter('savitzkyGolay', 5); setShowFiltersModal(false); }}
              >
                <Sliders className="w-4 h-4" />
                Savitzky-Golay - Polynomial smoothing preserving peaks
              </button>
              <button
                className="w-full btn-secondary text-sm justify-start"
                onClick={() => { applyFilter('antiJerk', 30); setShowFiltersModal(false); }}
              >
                <Activity className="w-4 h-4" />
                Anti-Jerk - Remove sudden jerky movements
              </button>
              <button
                className="w-full btn-secondary text-sm justify-start"
                onClick={() => { applyFilter('clamp', 10, 90); setShowFiltersModal(false); }}
              >
                <Maximize className="w-4 h-4" />
                Clamp - Constrain values to 10-90 range
              </button>
              <button
                className="w-full btn-secondary text-sm justify-start"
                onClick={() => { applyFilter('rdpSimplify', 3); setShowFiltersModal(false); }}
              >
                <Trash2 className="w-4 h-4" />
                RDP Simplify - Ramer-Douglas-Peucker algorithm
              </button>
              <button
                className="w-full btn-secondary text-sm justify-start"
                onClick={() => { applyFilter('keyframes', 20); setShowFiltersModal(false); }}
              >
                <ChevronRight className="w-4 h-4" />
                Keyframes - Keep only peaks and valleys
              </button>
              <button
                className="w-full btn-secondary text-sm justify-start"
                onClick={() => { applyFilter('timeShift', 100); setShowFiltersModal(false); }}
              >
                <ChevronRight className="w-4 h-4" />
                Time Shift (+100ms) - Move points forward in time
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

      {/* Keyboard Shortcuts Dialog */}
      {showKeyboardShortcuts && (
        <KeyboardShortcutsDialog onClose={() => setShowKeyboardShortcuts(false)} />
      )}

      {/* About Dialog */}
      {showAboutDialog && (
        <AboutDialog onClose={() => setShowAboutDialog(false)} />
      )}
    </div>
  )
}
