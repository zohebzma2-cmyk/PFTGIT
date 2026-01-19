import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Plus,
  FolderOpen,
  MoreVertical,
  Play,
  Clock,
  CheckCircle,
  Video,
  Trash2,
  Download,
  Copy,
  Upload,
  Search,
  FileText,
} from 'lucide-react'

// Types
interface FunscriptPoint {
  at: number
  pos: number
}

interface MediaItem {
  id: string
  uri: string
  filename: string
  mimeType: string
  duration: number
  width: number
  height: number
  thumbnailUri?: string
  createdAt: string
}

interface Project {
  id: string
  name: string
  description?: string
  status: 'draft' | 'finished'
  mediaId?: string
  media?: MediaItem
  points: FunscriptPoint[]
  duration: number
  createdAt: string
  updatedAt: string
  thumbnailUri?: string
}

type TabType = 'drafts' | 'finished' | 'media'

// Local storage helpers
const STORAGE_KEY = 'fungen_projects'
const MEDIA_STORAGE_KEY = 'fungen_media'

function loadProjects(): Project[] {
  try {
    const data = localStorage.getItem(STORAGE_KEY)
    return data ? JSON.parse(data) : []
  } catch {
    return []
  }
}

function saveProjects(projects: Project[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(projects))
}

function loadMedia(): MediaItem[] {
  try {
    const data = localStorage.getItem(MEDIA_STORAGE_KEY)
    return data ? JSON.parse(data) : []
  } catch {
    return []
  }
}

function saveMedia(media: MediaItem[]) {
  localStorage.setItem(MEDIA_STORAGE_KEY, JSON.stringify(media))
}

function exportFunscript(project: Project): string {
  return JSON.stringify({
    version: '1.0',
    inverted: false,
    range: 100,
    actions: project.points.map(p => ({ at: p.at, pos: p.pos })),
    metadata: {
      creator: 'FunGen',
      title: project.name,
      description: project.description || '',
      duration: project.duration,
    },
  }, null, 2)
}

function TabButton({
  label,
  count,
  active,
  onClick,
}: {
  label: string
  count: number
  active: boolean
  onClick: () => void
}) {
  return (
    <button
      onClick={onClick}
      className={`px-4 py-2 rounded-lg font-medium transition-colors flex items-center gap-2 ${
        active
          ? 'bg-primary text-bg-base'
          : 'bg-bg-elevated text-text-secondary hover:text-text-primary'
      }`}
    >
      {label}
      {count > 0 && (
        <span
          className={`px-2 py-0.5 rounded-full text-xs ${
            active ? 'bg-white/20' : 'bg-bg-surface'
          }`}
        >
          {count}
        </span>
      )}
    </button>
  )
}

function ProjectCard({
  project,
  onOpen,
  onDelete,
  onExport,
  onDuplicate,
}: {
  project: Project
  onOpen: () => void
  onDelete: () => void
  onExport: () => void
  onDuplicate: () => void
}) {
  const [showMenu, setShowMenu] = useState(false)

  const formatDuration = (ms: number) => {
    const seconds = Math.floor(ms / 1000)
    const minutes = Math.floor(seconds / 60)
    const remainingSeconds = seconds % 60
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`
  }

  return (
    <div className="card group hover:border-primary/50 transition-all relative">
      {/* Thumbnail */}
      <div
        className="aspect-video bg-bg-overlay rounded-md mb-4 flex items-center justify-center group-hover:bg-bg-elevated transition-colors cursor-pointer"
        onClick={onOpen}
      >
        {project.status === 'finished' ? (
          <CheckCircle className="w-12 h-12 text-status-success" />
        ) : (
          <FileText className="w-12 h-12 text-text-disabled" />
        )}
      </div>

      {/* Info */}
      <div className="flex items-start justify-between">
        <div className="flex-1 min-w-0 cursor-pointer" onClick={onOpen}>
          <h3 className="font-semibold text-text-primary truncate">{project.name}</h3>
          <p className="text-sm text-text-muted">
            {project.points.length} points • {formatDuration(project.duration)}
          </p>
          <div className="flex items-center gap-1 mt-2 text-xs text-text-disabled">
            <Clock className="w-3 h-3" />
            <span>{new Date(project.updatedAt).toLocaleDateString()}</span>
          </div>
        </div>

        {/* Actions */}
        <div className="relative">
          <button
            className="btn-icon"
            onClick={() => setShowMenu(!showMenu)}
            title="More"
          >
            <MoreVertical className="w-4 h-4" />
          </button>

          {showMenu && (
            <>
              <div className="fixed inset-0 z-10" onClick={() => setShowMenu(false)} />
              <div className="absolute right-0 top-8 z-20 bg-bg-elevated border border-border rounded-lg shadow-lg py-1 min-w-[140px]">
                <button
                  className="w-full px-3 py-2 text-left text-sm hover:bg-bg-surface flex items-center gap-2"
                  onClick={() => {
                    onOpen()
                    setShowMenu(false)
                  }}
                >
                  <Play className="w-4 h-4" />
                  Open
                </button>
                <button
                  className="w-full px-3 py-2 text-left text-sm hover:bg-bg-surface flex items-center gap-2"
                  onClick={() => {
                    onExport()
                    setShowMenu(false)
                  }}
                >
                  <Download className="w-4 h-4" />
                  Export
                </button>
                <button
                  className="w-full px-3 py-2 text-left text-sm hover:bg-bg-surface flex items-center gap-2"
                  onClick={() => {
                    onDuplicate()
                    setShowMenu(false)
                  }}
                >
                  <Copy className="w-4 h-4" />
                  Duplicate
                </button>
                <hr className="my-1 border-border" />
                <button
                  className="w-full px-3 py-2 text-left text-sm hover:bg-bg-surface flex items-center gap-2 text-status-error"
                  onClick={() => {
                    onDelete()
                    setShowMenu(false)
                  }}
                >
                  <Trash2 className="w-4 h-4" />
                  Delete
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}

function MediaCard({
  media,
  onSelect,
  onDelete,
}: {
  media: MediaItem
  onSelect: () => void
  onDelete: () => void
}) {
  const formatDuration = (ms: number) => {
    const seconds = Math.floor(ms / 1000)
    const minutes = Math.floor(seconds / 60)
    const remainingSeconds = seconds % 60
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`
  }

  return (
    <div className="card group hover:border-primary/50 transition-all relative">
      <div
        className="aspect-video bg-bg-overlay rounded-md mb-3 flex items-center justify-center group-hover:bg-bg-elevated transition-colors cursor-pointer"
        onClick={onSelect}
      >
        <Video className="w-10 h-10 text-text-disabled" />
      </div>
      <p className="font-medium text-text-primary truncate text-sm" title={media.filename}>
        {media.filename}
      </p>
      <p className="text-xs text-text-muted mt-1">{formatDuration(media.duration)}</p>
      <button
        className="absolute top-2 right-2 p-1.5 bg-bg-elevated/80 rounded-md opacity-0 group-hover:opacity-100 transition-opacity hover:bg-status-error/20"
        onClick={(e) => {
          e.stopPropagation()
          onDelete()
        }}
        title="Delete"
      >
        <Trash2 className="w-4 h-4 text-status-error" />
      </button>
    </div>
  )
}

function EmptyState({
  icon: Icon,
  title,
  subtitle,
  actionLabel,
  onAction,
}: {
  icon: React.ElementType
  title: string
  subtitle: string
  actionLabel?: string
  onAction?: () => void
}) {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-center">
      <div className="w-16 h-16 rounded-full bg-bg-elevated flex items-center justify-center mb-4">
        <Icon className="w-8 h-8 text-text-muted" />
      </div>
      <h2 className="text-lg font-semibold text-text-primary mb-2">{title}</h2>
      <p className="text-text-secondary mb-6 max-w-sm">{subtitle}</p>
      {actionLabel && onAction && (
        <button onClick={onAction} className="btn-primary">
          <Plus className="w-5 h-5" />
          {actionLabel}
        </button>
      )}
    </div>
  )
}

export default function ProjectsPage() {
  const navigate = useNavigate()
  const [activeTab, setActiveTab] = useState<TabType>('drafts')
  const [projects, setProjects] = useState<Project[]>([])
  const [mediaLibrary, setMediaLibrary] = useState<MediaItem[]>([])
  const [searchQuery, setSearchQuery] = useState('')

  // Load data from localStorage
  useEffect(() => {
    setProjects(loadProjects())
    setMediaLibrary(loadMedia())
  }, [])

  const drafts = projects.filter((p) => p.status === 'draft')
  const finished = projects.filter((p) => p.status === 'finished')

  const filteredItems = () => {
    const query = searchQuery.toLowerCase()
    if (activeTab === 'drafts') {
      return query ? drafts.filter((p) => p.name.toLowerCase().includes(query)) : drafts
    }
    if (activeTab === 'finished') {
      return query ? finished.filter((p) => p.name.toLowerCase().includes(query)) : finished
    }
    return query
      ? mediaLibrary.filter((m) => m.filename.toLowerCase().includes(query))
      : mediaLibrary
  }

  const handleCreateProject = (name?: string, media?: MediaItem) => {
    const projectName = name || `Untitled Project ${projects.length + 1}`
    const now = new Date().toISOString()
    const newProject: Project = {
      id: `project-${Date.now()}`,
      name: projectName,
      status: 'draft',
      mediaId: media?.id,
      media,
      points: [],
      duration: media?.duration || 0,
      createdAt: now,
      updatedAt: now,
    }

    const updatedProjects = [newProject, ...projects]
    setProjects(updatedProjects)
    saveProjects(updatedProjects)

    // Store current project ID in sessionStorage for editor
    sessionStorage.setItem('currentProjectId', newProject.id)
    navigate('/editor')
  }

  const handleOpenProject = (project: Project) => {
    sessionStorage.setItem('currentProjectId', project.id)
    navigate('/editor')
  }

  const handleDeleteProject = (project: Project) => {
    if (confirm(`Delete "${project.name}"? This cannot be undone.`)) {
      const updatedProjects = projects.filter((p) => p.id !== project.id)
      setProjects(updatedProjects)
      saveProjects(updatedProjects)
    }
  }

  const handleExportProject = (project: Project) => {
    const funscriptJson = exportFunscript(project)
    const blob = new Blob([funscriptJson], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${project.name.replace(/[^a-z0-9]/gi, '_')}.funscript`
    a.click()
    URL.revokeObjectURL(url)
  }

  const handleDuplicateProject = (project: Project) => {
    const now = new Date().toISOString()
    const duplicate: Project = {
      ...project,
      id: `project-${Date.now()}`,
      name: `${project.name} (Copy)`,
      status: 'draft',
      createdAt: now,
      updatedAt: now,
    }
    const updatedProjects = [duplicate, ...projects]
    setProjects(updatedProjects)
    saveProjects(updatedProjects)
  }

  const handleAddMedia = () => {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = 'video/*'
    input.onchange = async (e) => {
      const file = (e.target as HTMLInputElement).files?.[0]
      if (file) {
        // Create object URL for the video
        const uri = URL.createObjectURL(file)

        // Get video duration
        const video = document.createElement('video')
        video.src = uri
        video.onloadedmetadata = () => {
          const media: MediaItem = {
            id: `media-${Date.now()}`,
            uri,
            filename: file.name,
            mimeType: file.type,
            duration: video.duration * 1000,
            width: video.videoWidth,
            height: video.videoHeight,
            createdAt: new Date().toISOString(),
          }

          const updatedMedia = [media, ...mediaLibrary]
          setMediaLibrary(updatedMedia)
          saveMedia(updatedMedia)

          if (confirm('Create a new project with this video?')) {
            handleCreateProject(file.name.replace(/\.[^/.]+$/, ''), media)
          }
        }
      }
    }
    input.click()
  }

  const handleDeleteMedia = (media: MediaItem) => {
    if (confirm(`Remove "${media.filename}" from your library?`)) {
      const updatedMedia = mediaLibrary.filter((m) => m.id !== media.id)
      setMediaLibrary(updatedMedia)
      saveMedia(updatedMedia)
    }
  }

  const handleSelectMedia = (media: MediaItem) => {
    handleCreateProject(media.filename.replace(/\.[^/.]+$/, ''), media)
  }

  const renderContent = () => {
    const items = filteredItems()

    if (activeTab === 'drafts') {
      if (items.length === 0) {
        return (
          <EmptyState
            icon={FolderOpen}
            title="No Drafts"
            subtitle="Your work-in-progress projects will appear here"
            actionLabel="New Project"
            onAction={() => handleCreateProject()}
          />
        )
      }
      return (
        <div className="grid md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {(items as Project[]).map((project) => (
            <ProjectCard
              key={project.id}
              project={project}
              onOpen={() => handleOpenProject(project)}
              onDelete={() => handleDeleteProject(project)}
              onExport={() => handleExportProject(project)}
              onDuplicate={() => handleDuplicateProject(project)}
            />
          ))}
        </div>
      )
    }

    if (activeTab === 'finished') {
      if (items.length === 0) {
        return (
          <EmptyState
            icon={CheckCircle}
            title="No Finished Scripts"
            subtitle="Completed funscripts will appear here when you mark them as done"
          />
        )
      }
      return (
        <div className="grid md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {(items as Project[]).map((project) => (
            <ProjectCard
              key={project.id}
              project={project}
              onOpen={() => handleOpenProject(project)}
              onDelete={() => handleDeleteProject(project)}
              onExport={() => handleExportProject(project)}
              onDuplicate={() => handleDuplicateProject(project)}
            />
          ))}
        </div>
      )
    }

    if (activeTab === 'media') {
      if (items.length === 0) {
        return (
          <EmptyState
            icon={Video}
            title="No Videos"
            subtitle="Add videos to your library to create funscripts"
            actionLabel="Add Video"
            onAction={handleAddMedia}
          />
        )
      }
      return (
        <div className="grid md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
          {(items as MediaItem[]).map((media) => (
            <MediaCard
              key={media.id}
              media={media}
              onSelect={() => handleSelectMedia(media)}
              onDelete={() => handleDeleteMedia(media)}
            />
          ))}
        </div>
      )
    }

    return null
  }

  return (
    <div className="flex-1 overflow-auto p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-text-primary">Library</h1>
          <p className="text-text-secondary">Manage your projects and media</p>
        </div>
        <div className="flex items-center gap-3">
          {activeTab === 'media' ? (
            <button onClick={handleAddMedia} className="btn-primary">
              <Upload className="w-5 h-5" />
              Add Video
            </button>
          ) : (
            <button onClick={() => handleCreateProject()} className="btn-primary">
              <Plus className="w-5 h-5" />
              New Project
            </button>
          )}
        </div>
      </div>

      {/* Tabs & Search */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-6">
        <div className="flex gap-2">
          <TabButton
            label="Drafts"
            count={drafts.length}
            active={activeTab === 'drafts'}
            onClick={() => setActiveTab('drafts')}
          />
          <TabButton
            label="Finished"
            count={finished.length}
            active={activeTab === 'finished'}
            onClick={() => setActiveTab('finished')}
          />
          <TabButton
            label="Media"
            count={mediaLibrary.length}
            active={activeTab === 'media'}
            onClick={() => setActiveTab('media')}
          />
        </div>

        <div className="relative w-full sm:w-64">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-muted" />
          <input
            type="text"
            placeholder="Search..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-9 pr-4 py-2 bg-bg-elevated border border-border rounded-lg text-text-primary placeholder-text-muted focus:outline-none focus:border-primary"
          />
        </div>
      </div>

      {/* Content */}
      {renderContent()}
    </div>
  )
}
