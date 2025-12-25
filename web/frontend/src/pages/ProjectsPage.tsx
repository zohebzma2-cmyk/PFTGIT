import { useState } from 'react'
import { Link } from 'react-router-dom'
import {
  Plus,
  FolderOpen,
  MoreVertical,
  Play,
  Clock
} from 'lucide-react'

// Mock data - will be replaced with API calls
const mockProjects = [
  {
    id: '1',
    name: 'Sample Project 1',
    description: 'A test project',
    videoPath: '/uploads/video1.mp4',
    createdAt: '2024-01-15T10:00:00Z',
    updatedAt: '2024-01-15T12:30:00Z',
  },
]

export default function ProjectsPage() {
  const [projects] = useState(mockProjects)

  return (
    <div className="flex-1 overflow-auto p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-text-primary">Projects</h1>
          <p className="text-text-secondary">Manage your funscript projects</p>
        </div>
        <Link to="/editor" className="btn-primary">
          <Plus className="w-5 h-5" />
          New Project
        </Link>
      </div>

      {/* Projects Grid */}
      {projects.length > 0 ? (
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
          {projects.map((project) => (
            <div
              key={project.id}
              className="card group hover:border-primary/50 transition-all"
            >
              {/* Thumbnail */}
              <div className="aspect-video bg-bg-overlay rounded-md mb-4 flex items-center justify-center group-hover:bg-bg-elevated transition-colors">
                <FolderOpen className="w-12 h-12 text-text-disabled" />
              </div>

              {/* Info */}
              <div className="flex items-start justify-between">
                <div className="flex-1 min-w-0">
                  <h3 className="font-semibold text-text-primary truncate">
                    {project.name}
                  </h3>
                  <p className="text-sm text-text-muted truncate">
                    {project.description || 'No description'}
                  </p>
                  <div className="flex items-center gap-1 mt-2 text-xs text-text-disabled">
                    <Clock className="w-3 h-3" />
                    <span>
                      {new Date(project.updatedAt).toLocaleDateString()}
                    </span>
                  </div>
                </div>

                {/* Actions */}
                <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                  <Link
                    to={`/editor/${project.id}`}
                    className="btn-icon"
                    title="Open"
                  >
                    <Play className="w-4 h-4" />
                  </Link>
                  <button className="btn-icon" title="More">
                    <MoreVertical className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      ) : (
        /* Empty State */
        <div className="flex flex-col items-center justify-center py-16 text-center">
          <div className="w-16 h-16 rounded-full bg-bg-elevated flex items-center justify-center mb-4">
            <FolderOpen className="w-8 h-8 text-text-muted" />
          </div>
          <h2 className="text-lg font-semibold text-text-primary mb-2">
            No projects yet
          </h2>
          <p className="text-text-secondary mb-6 max-w-sm">
            Create your first project to start generating funscripts with AI.
          </p>
          <Link to="/editor" className="btn-primary">
            <Plus className="w-5 h-5" />
            Create Project
          </Link>
        </div>
      )}
    </div>
  )
}
