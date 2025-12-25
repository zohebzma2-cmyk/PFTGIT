import { Link } from 'react-router-dom'
import {
  Upload,
  Wand2,
  Play,
  Zap,
  ArrowRight,
  FileText
} from 'lucide-react'

const features = [
  {
    icon: Upload,
    title: 'Upload Video',
    description: 'Support for MP4, MKV, AVI, and more formats',
  },
  {
    icon: Wand2,
    title: 'AI Generation',
    description: 'Automatic funscript creation using AI motion detection',
  },
  {
    icon: Play,
    title: 'Live Preview',
    description: 'Real-time playback with device synchronization',
  },
]

export default function HomePage() {
  return (
    <div className="flex-1 overflow-auto">
      {/* Hero Section */}
      <div className="gradient-spotify">
        <div className="max-w-5xl mx-auto px-6 py-16">
          <div className="flex items-center gap-2 text-primary mb-4">
            <Zap className="w-5 h-5" />
            <span className="text-sm font-medium">AI-Powered</span>
          </div>

          <h1 className="text-4xl md:text-5xl font-bold text-text-primary mb-4">
            Generate Funscripts
            <br />
            <span className="text-primary">Automatically</span>
          </h1>

          <p className="text-lg text-text-secondary max-w-xl mb-8">
            Transform your videos into interactive experiences with our
            AI-powered funscript generator. Fast, accurate, and easy to use.
          </p>

          <div className="flex flex-wrap gap-4">
            <Link to="/editor" className="btn-primary">
              <Upload className="w-5 h-5" />
              Start New Project
            </Link>
            <Link to="/projects" className="btn-secondary">
              <FileText className="w-5 h-5" />
              View Projects
            </Link>
          </div>
        </div>
      </div>

      {/* Features */}
      <div className="max-w-5xl mx-auto px-6 py-16">
        <h2 className="text-2xl font-bold text-text-primary mb-8">How It Works</h2>

        <div className="grid md:grid-cols-3 gap-6">
          {features.map(({ icon: Icon, title, description }, index) => (
            <div key={title} className="card group hover:border-primary/50 transition-colors">
              <div className="flex items-start gap-4">
                <div className="flex-shrink-0 w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center group-hover:bg-primary/20 transition-colors">
                  <Icon className="w-6 h-6 text-primary" />
                </div>
                <div>
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-xs font-medium text-primary bg-primary/10 px-2 py-0.5 rounded">
                      Step {index + 1}
                    </span>
                  </div>
                  <h3 className="font-semibold text-text-primary mb-1">{title}</h3>
                  <p className="text-sm text-text-secondary">{description}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Quick Actions */}
      <div className="max-w-5xl mx-auto px-6 pb-16">
        <div className="card bg-bg-elevated border-border">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-semibold text-text-primary mb-1">
                Ready to get started?
              </h3>
              <p className="text-text-secondary">
                Upload a video and let our AI do the work.
              </p>
            </div>
            <Link
              to="/editor"
              className="btn-primary"
            >
              Open Editor
              <ArrowRight className="w-4 h-4" />
            </Link>
          </div>
        </div>
      </div>
    </div>
  )
}
