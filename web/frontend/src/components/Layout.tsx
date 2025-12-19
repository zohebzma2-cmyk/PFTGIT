import { Outlet, Link, useLocation } from 'react-router-dom'
import {
  Home,
  FolderOpen,
  Sliders,
  Settings,
  HelpCircle,
  Zap
} from 'lucide-react'
import { clsx } from 'clsx'

const navItems = [
  { path: '/', icon: Home, label: 'Home' },
  { path: '/projects', icon: FolderOpen, label: 'Projects' },
  { path: '/editor', icon: Sliders, label: 'Editor' },
]

export default function Layout() {
  const location = useLocation()

  return (
    <div className="min-h-screen bg-bg-base flex">
      {/* Sidebar */}
      <aside className="w-64 bg-bg-surface border-r border-border flex flex-col">
        {/* Logo */}
        <div className="p-4 border-b border-border">
          <Link to="/" className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-primary flex items-center justify-center">
              <Zap className="w-6 h-6 text-bg-base" />
            </div>
            <div>
              <h1 className="font-bold text-lg text-text-primary">FunGen</h1>
              <p className="text-xs text-text-muted">AI Funscript Generator</p>
            </div>
          </Link>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-2">
          <ul className="space-y-1">
            {navItems.map(({ path, icon: Icon, label }) => (
              <li key={path}>
                <Link
                  to={path}
                  className={clsx(
                    'flex items-center gap-3 px-3 py-2 rounded-lg transition-colors',
                    location.pathname === path
                      ? 'bg-bg-elevated text-text-primary'
                      : 'text-text-secondary hover:bg-bg-elevated hover:text-text-primary'
                  )}
                >
                  <Icon className="w-5 h-5" />
                  <span className="font-medium">{label}</span>
                </Link>
              </li>
            ))}
          </ul>
        </nav>

        {/* Bottom actions */}
        <div className="p-2 border-t border-border">
          <button className="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-text-secondary hover:bg-bg-elevated hover:text-text-primary transition-colors">
            <Settings className="w-5 h-5" />
            <span className="font-medium">Settings</span>
          </button>
          <button className="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-text-secondary hover:bg-bg-elevated hover:text-text-primary transition-colors">
            <HelpCircle className="w-5 h-5" />
            <span className="font-medium">Help</span>
          </button>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 flex flex-col overflow-hidden">
        <Outlet />
      </main>
    </div>
  )
}
