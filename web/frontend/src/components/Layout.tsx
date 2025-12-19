import { Outlet, Link, useLocation, useNavigate } from 'react-router-dom'
import {
  FolderOpen,
  Sliders,
  Settings,
  HelpCircle,
  Zap,
  LogOut,
  User
} from 'lucide-react'
import { clsx } from 'clsx'
import { useAuthStore } from '../store/authStore'
import { useState } from 'react'

const navItems = [
  { path: '/projects', icon: FolderOpen, label: 'Projects' },
  { path: '/editor', icon: Sliders, label: 'Editor' },
]

export default function Layout() {
  const location = useLocation()
  const navigate = useNavigate()
  const { user, logout } = useAuthStore()
  const [showUserMenu, setShowUserMenu] = useState(false)

  const handleLogout = () => {
    logout()
    navigate('/')
  }

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
                    location.pathname === path || location.pathname.startsWith(path + '/')
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
        <div className="p-2 border-t border-border space-y-1">
          <button className="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-text-secondary hover:bg-bg-elevated hover:text-text-primary transition-colors">
            <Settings className="w-5 h-5" />
            <span className="font-medium">Settings</span>
          </button>
          <button className="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-text-secondary hover:bg-bg-elevated hover:text-text-primary transition-colors">
            <HelpCircle className="w-5 h-5" />
            <span className="font-medium">Help</span>
          </button>
        </div>

        {/* User menu */}
        <div className="p-2 border-t border-border relative">
          <button
            onClick={() => setShowUserMenu(!showUserMenu)}
            className="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-text-secondary hover:bg-bg-elevated hover:text-text-primary transition-colors"
          >
            <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center">
              <User className="w-4 h-4 text-primary" />
            </div>
            <div className="flex-1 text-left">
              <p className="text-sm font-medium text-text-primary truncate">
                {user?.display_name || user?.username}
              </p>
              <p className="text-xs text-text-muted truncate">{user?.email}</p>
            </div>
          </button>

          {/* Dropdown menu */}
          {showUserMenu && (
            <div className="absolute bottom-full left-2 right-2 mb-1 bg-bg-elevated border border-border rounded-lg shadow-lg overflow-hidden">
              <button
                onClick={handleLogout}
                className="w-full flex items-center gap-2 px-3 py-2 text-left text-text-secondary hover:bg-bg-highlight hover:text-text-primary transition-colors"
              >
                <LogOut className="w-4 h-4" />
                <span className="text-sm">Sign out</span>
              </button>
            </div>
          )}
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 flex flex-col overflow-hidden">
        <Outlet />
      </main>
    </div>
  )
}
