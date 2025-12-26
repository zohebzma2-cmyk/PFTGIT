/**
 * Authentication Provider Component
 * Handles initial auth state and token refresh
 */

import { useEffect } from 'react'
import { useAuthStore } from '../store/authStore'
import { authApi } from '../api/auth'

interface AuthProviderProps {
  children: React.ReactNode
}

export function AuthProvider({ children }: AuthProviderProps) {
  const { token, setAuth, logout, setLoading } = useAuthStore()

  useEffect(() => {
    async function validateToken() {
      if (!token) {
        setLoading(false)
        return
      }

      try {
        // Validate token by fetching current user
        const user = await authApi.me()
        setAuth(user, token)
      } catch (error) {
        // Token is invalid, logout
        console.error('Token validation failed:', error)
        logout()
      }
    }

    validateToken()
  }, [])

  return <>{children}</>
}
