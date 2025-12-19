/**
 * Authentication API client
 */

import { useAuthStore } from '../store/authStore'

const API_BASE = '/api/auth'

class AuthApiError extends Error {
  constructor(public status: number, message: string) {
    super(message)
    this.name = 'AuthApiError'
  }
}

async function authRequest<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const token = useAuthStore.getState().token

  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...options.headers as Record<string, string>,
  }

  if (token) {
    headers['Authorization'] = `Bearer ${token}`
  }

  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers,
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
    throw new AuthApiError(response.status, error.detail || 'Request failed')
  }

  return response.json()
}

// Types
export interface RegisterRequest {
  email: string
  username: string
  password: string
  display_name?: string
}

export interface LoginRequest {
  email: string
  password: string
}

export interface User {
  id: string
  email: string
  username: string
  display_name: string | null
  avatar_url: string | null
  is_active: boolean
  is_verified: boolean
  created_at: string
}

export interface Token {
  access_token: string
  token_type: string
  expires_in: number
}

export interface AuthResponse {
  token: Token
  user: User
}

// API functions
export const authApi = {
  register: async (data: RegisterRequest): Promise<AuthResponse> => {
    return authRequest<AuthResponse>('/register', {
      method: 'POST',
      body: JSON.stringify(data),
    })
  },

  login: async (data: LoginRequest): Promise<AuthResponse> => {
    return authRequest<AuthResponse>('/login', {
      method: 'POST',
      body: JSON.stringify(data),
    })
  },

  me: async (): Promise<User> => {
    return authRequest<User>('/me')
  },

  updateProfile: async (data: { display_name?: string; avatar_url?: string }): Promise<User> => {
    const params = new URLSearchParams()
    if (data.display_name) params.append('display_name', data.display_name)
    if (data.avatar_url) params.append('avatar_url', data.avatar_url)

    return authRequest<User>(`/me?${params.toString()}`, {
      method: 'PUT',
    })
  },

  changePassword: async (currentPassword: string, newPassword: string): Promise<void> => {
    await authRequest('/change-password', {
      method: 'POST',
      body: JSON.stringify({
        current_password: currentPassword,
        new_password: newPassword,
      }),
    })
  },

  refreshToken: async (): Promise<Token> => {
    return authRequest<Token>('/refresh', {
      method: 'POST',
    })
  },
}
