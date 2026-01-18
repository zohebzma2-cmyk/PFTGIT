/**
 * API Client for FunGen Backend
 */

import { useAuthStore } from '../store/authStore'

// In development, Vite proxies /api to the backend
// In production, use VITE_API_URL environment variable
const API_BASE = import.meta.env.VITE_API_URL
  ? `${import.meta.env.VITE_API_URL}/api`
  : '/api'

class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message)
    this.name = 'ApiError'
  }
}

function getAuthHeaders(): Record<string, string> {
  const token = useAuthStore.getState().token
  if (token) {
    return { 'Authorization': `Bearer ${token}` }
  }
  return {}
}

async function request<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE}${endpoint}`

  const response = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
      ...getAuthHeaders(),
      ...options.headers,
    },
    ...options,
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
    throw new ApiError(response.status, error.detail || 'Request failed')
  }

  return response.json()
}

// Projects API
export const projectsApi = {
  list: () => request<Project[]>('/projects'),
  get: (id: string) => request<Project>(`/projects/${id}`),
  create: (data: ProjectCreate) =>
    request<Project>('/projects', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  update: (id: string, data: ProjectUpdate) =>
    request<Project>(`/projects/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    }),
  delete: (id: string) =>
    request<void>(`/projects/${id}`, { method: 'DELETE' }),
}

// Videos API
export const videosApi = {
  upload: async (file: File) => {
    const formData = new FormData()
    formData.append('file', file)

    const response = await fetch(`${API_BASE}/videos/upload`, {
      method: 'POST',
      headers: {
        ...getAuthHeaders(),
        // Note: Don't set Content-Type for FormData - browser sets it with boundary
      },
      body: formData,
    })

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Upload failed' }))
      throw new ApiError(response.status, error.detail)
    }

    return response.json() as Promise<VideoMetadata>
  },
  get: (id: string) => request<VideoMetadata>(`/videos/${id}`),
  delete: (id: string) => request<void>(`/videos/${id}`, { method: 'DELETE' }),
  getStreamUrl: (id: string) => `${API_BASE}/videos/${id}/stream`,
  getThumbnailUrl: (id: string, timeMs = 0) =>
    `${API_BASE}/videos/${id}/thumbnail?time_ms=${timeMs}`,
}

// Funscripts API
export const funscriptsApi = {
  list: () => request<Funscript[]>('/funscripts'),
  get: (id: string) => request<Funscript>(`/funscripts/${id}`),
  create: (data: FunscriptCreate) =>
    request<Funscript>('/funscripts', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  update: (id: string, data: FunscriptUpdate) =>
    request<Funscript>(`/funscripts/${id}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    }),
  delete: (id: string) =>
    request<void>(`/funscripts/${id}`, { method: 'DELETE' }),
  addPoint: (id: string, point: FunscriptPoint) =>
    request<void>(`/funscripts/${id}/points`, {
      method: 'POST',
      body: JSON.stringify(point),
    }),
  deletePoint: (id: string, timeMs: number) =>
    request<void>(`/funscripts/${id}/points/${timeMs}`, { method: 'DELETE' }),
  getExportUrl: (id: string) => `${API_BASE}/funscripts/${id}/export`,
}

// Processing API
export const processingApi = {
  createJob: (data: ProcessingJobCreate) =>
    request<ProcessingJob>('/processing/jobs', {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  listJobs: () => request<ProcessingJob[]>('/processing/jobs'),
  getJob: (id: string) => request<ProcessingJob>(`/processing/jobs/${id}`),
  cancelJob: (id: string) =>
    request<void>(`/processing/jobs/${id}/cancel`, { method: 'POST' }),
}

// Devices API
export const devicesApi = {
  list: () => request<Device[]>('/devices'),
  connectHandy: (connectionKey: string) =>
    request<Device>('/devices/handy/connect', {
      method: 'POST',
      body: JSON.stringify({ connection_key: connectionKey }),
    }),
  disconnect: (id: string) =>
    request<void>(`/devices/${id}/disconnect`, { method: 'POST' }),
  sync: (id: string, data: DeviceSyncRequest) =>
    request<void>(`/devices/${id}/sync`, {
      method: 'POST',
      body: JSON.stringify(data),
    }),
  pause: (id: string) =>
    request<void>(`/devices/${id}/pause`, { method: 'POST' }),
  resume: (id: string, positionMs = 0) =>
    request<void>(`/devices/${id}/resume?position_ms=${positionMs}`, {
      method: 'POST',
    }),
}

// Types
export interface Project {
  id: string
  name: string
  description: string | null
  is_public: boolean
  video_count: number
  funscript_count: number
  created_at: string
  updated_at: string
}

export interface ProjectCreate {
  name: string
  description?: string
}

export interface ProjectUpdate {
  name?: string
  description?: string
}

export interface VideoMetadata {
  id: string
  filename: string
  original_filename: string
  duration_ms: number | null
  width: number | null
  height: number | null
  fps: number | null
  frame_count: number | null
  file_size: number
  project_id: string
  created_at: string
}

export interface FunscriptPoint {
  at: number
  pos: number
}

export interface Funscript {
  id: string
  name: string
  video_id: string | null
  actions: FunscriptPoint[]
  metadata: Record<string, unknown>
  inverted: boolean
  range: number
}

export interface FunscriptCreate {
  name: string
  video_id?: string
  actions?: FunscriptPoint[]
}

export interface FunscriptUpdate {
  name?: string
  actions?: FunscriptPoint[]
  inverted?: boolean
  range?: number
}

export interface ProcessingJob {
  id: string
  video_id: string
  funscript_id: string | null
  stage: string
  progress: number
  message: string
  created_at: string
  started_at: string | null
  completed_at: string | null
  error: string | null
  actions: FunscriptPoint[] | null
}

export interface ProcessingJobCreate {
  video_id: string
  settings?: Record<string, unknown>
}

export interface Device {
  id: string
  type: 'handy' | 'bluetooth' | 'buttplug'
  name: string
  status: 'disconnected' | 'connecting' | 'connected' | 'syncing' | 'error'
  connection_key: string | null
  firmware_version: string | null
  last_error: string | null
  // Bluetooth-specific properties
  bluetooth_id?: string
  battery_level?: number
  features?: string[]
}

export interface DeviceSyncRequest {
  funscript_id: string
  video_position_ms?: number
  server_time_offset_ms?: number
}
