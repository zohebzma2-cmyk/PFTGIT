// Shared types - compatible with web app

export interface User {
  id: string;
  email: string;
  username: string;
  createdAt: string;
}

export interface AuthResponse {
  user: User;
  token: string;
}

export interface Project {
  id: string;
  name: string;
  description?: string;
  videoId?: string;
  funscriptId?: string;
  userId: string;
  createdAt: string;
  updatedAt: string;
}

export interface VideoMetadata {
  id: string;
  filename: string;
  originalName: string;
  mimeType: string;
  size: number;
  duration: number;
  width: number;
  height: number;
  fps: number;
  uploadedAt: string;
}

export interface FunscriptPoint {
  at: number;  // timestamp in milliseconds
  pos: number; // position 0-100
}

export interface Funscript {
  id: string;
  name: string;
  version: string;
  actions: FunscriptPoint[];
  metadata?: {
    duration?: number;
    averageSpeed?: number;
    pointCount?: number;
  };
  createdAt: string;
  updatedAt: string;
}

export interface Chapter {
  time: number;
  name: string;
}

export interface ProcessingJob {
  id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  videoId: string;
  funscriptId?: string;
  settings: ProcessingSettings;
  error?: string;
  createdAt: string;
  updatedAt: string;
}

export interface ProcessingSettings {
  confidenceThreshold: number;
  smoothingFactor: number;
  minPointDistance: number;
  autoSmoothPeaks: boolean;
  invertOutput: boolean;
  limitRange: boolean;
  rangeMin?: number;
  rangeMax?: number;
}

export interface Device {
  id: string;
  type: 'handy' | 'buttplug';
  name: string;
  connectionKey?: string;
  status: 'disconnected' | 'connecting' | 'connected' | 'error';
}

export interface WebSocketMessage {
  type: 'job_progress' | 'job_completed' | 'job_failed' | 'device_status';
  payload: any;
}
