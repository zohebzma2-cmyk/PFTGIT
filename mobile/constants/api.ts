import Constants from 'expo-constants';

export const API_BASE_URL = Constants.expoConfig?.extra?.apiUrl || 'https://pftgit.onrender.com';

export const API_ENDPOINTS = {
  // Auth
  login: '/api/auth/login',
  register: '/api/auth/register',
  me: '/api/auth/me',

  // Projects
  projects: '/api/projects',
  project: (id: string) => `/api/projects/${id}`,

  // Videos
  videos: '/api/videos',
  uploadVideo: '/api/videos/upload',
  videoStream: (id: string) => `/api/videos/${id}/stream`,

  // Processing
  processingJobs: '/api/processing/jobs',
  processingJob: (id: string) => `/api/processing/jobs/${id}`,

  // Funscripts
  funscripts: '/api/funscripts',
  funscript: (id: string) => `/api/funscripts/${id}`,

  // Devices
  handyConnect: '/api/devices/handy/connect',
  handySync: '/api/devices/handy/sync',

  // WebSocket
  websocket: '/ws',
};

export default API_ENDPOINTS;
