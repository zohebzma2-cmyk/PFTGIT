import * as SecureStore from 'expo-secure-store';
import { API_BASE_URL, API_ENDPOINTS } from '../constants/api';
import type {
  AuthResponse,
  User,
  Project,
  VideoMetadata,
  Funscript,
  ProcessingJob,
  ProcessingSettings
} from './types';

const TOKEN_KEY = 'auth_token';

// Get stored auth token
async function getToken(): Promise<string | null> {
  return await SecureStore.getItemAsync(TOKEN_KEY);
}

// Set auth token
export async function setToken(token: string): Promise<void> {
  await SecureStore.setItemAsync(TOKEN_KEY, token);
}

// Remove auth token
export async function removeToken(): Promise<void> {
  await SecureStore.deleteItemAsync(TOKEN_KEY);
}

// Get authorization headers
async function getAuthHeaders(): Promise<Record<string, string>> {
  const token = await getToken();
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  };
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  return headers;
}

// Base fetch wrapper with auth
async function fetchWithAuth<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;
  const headers = await getAuthHeaders();

  const response = await fetch(url, {
    ...options,
    headers: {
      ...headers,
      ...options.headers,
    },
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ message: 'Request failed' }));
    throw new Error(error.message || error.detail || 'Request failed');
  }

  return response.json();
}

// Auth API
export const authApi = {
  login: async (email: string, password: string): Promise<AuthResponse> => {
    const response = await fetchWithAuth<AuthResponse>(API_ENDPOINTS.login, {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    });
    await setToken(response.token);
    return response;
  },

  register: async (email: string, username: string, password: string): Promise<AuthResponse> => {
    const response = await fetchWithAuth<AuthResponse>(API_ENDPOINTS.register, {
      method: 'POST',
      body: JSON.stringify({ email, username, password }),
    });
    await setToken(response.token);
    return response;
  },

  getMe: async (): Promise<User> => {
    return fetchWithAuth<User>(API_ENDPOINTS.me);
  },

  logout: async (): Promise<void> => {
    await removeToken();
  },
};

// Projects API
export const projectsApi = {
  list: async (): Promise<Project[]> => {
    return fetchWithAuth<Project[]>(API_ENDPOINTS.projects);
  },

  get: async (id: string): Promise<Project> => {
    return fetchWithAuth<Project>(API_ENDPOINTS.project(id));
  },

  create: async (data: { name: string; description?: string }): Promise<Project> => {
    return fetchWithAuth<Project>(API_ENDPOINTS.projects, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },

  update: async (id: string, data: Partial<Project>): Promise<Project> => {
    return fetchWithAuth<Project>(API_ENDPOINTS.project(id), {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  },

  delete: async (id: string): Promise<void> => {
    await fetchWithAuth(API_ENDPOINTS.project(id), {
      method: 'DELETE',
    });
  },
};

// Videos API
export const videosApi = {
  upload: async (
    uri: string,
    filename: string,
    onProgress?: (progress: number) => void
  ): Promise<VideoMetadata> => {
    const token = await getToken();
    const formData = new FormData();

    formData.append('file', {
      uri,
      name: filename,
      type: 'video/mp4',
    } as any);

    const xhr = new XMLHttpRequest();

    return new Promise((resolve, reject) => {
      xhr.upload.addEventListener('progress', (event) => {
        if (event.lengthComputable && onProgress) {
          onProgress(event.loaded / event.total);
        }
      });

      xhr.addEventListener('load', () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(JSON.parse(xhr.responseText));
        } else {
          reject(new Error('Upload failed'));
        }
      });

      xhr.addEventListener('error', () => reject(new Error('Upload failed')));

      xhr.open('POST', `${API_BASE_URL}${API_ENDPOINTS.uploadVideo}`);
      if (token) {
        xhr.setRequestHeader('Authorization', `Bearer ${token}`);
      }
      xhr.send(formData);
    });
  },

  getStreamUrl: (id: string): string => {
    return `${API_BASE_URL}${API_ENDPOINTS.videoStream(id)}`;
  },
};

// Processing API
export const processingApi = {
  startJob: async (videoId: string, settings: ProcessingSettings): Promise<ProcessingJob> => {
    return fetchWithAuth<ProcessingJob>(API_ENDPOINTS.processingJobs, {
      method: 'POST',
      body: JSON.stringify({ videoId, settings }),
    });
  },

  getJob: async (id: string): Promise<ProcessingJob> => {
    return fetchWithAuth<ProcessingJob>(API_ENDPOINTS.processingJob(id));
  },
};

// Funscripts API
export const funscriptsApi = {
  get: async (id: string): Promise<Funscript> => {
    return fetchWithAuth<Funscript>(API_ENDPOINTS.funscript(id));
  },

  update: async (id: string, actions: { at: number; pos: number }[]): Promise<Funscript> => {
    return fetchWithAuth<Funscript>(API_ENDPOINTS.funscript(id), {
      method: 'PUT',
      body: JSON.stringify({ actions }),
    });
  },
};

// Device API
export const devicesApi = {
  connectHandy: async (connectionKey: string): Promise<{ success: boolean }> => {
    return fetchWithAuth<{ success: boolean }>(API_ENDPOINTS.handyConnect, {
      method: 'POST',
      body: JSON.stringify({ connectionKey }),
    });
  },

  syncHandy: async (funscriptId: string): Promise<{ success: boolean }> => {
    return fetchWithAuth<{ success: boolean }>(API_ENDPOINTS.handySync, {
      method: 'POST',
      body: JSON.stringify({ funscriptId }),
    });
  },
};

export default {
  auth: authApi,
  projects: projectsApi,
  videos: videosApi,
  processing: processingApi,
  funscripts: funscriptsApi,
  devices: devicesApi,
};
