import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import AsyncStorage from '@react-native-async-storage/async-storage';

export interface FunscriptPoint {
  at: number;
  pos: number;
}

export interface MediaItem {
  id: string;
  uri: string;
  filename: string;
  mimeType: string;
  duration: number;
  width: number;
  height: number;
  thumbnailUri?: string;
  createdAt: string;
}

export interface Project {
  id: string;
  name: string;
  description?: string;
  status: 'draft' | 'finished';
  mediaId?: string;
  media?: MediaItem;
  points: FunscriptPoint[];
  duration: number;
  createdAt: string;
  updatedAt: string;
  thumbnailUri?: string;
}

interface ProjectState {
  projects: Project[];
  mediaLibrary: MediaItem[];
  currentProjectId: string | null;

  // Project actions
  createProject: (name: string, media?: MediaItem) => Project;
  updateProject: (id: string, updates: Partial<Project>) => void;
  deleteProject: (id: string) => void;
  getProject: (id: string) => Project | undefined;
  setCurrentProject: (id: string | null) => void;
  markAsFinished: (id: string) => void;
  markAsDraft: (id: string) => void;
  duplicateProject: (id: string) => Project;

  // Media actions
  addMedia: (media: MediaItem) => void;
  removeMedia: (id: string) => void;
  getMedia: (id: string) => MediaItem | undefined;

  // Filter helpers
  getDrafts: () => Project[];
  getFinishedProjects: () => Project[];
}

export const useProjectStore = create<ProjectState>()(
  persist(
    (set, get) => ({
      projects: [],
      mediaLibrary: [],
      currentProjectId: null,

      createProject: (name: string, media?: MediaItem) => {
        const now = new Date().toISOString();
        const newProject: Project = {
          id: `project-${Date.now()}`,
          name,
          status: 'draft',
          mediaId: media?.id,
          media,
          points: [],
          duration: media?.duration || 0,
          createdAt: now,
          updatedAt: now,
          thumbnailUri: media?.thumbnailUri,
        };

        set((state) => ({
          projects: [newProject, ...state.projects],
          currentProjectId: newProject.id,
        }));

        return newProject;
      },

      updateProject: (id: string, updates: Partial<Project>) => {
        set((state) => ({
          projects: state.projects.map((p) =>
            p.id === id
              ? { ...p, ...updates, updatedAt: new Date().toISOString() }
              : p
          ),
        }));
      },

      deleteProject: (id: string) => {
        set((state) => ({
          projects: state.projects.filter((p) => p.id !== id),
          currentProjectId:
            state.currentProjectId === id ? null : state.currentProjectId,
        }));
      },

      getProject: (id: string) => {
        return get().projects.find((p) => p.id === id);
      },

      setCurrentProject: (id: string | null) => {
        set({ currentProjectId: id });
      },

      markAsFinished: (id: string) => {
        set((state) => ({
          projects: state.projects.map((p) =>
            p.id === id
              ? { ...p, status: 'finished', updatedAt: new Date().toISOString() }
              : p
          ),
        }));
      },

      markAsDraft: (id: string) => {
        set((state) => ({
          projects: state.projects.map((p) =>
            p.id === id
              ? { ...p, status: 'draft', updatedAt: new Date().toISOString() }
              : p
          ),
        }));
      },

      duplicateProject: (id: string) => {
        const original = get().getProject(id);
        if (!original) throw new Error('Project not found');

        const now = new Date().toISOString();
        const duplicate: Project = {
          ...original,
          id: `project-${Date.now()}`,
          name: `${original.name} (Copy)`,
          status: 'draft',
          createdAt: now,
          updatedAt: now,
        };

        set((state) => ({
          projects: [duplicate, ...state.projects],
        }));

        return duplicate;
      },

      addMedia: (media: MediaItem) => {
        set((state) => ({
          mediaLibrary: [media, ...state.mediaLibrary],
        }));
      },

      removeMedia: (id: string) => {
        set((state) => ({
          mediaLibrary: state.mediaLibrary.filter((m) => m.id !== id),
        }));
      },

      getMedia: (id: string) => {
        return get().mediaLibrary.find((m) => m.id === id);
      },

      getDrafts: () => {
        return get().projects.filter((p) => p.status === 'draft');
      },

      getFinishedProjects: () => {
        return get().projects.filter((p) => p.status === 'finished');
      },
    }),
    {
      name: 'fungen-projects-storage',
      storage: createJSONStorage(() => AsyncStorage),
    }
  )
);

// Export funscript helper
export function exportFunscript(project: Project): string {
  const funscript = {
    version: '1.0',
    inverted: false,
    range: 100,
    actions: project.points.map((p) => ({
      at: p.at,
      pos: p.pos,
    })),
    metadata: {
      creator: 'FunGen',
      description: project.description || '',
      duration: project.duration,
      license: '',
      notes: '',
      performers: [],
      script_url: '',
      tags: [],
      title: project.name,
      type: 'basic',
      video_url: '',
    },
  };

  return JSON.stringify(funscript, null, 2);
}

// Import funscript helper
export function importFunscript(json: string): { points: FunscriptPoint[]; metadata?: any } {
  const data = JSON.parse(json);
  const points: FunscriptPoint[] = (data.actions || []).map((a: any) => ({
    at: a.at,
    pos: a.pos,
  }));
  return { points, metadata: data.metadata };
}
