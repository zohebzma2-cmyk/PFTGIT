import { create } from 'zustand';
import type { FunscriptPoint, VideoMetadata, Chapter, ProcessingSettings } from '../api/types';

interface EditorState {
  // Video state
  video: VideoMetadata | null;
  videoUri: string | null;
  currentTime: number;
  duration: number;
  isPlaying: boolean;

  // Funscript state
  points: FunscriptPoint[];
  selectedPoints: number[];
  chapters: Chapter[];

  // Editor state
  zoom: number;
  scrollOffset: number;
  tool: 'select' | 'draw' | 'erase' | 'move';
  showTimeline2: boolean;

  // Processing state
  isProcessing: boolean;
  processingProgress: number;

  // Settings
  settings: ProcessingSettings;

  // History for undo/redo
  history: FunscriptPoint[][];
  historyIndex: number;

  // Actions
  setVideo: (video: VideoMetadata | null, uri: string | null) => void;
  setCurrentTime: (time: number) => void;
  setDuration: (duration: number) => void;
  setIsPlaying: (isPlaying: boolean) => void;

  setPoints: (points: FunscriptPoint[]) => void;
  addPoint: (point: FunscriptPoint) => void;
  updatePoint: (index: number, point: FunscriptPoint) => void;
  deletePoints: (indices: number[]) => void;
  clearPoints: () => void;

  selectPoint: (index: number, multi?: boolean) => void;
  selectAllPoints: () => void;
  deselectAllPoints: () => void;

  addChapter: (chapter: Chapter) => void;
  deleteChapter: (index: number) => void;
  clearChapters: () => void;
  setChapters: (chapters: Chapter[]) => void;

  setZoom: (zoom: number) => void;
  setScrollOffset: (offset: number) => void;
  setTool: (tool: 'select' | 'draw' | 'erase' | 'move') => void;
  setShowTimeline2: (show: boolean) => void;

  setProcessing: (isProcessing: boolean, progress?: number) => void;
  setSettings: (settings: Partial<ProcessingSettings>) => void;

  undo: () => void;
  redo: () => void;

  reset: () => void;
}

const defaultSettings: ProcessingSettings = {
  confidenceThreshold: 0.5,
  smoothingFactor: 0.3,
  minPointDistance: 100,
  autoSmoothPeaks: true,
  invertOutput: false,
  limitRange: true,
  rangeMin: 0,
  rangeMax: 100,
};

export const useEditorStore = create<EditorState>((set, get) => ({
  // Initial state
  video: null,
  videoUri: null,
  currentTime: 0,
  duration: 0,
  isPlaying: false,

  points: [],
  selectedPoints: [],
  chapters: [],

  zoom: 1,
  scrollOffset: 0,
  tool: 'select',
  showTimeline2: false,

  isProcessing: false,
  processingProgress: 0,

  settings: defaultSettings,

  history: [[]],
  historyIndex: 0,

  // Video actions
  setVideo: (video, uri) => set({ video, videoUri: uri }),
  setCurrentTime: (currentTime) => set({ currentTime }),
  setDuration: (duration) => set({ duration }),
  setIsPlaying: (isPlaying) => set({ isPlaying }),

  // Points actions with history
  setPoints: (points) => {
    const { history, historyIndex } = get();
    const newHistory = history.slice(0, historyIndex + 1);
    newHistory.push(points);
    set({
      points,
      history: newHistory,
      historyIndex: newHistory.length - 1,
    });
  },

  addPoint: (point) => {
    const { points } = get();
    const newPoints = [...points, point].sort((a, b) => a.at - b.at);
    get().setPoints(newPoints);
  },

  updatePoint: (index, point) => {
    const { points } = get();
    const newPoints = [...points];
    newPoints[index] = point;
    get().setPoints(newPoints.sort((a, b) => a.at - b.at));
  },

  deletePoints: (indices) => {
    const { points } = get();
    const newPoints = points.filter((_, i) => !indices.includes(i));
    get().setPoints(newPoints);
    set({ selectedPoints: [] });
  },

  clearPoints: () => {
    get().setPoints([]);
    set({ selectedPoints: [] });
  },

  // Selection actions
  selectPoint: (index, multi = false) => {
    const { selectedPoints } = get();
    if (multi) {
      if (selectedPoints.includes(index)) {
        set({ selectedPoints: selectedPoints.filter((i) => i !== index) });
      } else {
        set({ selectedPoints: [...selectedPoints, index] });
      }
    } else {
      set({ selectedPoints: [index] });
    }
  },

  selectAllPoints: () => {
    const { points } = get();
    set({ selectedPoints: points.map((_, i) => i) });
  },

  deselectAllPoints: () => set({ selectedPoints: [] }),

  // Chapter actions
  addChapter: (chapter) => {
    const { chapters } = get();
    const newChapters = [...chapters, chapter].sort((a, b) => a.time - b.time);
    set({ chapters: newChapters });
  },

  deleteChapter: (index) => {
    const { chapters } = get();
    set({ chapters: chapters.filter((_, i) => i !== index) });
  },

  clearChapters: () => set({ chapters: [] }),
  setChapters: (chapters) => set({ chapters }),

  // Editor state actions
  setZoom: (zoom) => set({ zoom: Math.max(0.1, Math.min(10, zoom)) }),
  setScrollOffset: (scrollOffset) => set({ scrollOffset }),
  setTool: (tool) => set({ tool }),
  setShowTimeline2: (showTimeline2) => set({ showTimeline2 }),

  // Processing actions
  setProcessing: (isProcessing, progress = 0) => set({ isProcessing, processingProgress: progress }),
  setSettings: (newSettings) => set((state) => ({
    settings: { ...state.settings, ...newSettings },
  })),

  // History actions
  undo: () => {
    const { historyIndex, history } = get();
    if (historyIndex > 0) {
      const newIndex = historyIndex - 1;
      set({
        points: history[newIndex],
        historyIndex: newIndex,
        selectedPoints: [],
      });
    }
  },

  redo: () => {
    const { historyIndex, history } = get();
    if (historyIndex < history.length - 1) {
      const newIndex = historyIndex + 1;
      set({
        points: history[newIndex],
        historyIndex: newIndex,
        selectedPoints: [],
      });
    }
  },

  // Reset all state
  reset: () => set({
    video: null,
    videoUri: null,
    currentTime: 0,
    duration: 0,
    isPlaying: false,
    points: [],
    selectedPoints: [],
    chapters: [],
    zoom: 1,
    scrollOffset: 0,
    tool: 'select',
    showTimeline2: false,
    isProcessing: false,
    processingProgress: 0,
    history: [[]],
    historyIndex: 0,
  }),
}));

export default useEditorStore;
