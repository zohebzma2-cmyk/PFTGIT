/**
 * Editor mode store using Zustand
 * Manages Simple vs Expert mode for the editor interface
 */

import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export type EditorMode = 'simple' | 'expert'

interface ModeState {
  // State
  mode: EditorMode

  // Actions
  setMode: (mode: EditorMode) => void
  toggleMode: () => void
}

export const useModeStore = create<ModeState>()(
  persist(
    (set) => ({
      // Initial state - default to simple mode for new users
      mode: 'simple',

      // Actions
      setMode: (mode) => set({ mode }),

      toggleMode: () =>
        set((state) => ({
          mode: state.mode === 'simple' ? 'expert' : 'simple',
        })),
    }),
    {
      name: 'fungen-mode',
    }
  )
)

// Selector hooks for convenience
export const useEditorMode = () => useModeStore((state) => state.mode)
export const useIsExpertMode = () => useModeStore((state) => state.mode === 'expert')
export const useIsSimpleMode = () => useModeStore((state) => state.mode === 'simple')
