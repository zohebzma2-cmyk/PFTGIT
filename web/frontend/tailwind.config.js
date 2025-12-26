/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      // Spotify-inspired color palette
      colors: {
        // Primary accent - Spotify Green
        primary: {
          DEFAULT: '#1db954',
          hover: '#1ed760',
          dark: '#17a043',
          dim: 'rgba(29, 185, 84, 0.2)',
        },
        // Backgrounds (darkest to lightest)
        bg: {
          base: '#121212',
          surface: '#181818',
          elevated: '#282828',
          highlight: '#3e3e3e',
          overlay: '#0a0a0a',
        },
        // Text colors
        text: {
          primary: '#ffffff',
          secondary: '#b3b3b3',
          muted: '#727272',
          disabled: '#535353',
        },
        // Status colors
        success: '#1db954',
        warning: '#f5a623',
        error: '#e84c3c',
        info: '#3498db',
        // Border colors
        border: {
          DEFAULT: '#282828',
          light: '#3e3e3e',
        },
      },
      // Custom spacing
      spacing: {
        'xs': '4px',
        'sm': '8px',
        'md': '12px',
        'lg': '16px',
        'xl': '24px',
      },
      // Border radius
      borderRadius: {
        'sm': '4px',
        'md': '6px',
        'lg': '8px',
        'full': '9999px',
      },
      // Font family
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['JetBrains Mono', 'Menlo', 'Monaco', 'monospace'],
      },
      // Animations
      animation: {
        'spin-slow': 'spin 2s linear infinite',
        'pulse-slow': 'pulse 3s ease-in-out infinite',
        'fade-in': 'fadeIn 0.2s ease-out',
        'slide-up': 'slideUp 0.2s ease-out',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
    },
  },
  plugins: [],
}
