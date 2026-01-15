import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
  ],
  server: {
    host: '0.0.0.0',
    proxy: {
      '/api': {
        target: 'http://[::1]:8000',
        changeOrigin: true,
      },
      '/ws/ws': {
        target: 'http://[::1]:8000',
        ws: true,
        changeOrigin: true,
      },
    },
  },
})
