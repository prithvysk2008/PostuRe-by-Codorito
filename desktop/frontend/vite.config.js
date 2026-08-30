import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// base: './' is required for Electron — the built app is loaded via the
// file:// protocol, and Vite's default absolute asset paths ("/assets/...")
// don't resolve from a file:// root the way they do on an http:// dev server.
export default defineConfig({
  plugins: [react()],
  base: './',
  server: { port: 5173, strictPort: true },
})
