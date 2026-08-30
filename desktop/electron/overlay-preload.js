// Preload for the ambient-glow overlay windows. These windows never load
// any app code beyond overlay.html, so this only needs one channel: the
// live status the main window's renderer forwards over IPC (see preload.js).
const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('overlayBridge', {
  onAmbientStatus: (cb) => ipcRenderer.on('ambient-status', (_event, payload) => cb(payload)),
})
