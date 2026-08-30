// The renderer only needs fetch()/WebSocket to the local backend (127.0.0.1),
// both of which work fine under contextIsolation without any bridged API.
// This is a placeholder for the day something genuinely needs main-process
// access (e.g. a native save dialog instead of the <a download> approach
// currently used for the Spine Card).
const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('postureDesktop', {
  platform: process.platform,
  // Forwards the live posture status to the main process, which relays it
  // to the always-on-top ambient-glow overlay windows (see overlay.html) —
  // that's what keeps the glow visible even when this window isn't focused.
  sendAmbientStatus: (payload) => ipcRenderer.send('ambient-status', payload),
})
