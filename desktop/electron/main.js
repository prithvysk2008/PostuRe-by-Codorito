// Electron main process.
//
// Owns the whole "no terminal, no manual steps" promise: on launch it spawns
// the Python engine as an invisible child process, waits for it to come up,
// then opens a native window pointed at the React frontend. The user never
// sees any of this — no console, no "streamlit run", no browser.
const { app, BrowserWindow, ipcMain, screen } = require('electron')
const { spawn } = require('child_process')
const http = require('http')
const path = require('path')
const fs = require('fs')

const REPO_ROOT = path.resolve(__dirname, '..', '..')
const VENV_PYTHON = path.join(REPO_ROOT, '.venv', 'bin', 'python3')
const BACKEND_URL = 'http://127.0.0.1:8765'
const DEV_FRONTEND_URL = 'http://localhost:5173'

let backendProcess = null
let mainWindow = null
let overlayWindows = []

function bundledBackendPath() {
  // Set by electron-builder's `extraResources` (see ../package.json) once
  // Step 6's PyInstaller build has produced backend/dist/posture-backend.
  // --onedir mode (see build_pyinstaller.sh) means this is a *directory*
  // containing the actual executable of the same name, not the exe itself.
  return path.join(process.resourcesPath, 'backend', 'posture-backend', 'posture-backend')
}

// mediapipe's drawing_utils imports matplotlib at module load time even
// though this app never calls it (see build_entry.py for the full story) —
// MPLBACKEND=Agg skips matplotlib's interactive-backend auto-detection,
// which otherwise probes several GUI toolkits before settling on one.
const BACKEND_ENV = { ...process.env, MPLBACKEND: 'Agg' }

function startBackend() {
  if (app.isPackaged) {
    const exe = bundledBackendPath()
    backendProcess = spawn(exe, [], { stdio: 'ignore', env: BACKEND_ENV })
  } else {
    // Dev mode: reuse the project's own virtualenv rather than requiring a
    // separate one for Electron development.
    if (!fs.existsSync(VENV_PYTHON)) {
      console.error(
        `Expected a Python virtualenv at ${VENV_PYTHON}. ` +
        'Run: python3 -m venv .venv && .venv/bin/pip install -r desktop/backend/requirements.txt'
      )
    }
    backendProcess = spawn(VENV_PYTHON, ['-m', 'desktop.backend.server'], {
      cwd: REPO_ROOT,
      stdio: 'inherit',
      env: BACKEND_ENV,
    })
  }

  backendProcess.on('error', (err) => {
    console.error('Failed to start the PostuRe engine:', err)
  })
  backendProcess.on('exit', (code, signal) => {
    if (code !== 0 && code !== null) {
      console.error(`PostuRe engine exited unexpectedly (code ${code}, signal ${signal})`)
    }
    backendProcess = null
  })
}

function stopBackend() {
  if (backendProcess) {
    backendProcess.kill()
    backendProcess = null
  }
}

// 60s, not 30s: MediaPipe's own transitive matplotlib import does a one-time
// font-cache build on a genuinely fresh machine (first-ever launch only —
// it's cached under ~/.matplotlib after that), which can take 30-40s.
function waitForBackend(timeoutMs = 60000, intervalMs = 400) {
  const deadline = Date.now() + timeoutMs
  return new Promise((resolve, reject) => {
    const attempt = () => {
      const req = http.get(`${BACKEND_URL}/api/ping`, (res) => {
        res.resume()
        if (res.statusCode === 200) resolve()
        else scheduleRetry()
      })
      req.on('error', scheduleRetry)
    }
    const scheduleRetry = () => {
      if (Date.now() > deadline) {
        reject(new Error('PostuRe engine did not start within the expected time.'))
        return
      }
      setTimeout(attempt, intervalMs)
    }
    attempt()
  })
}

async function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 920,
    minWidth: 900,
    minHeight: 640,
    backgroundColor: '#0b1e33',
    title: 'PostuRe:',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      // Keeps the WebSocket-driven UI (and the ambient-status IPC it feeds
      // to the overlay windows) updating at full rate even when this
      // window is unfocused or occluded, not just when it's minimized.
      backgroundThrottling: false,
    },
  })

  mainWindow.on('closed', () => {
    mainWindow = null
    destroyOverlayWindows()
  })

  try {
    await waitForBackend()
  } catch (err) {
    console.error(err)
    // Graceful degradation over a silent hang: load the frontend anyway —
    // useBackend.js's WebSocket retry loop will keep trying and the UI
    // shows a "connecting" banner instead of a blank window.
  }

  if (app.isPackaged) {
    mainWindow.loadFile(path.join(__dirname, '..', 'frontend', 'dist', 'index.html'))
  } else {
    mainWindow.loadURL(DEV_FRONTEND_URL)
  }
}

// ----------------------------------------------------------------------------
// AMBIENT GLOW OVERLAY — a transparent, click-through, always-on-top window
// per display, so the amber/red posture glow stays visible while the user is
// tabbed away to a different app (or a different full-screen Space), not just
// while PostuRe itself has focus. Driven by 'ambient-status' IPC messages the
// main window's renderer forwards from its live WebSocket tick (see
// App.jsx) — these windows never talk to the backend directly.
// ----------------------------------------------------------------------------
function destroyOverlayWindows() {
  overlayWindows.forEach((w) => { if (!w.isDestroyed()) w.destroy() })
  overlayWindows = []
}

function createOverlayWindows() {
  destroyOverlayWindows()
  overlayWindows = screen.getAllDisplays().map((display) => {
    const win = new BrowserWindow({
      x: display.bounds.x,
      y: display.bounds.y,
      width: display.bounds.width,
      height: display.bounds.height,
      transparent: true,
      frame: false,
      hasShadow: false,
      roundedCorners: false,
      resizable: false,
      movable: false,
      minimizable: false,
      maximizable: false,
      fullscreenable: false,
      skipTaskbar: true,
      focusable: false,
      show: true,
      alwaysOnTop: true,
      webPreferences: {
        preload: path.join(__dirname, 'overlay-preload.js'),
        contextIsolation: true,
        nodeIntegration: false,
        backgroundThrottling: false,
      },
    })
    win.setIgnoreMouseEvents(true, { forward: true })
    // 'screen-saver' is the highest standard Electron level on macOS — needed
    // so the glow still shows over another app that's full-screen in its own
    // Space, not just over normal windows.
    win.setAlwaysOnTop(true, 'screen-saver')
    win.setVisibleOnAllWorkspaces(true, { visibleOnFullScreenScreen: true })
    win.loadFile(path.join(__dirname, 'overlay.html'))

    // Dev-only self-test: pulse red for 3s as soon as the overlay loads, with
    // no session/backend/renderer involved. If you see this flash on launch
    // but the real glow still doesn't show up during a session, the bug is
    // in the App.jsx -> preload -> ipcMain relay, not in the overlay window
    // itself (transparency/always-on-top/click-through). If you *don't* see
    // this flash, the bug is in the overlay window/macOS window-level setup.
    if (!app.isPackaged) {
      win.webContents.once('did-finish-load', () => {
        win.webContents.send('ambient-status', { status: 'BAD', enabled: true })
        setTimeout(() => {
          if (!win.isDestroyed()) win.webContents.send('ambient-status', { status: 'IDLE', enabled: true })
        }, 3000)
      })
    }

    return win
  })
}

ipcMain.on('ambient-status', (_event, payload) => {
  overlayWindows.forEach((w) => { if (!w.isDestroyed()) w.webContents.send('ambient-status', payload) })
})

app.whenReady().then(() => {
  startBackend()
  createWindow()
  createOverlayWindows()

  // A demo laptop plugged into a projector mid-session is exactly the kind
  // of display change this should survive without a restart.
  screen.on('display-added', createOverlayWindows)
  screen.on('display-removed', createOverlayWindows)

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow()
      createOverlayWindows()
    }
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})

app.on('before-quit', () => {
  stopBackend()
  destroyOverlayWindows()
})
app.on('will-quit', stopBackend)
