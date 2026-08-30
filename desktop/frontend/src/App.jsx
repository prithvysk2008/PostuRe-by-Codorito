import { useEffect, useState } from 'react'
import LiveView from './components/LiveView.jsx'
import IdleScreen from './components/IdleScreen.jsx'
import SummaryScreen from './components/SummaryScreen.jsx'
import SettingsPopover from './components/SettingsPopover.jsx'
import ThemeToggle from './components/ThemeToggle.jsx'
import Fab from './components/Fab.jsx'
import { useBackend } from './useBackend.js'

const DEFAULT_SETTINGS = {
  sensitivity: 1.0, break_min: 20, ambient: true, audio: true, audio_volume: 70,
  fatigue: true, skeleton: true, show_angle: true, snapshot: true,
  cam_index: 0, complexity: 1, face_every: 2,
}

export default function App() {
  const {
    connected, tick, summary, history, spineCard,
    startSession, stopSession, recalibrate, updateSettings, requestSpineCard, refreshHistory, goHome,
  } = useBackend()

  const [settings, setSettings] = useState(DEFAULT_SETTINGS)
  const [theme, setTheme] = useState(() => {
    try { return localStorage.getItem('posture-theme') || 'dark' } catch { return 'dark' }
  })
  const running = !!tick?.running

  const onChangeSettings = (patch) => {
    setSettings((prev) => {
      const next = { ...prev, ...patch }
      updateSettings(next)
      return next
    })
  }

  const onToggleTheme = () => {
    setTheme((prev) => {
      const next = prev === 'light' ? 'dark' : 'light'
      try { localStorage.setItem('posture-theme', next) } catch { /* private-window storage can throw */ }
      return next
    })
  }

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
  }, [theme])

  // Refresh the idle-screen history once a session's summary lands.
  useEffect(() => {
    if (summary) refreshHistory()
  }, [summary, refreshHistory])

  // Forward the live status to the main process, which relays it to the
  // always-on-top ambient-glow overlay windows — that's what keeps the glow
  // visible while the user is tabbed away, not just while this window has
  // focus. Explicitly sends "IDLE" (no glow) once the session stops, rather
  // than leaving the overlay showing whatever status the last frame had.
  useEffect(() => {
    window.postureDesktop?.sendAmbientStatus?.({
      status: running ? (tick?.status || 'IDLE') : 'IDLE',
      enabled: settings.ambient,
    })
  }, [running, tick?.status, settings.ambient])

  return (
    <div className="app">
      <div className="main">
        <div className="masthead">
          <div className="masthead-left">
            <div className="wordmark" style={{ fontSize: '2rem' }}>Postu<span>Re:</span></div>
            <div className="tag">Craniovertebral tracking &middot; Fatigue detection &middot; 100% on-device</div>
          </div>
          <div className="masthead-actions">
            <ThemeToggle theme={theme} onToggle={onToggleTheme} />
            <SettingsPopover
              settings={settings}
              onChange={onChangeSettings}
              running={running}
              onRecalibrate={recalibrate}
            />
          </div>
        </div>
        <div className="rule" />

        {!connected && (
          <div className="banner">
            <span className="kicker">CONNECTING</span>
            <span>Waiting for the on-device engine to start…</span>
          </div>
        )}

        {running ? (
          <LiveView tick={tick} />
        ) : summary ? (
          <SummaryScreen
            summary={summary}
            requestSpineCard={requestSpineCard}
            spineCard={spineCard}
            onStartAnother={startSession}
            onGoHome={goHome}
          />
        ) : (
          <IdleScreen history={history} />
        )}
      </div>

      {!summary && <Fab running={running} onStart={startSession} onStop={stopSession} />}
    </div>
  )
}
