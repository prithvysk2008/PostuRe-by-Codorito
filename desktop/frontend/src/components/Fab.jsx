function PlayIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z" /></svg>
  )
}

function StopIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2" /></svg>
  )
}

// The single floating action button that replaced the sidebar's Start/Stop
// buttons — same control, now a native-feeling pill anchored to the window
// instead of a fixed sidebar row.
export default function Fab({ running, onStart, onStop }) {
  return (
    <button className={`fab${running ? ' running' : ''}`} onClick={running ? onStop : onStart}>
      {running ? <StopIcon /> : <PlayIcon />}
      <span>{running ? 'Stop session' : 'Start session'}</span>
    </button>
  )
}
