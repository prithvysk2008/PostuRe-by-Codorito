import { useEffect, useRef, useState } from 'react'

function GearIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.6 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.6a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
    </svg>
  )
}

// Replaces the removed sidebar's Coaching/Feedback/Overlay controls with a
// frosted-glass popover anchored under a settings gear — Performance &
// camera was dropped entirely (not carried over) per an explicit decision,
// and Start/Stop/Recalibrate moved to the floating action button / here.
export default function SettingsPopover({ settings, onChange, running, onRecalibrate }) {
  const [open, setOpen] = useState(false)
  const ref = useRef(null)

  useEffect(() => {
    if (!open) return
    const onClickOutside = (e) => {
      if (ref.current && !ref.current.contains(e.target)) setOpen(false)
    }
    document.addEventListener('mousedown', onClickOutside)
    return () => document.removeEventListener('mousedown', onClickOutside)
  }, [open])

  const set = (key) => (e) => {
    const val = e.target.type === 'checkbox' ? e.target.checked
      : e.target.type === 'range' || e.target.type === 'number' ? Number(e.target.value)
      : e.target.value
    onChange({ [key]: val })
  }

  return (
    <div className="settings-anchor" ref={ref}>
      <button className="icon-btn" aria-label="Settings" title="Settings" onClick={() => setOpen((v) => !v)}>
        <GearIcon />
      </button>

      <div className={`settings-popover${open ? ' open' : ''}`}>
        <div>
          <div className="section-title">Coaching</div>
          <div className="field" style={{ marginTop: 8 }}>
            <div className="field-row"><label>Sensitivity</label><span>{settings.sensitivity.toFixed(1)}</span></div>
            <input type="range" min="0.6" max="1.8" step="0.1" value={settings.sensitivity} onChange={set('sensitivity')} />
          </div>
          <div className="field" style={{ marginTop: 10 }}>
            <div className="field-row"><label>Stretch break every (min)</label><span>{settings.break_min}</span></div>
            <input type="range" min="5" max="90" step="1" value={settings.break_min} onChange={set('break_min')} />
          </div>
          {running && (
            <button className="btn" style={{ marginTop: 10 }} onClick={onRecalibrate}>Recalibrate baseline</button>
          )}
        </div>

        <div>
          <div className="section-title">Feedback</div>
          <label className="checkbox-row" style={{ marginTop: 8 }}>
            <input type="checkbox" checked={settings.ambient} onChange={set('ambient')} /> Ambient screen glow
          </label>
          <label className="checkbox-row" style={{ marginTop: 8 }}>
            <input type="checkbox" checked={settings.audio} onChange={set('audio')} /> Audio nudges
          </label>
          <div className="field" style={{ marginTop: 10 }}>
            <div className="field-row"><label>Nudge volume</label><span>{settings.audio_volume}</span></div>
            <input type="range" min="0" max="100" step="5" value={settings.audio_volume}
                   disabled={!settings.audio} onChange={set('audio_volume')} />
          </div>
          <label className="checkbox-row" style={{ marginTop: 10 }}>
            <input type="checkbox" checked={settings.fatigue} onChange={set('fatigue')} /> Fatigue engine (eyes + yawns)
          </label>
        </div>

        <div>
          <div className="section-title">Overlay</div>
          <label className="checkbox-row" style={{ marginTop: 8 }}>
            <input type="checkbox" checked={settings.skeleton} onChange={set('skeleton')} /> Draw skeleton
          </label>
          <label className="checkbox-row" style={{ marginTop: 8 }}>
            <input type="checkbox" checked={settings.show_angle} onChange={set('show_angle')} /> Show CVA angle
          </label>
          <label className="checkbox-row" style={{ marginTop: 8 }}>
            <input type="checkbox" checked={settings.snapshot} onChange={set('snapshot')} /> Evolution snapshot
          </label>
        </div>
      </div>
    </div>
  )
}
