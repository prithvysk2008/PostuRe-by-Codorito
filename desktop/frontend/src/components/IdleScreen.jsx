import { onGradientMouseMove, scoreColor } from '../format.js'

function friendlyDate(iso) {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso.split('T')[0]
  const today = new Date(); today.setHours(0, 0, 0, 0)
  const dd = new Date(d); dd.setHours(0, 0, 0, 0)
  const deltaDays = Math.round((today - dd) / 86400000)
  if (deltaDays === 0) return 'Today'
  if (deltaDays === 1) return 'Yesterday'
  if (deltaDays >= 2 && deltaDays <= 6) return dd.toLocaleDateString('en-US', { weekday: 'long' })
  return dd.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
}

function friendlyTime(iso) {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return ''
  return d.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' })
}

export default function IdleScreen({ history }) {
  const sessions = history?.sessions || []
  const bestAvg = sessions.length ? Math.max(...sessions.map((s) => s.avg_score)) : 0
  const bestAge = sessions.length ? Math.min(...sessions.map((s) => s.spine_age)) : 0

  return (
    <div>
      <div className="banner">
        <span className="kicker">READY</span>
        <span>Press <b>Start session</b> below. Five seconds of calibration, then the coaching is live.</span>
      </div>

      <div className="cards-grid" style={{ gridTemplateColumns: 'repeat(3, 1fr)', marginBottom: 22 }}>
        <div className="card">
          <div className="label">01 &middot; MEASURE</div>
          <div className="sub" style={{ color: 'var(--paper)', fontSize: '.9rem' }}>
            Craniovertebral angle, neck collapse, head pitch, chair slide and shoulder tilt — all normalised
            by your shoulder width, so leaning closer to the camera doesn't fake a score.
          </div>
        </div>
        <div className="card">
          <div className="label">02 &middot; PREDICT</div>
          <div className="sub" style={{ color: 'var(--paper)', fontSize: '.9rem' }}>
            A 45-second regression on your score spots the slump forming and nudges you before it happens,
            instead of scolding you after.
          </div>
        </div>
        <div className="card">
          <div className="label">03 &middot; REWARD</div>
          <div className="sub" style={{ color: 'var(--paper)', fontSize: '.9rem' }}>
            Streaks, recovery saves for fixing a slouch fast, and one shareable Spine Card at the end.
            Ambient colour and soft chimes only — no pop-ups.
          </div>
        </div>
      </div>

      {sessions.length > 0 && (
        <>
          <div className="section-title" style={{ marginBottom: 10 }}>Session history &middot; stored on this machine only</div>
          <div className="hist-strip">
            <div className="hist-stat gradient" onMouseMove={onGradientMouseMove}><b>{sessions.length}</b><span>Sessions</span></div>
            <div className="hist-stat gradient" onMouseMove={onGradientMouseMove}><b>{bestAvg.toFixed(0)}</b><span>Best avg score</span></div>
            <div className="hist-stat gradient" onMouseMove={onGradientMouseMove}><b>{bestAge}</b><span>Best spine age</span></div>
            <div className="hist-stat gradient" onMouseMove={onGradientMouseMove}><b>{history?.daily_streak ?? 0}</b><span>Day streak</span></div>
          </div>
          {[...sessions].slice(-8).reverse().map((s, i) => {
            const c = scoreColor(s.avg_score)
            return (
              <div className="hist-row" key={i}>
                <div>{friendlyDate(s.at)}</div>
                <div className="dim">{friendlyTime(s.at)}</div>
                <div className="dim">{s.minutes.toFixed(0)} min</div>
                <div className="hist-bar"><i style={{ width: `${Math.max(0, Math.min(100, s.avg_score))}%`, background: c }} /></div>
                <div style={{ color: c, fontFamily: 'var(--mono)', fontWeight: 700, textAlign: 'right' }}>{s.avg_score.toFixed(0)}</div>
                <div className="dim" style={{ textAlign: 'right', fontSize: '.68rem' }}>age {s.spine_age}</div>
              </div>
            )
          })}
        </>
      )}
    </div>
  )
}
