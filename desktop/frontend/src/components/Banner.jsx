// Ported from posture_app.py's render_banner(): same priority order —
// framing problems beat everything else since bad framing makes every
// other reading untrustworthy, then "no subject", then a predicted slump,
// then whichever metric is dominating the score, then fatigue, then "aligned".
function bannerInfo(state) {
  if (state.framing) return { kicker: 'FRAMING', msg: state.framing, cls: 'watch' }
  if (state.status === 'IDLE') {
    return { kicker: 'NO SUBJECT', msg: "Step back into frame — I can't see your shoulders.", cls: '' }
  }
  if (state.predicting && state.status !== 'BAD') {
    return {
      kicker: 'PREDICTED', cls: 'watch',
      msg: `You're drifting. At this rate you'll be slouching in ~${Math.round(state.eta)}s — reset now, before it costs you.`,
    }
  }
  if (state.tip) {
    return {
      kicker: state.status === 'BAD' ? 'CORRECT' : 'ADJUST',
      msg: state.tip,
      cls: state.status === 'BAD' ? 'bad' : state.status === 'WATCH' ? 'watch' : 'good',
    }
  }
  if (state.microsleeps && state.fatigue > 60) {
    return { kicker: 'FATIGUE', msg: 'Eyes are closing for too long. Take a real break.', cls: 'bad' }
  }
  return { kicker: 'ALIGNED', msg: 'Cervical loading is in a healthy range. Keep it here.', cls: 'good' }
}

export default function Banner({ state }) {
  if (!state) return null
  if (!state.calibrated) {
    return (
      <div className="banner">
        <span className="kicker">CALIBRATING</span>
        <span>Sit the way you want to sit for the next hour. Everything after this is measured against this exact posture.</span>
      </div>
    )
  }
  if (state.on_break) return null
  const { kicker, msg, cls } = bannerInfo(state)
  return (
    <div className={`banner ${cls}`}>
      <span className="kicker">{kicker}</span>
      <span>{msg}</span>
    </div>
  )
}
