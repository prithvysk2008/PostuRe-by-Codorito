export default function BreakOverlay({ state }) {
  if (!state?.on_break || !state.break) return null
  const { seconds_left, exercises, index, just_completed } = state.break
  const current = exercises?.[index]
  const prevName = index > 0 ? exercises[index - 1]?.name : null

  return (
    <div className="overlay-panel" style={{ marginTop: 12 }}>
      <div className="big">{seconds_left}</div>
      {current ? (
        <>
          <h3>{current.name}</h3>
          <p>{current.how}</p>
          {just_completed && prevName && (
            <div style={{
              display: 'inline-flex', gap: 6, marginTop: 8, padding: '5px 12px', borderRadius: 4,
              background: 'rgba(53,230,166,.14)', border: '1px solid rgba(53,230,166,.4)',
              color: 'var(--good)', fontFamily: 'var(--mono)', fontSize: '.68rem',
            }}>
              &#10003; {prevName} done
            </div>
          )}
          <p style={{ marginTop: 10, fontSize: '.7rem', letterSpacing: '.14em', color: 'var(--graphite)' }}>
            MOVE {index + 1} OF {exercises.length} &middot; {current.group?.toUpperCase()} &middot; SCORING PAUSED
          </p>
        </>
      ) : (
        <p>Stretch break</p>
      )}
    </div>
  )
}
