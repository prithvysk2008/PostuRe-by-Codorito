export function fmtClock(seconds) {
  seconds = Math.max(0, Math.floor(seconds || 0))
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = seconds % 60
  if (h) return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
  return `${m}:${String(s).padStart(2, '0')}`
}

export const STATUS_COLOR = {
  GOOD: 'var(--good)', WATCH: 'var(--hazard)', BAD: 'var(--critical)', IDLE: 'var(--drafting)',
}

// Mirrors PostureEngine's status_for() thresholds (T_GOOD=80, T_WATCH=60).
export function scoreColor(score) {
  if (score >= 80) return 'var(--good)'
  if (score >= 60) return 'var(--hazard)'
  return 'var(--critical)'
}

// Origin-shift interaction shared by every gradient tile: hovering makes the
// tile's own multicolor gradient track the cursor (see .card.gradient in
// index.css) instead of drawing a highlight on top of it.
export function onGradientMouseMove(e) {
  const rect = e.currentTarget.getBoundingClientRect()
  e.currentTarget.style.setProperty('--mx', `${((e.clientX - rect.left) / rect.width) * 100}%`)
  e.currentTarget.style.setProperty('--my', `${((e.clientY - rect.top) / rect.height) * 100}%`)
}
