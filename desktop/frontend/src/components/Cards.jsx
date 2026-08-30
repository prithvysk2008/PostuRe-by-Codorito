import Gauge from './Gauge.jsx'
import { fmtClock, onGradientMouseMove, STATUS_COLOR } from '../format.js'

function Card({ label, value, unit, sub, color, bar, children }) {
  return (
    <div className="card gradient" onMouseMove={onGradientMouseMove}>
      <div className="label">{label}</div>
      <div className="value" style={{ color }}>{value}{unit && <span className="unit">{unit}</span>}</div>
      <div className="sub">{sub}</div>
      {bar != null && (
        <div className="bar"><i style={{ width: `${Math.max(0, Math.min(1, bar)) * 100}%`, background: color }} /></div>
      )}
      {children}
    </div>
  )
}

export default function Cards({ state }) {
  if (!state || !state.calibrated) return null
  const color = STATUS_COLOR[state.status] || STATUS_COLOR.IDLE

  let trendColor = 'var(--graphite)'
  let trendSub = 'Holding your baseline.'
  if (state.trend_label === 'RISING') { trendColor = 'var(--good)'; trendSub = "You're straightening up." }
  else if (state.trend_label === 'FALLING') {
    trendColor = 'var(--critical)'
    trendSub = state.eta ? `Slump predicted in ~${Math.round(state.eta)}s at this rate.` : 'Posture is decaying steadily.'
  }

  const fatColor = state.fatigue < 25 ? 'var(--good)' : state.fatigue < 50 ? 'var(--hazard)' : 'var(--critical)'
  let fatSub = `${Math.round(state.blink_rate)} blinks/min · ${state.yawns} yawns`
  if (state.microsleeps) fatSub += ` · ${state.microsleeps} micro-sleep${state.microsleeps > 1 ? 's' : ''}`
  if (state.dry_eye) fatSub += ' · dry-eye risk'

  let streakSub = `Best ${fmtClock(state.best_streak_s)}`
  if (state.recoveries) streakSub += ` · ${state.recoveries} save${state.recoveries > 1 ? 's' : ''} (+${state.bonus})`

  const nextBreak = state.next_break_s == null ? '—' : fmtClock(state.next_break_s)
  const cvaPct = state.cva != null ? Math.max(0, Math.min(1, (state.cva - 30) / 30)) : 0

  return (
    <div className="cards-grid">
      <Card label="Posture score" value={Math.round(state.score)} unit="/100" sub={state.status_text} color={color} bar={state.score / 100} />
      <Card label="Fatigue" value={state.fatigue_label?.toUpperCase()} sub={fatSub} color={fatColor} bar={state.fatigue / 100} />
      <Card label="Streak" value={fmtClock(state.streak_s)} sub={streakSub} color={state.streak_s > 0 ? 'var(--good)' : 'var(--graphite)'} />
      <Card label="Trend" value={state.trend_label} sub={trendSub} color={trendColor} />
      <Card label="CVA" value={state.cva != null ? Math.round(state.cva) : '—'} unit="°"
            sub={state.cva_baseline != null ? `Baseline ${Math.round(state.cva_baseline)}°` : 'Not calibrated'}
            color="var(--drafting)">
        <Gauge pct={cvaPct} topLabel="60°" bottomLabel="30°" height={64} />
      </Card>
      <Card label="Next break" value={nextBreak} sub={`${state.breaks_taken} taken · day streak ${state.day_streak}`} color="var(--drafting)" />
    </div>
  )
}
