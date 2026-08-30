// A small vertical "measurement gauge" — the one recurring visual motif,
// reused for the live CVA reading and the end-of-session Spine Age.
export default function Gauge({ pct, topLabel, bottomLabel, height = 60 }) {
  const clamped = Math.max(0, Math.min(1, pct ?? 0))
  const fillH = clamped * height
  return (
    <div className="gauge-wrap">
      <div className="gauge-track" style={{ height }}>
        <div className="gauge-fill" style={{ height: fillH }} />
      </div>
      <div className="gauge-labels" style={{ height }}>
        <span>{topLabel}</span>
        <span>{bottomLabel}</span>
      </div>
    </div>
  )
}
