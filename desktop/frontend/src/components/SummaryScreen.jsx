import { useEffect, useMemo, useState } from 'react'
import Gauge from './Gauge.jsx'
import { fmtClock, onGradientMouseMove, scoreColor } from '../format.js'

const STYLE_LABELS = { minimal: 'Minimal', data: 'Data-rich', story: 'Story (9:16)' }

function CmpBarCard({ label, prevVal, curVal, maxVal, better, fmt }) {
  const frac = (v) => Math.max(0.04, Math.min(1, v / maxVal))
  const diff = curVal - prevVal
  const improved = better === 'lower' ? diff < 0 : diff > 0
  const deltaColor = improved ? 'var(--good)' : Math.abs(diff) < 1e-6 ? 'var(--graphite)' : 'var(--hazard)'
  const deltaTxt = Math.abs(diff) < 0.5 ? 'No change vs last session' : `${diff > 0 ? '+' : ''}${diff.toFixed(0)} vs last session`
  const curColor = improved ? 'var(--good)' : 'var(--drafting)'
  return (
    <div className="card gradient" onMouseMove={onGradientMouseMove}>
      <div className="label">{label}</div>
      <div className="cmp-track"><i style={{ width: `${frac(prevVal) * 100}%`, background: 'var(--graphite)' }} /><span>Last {fmt(prevVal)}</span></div>
      <div className="cmp-track"><i style={{ width: `${frac(curVal) * 100}%`, background: curColor }} /><span>Now {fmt(curVal)}</span></div>
      <div className="cmp-delta" style={{ color: deltaColor }}>{deltaTxt}</div>
    </div>
  )
}

function Comparison({ summary }) {
  const total = Object.values(summary.time_in).reduce((a, b) => a + b, 0) || 1
  const good = (summary.time_in.GOOD / total) * 100
  const watch = (summary.time_in.WATCH / total) * 100
  const bad = (summary.time_in.BAD / total) * 100
  const prevList = summary.prev_sessions || []
  const prev = prevList[prevList.length - 1]

  const hist = [...prevList.slice(-7), { avg_score: summary.avg }]
  const pts = hist.length >= 2
    ? hist.map((s, i) => `${(i / (hist.length - 1)) * 600},${100 - Math.max(0, Math.min(100, s.avg_score))}`)
    : null

  return (
    <>
      <div className="section-title" style={{ margin: '26px 0 8px' }}>This session vs your history</div>
      <div className="cards-grid" style={{ gridTemplateColumns: 'repeat(3, 1fr)' }}>
        <div className="card gradient" onMouseMove={onGradientMouseMove}>
          <div className="label">Time split</div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 14, marginTop: 8 }}>
            <div className="ring" style={{
              background: `conic-gradient(var(--good) 0% ${good}%, var(--hazard) ${good}% ${good + watch}%, var(--critical) ${good + watch}% 100%)`,
              position: 'relative',
            }}>
              <div style={{ position: 'absolute', inset: '22%', borderRadius: '50%', background: 'var(--ink-2)' }} />
            </div>
            <div className="donut-legend">
              <span><i style={{ background: 'var(--good)' }} />Aligned {good.toFixed(0)}%</span>
              <span><i style={{ background: 'var(--hazard)' }} />Drifting {watch.toFixed(0)}%</span>
              <span><i style={{ background: 'var(--critical)' }} />Slouched {bad.toFixed(0)}%</span>
            </div>
          </div>
        </div>
        {prev ? (
          <>
            <CmpBarCard label="Avg score" prevVal={prev.avg_score} curVal={summary.avg} maxVal={100} better="higher" fmt={(v) => v.toFixed(0)} />
            <CmpBarCard label="Spine age" prevVal={prev.spine_age} curVal={summary.age} maxVal={79} better="lower" fmt={(v) => v.toFixed(0)} />
          </>
        ) : (
          <div className="card gradient" onMouseMove={onGradientMouseMove} style={{ gridColumn: 'span 2' }}>
            <div className="label">First session</div>
            <div className="sub">Finish a few more sessions and comparisons will show up here.</div>
          </div>
        )}
      </div>
      {pts && (
        <div className="card gradient" onMouseMove={onGradientMouseMove} style={{ marginTop: 14 }}>
          <svg viewBox="0 0 600 100" preserveAspectRatio="none" style={{ width: '100%', height: 70, display: 'block' }}>
            <polyline points={pts.join(' ')} fill="none" stroke="var(--drafting)" strokeWidth="2.4"
                      strokeLinejoin="round" strokeLinecap="round" vectorEffect="non-scaling-stroke" />
          </svg>
          <div className="sub">Last {hist.length} session{hist.length > 1 ? 's' : ''} &middot; avg score trend</div>
        </div>
      )}
    </>
  )
}

export default function SummaryScreen({ summary, requestSpineCard, spineCard, onStartAnother, onGoHome }) {
  const [style, setStyle] = useState('minimal')

  useEffect(() => {
    if (!spineCard[style]) requestSpineCard(style)
  }, [style, spineCard, requestSpineCard])

  const ageColor = summary.age <= 30 ? 'var(--good)' : summary.age <= 46 ? 'var(--hazard)' : 'var(--critical)'
  const agePct = 1 - Math.max(0, Math.min(1, (summary.age - 18) / (79 - 18)))
  const total = useMemo(() => Object.values(summary.time_in).reduce((a, b) => a + b, 0) || 1, [summary])

  const png = spineCard[style]
  const downloadHref = png ? `data:image/png;base64,${png}` : null

  return (
    <div>
      <div className="hero">
        <div className="cap">Spine Age</div>
        <div className="big" style={{ color: ageColor }}>{summary.age}</div>
        <h2 style={{ margin: '12px 0 2px' }}>{summary.label}</h2>
        <p style={{ color: 'var(--graphite)', maxWidth: 620, margin: '8px auto 0' }}>{summary.note}</p>
        <p className="explain">
          This isn't a literal age prediction — it's a wear score for this one session, where lower is
          always better. Even a genuinely perfect session lands in the low-to-mid 20s, so a young, healthy
          person seeing a number around there means they did almost everything right, not that something
          is wrong.
        </p>
        <div style={{ marginTop: 20, display: 'flex', justifyContent: 'center' }}>
          <Gauge pct={agePct} topLabel="BEST · 18" bottomLabel="WORST · 79" height={110} />
        </div>
      </div>

      <div className="banner" style={{ marginTop: 18 }}>
        <span className="kicker">FOR NEXT TIME</span>
        <span>{summary.suggestion}</span>
      </div>

      <div className="cards-grid" style={{ gridTemplateColumns: 'repeat(4, 1fr)' }}>
        <div className="card gradient" onMouseMove={onGradientMouseMove}>
          <div className="label">Session</div>
          <div className="value" style={{ color: 'var(--drafting)' }}>{fmtClock(summary.duration)}</div>
          <div className="sub">{summary.breaks} stretch breaks</div>
        </div>
        <div className="card gradient" onMouseMove={onGradientMouseMove}>
          <div className="label">Average score</div>
          <div className="value" style={{ color: scoreColor(summary.avg) }}>{summary.avg.toFixed(0)}<span className="unit">/100</span></div>
          <div className="sub">{((summary.time_in.GOOD / total) * 100).toFixed(0)}% aligned &middot; {((summary.time_in.BAD / total) * 100).toFixed(0)}% slouched</div>
          <div className="bar"><i style={{ width: `${summary.avg}%`, background: scoreColor(summary.avg) }} /></div>
        </div>
        <div className="card gradient" onMouseMove={onGradientMouseMove}>
          <div className="label">Best streak</div>
          <div className="value" style={{ color: 'var(--good)' }}>{fmtClock(summary.best_streak)}</div>
          <div className="sub">{summary.recoveries} recovery saves &middot; +{summary.bonus} bonus</div>
        </div>
        <div className="card gradient" onMouseMove={onGradientMouseMove}>
          <div className="label">Fatigue</div>
          <div className="value" style={{ color: summary.microsleeps ? 'var(--critical)' : 'var(--good)' }}>
            {summary.blinks}<span className="unit"> blinks</span>
          </div>
          <div className="sub">{summary.blink_rate.toFixed(0)}/min &middot; {summary.yawns} yawns &middot; {summary.microsleeps} micro-sleeps</div>
        </div>
      </div>

      <Comparison summary={summary} />

      {summary.baseline_jpg && summary.current_jpg && (
        <>
          <div className="section-title" style={{ margin: '22px 0 8px' }}>How you started vs how you finished</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
            <figure style={{ margin: 0 }}>
              <img src={`data:image/jpeg;base64,${summary.baseline_jpg}`} alt="baseline" style={{ width: '100%', borderRadius: 8, border: '1px solid var(--line)' }} />
              <figcaption className="sub">Calibrated baseline</figcaption>
            </figure>
            <figure style={{ margin: 0 }}>
              <img src={`data:image/jpeg;base64,${summary.current_jpg}`} alt="final frame" style={{ width: '100%', borderRadius: 8, border: '1px solid var(--line)' }} />
              <figcaption className="sub">Final frame</figcaption>
            </figure>
          </div>
        </>
      )}

      <div className="section-title" style={{ margin: '26px 0 8px' }}>Spine Card</div>
      <div className="style-picker">
        {Object.entries(STYLE_LABELS).map(([key, lbl]) => (
          <label key={key}>
            <input type="radio" name="spine-card-style" checked={style === key} onChange={() => setStyle(key)} />
            {lbl}
          </label>
        ))}
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, alignItems: 'start' }}>
        <div>
          {png ? (
            <>
              <img src={downloadHref} alt="Spine Card preview" style={{ width: style === 'story' ? 240 : 340, borderRadius: 8, border: '1px solid var(--line)' }} />
              <div style={{ marginTop: 10 }}>
                <a className="btn" href={downloadHref} download={`spine-card-${summary.age}-${style}.png`}
                   style={{ display: 'inline-block', textDecoration: 'none', width: 'auto', padding: '10px 18px' }}>
                  Download Spine Card
                </a>
              </div>
            </>
          ) : (
            <div className="dim">Rendering Spine Card…</div>
          )}
        </div>
        <div>
          <button className="btn primary" onClick={onStartAnother}>Start another session</button>
          <div style={{ marginTop: 10 }}>
            <button className="btn" onClick={onGoHome}>Back to home</button>
          </div>
        </div>
      </div>

      <div className="disclaimer">
        <span className="kicker">NOT MEDICAL ADVICE</span>
        <p>
          PostuRe estimates posture and fatigue from webcam video using on-device computer
          vision — it is a wellness and awareness tool, not a diagnostic or medical device.
          Spine age, scores, and coaching tips are session estimates, not clinical
          assessments. If you have persistent pain, numbness, or other symptoms, please
          consult a qualified doctor, physiotherapist, or ergonomics specialist.
        </p>
      </div>
    </div>
  )
}
