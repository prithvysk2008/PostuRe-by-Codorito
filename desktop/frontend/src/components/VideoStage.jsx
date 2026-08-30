import { STATUS_COLOR } from '../format.js'

export default function VideoStage({ tick }) {
  const frame = tick?.frame
  const status = tick?.status || 'IDLE'
  const statusText = tick?.state?.status_text || (status === 'IDLE' ? 'NO SUBJECT' : status)
  const color = STATUS_COLOR[status] || STATUS_COLOR.IDLE

  return (
    <div className="video-stage" style={{ borderColor: color }}>
      {frame ? (
        <img src={`data:image/jpeg;base64,${frame}`} alt="live posture feed" />
      ) : (
        <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--graphite)' }}>
          Starting camera…
        </div>
      )}
      {frame && <div className="badge" style={{ color, borderColor: color }}>{statusText}</div>}
      {frame && <div className="fps-tag">{Math.round(tick?.fps || 0)} FPS &middot; ON-DEVICE</div>}
    </div>
  )
}
