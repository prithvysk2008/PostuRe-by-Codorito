import { useCallback, useEffect, useRef, useState } from 'react'

// The Python backend binds 127.0.0.1:8765 (see desktop/backend/server.py).
// Electron spawns that process on launch, but it takes a moment to come up
// (loading MediaPipe models etc.), so this hook retries the connection
// rather than failing once and giving up.
const WS_URL = 'ws://127.0.0.1:8765/ws'
const HTTP_BASE = 'http://127.0.0.1:8765'

export function useBackend() {
  const [connected, setConnected] = useState(false)
  const [tick, setTick] = useState(null)       // latest {running, frame, status, fps, state}
  const [summary, setSummary] = useState(null) // set when a session ends
  const [history, setHistory] = useState(null) // store data for the idle screen
  const [spineCard, setSpineCard] = useState({}) // { [style]: base64png }
  const wsRef = useRef(null)
  const retryRef = useRef(null)

  const connect = useCallback(() => {
    const ws = new WebSocket(WS_URL)
    wsRef.current = ws

    ws.onopen = () => {
      setConnected(true)
      ws.send(JSON.stringify({ type: 'get_history' }))
    }
    ws.onclose = () => {
      setConnected(false)
      retryRef.current = setTimeout(connect, 800)
    }
    ws.onerror = () => {
      try { ws.close() } catch { /* already closing */ }
    }
    ws.onmessage = (evt) => {
      let msg
      try { msg = JSON.parse(evt.data) } catch { return }
      if (msg.type === 'tick') {
        setTick(msg)
        if (msg.chime?.wav_b64) {
          try {
            new Audio(`data:audio/wav;base64,${msg.chime.wav_b64}`).play().catch(() => {})
          } catch { /* audio playback isn't essential — never let it break the session */ }
        }
      }
      else if (msg.type === 'summary') setSummary(msg.summary)
      else if (msg.type === 'history') setHistory(msg.store)
      else if (msg.type === 'spine_card') {
        setSpineCard((prev) => ({ ...prev, [msg.style]: msg.png_b64 }))
      }
    }
  }, [])

  useEffect(() => {
    connect()
    return () => {
      clearTimeout(retryRef.current)
      wsRef.current?.close()
    }
  }, [connect])

  const send = useCallback((obj) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(obj))
    }
  }, [])

  const startSession = useCallback(() => {
    setSummary(null)
    send({ type: 'start_session' })
  }, [send])
  const stopSession = useCallback(() => send({ type: 'stop_session' }), [send])
  const goHome = useCallback(() => setSummary(null), [])
  const recalibrate = useCallback(() => send({ type: 'recalibrate' }), [send])
  const updateSettings = useCallback((settings) => send({ type: 'update_settings', settings }), [send])
  const requestSpineCard = useCallback((style) => send({ type: 'get_spine_card', style }), [send])
  const refreshHistory = useCallback(() => send({ type: 'get_history' }), [send])

  return {
    connected, tick, summary, history, spineCard,
    startSession, stopSession, recalibrate, updateSettings, requestSpineCard, refreshHistory, goHome,
    httpBase: HTTP_BASE,
  }
}
