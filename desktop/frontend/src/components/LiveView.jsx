import Banner from './Banner.jsx'
import BreakOverlay from './BreakOverlay.jsx'
import Cards from './Cards.jsx'
import VideoStage from './VideoStage.jsx'

export default function LiveView({ tick }) {
  const state = tick?.state || {}
  return (
    <>
      <Banner state={state} />
      <div className="live-stack">
        <VideoStage tick={tick} />
        <Cards state={state} />
        <BreakOverlay state={state} />
      </div>
    </>
  )
}
