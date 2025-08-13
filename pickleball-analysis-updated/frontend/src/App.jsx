import React, { useRef, useState, useEffect } from 'react'
import VideoPlayer from './components/VideoPlayer'

export default function App(){
  const [fileUrl, setFileUrl] = useState(null)
  const [playing, setPlaying] = useState(false)
  const [tips, setTips] = useState([])
  const overlayRef = useRef(null)

  const handleFile = (e) => {
    const f = e.target.files[0]
    if (f) {
      setFileUrl(URL.createObjectURL(f))
      setTips([])
    }
  }

  // Called for each captured frame's canvas element
  const onFrame = async (canvas) => {
    // Send a downscaled JPEG every ~6th frame to reduce load
    try {
      const blob = await new Promise((res) => canvas.toBlob(res, 'image/jpeg', 0.7))
      if (!blob) return
      const fd = new FormData()
      fd.append('frame', blob, 'frame.jpg')
      const resp = await fetch('http://localhost:8000/api/analyze', { method: 'POST', body: fd })
      if (!resp.ok) return
      const data = await resp.json()
      drawOverlay(canvas, data)
      if (data.tips) setTips(data.tips)
    } catch (e) {
      // ignore network errors for now
    }
  }

  const drawOverlay = (canvas, data) => {
    const ctx = canvas.getContext('2d')
    // draw detection overlays lightly on the same canvas
    if (!data) return
    if (data.ball) {
      ctx.beginPath()
      ctx.strokeStyle = 'orange'
      ctx.lineWidth = 3
      ctx.arc(data.ball.x, data.ball.y, Math.max(4, data.ball.r||6), 0, Math.PI*2)
      ctx.stroke()
    }
    if (data.traj && data.traj.length>1) {
      ctx.beginPath(); ctx.strokeStyle = 'yellow'; ctx.lineWidth = 2
      ctx.moveTo(data.traj[0].x, data.traj[0].y)
      data.traj.forEach(p => ctx.lineTo(p.x,p.y))
      ctx.stroke()
    }
    if (data.players) {
      data.players.forEach(p => {
        ctx.fillStyle = 'rgba(0,150,255,0.8)'
        ctx.beginPath(); ctx.arc(p.x, p.y, 12, 0, Math.PI*2); ctx.fill()
      })
    }
  }

  return (
    <div className="app">
      <h1>Pickleball Analysis — Local</h1>
      <div className="upload">
        <input type="file" accept="video/*" onChange={handleFile} />
      </div>

      {fileUrl ? (
        <>
          <VideoPlayer src={fileUrl} onFrame={onFrame} playing={playing} />
          <div className="controls">
            <button className="button" onClick={() => setPlaying(p => !p)}>{playing ? 'Stop' : 'Start'}</button>
          </div>
        </>
      ) : (
        <div>Please upload a pickleball video file to start.</div>
      )}

      <div className="tips">
        <h3>Tips & Summary</h3>
        {tips.length ? (
          <ol>
            {tips.map((t,i) => <li key={i}>{t}</li>)}
          </ol>
        ) : (
          <div>No tips yet. Play a video and the server will return tips in real time.</div>
        )}
      </div>
    </div>
  )
}
