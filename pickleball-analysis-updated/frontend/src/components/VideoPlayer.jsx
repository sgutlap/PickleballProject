import React, { useRef, useEffect } from 'react'

export default function VideoPlayer({ src, onFrame, playing }) {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)

  useEffect(() => {
    let raf
    const capture = () => {
      const v = videoRef.current
      const c = canvasRef.current
      if (v && c && v.readyState >= 2 && !v.paused && !v.ended) {
        c.width = v.videoWidth; c.height = v.videoHeight
        const ctx = c.getContext('2d')
        ctx.drawImage(v, 0, 0, c.width, c.height)
        onFrame && onFrame(c)
      }
      raf = requestAnimationFrame(capture)
    }
    raf = requestAnimationFrame(capture)
    return () => cancelAnimationFrame(raf)
  }, [onFrame])

  return (
    <div className="video-wrap">
      <video ref={videoRef} src={src} controls style={{width:'100%'}} crossOrigin="anonymous" />
      <canvas ref={canvasRef} className="overlay" />
    </div>
  )
}
