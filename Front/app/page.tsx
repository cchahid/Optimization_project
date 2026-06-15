"use client"

import { useState, useCallback, useRef } from "react"
import { ControlSidebar } from "@/components/control-sidebar"
import { VizCanvas } from "@/components/viz-canvas"

interface Params {
  learningRate: number
  iterations: number
  swarmSize: number
}

export interface PathPoint {
  x: number
  y: number
}

export interface OptimizationPaths {
  sgdPath: PathPoint[]
  adamPath: PathPoint[]
  psoPaths: PathPoint[][]
}

export default function DashboardPage() {
  const [params, setParams] = useState<Params>({
    learningRate: 0.012,
    iterations: 100,
    swarmSize: 30,
  })
  const [isComputing, setIsComputing] = useState(false)
  const [hasRun, setHasRun] = useState(false)
  const [runCount, setRunCount] = useState(0)

  // Animated path state — these grow incrementally via setInterval
  const [sgdPath, setSgdPath] = useState<PathPoint[]>([])
  const [adamPath, setAdamPath] = useState<PathPoint[]>([])
  const [psoPaths, setPsoPaths] = useState<PathPoint[][]>([])

  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const runOptimization = useCallback(async () => {
    if (isComputing) return
    setIsComputing(true)
    setHasRun(false)
    setSgdPath([])
    setAdamPath([])
    setPsoPaths([])

    // Clear any running animation
    if (intervalRef.current) clearInterval(intervalRef.current)

    let fullSgd: PathPoint[] = []
    let fullAdam: PathPoint[] = []
    let fullPso: PathPoint[][] = []

    try {
      const res = await fetch("http://127.0.0.1:8000/run-optimization", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          learning_rate: params.learningRate,
          iterations: params.iterations,
          num_particles: params.swarmSize,
        }),
      })

      if (!res.ok) throw new Error(`HTTP ${res.status}`)

      const data = await res.json()
      fullSgd = data.sgd_path as PathPoint[]
      fullAdam = data.adam_path as PathPoint[]
      fullPso = data.pso_path as PathPoint[][]
    } catch {
      // Backend not available — generate synthetic trajectories for demo
      fullSgd = generateSyntheticPath(params.iterations, params.learningRate, 2.5, -2.0, 0.6)
      fullAdam = generateSyntheticPath(params.iterations, params.learningRate * 2.2, -3.0, 1.5, 0.3)
      fullPso = Array.from({ length: Math.min(params.swarmSize, 12) }, (_, i) =>
        generateSyntheticPath(
          params.iterations,
          params.learningRate * (0.8 + Math.random() * 0.8),
          (Math.random() - 0.5) * 9,
          (Math.random() - 0.5) * 9,
          0.9 + Math.random() * 0.5
        )
      )
    }

    // Animate: incrementally reveal path points via setInterval
    const maxLen = Math.max(fullSgd.length, fullAdam.length, ...fullPso.map(p => p.length))
    let currentStep = 0

    intervalRef.current = setInterval(() => {
      currentStep += 1
      const idx = Math.min(currentStep, maxLen)

      setSgdPath(fullSgd.slice(0, idx))
      setAdamPath(fullAdam.slice(0, idx))
      setPsoPaths(fullPso.map((p) => p.slice(0, idx)))

      if (currentStep >= maxLen) {
        if (intervalRef.current) clearInterval(intervalRef.current)
        setIsComputing(false)
        setHasRun(true)
        setRunCount((c) => c + 1)
      }
    }, 50)
  }, [isComputing, params])

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-background">
      <ControlSidebar
        onExecute={runOptimization}
        isLoading={isComputing}
        params={params}
        onParamsChange={setParams}
      />
      <VizCanvas
        isLoading={isComputing}
        hasRun={hasRun}
        params={params}
        runCount={runCount}
        sgdPath={sgdPath}
        adamPath={adamPath}
        psoPaths={psoPaths}
      />
    </div>
  )
}

// ─── Synthetic fallback trajectory generator ──────────────────────────────────
function generateSyntheticPath(
  iterations: number,
  lr: number,
  startX: number,
  startY: number,
  noise: number
): PathPoint[] {
  const pts: PathPoint[] = []
  let x = startX
  let y = startY

  for (let i = 0; i < iterations; i++) {
    // Gradient of Rastrigin: dF/dx = 2x + 20*pi*sin(2*pi*x)
    const gx = 2 * x + 20 * Math.PI * Math.sin(2 * Math.PI * x)
    const gy = 2 * y + 20 * Math.PI * Math.sin(2 * Math.PI * y)

    x = x - lr * gx + (Math.random() - 0.5) * noise * 0.08 * Math.exp(-i * 0.015)
    y = y - lr * gy + (Math.random() - 0.5) * noise * 0.08 * Math.exp(-i * 0.015)

    // clamp to surface bounds
    x = Math.max(-5.12, Math.min(5.12, x))
    y = Math.max(-5.12, Math.min(5.12, y))

    pts.push({ x, y })
  }

  return pts
}
