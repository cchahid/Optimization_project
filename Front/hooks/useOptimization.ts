import { useState, useCallback, useRef } from "react"
import { toast } from "sonner"

export interface PathPoint {
  x: number
  y: number
}

interface OptimizationParams {
  learningRate: number
  iterations: number
  swarmSize: number
}

export function useOptimization() {
  const [isOptimizing, setIsOptimizing] = useState(false)
  const [hasOptimized, setHasOptimized] = useState(false)
  const [runCount, setRunCount] = useState(0)

  const [sgdPath, setSgdPath] = useState<PathPoint[]>([])
  const [adamPath, setAdamPath] = useState<PathPoint[]>([])
  const [psoPaths, setPsoPaths] = useState<PathPoint[][]>([])

  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const runOptimization = useCallback(async (params: OptimizationParams) => {
    if (isOptimizing) return
    setIsOptimizing(true)
    setHasOptimized(false)
    setSgdPath([])
    setAdamPath([])
    setPsoPaths([])

    if (intervalRef.current) clearInterval(intervalRef.current)

    let isSuccess = false

    try {
      // Updated endpoint URL to point directly to your public Codespace proxy port
      const res = await fetch("https://laughing-orbit-jx7v5ww7rwp3p4q-8000.app.github.dev/run-optimization", {
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
      const fullSgd = data.sgd_path as PathPoint[]
      const fullAdam = data.adam_path as PathPoint[]
      const fullPso = data.pso_path as PathPoint[][]

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
          setIsOptimizing(false)
          setHasOptimized(true)
          setRunCount((c) => c + 1)
        }
      }, 50)
      
      isSuccess = true
    } catch (err) {
      console.error("Failed to run optimization", err)
      toast.error('Backend Error: Could not connect to FastAPI. Please ensure the server is running on port 8000.')
    } finally {
      if (!isSuccess) {
        setIsOptimizing(false)
      }
    }
  }, [isOptimizing])

  return { isOptimizing, hasOptimized, runCount, sgdPath, adamPath, psoPaths, runOptimization }
}