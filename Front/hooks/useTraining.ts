import { useState, useCallback } from "react"
import { toast } from "sonner"

export interface TrainingData {
  epochs: number[]
  sgd_loss: number[]
  adam_loss: number[]
  pso_loss: number[]
}

export function useTraining() {
  const [isTraining, setIsTraining] = useState(false)
  const [trainingData, setTrainingData] = useState<TrainingData | null>(null)

  const runTraining = useCallback(async () => {
    if (isTraining) return
    setIsTraining(true)
    try {
      const res = await fetch("http://127.0.0.1:8000/training-convergence")
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setTrainingData(data)
    } catch (err) {
      console.error("Failed to fetch training data", err)
      toast.error('Backend Error: Could not connect to FastAPI. Please ensure the server is running on port 8000.')
      setTrainingData(null)
    } finally {
      setIsTraining(false)
    }
  }, [isTraining])

  return { isTraining, trainingData, runTraining }
}
