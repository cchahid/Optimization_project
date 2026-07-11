"use client"

import { useState } from "react"
import { ControlSidebar } from "@/components/control-sidebar"
import { VizCanvas } from "@/components/viz-canvas"
import { useOptimization } from "@/hooks/useOptimization"
import { useTraining } from "@/hooks/useTraining"

export interface Params {
  learningRate: number
  iterations: number
  swarmSize: number
}

export type { PathPoint } from "@/hooks/useOptimization"

export default function DashboardPage() {
  const [params, setParams] = useState<Params>({
    learningRate: 0.012,
    iterations: 100,
    swarmSize: 30,
  })
  const [showBaseline, setShowBaseline] = useState(false)
  
  const { isOptimizing, hasOptimized, runCount, sgdPath, adamPath, psoPaths, runOptimization } = useOptimization()
  const { isTraining, trainingData, runTraining } = useTraining()

  const handleBaseline = async () => {
    if (!trainingData && !isTraining) {
      await runTraining()
    }
    setShowBaseline((prev) => !prev)
  }

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-background">
      <ControlSidebar
        onExecuteTraining={runTraining}
        onToggleBaseline={handleBaseline}
        onExecuteOptimization={() => runOptimization(params)}
        isTraining={isTraining}
        isOptimizing={isOptimizing}
        isBaselineVisible={showBaseline}
        params={params}
        onParamsChange={setParams}
      />
      <VizCanvas
        isOptimizing={isOptimizing}
        hasOptimized={hasOptimized}
        params={params}
        runCount={runCount}
        sgdPath={sgdPath}
        adamPath={adamPath}
        psoPaths={psoPaths}
        isTraining={isTraining}
        trainingData={trainingData}
        showBaseline={showBaseline}
      />
    </div>
  )
}
