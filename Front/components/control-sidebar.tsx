"use client"

import { useState } from "react"
import { Slider } from "@/components/ui/slider"
import { Button } from "@/components/ui/button"
import { Separator } from "@/components/ui/separator"
import {
  Play,
  Loader2,
  SlidersHorizontal,
  FlaskConical,
  Activity,
  Users,
  Cpu,
  Zap,
  Info,
} from "lucide-react"
import { cn } from "@/lib/utils"

interface ControlSidebarProps {
  onExecute: () => void
  isLoading: boolean
  params: {
    learningRate: number
    iterations: number
    swarmSize: number
  }
  onParamsChange: (params: { learningRate: number; iterations: number; swarmSize: number }) => void
}

function SliderRow({
  label,
  icon: Icon,
  value,
  displayValue,
  min,
  max,
  step,
  onChange,
  unit,
}: {
  label: string
  icon: React.ElementType
  value: number[]
  displayValue: string
  min: number
  max: number
  step: number
  onChange: (val: number[]) => void
  unit?: string
}) {
  return (
    <div className="flex flex-col gap-2.5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5">
          <Icon className="size-3.5 text-muted-foreground" />
          <span className="text-xs font-medium text-label tracking-wide uppercase">{label}</span>
        </div>
        <span className="font-mono text-xs text-accent-cyan tabular-nums">
          {displayValue}
          {unit && <span className="text-muted-foreground ml-0.5">{unit}</span>}
        </span>
      </div>
      <Slider
        value={value}
        min={min}
        max={max}
        step={step}
        onValueChange={onChange}
        className="w-full"
      />
      <div className="flex justify-between">
        <span className="font-mono text-[10px] text-muted-foreground/60">{min}</span>
        <span className="font-mono text-[10px] text-muted-foreground/60">{max}</span>
      </div>
    </div>
  )
}

function MetricChip({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between px-3 py-2 rounded bg-panel border border-panel-border/50">
      <span className="text-[10px] uppercase tracking-widest text-muted-foreground font-medium">{label}</span>
      <span className="font-mono text-xs text-foreground tabular-nums">{value}</span>
    </div>
  )
}

export function ControlSidebar({ onExecute, isLoading, params, onParamsChange }: ControlSidebarProps) {
  const lrDisplay = params.learningRate.toFixed(4)
  const iterDisplay = String(params.iterations).padStart(3, "0")
  const swarmDisplay = String(params.swarmSize).padStart(2, "0")

  // Derived computed params
  const totalSteps = params.iterations * 3
  const convergenceEst = (params.learningRate * params.swarmSize * 0.4).toFixed(4)

  return (
    <aside className="w-[300px] min-w-[300px] h-screen bg-card border-r border-panel-border flex flex-col overflow-y-auto">
      {/* Header */}
      <div className="px-5 pt-6 pb-5 border-b border-panel-border">
        <div className="flex items-start gap-2.5 mb-3">
          <div className="mt-0.5 size-7 rounded bg-panel border border-panel-border flex items-center justify-center flex-shrink-0">
            <FlaskConical className="size-3.5 text-accent-cyan" />
          </div>
          <div>
            <h1 className="font-heading text-[15px] font-semibold leading-snug text-foreground text-balance">
              Empirical Analysis of Optimization
            </h1>
            <p className="text-[10px] text-muted-foreground mt-0.5 tracking-wide font-mono">
              TRAJECTORY VISUALIZER v2.4.1
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2 mt-3">
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[9px] font-mono uppercase tracking-widest bg-accent-cyan/10 text-accent-cyan border border-accent-cyan/20">
            <span className="size-1.5 rounded-full bg-accent-cyan inline-block animate-pulse" />
            READY
          </span>
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[9px] font-mono uppercase tracking-widest bg-panel border border-panel-border text-muted-foreground">
            <Cpu className="size-2.5" />
            3 SOLVERS
          </span>
        </div>
      </div>

      {/* Parameters */}
      <div className="px-5 py-5 flex flex-col gap-1">
        <div className="flex items-center gap-1.5 mb-4">
          <SlidersHorizontal className="size-3 text-muted-foreground" />
          <span className="text-[9px] uppercase tracking-widest text-muted-foreground font-semibold">
            Hyperparameter Configuration
          </span>
        </div>

        <div className="flex flex-col gap-6">
          <SliderRow
            label="Learning Rate"
            icon={Zap}
            value={[params.learningRate]}
            displayValue={lrDisplay}
            min={0.001}
            max={0.1}
            step={0.001}
            onChange={([v]) => onParamsChange({ ...params, learningRate: v })}
          />

          <Separator className="opacity-30" />

          <SliderRow
            label="Iterations"
            icon={Activity}
            value={[params.iterations]}
            displayValue={iterDisplay}
            min={10}
            max={200}
            step={1}
            onChange={([v]) => onParamsChange({ ...params, iterations: v })}
          />

          <Separator className="opacity-30" />

          <SliderRow
            label="Swarm Size"
            icon={Users}
            value={[params.swarmSize]}
            displayValue={swarmDisplay}
            min={10}
            max={50}
            step={1}
            onChange={([v]) => onParamsChange({ ...params, swarmSize: v })}
            unit=" agents"
          />
        </div>
      </div>

      <Separator className="opacity-30 mx-5" />

      {/* Computed metrics */}
      <div className="px-5 py-4 flex flex-col gap-2">
        <div className="flex items-center gap-1.5 mb-2">
          <Info className="size-3 text-muted-foreground" />
          <span className="text-[9px] uppercase tracking-widest text-muted-foreground font-semibold">
            Derived Parameters
          </span>
        </div>
        <MetricChip label="Total Eval Steps" value={String(totalSteps).padStart(5, "0")} />
        <MetricChip label="Convergence Est." value={convergenceEst} />
        <MetricChip label="Solvers" value="SGD / Adam / PSO" />
        <MetricChip label="Loss Surface" value="Rosenbrock-6D" />
      </div>

      <Separator className="opacity-30 mx-5" />

      {/* Solver toggle rows */}
      <div className="px-5 py-4 flex flex-col gap-2">
        <div className="flex items-center gap-1.5 mb-2">
          <span className="text-[9px] uppercase tracking-widest text-muted-foreground font-semibold">
            Active Solvers
          </span>
        </div>
        <SolverRow name="SGD" color="accent-amber" active />
        <SolverRow name="Adam" color="accent-cyan" active />
        <SolverRow name="PSO" color="accent-white" active />
      </div>

      {/* Spacer */}
      <div className="flex-1" />

      {/* Execute button */}
      <div className="px-5 py-5 border-t border-panel-border">
        <Button
          onClick={onExecute}
          disabled={isLoading}
          className={cn(
            "w-full font-mono text-xs tracking-widest uppercase h-10",
            "bg-primary text-primary-foreground",
            "hover:bg-primary/90 transition-colors",
            "disabled:opacity-60"
          )}
        >
          {isLoading ? (
            <>
              <Loader2 className="size-3.5 animate-spin mr-2" />
              Computing Trajectories...
            </>
          ) : (
            <>
              <Play className="size-3.5 mr-2" />
              Execute Optimization
            </>
          )}
        </Button>
        <p className="text-center text-[9px] font-mono text-muted-foreground/50 mt-2 tracking-wider">
          EST. RUNTIME ≈ {(params.iterations * 0.012).toFixed(2)}s
        </p>
      </div>
    </aside>
  )
}

function SolverRow({ name, color, active }: { name: string; color: string; active: boolean }) {
  const dotColor = {
    "accent-amber": "bg-accent-amber",
    "accent-cyan": "bg-accent-cyan",
    "accent-white": "bg-accent-white",
  }[color] ?? "bg-muted-foreground"

  const textColor = {
    "accent-amber": "text-accent-amber",
    "accent-cyan": "text-accent-cyan",
    "accent-white": "text-accent-white",
  }[color] ?? "text-foreground"

  return (
    <div className="flex items-center justify-between px-3 py-1.5 rounded border border-panel-border/40 bg-panel/40">
      <div className="flex items-center gap-2">
        <span className={cn("size-1.5 rounded-full flex-shrink-0", dotColor)} />
        <span className={cn("font-mono text-[11px] font-semibold tracking-widest", textColor)}>{name}</span>
      </div>
      <span className={cn(
        "text-[9px] font-mono uppercase tracking-wider px-1.5 py-0.5 rounded",
        active
          ? "text-accent-cyan/80 bg-accent-cyan/10"
          : "text-muted-foreground bg-panel"
      )}>
        {active ? "ON" : "OFF"}
      </span>
    </div>
  )
}
