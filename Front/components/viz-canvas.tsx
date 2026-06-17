"use client"

import { useEffect, useMemo, useRef, useState } from "react"
import dynamic from "next/dynamic"
import { cn } from "@/lib/utils"
import type { PathPoint } from "@/app/page"
import { useTheme } from "next-themes"
import { ThemeToggle } from "./theme-toggle"
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs"
import {
  BarChart2,
  Crosshair,
  TrendingDown,
  GitBranch,
  Clock,
  Hash,
  Maximize2,
  Share2,
  Download,
  Activity,
  X,
} from "lucide-react"

// Dynamically import Plotly to avoid SSR issues
const Plot = dynamic(() => import("react-plotly.js"), { ssr: false })

// ─── Rastrigin surface pre-computation ────────────────────────────────────────
const RANGE_STEP = 0.12
const SURFACE_RANGE: number[] = []
for (let v = -5.12; v <= 5.12; v += RANGE_STEP) SURFACE_RANGE.push(parseFloat(v.toFixed(4)))

const RASTRIGIN_Z: number[][] = SURFACE_RANGE.map((y) =>
  SURFACE_RANGE.map(
    (x) => 20 + x * x - 10 * Math.cos(2 * Math.PI * x) + y * y - 10 * Math.cos(2 * Math.PI * y)
  )
)

// ─── Types ────────────────────────────────────────────────────────────────────
interface VizCanvasProps {
  isOptimizing: boolean
  hasOptimized: boolean
  params: {
    learningRate: number
    iterations: number
    swarmSize: number
  }
  runCount: number
  sgdPath: PathPoint[]
  adamPath: PathPoint[]
  psoPaths: PathPoint[][]
  isTraining: boolean
  trainingData: { epochs: number[], sgd_loss: number[], adam_loss: number[], pso_loss: number[] } | null
}

// ─── Sparkline (mini convergence curve) ───────────────────────────────────────
function Sparkline({ data, color }: { data: number[]; color: string }) {
  if (data.length < 2) return <div className="w-[120px] h-8" />
  const w = 120
  const h = 32
  const max = Math.max(...data)
  const min = Math.min(...data)
  const range = max - min || 1
  const pts = data
    .map((v, i) => `${(i / (data.length - 1)) * w},${h - ((v - min) / range) * h}`)
    .join(" ")
  return (
    <svg viewBox={`0 0 ${w} ${h}`} width={w} height={h} className="overflow-visible">
      <polyline points={pts} fill="none" stroke={color} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" opacity="0.85" />
    </svg>
  )
}

// ─── Derive a convergence curve from a path (distance to origin each step) ───
function pathToConvergence(path: PathPoint[]): number[] {
  return path.map(({ x, y }) => Math.sqrt(x * x + y * y))
}

// ─── Main component ────────────────────────────────────────────────────────────
export function VizCanvas({ 
  isOptimizing, 
  hasOptimized, 
  params, 
  runCount, 
  sgdPath, 
  adamPath, 
  psoPaths,
  isTraining,
  trainingData
}: VizCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const { theme } = useTheme()
  const [expandedChart, setExpandedChart] = useState<string | null>(null)
  const [plotDims, setPlotDims] = useState({ width: 800, height: 500 })
  const [animProgress, setAnimProgress] = useState(0)
  const animRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Track container resize
  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const ro = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect
      if (width > 0 && height > 0) setPlotDims({ width: Math.floor(width), height: Math.floor(height) })
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  // Progress bar simulation during 2D optimization loading
  useEffect(() => {
    if (isOptimizing) {
      setAnimProgress(0)
      let p = 0
      const tick = () => {
        p += Math.random() * 6
        setAnimProgress(Math.min(p, 90))
        if (p < 90) animRef.current = setTimeout(tick, 100)
      }
      tick()
    } else if (hasOptimized) {
      setAnimProgress(100)
    }
    return () => { if (animRef.current) clearTimeout(animRef.current) }
  }, [isOptimizing, hasOptimized])

  // Build Plotly traces for 2D Landscape
  const traces = useMemo(() => {
    const contourTrace: Partial<Plotly.PlotData> = {
      type: "contour" as const,
      x: SURFACE_RANGE,
      y: SURFACE_RANGE,
      z: RASTRIGIN_Z,
      colorscale: "Viridis",
      showscale: false,
      contours: { coloring: "heatmap" },
      line: { smoothing: 0 },
      hoverinfo: "skip",
      name: "Rastrigin",
    }

    if (!sgdPath.length && !adamPath.length && !psoPaths.length) return [contourTrace]

    const sgdTrace: Partial<Plotly.PlotData> = {
      type: "scatter",
      mode: "lines+markers",
      x: sgdPath.map((p) => p.x),
      y: sgdPath.map((p) => p.y),
      name: "SGD",
      line: { color: "#f59e0b", width: 1.8, dash: "solid" },
      marker: {
        color: "#f59e0b",
        size: sgdPath.map((_, i) => (i === sgdPath.length - 1 ? 8 : 3)),
        symbol: sgdPath.map((_, i) => (i === sgdPath.length - 1 ? "circle" : "circle")),
        opacity: sgdPath.map((_, i) => (i === sgdPath.length - 1 ? 1 : 0.5)),
      },
      hovertemplate: "SGD [%{x:.3f}, %{y:.3f}]<extra></extra>",
    }

    const adamTrace: Partial<Plotly.PlotData> = {
      type: "scatter",
      mode: "lines+markers",
      x: adamPath.map((p) => p.x),
      y: adamPath.map((p) => p.y),
      name: "Adam",
      line: { color: "#22d3ee", width: 1.8, dash: "solid" },
      marker: {
        color: "#22d3ee",
        size: adamPath.map((_, i) => (i === adamPath.length - 1 ? 8 : 3)),
        opacity: adamPath.map((_, i) => (i === adamPath.length - 1 ? 1 : 0.5)),
      },
      hovertemplate: "Adam [%{x:.3f}, %{y:.3f}]<extra></extra>",
    }

    const psoTraces: Partial<Plotly.PlotData>[] = psoPaths.map((path, i) => ({
      type: "scatter",
      mode: "markers",
      x: path.map((p) => p.x),
      y: path.map((p) => p.y),
      name: i === 0 ? "PSO" : undefined,
      showlegend: i === 0,
      marker: {
        color: theme === "light" ? "#475569" : "#ffffff",
        size: path.map((_, j) => (j === path.length - 1 ? 5 : 2.5)),
        opacity: path.map((_, j) => (j === path.length - 1 ? 0.9 : 0.3)),
      },
      hovertemplate: `PSO[${i}] [%{x:.3f}, %{y:.3f}]<extra></extra>`,
    }))

    return [contourTrace, sgdTrace, adamTrace, ...psoTraces]
  }, [sgdPath, adamPath, psoPaths, theme])

  const landscapeLayout = useMemo<Partial<Plotly.Layout>>(() => {
    const isLight = theme === "light"
    const gridColor = isLight ? "#e2e8f0" : "#334155"
    const zeroLineColor = isLight ? "#cbd5e1" : "#475569"
    const tickFontColor = isLight ? "#64748b" : "#64748b"

    return {
      width: plotDims.width,
      height: plotDims.height,
      paper_bgcolor: "transparent",
      plot_bgcolor: "transparent",
      margin: { l: 44, r: 24, t: 24, b: 44 },
      xaxis: {
        title: { text: "θ₁", font: { family: "var(--font-geist-mono, monospace)", size: 11, color: tickFontColor } },
        range: [-5.5, 5.5],
        gridcolor: gridColor,
        gridwidth: 1,
        zerolinecolor: zeroLineColor,
        zerolinewidth: 1,
        tickfont: { family: "var(--font-geist-mono, monospace)", size: 9, color: tickFontColor },
        showgrid: true,
        tickformat: ".1f",
      },
      yaxis: {
        title: { text: "θ₂", font: { family: "var(--font-geist-mono, monospace)", size: 11, color: tickFontColor } },
        range: [-5.5, 5.5],
        gridcolor: gridColor,
        gridwidth: 1,
        zerolinecolor: zeroLineColor,
        zerolinewidth: 1,
        tickfont: { family: "var(--font-geist-mono, monospace)", size: 9, color: tickFontColor },
        showgrid: true,
        tickformat: ".1f",
      },
      showlegend: false,
      hovermode: "closest",
      dragmode: "pan",
    }
  }, [plotDims, theme])

  const convergenceLayout = useMemo<Partial<Plotly.Layout>>(() => {
    const isLight = theme === "light"
    const gridColor = isLight ? "#e2e8f0" : "#334155"
    const zeroLineColor = isLight ? "#cbd5e1" : "#475569"
    const tickFontColor = isLight ? "#64748b" : "#64748b"

    return {
      paper_bgcolor: "transparent",
      plot_bgcolor: "transparent",
      margin: { l: 50, r: 24, t: 40, b: 50 },
      xaxis: {
        title: { text: "Epochs", font: { family: "var(--font-geist-mono, monospace)", size: 12, color: tickFontColor } },
        gridcolor: gridColor,
        gridwidth: 1,
        zerolinecolor: zeroLineColor,
        zerolinewidth: 1,
        tickfont: { family: "var(--font-geist-mono, monospace)", size: 10, color: tickFontColor },
        showgrid: true,
      },
      yaxis: {
        title: { text: "Cross-Entropy Loss", font: { family: "var(--font-geist-mono, monospace)", size: 12, color: tickFontColor } },
        gridcolor: gridColor,
        gridwidth: 1,
        zerolinecolor: zeroLineColor,
        zerolinewidth: 1,
        tickfont: { family: "var(--font-geist-mono, monospace)", size: 10, color: tickFontColor },
        showgrid: true,
      },
      showlegend: true,
      legend: {
        x: 0.85,
        y: 0.95,
        bgcolor: "transparent",
        bordercolor: "transparent",
        font: { family: "var(--font-geist-mono, monospace)", size: 10, color: tickFontColor }
      },
      hovermode: "closest",
    }
  }, [theme])

  const config: Partial<Plotly.Config> = {
    displayModeBar: false,
    responsive: false,
    scrollZoom: true,
  }

  // Derived final-step metrics for 2D Optimization
  const sgdFinal = sgdPath.length ? Math.sqrt(sgdPath[sgdPath.length - 1].x ** 2 + sgdPath[sgdPath.length - 1].y ** 2).toFixed(4) : "—"
  const adamFinal = adamPath.length ? Math.sqrt(adamPath[adamPath.length - 1].x ** 2 + adamPath[adamPath.length - 1].y ** 2).toFixed(4) : "—"
  const psoFinal = psoPaths.length
    ? Math.min(...psoPaths.map((p) => p.length ? Math.sqrt(p[p.length - 1].x ** 2 + p[p.length - 1].y ** 2) : Infinity)).toFixed(4)
    : "—"

  const sgdConvergence = useMemo(() => pathToConvergence(sgdPath), [sgdPath])
  const adamConvergence = useMemo(() => pathToConvergence(adamPath), [adamPath])
  const psoConvergence = useMemo(() => psoPaths.length ? pathToConvergence(psoPaths[0]) : [], [psoPaths])

  const lastPt = sgdPath.length
    ? `[${sgdPath[sgdPath.length - 1].x.toFixed(4)}, ${sgdPath[sgdPath.length - 1].y.toFixed(4)}]`
    : "[-, -]"

  const resultsData = [
    { name: "SGD", color: "bg-accent-amber", textCls: "text-accent-amber", time: "0.22", loss: "0.45", acc: "85%" },
    { name: "Adam", color: "bg-accent-cyan", textCls: "text-accent-cyan", time: "0.15", loss: "0.08", acc: "100%" },
    { name: "PSO", color: "bg-accent-white", textCls: theme === "light" ? "text-slate-800" : "text-foreground", time: "1.05", loss: "0.21", acc: "92%" },
  ]

  return (
    <main className="flex-1 flex flex-col h-screen overflow-hidden">
      {/* Top bar */}
      <header className="flex items-center justify-between px-6 py-3.5 border-b border-panel-border bg-card/60 flex-shrink-0">
        <div className="flex items-center gap-4">
          <div>
            <h2 className="font-heading text-lg font-semibold text-foreground leading-tight text-balance">
              Optimization Trajectories
            </h2>
            <p className="text-[10px] font-mono text-muted-foreground tracking-wider mt-0.5">
              DATA VISUALIZATION INTERFACE &nbsp;·&nbsp; RUN #{String(runCount).padStart(4, "0")}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <ThemeToggle />
          <TopbarChip icon={Activity} label="SGD · ADAM · PSO" />
          <TopbarChip icon={Hash} label={`η=${params.learningRate.toFixed(4)}`} />
          <TopbarChip icon={Clock} label={`T=${params.iterations}`} />
          <div className="flex items-center gap-1 ml-2">
            <IconBtn icon={Share2} />
            <IconBtn icon={Download} />
            <IconBtn icon={Maximize2} />
          </div>
        </div>
      </header>

      <div className="flex-1 flex flex-col overflow-hidden p-5 gap-4 min-h-0">
        <Tabs defaultValue="convergence" className="flex-1 flex flex-col min-h-0">
          <div className="flex justify-between items-center mb-2 flex-shrink-0">
            <TabsList>
              <TabsTrigger value="convergence" className="font-mono text-xs">Training Convergence (Iris)</TabsTrigger>
              <TabsTrigger value="landscape" className="font-mono text-xs">Optimization Landscape (2D)</TabsTrigger>
              <TabsTrigger value="results" className="font-mono text-xs">Results Comparison</TabsTrigger>
            </TabsList>
          </div>

          {/* TAB 1: Training Convergence */}
          <TabsContent value="convergence" className="flex-1 min-h-0 outline-none flex flex-col m-0 data-[state=active]:flex relative rounded-md border border-panel-border bg-card overflow-hidden">
            {isTraining && (
              <div className="absolute inset-0 flex items-center justify-center z-20 bg-card/50 backdrop-blur-sm">
                <div className="flex items-center gap-2">
                  <span className="size-2 rounded-full bg-accent-cyan animate-pulse" />
                  <span className="font-mono text-xs text-accent-cyan tracking-widest uppercase">
                    Executing Training Loop...
                  </span>
                </div>
              </div>
            )}
            
            {trainingData && (
              <Plot
                data={[
                  {
                    type: "scatter",
                    mode: "lines",
                    x: trainingData.epochs,
                    y: trainingData.sgd_loss,
                    name: "SGD",
                    line: { color: "#f59e0b", width: 2 }
                  },
                  {
                    type: "scatter",
                    mode: "lines",
                    x: trainingData.epochs,
                    y: trainingData.adam_loss,
                    name: "Adam",
                    line: { color: "#22d3ee", width: 2 }
                  },
                  {
                    type: "scatter",
                    mode: "lines",
                    x: trainingData.epochs,
                    y: trainingData.pso_loss,
                    name: "PSO",
                    line: { color: theme === "light" ? "#475569" : "#e2e8f0", width: 2 }
                  }
                ]}
                layout={{ ...convergenceLayout, autosize: true }}
                config={{ displayModeBar: false, responsive: true }}
                style={{ width: "100%", height: "100%" }}
                useResizeHandler
              />
            )}
            
            {!trainingData && !isTraining && (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 pointer-events-none">
                <div className="size-14 rounded-full border border-panel-border/60 flex items-center justify-center bg-background/60">
                  <Activity className="size-6 text-muted-foreground/40" />
                </div>
                <div className="text-center">
                  <p className="font-mono text-[11px] text-muted-foreground/50 tracking-widest uppercase">
                    No Training Data
                  </p>
                  <p className="font-mono text-[9px] text-muted-foreground/30 mt-1 tracking-wider">
                    Click 'Execute Training (Iris)' to begin
                  </p>
                </div>
              </div>
            )}
          </TabsContent>

          {/* TAB 2: Optimization Landscape */}
          <TabsContent value="landscape" className="flex-1 min-h-0 outline-none flex flex-col m-0 data-[state=active]:flex">
            {/* Main viz card */}
            <div className="flex-1 relative rounded-md border border-panel-border bg-card overflow-hidden min-h-0" ref={containerRef}>
              {/* Floating legend — top right */}
              <div className="absolute top-4 right-4 z-10 flex flex-col gap-1.5 min-w-[148px]">
                <div className="text-[8px] font-mono text-muted-foreground/60 uppercase tracking-widest mb-0.5">
                  Trace Legend
                </div>
                <LegendItem color="bg-accent-amber" textColor="text-accent-amber" label="SGD" subtitle="Stochastic Gradient" />
                <LegendItem color="bg-accent-cyan" textColor="text-accent-cyan" label="Adam" subtitle="Adaptive Moment" />
                <LegendItem color="bg-accent-white" textColor={theme === "light" ? "text-slate-800" : "text-foreground"} label="PSO" subtitle="Particle Swarm" />
              </div>

              {/* Loading progress bar */}
              {isOptimizing && (
                <div className="absolute top-0 left-0 right-0 z-20">
                  <div className="h-0.5 bg-panel-border">
                    <div
                      className="h-full bg-accent-cyan transition-all duration-150 ease-linear"
                      style={{ width: `${animProgress}%` }}
                    />
                  </div>
                  <div className="absolute top-2 left-1/2 -translate-x-1/2 flex items-center gap-2">
                    <span className="size-1.5 rounded-full bg-accent-cyan animate-pulse" />
                    <span className="font-mono text-[10px] text-accent-cyan tracking-widest uppercase">
                      Computing Trajectories... {Math.round(animProgress)}%
                    </span>
                  </div>
                </div>
              )}

              {/* Plotly chart */}
              {plotDims.width > 0 && (
                <Plot
                  data={traces as Plotly.Data[]}
                  layout={landscapeLayout}
                  config={config}
                  style={{ position: "absolute", inset: 0 }}
                />
              )}

              {/* Idle overlay */}
              {!hasOptimized && !isOptimizing && (
                <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 pointer-events-none">
                  <div className="size-14 rounded-full border border-panel-border/60 flex items-center justify-center bg-background/60">
                    <BarChart2 className="size-6 text-muted-foreground/40" />
                  </div>
                  <div className="text-center">
                    <p className="font-mono text-[11px] text-muted-foreground/50 tracking-widest uppercase">
                      Plotly.js Canvas Injection Zone
                    </p>
                    <p className="font-mono text-[9px] text-muted-foreground/30 mt-1 tracking-wider">
                      Configure parameters · Execute 2D optimization
                    </p>
                  </div>
                </div>
              )}

              {/* Bottom-left coordinate readout */}
              <div className="absolute bottom-3 left-3 flex items-center gap-2 z-10">
                <Crosshair className="size-3 text-muted-foreground/40" />
                <span className="font-mono text-[9px] text-muted-foreground/40 tabular-nums">{lastPt}</span>
              </div>
            </div>

            {/* Bottom stats row */}
            <div className="flex gap-3 flex-shrink-0 mt-4">
              <StatCard icon={TrendingDown} label="‖θ‖ SGD" value={sgdFinal} accent="amber" />
              <StatCard icon={TrendingDown} label="‖θ‖ Adam" value={adamFinal} accent="cyan" />
              <StatCard icon={TrendingDown} label="‖θ‖ PSO best" value={psoFinal} accent="white" />
              <StatCard icon={GitBranch} label="Eval Calls" value={hasOptimized ? String(params.iterations * 3) : "—"} accent="muted" />
              <StatCard
                icon={BarChart2}
                label="Δ‖θ‖ (Adam−SGD)"
                value={sgdFinal !== "—" && adamFinal !== "—"
                  ? (parseFloat(adamFinal) - parseFloat(sgdFinal)).toFixed(4)
                  : "—"}
                accent="muted"
              />
            </div>

            {/* Convergence sparkline row */}
            {hasOptimized && (
              <div className="grid grid-cols-3 gap-3 flex-shrink-0 mt-4">
                <ConvergenceCard name="SGD" colorClass="bg-accent-amber" textColor="text-accent-amber" data={sgdConvergence} svgColor="#f59e0b" finalNorm={sgdFinal} onClick={() => setExpandedChart("SGD")} />
                <ConvergenceCard name="Adam" colorClass="bg-accent-cyan" textColor="text-accent-cyan" data={adamConvergence} svgColor="#22d3ee" finalNorm={adamFinal} onClick={() => setExpandedChart("Adam")} />
                <ConvergenceCard name="PSO" colorClass="bg-accent-white" textColor={theme === "light" ? "text-slate-800" : "text-foreground"} data={psoConvergence} svgColor={theme === "light" ? "#64748b" : "#e2e8f0"} finalNorm={psoFinal} onClick={() => setExpandedChart("PSO")} />
              </div>
            )}
          </TabsContent>

          {/* TAB 3: Results Comparison */}
          <TabsContent value="results" className="flex-1 min-h-0 outline-none flex flex-col m-0 data-[state=active]:flex rounded-md border border-panel-border bg-card p-6 overflow-y-auto">
            <div className="max-w-4xl mx-auto w-full space-y-8">
              
              <div>
                <h3 className="font-heading text-lg font-semibold text-foreground mb-4">Algorithm Performance Summary</h3>
                <div className="rounded-md border border-panel-border bg-background/50 overflow-hidden">
                  <table className="w-full text-sm text-left">
                    <thead className="bg-panel border-b border-panel-border text-[10px] uppercase font-mono text-muted-foreground tracking-wider">
                      <tr>
                        <th className="px-5 py-3.5 font-medium">Optimizer</th>
                        <th className="px-5 py-3.5 font-medium text-right">Execution Time (s)</th>
                        <th className="px-5 py-3.5 font-medium text-right">Final Train Loss</th>
                        <th className="px-5 py-3.5 font-medium text-right">Test Accuracy</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-panel-border/50 font-mono">
                      {resultsData.map((row) => (
                        <tr key={row.name} className="hover:bg-muted/30 transition-colors">
                          <td className="px-5 py-4">
                            <div className="flex items-center gap-2.5">
                              <span className={cn("size-2 rounded-full", row.color)} />
                              <span className={cn("font-bold tracking-wider", row.textCls)}>{row.name}</span>
                            </div>
                          </td>
                          <td className="px-5 py-4 text-right tabular-nums text-muted-foreground">{row.time}</td>
                          <td className="px-5 py-4 text-right tabular-nums text-foreground">{row.loss}</td>
                          <td className="px-5 py-4 text-right tabular-nums font-semibold text-accent-cyan">{row.acc}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-6">
                {/* Execution Time Chart */}
                <div className="p-6 rounded-md border border-panel-border bg-background/50 flex flex-col">
                  <h4 className="font-mono text-[10px] text-muted-foreground tracking-widest uppercase mb-5">Execution Time (Lower is Better)</h4>
                  <div className="space-y-5 flex-1 flex flex-col justify-center">
                    {resultsData.map(row => (
                      <div key={row.name} className="space-y-1.5">
                        <div className="flex justify-between text-xs font-mono">
                          <span className={row.textCls}>{row.name}</span>
                          <span className="text-muted-foreground tabular-nums">{row.time}s</span>
                        </div>
                        <div className="h-1.5 bg-panel-border rounded-full overflow-hidden">
                          <div 
                            className={cn("h-full transition-all duration-1000", row.color)} 
                            style={{ width: `${(parseFloat(row.time) / 1.05) * 100}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Final Loss Chart */}
                <div className="p-6 rounded-md border border-panel-border bg-background/50 flex flex-col">
                  <h4 className="font-mono text-[10px] text-muted-foreground tracking-widest uppercase mb-5">Final Train Loss (Lower is Better)</h4>
                  <div className="space-y-5 flex-1 flex flex-col justify-center">
                    {resultsData.map(row => (
                      <div key={row.name} className="space-y-1.5">
                        <div className="flex justify-between text-xs font-mono">
                          <span className={row.textCls}>{row.name}</span>
                          <span className="text-muted-foreground tabular-nums">{row.loss}</span>
                        </div>
                        <div className="h-1.5 bg-panel-border rounded-full overflow-hidden">
                          <div 
                            className={cn("h-full transition-all duration-1000", row.color)} 
                            style={{ width: `${(parseFloat(row.loss) / 0.45) * 100}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

            </div>
          </TabsContent>
        </Tabs>
      </div>

      {/* Interactive Chart Modal (for Optimization Landscape) */}
      {expandedChart && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-background/80 backdrop-blur-sm p-6">
          <div className="relative flex flex-col w-full max-w-4xl h-[600px] bg-card border border-panel-border rounded-lg shadow-2xl overflow-hidden animate-in zoom-in-95 duration-200">
            <div className="flex items-center justify-between px-6 py-4 border-b border-panel-border">
              <h3 className="font-heading text-lg font-semibold text-foreground">
                {expandedChart} Convergence History
              </h3>
              <button 
                onClick={() => setExpandedChart(null)}
                className="p-1 rounded-md hover:bg-muted/50 text-muted-foreground transition-colors"
              >
                <X className="size-5" />
              </button>
            </div>
            <div className="flex-1 p-2 relative">
              <Plot
                data={[{
                  type: "scatter",
                  mode: "lines",
                  y: expandedChart === "SGD" ? sgdConvergence 
                    : expandedChart === "Adam" ? adamConvergence 
                    : psoConvergence,
                  line: {
                    color: expandedChart === "SGD" ? "#f59e0b" 
                      : expandedChart === "Adam" ? "#22d3ee" 
                      : (theme === "light" ? "#64748b" : "#e2e8f0"),
                    width: 2
                  }
                }]}
                layout={{
                  autosize: true,
                  paper_bgcolor: "transparent",
                  plot_bgcolor: "transparent",
                  margin: { l: 50, r: 20, t: 20, b: 40 },
                  xaxis: {
                    title: { text: "Iterations", font: { color: theme === "light" ? "#64748b" : "#94a3b8" } },
                    gridcolor: theme === "light" ? "#e2e8f0" : "#334155",
                    tickfont: { color: theme === "light" ? "#64748b" : "#94a3b8" }
                  },
                  yaxis: {
                    title: { text: "Distance to Origin ‖θ‖", font: { color: theme === "light" ? "#64748b" : "#94a3b8" } },
                    gridcolor: theme === "light" ? "#e2e8f0" : "#334155",
                    tickfont: { color: theme === "light" ? "#64748b" : "#94a3b8" }
                  }
                }}
                config={{ displayModeBar: false, responsive: true }}
                style={{ width: "100%", height: "100%" }}
                useResizeHandler
              />
            </div>
          </div>
        </div>
      )}
    </main>
  )
}

// ─── Sub-components ────────────────────────────────────────────────────────────

function ConvergenceCard({
  name, colorClass, textColor, data, svgColor, finalNorm, onClick
}: {
  name: string
  colorClass: string
  textColor: string
  data: number[]
  svgColor: string
  finalNorm: string
  onClick?: () => void
}) {
  const delta = data.length > 1 ? (data[data.length - 1] - data[0]).toFixed(4) : "—"
  return (
    <div 
      onClick={onClick}
      className={cn(
        "flex items-center justify-between gap-4 px-3 py-2.5 rounded border border-panel-border/50 bg-card/60 transition-colors",
        onClick && "cursor-pointer hover:bg-muted/30"
      )}
    >
      <div className="flex items-center gap-2 min-w-[56px]">
        <span className={cn("size-1.5 rounded-full flex-shrink-0", colorClass)} />
        <span className={cn("font-mono text-[11px] font-bold tracking-widest", textColor)}>{name}</span>
      </div>
      <Sparkline data={data} color={svgColor} />
      <div className="text-right min-w-[64px]">
        <div className="font-mono text-xs text-foreground tabular-nums">{finalNorm}</div>
        <div className="font-mono text-[10px] text-muted-foreground tabular-nums">{delta}</div>
      </div>
    </div>
  )
}

function LegendItem({ color, textColor, label, subtitle }: { color: string; textColor: string; label: string; subtitle: string }) {
  return (
    <div className="flex items-center gap-2 px-2.5 py-1.5 rounded bg-background/70 border border-panel-border/50">
      <span className={cn("size-1.5 rounded-full flex-shrink-0", color)} />
      <div>
        <span className={cn("font-mono text-[10px] font-bold tracking-widest", textColor)}>{label}</span>
        <span className="font-mono text-[9px] text-muted-foreground/60 ml-1.5">{subtitle}</span>
      </div>
    </div>
  )
}

function StatCard({ icon: Icon, label, value, accent }: { icon: React.ElementType; label: string; value: string; accent: "amber" | "cyan" | "white" | "muted" }) {
  const accentCls = { amber: "text-accent-amber", cyan: "text-accent-cyan", white: "text-foreground", muted: "text-muted-foreground" }[accent]
  return (
    <div className="flex-1 px-4 py-3 rounded-md border border-panel-border bg-card flex flex-col gap-1.5 min-w-0">
      <div className="flex items-center gap-1.5">
        <Icon className={cn("size-3", accentCls)} />
        <span className="text-[9px] uppercase tracking-widest text-muted-foreground font-medium truncate">{label}</span>
      </div>
      <span className={cn("font-mono text-sm font-semibold tabular-nums", accentCls)}>{value}</span>
    </div>
  )
}

function TopbarChip({ icon: Icon, label }: { icon: React.ElementType; label: string }) {
  return (
    <div className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-panel border border-panel-border">
      <Icon className="size-3 text-muted-foreground" />
      <span className="font-mono text-[10px] text-muted-foreground tracking-wider">{label}</span>
    </div>
  )
}

function IconBtn({ icon: Icon }: { icon: React.ElementType }) {
  return (
    <button className="size-7 rounded border border-panel-border bg-panel flex items-center justify-center hover:bg-muted/50 transition-colors">
      <Icon className="size-3 text-muted-foreground" />
    </button>
  )
}
