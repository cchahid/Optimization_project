"use client"

import { useEffect, useMemo, useRef, useState } from "react"
import dynamic from "next/dynamic"
import { cn } from "@/lib/utils"
import type { PathPoint } from "@/app/page"
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
} from "lucide-react"

// Dynamically import Plotly to avoid SSR issues (it needs window)
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
  isLoading: boolean
  hasRun: boolean
  params: {
    learningRate: number
    iterations: number
    swarmSize: number
  }
  runCount: number
  sgdPath: PathPoint[]
  adamPath: PathPoint[]
  psoPaths: PathPoint[][]
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
export function VizCanvas({ isLoading, hasRun, params, runCount, sgdPath, adamPath, psoPaths }: VizCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null)
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

  // Progress bar simulation during loading
  useEffect(() => {
    if (isLoading) {
      setAnimProgress(0)
      let p = 0
      const tick = () => {
        p += Math.random() * 6
        setAnimProgress(Math.min(p, 90))
        if (p < 90) animRef.current = setTimeout(tick, 100)
      }
      tick()
    } else if (hasRun) {
      setAnimProgress(100)
    }
    return () => { if (animRef.current) clearTimeout(animRef.current) }
  }, [isLoading, hasRun])

  // Build Plotly traces
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
        color: "#ffffff",
        size: path.map((_, j) => (j === path.length - 1 ? 5 : 2.5)),
        opacity: path.map((_, j) => (j === path.length - 1 ? 0.9 : 0.3)),
      },
      hovertemplate: `PSO[${i}] [%{x:.3f}, %{y:.3f}]<extra></extra>`,
    }))

    return [contourTrace, sgdTrace, adamTrace, ...psoTraces]
  }, [sgdPath, adamPath, psoPaths])

  const layout = useMemo<Partial<Plotly.Layout>>(() => ({
    width: plotDims.width,
    height: plotDims.height,
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    margin: { l: 44, r: 24, t: 24, b: 44 },
    xaxis: {
      title: { text: "θ₁", font: { family: "var(--font-geist-mono, monospace)", size: 11, color: "#64748b" } },
      range: [-5.5, 5.5],
      gridcolor: "#334155",
      gridwidth: 1,
      zerolinecolor: "#475569",
      zerolinewidth: 1,
      tickfont: { family: "var(--font-geist-mono, monospace)", size: 9, color: "#64748b" },
      showgrid: true,
      tickformat: ".1f",
    },
    yaxis: {
      title: { text: "θ₂", font: { family: "var(--font-geist-mono, monospace)", size: 11, color: "#64748b" } },
      range: [-5.5, 5.5],
      gridcolor: "#334155",
      gridwidth: 1,
      zerolinecolor: "#475569",
      zerolinewidth: 1,
      tickfont: { family: "var(--font-geist-mono, monospace)", size: 9, color: "#64748b" },
      showgrid: true,
      tickformat: ".1f",
    },
    showlegend: false,
    hovermode: "closest",
    dragmode: "pan",
  }), [plotDims])

  const config: Partial<Plotly.Config> = {
    displayModeBar: false,
    responsive: false,
    scrollZoom: true,
  }

  // Derived final-step metrics
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
              LOSS SURFACE: RASTRIGIN (2D) &nbsp;·&nbsp; RUN #{String(runCount).padStart(4, "0")}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
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
        {/* Main viz card */}
        <div className="flex-1 relative rounded-md border border-panel-border bg-card overflow-hidden min-h-0" ref={containerRef}>

          {/* Floating legend — top right */}
          <div className="absolute top-4 right-4 z-10 flex flex-col gap-1.5 min-w-[148px]">
            <div className="text-[8px] font-mono text-muted-foreground/60 uppercase tracking-widest mb-0.5">
              Trace Legend
            </div>
            <LegendItem color="bg-accent-amber" textColor="text-accent-amber" label="SGD" subtitle="Stochastic Gradient" />
            <LegendItem color="bg-accent-cyan" textColor="text-accent-cyan" label="Adam" subtitle="Adaptive Moment" />
            <LegendItem color="bg-accent-white" textColor="text-foreground" label="PSO" subtitle="Particle Swarm" />
          </div>

          {/* Loading progress bar */}
          {isLoading && (
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
              layout={layout}
              config={config}
              style={{ position: "absolute", inset: 0 }}
            />
          )}

          {/* Idle overlay */}
          {!hasRun && !isLoading && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 pointer-events-none">
              <div className="size-14 rounded-full border border-panel-border/60 flex items-center justify-center bg-background/60">
                <BarChart2 className="size-6 text-muted-foreground/40" />
              </div>
              <div className="text-center">
                <p className="font-mono text-[11px] text-muted-foreground/50 tracking-widest uppercase">
                  Plotly.js Canvas Injection Zone
                </p>
                <p className="font-mono text-[9px] text-muted-foreground/30 mt-1 tracking-wider">
                  Configure parameters · Execute optimization
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
        <div className="flex gap-3 flex-shrink-0">
          <StatCard icon={TrendingDown} label="‖θ‖ SGD" value={sgdFinal} accent="amber" />
          <StatCard icon={TrendingDown} label="‖θ‖ Adam" value={adamFinal} accent="cyan" />
          <StatCard icon={TrendingDown} label="‖θ‖ PSO best" value={psoFinal} accent="white" />
          <StatCard icon={GitBranch} label="Eval Calls" value={hasRun ? String(params.iterations * 3) : "—"} accent="muted" />
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
        {hasRun && (
          <div className="grid grid-cols-3 gap-3 flex-shrink-0">
            <ConvergenceCard name="SGD" colorClass="bg-accent-amber" textColor="text-accent-amber" data={sgdConvergence} svgColor="#f59e0b" finalNorm={sgdFinal} />
            <ConvergenceCard name="Adam" colorClass="bg-accent-cyan" textColor="text-accent-cyan" data={adamConvergence} svgColor="#22d3ee" finalNorm={adamFinal} />
            <ConvergenceCard name="PSO" colorClass="bg-accent-white" textColor="text-foreground" data={psoConvergence} svgColor="#e2e8f0" finalNorm={psoFinal} />
          </div>
        )}
      </div>
    </main>
  )
}

// ─── Sub-components ────────────────────────────────────────────────────────────

function ConvergenceCard({
  name, colorClass, textColor, data, svgColor, finalNorm,
}: {
  name: string
  colorClass: string
  textColor: string
  data: number[]
  svgColor: string
  finalNorm: string
}) {
  const delta = data.length > 1 ? (data[data.length - 1] - data[0]).toFixed(4) : "—"
  return (
    <div className="flex items-center justify-between gap-4 px-3 py-2.5 rounded border border-panel-border/50 bg-card/60">
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
