"use client"

import * as React from "react"
import { Moon, Sun } from "lucide-react"
import { useTheme } from "next-themes"

export function ThemeToggle() {
  const { theme, setTheme } = useTheme()

  return (
    <button
      onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
      className="relative size-7 rounded border border-panel-border bg-panel flex items-center justify-center hover:bg-muted/50 transition-colors group"
      aria-label="Toggle theme"
    >
      <Sun className="h-3.5 w-3.5 text-muted-foreground transition-all scale-100 rotate-0 dark:-rotate-90 dark:scale-0" />
      <Moon className="absolute h-3.5 w-3.5 text-muted-foreground transition-all scale-0 rotate-90 dark:rotate-0 dark:scale-100" />
    </button>
  )
}
