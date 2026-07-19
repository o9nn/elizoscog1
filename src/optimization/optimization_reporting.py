#!/usr/bin/env python3
"""
Optimization Trajectory Visualization & Evolutionary Optimization Reports
Phase 5 Implementation: Adaptive Optimization & Continuous Learning

Provides visualization of optimization trajectories through the fitness
landscape and generation of evolutionary optimization reports in JSON,
Markdown, and self-contained HTML formats.
"""

import json
import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from .adaptive_optimization import (
    FitnessLandscapeMapper,
    AdaptiveOptimizationEngine,
)

logger = logging.getLogger(__name__)


class OptimizationTrajectoryVisualizer:
    """Visualizes optimization trajectories through the fitness landscape"""

    def __init__(self, fitness_mapper: FitnessLandscapeMapper):
        self.fitness_mapper = fitness_mapper

    def generate_ascii_trajectory(self, width: int = 60, height: int = 12) -> str:
        """Generate an ASCII chart of fitness over the optimization trajectory"""
        trajectory_data = self.fitness_mapper.get_optimization_trajectory()
        trajectory = trajectory_data.get("trajectory", [])

        if not trajectory:
            return "No trajectory data available"

        fitness_values = [p["fitness"] for p in trajectory]

        # Resample to fit the requested width
        if len(fitness_values) > width:
            step = len(fitness_values) / width
            fitness_values = [
                fitness_values[int(i * step)] for i in range(width)
            ]

        min_fit = min(fitness_values)
        max_fit = max(fitness_values)
        fit_range = max_fit - min_fit

        # Build the chart grid
        grid = [[" " for _ in range(len(fitness_values))] for _ in range(height)]
        for col, fitness in enumerate(fitness_values):
            if fit_range > 0:
                row = int((fitness - min_fit) / fit_range * (height - 1))
            else:
                row = height // 2
            grid[height - 1 - row][col] = "█"

        lines = ["Optimization Trajectory (fitness over time)"]
        lines.append(f"max: {max_fit:.4f}")
        for row in grid:
            lines.append("│" + "".join(row))
        lines.append("└" + "─" * len(fitness_values))
        lines.append(f"min: {min_fit:.4f}   points: {len(trajectory)}")

        statistics = trajectory_data.get("statistics", {})
        if statistics:
            lines.append(
                f"improvement: {statistics.get('fitness_improvement', 0.0):+.4f}   "
                f"trend: {statistics.get('convergence_trend', 'unknown')}"
            )

        return "\n".join(lines)

    def generate_html_visualization(self, title: str = "Optimization Trajectory") -> str:
        """Generate a self-contained HTML visualization of the trajectory"""
        trajectory_data = self.fitness_mapper.get_optimization_trajectory()
        landscape_summary = self.fitness_mapper.get_landscape_summary()
        peaks = self.fitness_mapper.identify_fitness_peaks()

        payload = json.dumps({
            "trajectory": trajectory_data.get("trajectory", []),
            "statistics": trajectory_data.get("statistics", {}),
            "landscape_summary": landscape_summary,
            "fitness_peaks": peaks[:10],
        })

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #1a1a2e; color: #eee; margin: 20px; }}
  h1 {{ color: #4fc3f7; }}
  .panel {{ background: #16213e; border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
  canvas {{ background: #0f1626; border-radius: 4px; width: 100%; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #333; padding: 6px 10px; text-align: left; }}
  th {{ background: #0f3460; }}
</style>
</head>
<body>
<h1>📈 {title}</h1>
<div class="panel">
  <h2>Fitness Trajectory</h2>
  <canvas id="trajectory-chart" width="900" height="300"></canvas>
</div>
<div class="panel">
  <h2>Statistics</h2>
  <div id="statistics"></div>
</div>
<div class="panel">
  <h2>Top Fitness Peaks</h2>
  <div id="peaks"></div>
</div>
<script>
const data = {payload};

function drawTrajectory() {{
  const canvas = document.getElementById('trajectory-chart');
  const ctx = canvas.getContext('2d');
  const points = data.trajectory.map(p => p.fitness);
  if (points.length === 0) return;

  const minF = Math.min(...points);
  const maxF = Math.max(...points);
  const range = (maxF - minF) || 1;
  const pad = 20;
  const w = canvas.width - 2 * pad;
  const h = canvas.height - 2 * pad;

  ctx.strokeStyle = '#4fc3f7';
  ctx.lineWidth = 2;
  ctx.beginPath();
  points.forEach((f, i) => {{
    const x = pad + (i / Math.max(points.length - 1, 1)) * w;
    const y = pad + (1 - (f - minF) / range) * h;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }});
  ctx.stroke();
}}

function renderStatistics() {{
  const s = data.statistics || {{}};
  const rows = Object.entries(s).map(([k, v]) =>
    `<tr><td>${{k}}</td><td>${{JSON.stringify(v)}}</td></tr>`).join('');
  document.getElementById('statistics').innerHTML =
    `<table><tr><th>Metric</th><th>Value</th></tr>${{rows}}</table>`;
}}

function renderPeaks() {{
  const peaks = data.fitness_peaks || [];
  const rows = peaks.map(p =>
    `<tr><td>${{p.fitness.toFixed(4)}}</td><td>${{JSON.stringify(p.parameters)}}</td><td>${{p.generation}}</td></tr>`).join('');
  document.getElementById('peaks').innerHTML =
    `<table><tr><th>Fitness</th><th>Parameters</th><th>Generation</th></tr>${{rows}}</table>`;
}}

drawTrajectory();
renderStatistics();
renderPeaks();
</script>
</body>
</html>"""

    def save_visualization(self, filename: str, title: str = "Optimization Trajectory") -> bool:
        """Save the HTML visualization to a file"""
        try:
            with open(filename, "w") as f:
                f.write(self.generate_html_visualization(title))
            logger.info(f"📊 Saved optimization trajectory visualization to {filename}")
            return True
        except (OSError, IOError) as e:
            logger.error(f"Failed to save visualization: {e}")
            return False


class EvolutionaryOptimizationReporter:
    """Generates evolutionary optimization reports from an adaptive optimization engine"""

    def __init__(self, engine: AdaptiveOptimizationEngine):
        self.engine = engine

    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive evolutionary optimization report as a dictionary"""
        status = self.engine.get_comprehensive_status()
        trajectory = self.engine.fitness_mapper.get_optimization_trajectory()
        peaks = self.engine.fitness_mapper.identify_fitness_peaks()
        validation = self.engine.validate_adaptive_improvement()

        return {
            "report_type": "evolutionary_optimization_report",
            "generated_at": datetime.now().isoformat(),
            "timestamp": time.time(),
            "engine_status": status.get("engine_status", {}),
            "performance_monitoring": status.get("performance_monitoring", {}),
            "parameter_optimization": status.get("parameter_optimization", {}),
            "fitness_landscape": status.get("fitness_landscape", {}),
            "effectiveness_metrics": status.get("effectiveness_metrics", {}),
            "optimization_trajectory": trajectory,
            "fitness_peaks": peaks[:10],
            "improvement_validation": validation,
            "parameters": [
                {
                    "name": p.name,
                    "current_value": p.current_value,
                    "min_value": p.min_value,
                    "max_value": p.max_value,
                    "trend": p.get_trend(),
                    "fitness_contribution": p.fitness_contribution,
                }
                for p in self.engine.parameters
            ],
        }

    def generate_markdown_report(self) -> str:
        """Generate an evolutionary optimization report in Markdown format"""
        report = self.generate_report()
        engine_status = report["engine_status"]
        validation = report["improvement_validation"]
        trajectory_stats = report["optimization_trajectory"].get("statistics", {})

        lines = [
            "# 📈 Evolutionary Optimization Report",
            "",
            f"**Generated:** {report['generated_at']}",
            "",
            "## Engine Status",
            "",
            f"- **Strategy:** {engine_status.get('strategy', 'unknown')}",
            f"- **Adaptation Cycles:** {engine_status.get('adaptation_cycles', 0)}",
            f"- **Total Improvements:** {engine_status.get('total_improvements', 0)}",
            f"- **Running:** {engine_status.get('running', False)}",
            "",
            "## Optimization Trajectory",
            "",
        ]

        if trajectory_stats:
            lines.extend([
                f"- **Total Points:** {trajectory_stats.get('total_points', 0)}",
                f"- **Fitness Improvement:** {trajectory_stats.get('fitness_improvement', 0.0):+.4f}",
                f"- **Average Fitness:** {trajectory_stats.get('average_fitness', 0.0):.4f}",
                f"- **Convergence Trend:** {trajectory_stats.get('convergence_trend', 'unknown')}",
            ])
        else:
            lines.append("_No trajectory data available._")

        lines.extend(["", "## Adaptive Parameters", ""])
        lines.append("| Parameter | Current Value | Range | Trend |")
        lines.append("|-----------|---------------|-------|-------|")
        for p in report["parameters"]:
            lines.append(
                f"| {p['name']} | {p['current_value']:.4f} | "
                f"[{p['min_value']}, {p['max_value']}] | {p['trend']} |"
            )

        lines.extend(["", "## Fitness Peaks", ""])
        peaks = report["fitness_peaks"]
        if peaks:
            lines.append("| Fitness | Generation | Parameters |")
            lines.append("|---------|------------|------------|")
            for peak in peaks[:5]:
                lines.append(
                    f"| {peak['fitness']:.4f} | {peak['generation']} | "
                    f"`{json.dumps(peak['parameters'])}` |"
                )
        else:
            lines.append("_No fitness peaks identified yet._")

        lines.extend(["", "## Improvement Validation", ""])
        if validation.get("status") == "insufficient_data":
            lines.append(f"_{validation.get('message', 'Insufficient data')}_")
        else:
            summary = validation.get("validation_summary", {})
            lines.extend([
                f"- **Overall Success:** {summary.get('overall_success', False)}",
                f"- **Improvement:** {validation.get('improvement_percentage', 0.0):.2f}%",
                f"- **Stability Score:** {validation.get('stability_score', 0.0):.4f}",
                f"- **Convergence Achieved:** {validation.get('convergence_achieved', False)}",
            ])

        lines.append("")
        return "\n".join(lines)

    def save_report(self, filename: str, format: str = "json") -> bool:
        """Save the report to a file in JSON or Markdown format"""
        try:
            if format == "markdown":
                content = self.generate_markdown_report()
            elif format == "json":
                content = json.dumps(self.generate_report(), indent=2, default=str)
            else:
                raise ValueError(f"Unsupported report format: {format}")

            with open(filename, "w") as f:
                f.write(content)
            logger.info(f"📄 Saved evolutionary optimization report to {filename}")
            return True
        except (OSError, IOError) as e:
            logger.error(f"Failed to save report: {e}")
            return False
