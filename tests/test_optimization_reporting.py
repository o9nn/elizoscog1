#!/usr/bin/env python3
"""
Test Suite for Optimization Trajectory Visualization & Evolutionary Reports
Phase 5 Implementation: Validates visualization and report generation components
"""

import asyncio
import unittest
import sys
import os
import json
import tempfile
import logging

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.optimization.adaptive_optimization import (
    AdaptiveParameter, FitnessLandscapeMapper, AdaptiveOptimizationEngine
)
from src.optimization.optimization_reporting import (
    OptimizationTrajectoryVisualizer, EvolutionaryOptimizationReporter
)

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _make_populated_mapper():
    """Create a fitness landscape mapper with sample evaluation points"""
    mapper = FitnessLandscapeMapper(["param1", "param2"])
    for i in range(20):
        mapper.add_evaluation_point(
            {"param1": 0.1 + i * 0.04, "param2": 0.9 - i * 0.03},
            fitness=0.5 + i * 0.02,
            generation=i
        )
    return mapper


class TestOptimizationTrajectoryVisualizer(unittest.TestCase):
    """Test the OptimizationTrajectoryVisualizer class"""

    def setUp(self):
        self.mapper = _make_populated_mapper()
        self.visualizer = OptimizationTrajectoryVisualizer(self.mapper)

    def test_ascii_trajectory_generation(self):
        """Test ASCII trajectory chart generation"""
        chart = self.visualizer.generate_ascii_trajectory()

        self.assertIn("Optimization Trajectory", chart)
        self.assertIn("█", chart)
        self.assertIn("points: 20", chart)
        self.assertIn("improvement:", chart)

    def test_ascii_trajectory_empty_mapper(self):
        """Test ASCII chart with no trajectory data"""
        empty_mapper = FitnessLandscapeMapper(["param1"])
        visualizer = OptimizationTrajectoryVisualizer(empty_mapper)

        chart = visualizer.generate_ascii_trajectory()
        self.assertEqual(chart, "No trajectory data available")

    def test_html_visualization_generation(self):
        """Test HTML visualization generation"""
        html = self.visualizer.generate_html_visualization()

        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("trajectory-chart", html)
        self.assertIn("fitness_peaks", html)
        self.assertIn("Optimization Trajectory", html)

    def test_save_visualization(self):
        """Test saving HTML visualization to file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "trajectory.html")
            result = self.visualizer.save_visualization(filename)

            self.assertTrue(result)
            self.assertTrue(os.path.exists(filename))
            with open(filename) as f:
                self.assertIn("<!DOCTYPE html>", f.read())


class TestEvolutionaryOptimizationReporter(unittest.TestCase):
    """Test the EvolutionaryOptimizationReporter class"""

    def setUp(self):
        parameters = [
            AdaptiveParameter("param1", 0.5, 0.0, 1.0),
            AdaptiveParameter("param2", 0.3, 0.0, 1.0)
        ]
        self.engine = AdaptiveOptimizationEngine(parameters)

        # Populate the fitness mapper with trajectory data
        for i in range(15):
            self.engine.fitness_mapper.add_evaluation_point(
                {"param1": 0.5 + i * 0.02, "param2": 0.3 + i * 0.01},
                fitness=0.6 + i * 0.02,
                generation=i
            )

        self.reporter = EvolutionaryOptimizationReporter(self.engine)

    def test_report_generation(self):
        """Test comprehensive report generation"""
        report = self.reporter.generate_report()

        self.assertEqual(report["report_type"], "evolutionary_optimization_report")
        self.assertIn("engine_status", report)
        self.assertIn("optimization_trajectory", report)
        self.assertIn("fitness_peaks", report)
        self.assertIn("improvement_validation", report)
        self.assertEqual(len(report["parameters"]), 2)
        self.assertEqual(report["parameters"][0]["name"], "param1")

    def test_report_trajectory_statistics(self):
        """Test that trajectory statistics are included in the report"""
        report = self.reporter.generate_report()
        stats = report["optimization_trajectory"]["statistics"]

        self.assertEqual(stats["total_points"], 15)
        self.assertGreater(stats["fitness_improvement"], 0)

    def test_markdown_report_generation(self):
        """Test Markdown report generation"""
        markdown = self.reporter.generate_markdown_report()

        self.assertIn("# 📈 Evolutionary Optimization Report", markdown)
        self.assertIn("## Engine Status", markdown)
        self.assertIn("## Optimization Trajectory", markdown)
        self.assertIn("## Adaptive Parameters", markdown)
        self.assertIn("| param1 |", markdown)
        self.assertIn("## Improvement Validation", markdown)

    def test_save_json_report(self):
        """Test saving report as JSON"""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "report.json")
            result = self.reporter.save_report(filename, format="json")

            self.assertTrue(result)
            with open(filename) as f:
                data = json.load(f)
            self.assertEqual(data["report_type"], "evolutionary_optimization_report")

    def test_save_markdown_report(self):
        """Test saving report as Markdown"""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "report.md")
            result = self.reporter.save_report(filename, format="markdown")

            self.assertTrue(result)
            with open(filename) as f:
                self.assertIn("# 📈 Evolutionary Optimization Report", f.read())

    def test_save_report_invalid_format(self):
        """Test that an unsupported format raises ValueError"""
        with self.assertRaises(ValueError):
            self.reporter.save_report("report.txt", format="txt")

    def test_report_with_insufficient_validation_data(self):
        """Test report generation with insufficient adaptation history"""
        report = self.reporter.generate_report()
        validation = report["improvement_validation"]

        self.assertEqual(validation["status"], "insufficient_data")


class TestPackageExports(unittest.TestCase):
    """Test that adaptive optimization components are exported from the package"""

    def test_package_level_imports(self):
        """Test importing adaptive optimization components from src.optimization"""
        from src.optimization import (
            AdaptiveParameter, AdaptiveOptimizationEngine,
            OptimizationTrajectoryVisualizer, EvolutionaryOptimizationReporter,
            create_adaptive_optimization_engine
        )

        self.assertIsNotNone(AdaptiveParameter)
        self.assertIsNotNone(AdaptiveOptimizationEngine)
        self.assertIsNotNone(OptimizationTrajectoryVisualizer)
        self.assertIsNotNone(EvolutionaryOptimizationReporter)
        self.assertIsNotNone(create_adaptive_optimization_engine)


if __name__ == "__main__":
    unittest.main()
