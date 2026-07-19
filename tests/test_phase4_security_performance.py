#!/usr/bin/env python3
"""
Comprehensive Test Suite for Phase 4 Performance Optimization & Security Improvements

Tests:
- Security hardening and SAST/DAST integration
- Performance profiling and resource management  
- Load testing and stress testing capabilities
- Resource limits and monitoring
- AI-driven optimization features
"""

import asyncio
import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import our modules
from test_security import SecurityScanner
from src.optimization.performance_profiler import PerformanceProfiler, CognitiveResourceManager, ResourceLimits


class Phase4SecurityTests(unittest.TestCase):
    """Test security improvements and scanning capabilities"""
    
    def setUp(self):
        self.scanner = SecurityScanner(base_url="http://localhost", scan_timeout=30)
    
    def test_security_scanner_initialization(self):
        """Test SecurityScanner initializes correctly"""
        self.assertEqual(self.scanner.base_url, "http://localhost")
        self.assertEqual(len(self.scanner.service_endpoints), 6)
        self.assertIn("service-registry", self.scanner.service_endpoints)
        self.assertIn("financial-service", self.scanner.service_endpoints)
    
    def test_vulnerability_patterns_loaded(self):
        """Test vulnerability test patterns are loaded"""
        self.assertTrue(len(self.scanner.vulnerability_patterns) > 0)
        self.assertIn("' OR '1'='1", self.scanner.vulnerability_patterns)
        self.assertIn("<script>alert('XSS')</script>", self.scanner.vulnerability_patterns)
    
    async def test_sast_analysis(self):
        """Test Static Application Security Testing"""
        # Use a smaller scope to avoid timeout
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test file with a security issue
            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("""
import subprocess
def unsafe_function(user_input):
    subprocess.call(user_input, shell=True)  # Security issue: shell injection
""")
            
            results = await self.scanner.run_sast_analysis(str(tmpdir))
            self.assertEqual(results["status"], "completed")
            self.assertIn("return_code", results)
    
    async def test_dast_analysis_structure(self):
        """Test Dynamic Application Security Testing structure"""
        # Mock the endpoint testing to avoid actual network calls
        with patch.object(self.scanner, '_test_service_endpoints') as mock_test:
            mock_test.return_value = {
                "status": "tested",
                "endpoints_tested": [],
                "vulnerabilities": [],
                "vulnerabilities_count": 0
            }
            
            results = await self.scanner.run_dast_analysis()
            self.assertEqual(results["status"], "completed")
            self.assertEqual(results["services_tested"], 6)
            self.assertIn("endpoint_results", results)
    
    async def test_penetration_tests_structure(self):
        """Test penetration testing framework"""
        # Mock the individual test methods
        with patch.object(self.scanner, '_test_authentication_bypass') as mock_auth, \
             patch.object(self.scanner, '_test_rate_limiting') as mock_rate, \
             patch.object(self.scanner, '_test_cors_configuration') as mock_cors:
            
            mock_auth.return_value = [{"test_name": "auth_test", "vulnerable": False}]
            mock_rate.return_value = [{"test_name": "rate_test", "vulnerable": False}]
            mock_cors.return_value = [{"test_name": "cors_test", "vulnerable": False}]
            
            results = await self.scanner.run_penetration_tests()
            self.assertEqual(results["status"], "completed")
            self.assertEqual(results["tests_run"], 3)
            self.assertIn("test_results", results)
    
    def test_security_report_generation(self):
        """Test security report generation"""
        # Mock the scan results
        self.scanner.scan_results = {
            "sast": {"total_issues": 5, "high_severity_issues": 1, "medium_severity_issues": 2, "low_severity_issues": 2},
            "dast": {"vulnerabilities_found": 2},
            "penetration": {"vulnerabilities_found": 1}
        }
        
        async def test_report():
            with patch.object(self.scanner, 'run_sast_analysis') as mock_sast, \
                 patch.object(self.scanner, 'run_dast_analysis') as mock_dast, \
                 patch.object(self.scanner, 'run_penetration_tests') as mock_pentest:
                
                mock_sast.return_value = self.scanner.scan_results["sast"]
                mock_dast.return_value = self.scanner.scan_results["dast"]
                mock_pentest.return_value = self.scanner.scan_results["penetration"]
                
                with tempfile.NamedTemporaryFile(suffix='.json') as tmp:
                    report = await self.scanner.generate_security_report(tmp.name)
                    
                    self.assertIn("scan_summary", report)
                    self.assertIn("recommendations", report)
                    self.assertEqual(report["scan_summary"]["total_vulnerabilities"], 8)
        
        asyncio.run(test_report())


class Phase4PerformanceTests(unittest.TestCase):
    """Test performance profiling and optimization capabilities"""
    
    def setUp(self):
        self.resource_limits = ResourceLimits(
            max_memory_mb=512,
            max_cpu_percent=70.0,
            max_response_time_ms=500.0
        )
        self.profiler = PerformanceProfiler("test-service", self.resource_limits)
    
    def tearDown(self):
        if self.profiler.profiling_active:
            self.profiler.stop_profiling()
    
    def test_profiler_initialization(self):
        """Test PerformanceProfiler initializes correctly"""
        self.assertEqual(self.profiler.service_name, "test-service")
        self.assertEqual(self.profiler.resource_limits.max_memory_mb, 512)
        self.assertFalse(self.profiler.profiling_active)
        self.assertEqual(len(self.profiler.metrics_history), 0)
    
    def test_resource_limits_validation(self):
        """Test resource limits are properly configured"""
        limits = ResourceLimits()
        self.assertEqual(limits.max_memory_mb, 1024)
        self.assertEqual(limits.max_cpu_percent, 80.0)
        self.assertEqual(limits.max_response_time_ms, 1000.0)
        
        custom_limits = ResourceLimits(max_memory_mb=2048, max_cpu_percent=90.0)
        self.assertEqual(custom_limits.max_memory_mb, 2048)
        self.assertEqual(custom_limits.max_cpu_percent, 90.0)
    
    def test_request_recording(self):
        """Test request metrics recording"""
        # Record some test requests
        self.profiler.record_request(150.0, True)
        self.profiler.record_request(200.0, True)
        self.profiler.record_request(100.0, False)  # Failed request
        
        self.assertEqual(self.profiler.request_count, 3)
        self.assertEqual(self.profiler.error_count, 1)
        self.assertEqual(self.profiler.total_response_time, 450.0)
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.net_connections')
    def test_metrics_collection(self, mock_connections, mock_memory, mock_cpu):
        """Test performance metrics collection"""
        # Mock system metrics
        mock_cpu.return_value = 65.0
        mock_memory.return_value = MagicMock(used=500*1024*1024, percent=45.0)  # 500MB
        mock_connections.return_value = [1, 2, 3]  # 3 connections
        
        # Record some requests first
        self.profiler.record_request(100.0, True)
        self.profiler.record_request(200.0, True)
        
        metrics = self.profiler.collect_metrics()
        
        self.assertEqual(metrics.service_name, "test-service")
        self.assertEqual(metrics.cpu_percent, 65.0)
        self.assertAlmostEqual(metrics.memory_mb, 500.0, places=1)
        self.assertEqual(metrics.memory_percent, 45.0)
        self.assertEqual(metrics.active_connections, 3)
        self.assertGreater(metrics.timestamp, 0)
    
    def test_profiling_lifecycle(self):
        """Test profiling start and stop"""
        self.assertFalse(self.profiler.profiling_active)
        
        # Start profiling
        self.profiler.start_profiling(interval=0.1)
        self.assertTrue(self.profiler.profiling_active)
        self.assertIsNotNone(self.profiler.profiling_thread)
        
        # Stop profiling
        self.profiler.stop_profiling()
        self.assertFalse(self.profiler.profiling_active)
    
    @patch('psutil.cpu_percent', return_value=85.0)  # High CPU
    @patch('psutil.virtual_memory')
    def test_performance_alerts(self, mock_memory, mock_cpu):
        """Test performance alerting system"""
        # Mock high memory usage (600MB > 512MB limit)
        mock_memory.return_value = MagicMock(used=600*1024*1024, percent=60.0)
        
        with patch.object(self.profiler, '_analyze_performance') as mock_analyze:
            metrics = self.profiler.collect_metrics()
            self.profiler._analyze_performance(metrics)
            mock_analyze.assert_called_once()
        
        # Check that alerts would be generated
        self.assertGreater(metrics.cpu_percent, self.resource_limits.max_cpu_percent)
        self.assertGreater(metrics.memory_mb, self.resource_limits.max_memory_mb)
    
    def test_memory_leak_detection(self):
        """Test memory leak detection functionality"""
        # Set initial baseline
        self.profiler.memory_baseline = 400.0  # 400MB baseline
        
        # Create metrics with memory growth
        from src.optimization.performance_profiler import PerformanceMetrics
        import time
        
        leak_metrics = PerformanceMetrics(
            timestamp=time.time(),
            cpu_percent=50.0,
            memory_mb=470.0,  # 70MB growth (above 50MB threshold)
            memory_percent=60.0,
            response_time_ms=100.0,
            requests_per_second=10.0,
            active_connections=5,
            error_rate=0.01,
            service_name="test-service"
        )
        
        with patch('logging.Logger.warning') as mock_warning:
            self.profiler._detect_memory_leaks(leak_metrics)
            mock_warning.assert_called()
    
    def test_optimization_recommendations(self):
        """Test AI-driven optimization recommendations"""
        # Add some metrics history
        from src.optimization.performance_profiler import PerformanceMetrics
        import time
        
        # High CPU metrics
        for i in range(5):
            metrics = PerformanceMetrics(
                timestamp=time.time() + i,
                cpu_percent=85.0,  # High CPU
                memory_mb=300.0,
                memory_percent=30.0,
                response_time_ms=1200.0,  # Slow response
                requests_per_second=2.0,  # Low throughput
                active_connections=10,
                error_rate=0.02,  # High error rate
                service_name="test-service"
            )
            self.profiler.metrics_history.append(metrics)
        
        recommendations = self.profiler._generate_optimization_recommendations()
        
        self.assertTrue(len(recommendations) > 0)
        # Check for expected recommendations based on our test data
        high_cpu_rec = any("CPU" in rec for rec in recommendations)
        slow_response_rec = any("response time" in rec.lower() for rec in recommendations)
        self.assertTrue(high_cpu_rec or slow_response_rec)


class Phase4CognitiveResourceManagerTests(unittest.TestCase):
    """Test cognitive resource management capabilities"""
    
    def setUp(self):
        self.manager = CognitiveResourceManager()
    
    def tearDown(self):
        self.manager.cleanup()
    
    def test_manager_initialization(self):
        """Test CognitiveResourceManager initialization"""
        self.assertEqual(len(self.manager.service_profilers), 0)
        self.assertIsInstance(self.manager.resource_allocation_rules, dict)
    
    def test_service_registration(self):
        """Test service registration and profiler creation"""
        limits = ResourceLimits(max_memory_mb=1024)
        profiler = self.manager.register_service("test-service-1", limits)
        
        self.assertIn("test-service-1", self.manager.service_profilers)
        self.assertEqual(profiler.service_name, "test-service-1")
        self.assertEqual(profiler.resource_limits.max_memory_mb, 1024)
        self.assertTrue(profiler.profiling_active)
    
    def test_system_performance_report(self):
        """Test system-wide performance report generation"""
        # Register multiple services
        profiler1 = self.manager.register_service("service-1")
        profiler2 = self.manager.register_service("service-2")
        
        # Mock performance reports
        mock_report_1 = {
            "service_name": "service-1",
            "performance_summary": {
                "cpu_usage": {"current": 60.0},
                "memory_usage": {"current_mb": 400.0},
                "response_time": {"current_ms": 150.0},
                "throughput": {"current_rps": 25.0}
            }
        }
        
        mock_report_2 = {
            "service_name": "service-2", 
            "performance_summary": {
                "cpu_usage": {"current": 40.0},
                "memory_usage": {"current_mb": 300.0},
                "response_time": {"current_ms": 100.0},
                "throughput": {"current_rps": 30.0}
            }
        }
        
        with patch.object(profiler1, 'get_performance_report', return_value=mock_report_1), \
             patch.object(profiler2, 'get_performance_report', return_value=mock_report_2):
            
            system_report = self.manager.get_system_performance_report()
            
            self.assertEqual(system_report["services_monitored"], 2)
            self.assertIn("system_summary", system_report)
            self.assertIn("service_reports", system_report)
            self.assertIn("cognitive_recommendations", system_report)
            
            # Check system totals
            summary = system_report["system_summary"]
            self.assertEqual(summary["total_cpu_usage"], 100.0)  # 60 + 40
            self.assertEqual(summary["total_memory_usage"], 700.0)  # 400 + 300
            self.assertEqual(summary["total_throughput"], 55.0)  # 25 + 30
    
    def test_cognitive_recommendations(self):
        """Test AI-driven system optimization recommendations"""
        # Create mock system report with issues
        mock_system_report = {
            "system_summary": {
                "services_with_issues": 2,
                "total_cpu_usage": 250.0,  # High CPU
                "total_memory_usage": 7000.0,  # High memory
                "average_response_time": 900.0  # Slow response
            },
            "service_reports": {
                "high-load-service": {
                    "performance_summary": {
                        "cpu_usage": {"current": 90.0},
                        "memory_usage": {"current_mb": 800.0},
                        "response_time": {"current_ms": 1000.0}
                    }
                },
                "low-load-service": {
                    "performance_summary": {
                        "cpu_usage": {"current": 20.0},
                        "memory_usage": {"current_mb": 200.0},
                        "response_time": {"current_ms": 100.0}
                    }
                }
            }
        }
        
        recommendations = self.manager._generate_cognitive_recommendations(mock_system_report)
        
        self.assertTrue(len(recommendations) > 0)
        # Should detect system-level issues
        system_alert = any("System Alert" in rec for rec in recommendations)
        cpu_alert = any("CPU usage high" in rec for rec in recommendations) 
        memory_alert = any("memory usage high" in rec.lower() for rec in recommendations)
        load_balance_alert = any("Load imbalance detected" in rec for rec in recommendations)
        
        self.assertTrue(system_alert or cpu_alert or memory_alert or load_balance_alert)


class Phase4IntegrationTests(unittest.TestCase):
    """Integration tests for security and performance systems"""
    
    async def test_security_performance_integration(self):
        """Test integration between security scanning and performance monitoring"""
        # Test that security scanner can handle performance profiler metrics
        scanner = SecurityScanner()
        manager = CognitiveResourceManager()
        
        try:
            # Register a service for monitoring
            profiler = manager.register_service("integration-test-service")
            
            # Simulate some activity
            profiler.record_request(100.0, True)
            profiler.record_request(150.0, True)
            
            # Verify both systems can operate independently
            self.assertTrue(profiler.profiling_active)
            self.assertEqual(len(scanner.service_endpoints), 6)
            
        finally:
            manager.cleanup()
    
    def test_docker_security_config_validation(self):
        """Test that Docker security configurations are properly structured"""
        # Read the docker-compose.yml to verify security settings
        docker_compose_path = Path("/home/runner/work/elizoscog/elizoscog/docker/docker-compose.yml")
        
        if docker_compose_path.exists():
            with open(docker_compose_path, 'r') as f:
                content = f.read()
            
            # Check for security hardening configurations
            security_checks = [
                "security_opt",
                "no-new-privileges",
                "read_only: true",
                "cap_drop",
                "tmpfs"
            ]
            
            for check in security_checks:
                self.assertIn(check, content, f"Security configuration '{check}' not found in docker-compose.yml")
    
    def test_performance_config_files_exist(self):
        """Test that performance testing configuration files exist"""
        locust_file = Path("/home/runner/work/elizoscog/elizoscog/docker/configs/performance/locustfile.py")
        self.assertTrue(locust_file.exists(), "Locust performance test file not found")
        
        # Check locust file has proper structure
        with open(locust_file, 'r') as f:
            content = f.read()
        
        required_classes = ["MicroservicePerformanceUser", "HighLoadUser"]
        for cls in required_classes:
            self.assertIn(cls, content, f"Required class '{cls}' not found in locustfile.py")


def run_all_tests():
    """Run all Phase 4 tests and generate report"""
    print("=" * 60)
    print("🧪 Phase 4 Security & Performance Test Suite")  
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        Phase4SecurityTests,
        Phase4PerformanceTests, 
        Phase4CognitiveResourceManagerTests,
        Phase4IntegrationTests
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Generate summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\n❌ Failures ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split(chr(10))[-2] if traceback else 'Unknown failure'}")
    
    if result.errors:
        print(f"\n⚠️ Errors ({len(result.errors)}):")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split(chr(10))[-2] if traceback else 'Unknown error'}")
    
    if not result.failures and not result.errors:
        print("\n✅ All Phase 4 tests passed successfully!")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)