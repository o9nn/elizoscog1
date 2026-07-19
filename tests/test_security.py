#!/usr/bin/env python3
"""
Security Testing Suite for ElizaOS-OpenCog-GnuCash Microservices

This module provides comprehensive security testing including:
- Static Application Security Testing (SAST)
- Dynamic Application Security Testing (DAST)  
- Penetration testing for microservice endpoints
- Vulnerability scanning and reporting
"""

import asyncio
import json
import subprocess
import logging
import requests
import time
from typing import Dict, List, Any
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SecurityScanner:
    """
    Comprehensive security scanner for microservice infrastructure
    """
    
    def __init__(self, base_url: str = "http://localhost", scan_timeout: int = 300):
        self.base_url = base_url
        self.scan_timeout = scan_timeout
        self.scan_results = {}
        
        # Service endpoints for testing
        self.service_endpoints = {
            "service-registry": 8001,
            "load-balancer": 8002,
            "orchestrator": 8003,
            "financial-service": 8010,
            "reasoning-service": 8011,
            "ggml-service": 8012
        }
        
        # Common vulnerability test patterns
        self.vulnerability_patterns = [
            # SQL Injection patterns
            "' OR '1'='1",
            "'; DROP TABLE users; --",
            "' UNION SELECT * FROM users --",
            
            # XSS patterns
            "<script>alert('XSS')</script>",
            "javascript:alert('XSS')",
            "<img src=x onerror=alert('XSS')>",
            
            # Path traversal patterns
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\drivers\\etc\\hosts",
            
            # Command injection patterns
            "; cat /etc/passwd",
            "| whoami",
            "&& id",
            
            # NoSQL injection patterns
            {"$ne": None},
            {"$gt": ""},
            {"$regex": ".*"}
        ]
    
    async def run_sast_analysis(self, source_path: str = "/home/runner/work/elizoscog/elizoscog") -> Dict[str, Any]:
        """
        Run Static Application Security Testing (SAST) using bandit
        """
        logger.info("Starting SAST analysis with bandit...")
        
        try:
            # Run bandit for Python security analysis
            cmd = [
                "python", "-m", "bandit", 
                "-r", source_path,
                "-f", "json",
                "-o", "/tmp/sast_report.json",
                "--skip", "B101,B601"  # Skip common false positives
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.scan_timeout)
            
            # Parse results
            sast_results = {
                "status": "completed",
                "return_code": result.returncode,
                "high_severity_issues": 0,
                "medium_severity_issues": 0,
                "low_severity_issues": 0,
                "total_issues": 0,
                "scan_time": time.time()
            }
            
            try:
                if Path("/tmp/sast_report.json").exists():
                    with open("/tmp/sast_report.json", "r") as f:
                        bandit_data = json.load(f)
                    
                    # Count issues by severity
                    for result_item in bandit_data.get("results", []):
                        severity = result_item.get("issue_severity", "").lower()
                        if severity == "high":
                            sast_results["high_severity_issues"] += 1
                        elif severity == "medium":
                            sast_results["medium_severity_issues"] += 1
                        elif severity == "low":
                            sast_results["low_severity_issues"] += 1
                    
                    sast_results["total_issues"] = len(bandit_data.get("results", []))
                    sast_results["details"] = bandit_data
                    
            except (FileNotFoundError, json.JSONDecodeError) as e:
                logger.warning(f"Could not parse SAST results: {e}")
                sast_results["error"] = str(e)
            
            self.scan_results["sast"] = sast_results
            logger.info(f"SAST analysis completed. Total issues: {sast_results['total_issues']}")
            
            return sast_results
            
        except subprocess.TimeoutExpired:
            logger.error("SAST analysis timed out")
            return {"status": "timeout", "error": "SAST scan timed out"}
        except Exception as e:
            logger.error(f"SAST analysis failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def run_dast_analysis(self) -> Dict[str, Any]:
        """
        Run Dynamic Application Security Testing (DAST) against live services
        """
        logger.info("Starting DAST analysis...")
        
        dast_results = {
            "status": "completed",
            "services_tested": 0,
            "vulnerabilities_found": 0,
            "endpoint_results": {},
            "scan_time": time.time()
        }
        
        for service_name, port in self.service_endpoints.items():
            service_url = f"{self.base_url}:{port}"
            logger.info(f"Testing service: {service_name} at {service_url}")
            
            service_results = await self._test_service_endpoints(service_name, service_url)
            dast_results["endpoint_results"][service_name] = service_results
            dast_results["services_tested"] += 1
            
            # Count vulnerabilities
            dast_results["vulnerabilities_found"] += service_results.get("vulnerabilities_count", 0)
        
        self.scan_results["dast"] = dast_results
        logger.info(f"DAST analysis completed. Services tested: {dast_results['services_tested']}")
        
        return dast_results
    
    async def _test_service_endpoints(self, service_name: str, base_url: str) -> Dict[str, Any]:
        """
        Test individual service endpoints for security vulnerabilities
        """
        service_results = {
            "status": "tested",
            "endpoints_tested": [],
            "vulnerabilities": [],
            "vulnerabilities_count": 0
        }
        
        # Common endpoints to test
        test_endpoints = [
            "/health",
            "/metrics",
            "/status",
            "/api/v1/",
            "/api/v1/services",
            "/admin",
            "/debug"
        ]
        
        for endpoint in test_endpoints:
            endpoint_url = f"{base_url}{endpoint}"
            endpoint_results = await self._test_endpoint_security(endpoint_url, endpoint)
            
            service_results["endpoints_tested"].append({
                "endpoint": endpoint,
                "url": endpoint_url,
                "results": endpoint_results
            })
            
            # Collect vulnerabilities
            if endpoint_results.get("vulnerabilities"):
                service_results["vulnerabilities"].extend(endpoint_results["vulnerabilities"])
                service_results["vulnerabilities_count"] += len(endpoint_results["vulnerabilities"])
        
        return service_results
    
    async def _test_endpoint_security(self, url: str, endpoint: str) -> Dict[str, Any]:
        """
        Test individual endpoint for common security vulnerabilities
        """
        results = {
            "accessible": False,
            "response_time": 0,
            "status_code": None,
            "security_headers": {},
            "vulnerabilities": []
        }
        
        try:
            start_time = time.time()
            
            # Basic connectivity test
            response = requests.get(url, timeout=10, allow_redirects=False)
            results["accessible"] = True
            results["response_time"] = (time.time() - start_time) * 1000
            results["status_code"] = response.status_code
            
            # Check security headers
            security_headers = {
                "Content-Security-Policy": response.headers.get("Content-Security-Policy"),
                "X-Frame-Options": response.headers.get("X-Frame-Options"),
                "X-Content-Type-Options": response.headers.get("X-Content-Type-Options"),
                "Strict-Transport-Security": response.headers.get("Strict-Transport-Security"),
                "X-XSS-Protection": response.headers.get("X-XSS-Protection")
            }
            results["security_headers"] = security_headers
            
            # Check for missing security headers
            missing_headers = []
            for header, value in security_headers.items():
                if not value:
                    missing_headers.append(header)
            
            if missing_headers:
                results["vulnerabilities"].append({
                    "type": "Missing Security Headers",
                    "severity": "medium",
                    "description": f"Missing headers: {', '.join(missing_headers)}",
                    "endpoint": endpoint
                })
            
            # Test for information disclosure
            if response.status_code == 200:
                content = response.text.lower()
                
                # Check for server information disclosure
                server_header = response.headers.get("Server", "").lower()
                if any(tech in server_header for tech in ["apache", "nginx", "python", "flask", "django"]):
                    results["vulnerabilities"].append({
                        "type": "Server Information Disclosure",
                        "severity": "low",
                        "description": f"Server header reveals technology: {server_header}",
                        "endpoint": endpoint
                    })
                
                # Check for debug information in response
                if any(debug in content for debug in ["debug", "traceback", "exception", "error"]):
                    results["vulnerabilities"].append({
                        "type": "Debug Information Disclosure",
                        "severity": "medium",
                        "description": "Response may contain debug information",
                        "endpoint": endpoint
                    })
            
            # Test injection vulnerabilities (if endpoint accepts parameters)
            if endpoint in ["/api/v1/", "/api/v1/services"]:
                await self._test_injection_vulnerabilities(url, endpoint, results)
                
        except requests.RequestException as e:
            results["error"] = str(e)
            logger.debug(f"Could not test endpoint {url}: {e}")
        
        return results
    
    async def _test_injection_vulnerabilities(self, url: str, endpoint: str, results: Dict[str, Any]):
        """
        Test for injection vulnerabilities using common payloads
        """
        for pattern in self.vulnerability_patterns[:5]:  # Test first 5 patterns to avoid overwhelming
            try:
                # Test as query parameter
                test_url = f"{url}?test={pattern}"
                response = requests.get(test_url, timeout=5)
                
                # Simple heuristic checks
                if response.status_code == 500:
                    results["vulnerabilities"].append({
                        "type": "Potential Injection Vulnerability",
                        "severity": "high",
                        "description": f"Server error (500) triggered by payload: {str(pattern)[:50]}",
                        "endpoint": endpoint
                    })
                
                # Check for SQL error messages
                error_indicators = ["sql", "mysql", "postgres", "oracle", "sqlite", "syntax error"]
                content = response.text.lower()
                for indicator in error_indicators:
                    if indicator in content:
                        results["vulnerabilities"].append({
                            "type": "SQL Error Information Disclosure",
                            "severity": "medium", 
                            "description": f"SQL error message detected in response",
                            "endpoint": endpoint
                        })
                        break
                        
            except requests.RequestException:
                continue  # Skip failed requests
    
    async def run_penetration_tests(self) -> Dict[str, Any]:
        """
        Run basic penetration testing against microservice endpoints
        """
        logger.info("Starting penetration testing...")
        
        pentest_results = {
            "status": "completed",
            "tests_run": 0,
            "vulnerabilities_found": 0,
            "test_results": [],
            "scan_time": time.time()
        }
        
        # Authentication bypass tests
        auth_tests = await self._test_authentication_bypass()
        pentest_results["test_results"].extend(auth_tests)
        pentest_results["tests_run"] += len(auth_tests)
        
        # Rate limiting tests
        rate_limit_tests = await self._test_rate_limiting()
        pentest_results["test_results"].extend(rate_limit_tests)
        pentest_results["tests_run"] += len(rate_limit_tests)
        
        # CORS configuration tests
        cors_tests = await self._test_cors_configuration()
        pentest_results["test_results"].extend(cors_tests)
        pentest_results["tests_run"] += len(cors_tests)
        
        # Count total vulnerabilities
        for test_result in pentest_results["test_results"]:
            if test_result.get("vulnerable", False):
                pentest_results["vulnerabilities_found"] += 1
        
        self.scan_results["penetration"] = pentest_results
        logger.info(f"Penetration testing completed. Tests run: {pentest_results['tests_run']}")
        
        return pentest_results
    
    async def _test_authentication_bypass(self) -> List[Dict[str, Any]]:
        """Test for authentication bypass vulnerabilities"""
        auth_tests = []
        
        for service_name, port in self.service_endpoints.items():
            test_result = {
                "test_name": "Authentication Bypass Test",
                "service": service_name,
                "vulnerable": False,
                "details": ""
            }
            
            try:
                # Test access to potentially protected endpoints
                protected_endpoints = ["/admin", "/api/v1/admin", "/config", "/debug"]
                
                for endpoint in protected_endpoints:
                    url = f"{self.base_url}:{port}{endpoint}"
                    response = requests.get(url, timeout=5)
                    
                    # If we get 200 instead of 401/403, it might be unprotected
                    if response.status_code == 200:
                        test_result["vulnerable"] = True
                        test_result["details"] = f"Potentially unprotected endpoint: {endpoint}"
                        break
                        
            except requests.RequestException:
                test_result["details"] = "Service unavailable for testing"
            
            auth_tests.append(test_result)
        
        return auth_tests
    
    async def _test_rate_limiting(self) -> List[Dict[str, Any]]:
        """Test for rate limiting implementation"""
        rate_limit_tests = []
        
        for service_name, port in self.service_endpoints.items():
            test_result = {
                "test_name": "Rate Limiting Test",
                "service": service_name,
                "vulnerable": False,
                "details": ""
            }
            
            try:
                # Send multiple rapid requests
                url = f"{self.base_url}:{port}/health"
                responses = []
                
                for i in range(10):
                    response = requests.get(url, timeout=2)
                    responses.append(response.status_code)
                    time.sleep(0.1)  # Brief pause between requests
                
                # Check if all requests succeeded (no rate limiting)
                if all(status == 200 for status in responses):
                    test_result["vulnerable"] = True
                    test_result["details"] = "No rate limiting detected - all 10 rapid requests succeeded"
                else:
                    test_result["details"] = f"Rate limiting may be present - response codes: {set(responses)}"
                    
            except requests.RequestException:
                test_result["details"] = "Service unavailable for testing"
            
            rate_limit_tests.append(test_result)
        
        return rate_limit_tests
    
    async def _test_cors_configuration(self) -> List[Dict[str, Any]]:
        """Test CORS configuration for security issues"""
        cors_tests = []
        
        for service_name, port in self.service_endpoints.items():
            test_result = {
                "test_name": "CORS Configuration Test",
                "service": service_name,
                "vulnerable": False,
                "details": ""
            }
            
            try:
                url = f"{self.base_url}:{port}/health"
                headers = {"Origin": "http://malicious-site.com"}
                response = requests.get(url, headers=headers, timeout=5)
                
                cors_header = response.headers.get("Access-Control-Allow-Origin", "")
                
                if cors_header == "*":
                    test_result["vulnerable"] = True
                    test_result["details"] = "CORS configured to allow all origins (*)"
                elif "malicious-site.com" in cors_header:
                    test_result["vulnerable"] = True
                    test_result["details"] = "CORS allows untrusted origin"
                else:
                    test_result["details"] = f"CORS header: {cors_header or 'Not set'}"
                    
            except requests.RequestException:
                test_result["details"] = "Service unavailable for testing"
            
            cors_tests.append(test_result)
        
        return cors_tests
    
    async def generate_security_report(self, output_path: str = "/tmp/security_report.json") -> Dict[str, Any]:
        """
        Generate comprehensive security report
        """
        logger.info("Generating comprehensive security report...")
        
        # Run all security tests
        sast_results = await self.run_sast_analysis()
        dast_results = await self.run_dast_analysis()
        pentest_results = await self.run_penetration_tests()
        
        # Generate comprehensive report
        security_report = {
            "scan_timestamp": time.time(),
            "scan_summary": {
                "total_vulnerabilities": 0,
                "high_severity": 0,
                "medium_severity": 0,
                "low_severity": 0,
                "services_tested": len(self.service_endpoints),
                "scan_duration": 0
            },
            "sast_results": sast_results,
            "dast_results": dast_results,
            "penetration_test_results": pentest_results,
            "recommendations": []
        }
        
        # Calculate totals
        security_report["scan_summary"]["total_vulnerabilities"] = (
            sast_results.get("total_issues", 0) + 
            dast_results.get("vulnerabilities_found", 0) + 
            pentest_results.get("vulnerabilities_found", 0)
        )
        
        security_report["scan_summary"]["high_severity"] = sast_results.get("high_severity_issues", 0)
        security_report["scan_summary"]["medium_severity"] = sast_results.get("medium_severity_issues", 0)
        security_report["scan_summary"]["low_severity"] = sast_results.get("low_severity_issues", 0)
        
        # Generate recommendations
        if security_report["scan_summary"]["total_vulnerabilities"] > 0:
            security_report["recommendations"].extend([
                "Review and fix high severity vulnerabilities immediately",
                "Implement proper input validation and sanitization",
                "Add security headers to all HTTP responses",
                "Configure rate limiting for all public endpoints",
                "Review CORS configuration for overly permissive settings",
                "Implement proper authentication and authorization",
                "Enable security monitoring and alerting"
            ])
        
        # Calculate pass rate
        total_checks = sast_results.get("total_issues", 0) + 100  # Base score of 100
        passed_checks = max(0, 100 - security_report["scan_summary"]["total_vulnerabilities"])
        pass_rate = (passed_checks / total_checks) * 100
        security_report["scan_summary"]["pass_rate"] = min(100, max(0, pass_rate))
        
        # Save report
        try:
            with open(output_path, "w") as f:
                json.dump(security_report, f, indent=2, default=str)
            logger.info(f"Security report saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save security report: {e}")
        
        self.scan_results["comprehensive_report"] = security_report
        
        logger.info(f"Security scan completed. Pass rate: {pass_rate:.1f}%")
        return security_report


async def main():
    """
    Main function to run comprehensive security testing
    """
    print("=" * 60)
    print("🔒 ElizaOS-OpenCog-GnuCash Security Testing Suite")
    print("=" * 60)
    
    scanner = SecurityScanner()
    
    try:
        # Run comprehensive security scan
        report = await scanner.generate_security_report()
        
        print("\n📊 Security Scan Summary:")
        print(f"   Total vulnerabilities found: {report['scan_summary']['total_vulnerabilities']}")
        print(f"   High severity: {report['scan_summary']['high_severity']}")
        print(f"   Medium severity: {report['scan_summary']['medium_severity']}")
        print(f"   Low severity: {report['scan_summary']['low_severity']}")
        print(f"   Security pass rate: {report['scan_summary']['pass_rate']:.1f}%")
        
        # Check if we meet the 99.9% target
        if report['scan_summary']['pass_rate'] >= 99.9:
            print("✅ SUCCESS: 99.9% security pass rate achieved!")
        else:
            print(f"❌ NEEDS IMPROVEMENT: {99.9 - report['scan_summary']['pass_rate']:.1f}% improvement needed")
        
        print(f"\n📋 Recommendations ({len(report.get('recommendations', []))}):")
        for i, rec in enumerate(report.get('recommendations', [])[:5], 1):
            print(f"   {i}. {rec}")
        
        return report
        
    except Exception as e:
        print(f"❌ Security testing failed: {e}")
        logger.error(f"Security testing failed: {e}")
        return None


if __name__ == "__main__":
    asyncio.run(main())