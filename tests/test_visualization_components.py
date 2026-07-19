#!/usr/bin/env python3
"""
Comprehensive Test Suite for Visualization Components

Tests all visualization components including AtomSpace Explorer,
Cognitive Flowchart Generator, Financial Dashboard, and Hypergraph Visualizer.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from visualization.atomspace_explorer import AtomSpaceExplorer
    from visualization.cognitive_flowchart import CognitiveFlowchartGenerator
    from visualization.financial_dashboard import FinancialDashboard
    from visualization.hypergraph_visualizer import HypergraphVisualizer
except ImportError as e:
    print(f"⚠️  Warning: Could not import visualization components: {e}")
    print("Running basic test structure verification...")


def test_atomspace_explorer():
    """Test AtomSpace Explorer functionality"""
    print("🧠 Testing AtomSpace Explorer...")
    
    try:
        # Initialize explorer
        explorer = AtomSpaceExplorer({
            'max_nodes': 20,
            'enable_3d': True,
            'layout_algorithm': 'force_directed'
        })
        
        # Load sample data
        success = explorer.load_atomspace_data("test_atomspace.scm")
        assert success, "Failed to load AtomSpace data"
        
        # Generate visualization data
        viz_data = explorer.generate_visualization_json()
        assert 'nodes' in viz_data, "Missing nodes in visualization data"
        assert 'edges' in viz_data, "Missing edges in visualization data"
        assert len(viz_data['nodes']) > 0, "No nodes generated"
        
        # Test cognitive insights
        insights = explorer.get_cognitive_insights()
        assert 'node_type_distribution' in insights, "Missing node type distribution"
        assert 'attention_patterns' in insights, "Missing attention patterns"
        
        # Generate HTML visualization
        html_content = explorer.generate_html_visualization()
        assert '<html' in html_content, "Invalid HTML generated"
        assert 'AtomSpace Explorer' in html_content, "Missing title"
        
        # Test file saving
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp:
            success = explorer.save_visualization(tmp.name)
            assert success, "Failed to save visualization"
            assert os.path.exists(tmp.name), "Visualization file not created"
            os.unlink(tmp.name)
        
        print("✅ AtomSpace Explorer: All tests passed")
        return True
        
    except Exception as e:
        print(f"❌ AtomSpace Explorer test failed: {e}")
        return False


def test_cognitive_flowchart():
    """Test Cognitive Flowchart Generator functionality"""
    print("🔄 Testing Cognitive Flowchart Generator...")
    
    try:
        # Initialize generator
        generator = CognitiveFlowchartGenerator({
            'layout': 'hierarchical',
            'enable_animation': True,
            'interactive': True
        })
        
        # Create integration flowchart
        integration_flow = generator.create_integration_flowchart()
        assert 'nodes' in integration_flow, "Missing nodes in integration flowchart"
        assert 'edges' in integration_flow, "Missing edges in integration flowchart"
        assert len(integration_flow['nodes']) > 5, "Too few nodes in integration flowchart"
        
        # Create financial reasoning flowchart
        financial_flow = generator.create_financial_reasoning_flowchart()
        assert 'nodes' in financial_flow, "Missing nodes in financial flowchart"
        assert 'edges' in financial_flow, "Missing edges in financial flowchart"
        
        # Generate HTML visualization
        html_content = generator.generate_html_flowchart(integration_flow)
        assert '<html' in html_content, "Invalid HTML generated"
        assert '3D Hypergraph Explorer' in html_content or 'flowchart' in html_content.lower(), "Missing flowchart content"
        
        # Test file saving
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp:
            success = generator.save_flowchart(integration_flow, tmp.name)
            assert success, "Failed to save flowchart"
            assert os.path.exists(tmp.name), "Flowchart file not created"
            os.unlink(tmp.name)
        
        # Test generating all flowcharts
        with tempfile.TemporaryDirectory() as tmp_dir:
            saved_files = generator.generate_all_flowcharts(tmp_dir)
            assert len(saved_files) > 0, "No flowcharts generated"
            for file in saved_files:
                assert os.path.exists(file), f"Generated file does not exist: {file}"
        
        print("✅ Cognitive Flowchart Generator: All tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Cognitive Flowchart Generator test failed: {e}")
        return False


def test_financial_dashboard():
    """Test Financial Dashboard functionality"""
    print("💰 Testing Financial Dashboard...")
    
    try:
        # Initialize dashboard
        dashboard = FinancialDashboard({
            'currency': 'USD',
            'enable_predictions': True,
            'show_cognitive_insights': True
        })
        
        # Load sample financial data
        success = dashboard.load_gnucash_data("test_gnucash.sqlite")
        assert success, "Failed to load GnuCash data"
        
        # Generate cognitive insights
        insights = dashboard.generate_cognitive_insights()
        assert 'spending_patterns' in insights, "Missing spending patterns"
        assert 'risk_assessment' in insights, "Missing risk assessment"
        assert 'optimization_suggestions' in insights, "Missing optimization suggestions"
        
        # Generate predictions
        predictions = dashboard.generate_predictions()
        assert 'next_month_expenses' in predictions, "Missing expense predictions"
        assert 'investment_projections' in predictions, "Missing investment projections"
        assert 'cash_flow_forecast' in predictions, "Missing cash flow forecast"
        
        # Get key metrics
        metrics = dashboard.get_key_metrics()
        assert len(metrics) > 0, "No key metrics generated"
        assert all(hasattr(m, 'name') and hasattr(m, 'value') for m in metrics), "Invalid metric structure"
        
        # Generate HTML dashboard
        html_content = dashboard.generate_dashboard_html()
        assert '<html' in html_content, "Invalid HTML generated"
        assert 'Financial Intelligence' in html_content, "Missing dashboard title"
        assert 'Chart.js' in html_content or 'chart' in html_content.lower(), "Missing chart library"
        
        # Test file saving
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp:
            success = dashboard.save_dashboard(tmp.name)
            assert success, "Failed to save dashboard"
            assert os.path.exists(tmp.name), "Dashboard file not created"
            os.unlink(tmp.name)
        
        print("✅ Financial Dashboard: All tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Financial Dashboard test failed: {e}")
        return False


def test_hypergraph_visualizer():
    """Test Hypergraph Visualizer functionality"""
    print("🔗 Testing Hypergraph Visualizer...")
    
    try:
        # Initialize visualizer
        visualizer = HypergraphVisualizer({
            'dimensions': 3,
            'layout_algorithm': 'force_directed_3d',
            'enable_physics': True,
            'interactive': True
        })
        
        # Create AtomSpace hypergraph
        atomspace_graph = visualizer.create_atomspace_hypergraph()
        assert 'nodes' in atomspace_graph, "Missing nodes in AtomSpace hypergraph"
        assert 'hyperedges' in atomspace_graph, "Missing hyperedges in AtomSpace hypergraph"
        assert len(atomspace_graph['nodes']) > 0, "No nodes in AtomSpace hypergraph"
        
        # Create cognitive reasoning hypergraph
        reasoning_graph = visualizer.create_cognitive_reasoning_hypergraph()
        assert 'nodes' in reasoning_graph, "Missing nodes in reasoning hypergraph"
        assert 'hyperedges' in reasoning_graph, "Missing hyperedges in reasoning hypergraph"
        
        # Test force-directed layout
        original_positions = [(node['position'][0], node['position'][1], node['position'][2]) 
                            for node in atomspace_graph['nodes']]
        
        visualizer.apply_force_directed_layout(10)  # Few iterations for testing
        
        # Positions should potentially change (though not guaranteed in few iterations)
        print("  Layout optimization completed")
        
        # Generate 3D HTML visualization
        html_content = visualizer.generate_3d_visualization_html(atomspace_graph)
        assert '<html' in html_content, "Invalid HTML generated"
        assert 'Three.js' in html_content or 'three.js' in html_content.lower(), "Missing Three.js library"
        assert '3D Hypergraph' in html_content, "Missing 3D hypergraph title"
        
        # Test file saving
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp:
            success = visualizer.save_visualization(atomspace_graph, tmp.name)
            assert success, "Failed to save hypergraph visualization"
            assert os.path.exists(tmp.name), "Hypergraph file not created"
            os.unlink(tmp.name)
        
        # Test generating all visualizations
        with tempfile.TemporaryDirectory() as tmp_dir:
            saved_files = visualizer.generate_all_visualizations(tmp_dir)
            assert len(saved_files) > 0, "No hypergraph visualizations generated"
            for file in saved_files:
                assert os.path.exists(file), f"Generated file does not exist: {file}"
        
        print("✅ Hypergraph Visualizer: All tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Hypergraph Visualizer test failed: {e}")
        return False


def test_integration_workflows():
    """Test integration between visualization components"""
    print("🔗 Testing Integration Workflows...")
    
    try:
        # Test data flow between components
        explorer = AtomSpaceExplorer({'max_nodes': 10, 'enable_3d': True})
        generator = CognitiveFlowchartGenerator()
        dashboard = FinancialDashboard()
        visualizer = HypergraphVisualizer()
        
        # Load data in all components
        explorer.load_atomspace_data("test_data")
        dashboard.load_gnucash_data("test_data")
        
        # Generate data from each component
        atomspace_data = explorer.generate_visualization_json()
        flowchart_data = generator.create_integration_flowchart()
        financial_insights = dashboard.generate_cognitive_insights()
        hypergraph_data = visualizer.create_atomspace_hypergraph()
        
        # Verify data compatibility
        assert all('nodes' in data for data in [atomspace_data, flowchart_data, hypergraph_data]), \
            "Inconsistent data structure across components"
        
        # Test cross-component analysis
        total_cognitive_elements = (
            len(atomspace_data['nodes']) +
            len(flowchart_data['nodes']) +
            len(hypergraph_data['nodes'])
        )
        
        assert total_cognitive_elements > 10, "Insufficient cognitive elements for integration"
        
        # Test unified visualization generation
        with tempfile.TemporaryDirectory() as tmp_dir:
            files_generated = []
            
            # Generate from each component
            if explorer.save_visualization(f"{tmp_dir}/atomspace.html"):
                files_generated.append("atomspace.html")
            
            if generator.save_flowchart(flowchart_data, f"{tmp_dir}/flowchart.html"):
                files_generated.append("flowchart.html")
            
            if dashboard.save_dashboard(f"{tmp_dir}/dashboard.html"):
                files_generated.append("dashboard.html")
            
            if visualizer.save_visualization(hypergraph_data, f"{tmp_dir}/hypergraph.html"):
                files_generated.append("hypergraph.html")
            
            assert len(files_generated) >= 3, f"Only {len(files_generated)} visualization files generated"
        
        print("✅ Integration Workflows: All tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Integration Workflows test failed: {e}")
        return False


def run_comprehensive_test_suite():
    """Run all visualization component tests"""
    print("🧪 Starting Comprehensive Visualization Component Test Suite")
    print("=" * 60)
    
    test_results = {
        'AtomSpace Explorer': False,
        'Cognitive Flowchart': False,
        'Financial Dashboard': False,
        'Hypergraph Visualizer': False,
        'Integration Workflows': False
    }
    
    try:
        # Run individual component tests
        test_results['AtomSpace Explorer'] = test_atomspace_explorer()
        test_results['Cognitive Flowchart'] = test_cognitive_flowchart()
        test_results['Financial Dashboard'] = test_financial_dashboard()
        test_results['Hypergraph Visualizer'] = test_hypergraph_visualizer()
        test_results['Integration Workflows'] = test_integration_workflows()
        
    except Exception as e:
        print(f"💥 Test suite execution error: {e}")
    
    # Report results
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<30} {status}")
    
    print("-" * 60)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL VISUALIZATION COMPONENT TESTS PASSED!")
        print("🚀 Visualization framework is ready for production use")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} tests failed")
        print("🔧 Please review and fix failing components before deployment")
    
    return passed_tests == total_tests


def main():
    """Main test execution"""
    success = run_comprehensive_test_suite()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()