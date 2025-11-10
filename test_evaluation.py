#!/usr/bin/env python3
"""
Comprehensive Test Suite for NeuroLogicX Research Evaluation Pipeline
Validates all components and generates research-ready test results
"""

import json
import time
import sys
from pathlib import Path
from datetime import datetime

# Import research evaluation components
try:
    from evaluation import (
        ResearchDatasetGenerator,
        ResearchEvaluationPipeline, 
        run_complete_research_evaluation,
        EvaluationResult,
        ComparisonResult
    )
    from logic_engine import ResearchBABITaskProcessor
    EVALUATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import evaluation modules: {e}")
    EVALUATION_AVAILABLE = False


class NeuroLogicXTestSuite:
    """Comprehensive test suite for NeuroLogicX research evaluation"""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = time.time()
        
    def log_test(self, test_name: str, status: str, details: str = ""):
        """Log test results with timing"""
        test_duration = time.time() - self.start_time
        self.test_results[test_name] = {
            "status": status,
            "details": details,
            "duration_seconds": round(test_duration, 3),
            "timestamp": datetime.now().isoformat()
        }
        print(f"[{status}] {test_name}: {details}")

    def test_environment_setup(self):
        """Test that all required dependencies are available"""
        print("Testing environment setup...")
        
        # Check Python version
        python_version = sys.version_info
        if python_version >= (3, 8):
            self.log_test("Python Version", "PASS", f"Python {python_version.major}.{python_version.minor}.{python_version.micro}")
        else:
            self.log_test("Python Version", "FAIL", "Python 3.8+ required")
            return False

        # Check evaluation availability
        if EVALUATION_AVAILABLE:
            self.log_test("Evaluation Imports", "PASS", "All evaluation modules imported successfully")
        else:
            self.log_test("Evaluation Imports", "FAIL", "Could not import evaluation modules")
            return False

        return True

    def test_dataset_generation(self):
        """Test research-grade dataset generation"""
        print("\nTesting research dataset generation...")
        
        try:
            generator = ResearchDatasetGenerator(seed=42)  # Fixed seed for reproducibility
            
            # Test individual task types
            test_cases = {
                "Simple Location": generator.generate_simple_location_task(),
                "Sequential Movement": generator.generate_sequential_location_task(),
                "Multiple People": generator.generate_multiple_people_task(),
                "Object Transfer": generator.generate_object_transfer_task(),
                "Complex Reasoning": generator.generate_complex_reasoning_task()
            }
            
            for task_type, task in test_cases.items():
                if self._validate_task(task, task_type):
                    self.log_test(f"Task Generation - {task_type}", "PASS", 
                                f"Generated valid {task_type.lower()} task")
                else:
                    self.log_test(f"Task Generation - {task_type}", "FAIL", 
                                "Invalid task structure")
                    return False
            
            # Generate comprehensive dataset
            dataset = generator.generate_research_dataset(n_samples=50)
            dataset_stats = generator.get_dataset_statistics(dataset)
            
            if len(dataset) == 50:
                self.log_test("Dataset Generation", "PASS", 
                            f"Generated {len(dataset)} samples with {dataset_stats['dataset_complexity_score']:.1f} avg complexity")
            else:
                self.log_test("Dataset Generation", "FAIL", 
                            f"Expected 50 samples, got {len(dataset)}")
                return False
            
            return dataset
            
        except Exception as e:
            self.log_test("Dataset Generation", "ERROR", f"Exception: {str(e)}")
            return None

    def _validate_task(self, task: dict, task_type: str) -> bool:
        """Validate task structure and content"""
        required_fields = ['story', 'question', 'answer', 'task_type', 'difficulty']
        
        for field in required_fields:
            if field not in task:
                print(f"Missing required field: {field}")
                return False
        
        if not isinstance(task['story'], list) or len(task['story']) == 0:
            print("Invalid story format")
            return False
            
        if not task['question'] or not task['answer']:
            print("Empty question or answer")
            return False
            
        return True

    def test_individual_systems(self, test_dataset: list):
        """Test each AI system with standardized test cases"""
        print("\nTesting individual AI systems...")
        
        try:
            pipeline = ResearchEvaluationPipeline()
            
            # Use first 3 tasks for system testing
            test_cases = test_dataset[:3] if test_dataset else self._create_fallback_test_cases()
            
            system_results = {}
            
            for system_name, system in pipeline.systems.items():
                print(f"\nTesting {system_name}...")
                system_performance = []
                
                for i, test_case in enumerate(test_cases):
                    try:
                        start_time = time.time()
                        
                        if system_name == "NeuroLogicX":
                            # Use the processor for NeuroLogicX
                            trace = system.process_task(test_case['story'], test_case['question'])
                            answer = trace.final_answer
                            confidence = trace.confidence
                            processing_time = trace.processing_time
                        else:
                            # Use answer_question for baselines
                            answer, confidence = system.answer_question(test_case['story'], test_case['question'])
                            processing_time = time.time() - start_time
                        
                        expected = test_case['answer'].lower().strip()
                        predicted = answer.lower().strip()
                        correct = predicted == expected
                        
                        system_performance.append({
                            'test_case': i,
                            'correct': correct,
                            'confidence': confidence,
                            'processing_time': processing_time,
                            'expected': expected,
                            'predicted': predicted
                        })
                        
                        status = "✓" if correct else "✗"
                        print(f"  Test {i+1}: {status} '{predicted}' (expected: '{expected}') "
                              f"conf: {confidence:.3f}, time: {processing_time:.3f}s")
                        
                    except Exception as e:
                        print(f"  Test {i+1}: ERROR - {str(e)}")
                        system_performance.append({
                            'test_case': i,
                            'correct': False,
                            'confidence': 0.0,
                            'processing_time': 0.0,
                            'error': str(e)
                        })
                
                # Calculate system accuracy
                correct_count = sum(1 for result in system_performance if result.get('correct', False))
                accuracy = correct_count / len(system_performance) if system_performance else 0.0
                
                system_results[system_name] = {
                    'accuracy': accuracy,
                    'correct_count': correct_count,
                    'total_tests': len(system_performance),
                    'details': system_performance
                }
                
                self.log_test(f"System Test - {system_name}", "PASS" if accuracy > 0 else "WARN",
                            f"Accuracy: {accuracy:.1%} ({correct_count}/{len(system_performance)})")
            
            return system_results
            
        except Exception as e:
            self.log_test("System Testing", "ERROR", f"Exception: {str(e)}")
            return None

    def _create_fallback_test_cases(self):
        """Create fallback test cases if dataset generation fails"""
        return [
            {
                'story': ["Mary moved to the bathroom.", "John went to the kitchen."],
                'question': "Where is Mary?",
                'answer': "bathroom",
                'task_type': "simple_location",
                'difficulty': 1
            },
            {
                'story': ["John went to the office.", "Mary traveled to the garden.", "John moved to the hallway."],
                'question': "Where is John?",
                'answer': "hallway", 
                'task_type': "sequential_location",
                'difficulty': 2
            }
        ]

    def test_evaluation_pipeline(self):
        """Test the complete research evaluation pipeline"""
        print("\nTesting research evaluation pipeline...")
        
        try:
            pipeline = ResearchEvaluationPipeline()
            
            # Test with small dataset for speed
            print("Running evaluation with 50 samples...")
            comparison = pipeline.run_research_evaluation(n_samples=50, use_cross_validation=False)
            
            if comparison and isinstance(comparison, ComparisonResult):
                # Validate evaluation results
                valid_results = self._validate_evaluation_results(comparison)
                
                if valid_results:
                    self.log_test("Evaluation Pipeline", "PASS", 
                                f"Successfully evaluated {len(comparison.results)} systems")
                    
                    # Print summary
                    print("\nEvaluation Summary:")
                    print("-" * 40)
                    for i, system_name in enumerate(comparison.performance_ranking):
                        result = comparison.results[system_name]
                        print(f"{i+1}. {system_name}: {result.accuracy:.1%} accuracy "
                              f"({result.correct_answers}/{result.total_questions})")
                    
                    return comparison
                else:
                    self.log_test("Evaluation Pipeline", "FAIL", "Invalid evaluation results")
                    return None
            else:
                self.log_test("Evaluation Pipeline", "FAIL", "Evaluation returned invalid object")
                return None
                
        except Exception as e:
            self.log_test("Evaluation Pipeline", "ERROR", f"Exception: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _validate_evaluation_results(self, comparison: ComparisonResult) -> bool:
        """Validate evaluation results structure and content"""
        if not comparison.results:
            print("No results in comparison")
            return False
            
        for system_name, result in comparison.results.items():
            if not isinstance(result, EvaluationResult):
                print(f"Invalid result type for {system_name}")
                return False
                
            if result.accuracy < 0 or result.accuracy > 1:
                print(f"Invalid accuracy for {system_name}: {result.accuracy}")
                return False
                
            if result.correct_answers > result.total_questions:
                print(f"More correct answers than total questions for {system_name}")
                return False
                
        return True

    def test_research_outputs(self, comparison: ComparisonResult):
        """Test generation of research outputs (plots, tables, reports)"""
        print("\nTesting research output generation...")
        
        try:
            pipeline = ResearchEvaluationPipeline()
            
            # Test paper results generation
            paper_results = pipeline.generate_research_paper_results()
            
            if paper_results and 'main_results' in paper_results:
                self.log_test("Paper Results Generation", "PASS", 
                            "Successfully generated paper-ready results")
                
                # Save sample outputs
                output_dir = Path("test_outputs")
                output_dir.mkdir(exist_ok=True)
                
                # Save paper results
                with open(output_dir / "test_paper_results.json", "w") as f:
                    json.dump(paper_results, f, indent=2)
                
                # Test plot generation (if comparison available)
                if comparison:
                    try:
                        pipeline.create_research_plots(comparison, save_dir=str(output_dir / "plots"))
                        self.log_test("Plot Generation", "PASS", "Generated research plots")
                    except Exception as e:
                        self.log_test("Plot Generation", "WARN", f"Could not generate plots: {e}")
                
                # Test table generation
                try:
                    pipeline.export_research_tables(comparison, save_dir=str(output_dir / "tables"))
                    self.log_test("Table Generation", "PASS", "Generated LaTeX tables")
                except Exception as e:
                    self.log_test("Table Generation", "WARN", f"Could not generate tables: {e}")
                
                return paper_results
            else:
                self.log_test("Paper Results Generation", "FAIL", "Failed to generate paper results")
                return None
                
        except Exception as e:
            self.log_test("Research Outputs", "ERROR", f"Exception: {str(e)}")
            return None

    def generate_comprehensive_report(self):
        """Generate comprehensive test report"""
        print("\nGenerating comprehensive test report...")
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result['status'] == 'PASS')
        failed_tests = sum(1 for result in self.test_results.values() if result['status'] == 'FAIL')
        warning_tests = sum(1 for result in self.test_results.values() if result['status'] == 'WARN')
        
        report = {
            "test_suite": "NeuroLogicX Research Evaluation Test Suite",
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "warning_tests": warning_tests,
                "success_rate": f"{(passed_tests / total_tests * 100):.1f}%" if total_tests > 0 else "0%"
            },
            "test_results": self.test_results,
            "system_status": "READY" if failed_tests == 0 else "ISSUES_DETECTED"
        }
        
        # Save detailed report
        with open("neuro_logicx_test_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        # Print executive summary
        print("\n" + "="*70)
        print("NEUROLOGICX TEST SUITE EXECUTIVE SUMMARY")
        print("="*70)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} | Failed: {failed_tests} | Warnings: {warning_tests}")
        print(f"Success Rate: {report['summary']['success_rate']}")
        print(f"System Status: {report['system_status']}")
        print("="*70)
        
        # Show critical failures
        if failed_tests > 0:
            print("\nCRITICAL FAILURES:")
            for test_name, result in self.test_results.items():
                if result['status'] == 'FAIL':
                    print(f"  ✗ {test_name}: {result['details']}")
        
        # Show warnings
        if warning_tests > 0:
            print("\nWARNINGS:")
            for test_name, result in self.test_results.items():
                if result['status'] == 'WARN':
                    print(f"  ⚠ {test_name}: {result['details']}")
        
        return report

    def run_complete_test_suite(self):
        """Run the complete test suite"""
        print("NeuroLogicX Research Evaluation - Comprehensive Test Suite")
        print("=" * 70)
        
        # Track overall success
        all_tests_passed = True
        
        # 1. Environment setup
        if not self.test_environment_setup():
            print("❌ Environment setup failed. Cannot proceed with tests.")
            return False
        
        # 2. Dataset generation
        test_dataset = self.test_dataset_generation()
        if not test_dataset:
            print("❌ Dataset generation failed.")
            all_tests_passed = False
        
        # 3. Individual system testing
        system_results = self.test_individual_systems(test_dataset)
        if not system_results:
            print("⚠ System testing completed with issues")
        
        # 4. Evaluation pipeline
        comparison = self.test_evaluation_pipeline()
        if not comparison:
            print("❌ Evaluation pipeline test failed.")
            all_tests_passed = False
        
        # 5. Research outputs (only if evaluation succeeded)
        if comparison:
            research_outputs = self.test_research_outputs(comparison)
            if not research_outputs:
                print("⚠ Research output generation completed with issues")
        
        # 6. Generate final report
        report = self.generate_comprehensive_report()
        
        print(f"\n🎯 TEST SUITE COMPLETED: {'SUCCESS' if all_tests_passed else 'ISSUES DETECTED'}")
        print("Generated files:")
        print("  • neuro_logicx_test_report.json - Comprehensive test results")
        print("  • test_outputs/ - Sample research outputs")
        print("  • test_paper_results.json - Sample paper-ready results")
        
        return all_tests_passed


def quick_demonstration():
    """Quick demonstration for researchers"""
    print("\n" + "="*60)
    print("QUICK RESEARCH DEMONSTRATION")
    print("="*60)
    
    try:
        # Initialize pipeline
        pipeline = ResearchEvaluationPipeline()
        
        # Generate sample dataset
        generator = ResearchDatasetGenerator()
        sample_data = generator.generate_research_dataset(n_samples=20)
        
        print(f"Generated {len(sample_data)} research samples")
        print(f"Sample task: '{sample_data[0]['question']}' -> '{sample_data[0]['answer']}'")
        
        # Quick evaluation
        print("\nRunning quick research evaluation...")
        comparison = pipeline.run_research_evaluation(n_samples=30, use_cross_validation=False)
        
        if comparison:
            print("\nResearch Results:")
            print("-" * 40)
            for system_name, result in comparison.results.items():
                print(f"{system_name:15}: {result.accuracy:.1%} accuracy")
            
            print(f"\nBest System: {comparison.best_system}")
            print("Statistical Significance:")
            for test_name, test_result in comparison.statistical_tests.items():
                stars = "**" if test_result.p_value < 0.01 else "*" if test_result.p_value < 0.05 else "ns"
                print(f"  {test_name}: p={test_result.p_value:.4f} ({stars})")
        
    except Exception as e:
        print(f"Demonstration error: {e}")


def main():
    """Main test execution"""
    test_suite = NeuroLogicXTestSuite()
    
    # Run complete test suite
    success = test_suite.run_complete_test_suite()
    
    # Show quick demonstration
    quick_demonstration()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()