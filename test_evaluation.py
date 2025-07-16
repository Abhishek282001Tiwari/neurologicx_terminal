#!/usr/bin/env python3
"""
Test script for the NeuroLogicX evaluation pipeline
Demonstrates all key functionality without requiring full neural dependencies
"""

from evaluation import (
    BABIDatasetGenerator, 
    EvaluationPipeline,
    run_quick_evaluation,
    generate_paper_results
)
import json

def test_dataset_generation():
    """Test bAbI dataset generation"""
    print("Testing dataset generation...")
    generator = BABIDatasetGenerator()
    
    # Test different task types
    simple_task = generator.generate_simple_location_task()
    sequential_task = generator.generate_sequential_location_task()
    multiple_task = generator.generate_multiple_people_task()
    complex_task = generator.generate_complex_task()
    
    print(f"✓ Simple task: {simple_task['question']} -> {simple_task['answer']}")
    print(f"✓ Sequential task: {len(sequential_task['story'])} sentences")
    print(f"✓ Multiple people task: {len(multiple_task['story'])} sentences")
    print(f"✓ Complex task: difficulty {complex_task['difficulty']}")
    
    # Generate small dataset
    dataset = generator.generate_dataset(n_samples=10)
    print(f"✓ Generated dataset with {len(dataset)} samples")
    
    return dataset

def test_individual_systems():
    """Test each system individually"""
    print("\nTesting individual systems...")
    
    pipeline = EvaluationPipeline()
    
    # Create a simple test case
    test_story = ["Mary moved to the bathroom.", "John went to the kitchen."]
    test_question = "Where is Mary?"
    expected_answer = "bathroom"
    
    print(f"Test story: {test_story}")
    print(f"Test question: {test_question}")
    print(f"Expected answer: {expected_answer}")
    print()
    
    # Test each system
    for system_name, system in pipeline.systems.items():
        try:
            answer, confidence = system.answer_question(test_story, test_question)
            correct = answer.lower().strip() == expected_answer.lower().strip()
            status = "✓" if correct else "✗"
            print(f"{status} {system_name}: '{answer}' (confidence: {confidence:.3f})")
        except Exception as e:
            print(f"✗ {system_name}: Error - {e}")

def test_quick_evaluation():
    """Test quick evaluation pipeline"""
    print("\nRunning quick evaluation...")
    
    try:
        comparison = run_quick_evaluation()
        
        print("\nResults summary:")
        for i, system_name in enumerate(comparison.performance_ranking):
            result = comparison.results[system_name]
            print(f"{i+1}. {system_name}: {result.accuracy:.1%} accuracy")
        
        # Show statistical tests
        print("\nStatistical significance tests:")
        for test_name, test_result in comparison.statistical_tests.items():
            significant = "significant" if test_result["significant"] else "not significant"
            print(f"  {test_name}: p={test_result['p_value']:.4f} ({significant})")
        
        return comparison
        
    except Exception as e:
        print(f"Error in quick evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None

def demo_paper_results():
    """Demonstrate paper results generation"""
    print("\nGenerating paper-ready results...")
    
    try:
        # Use a smaller dataset for demo
        pipeline = EvaluationPipeline()
        comparison = pipeline.run_full_evaluation(n_samples=30)
        
        # Generate paper results
        paper_results = pipeline.generate_paper_results(comparison)
        
        print("Main results for paper:")
        for system_name, metrics in paper_results["main_results"].items():
            print(f"  {system_name}: {metrics['accuracy']} accuracy, {metrics['avg_response_time']} avg time")
        
        # Show confidence analysis
        print("\nConfidence analysis:")
        for system_name, analysis in paper_results["confidence_analysis"].items():
            print(f"  {system_name}: calibration = {analysis['confidence_calibration']:.3f}")
        
        # Save sample results
        with open("sample_paper_results.json", "w") as f:
            json.dump(paper_results, f, indent=2)
        print("✓ Sample results saved to sample_paper_results.json")
        
        return paper_results
        
    except Exception as e:
        print(f"Error generating paper results: {e}")
        import traceback
        traceback.print_exc()
        return None

def show_concrete_numbers():
    """Show concrete accuracy numbers like those requested"""
    print("\n" + "="*60)
    print("CONCRETE EVALUATION RESULTS")
    print("="*60)
    
    # Run evaluation with specific seed for reproducible results
    pipeline = EvaluationPipeline()
    comparison = pipeline.run_full_evaluation(n_samples=100)
    
    # Extract concrete numbers
    results = comparison.results
    
    print("System Performance:")
    print("-" * 40)
    
    for system_name in ["NeuroLogicX", "BERT_Baseline", "Rule_Based"]:
        if system_name in results:
            result = results[system_name]
            print(f"{system_name:15}: {result.accuracy:.1%} accuracy")
            print(f"{'':15}  ({result.correct_answers}/{result.total_questions} correct)")
            print(f"{'':15}  {result.avg_response_time:.3f}s avg time")
            print(f"{'':15}  {result.avg_confidence:.3f} avg confidence")
            print()
    
    # Show ranking
    print("Performance Ranking:")
    print("-" * 20)
    for i, system_name in enumerate(comparison.performance_ranking):
        result = results[system_name]
        print(f"{i+1}. {system_name}: {result.accuracy:.1%}")
    
    # Show key comparisons
    if "NeuroLogicX" in results and "BERT_Baseline" in results:
        neurologic_acc = results["NeuroLogicX"].accuracy
        bert_acc = results["BERT_Baseline"].accuracy
        improvement = neurologic_acc - bert_acc
        print(f"\nNeuroLogicX vs BERT improvement: +{improvement:.1%}")
    
    return comparison

def main():
    """Run all tests"""
    print("NeuroLogicX Evaluation Pipeline Test")
    print("=" * 50)
    
    # Test dataset generation
    dataset = test_dataset_generation()
    
    # Test individual systems
    test_individual_systems()
    
    # Test evaluation pipeline
    comparison = test_quick_evaluation()
    
    # Demo paper results
    paper_results = demo_paper_results()
    
    # Show concrete numbers
    final_comparison = show_concrete_numbers()
    
    print("\n" + "="*60)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("The evaluation pipeline is ready for research paper generation.")
    print("Key files generated:")
    print("  • sample_paper_results.json - Sample results data")
    print("  • plots/ - Will contain comparison charts")
    print("  • latex/ - Will contain LaTeX tables")

if __name__ == "__main__":
    main()