"""
Comprehensive Evaluation Pipeline for NeuroLogicX Research
Implements baseline comparisons, statistical testing, and paper-ready results export
"""

import random
import time
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
import re

# Import our systems
from logic_engine import BABITaskProcessor, ReasoningTrace

try:
    from transformers import AutoTokenizer, AutoModel, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: Transformers not available. BERT baseline will use fallback.")


@dataclass
class EvaluationResult:
    """Store evaluation results for a single system"""
    system_name: str
    accuracy: float
    total_questions: int
    correct_answers: int
    avg_confidence: float
    avg_response_time: float
    detailed_results: List[Dict]
    metadata: Dict[str, Any] = None


@dataclass
class ComparisonResult:
    """Store comparison results between systems"""
    results: Dict[str, EvaluationResult]
    statistical_tests: Dict[str, Dict]
    best_system: str
    performance_ranking: List[str]


class BABIDatasetGenerator:
    """Generate bAbI-style datasets for evaluation"""
    
    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        
        self.people = ["Mary", "John", "Sandra", "Daniel", "David", "Sarah", "Michael", "Lisa"]
        self.locations = ["bathroom", "kitchen", "hallway", "garden", "office", "bedroom", "living room", "garage"]
        self.actions = ["moved", "went", "traveled", "walked"]
        self.objects = ["apple", "book", "ball", "key", "phone", "laptop", "cup", "plate"]
        
    def generate_simple_location_task(self) -> Dict[str, Any]:
        """Generate a simple location tracking task"""
        person = random.choice(self.people)
        location = random.choice(self.locations)
        action = random.choice(self.actions)
        
        story = [f"{person} {action} to the {location}."]
        question = f"Where is {person}?"
        answer = location
        
        return {
            "story": story,
            "question": question,
            "answer": answer,
            "task_type": "simple_location",
            "difficulty": 1
        }
    
    def generate_sequential_location_task(self) -> Dict[str, Any]:
        """Generate a task with sequential movements"""
        person = random.choice(self.people)
        location1 = random.choice(self.locations)
        location2 = random.choice([l for l in self.locations if l != location1])
        action1, action2 = random.choices(self.actions, k=2)
        
        story = [
            f"{person} {action1} to the {location1}.",
            f"{person} {action2} to the {location2}."
        ]
        question = f"Where is {person}?"
        answer = location2  # Latest location
        
        return {
            "story": story,
            "question": question,
            "answer": answer,
            "task_type": "sequential_location",
            "difficulty": 2
        }
    
    def generate_multiple_people_task(self) -> Dict[str, Any]:
        """Generate a task with multiple people"""
        people = random.sample(self.people, 3)
        locations = random.sample(self.locations, 3)
        actions = random.choices(self.actions, k=3)
        
        story = []
        person_locations = {}
        
        for person, location, action in zip(people, locations, actions):
            story.append(f"{person} {action} to the {location}.")
            person_locations[person] = location
        
        # Ask about one person
        target_person = random.choice(people)
        question = f"Where is {target_person}?"
        answer = person_locations[target_person]
        
        return {
            "story": story,
            "question": question,
            "answer": answer,
            "task_type": "multiple_people",
            "difficulty": 2
        }
    
    def generate_complex_task(self) -> Dict[str, Any]:
        """Generate a complex task with multiple movements and people"""
        people = random.sample(self.people, 2)
        locations = random.sample(self.locations, 4)
        actions = random.choices(self.actions, k=4)
        
        story = []
        person_locations = {}
        
        # Initial movements
        for i, person in enumerate(people):
            location = locations[i]
            action = actions[i]
            story.append(f"{person} {action} to the {location}.")
            person_locations[person] = location
        
        # Additional movements
        for i in range(2):
            person = random.choice(people)
            new_location = random.choice([l for l in locations if l != person_locations[person]])
            action = random.choice(actions)
            story.append(f"{person} {action} to the {new_location}.")
            person_locations[person] = new_location
        
        target_person = random.choice(people)
        question = f"Where is {target_person}?"
        answer = person_locations[target_person]
        
        return {
            "story": story,
            "question": question,
            "answer": answer,
            "task_type": "complex_movement",
            "difficulty": 3
        }
    
    def generate_dataset(self, n_samples: int = 100) -> List[Dict]:
        """Generate a complete evaluation dataset"""
        dataset = []
        
        # Distribution of task types
        simple_count = int(n_samples * 0.3)
        sequential_count = int(n_samples * 0.3)
        multiple_count = int(n_samples * 0.2)
        complex_count = n_samples - simple_count - sequential_count - multiple_count
        
        # Generate tasks
        for _ in range(simple_count):
            dataset.append(self.generate_simple_location_task())
        
        for _ in range(sequential_count):
            dataset.append(self.generate_sequential_location_task())
        
        for _ in range(multiple_count):
            dataset.append(self.generate_multiple_people_task())
        
        for _ in range(complex_count):
            dataset.append(self.generate_complex_task())
        
        # Shuffle dataset
        random.shuffle(dataset)
        
        return dataset


class BERTBaseline:
    """BERT-based baseline for comparison"""
    
    def __init__(self):
        self.model_name = "bert-base-uncased"
        self.qa_pipeline = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize BERT model for question answering"""
        if TRANSFORMERS_AVAILABLE:
            try:
                self.qa_pipeline = pipeline(
                    "question-answering", 
                    model=self.model_name,
                    tokenizer=self.model_name
                )
                print(f"✓ BERT baseline initialized with {self.model_name}")
            except Exception as e:
                print(f"Warning: Could not load BERT model: {e}")
                self.qa_pipeline = None
        else:
            print("Warning: BERT baseline using rule-based fallback")
    
    def answer_question(self, story: List[str], question: str) -> Tuple[str, float]:
        """Answer a question using BERT"""
        if self.qa_pipeline:
            # Combine story into context
            context = " ".join(story)
            
            try:
                result = self.qa_pipeline(question=question, context=context)
                answer = result['answer'].lower().strip()
                confidence = result['score']
                
                # Post-process answer to match expected format
                answer = self._postprocess_answer(answer, story)
                
                return answer, confidence
            except Exception as e:
                print(f"BERT error: {e}")
                return self._fallback_answer(story, question)
        else:
            return self._fallback_answer(story, question)
    
    def _postprocess_answer(self, answer: str, story: List[str]) -> str:
        """Post-process BERT answer to match expected format"""
        # Extract location names from story
        locations = ["bathroom", "kitchen", "hallway", "garden", "office", 
                    "bedroom", "living room", "garage"]
        
        # Find the best matching location
        for location in locations:
            if location in answer or answer in location:
                return location
        
        # If no direct match, use the original answer
        return answer
    
    def _fallback_answer(self, story: List[str], question: str) -> Tuple[str, float]:
        """Fallback answer when BERT is not available"""
        # Simple rule-based fallback
        question_lower = question.lower()
        
        if "where is" in question_lower:
            # Extract person name
            person_match = re.search(r'where is (\w+)', question_lower)
            if person_match:
                person = person_match.group(1)
                
                # Find last mentioned location for this person
                for sentence in reversed(story):
                    if person.lower() in sentence.lower():
                        locations = ["bathroom", "kitchen", "hallway", "garden", 
                                   "office", "bedroom", "living room", "garage"]
                        for location in locations:
                            if location in sentence.lower():
                                return location, 0.5
        
        return "unknown", 0.1


class RuleBasedBaseline:
    """Rule-based baseline using only symbolic reasoning"""
    
    def __init__(self):
        self.entities_patterns = {
            'person': r'\b[A-Z][a-z]+\b',
            'location': r'\b(bathroom|kitchen|hallway|garden|office|bedroom|living room|garage)\b',
            'action': r'\b(moved|went|traveled|walked)\b'
        }
        
    def extract_facts(self, story: List[str]) -> List[Tuple[str, str]]:
        """Extract facts from story using rules"""
        facts = []
        
        for sentence in story:
            # Extract entities
            people = re.findall(self.entities_patterns['person'], sentence)
            locations = re.findall(self.entities_patterns['location'], sentence, re.IGNORECASE)
            actions = re.findall(self.entities_patterns['action'], sentence, re.IGNORECASE)
            
            # Create facts
            if people and locations and actions:
                person = people[0].lower()
                location = locations[0].lower()
                facts.append((person, location))
        
        return facts
    
    def answer_question(self, story: List[str], question: str) -> Tuple[str, float]:
        """Answer question using rule-based reasoning"""
        facts = self.extract_facts(story)
        question_lower = question.lower()
        
        if "where is" in question_lower:
            # Extract person name from question
            person_match = re.search(r'where is (\w+)', question_lower)
            if person_match:
                target_person = person_match.group(1).lower()
                
                # Find the most recent location for this person
                for person, location in reversed(facts):
                    if person == target_person:
                        return location, 0.95
        
        return "unknown", 0.1


class NeuroLogicXEvaluator:
    """Wrapper for NeuroLogicX evaluation"""
    
    def __init__(self):
        self.processor = BABITaskProcessor()
    
    def answer_question(self, story: List[str], question: str) -> Tuple[str, float]:
        """Answer question using NeuroLogicX"""
        try:
            trace = self.processor.process_task(story, question)
            return trace.final_answer.lower().strip(), trace.confidence
        except Exception as e:
            print(f"NeuroLogicX error: {e}")
            return "unknown", 0.0


class EvaluationPipeline:
    """Main evaluation pipeline"""
    
    def __init__(self):
        self.systems = {
            "NeuroLogicX": NeuroLogicXEvaluator(),
            "BERT_Baseline": BERTBaseline(), 
            "Rule_Based": RuleBasedBaseline()
        }
        self.dataset_generator = BABIDatasetGenerator()
        
    def evaluate_system(self, system_name: str, dataset: List[Dict]) -> EvaluationResult:
        """Evaluate a single system on the dataset"""
        system = self.systems[system_name]
        results = []
        correct = 0
        total_time = 0
        confidences = []
        
        print(f"Evaluating {system_name}...")
        
        for i, task in enumerate(dataset):
            start_time = time.time()
            
            predicted_answer, confidence = system.answer_question(
                task["story"], task["question"]
            )
            
            response_time = time.time() - start_time
            total_time += response_time
            
            expected_answer = task["answer"].lower().strip()
            is_correct = predicted_answer == expected_answer
            
            if is_correct:
                correct += 1
            
            confidences.append(confidence)
            
            results.append({
                "task_id": i,
                "story": task["story"],
                "question": task["question"],
                "expected": expected_answer,
                "predicted": predicted_answer,
                "correct": is_correct,
                "confidence": confidence,
                "response_time": response_time,
                "task_type": task["task_type"],
                "difficulty": task["difficulty"]
            })
            
            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{len(dataset)} tasks...")
        
        accuracy = correct / len(dataset)
        avg_confidence = np.mean(confidences)
        avg_response_time = total_time / len(dataset)
        
        return EvaluationResult(
            system_name=system_name,
            accuracy=accuracy,
            total_questions=len(dataset),
            correct_answers=correct,
            avg_confidence=avg_confidence,
            avg_response_time=avg_response_time,
            detailed_results=results
        )
    
    def run_full_evaluation(self, n_samples: int = 200) -> ComparisonResult:
        """Run comprehensive evaluation across all systems"""
        print(f"Generating evaluation dataset with {n_samples} samples...")
        dataset = self.dataset_generator.generate_dataset(n_samples)
        
        results = {}
        for system_name in self.systems.keys():
            results[system_name] = self.evaluate_system(system_name, dataset)
        
        # Statistical significance testing
        statistical_tests = self._perform_statistical_tests(results)
        
        # Rank systems by accuracy
        performance_ranking = sorted(
            results.keys(), 
            key=lambda x: results[x].accuracy, 
            reverse=True
        )
        
        best_system = performance_ranking[0]
        
        return ComparisonResult(
            results=results,
            statistical_tests=statistical_tests,
            best_system=best_system,
            performance_ranking=performance_ranking
        )
    
    def _perform_statistical_tests(self, results: Dict[str, EvaluationResult]) -> Dict:
        """Perform statistical significance tests between systems"""
        tests = {}
        
        # Pairwise t-tests
        system_names = list(results.keys())
        for i, sys1 in enumerate(system_names):
            for sys2 in system_names[i+1:]:
                # Extract correctness for each system
                correct1 = [r["correct"] for r in results[sys1].detailed_results]
                correct2 = [r["correct"] for r in results[sys2].detailed_results]
                
                # Perform t-test (convert boolean to int)
                correct1_int = [int(x) for x in correct1]
                correct2_int = [int(x) for x in correct2]
                statistic, p_value = stats.ttest_rel(correct1_int, correct2_int)
                
                tests[f"{sys1}_vs_{sys2}"] = {
                    "statistic": statistic,
                    "p_value": p_value,
                    "significant": p_value < 0.05,
                    "effect_size": abs(results[sys1].accuracy - results[sys2].accuracy)
                }
        
        return tests
    
    def generate_paper_results(self, comparison: ComparisonResult) -> Dict[str, Any]:
        """Generate results formatted for research paper"""
        paper_results = {
            "main_results": {},
            "statistical_significance": {},
            "performance_by_difficulty": {},
            "confidence_analysis": {},
            "timing_analysis": {}
        }
        
        # Main results
        for system_name, result in comparison.results.items():
            paper_results["main_results"][system_name] = {
                "accuracy": f"{result.accuracy:.1%}",
                "accuracy_numeric": result.accuracy,
                "correct": result.correct_answers,
                "total": result.total_questions,
                "avg_confidence": f"{result.avg_confidence:.3f}",
                "avg_response_time": f"{result.avg_response_time:.3f}s"
            }
        
        # Statistical significance
        paper_results["statistical_significance"] = comparison.statistical_tests
        
        # Performance by difficulty
        for system_name, result in comparison.results.items():
            diff_results = defaultdict(list)
            for r in result.detailed_results:
                diff_results[r["difficulty"]].append(r["correct"])
            
            paper_results["performance_by_difficulty"][system_name] = {
                f"difficulty_{d}": f"{np.mean(correct):.1%}" 
                for d, correct in diff_results.items()
            }
        
        # Confidence analysis
        for system_name, result in comparison.results.items():
            correct_confidences = [r["confidence"] for r in result.detailed_results if r["correct"]]
            incorrect_confidences = [r["confidence"] for r in result.detailed_results if not r["correct"]]
            
            # Calculate correlation, handle NaN values
            confidences = [r["confidence"] for r in result.detailed_results]
            correct_vals = [int(r["correct"]) for r in result.detailed_results]
            
            if len(result.detailed_results) > 1 and len(set(correct_vals)) > 1:
                corr_matrix = np.corrcoef(confidences, correct_vals)
                calibration = float(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0
            else:
                calibration = 0.0
            
            paper_results["confidence_analysis"][system_name] = {
                "avg_confidence_correct": float(np.mean(correct_confidences)) if correct_confidences else 0.0,
                "avg_confidence_incorrect": float(np.mean(incorrect_confidences)) if incorrect_confidences else 0.0,
                "confidence_calibration": calibration
            }
        
        return paper_results
    
    def create_comparison_plots(self, comparison: ComparisonResult, save_dir: str = "./plots"):
        """Create comparison plots for the paper"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Set style for publication-quality plots
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Accuracy comparison bar plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        systems = list(comparison.results.keys())
        accuracies = [comparison.results[s].accuracy for s in systems]
        
        bars = ax.bar(systems, accuracies, color=['#2E86AB', '#A23B72', '#F18F01'])
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('System Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/accuracy_comparison.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_dir}/accuracy_comparison.pdf", bbox_inches='tight')
        plt.close()
        
        # 2. Performance by task difficulty
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        difficulty_data = defaultdict(dict)
        for system_name, result in comparison.results.items():
            diff_results = defaultdict(list)
            for r in result.detailed_results:
                diff_results[r["difficulty"]].append(r["correct"])
            
            for d, correct in diff_results.items():
                difficulty_data[d][system_name] = np.mean(correct)
        
        x = np.arange(len(difficulty_data))
        width = 0.25
        
        for i, system in enumerate(systems):
            accuracies = [difficulty_data[d].get(system, 0) for d in sorted(difficulty_data.keys())]
            ax.bar(x + i*width, accuracies, width, label=system)
        
        ax.set_xlabel('Task Difficulty Level')
        ax.set_ylabel('Accuracy')
        ax.set_title('Performance by Task Difficulty')
        ax.set_xticks(x + width)
        ax.set_xticklabels([f'Level {d}' for d in sorted(difficulty_data.keys())])
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/difficulty_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Response time vs accuracy scatter plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        for system_name, result in comparison.results.items():
            ax.scatter(result.avg_response_time, result.accuracy, 
                      s=200, label=system_name, alpha=0.7)
            ax.annotate(system_name, 
                       (result.avg_response_time, result.accuracy),
                       xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel('Average Response Time (seconds)')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy vs Response Time Trade-off')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/speed_accuracy_tradeoff.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plots saved to {save_dir}/")
    
    def export_latex_tables(self, comparison: ComparisonResult, save_dir: str = "./latex"):
        """Export LaTeX tables for research paper"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Main results table
        latex_main = "\\begin{table}[htb]\n\\centering\n"
        latex_main += "\\caption{Performance Comparison of NeuroLogicX vs Baselines}\n"
        latex_main += "\\label{tab:main_results}\n"
        latex_main += "\\begin{tabular}{lccccc}\n"
        latex_main += "\\toprule\n"
        latex_main += "System & Accuracy & Correct/Total & Avg Confidence & Response Time (s) & Rank \\\\\n"
        latex_main += "\\midrule\n"
        
        for i, system_name in enumerate(comparison.performance_ranking):
            result = comparison.results[system_name]
            clean_name = system_name.replace("_", " ")
            if system_name == "NeuroLogicX":
                clean_name = "\\textbf{NeuroLogicX (Ours)}"
            
            latex_main += f"{clean_name} & "
            latex_main += f"{result.accuracy:.1%} & "
            latex_main += f"{result.correct_answers}/{result.total_questions} & "
            latex_main += f"{result.avg_confidence:.3f} & "
            latex_main += f"{result.avg_response_time:.3f} & "
            latex_main += f"{i+1} \\\\\n"
        
        latex_main += "\\bottomrule\n"
        latex_main += "\\end{tabular}\n"
        latex_main += "\\end{table}\n"
        
        with open(f"{save_dir}/main_results.tex", "w") as f:
            f.write(latex_main)
        
        # Statistical significance table
        latex_stats = "\\begin{table}[htb]\n\\centering\n"
        latex_stats += "\\caption{Statistical Significance Tests (p-values)}\n"
        latex_stats += "\\label{tab:significance}\n"
        latex_stats += "\\begin{tabular}{lcc}\n"
        latex_stats += "\\toprule\n"
        latex_stats += "Comparison & p-value & Significant \\\\\n"
        latex_stats += "\\midrule\n"
        
        for test_name, test_result in comparison.statistical_tests.items():
            comparison_name = test_name.replace("_", " ").replace(" vs ", " vs. ")
            p_val = test_result["p_value"]
            significant = "Yes" if test_result["significant"] else "No"
            
            latex_stats += f"{comparison_name} & "
            latex_stats += f"{p_val:.4f} & "
            latex_stats += f"{significant} \\\\\n"
        
        latex_stats += "\\bottomrule\n"
        latex_stats += "\\end{tabular}\n"
        latex_stats += "\\end{table}\n"
        
        with open(f"{save_dir}/significance_tests.tex", "w") as f:
            f.write(latex_stats)
        
        print(f"LaTeX tables saved to {save_dir}/")
    
    def save_full_results(self, comparison: ComparisonResult, filename: str = "evaluation_results.json"):
        """Save complete evaluation results to JSON"""
        # Convert results to serializable format
        serializable_results = {}
        for system_name, result in comparison.results.items():
            serializable_results[system_name] = asdict(result)
        
        full_results = {
            "timestamp": datetime.now().isoformat(),
            "systems_evaluated": list(comparison.results.keys()),
            "best_system": comparison.best_system,
            "performance_ranking": comparison.performance_ranking,
            "results": serializable_results,
            "statistical_tests": comparison.statistical_tests
        }
        
        with open(filename, "w") as f:
            json.dump(full_results, f, indent=2)
        
        print(f"Complete results saved to {filename}")


def run_quick_evaluation():
    """Run a quick evaluation with smaller dataset"""
    pipeline = EvaluationPipeline()
    print("Running quick evaluation (50 samples)...")
    comparison = pipeline.run_full_evaluation(n_samples=50)
    
    print("\n" + "="*60)
    print("QUICK EVALUATION RESULTS")
    print("="*60)
    
    for i, system_name in enumerate(comparison.performance_ranking):
        result = comparison.results[system_name]
        print(f"{i+1}. {system_name}: {result.accuracy:.1%} accuracy "
              f"({result.correct_answers}/{result.total_questions})")
    
    return comparison


def run_full_evaluation():
    """Run comprehensive evaluation with full dataset"""
    pipeline = EvaluationPipeline()
    print("Running comprehensive evaluation (200 samples)...")
    comparison = pipeline.run_full_evaluation(n_samples=200)
    
    print("\n" + "="*60)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("="*60)
    
    for i, system_name in enumerate(comparison.performance_ranking):
        result = comparison.results[system_name]
        print(f"{i+1}. {system_name}:")
        print(f"   Accuracy: {result.accuracy:.1%}")
        print(f"   Correct: {result.correct_answers}/{result.total_questions}")
        print(f"   Avg Confidence: {result.avg_confidence:.3f}")
        print(f"   Avg Response Time: {result.avg_response_time:.3f}s")
        print()
    
    return comparison


def generate_paper_results():
    """Generate all results needed for research paper"""
    pipeline = EvaluationPipeline()
    
    print("Generating comprehensive results for research paper...")
    comparison = pipeline.run_full_evaluation(n_samples=200)
    
    # Generate paper-formatted results
    paper_results = pipeline.generate_paper_results(comparison)
    
    # Create plots
    pipeline.create_comparison_plots(comparison)
    
    # Export LaTeX tables
    pipeline.export_latex_tables(comparison)
    
    # Save complete results
    pipeline.save_full_results(comparison)
    
    print("\n" + "="*60)
    print("PAPER RESULTS GENERATED")
    print("="*60)
    print("Key findings:")
    
    best_system = comparison.best_system
    best_result = comparison.results[best_system]
    print(f"• Best performing system: {best_system}")
    print(f"• Best accuracy: {best_result.accuracy:.1%}")
    
    # Show key comparisons
    if "NeuroLogicX" in comparison.results:
        neurologic_result = comparison.results["NeuroLogicX"]
        print(f"• NeuroLogicX accuracy: {neurologic_result.accuracy:.1%}")
        
        for system_name, result in comparison.results.items():
            if system_name != "NeuroLogicX":
                improvement = neurologic_result.accuracy - result.accuracy
                print(f"• vs {system_name}: +{improvement:.1%} improvement")
    
    return comparison, paper_results


def create_comparison_plots():
    """Standalone function to create comparison plots"""
    pipeline = EvaluationPipeline()
    comparison = pipeline.run_full_evaluation(n_samples=100)
    pipeline.create_comparison_plots(comparison)
    return comparison


def export_latex_tables():
    """Standalone function to export LaTeX tables"""
    pipeline = EvaluationPipeline()
    comparison = pipeline.run_full_evaluation(n_samples=100)
    pipeline.export_latex_tables(comparison)
    return comparison


if __name__ == "__main__":
    # Run comprehensive evaluation
    print("NeuroLogicX Evaluation Pipeline")
    print("="*50)
    
    # Quick test first
    print("1. Running quick evaluation...")
    quick_results = run_quick_evaluation()
    
    print("\n2. Running full evaluation...")
    full_results = run_full_evaluation()
    
    print("\n3. Generating paper results...")
    comparison, paper_results = generate_paper_results()
    
    print("\nEvaluation complete! Check the generated files:")
    print("• plots/ directory for figures")
    print("• latex/ directory for tables") 
    print("• evaluation_results.json for complete data")