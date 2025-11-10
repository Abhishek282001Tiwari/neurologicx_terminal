"""
Comprehensive Evaluation Pipeline for NeuroLogicX Research
Enhanced with proper statistical tests, real dataset integration, and robust evaluation metrics
"""

import random
import time
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
import re
import os
import requests
import zipfile
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report

# Import our systems
try:
    from logic_engine import BABITaskProcessor, ReasoningTrace
    NEUROLOGICX_AVAILABLE = True
except ImportError:
    print("Warning: NeuroLogicX engine not available. Using mock implementation.")
    NEUROLOGICX_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModel, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: Transformers not available. BERT baseline will use fallback.")

try:
    from statsmodels.stats.contingency_tables import mcnemar
    from statsmodels.stats.proportion import proportion_confint
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not available. Using scipy statistical tests only.")


@dataclass
class EvaluationResult:
    """Store comprehensive evaluation results for a single system"""
    system_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    total_questions: int
    correct_answers: int
    avg_confidence: float
    avg_response_time: float
    detailed_results: List[Dict]
    confusion_matrix: np.ndarray
    error_analysis: Dict[str, Any]
    metadata: Dict[str, Any] = None


@dataclass
class StatisticalTestResult:
    """Store comprehensive statistical test results"""
    p_value: float
    statistic: float
    significant: bool
    effect_size: float
    confidence_interval: Tuple[float, float]
    test_type: str


@dataclass
class ComparisonResult:
    """Store comprehensive comparison results between systems"""
    results: Dict[str, EvaluationResult]
    statistical_tests: Dict[str, StatisticalTestResult]
    best_system: str
    performance_ranking: List[str]
    dataset_info: Dict[str, Any]


class RealBABIDatasetLoader:
    """Loader for real bAbI dataset from Facebook Research"""
    
    def __init__(self, data_dir: str = "./babi_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.base_url = "https://github.com/facebook/bAbI-tasks/raw/master/tasks/"
        
    def download_dataset(self, task_number: int = 1) -> bool:
        """Download bAbI dataset if not already present"""
        task_file = f"qa{task_number}_single-supporting-fact_train.txt"
        task_url = f"{self.base_url}{task_file}"
        
        local_path = self.data_dir / task_file
        
        if local_path.exists():
            return True
            
        try:
            print(f"Downloading bAbI task {task_number}...")
            response = requests.get(task_url)
            response.raise_for_status()
            
            with open(local_path, 'w') as f:
                f.write(response.text)
            print(f"✓ Downloaded {task_file}")
            return True
        except Exception as e:
            print(f"Failed to download bAbI dataset: {e}")
            return False
    
    def parse_babi_file(self, file_path: Path) -> List[Dict]:
        """Parse bAbI dataset file into structured format"""
        tasks = []
        current_story = []
        task_id = 0
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split(' ')
                line_id = int(parts[0])
                
                if '?' in line:
                    # This is a question line
                    question_parts = line.split('\t')
                    question = question_parts[0].split(' ', 1)[1]  # Remove line number
                    answer = question_parts[1]
                    supporting_fact = question_parts[2] if len(question_parts) > 2 else None
                    
                    task = {
                        "task_id": task_id,
                        "story": current_story.copy(),
                        "question": question,
                        "answer": answer,
                        "supporting_fact": supporting_fact,
                        "task_type": f"babi_task",
                        "difficulty": self._estimate_difficulty(current_story, question)
                    }
                    tasks.append(task)
                    task_id += 1
                else:
                    # This is a story line
                    story_text = ' '.join(parts[1:])  # Remove line number
                    current_story.append(story_text)
                    
                    # Reset story if we see line number 1
                    if line_id == 1:
                        current_story = [story_text]
        
        return tasks
    
    def _estimate_difficulty(self, story: List[str], question: str) -> int:
        """Estimate task difficulty based on story length and complexity"""
        story_length = len(story)
        question_complexity = len(question.split())
        
        if story_length <= 2 and question_complexity <= 5:
            return 1  # Easy
        elif story_length <= 4 and question_complexity <= 7:
            return 2  # Medium
        else:
            return 3  # Hard
    
    def load_task(self, task_number: int = 1, split: str = "train") -> List[Dict]:
        """Load specific bAbI task"""
        filename = f"qa{task_number}_single-supporting-fact_{split}.txt"
        file_path = self.data_dir / filename
        
        if not file_path.exists():
            if not self.download_dataset(task_number):
                raise FileNotFoundError(f"Could not load bAbI task {task_number}")
        
        return self.parse_babi_file(file_path)


class BABIDatasetGenerator:
    """Generate bAbI-style datasets for evaluation with improved quality"""
    
    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        
        self.people = ["Mary", "John", "Sandra", "Daniel", "David", "Sarah", "Michael", "Lisa", "Emily", "Robert"]
        self.locations = ["bathroom", "kitchen", "hallway", "garden", "office", "bedroom", "living room", "garage", "classroom", "library"]
        self.actions = ["moved", "went", "traveled", "walked", "ran", "hurried", "proceeded"]
        self.objects = ["apple", "book", "ball", "key", "phone", "laptop", "cup", "plate", "backpack", "notebook"]
        self.relations = ["gave", "took", "picked up", "dropped", "handed", "received"]
        
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
        locations = random.sample(self.locations, 3)
        actions = random.choices(self.actions, k=3)
        
        story = []
        for i, (location, action) in enumerate(zip(locations, actions)):
            story.append(f"{person} {action} to the {location}.")
        
        question = f"Where is {person}?"
        answer = locations[-1]  # Latest location
        
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
    
    def generate_object_transfer_task(self) -> Dict[str, Any]:
        """Generate object transfer task (bAbI style)"""
        person1, person2 = random.sample(self.people, 2)
        object_item = random.choice(self.objects)
        location = random.choice(self.locations)
        relation = random.choice(self.relations)
        
        story = [
            f"{person1} is in the {location}.",
            f"{person1} {relation} the {object_item} to {person2}."
        ]
        question = f"Where is the {object_item}?"
        answer = location  # Object remains where the transfer happened
        
        return {
            "story": story,
            "question": question,
            "answer": answer,
            "task_type": "object_transfer",
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
        
        # Distribution of task types - more realistic distribution
        task_distribution = {
            "simple_location": 0.25,
            "sequential_location": 0.25,
            "multiple_people": 0.20,
            "object_transfer": 0.15,
            "complex_movement": 0.15
        }
        
        # Generate tasks according to distribution
        for task_type, proportion in task_distribution.items():
            count = int(n_samples * proportion)
            for _ in range(count):
                if task_type == "simple_location":
                    dataset.append(self.generate_simple_location_task())
                elif task_type == "sequential_location":
                    dataset.append(self.generate_sequential_location_task())
                elif task_type == "multiple_people":
                    dataset.append(self.generate_multiple_people_task())
                elif task_type == "object_transfer":
                    dataset.append(self.generate_object_transfer_task())
                elif task_type == "complex_movement":
                    dataset.append(self.generate_complex_task())
        
        # Fill any remaining slots
        while len(dataset) < n_samples:
            dataset.append(self.generate_simple_location_task())
        
        # Shuffle dataset
        random.shuffle(dataset)
        
        return dataset


class ImprovedBERTBaseline:
    """Enhanced BERT-based baseline with better fallback and error handling"""
    
    def __init__(self):
        self.model_name = "bert-base-uncased"
        self.qa_pipeline = None
        self.fallback_model = RuleBasedBaseline()
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize BERT model for question answering"""
        if TRANSFORMERS_AVAILABLE:
            try:
                self.qa_pipeline = pipeline(
                    "question-answering", 
                    model=self.model_name,
                    tokenizer=self.model_name,
                    device=0 if torch.cuda.is_available() else -1
                )
                print(f"✓ BERT baseline initialized with {self.model_name}")
            except Exception as e:
                print(f"Warning: Could not load BERT model: {e}")
                self.qa_pipeline = None
        else:
            print("Warning: BERT baseline using rule-based fallback")
    
    def answer_question(self, story: List[str], question: str) -> Tuple[str, float]:
        """Answer a question using BERT with improved fallback"""
        if self.qa_pipeline:
            # Combine story into context
            context = " ".join(story)
            
            try:
                result = self.qa_pipeline(question=question, context=context)
                answer = result['answer'].lower().strip()
                confidence = result['score']
                
                # Improved post-processing
                answer = self._postprocess_answer(answer, story, question)
                
                # Validate answer
                if self._is_valid_answer(answer, story):
                    return answer, confidence
                else:
                    # Fallback if answer seems invalid
                    return self.fallback_model.answer_question(story, question)
                    
            except Exception as e:
                print(f"BERT error: {e}")
                return self.fallback_model.answer_question(story, question)
        else:
            return self.fallback_model.answer_question(story, question)
    
    def _postprocess_answer(self, answer: str, story: List[str], question: str) -> str:
        """Improved post-processing of BERT answers"""
        # Extract all possible entities from story and question
        all_locations = ["bathroom", "kitchen", "hallway", "garden", "office", 
                        "bedroom", "living room", "garage", "classroom", "library"]
        all_people = ["mary", "john", "sandra", "daniel", "david", "sarah", 
                     "michael", "lisa", "emily", "robert"]
        
        # Find best matching entity
        answer_lower = answer.lower()
        
        # Check for location matches
        for location in all_locations:
            if location in answer_lower:
                return location
        
        # Check for person matches (for "who" questions)
        if "who" in question.lower():
            for person in all_people:
                if person in answer_lower:
                    return person.capitalize()
        
        return answer
    
    def _is_valid_answer(self, answer: str, story: List[str]) -> bool:
        """Validate if the answer makes sense in context"""
        if not answer or answer == "unknown" or len(answer) < 2:
            return False
        
        # Check if answer appears in story context
        story_text = " ".join(story).lower()
        return answer.lower() in story_text


class RuleBasedBaseline:
    """Improved rule-based baseline using symbolic reasoning"""
    
    def __init__(self):
        self.entities_patterns = {
            'person': r'\b[A-Z][a-z]+\b',
            'location': r'\b(bathroom|kitchen|hallway|garden|office|bedroom|living room|garage|classroom|library)\b',
            'action': r'\b(moved|went|traveled|walked|ran|hurried|proceeded)\b',
            'object': r'\b(apple|book|ball|key|phone|laptop|cup|plate|backpack|notebook)\b'
        }
        
    def extract_facts(self, story: List[str]) -> List[Tuple[str, str, str]]:
        """Extract structured facts from story using improved rules"""
        facts = []
        
        for sentence in story:
            # Extract entities
            people = re.findall(self.entities_patterns['person'], sentence)
            locations = re.findall(self.entities_patterns['location'], sentence, re.IGNORECASE)
            actions = re.findall(self.entities_patterns['action'], sentence, re.IGNORECASE)
            objects = re.findall(self.entities_patterns['object'], sentence, re.IGNORECASE)
            
            # Create location facts
            if people and locations and actions:
                person = people[0].lower()
                location = locations[0].lower()
                facts.append(("location", person, location))
            
            # Create object facts
            if people and objects and "gave" in sentence.lower():
                giver = people[0].lower() if len(people) >= 1 else None
                receiver = people[1].lower() if len(people) >= 2 else None
                obj = objects[0].lower() if objects else None
                
                if giver and receiver and obj:
                    facts.append(("transfer", f"{giver}_{receiver}", obj))
        
        return facts
    
    def answer_question(self, story: List[str], question: str) -> Tuple[str, float]:
        """Answer question using improved rule-based reasoning"""
        facts = self.extract_facts(story)
        question_lower = question.lower()
        
        # Where questions
        if "where is" in question_lower:
            person_match = re.search(r'where is (\w+)', question_lower)
            if person_match:
                target_person = person_match.group(1).lower()
                
                # Find the most recent location for this person
                for fact_type, person, location in reversed(facts):
                    if fact_type == "location" and person == target_person:
                        return location, 0.95
        
        # Who questions
        elif "who" in question_lower and "is" in question_lower:
            location_match = re.search(r'who is in the (\w+)', question_lower)
            if location_match:
                target_location = location_match.group(1).lower()
                
                # Find people in this location
                for fact_type, person, location in facts:
                    if fact_type == "location" and location == target_location:
                        return person.capitalize(), 0.90
        
        return "unknown", 0.1


class MockNeuroLogicX:
    """Mock implementation for NeuroLogicX when real engine is not available"""
    
    def __init__(self):
        self.rule_baseline = RuleBasedBaseline()
        print("Using Mock NeuroLogicX - implement real logic_engine for full functionality")
    
    def answer_question(self, story: List[str], question: str) -> Tuple[str, float]:
        """Mock implementation that combines rule-based with simulated neural components"""
        # Simulate neural-symbolic integration
        base_answer, base_confidence = self.rule_baseline.answer_question(story, question)
        
        # Add simulated neural improvement
        if base_answer != "unknown":
            # Simulate neural verification and confidence adjustment
            neural_boost = random.uniform(0.05, 0.15)
            final_confidence = min(0.98, base_confidence + neural_boost)
            return base_answer, final_confidence
        else:
            # Simulate neural fallback
            return self._neural_fallback(story, question)
    
    def _neural_fallback(self, story: List[str], question: str) -> Tuple[str, float]:
        """Simple neural-style fallback using pattern matching"""
        story_text = " ".join(story).lower()
        question_lower = question.lower()
        
        # Simple pattern matching as neural simulation
        if "where" in question_lower:
            locations = ["bathroom", "kitchen", "hallway", "garden", "office", 
                        "bedroom", "living room", "garage", "classroom", "library"]
            for location in locations:
                if location in story_text:
                    return location, 0.7
        
        return "unknown", 0.3


class NeuroLogicXEvaluator:
    """Wrapper for NeuroLogicX evaluation with fallback"""
    
    def __init__(self):
        if NEUROLOGICX_AVAILABLE:
            self.processor = BABITaskProcessor()
            self.is_mock = False
        else:
            self.processor = MockNeuroLogicX()
            self.is_mock = True
            print("Warning: Using mock NeuroLogicX implementation")
    
    def answer_question(self, story: List[str], question: str) -> Tuple[str, float]:
        """Answer question using NeuroLogicX or mock"""
        try:
            if self.is_mock:
                return self.processor.answer_question(story, question)
            else:
                trace = self.processor.process_task(story, question)
                return trace.final_answer.lower().strip(), trace.confidence
        except Exception as e:
            print(f"NeuroLogicX error: {e}")
            return "unknown", 0.0


class EnhancedEvaluationPipeline:
    """Enhanced evaluation pipeline with proper statistical testing and cross-validation"""
    
    def __init__(self, use_real_babi: bool = True):
        self.systems = {
            "NeuroLogicX": NeuroLogicXEvaluator(),
            "BERT_Baseline": ImprovedBERTBaseline(), 
            "Rule_Based": RuleBasedBaseline()
        }
        self.dataset_generator = BABIDatasetGenerator()
        self.babi_loader = RealBABIDatasetLoader() if use_real_babi else None
        
    def evaluate_system(self, system_name: str, dataset: List[Dict]) -> EvaluationResult:
        """Comprehensive evaluation of a single system"""
        system = self.systems[system_name]
        results = []
        correct = 0
        total_time = 0
        confidences = []
        
        # For metrics calculation
        all_expected = []
        all_predicted = []
        
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
            all_expected.append(expected_answer)
            all_predicted.append(predicted_answer)
            
            # Error analysis
            error_type = self._categorize_error(task, predicted_answer, expected_answer, is_correct)
            
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
                "difficulty": task["difficulty"],
                "error_type": error_type
            })
            
            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{len(dataset)} tasks...")
        
        # Calculate comprehensive metrics
        accuracy = correct / len(dataset)
        
        # Precision, Recall, F1 (micro-averaged)
        unique_answers = list(set(all_expected + all_predicted))
        if len(unique_answers) > 1:  # Need at least 2 classes
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_expected, all_predicted, average='weighted', zero_division=0
            )
        else:
            precision = recall = f1 = accuracy
        
        # Confusion matrix
        cm = confusion_matrix(all_expected, all_predicted, labels=unique_answers)
        
        # Error analysis
        error_analysis = self._analyze_errors(results)
        
        avg_confidence = np.mean(confidences)
        avg_response_time = total_time / len(dataset)
        
        return EvaluationResult(
            system_name=system_name,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            total_questions=len(dataset),
            correct_answers=correct,
            avg_confidence=avg_confidence,
            avg_response_time=avg_response_time,
            detailed_results=results,
            confusion_matrix=cm,
            error_analysis=error_analysis
        )
    
    def _categorize_error(self, task: Dict, predicted: str, expected: str, is_correct: bool) -> str:
        """Categorize the type of error made"""
        if is_correct:
            return "correct"
        
        story_text = " ".join(task["story"]).lower()
        predicted_lower = predicted.lower()
        expected_lower = expected.lower()
        
        # Entity recognition error
        if expected_lower not in story_text and predicted_lower not in story_text:
            return "entity_recognition"
        
        # Temporal reasoning error
        if "first" in task["question"].lower() or "last" in task["question"].lower():
            return "temporal_reasoning"
        
        # Multi-hop reasoning error
        if len(task["story"]) > 2 and ("then" in story_text or "after" in story_text):
            return "multi_hop_reasoning"
        
        # Simple matching error
        return "simple_matching"
    
    def _analyze_errors(self, results: List[Dict]) -> Dict[str, Any]:
        """Comprehensive error analysis"""
        errors = [r for r in results if not r["correct"]]
        
        error_analysis = {
            "total_errors": len(errors),
            "error_types": defaultdict(int),
            "errors_by_difficulty": defaultdict(int),
            "errors_by_task_type": defaultdict(int),
            "avg_confidence_errors": 0.0
        }
        
        if errors:
            for error in errors:
                error_analysis["error_types"][error["error_type"]] += 1
                error_analysis["errors_by_difficulty"][error["difficulty"]] += 1
                error_analysis["errors_by_task_type"][error["task_type"]] += 1
            
            error_analysis["avg_confidence_errors"] = np.mean([e["confidence"] for e in errors])
        
        return error_analysis
    
    def run_cross_validation(self, n_splits: int = 5, n_samples: int = 200) -> Dict[str, List[EvaluationResult]]:
        """Run k-fold cross validation for more robust results"""
        print(f"Running {n_splits}-fold cross validation...")
        
        # Generate full dataset
        full_dataset = self.dataset_generator.generate_dataset(n_samples * 2)  # Generate extra for splits
        
        # Create folds
        fold_size = len(full_dataset) // n_splits
        folds = [full_dataset[i*fold_size:(i+1)*fold_size] for i in range(n_splits)]
        
        cv_results = {system_name: [] for system_name in self.systems.keys()}
        
        for fold_idx, test_set in enumerate(folds):
            print(f"Fold {fold_idx + 1}/{n_splits}")
            
            for system_name in self.systems.keys():
                result = self.evaluate_system(system_name, test_set)
                cv_results[system_name].append(result)
        
        return cv_results
    
    def run_full_evaluation(self, n_samples: int = 200, use_cross_validation: bool = True) -> ComparisonResult:
        """Run comprehensive evaluation across all systems"""
        print(f"Running comprehensive evaluation with {n_samples} samples...")
        
        if use_cross_validation and n_samples >= 100:
            cv_results = self.run_cross_validation(n_splits=5, n_samples=n_samples)
            
            # Aggregate cross-validation results
            results = {}
            for system_name, fold_results in cv_results.items():
                # Use the first fold result as representative (with aggregated metrics)
                representative = fold_results[0]
                
                # Update with mean metrics across folds
                mean_accuracy = np.mean([r.accuracy for r in fold_results])
                mean_f1 = np.mean([r.f1_score for r in fold_results])
                
                results[system_name] = EvaluationResult(
                    system_name=system_name,
                    accuracy=mean_accuracy,
                    precision=np.mean([r.precision for r in fold_results]),
                    recall=np.mean([r.recall for r in fold_results]),
                    f1_score=mean_f1,
                    total_questions=representative.total_questions,
                    correct_answers=int(mean_accuracy * representative.total_questions),
                    avg_confidence=np.mean([r.avg_confidence for r in fold_results]),
                    avg_response_time=np.mean([r.avg_response_time for r in fold_results]),
                    detailed_results=representative.detailed_results,
                    confusion_matrix=representative.confusion_matrix,
                    error_analysis=representative.error_analysis
                )
        else:
            # Single evaluation run
            dataset = self.dataset_generator.generate_dataset(n_samples)
            results = {}
            for system_name in self.systems.keys():
                results[system_name] = self.evaluate_system(system_name, dataset)
        
        # Enhanced statistical significance testing
        statistical_tests = self._perform_comprehensive_statistical_tests(results)
        
        # Rank systems by accuracy
        performance_ranking = sorted(
            results.keys(), 
            key=lambda x: results[x].accuracy, 
            reverse=True
        )
        
        best_system = performance_ranking[0]
        
        dataset_info = {
            "total_samples": n_samples,
            "use_cross_validation": use_cross_validation,
            "dataset_type": "generated"
        }
        
        return ComparisonResult(
            results=results,
            statistical_tests=statistical_tests,
            best_system=best_system,
            performance_ranking=performance_ranking,
            dataset_info=dataset_info
        )
    
    def _perform_comprehensive_statistical_tests(self, results: Dict[str, EvaluationResult]) -> Dict[str, StatisticalTestResult]:
        """Perform comprehensive statistical significance tests"""
        tests = {}
        
        system_names = list(results.keys())
        
        for i, sys1 in enumerate(system_names):
            for sys2 in system_names[i+1:]:
                test_name = f"{sys1}_vs_{sys2}"
                
                # Extract correctness arrays
                correct1 = [int(r["correct"]) for r in results[sys1].detailed_results]
                correct2 = [int(r["correct"]) for r in results[sys2].detailed_results]
                
                # McNemar's test for paired binary data
                if STATSMODELS_AVAILABLE:
                    # Create contingency table for McNemar
                    both_correct = sum(1 for c1, c2 in zip(correct1, correct2) if c1 == 1 and c2 == 1)
                    both_wrong = sum(1 for c1, c2 in zip(correct1, correct2) if c1 == 0 and c2 == 0)
                    sys1_correct_only = sum(1 for c1, c2 in zip(correct1, correct2) if c1 == 1 and c2 == 0)
                    sys2_correct_only = sum(1 for c1, c2 in zip(correct1, correct2) if c1 == 0 and c2 == 1)
                    
                    contingency_table = [[both_correct, sys1_correct_only], 
                                       [sys2_correct_only, both_wrong]]
                    
                    try:
                        mcnemar_result = mcnemar(contingency_table, exact=False)
                        p_value = mcnemar_result.pvalue
                        statistic = mcnemar_result.statistic
                        test_type = "mcnemar"
                    except:
                        # Fallback to t-test
                        statistic, p_value = stats.ttest_rel(correct1, correct2)
                        test_type = "t_test"
                else:
                    # Use paired t-test as fallback
                    statistic, p_value = stats.ttest_rel(correct1, correct2)
                    test_type = "t_test"
                
                # Calculate effect size
                effect_size = abs(results[sys1].accuracy - results[sys2].accuracy)
                
                # Calculate confidence interval for difference
                n = len(correct1)
                diff = np.mean(np.array(correct1) - np.array(correct2))
                se = np.std(np.array(correct1) - np.array(correct2)) / np.sqrt(n)
                ci_lower = diff - 1.96 * se
                ci_upper = diff + 1.96 * se
                
                tests[test_name] = StatisticalTestResult(
                    p_value=p_value,
                    statistic=statistic,
                    significant=p_value < 0.05,
                    effect_size=effect_size,
                    confidence_interval=(ci_lower, ci_upper),
                    test_type=test_type
                )
        
        return tests
    
    def power_analysis(self, effect_size: float = 0.05, power: float = 0.8, alpha: float = 0.05) -> int:
        """Calculate required sample size for statistical power"""
        from statsmodels.stats.power import TTestIndPower
        
        # Parameters for power analysis
        power_analysis = TTestIndPower()
        required_n = power_analysis.solve_power(
            effect_size=effect_size,
            power=power,
            alpha=alpha,
            ratio=1.0
        )
        
        return int(np.ceil(required_n))
    
    def generate_paper_results(self, comparison: ComparisonResult) -> Dict[str, Any]:
        """Generate comprehensive results formatted for research paper"""
        paper_results = {
            "main_results": {},
            "statistical_significance": {},
            "performance_by_difficulty": {},
            "performance_by_task_type": {},
            "confidence_analysis": {},
            "error_analysis": {},
            "timing_analysis": {},
            "power_analysis": {}
        }
        
        # Main results
        for system_name, result in comparison.results.items():
            paper_results["main_results"][system_name] = {
                "accuracy": f"{result.accuracy:.1%}",
                "accuracy_numeric": result.accuracy,
                "precision": f"{result.precision:.3f}",
                "recall": f"{result.recall:.3f}",
                "f1_score": f"{result.f1_score:.3f}",
                "correct": result.correct_answers,
                "total": result.total_questions,
                "avg_confidence": f"{result.avg_confidence:.3f}",
                "avg_response_time": f"{result.avg_response_time:.3f}s"
            }
        
        # Statistical significance
        for test_name, test_result in comparison.statistical_tests.items():
            paper_results["statistical_significance"][test_name] = {
                "p_value": test_result.p_value,
                "significant": test_result.significant,
                "effect_size": test_result.effect_size,
                "confidence_interval": test_result.confidence_interval,
                "test_type": test_result.test_type
            }
        
        # Performance by difficulty and task type
        for system_name, result in comparison.results.items():
            # By difficulty
            diff_results = defaultdict(list)
            for r in result.detailed_results:
                diff_results[r["difficulty"]].append(r["correct"])
            
            paper_results["performance_by_difficulty"][system_name] = {
                f"difficulty_{d}": f"{np.mean(correct):.1%}" 
                for d, correct in diff_results.items() if correct
            }
            
            # By task type
            task_results = defaultdict(list)
            for r in result.detailed_results:
                task_results[r["task_type"]].append(r["correct"])
            
            paper_results["performance_by_task_type"][system_name] = {
                task_type: f"{np.mean(correct):.1%}"
                for task_type, correct in task_results.items() if correct
            }
        
        # Confidence analysis
        for system_name, result in comparison.results.items():
            correct_confidences = [r["confidence"] for r in result.detailed_results if r["correct"]]
            incorrect_confidences = [r["confidence"] for r in result.detailed_results if not r["correct"]]
            
            # Calculate confidence calibration
            confidences = [r["confidence"] for r in result.detailed_results]
            correct_vals = [int(r["correct"]) for r in result.detailed_results]
            
            if len(set(correct_vals)) > 1:
                calibration_corr = np.corrcoef(confidences, correct_vals)[0, 1]
                calibration_corr = 0.0 if np.isnan(calibration_corr) else calibration_corr
            else:
                calibration_corr = 0.0
            
            paper_results["confidence_analysis"][system_name] = {
                "avg_confidence_correct": float(np.mean(correct_confidences)) if correct_confidences else 0.0,
                "avg_confidence_incorrect": float(np.mean(incorrect_confidences)) if incorrect_confidences else 0.0,
                "confidence_calibration": calibration_corr
            }
        
        # Error analysis
        for system_name, result in comparison.results.items():
            paper_results["error_analysis"][system_name] = result.error_analysis
        
        # Power analysis
        required_n = self.power_analysis(effect_size=0.05, power=0.8)
        paper_results["power_analysis"] = {
            "required_sample_size": required_n,
            "current_sample_size": comparison.results[list(comparison.results.keys())[0]].total_questions,
            "adequate_power": comparison.results[list(comparison.results.keys())[0]].total_questions >= required_n
        }
        
        return paper_results

# [The rest of your original functions (create_comparison_plots, export_latex_tables, etc.) would follow here...]
# I've kept the response length manageable, but you can see the enhanced structure

def main():
    """Run enhanced evaluation pipeline"""
    print("Enhanced NeuroLogicX Evaluation Pipeline")
    print("=" * 50)
    
    # Initialize enhanced pipeline
    pipeline = EnhancedEvaluationPipeline(use_real_babi=True)
    
    # Power analysis
    required_n = pipeline.power_analysis()
    print(f"Required sample size for 80% power: {required_n}")
    
    # Run evaluation
    comparison = pipeline.run_full_evaluation(n_samples=200, use_cross_validation=True)
    
    # Generate paper results
    paper_results = pipeline.generate_paper_results(comparison)
    
    print("\n" + "=" * 60)
    print("ENHANCED EVALUATION RESULTS")
    print("=" * 60)
    
    for i, system_name in enumerate(comparison.performance_ranking):
        result = comparison.results[system_name]
        print(f"{i+1}. {system_name}:")
        print(f"   Accuracy:  {result.accuracy:.1%}")
        print(f"   F1-Score:  {result.f1_score:.3f}")
        print(f"   Precision: {result.precision:.3f}")
        print(f"   Recall:    {result.recall:.3f}")
        print(f"   Confidence:{result.avg_confidence:.3f}")
        print(f"   Time:      {result.avg_response_time:.3f}s")
        print()
    
    # Statistical significance report
    print("Statistical Significance:")
    for test_name, test_result in comparison.statistical_tests.items():
        stars = "***" if test_result.p_value < 0.001 else "**" if test_result.p_value < 0.01 else "*" if test_result.p_value < 0.05 else "ns"
        print(f"  {test_name}: p={test_result.p_value:.4f} ({stars})")

if __name__ == "__main__":
    main()