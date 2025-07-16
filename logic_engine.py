"""
Neural-Symbolic Reasoning Engine for Terminal-Style AI App
Enhanced with BERT-based neural perception and symbolic reasoning capabilities
Supports bAbI-style reasoning tasks and research evaluation metrics
"""

import random
import re
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from collections import defaultdict

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    from sentence_transformers import SentenceTransformer
    NEURAL_IMPORTS_AVAILABLE = True
except ImportError:
    NEURAL_IMPORTS_AVAILABLE = False
    print("Warning: Neural network dependencies not available. Install with: pip install transformers torch sentence-transformers")


@dataclass
class Entity:
    """Represents an entity extracted from text"""
    name: str
    entity_type: str
    confidence: float = 1.0
    attributes: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}


@dataclass
class Predicate:
    """Represents a symbolic predicate"""
    name: str
    arguments: List[str]
    confidence: float = 1.0
    source: str = "symbolic"  # "neural" or "symbolic"
    
    def __str__(self):
        return f"{self.name}({', '.join(self.arguments)})"


@dataclass
class Rule:
    """Represents a logical rule"""
    conditions: List[Predicate]
    conclusion: Predicate
    confidence: float = 1.0
    
    def __str__(self):
        cond_str = " AND ".join(str(c) for c in self.conditions)
        return f"IF {cond_str} THEN {self.conclusion}"


@dataclass
class ReasoningStep:
    """Represents a step in the reasoning process"""
    step_type: str  # "fact", "rule_application", "conclusion"
    content: str
    predicates: List[Predicate] = None
    rule_used: Rule = None
    confidence: float = 1.0
    
    def __post_init__(self):
        if self.predicates is None:
            self.predicates = []


@dataclass
class ReasoningTrace:
    """Complete reasoning trace for transparency"""
    question: str
    story_facts: List[str]
    extracted_entities: List[Entity]
    symbolic_predicates: List[Predicate]
    reasoning_steps: List[ReasoningStep]
    final_answer: str
    confidence: float
    neural_embeddings: Optional[np.ndarray] = None


class NeuralPerceptionModule:
    """BERT-based neural perception for text understanding"""
    
    def __init__(self):
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.sentence_model = None
        self.entity_patterns = {
            'person': r'\b[A-Z][a-z]+\b',
            'location': r'\b(bathroom|kitchen|hallway|garden|office|bedroom|living room|garage)\b',
            'action': r'\b(moved|went|traveled|walked|ran|took|grabbed|picked up|put down)\b',
            'object': r'\b(apple|book|ball|key|phone|laptop|cup|plate)\b'
        }
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the neural models"""
        if NEURAL_IMPORTS_AVAILABLE:
            try:
                self.sentence_model = SentenceTransformer(self.model_name)
                print(f"✓ Neural perception module initialized with {self.model_name}")
            except Exception as e:
                print(f"Warning: Could not load neural model: {e}")
                self.sentence_model = None
        else:
            print("Warning: Neural models not available - using rule-based fallback")
    
    def encode_text(self, texts: List[str]) -> np.ndarray:
        """Encode texts into neural embeddings"""
        if self.sentence_model is not None:
            try:
                embeddings = self.sentence_model.encode(texts)
                return embeddings
            except Exception as e:
                print(f"Warning: Neural encoding failed: {e}")
        
        # Fallback to simple bag-of-words representation
        return self._fallback_encoding(texts)
    
    def _fallback_encoding(self, texts: List[str]) -> np.ndarray:
        """Fallback encoding when neural models are unavailable"""
        vocab = set()
        for text in texts:
            vocab.update(text.lower().split())
        vocab = sorted(list(vocab))
        
        embeddings = []
        for text in texts:
            words = text.lower().split()
            vector = [1 if word in words else 0 for word in vocab]
            embeddings.append(vector)
        
        return np.array(embeddings)
    
    def extract_entities(self, text: str) -> List[Entity]:
        """Extract entities from text using pattern matching and neural understanding"""
        entities = []
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity_name = match.group().lower()
                confidence = 0.9  # High confidence for pattern-based extraction
                entities.append(Entity(
                    name=entity_name,
                    entity_type=entity_type,
                    confidence=confidence
                ))
        
        return self._deduplicate_entities(entities)
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities, keeping highest confidence"""
        entity_map = {}
        for entity in entities:
            key = (entity.name, entity.entity_type)
            if key not in entity_map or entity.confidence > entity_map[key].confidence:
                entity_map[key] = entity
        return list(entity_map.values())


class SymbolicReasoningEngine:
    """Forward chaining symbolic reasoning engine"""
    
    def __init__(self):
        self.facts = set()
        self.rules = []
        self.reasoning_trace = []
        
        # Initialize with basic spatial reasoning rules
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize with common sense reasoning rules"""
        # Location rules
        self.add_rule(Rule(
            conditions=[Predicate("moved", ["X", "Y"])],
            conclusion=Predicate("at", ["X", "Y"]),
            confidence=0.95
        ))
        
        self.add_rule(Rule(
            conditions=[Predicate("went", ["X", "Y"])],
            conclusion=Predicate("at", ["X", "Y"]),
            confidence=0.95
        ))
        
        # Object possession rules
        self.add_rule(Rule(
            conditions=[Predicate("took", ["X", "Y"])],
            conclusion=Predicate("has", ["X", "Y"]),
            confidence=0.9
        ))
        
        self.add_rule(Rule(
            conditions=[Predicate("picked_up", ["X", "Y"])],
            conclusion=Predicate("has", ["X", "Y"]),
            confidence=0.9
        ))
    
    def add_fact(self, predicate: Predicate):
        """Add a fact to the knowledge base"""
        fact_str = str(predicate)
        if fact_str not in self.facts:
            self.facts.add(fact_str)
            self.reasoning_trace.append(ReasoningStep(
                step_type="fact",
                content=f"Added fact: {predicate}",
                predicates=[predicate],
                confidence=predicate.confidence
            ))
    
    def add_rule(self, rule: Rule):
        """Add a rule to the knowledge base"""
        self.rules.append(rule)
    
    def forward_chain(self, max_iterations: int = 10) -> List[Predicate]:
        """Apply forward chaining to derive new facts"""
        new_facts = []
        
        for iteration in range(max_iterations):
            facts_added = False
            
            for rule in self.rules:
                if self._can_apply_rule(rule):
                    new_predicate = self._apply_rule(rule)
                    if new_predicate and str(new_predicate) not in self.facts:
                        self.add_fact(new_predicate)
                        new_facts.append(new_predicate)
                        facts_added = True
                        
                        self.reasoning_trace.append(ReasoningStep(
                            step_type="rule_application",
                            content=f"Applied rule: {rule}",
                            predicates=[new_predicate],
                            rule_used=rule,
                            confidence=rule.confidence * new_predicate.confidence
                        ))
            
            if not facts_added:
                break
        
        return new_facts
    
    def _can_apply_rule(self, rule: Rule) -> bool:
        """Check if a rule can be applied given current facts"""
        for condition in rule.conditions:
            if not self._matches_any_fact(condition):
                return False
        return True
    
    def _matches_any_fact(self, condition: Predicate) -> bool:
        """Check if a condition matches any existing fact"""
        condition_pattern = self._predicate_to_pattern(condition)
        for fact_str in self.facts:
            if self._matches_pattern(fact_str, condition_pattern):
                return True
        return False
    
    def _predicate_to_pattern(self, predicate: Predicate) -> str:
        """Convert predicate with variables to regex pattern"""
        pattern = predicate.name + r"\("
        for i, arg in enumerate(predicate.arguments):
            if i > 0:
                pattern += r",\s*"
            if arg.isupper():  # Variable
                pattern += r"([^,)]+)"
            else:  # Constant
                pattern += re.escape(arg)
        pattern += r"\)"
        return pattern
    
    def _matches_pattern(self, fact_str: str, pattern: str) -> bool:
        """Check if fact matches pattern"""
        return bool(re.match(pattern, fact_str))
    
    def _apply_rule(self, rule: Rule) -> Optional[Predicate]:
        """Apply a rule and return the new predicate"""
        # Find variable bindings
        bindings = self._find_bindings(rule.conditions)
        if not bindings:
            return None
        
        # Apply bindings to conclusion
        new_args = []
        for arg in rule.conclusion.arguments:
            if arg in bindings:
                new_args.append(bindings[arg])
            else:
                new_args.append(arg)
        
        return Predicate(
            name=rule.conclusion.name,
            arguments=new_args,
            confidence=rule.confidence,
            source="symbolic"
        )
    
    def _find_bindings(self, conditions: List[Predicate]) -> Optional[Dict[str, str]]:
        """Find variable bindings for rule conditions"""
        # Simplified binding finder - in practice would be more sophisticated
        bindings = {}
        
        for condition in conditions:
            pattern = self._predicate_to_pattern(condition)
            for fact_str in self.facts:
                match = re.match(pattern, fact_str)
                if match:
                    var_index = 0
                    for arg in condition.arguments:
                        if arg.isupper():  # Variable
                            bindings[arg] = match.group(var_index + 1)
                            var_index += 1
                    break
        
        return bindings if bindings else None
    
    def query(self, query_predicate: Predicate) -> List[Dict[str, str]]:
        """Query the knowledge base"""
        results = []
        pattern = self._predicate_to_pattern(query_predicate)
        
        for fact_str in self.facts:
            match = re.match(pattern, fact_str)
            if match:
                binding = {}
                var_index = 0
                for arg in query_predicate.arguments:
                    if arg.isupper():  # Variable
                        binding[arg] = match.group(var_index + 1)
                        var_index += 1
                if binding:
                    results.append(binding)
        
        return results


class NeuralSymbolicTranslator:
    """Translates between neural embeddings and symbolic predicates"""
    
    def __init__(self):
        self.action_mappings = {
            'moved': ['moved', 'went', 'traveled', 'walked'],
            'took': ['took', 'grabbed', 'picked', 'picked up'],
            'put': ['put', 'placed', 'set', 'put down']
        }
        
        self.location_keywords = {
            'bathroom', 'kitchen', 'hallway', 'garden', 'office', 
            'bedroom', 'living room', 'garage', 'room'
        }
    
    def text_to_predicates(self, text: str, entities: List[Entity]) -> List[Predicate]:
        """Convert text and entities to symbolic predicates"""
        predicates = []
        text_lower = text.lower()
        
        # Extract entities by type
        people = [e.name for e in entities if e.entity_type == 'person']
        locations = [e.name for e in entities if e.entity_type == 'location']
        actions = [e.name for e in entities if e.entity_type == 'action']
        objects = [e.name for e in entities if e.entity_type == 'object']
        
        # Generate predicates based on sentence structure
        for action in actions:
            canonical_action = self._canonicalize_action(action)
            
            if canonical_action == 'moved' and people and locations:
                # Handle movement predicates
                for person in people:
                    for location in locations:
                        if self._person_action_location_in_text(text_lower, person, action, location):
                            predicates.append(Predicate(
                                name="moved",
                                arguments=[person, location],
                                confidence=0.9,
                                source="neural"
                            ))
            
            elif canonical_action == 'took' and people and objects:
                # Handle object acquisition
                for person in people:
                    for obj in objects:
                        if self._person_action_object_in_text(text_lower, person, action, obj):
                            predicates.append(Predicate(
                                name="took",
                                arguments=[person, obj],
                                confidence=0.9,
                                source="neural"
                            ))
        
        return predicates
    
    def _canonicalize_action(self, action: str) -> str:
        """Map action to canonical form"""
        for canonical, variants in self.action_mappings.items():
            if action in variants:
                return canonical
        return action
    
    def _person_action_location_in_text(self, text: str, person: str, action: str, location: str) -> bool:
        """Check if person-action-location pattern exists in text"""
        # Simple heuristic: person appears before action, action appears before location
        person_pos = text.find(person)
        action_pos = text.find(action)
        location_pos = text.find(location)
        
        return (person_pos >= 0 and action_pos >= 0 and location_pos >= 0 and
                person_pos < action_pos and action_pos < location_pos)
    
    def _person_action_object_in_text(self, text: str, person: str, action: str, obj: str) -> bool:
        """Check if person-action-object pattern exists in text"""
        person_pos = text.find(person)
        action_pos = text.find(action)
        obj_pos = text.find(obj)
        
        return (person_pos >= 0 and action_pos >= 0 and obj_pos >= 0 and
                person_pos < action_pos and action_pos < obj_pos)


class BABITaskProcessor:
    """Processes bAbI-style reasoning tasks"""
    
    def __init__(self):
        self.neural_module = NeuralPerceptionModule()
        self.symbolic_engine = SymbolicReasoningEngine()
        self.translator = NeuralSymbolicTranslator()
        self.evaluation_results = []
    
    def process_task(self, story_sentences: List[str], question: str) -> ReasoningTrace:
        """Process a complete bAbI-style task"""
        # Reset reasoning engine for new task
        self.symbolic_engine = SymbolicReasoningEngine()
        
        # Step 1: Neural perception - extract entities and encode text
        all_entities = []
        all_predicates = []
        
        for sentence in story_sentences:
            entities = self.neural_module.extract_entities(sentence)
            all_entities.extend(entities)
            
            # Translate to symbolic predicates
            predicates = self.translator.text_to_predicates(sentence, entities)
            all_predicates.extend(predicates)
            
            # Add predicates as facts
            for predicate in predicates:
                self.symbolic_engine.add_fact(predicate)
        
        # Step 2: Forward chaining reasoning
        derived_facts = self.symbolic_engine.forward_chain()
        
        # Step 3: Answer the question
        answer, confidence = self._answer_question(question, all_entities)
        
        # Step 4: Create reasoning trace
        trace = ReasoningTrace(
            question=question,
            story_facts=story_sentences,
            extracted_entities=self._deduplicate_entities(all_entities),
            symbolic_predicates=all_predicates + derived_facts,
            reasoning_steps=self.symbolic_engine.reasoning_trace.copy(),
            final_answer=answer,
            confidence=confidence,
            neural_embeddings=self.neural_module.encode_text(story_sentences + [question])
        )
        
        return trace
    
    def _answer_question(self, question: str, entities: List[Entity]) -> Tuple[str, float]:
        """Answer a question based on the knowledge base"""
        question_lower = question.lower()
        
        # Parse question type
        if "where is" in question_lower or "where's" in question_lower:
            return self._answer_location_question(question, entities)
        elif "who is" in question_lower or "who's" in question_lower:
            return self._answer_person_question(question, entities)
        elif "what" in question_lower:
            return self._answer_what_question(question, entities)
        else:
            return "I don't understand the question", 0.0
    
    def _answer_location_question(self, question: str, entities: List[Entity]) -> Tuple[str, float]:
        """Answer 'where is X?' questions"""
        # Extract person name from question
        people = [e.name for e in entities if e.entity_type == 'person']
        question_lower = question.lower()
        
        target_person = None
        for person in people:
            if person in question_lower:
                target_person = person
                break
        
        if not target_person:
            return "Person not found in story", 0.0
        
        # Query for location
        query = Predicate("at", [target_person, "X"])
        results = self.symbolic_engine.query(query)
        
        if results:
            location = results[0]["X"]
            confidence = 0.9
            
            # Add reasoning step
            self.symbolic_engine.reasoning_trace.append(ReasoningStep(
                step_type="conclusion",
                content=f"Concluded that {target_person} is at {location}",
                confidence=confidence
            ))
            
            return location, confidence
        else:
            return "Location unknown", 0.1
    
    def _answer_person_question(self, question: str, entities: List[Entity]) -> Tuple[str, float]:
        """Answer 'who is X?' questions"""
        # Implementation for person-based questions
        return "Person query not implemented", 0.0
    
    def _answer_what_question(self, question: str, entities: List[Entity]) -> Tuple[str, float]:
        """Answer 'what X?' questions"""
        # Implementation for what-based questions
        return "What query not implemented", 0.0
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities"""
        seen = set()
        unique_entities = []
        for entity in entities:
            key = (entity.name, entity.entity_type)
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        return unique_entities
    
    def evaluate_accuracy(self, test_cases: List[Dict]) -> Dict[str, float]:
        """Evaluate accuracy on a set of test cases"""
        correct = 0
        total = len(test_cases)
        results = []
        
        for test_case in test_cases:
            story = test_case['story']
            question = test_case['question']
            expected_answer = test_case['answer'].lower().strip()
            
            trace = self.process_task(story, question)
            predicted_answer = trace.final_answer.lower().strip()
            
            is_correct = predicted_answer == expected_answer
            if is_correct:
                correct += 1
            
            results.append({
                'story': story,
                'question': question,
                'expected': expected_answer,
                'predicted': predicted_answer,
                'correct': is_correct,
                'confidence': trace.confidence
            })
        
        accuracy = correct / total if total > 0 else 0.0
        self.evaluation_results = results
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'detailed_results': results
        }


class TerminalLogicEngine:
    """Enhanced logic engine with neural-symbolic reasoning capabilities"""
    
    def __init__(self):
        # Initialize neurosymbolic components
        self.babi_processor = BABITaskProcessor()
        self.current_story = []
        self.last_reasoning_trace = None
        
        # Original terminal commands
        self.commands = {
            'help': self._help_command,
            'dog': self._dog_command,
            'cat': self._cat_command,
            'add': self._add_command,
            'subtract': self._subtract_command,
            'multiply': self._multiply_command,
            'divide': self._divide_command,
            'clear': self._clear_command,
            'exit': self._exit_command,
            'date': self._date_command,
            'random': self._random_command,
            'echo': self._echo_command,
            
            # New neurosymbolic commands
            'reason': self._reason_command,
            'story': self._story_command,
            'evaluate': self._evaluate_command,
            'neural_status': self._neural_status_command,
            'demo': self._demo_command,
        }
        
        # Fun facts databases
        self.dog_facts = [
            "Dogs have been human companions for over 15,000 years.",
            "A dog's sense of smell is 10,000 to 100,000 times stronger than humans.",
            "Dogs can learn over 150 words and can count up to four or five.",
            "The Basenji dog is known as the 'barkless dog' but they can yodel.",
            "Dogs sweat through their paw pads and nose, not through their skin.",
            "A dog's mouth exerts 150-300 pounds of pressure per square inch.",
            "Dogs can see in color, but not as vividly as humans.",
            "The Norwegian Lundehund has six toes on each foot.",
            "Dogs have three eyelids: upper, lower, and a third for protection.",
            "Puppies are born deaf and blind but develop hearing and sight quickly."
        ]
        
        self.cat_facts = [
            "Cats sleep 12-16 hours per day, which is 70% of their lives.",
            "A group of cats is called a 'clowder' and a group of kittens is a 'kindle'.",
            "Cats have five toes on their front paws but only four on their back paws.",
            "A cat's hearing is much more sensitive than dogs and humans.",
            "Cats can rotate their ears 180 degrees.",
            "The oldest known pet cat existed 9,500 years ago in Cyprus.",
            "Cats have a third eyelid called a nictitating membrane.",
            "A cat's nose print is unique, like a human's fingerprint.",
            "Cats can make over 100 different vocal sounds.",
            "Adult cats only meow to communicate with humans, not other cats."
        ]
    
    def _help_command(self, args: List[str]) -> str:
        """Display available commands"""
        help_text = """
Available Commands:
==================

Basic Commands:
  help                    - Show this help message
  clear                   - Clear the terminal screen
  exit                    - Exit the terminal
  date                    - Show current date and time
  random                  - Generate a random number
  echo <text>             - Echo back the provided text

Fun Facts:
  dog                     - Get a random dog fact
  cat                     - Get a random cat fact

Math Operations:
  add <num1> <num2>       - Add two numbers
  subtract <num1> <num2>  - Subtract two numbers
  multiply <num1> <num2>  - Multiply two numbers
  divide <num1> <num2>    - Divide two numbers

🧠 Neural-Symbolic AI Commands:
  demo                    - Run bAbI reasoning demo
  story <story_text>      - Process story for reasoning
  reason <question>       - Answer question about loaded story
  evaluate                - Run evaluation on test cases
  neural_status          - Check neural model status

Examples:
  Basic: add 5 3          → 8
  AI: demo               → Shows bAbI reasoning demo
  AI: story Mary moved to the bathroom. John went to the hallway.
  AI: reason Where is Mary? → bathroom (with reasoning trace)
"""
        return help_text.strip()
    
    def _dog_command(self, args: List[str]) -> str:
        """Return a random dog fact"""
        return f"🐕 Dog Fact: {random.choice(self.dog_facts)}"
    
    def _cat_command(self, args: List[str]) -> str:
        """Return a random cat fact"""
        return f"🐱 Cat Fact: {random.choice(self.cat_facts)}"
    
    def _add_command(self, args: List[str]) -> str:
        """Add two numbers"""
        if len(args) != 2:
            return "Error: 'add' requires exactly 2 numbers. Usage: add <num1> <num2>"
        
        try:
            num1 = float(args[0])
            num2 = float(args[1])
            result = num1 + num2
            
            # Format result nicely (remove .0 for whole numbers)
            if result.is_integer():
                result = int(result)
            
            return f"{num1} + {num2} = {result}"
        except ValueError:
            return f"Error: Invalid numbers provided. '{args[0]}' and '{args[1]}' must be valid numbers."
    
    def _subtract_command(self, args: List[str]) -> str:
        """Subtract two numbers"""
        if len(args) != 2:
            return "Error: 'subtract' requires exactly 2 numbers. Usage: subtract <num1> <num2>"
        
        try:
            num1 = float(args[0])
            num2 = float(args[1])
            result = num1 - num2
            
            if result.is_integer():
                result = int(result)
            
            return f"{num1} - {num2} = {result}"
        except ValueError:
            return f"Error: Invalid numbers provided. '{args[0]}' and '{args[1]}' must be valid numbers."
    
    def _multiply_command(self, args: List[str]) -> str:
        """Multiply two numbers"""
        if len(args) != 2:
            return "Error: 'multiply' requires exactly 2 numbers. Usage: multiply <num1> <num2>"
        
        try:
            num1 = float(args[0])
            num2 = float(args[1])
            result = num1 * num2
            
            if result.is_integer():
                result = int(result)
            
            return f"{num1} × {num2} = {result}"
        except ValueError:
            return f"Error: Invalid numbers provided. '{args[0]}' and '{args[1]}' must be valid numbers."
    
    def _divide_command(self, args: List[str]) -> str:
        """Divide two numbers"""
        if len(args) != 2:
            return "Error: 'divide' requires exactly 2 numbers. Usage: divide <num1> <num2>"
        
        try:
            num1 = float(args[0])
            num2 = float(args[1])
            
            if num2 == 0:
                return "Error: Cannot divide by zero."
            
            result = num1 / num2
            
            if result.is_integer():
                result = int(result)
            
            return f"{num1} ÷ {num2} = {result}"
        except ValueError:
            return f"Error: Invalid numbers provided. '{args[0]}' and '{args[1]}' must be valid numbers."
    
    def _clear_command(self, args: List[str]) -> str:
        """Clear the terminal screen"""
        return "CLEAR_SCREEN"  # Special return value to signal screen clear
    
    def _exit_command(self, args: List[str]) -> str:
        """Exit the terminal"""
        return "Goodbye! Thanks for using the terminal interface. 👋"
    
    def _date_command(self, args: List[str]) -> str:
        """Show current date and time"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S %A")
        return f"Current date and time: {current_time}"
    
    def _random_command(self, args: List[str]) -> str:
        """Generate a random number"""
        if len(args) == 0:
            # Random number between 1 and 100
            num = random.randint(1, 100)
            return f"Random number: {num}"
        elif len(args) == 1:
            try:
                max_num = int(args[0])
                if max_num <= 0:
                    return "Error: Maximum number must be positive."
                num = random.randint(1, max_num)
                return f"Random number (1-{max_num}): {num}"
            except ValueError:
                return f"Error: '{args[0]}' is not a valid number."
        elif len(args) == 2:
            try:
                min_num = int(args[0])
                max_num = int(args[1])
                if min_num >= max_num:
                    return "Error: Minimum must be less than maximum."
                num = random.randint(min_num, max_num)
                return f"Random number ({min_num}-{max_num}): {num}"
            except ValueError:
                return f"Error: Invalid numbers provided."
        else:
            return "Error: 'random' takes 0, 1, or 2 arguments. Usage: random [max] or random [min] [max]"
    
    def _echo_command(self, args: List[str]) -> str:
        """Echo back the provided text"""
        if not args:
            return "Error: 'echo' requires text to echo. Usage: echo <text>"
        
        return " ".join(args)
    
    def _demo_command(self, args: List[str]) -> str:
        """Run a bAbI reasoning demonstration"""
        try:
            # Demo story and question
            demo_story = [
                "Mary moved to the bathroom.",
                "John went to the hallway.",
                "Sandra moved to the garden."
            ]
            demo_question = "Where is Mary?"
            
            # Process the task
            trace = self.babi_processor.process_task(demo_story, demo_question)
            self.last_reasoning_trace = trace
            
            # Format the response
            result = "🧠 Neural-Symbolic Reasoning Demo\n"
            result += "=" * 35 + "\n\n"
            result += "Story:\n"
            for i, sentence in enumerate(demo_story, 1):
                result += f"  {i}. {sentence}\n"
            
            result += f"\nQuestion: {demo_question}\n"
            result += f"Answer: {trace.final_answer}\n"
            result += f"Confidence: {trace.confidence:.2f}\n\n"
            
            result += "Reasoning Process:\n"
            for i, step in enumerate(trace.reasoning_steps[:5], 1):  # Show first 5 steps
                result += f"  {i}. {step.content}\n"
            
            if len(trace.reasoning_steps) > 5:
                result += f"  ... ({len(trace.reasoning_steps) - 5} more steps)\n"
            
            result += f"\nEntities Found: {len(trace.extracted_entities)}\n"
            result += f"Predicates Generated: {len(trace.symbolic_predicates)}\n"
            
            return result
            
        except Exception as e:
            return f"Demo error: {str(e)}"
    
    def _story_command(self, args: List[str]) -> str:
        """Process a story for reasoning"""
        if not args:
            return "Error: 'story' requires text. Usage: story <story_text>"
        
        try:
            story_text = " ".join(args)
            # Split into sentences (simple approach)
            sentences = [s.strip() for s in story_text.split('.') if s.strip()]
            
            self.current_story = sentences
            
            # Show what was processed
            result = f"📖 Story loaded ({len(sentences)} sentences):\n"
            for i, sentence in enumerate(sentences, 1):
                result += f"  {i}. {sentence}.\n"
            
            result += "\nUse 'reason <question>' to ask questions about this story."
            return result
            
        except Exception as e:
            return f"Story processing error: {str(e)}"
    
    def _reason_command(self, args: List[str]) -> str:
        """Answer a question about the loaded story"""
        if not args:
            return "Error: 'reason' requires a question. Usage: reason <question>"
        
        if not self.current_story:
            return "Error: No story loaded. Use 'story <text>' first."
        
        try:
            question = " ".join(args)
            
            # Process the reasoning task
            trace = self.babi_processor.process_task(self.current_story, question)
            self.last_reasoning_trace = trace
            
            # Format detailed response
            result = "🔍 Neural-Symbolic Reasoning Result\n"
            result += "=" * 38 + "\n\n"
            result += f"Question: {question}\n"
            result += f"Answer: {trace.final_answer}\n"
            result += f"Confidence: {trace.confidence:.2f}\n\n"
            
            result += "📊 Analysis:\n"
            result += f"  • Entities extracted: {len(trace.extracted_entities)}\n"
            result += f"  • Symbolic predicates: {len(trace.symbolic_predicates)}\n"
            result += f"  • Reasoning steps: {len(trace.reasoning_steps)}\n\n"
            
            if trace.extracted_entities:
                result += "🏷️  Entities found:\n"
                for entity in trace.extracted_entities[:5]:  # Show first 5
                    result += f"  • {entity.name} ({entity.entity_type})\n"
            
            result += "\n🧮 Reasoning trace:\n"
            for i, step in enumerate(trace.reasoning_steps, 1):
                result += f"  {i}. {step.content}\n"
                if i >= 8:  # Limit output
                    remaining = len(trace.reasoning_steps) - i
                    if remaining > 0:
                        result += f"  ... ({remaining} more steps)\n"
                    break
            
            return result
            
        except Exception as e:
            return f"Reasoning error: {str(e)}"
    
    def _evaluate_command(self, args: List[str]) -> str:
        """Run evaluation on built-in test cases"""
        try:
            # Built-in test cases for evaluation
            test_cases = [
                {
                    'story': ["Mary moved to the bathroom.", "John went to the hallway."],
                    'question': "Where is Mary?",
                    'answer': "bathroom"
                },
                {
                    'story': ["John went to the kitchen.", "Mary traveled to the office."],
                    'question': "Where is John?",
                    'answer': "kitchen"
                },
                {
                    'story': ["Sandra moved to the garden.", "Daniel went to the bathroom.", "Sandra traveled to the kitchen."],
                    'question': "Where is Sandra?",
                    'answer': "kitchen"
                }
            ]
            
            # Run evaluation
            results = self.babi_processor.evaluate_accuracy(test_cases)
            
            # Format results
            result = "📈 Evaluation Results\n"
            result += "=" * 20 + "\n\n"
            result += f"Accuracy: {results['accuracy']:.2%}\n"
            result += f"Correct: {results['correct']}/{results['total']}\n\n"
            
            result += "📋 Detailed Results:\n"
            for i, test_result in enumerate(results['detailed_results'], 1):
                status = "✅" if test_result['correct'] else "❌"
                result += f"{status} Test {i}: {test_result['predicted']} "
                result += f"(expected: {test_result['expected']})\n"
            
            return result
            
        except Exception as e:
            return f"Evaluation error: {str(e)}"
    
    def _neural_status_command(self, args: List[str]) -> str:
        """Check neural model status"""
        try:
            result = "🔬 Neural-Symbolic System Status\n"
            result += "=" * 34 + "\n\n"
            
            result += f"Neural imports available: {'✅ Yes' if NEURAL_IMPORTS_AVAILABLE else '❌ No'}\n"
            
            if hasattr(self.babi_processor.neural_module, 'sentence_model'):
                model_status = "✅ Loaded" if self.babi_processor.neural_module.sentence_model else "❌ Failed"
                result += f"Sentence transformer: {model_status}\n"
                if self.babi_processor.neural_module.sentence_model:
                    result += f"Model: {self.babi_processor.neural_module.model_name}\n"
            
            result += f"Symbolic rules: {len(self.babi_processor.symbolic_engine.rules)}\n"
            result += f"Current story: {'✅ Loaded' if self.current_story else '❌ None'}\n"
            
            if self.current_story:
                result += f"Story sentences: {len(self.current_story)}\n"
            
            if self.last_reasoning_trace:
                result += f"Last reasoning: {len(self.last_reasoning_trace.reasoning_steps)} steps\n"
            
            return result
            
        except Exception as e:
            return f"Status check error: {str(e)}"
    
    def parse_command(self, command_string: str) -> tuple:
        """Parse a command string into command and arguments"""
        if not command_string.strip():
            return None, []
        
        parts = command_string.strip().split()
        command = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        
        return command, args
    
    def handle_command(self, command_string: str) -> str:
        """
        Main command handler function
        
        Args:
            command_string: Raw command input from user
            
        Returns:
            String response from command execution
        """
        # Handle empty input
        if not command_string.strip():
            return "Error: No command entered. Type 'help' for available commands."
        
        # Parse the command
        command, args = self.parse_command(command_string)
        
        if command is None:
            return "Error: Invalid command format."
        
        # Check if command exists
        if command not in self.commands:
            return f"Error: Unknown command '{command}'. Type 'help' for available commands."
        
        # Execute the command
        try:
            result = self.commands[command](args)
            return result
        except Exception as e:
            return f"Error executing command '{command}': {str(e)}"


# Global instance for easy import
_engine = TerminalLogicEngine()

def handle_command(command: str) -> str:
    """
    Main function to handle terminal commands
    
    Args:
        command: User input command string
        
    Returns:
        Response string from command execution
    """
    return _engine.handle_command(command)


# For testing purposes
if __name__ == "__main__":
    print("Testing Terminal Logic Engine...")
    print("=" * 40)
    
    test_commands = [
        "help",
        "dog",
        "cat", 
        "add 5 3",
        "subtract 10 4",
        "multiply 3 7",
        "divide 15 3",
        "divide 5 0",
        "add 2",
        "random",
        "random 50",
        "random 10 20",
        "echo Hello World!",
        "date",
        "invalid_command",
        ""
    ]
    
    for cmd in test_commands:
        print(f"\n$ {cmd}")
        result = handle_command(cmd)
        print(result)