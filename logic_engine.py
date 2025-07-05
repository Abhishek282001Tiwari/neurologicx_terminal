"""
Logic Engine for Terminal-Style AI App
Handles command parsing and execution for the terminal interface
"""

import random
from datetime import datetime
from typing import List, Dict, Any


class TerminalLogicEngine:
    """Main logic engine for processing terminal commands"""
    
    def __init__(self):
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

Examples:
  add 5 3                 → 8
  subtract 10 4           → 6
  multiply 3 7            → 21
  divide 15 3             → 5
  echo Hello World        → Hello World
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