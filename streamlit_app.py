"""
Minimalist Terminal-Style Interface
A standalone terminal interface built with Streamlit
"""

import streamlit as st
from datetime import datetime
import random
import os

# Page configuration
st.set_page_config(
    page_title="Terminal Interface",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for terminal-style interface
st.markdown("""
<style>
    /* Hide Streamlit branding and menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {visibility: hidden;}
    
    /* Hide sidebar */
    .css-1d391kg {display: none;}
    section[data-testid="stSidebar"] {display: none;}
    
    /* Main container styling */
    .stApp {
        background-color: #000000 !important;
        color: #ffffff !important;
        font-family: 'Cambria', serif !important;
    }
    
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 2rem;
        padding-right: 2rem;
        background-color: #000000 !important;
        max-width: 100%;
    }
    
    /* Text input styling */
    .stTextInput > div > div > input {
        background-color: #000000 !important;
        color: #ffffff !important;
        font-family: 'Cambria', serif !important;
        font-size: 16px !important;
        border: 1px solid #333333 !important;
        padding: 10px !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #ffffff !important;
        box-shadow: none !important;
    }
    
    .stTextInput > label {
        color: #ffffff !important;
        font-family: 'Cambria', serif !important;
    }
    
    /* Text area styling */
    .stTextArea > div > div > textarea {
        background-color: #000000 !important;
        color: #ffffff !important;
        font-family: 'Cambria', serif !important;
        font-size: 16px !important;
        border: 1px solid #333333 !important;
        padding: 10px !important;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #ffffff !important;
        box-shadow: none !important;
    }
    
    .stTextArea > label {
        color: #ffffff !important;
        font-family: 'Cambria', serif !important;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #000000 !important;
        color: #ffffff !important;
        border: 1px solid #ffffff !important;
        font-family: 'Cambria', serif !important;
        font-size: 16px !important;
        padding: 0.5rem 1rem !important;
    }
    
    .stButton > button:hover {
        background-color: #333333 !important;
        border-color: #ffffff !important;
        color: #ffffff !important;
    }
    
    /* Terminal output styling */
    .terminal-output {
        background-color: #000000 !important;
        color: #ffffff !important;
        font-family: 'Cambria', serif !important;
        font-size: 16px !important;
        padding: 20px !important;
        border: 1px solid #333333 !important;
        white-space: pre-wrap !important;
        min-height: 200px !important;
        max-height: 400px !important;
        overflow-y: auto !important;
        margin-bottom: 20px !important;
    }
    
    /* Header styling */
    .terminal-header {
        color: #ffffff !important;
        font-family: 'Cambria', serif !important;
        font-size: 24px !important;
        font-weight: bold !important;
        margin-bottom: 10px !important;
    }
    
    .terminal-subtitle {
        color: #ffffff !important;
        font-family: 'Cambria', serif !important;
        font-size: 16px !important;
        margin-bottom: 30px !important;
    }
    
    /* Social links styling */
    .social-links {
        color: #ffffff !important;
        font-family: 'Cambria', serif !important;
        font-size: 16px !important;
        font-weight: bold !important;
        margin-top: 40px !important;
        padding: 20px !important;
        border-top: 1px solid #333333 !important;
    }
    
    /* Prompt styling */
    .terminal-prompt {
        color: #ffffff !important;
        font-family: 'Cambria', serif !important;
        font-size: 16px !important;
        margin-bottom: 10px !important;
    }
    
    /* Custom markdown styling */
    .stMarkdown {
        color: #ffffff !important;
        font-family: 'Cambria', serif !important;
    }
    
    /* Fix for any remaining Streamlit elements */
    div[data-testid="stMarkdownContainer"] p {
        color: #ffffff !important;
        font-family: 'Cambria', serif !important;
    }
    
    /* Column styling */
    .stColumn {
        background-color: #000000 !important;
    }
    
    /* Ensure all text is white */
    * {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'terminal_history' not in st.session_state:
    st.session_state.terminal_history = []
    st.session_state.terminal_history.append("Terminal Interface v1.0")
    st.session_state.terminal_history.append("Type 'help' for available commands")
    st.session_state.terminal_history.append("=" * 50)
    st.session_state.terminal_history.append("")

if 'command_count' not in st.session_state:
    st.session_state.command_count = 0

# Header
st.markdown('<div class="terminal-header">Terminal Interface</div>', unsafe_allow_html=True)
st.markdown('<div class="terminal-subtitle">Minimalist Code Editor Style</div>', unsafe_allow_html=True)

# Terminal output display
terminal_content = "\n".join(st.session_state.terminal_history)
st.markdown(f'<div class="terminal-output">{terminal_content}</div>', unsafe_allow_html=True)

# Command input
st.markdown('<div class="terminal-prompt">$ Enter command:</div>', unsafe_allow_html=True)

# Create columns for input and button
col1, col2 = st.columns([5, 1])

with col1:
    user_command = st.text_input(
        "",
        key="command_input",
        placeholder="Type your command here...",
        label_visibility="hidden"
    )

with col2:
    execute_button = st.button("Execute", type="primary")

# Process command
if execute_button or (user_command and st.session_state.get('last_command') != user_command):
    if user_command.strip():
        st.session_state.last_command = user_command
        st.session_state.command_count += 1
        
        # Add command to history
        st.session_state.terminal_history.append(f"$ {user_command}")
        
        # Process different commands
        command = user_command.strip().lower()
        args = user_command.strip().split()[1:] if len(user_command.strip().split()) > 1 else []
        
        if command == "help":
            st.session_state.terminal_history.extend([
                "",
                "Available commands:",
                "  help          - Show this help message",
                "  clear         - Clear terminal output",
                "  date          - Show current date and time",
                "  info          - Show system information",
                "  echo <text>   - Echo back the text",
                "  ls            - List directory contents (simulated)",
                "  pwd           - Show current directory",
                "  whoami        - Show current user",
                "  random        - Generate a random number",
                "  calc <expr>   - Simple calculator (e.g., calc 2+2)",
                "  uptime        - Show session uptime",
                ""
            ])
            
        elif command == "clear":
            st.session_state.terminal_history = []
            st.session_state.terminal_history.extend([
                "Terminal cleared.",
                "Type 'help' for available commands",
                ""
            ])
            
        elif command == "date":
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")
            st.session_state.terminal_history.extend([
                f"Current date and time: {current_time}",
                ""
            ])
            
        elif command == "info":
            st.session_state.terminal_history.extend([
                "",
                "System Information:",
                "  Platform: Streamlit Terminal Interface",
                "  Version: 1.0.0",
                f"  Commands executed: {st.session_state.command_count}",
                "  Font: Cambria",
                "  Theme: Dark Terminal",
                "  Background: Pure Black (#000000)",
                "  Text Color: White (#ffffff)",
                ""
            ])
            
        elif command.startswith("echo"):
            if args:
                text_to_echo = " ".join(args)
                st.session_state.terminal_history.extend([
                    text_to_echo,
                    ""
                ])
            else:
                st.session_state.terminal_history.extend([
                    "Usage: echo <text>",
                    ""
                ])
                
        elif command == "ls":
            st.session_state.terminal_history.extend([
                "drwxr-xr-x  streamlit_app.py",
                "drwxr-xr-x  requirements.txt", 
                "drwxr-xr-x  README.md",
                "drwxr-xr-x  .gitignore",
                ""
            ])
            
        elif command == "pwd":
            st.session_state.terminal_history.extend([
                "/terminal-interface",
                ""
            ])
            
        elif command == "whoami":
            st.session_state.terminal_history.extend([
                "terminal-user",
                ""
            ])
            
        elif command == "random":
            random_num = random.randint(1, 1000)
            st.session_state.terminal_history.extend([
                f"Random number: {random_num}",
                ""
            ])
            
        elif command.startswith("calc"):
            if args:
                try:
                    expression = " ".join(args)
                    # Simple and safe evaluation for basic math
                    allowed_chars = set('0123456789+-*/.()')
                    if all(c in allowed_chars or c.isspace() for c in expression):
                        result = eval(expression)
                        st.session_state.terminal_history.extend([
                            f"{expression} = {result}",
                            ""
                        ])
                    else:
                        st.session_state.terminal_history.extend([
                            "Error: Invalid characters in expression",
                            "Only numbers, +, -, *, /, (, ) are allowed",
                            ""
                        ])
                except:
                    st.session_state.terminal_history.extend([
                        "Error: Invalid mathematical expression",
                        "Example: calc 2+2 or calc (5*3)-1",
                        ""
                    ])
            else:
                st.session_state.terminal_history.extend([
                    "Usage: calc <expression>",
                    "Example: calc 2+2",
                    ""
                ])
                
        elif command == "uptime":
            uptime_info = f"Session commands: {st.session_state.command_count}"
            st.session_state.terminal_history.extend([
                f"Terminal uptime: {uptime_info}",
                ""
            ])
            
        elif command in ["exit", "quit"]:
            st.session_state.terminal_history.extend([
                "Cannot exit web terminal.",
                "Close browser tab to exit.",
                ""
            ])
            
        else:
            st.session_state.terminal_history.extend([
                f"Command not found: {user_command}",
                "Type 'help' for available commands.",
                ""
            ])
        
        # Rerun to update display
        st.rerun()

# Additional features section
st.markdown("---")

# Multi-line input
st.markdown('<div class="terminal-prompt">Multi-line Input:</div>', unsafe_allow_html=True)
multi_line_input = st.text_area(
    "",
    height=100,
    placeholder="Enter multiple lines of code or text here...",
    label_visibility="hidden"
)

if st.button("Process Multi-line"):
    if multi_line_input.strip():
        st.session_state.terminal_history.extend([
            "Multi-line input processed:",
            ""
        ])
        for line in multi_line_input.split('\n'):
            if line.strip():
                st.session_state.terminal_history.append(f"  {line}")
        st.session_state.terminal_history.append("")
        st.rerun()

# Clear terminal button
if st.button("Clear Terminal"):
    st.session_state.terminal_history = []
    st.session_state.terminal_history.extend([
        "Terminal cleared.",
        "Type 'help' for available commands",
        ""
    ])
    st.rerun()

# Social links footer
st.markdown("""
<div class="social-links">
<strong>Connect:</strong><br><br>
<strong>GitHub:</strong> github.com/Abhishek282001Tiwari<br>
<strong>Twitter:</strong> x.com/abhishekt282001<br>
<strong>LinkedIn:</strong> linkedin.com/in/abhishektiwari282001<br>
<strong>Email:</strong> abhishekt282001@gmail.com
</div>
""", unsafe_allow_html=True)