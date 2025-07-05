"""
Minimalist Terminal-Style Interface
A standalone terminal interface built with Streamlit
"""

import streamlit as st
from datetime import datetime
from logic_engine import handle_command

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
    st.session_state.terminal_history.append("Terminal AI Interface v2.0")
    st.session_state.terminal_history.append("Enhanced with AI command processing")
    st.session_state.terminal_history.append("Type 'help' for available commands")
    st.session_state.terminal_history.append("=" * 50)
    st.session_state.terminal_history.append("")

if 'command_count' not in st.session_state:
    st.session_state.command_count = 0

# Header
st.markdown('<div class="terminal-header">Terminal AI Interface</div>', unsafe_allow_html=True)
st.markdown('<div class="terminal-subtitle">Minimalist Code Editor Style with AI Commands</div>', unsafe_allow_html=True)

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
        placeholder="Type your command here (e.g., help, dog, add 5 3)...",
        label_visibility="hidden"
    )

with col2:
    execute_button = st.button("Execute", type="primary")

# Process command using logic engine
if execute_button or (user_command and st.session_state.get('last_command') != user_command):
    if user_command.strip():
        st.session_state.last_command = user_command
        st.session_state.command_count += 1
        
        # Add command to history
        st.session_state.terminal_history.append(f"$ {user_command}")
        
        # Process command using the logic engine
        try:
            response = handle_command(user_command)
            
            # Handle special responses
            if response == "CLEAR_SCREEN":
                st.session_state.terminal_history = []
                st.session_state.terminal_history.extend([
                    "Terminal cleared.",
                    "Type 'help' for available commands",
                    ""
                ])
            else:
                # Add response to history
                if response:
                    # Split multi-line responses properly
                    response_lines = response.split('\n')
                    st.session_state.terminal_history.extend(response_lines)
                else:
                    st.session_state.terminal_history.append("(no output)")
                
                st.session_state.terminal_history.append("")
                
        except Exception as e:
            st.session_state.terminal_history.extend([
                f"System Error: {str(e)}",
                "Please try again or type 'help' for available commands.",
                ""
            ])
        
        # Rerun to update display
        st.rerun()

# Additional features section
st.markdown("---")

# Multi-line input
st.markdown('<div class="terminal-prompt">Multi-line Command Input:</div>', unsafe_allow_html=True)
multi_line_input = st.text_area(
    "",
    height=100,
    placeholder="Enter multiple commands, one per line...\nExample:\ndog\nadd 5 3\ncat",
    label_visibility="hidden"
)

if st.button("Process Multi-line Commands"):
    if multi_line_input.strip():
        st.session_state.terminal_history.extend([
            "Processing multi-line commands:",
            ""
        ])
        
        # Process each line as a separate command
        for line in multi_line_input.split('\n'):
            line = line.strip()
            if line:
                st.session_state.terminal_history.append(f"$ {line}")
                
                try:
                    response = handle_command(line)
                    
                    if response == "CLEAR_SCREEN":
                        st.session_state.terminal_history = []
                        st.session_state.terminal_history.extend([
                            "Terminal cleared.",
                            "Type 'help' for available commands",
                            ""
                        ])
                        break  # Stop processing if clear is encountered
                    else:
                        if response:
                            response_lines = response.split('\n')
                            st.session_state.terminal_history.extend(response_lines)
                        st.session_state.terminal_history.append("")
                        
                except Exception as e:
                    st.session_state.terminal_history.extend([
                        f"Error processing '{line}': {str(e)}",
                        ""
                    ])
        
        st.rerun()

# Command examples
st.markdown('<div class="terminal-prompt">Quick Examples:</div>', unsafe_allow_html=True)

# Create example buttons
example_col1, example_col2, example_col3, example_col4 = st.columns(4)

with example_col1:
    if st.button("help"):
        st.session_state.terminal_history.append("$ help")
        response = handle_command("help")
        response_lines = response.split('\n')
        st.session_state.terminal_history.extend(response_lines)
        st.session_state.terminal_history.append("")
        st.rerun()

with example_col2:
    if st.button("dog"):
        st.session_state.terminal_history.append("$ dog")
        response = handle_command("dog")
        st.session_state.terminal_history.append(response)
        st.session_state.terminal_history.append("")
        st.rerun()

with example_col3:
    if st.button("add 7 3"):
        st.session_state.terminal_history.append("$ add 7 3")
        response = handle_command("add 7 3")
        st.session_state.terminal_history.append(response)
        st.session_state.terminal_history.append("")
        st.rerun()

with example_col4:
    if st.button("random"):
        st.session_state.terminal_history.append("$ random")
        response = handle_command("random")
        st.session_state.terminal_history.append(response)
        st.session_state.terminal_history.append("")
        st.rerun()

# Clear terminal button
if st.button("Clear Terminal"):
    response = handle_command("clear")
    if response == "CLEAR_SCREEN":
        st.session_state.terminal_history = []
        st.session_state.terminal_history.extend([
            "Terminal cleared.",
            "Type 'help' for available commands",
            ""
        ])
    st.rerun()

# Session info
st.markdown("---")
st.markdown(f'<div class="terminal-prompt">Session Info: Commands executed: {st.session_state.command_count}</div>', unsafe_allow_html=True)

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