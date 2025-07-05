# Terminal Interface - Minimalist Streamlit App

A minimalist terminal-style web interface built with Streamlit, featuring a pure black background and white Cambria font for a clean code editor experience.

## Features

- **Pure black background** with white text
- **Cambria font** throughout the interface
- **Terminal-style command interface** with command history
- **Multi-line input support** for code/text processing
- **Minimal design** - no emojis, icons, or unnecessary UI elements
- **Built-in commands**: help, clear, date, info, echo
- **Social links** in clean text format

## Project Structure

```
terminal-streamlit-app/
├── streamlit_app.py    # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── .gitignore         # Git ignore rules
```

## Local Development

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation & Running

1. **Clone or download this project**
   ```bash
   git clone <your-repo-url>
   cd terminal-streamlit-app
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If not, navigate to that URL manually

## Available Commands

Once the app is running, try these terminal commands:

- `help` - Show available commands
- `clear` - Clear terminal output
- `date` - Display current date and time
- `info` - Show system information
- `echo <text>` - Echo back the provided text
- `exit` or `quit` - Shows exit message (web terminal cannot be closed)

## Deployment via Streamlit Cloud

### Step 1: Push to GitHub

1. **Create a new GitHub repository**
2. **Push your code**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Terminal-style Streamlit app"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git push -u origin main
   ```

### Step 2: Deploy on Streamlit Cloud

1. **Go to [share.streamlit.io](https://share.streamlit.io)**
2. **Sign in with your GitHub account**
3. **Click "New app"**
4. **Select your repository**: `YOUR_USERNAME/YOUR_REPO_NAME`
5. **Main file path**: `streamlit_app.py`
6. **Click "Deploy"**

### Step 3: Access Your Live App

Your app will be available at:
`https://YOUR_REPO_NAME.streamlit.app`

## Customization

### Modifying the Interface

- **Colors**: Edit the CSS in `streamlit_app.py` under the `st.markdown()` section
- **Font**: Change `'Cambria'` to any web-safe font family
- **Commands**: Add new command handlers in the command processing section
- **Social Links**: Update the footer section with your information

### Adding New Commands

To add a new command, edit the command processing section in `streamlit_app.py`:

```python
elif command == "your_command":
    st.session_state.terminal_history.append("Your response here")
```

## Technical Details

- **Framework**: Streamlit 1.25.0+
- **Styling**: Custom CSS for terminal appearance
- **State Management**: Streamlit session state for command history
- **Responsive**: Works on desktop and mobile browsers
- **No external dependencies**: Only requires Streamlit

## Browser Compatibility

- ✅ Chrome (recommended)
- ✅ Firefox
- ✅ Safari
- ✅ Edge

## Contact & Social Links

- **GitHub**: [github.com/Abhishek282001Tiwari](https://github.com/Abhishek282001Tiwari)
- **Twitter**: [x.com/abhishekt282001](https://x.com/abhishekt282001)
- **LinkedIn**: [linkedin.com/in/abhishektiwari282001](https://linkedin.com/in/abhishektiwari282001)
- **Email**: abhishekt282001@gmail.com

## License

This project is open source and available under the [MIT License](LICENSE).

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

**Built with ❤️ using Streamlit**