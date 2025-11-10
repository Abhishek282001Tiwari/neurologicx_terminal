# 🧠 NeuroLogicX: Neurosymbolic AI Research Platform

> **A Modular Neurosymbolic Framework for General-Purpose Reasoning: Bridging Symbolic and Deep Learning for Interpretable AI**

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

## 📖 Research Paper

**Published Preprint**: [TechRxiv](https://www.techrxiv.org/users/942678/articles/1316379-a-modular-neurosymbolic-framework-for-general-purpose-reasoning-bridging-symbolic-and-deep-learning-for-interpretable-ai)

**Citation**:
```bibtex
@article{tiwari2025modular,
  title={A Modular Neurosymbolic Framework for General-Purpose Reasoning: Bridging Symbolic and Deep Learning for Interpretable AI},
  author={Tiwari, Abhishek Pankaj},
  journal={TechRxiv},
  year={2025},
  month={July},
  day={19},
  doi={10.36227/techrxiv.175295182.20276969/v1}
}

# Clone repository
git clone https://github.com/Abhishek282001Tiwari/NeuroLogicX
cd NeuroLogicX

# Install dependencies
pip install -r requirements.txt

# Launch the interface
streamlit run streamlit_app.py

Research Overview

NeuroLogicX represents a groundbreaking approach to artificial intelligence that seamlessly integrates neural networks with symbolic reasoning, achieving 94.2% accuracy on bAbI reasoning tasks while maintaining full explainability.

Key Research Contributions

Aspect	Traditional AI	NeuroLogicX
Accuracy	87.3% (BERT)	94.2% ✅
Explainability	Black-box	Full reasoning traces ✅
Reasoning Depth	Single-hop	Multi-hop temporal ✅
Confidence Calibration	Poor (0.423)	Excellent (0.847) ✅

Architecture

graph TB
    A[Input Text] --> B[Neural Perception<br/>BERT Embeddings]
    B --> C[Symbolic Translation<br/>Entity Extraction]
    C --> D[Forward Chaining<br/>Reasoning Engine]
    D --> E[Explainable Output<br/>Reasoning Traces]
    E --> F[Final Answer + Confidence]

    Core Components

🧠 Neural Perception Module

BERT-based entity recognition
Semantic role labeling
Neural embeddings for understanding
⚡ Symbolic Reasoning Engine

Forward-chaining inference
Logical rule application
Temporal reasoning capabilities
🔄 Neural-Symbolic Translator

Bridges neural and symbolic representations
Maintains explainability while leveraging neural power

Experimental Results

Performance Comparison

System	Accuracy	Confidence	Response Time	Rank
NeuroLogicX	94.2%	0.891	0.152s	1
Rule-Based Baseline	91.1%	0.950	0.045s	2
BERT Baseline	87.3%	0.734	0.089s	3
Statistical Significance

All improvements are statistically significant:

NeuroLogicX vs BERT: p = 0.0034 (**)
NeuroLogicX vs Rule-Based: p = 0.0182 (*)

💻 Terminal Interface Features

🎨 Design Philosophy

Pure black background - Optimal for extended research sessions
Cambria typography - Enhanced readability for technical content
Minimalist interface - Focus on cognitive tasks without distractions

🔧 Available Commands

# Research & Evaluation
demo                    # Run bAbI reasoning demonstration
story <text>           # Process story for reasoning
reason <question>      # Answer question about loaded story
evaluate               # Run comprehensive evaluation
neural_status          # Check system status

# Basic Operations
help                   # Show available commands
clear                  # Clear terminal output
date                   # Show current date and time
echo <text>            # Echo back provided text

🧪 Example Research Session

$ story Mary moved to the bathroom. John went to the hallway.
📖 Story loaded (2 sentences):
  1. Mary moved to the bathroom.
  2. John went to the hallway.

$ reason Where is Mary?
🔍 Neural-Symbolic Reasoning Result
Question: Where is Mary?
Answer: bathroom
Confidence: 0.92

🧮 Reasoning trace:
  1. Added fact: moved(mary, bathroom)
  2. Added fact: went(john, hallway)
  3. Applied rule: IF moved(X, Y) THEN at(X, Y)
  4. Concluded that mary is at bathroom

📁 Project Structure

NeuroLogicX/
├── 📄 streamlit_app.py          # Main web interface
├── 🧠 logic_engine.py           # Core neurosymbolic engine
├── 📊 evaluation.py             # Comprehensive evaluation pipeline
├── 📈 results/                  # Experimental results
│   ├── main_results.tex         # LaTeX table for paper
│   ├── significance_tests.tex   # Statistical analysis
│   └── paper_evidence.json      # Research evidence
├── 📋 requirements.txt          # Dependencies
└── 📖 README.md                 # This file

🛠️ Installation & Development

Prerequisites

Python 3.8+
4GB RAM minimum
Modern web browser

Detailed Setup

# 1. Create virtual environment
python -m venv neurologicx_env
source neurologicx_env/bin/activate  # Windows: neurologicx_env\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch research platform
streamlit run streamlit_app.py

# 4. Access at http://localhost:8501

Dependencies
streamlit>=1.25.0
transformers>=4.30.0
torch>=2.0.0
sentence-transformers>=2.2.0
numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0

📚 Research Extensions

Current Capabilities

✅ bAbI-style reasoning tasks
✅ Multi-hop temporal reasoning
✅ Explainable AI with full traces
✅ Confidence calibration
✅ Neural-symbolic integration
Planned Extensions

🔄 Larger-scale knowledge bases
🔄 Real-world dialog systems
🔄 Multi-modal reasoning
🔄 Distributed reasoning capabilities
👨‍🔬 Research Team

Principal Investigator: Abhishek Pankaj Tiwari

Connect

📧 Email: abhishekt282001@gmail.com
💼 LinkedIn: abhishektiwari282001
🐦 Twitter: abhishekt282001
🔬 GitHub: Abhishek282001Tiwari
📄 Citation & Attribution

If you use NeuroLogicX in your research, please cite:
@article{tiwari2025modular,
  title={A Modular Neurosymbolic Framework for General-Purpose Reasoning: Bridging Symbolic and Deep Learning for Interpretable AI},
  author={Tiwari, Abhishek Pankaj},
  journal={TechRxiv},
  year={2025},
  doi={10.36227/techrxiv.175295182.20276969/v1}
}

📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

🤝 Contributing

We welcome research collaborations and contributions! Please see our Contributing Guidelines for details.

Fork the repository
Create your research branch (git checkout -b research/feature)
Commit your changes (git commit -m 'Add research feature')
Push to the branch (git push origin research/feature)
Open a Pull Request

