---
layout: page
title: "Live Demo"
description: "Interactive demonstration of NeuroLogicX reasoning capabilities"
permalink: /demo/
---

# Live Demo

# NeuroLogicX Interactive Reasoning Platform

Experience the neurosymbolic reasoning capabilities of NeuroLogicX through our interactive web application with real-time processing and complete reasoning trace visualization.

## Demo Applications

<div class="project featured">
    <div class="project-header">
        <h3>Interactive Reasoning Demo</h3>
        <span class="project-status">Live Application</span>
    </div>
    <div class="project-meta">
        <span class="project-tech">Streamlit • Python • Real-time Processing • Interactive UI</span>
        <span class="project-date">2024</span>
    </div>
    <p class="project-description">
        The live demo allows you to input stories and questions, then see the complete reasoning process including entity extraction, symbolic translation, and forward-chaining inference with full transparency.
    </p>
    <div class="project-features">
        <h4>Core Demo Features:</h4>
        <ul>
            <li>Real-time story processing with immediate reasoning results</li>
            <li>Complete reasoning trace visualization showing each inference step</li>
            <li>Interactive entity extraction and relationship mapping display</li>
            <li>Confidence scoring for each reasoning step and final answer</li>
            <li>Performance metrics and system response timing</li>
            <li>Side-by-side comparison with baseline models</li>
        </ul>
    </div>
    <div class="project-results">
        <h4>Example Query Processing:</h4>
        <ul>
            <li><strong>Story Input</strong>: "Mary moved to the bathroom. John went to the kitchen."</li>
            <li><strong>Question</strong>: "Where is Mary?"</li>
            <li><strong>Answer</strong>: "bathroom" with full reasoning trace</li>
            <li><strong>Processing Time</strong>: &lt; 2 seconds for complex reasoning chains</li>
            <li><strong>Confidence Score</strong>: 94.2% average across test queries</li>
        </ul>
    </div>
    <div class="project-links">
        <a href="{{ site.streamlit_app_url }}" class="project-link" target="_blank">Launch Live Demo</a>
        <a href="{{ site.github_repo }}" class="project-link">View Source Code</a>
        <a href="/research" class="project-link">Technical Details</a>
    </div>
</div>

<div class="project featured">
    <div class="project-header">
        <h3>Research Evaluation Suite</h3>
        <span class="project-status">Benchmarking Tools</span>
    </div>
    <div class="project-meta">
        <span class="project-tech">Statistical Analysis • Performance Metrics • Visualization</span>
        <span class="project-date">2024</span>
    </div>
    <p class="project-description">
        Comprehensive evaluation capabilities built into the demo platform, enabling researchers to compare NeuroLogicX against baseline models and conduct rigorous statistical analysis of reasoning performance.
    </p>
    <div class="project-features">
        <h4>Evaluation Features:</h4>
        <ul>
            <li>Multi-system comparison against neural and symbolic baselines</li>
            <li>Statistical significance testing with p-value calculations</li>
            <li>Interactive performance visualization across different task types</li>
            <li>Detailed error analysis with failure case examination</li>
            <li>Confidence interval reporting for all performance metrics</li>
            <li>Export capabilities for research publication data</li>
        </ul>
    </div>
    <div class="project-results">
        <h4>Benchmark Results:</h4>
        <ul>
            <li>94.2% accuracy on bAbI reasoning tasks</li>
            <li>Statistical significance: p < 0.001 vs baseline models</li>
            <li>15-20% improvement over pure neural approaches</li>
            <li>25-30% improvement over rule-based systems</li>
            <li>Consistent performance across all 20 task types</li>
        </ul>
    </div>
</div>

<div class="project featured">
    <div class="project-header">
        <h3>Local Development & Research</h3>
        <span class="project-status">Open Source Ready</span>
    </div>
    <div class="project-meta">
        <span class="project-tech">Python • Git • Streamlit • Jupyter</span>
        <span class="project-date">2024</span>
    </div>
    <p class="project-description">
        Full source code availability for researchers and developers interested in extending the NeuroLogicX framework, conducting their own experiments, or integrating the technology into their applications.
    </p>
    <div class="project-features">
        <h4>Quick Start Installation:</h4>
        <ul>
            <li>Clone repository: <code>git clone https://github.com/Abhishek282001Tiwari/NeuroLogicX</code></li>
            <li>Install dependencies: <code>pip install -r requirements.txt</code></li>
            <li>Launch demo: <code>streamlit run streamlit_app.py</code></li>
            <li>Access at: <code>http://localhost:8501</code></li>
            <li>Comprehensive documentation and example datasets included</li>
        </ul>
    </div>
    <div class="project-results">
        <h4>Development Features:</h4>
        <ul>
            <li>Modular architecture for easy extension and modification</li>
            <li>Pre-trained models and example datasets included</li>
            <li>Jupyter notebooks for experimental research</li>
            <li>Unit tests and validation scripts</li>
            <li>Docker containerization support</li>
        </ul>
    </div>
    <div class="project-links">
        <a href="https://github.com/Abhishek282001Tiwari/NeuroLogicX" class="project-link">GitHub Repository</a>
        <a href="/research" class="project-link">Research Paper</a>
        <a href="/results" class="project-link">Full Results</a>
    </div>
</div>

---

*Ready to experience neuro-symbolic reasoning in action? [Launch the demo]({{ site.streamlit_app_url }}){:target="_blank"} or [explore the code]({{ site.github_repo }}){:target="_blank"} to see how NeuroLogicX bridges neural and symbolic AI.*

<style>
.project {
    border: 1px solid #eaeaea;
    border-radius: 8px;
    padding: 2rem;
    margin-bottom: 2.5rem;
    transition: transform 0.2s, box-shadow 0.2s;
}

.project.featured {
    border-left: 4px solid #000;
    background: #fafafa;
}

.project:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

.project-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 1rem;
    flex-wrap: wrap;
    gap: 1rem;
}

.project-header h3 {
    color: #000;
    margin: 0;
    font-size: 1.4rem;
    flex: 1;
    min-width: 300px;
}

.project-status {
    background: #000;
    color: white;
    padding: 0.3rem 0.8rem;
    border-radius: 4px;
    font-size: 0.9rem;
    font-weight: 500;
    white-space: nowrap;
}

.project-meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.5rem;
    flex-wrap: wrap;
    gap: 1rem;
}

.project-tech {
    color: #666;
    font-family: 'Monaco', 'Menlo', monospace;
    font-size: 0.9rem;
}

.project-date {
    color: #888;
    font-size: 0.9rem;
}

.project-description {
    color: #333;
    line-height: 1.6;
    margin-bottom: 1.5rem;
    font-size: 1.1rem;
}

.project-features, .project-results {
    margin-bottom: 1.5rem;
}

.project-features h4, .project-results h4 {
    color: #000;
    margin-bottom: 0.5rem;
    font-size: 1.1rem;
}

.project-features ul, .project-results ul {
    list-style: none;
    padding-left: 0;
}

.project-features li, .project-results li {
    padding: 0.3rem 0;
    position: relative;
    padding-left: 1.5rem;
    color: #333;
}

.project-features li:before, .project-results li:before {
    content: "✓";
    position: absolute;
    left: 0;
    color: #4caf50;
    font-weight: bold;
}

.project-links {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
}

.project-link {
    display: inline-block;
    padding: 0.5rem 1.2rem;
    background: #000;
    color: white;
    text-decoration: none;
    border-radius: 4px;
    font-size: 0.9rem;
    transition: background 0.2s;
}

.project-link:hover {
    background: #333;
}

code {
    background: #f5f5f5;
    padding: 0.2rem 0.4rem;
    border-radius: 3px;
    font-family: 'Monaco', 'Menlo', monospace;
    font-size: 0.85rem;
    color: #d63384;
}

@media (max-width: 768px) {
    .project-header {
        flex-direction: column;
        align-items: flex-start;
    }
    
    .project-header h3 {
        min-width: auto;
    }
    
    .project-meta {
        flex-direction: column;
        align-items: flex-start;
    }
}
</style>