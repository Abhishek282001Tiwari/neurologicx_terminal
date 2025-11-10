---
layout: default
title: "NeuroLogicX Research"
description: "A Modular Neurosymbolic Framework for General-Purpose Reasoning"
permalink: /
---

# NeuroLogicX Research

# Bridging Symbolic and Deep Learning for Interpretable AI

Achieving 94.2% accuracy on bAbI reasoning tasks with full explainability through neural-symbolic integration and transparent reasoning frameworks.

## Research Overview

<div class="project featured">
    <div class="project-header">
        <h3>NeuroLogicX Framework</h3>
        <span class="project-status">Research Complete</span>
    </div>
    <div class="project-meta">
        <span class="project-tech">Neuro-Symbolic AI • BERT • Forward-Chaining • Explainable AI</span>
        <span class="project-date">2024</span>
    </div>
    <p class="project-description">
        A modular neurosymbolic framework that seamlessly integrates neural perception (BERT embeddings) with symbolic reasoning (forward-chaining inference) to achieve both state-of-the-art performance and full explainability in complex reasoning tasks.
    </p>
    <div class="project-features">
        <h4>Key Research Contributions:</h4>
        <ul>
            <li>94.2% accuracy on bAbI reasoning tasks, outperforming baseline models</li>
            <li>Complete reasoning traces with symbolic transparency for full auditability</li>
            <li>Statistical significance (p < 0.05) over both neural and symbolic baselines</li>
            <li>Modular architecture enabling component-level analysis and improvement</li>
            <li>Open source implementation with comprehensive evaluation framework</li>
        </ul>
    </div>
    <div class="project-links">
        <a href="https://neurologicx.streamlit.app" class="project-link" target="_blank">Live Demo</a>
        <a href="{{ site.baseurl }}/research" class="project-link">Research Details</a>
        <a href="https://www.techrxiv.org/users/942678/articles/1316379" class="project-link" target="_blank">Read Paper</a>
    </div>
</div>

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