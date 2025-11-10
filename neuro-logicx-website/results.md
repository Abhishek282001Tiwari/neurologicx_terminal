---
layout: page
title: "Research Results"
description: "Comprehensive evaluation results and statistical analysis"
---

## Experimental Results

Comprehensive evaluation of NeuroLogicX against state-of-the-art baselines on bAbI reasoning tasks.

### Performance Summary

| System | Accuracy | F1-Score | Avg Confidence | Response Time |
|--------|----------|----------|----------------|---------------|
| **NeuroLogicX** | **94.2%** | 0.941 | 0.891 | 0.152s |
| Rule-Based Baseline | 91.1% | 0.911 | 0.950 | 0.045s |
| BERT Baseline | 87.3% | 0.873 | 0.734 | 0.089s |

### Statistical Significance

All performance improvements are statistically significant:

- **NeuroLogicX vs BERT Baseline**: p = 0.0034 (**)
- **NeuroLogicX vs Rule-Based**: p = 0.0182 (*)
- **BERT vs Rule-Based**: p = 0.0071 (**)

*Note: *p < 0.05, **p < 0.01*

### Performance by Task Difficulty

<div class="results-chart">
  <div id="difficultyChart"></div>
</div>

| System | Easy Tasks | Medium Tasks | Hard Tasks |
|--------|------------|--------------|------------|
| **NeuroLogicX** | **96.7%** | **93.1%** | **89.4%** |
| Rule-Based | 95.0% | 89.7% | 82.1% |
| BERT Baseline | 91.7% | 86.2% | 78.9% |

### Confidence Analysis

NeuroLogicX demonstrates superior confidence calibration:

| System | Calibration Score | Interpretation |
|--------|-------------------|----------------|
| **NeuroLogicX** | **0.847** | Excellent |
| Rule-Based | 0.000 | Overconfident |
| BERT Baseline | 0.423 | Moderate |

### Error Analysis

Detailed error categorization reveals system strengths:

| System | Total Errors | Primary Error Source |
|--------|--------------|---------------------|
| **NeuroLogicX** | **12** | Reasoning (42%) |
| Rule-Based | 18 | Pattern Matching (67%) |
| BERT Baseline | 25 | Context Understanding (60%) |

### Research Implications

1. **Effective Integration**: NeuroLogicX successfully bridges neural and symbolic AI
2. **Complex Reasoning**: Superior performance on difficult tasks demonstrates robust capabilities
3. **Explainability**: Full transparency without performance trade-offs
4. **Uncertainty Awareness**: Excellent confidence calibration for reliable deployment

### Interactive Results Explorer

<details>
<summary>View Detailed Results Data</summary>
<div class="interactive-results">
  <div id="resultsExplorer"></div>
</div>
</details>

[Download Complete Results PDF](/assets/neuro_logicx_results.pdf){: .btn .btn-outline }
[View Live Evaluation]({{ site.streamlit_app_url }}){: .btn .btn-primary }