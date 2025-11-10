
### **4. `demo.md`**
```markdown
---
layout: page
title: "Live Demo"
description: "Interactive demonstration of NeuroLogicX reasoning capabilities"
---

## Live NeuroLogicX Demo

Experience the neurosymbolic reasoning capabilities of NeuroLogicX through our interactive web application.

<div class="demo-section">
  <div class="demo-container">
    <div class="demo-description">
      <h3>Interactive Reasoning</h3>
      <p>
        The live demo allows you to input stories and questions, then see the complete 
        reasoning process including entity extraction, symbolic translation, and 
        forward-chaining inference.
      </p>
      
      <h4>Demo Features:</h4>
      <ul>
        <li>Real-time story processing</li>
        <li>Complete reasoning trace visualization</li>
        <li>Entity extraction display</li>
        <li>Confidence scoring</li>
        <li>Performance metrics</li>
      </ul>
      
      <div class="demo-examples">
        <h4>Example Queries:</h4>
        <div class="code-example">
          <strong>Story:</strong> "Mary moved to the bathroom. John went to the kitchen."<br>
          <strong>Question:</strong> "Where is Mary?"<br>
          <strong>Answer:</strong> "bathroom" (with full reasoning trace)
        </div>
      </div>
    </div>
    
    <div class="demo-action">
      <a href="{{ site.streamlit_app_url }}" class="demo-button" target="_blank">
        <span class="button-icon">🚀</span>
        <span class="button-text">Launch Live Demo</span>
      </a>
      <p class="demo-note">
        Opens in new window. Requires modern web browser.
      </p>
    </div>
  </div>
</div>

### Research Evaluation Demo

The demo also includes comprehensive evaluation capabilities:

- **System comparison** against baselines
- **Statistical significance testing**
- **Performance visualization**
- **Error analysis**

### Source Code Access

For researchers interested in the implementation:

```bash
git clone https://github.com/Abhishek282001Tiwari/NeuroLogicX
cd NeuroLogicX
pip install -r requirements.txt
streamlit run streamlit_app.py

<div class="resources"> <h3>Additional Resources</h3> <div class="resource-links"> <a href="{{ site.github_repo }}" class="resource-link" target="_blank"> GitHub Repository </a> <a href="/research" class="resource-link"> Technical Documentation </a> <a href="/results" class="resource-link"> Evaluation Results </a> </div> </div> ```

