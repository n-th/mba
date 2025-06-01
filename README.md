# Automated Commit Message Generation Research

This project implements a research methodology for analyzing and comparing automated commit message generation using Large Language Models (LLMs) against human-written commit messages.

## Research Methodology

The study follows these main steps:

1. **Repository Selection and Commit Extraction**
   - Select relevant repositories with significant commit history
   - Extract commit diffs and original commit messages
   - Define time intervals or commit sets for analysis

2. **Automated Message Generation**
   - Process code diffs using LLM
   - Generate descriptive commit messages
   - Include justifications for changes

3. **Comparison and Analysis**
   - Compare human vs. automated messages
   - Evaluate clarity, completeness, and coherence
   - Perform textual analysis (key terms, similarity measures)

4. **Results Discussion**
   - Analyze convergence and divergence points
   - Discuss model limitations
   - Present findings

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set up environment variables:

```bash
cp .env.example .env
# Edit .env with your configuration
```

## Usage

Run the main analysis script:

```bash
python main.py
```

## Ethical Considerations

- All analyzed commits are from public open-source repositories
- No user data collection or experimentation
- Privacy of contributors is preserved
