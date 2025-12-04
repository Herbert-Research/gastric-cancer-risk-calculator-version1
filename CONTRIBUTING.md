# Contributing to Gastric Cancer Risk Calculator

Thank you for your interest in contributing to this educational project.

## Scope

This repository is primarily an educational demonstration for a PhD portfolio. 
Contributions that improve scientific rigor, reproducibility, or documentation 
are welcome.

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Herbert-Research/gastric-cancer-risk-calculator.git
   cd gastric-cancer-risk-calculator
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

4. Run tests:
   ```bash
   pytest tests/ -v
   ```
