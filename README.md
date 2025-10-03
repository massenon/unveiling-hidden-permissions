# Unveiling Hidden Permissions: An LLM-based Analyzer

This repository contains the Python implementation of the framework described in the paper "Unveiling Hidden Permissions: An LLM Framework for Detecting Privacy and Security Concerns in AI Mobile Apps Reviews."

## Overview

The system analyzes user reviews from AI-powered mobile apps and correlates them with the permissions declared in the app's manifest. The goal is to automatically identify and prioritize "hidden permissions" declared permissions whose purpose is opaque or confusing to users, leading to privacy concerns.

## Features

- **Permission Extraction:** Uses `androguard` to parse declared permissions from Android APKs.
- **NLP Preprocessing:** Cleans and tokenizes user reviews using `spaCy`.
- **Risk Classification:** A fine-tuned RoBERTa model classifies reviews into risk categories (e.g., Unauthorized Tracking, Data Leakage).
- **Risk Scoring:** A custom algorithm calculates a final risk score to prioritize the most significant issues.

## Getting Started

### Prerequisites

- Python 3.8+
- Git

### Installation

1.  Clone this repository:
    ```bash
    git clone https://github.com/YourUsername/hidden-permissions-analyzer.git
    cd hidden-permissions-analyzer
    ```

2.  Create and activate a Python virtual environment:
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    ```

3.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    ```

## Usage

To run the main analysis pipeline with a sample review, execute:
```bash
python -m src.main_pipeline