# AI Job Risk Predictor  
Quantifying the Impact of AI and Automation on Job Roles

Live Application: https://ai-job-risk.vercel.app/

---

## Overview

The AI Job Risk Predictor is a full-stack machine learning application that estimates the likelihood of a job role being impacted or partially replaced by AI-driven automation.

Unlike traditional risk calculators, this system uses a **Hybrid AI Architecture**:
- Large Language Models (LLMs) for semantic understanding and feature extraction from raw job descriptions.
- A trained Gradient Boosting model (CatBoost) for precise, numerical risk estimation.

This separation allows the system to handle unstructured human input while preserving the reliability, stability, and interpretability of classical machine learning.

---

## Core Idea

LLMs are used for understanding language.  
Machine learning models are used for making deterministic predictions.

This project deliberately combines both, instead of misusing LLMs as black-box predictors.

---

## Tech Stack

### Frontend
- React (Vite)
- Framer Motion
- Lucide React
- Axios

### Backend
- Python
- FastAPI
- Pydantic

### Machine Learning
- CatBoostClassifier
- Pandas
- NumPy
- Joblib

### AI Integration
- OpenRouter API
  - Nvidia Nemotron

### Deployment
- Vercel (Frontend)
- Render (Backend)

---

## Backend Architecture

The backend is designed around a **two-step inference pipeline**, a pattern commonly used in production ML systems.

---

### Step 1: `/infer-features` — LLM Enrichment Layer

**Purpose:**  
Transform raw, unstructured user input into structured, ML-ready features.

**Input:**
- Job title
- Industry
- Responsibilities
- Seniority level
- Company size

**Logic:**
- An LLM interprets the job description and infers:
  - `ai_intensity_score`
  - `industry_ai_adoption_stage`
  - `job_description_embedding_cluster`
- A robust JSON-cleaning layer removes markdown noise and malformed output.
- A model fallback mechanism automatically switches LLMs if the primary model is unavailable.

This step focuses entirely on **data enrichment**, not prediction.

---

### Step 2: `/predict` — ML Prediction Layer

**Purpose:**  
Run a deterministic machine learning model on validated, structured features.

**Input:**
- User-reviewed or overridden inferred features

**Logic:**
- Loads a pre-trained CatBoostClassifier
- Enforces strict feature alignment:
  - Correct order
  - Correct data types
  - Exact training signature

**Output:**
- Probability score representing automation risk
- Binary classification (High Risk / Low Risk)

---

## Engineering Decisions

### Schema Separation
Separate Pydantic schemas are used for:
- Feature inference
- Final prediction

This avoids schema overloading and prevents premature validation failures when data is still incomplete.

### Human-in-the-Loop Validation
Users can review and override inferred features before prediction, improving trust and accuracy.

### Reliability and Safety
- Environment-variable based secret management
- No API keys in source control
- Explicit error handling for LLM failures

### Deployment Readiness
- Health-check endpoints for liveness monitoring
- Strict feature enforcement at prediction time
- CORS configured for production frontend access

---

## System Design Summary

The system cleanly separates:
- Language understanding
- Feature inference
- Mathematical prediction
- User validation

This mirrors how real-world AI systems are designed and deployed, rather than relying on a single opaque model.

---

Live Application: https://ai-job-risk.vercel.app/
