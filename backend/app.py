from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np
import requests
import json
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="AI Job Risk Predictor API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Relaxed for production debugging
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
MODEL_PATH = 'model/final_ai_job_replacement_model.pkl'
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Model failed to load: {e}")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=OPENROUTER_API_KEY,
)

@app.get("/")
def root():
    return {"message": "AI Job Risk Predictor API is running. Use /health for status."}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

# --- SCHEMAS ---

class JobInferenceRequest(BaseModel):
    """Step 1: Raw user input for LLM feature inference"""
    job_title: str
    industry: Optional[str] = "Not specified"
    experience_years: Optional[str] = "Not specified"
    job_responsibilities: Optional[str] = "Not specified"
    seniority_level: Optional[str] = "Mid"
    company_size: Optional[str] = "Mid-size"
    work_type: Optional[str] = "Engineering"
    ai_exposure: Optional[str] = "Medium"

class JobPredictionRequest(BaseModel):
    """Step 2: Validated ML features for final model prediction"""
    ai_intensity_score: float
    industry_ai_adoption_stage: str
    job_description_embedding_cluster: int
    seniority_level: str
    industry: Optional[str] = "Not specified"
    company_size: Optional[str] = "Not specified"

def clean_json_string(s):
    """Robustly extract JSON from LLM response, handling markdown blocks and noise."""
    try:
        # Remove markdown code blocks if present
        if "```json" in s:
            s = s.split("```json")[1].split("```")[0]
        elif "```" in s:
            s = s.split("```")[1].split("```")[0]
        
        start = s.find('{')
        end = s.rfind('}') + 1
        if start != -1 and end != 0:
            return s[start:end].strip()
    except:
        pass
    return s.strip()

@app.post("/infer-features")
def infer_features(data: JobInferenceRequest):
    if not OPENROUTER_API_KEY or len(OPENROUTER_API_KEY) < 10:
        raise HTTPException(status_code=500, detail="Invalid or missing OPENROUTER_API_KEY on server")

    prompt_content = f"""
Job Title: {data.job_title}
Industry: {data.industry}
Responsibilities: {data.job_responsibilities}
Seniority: {data.seniority_level}
 
Infer these ML features:
1. ai_intensity_score (0.0 to 1.0)
2. industry_ai_adoption_stage (Emerging, Growing, or Mature)
3. job_description_embedding_cluster (0 to 19)
 
Return ONLY a JSON object. No preamble.
{{
  "ai_intensity_score": float,
  "industry_ai_adoption_stage": "string",
  "job_description_embedding_cluster": int,
  "confidence": float
}}
"""

    def call_llm(model_name="nvidia/nemotron-3-nano-30b-a3b:free"):
        # Debug: Print masked key to logs (only first 4 chars)
        key_preview = f"{OPENROUTER_API_KEY[:4]}..." if OPENROUTER_API_KEY else "None"
        print(f"DEBUG: Calling OpenRouter with model {model_name}. Key preview: {key_preview}")
        
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a technical assistant that outputs only valid JSON."},
                    {"role": "user", "content": prompt_content}
                ],
                temperature=0.1,
                timeout=20
            )
            return {"content": completion.choices[0].message.content}
        except Exception as e:
            print(f"ERROR: OpenRouter call failed: {str(e)}")
            return {"error": str(e)}

    # Try primary model (Mistral 7B - High availability free model)
    llm_result = call_llm("nvidia/nemotron-3-nano-30b-a3b:free")
    
    # Fallback if primary fails
    if "error" in llm_result:
        llm_result = call_llm("google/gemma-2-9b-it:free")

    if "error" in llm_result:
        raise HTTPException(status_code=400, detail=f"LLM Provider Error: {llm_result['error']}")

    content = "No content"
    try:
        content = llm_result.get('content', 'No content')
        cleaned_content = clean_json_string(content)
        inferred = json.loads(cleaned_content)

        # Validation & Normalization
        inferred['ai_intensity_score'] = max(0.0, min(1.0, float(inferred.get('ai_intensity_score', 0.5))))
        inferred['job_description_embedding_cluster'] = max(0, min(19, int(inferred.get('job_description_embedding_cluster', 0))))
        
        valid_stages = ['Emerging', 'Growing', 'Mature']
        stage = str(inferred.get('industry_ai_adoption_stage', 'Growing')).capitalize()
        if stage not in valid_stages: stage = 'Growing'
        inferred['industry_ai_adoption_stage'] = stage
        
        confidence = float(inferred.get('confidence', 0.5))
        inferred['is_low_confidence'] = confidence < 0.6
        
        return inferred
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Parsing Error: {str(e)}. Raw Content: {content[:50]}...")

@app.post("/predict")
def predict(data: JobPredictionRequest):
    """
    Step 2: Takes validated ML features and returns the final risk prediction.
    This endpoint is strict and expects all required features.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="ML Model not loaded on server")

    try:
        # Prepare data for the model with EXACT column order as training
        # Features: industry, seniority_level, company_size, ai_intensity_score, 
        #           industry_ai_adoption_stage, job_description_embedding_cluster
        
        input_dict = {
            'industry': data.industry or "Not specified",
            'seniority_level': data.seniority_level,
            'company_size': data.company_size or "Not specified",
            'ai_intensity_score': float(data.ai_intensity_score),
            'industry_ai_adoption_stage': data.industry_ai_adoption_stage,
            'job_description_embedding_cluster': str(data.job_description_embedding_cluster)
        }
        
        # Create DataFrame and enforce column order
        features_order = [
            'industry', 
            'seniority_level', 
            'company_size', 
            'ai_intensity_score', 
            'industry_ai_adoption_stage', 
            'job_description_embedding_cluster'
        ]
        
        input_data = pd.DataFrame([input_dict])[features_order]
        
        # Get prediction and probability
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        # Handle different return types from different models (CatBoost vs Sklearn)
        if isinstance(prediction, (np.ndarray, list)):
            prediction = int(prediction[0])
        else:
            prediction = int(prediction)
            
        return {
            'risk_score': round(float(probability), 4),
            'is_high_risk': bool(prediction == 1),
            'message': 'High Risk' if prediction == 1 else 'Low Risk'
        }
    except Exception as e:
        print(f"PREDICTION ERROR: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
