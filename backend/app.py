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

@app.get("/health")
def health():
    return {"status": "ok"}

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

    def call_llm(model_name="google/gemma-3n-e2b-it:free"):
        # Debug: Print masked key to logs (only first 4 chars)
        key_preview = f"{OPENROUTER_API_KEY[:4]}..." if OPENROUTER_API_KEY else "None"
        print(f"DEBUG: Calling OpenRouter with model {model_name}. Key preview: {key_preview}")
        
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://ai-job-risk.vercel.app",
                    "X-Title": "AI Job Risk Predictor",
                },
                data=json.dumps({
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": "You are a technical assistant that outputs only valid JSON."},
                        {"role": "user", "content": prompt_content}
                    ],
                    "temperature": 0.1
                }),
                timeout=20
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"ERROR: OpenRouter call failed: {str(e)}")
            return {"error": str(e)}

    # Try primary model (Gemma 3 as requested in snippet)
    llm_result = call_llm("google/gemma-3n-e2b-it:free")
    
    # Fallback if primary fails
    if "error" in llm_result:
        llm_result = call_llm("google/gemma-2-9b-it:free")

    if "error" in llm_result:
        raise HTTPException(status_code=400, detail=f"LLM Provider Error: {llm_result['error']}")

    content = "No content"
    try:
        content = llm_result['choices'][0]['message']['content']
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
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Prepare data for the model. 
        # Note: We include industry and company_size if they were part of the training set.
        input_dict = {
            'ai_intensity_score': data.ai_intensity_score,
            'job_description_embedding_cluster': str(data.job_description_embedding_cluster),
            'industry_ai_adoption_stage': data.industry_ai_adoption_stage,
            'seniority_level': data.seniority_level,
            'industry': data.industry,
            'company_size': data.company_size
        }
        
        input_data = pd.DataFrame([input_dict])
        
        # Ensure we only pass features the model was trained on
        # If the model is a pipeline, it will handle filtering or we can do it here
        # For now, we pass the core features.
        
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        if isinstance(prediction, np.ndarray):
            prediction = int(prediction[0])
        else:
            prediction = int(prediction)
            
        return {
            'risk_score': round(float(probability), 4),
            'is_high_risk': bool(prediction == 1),
            'message': 'High Risk' if prediction == 1 else 'Low Risk'
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
