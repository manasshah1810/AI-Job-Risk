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
    allow_origins=[
        "http://localhost:5173",
        "https://ai-job-risk.vercel.app"
    ],
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

class JobInfo(BaseModel):
    job_title: str
    industry: Optional[str] = "Not specified"
    experience_years: Optional[str] = "Not specified"
    job_responsibilities: Optional[str] = "Not specified"
    seniority_level: str
    company_size: str
    work_type: str
    ai_exposure: str

class PredictionInput(BaseModel):
    ai_intensity_score: float
    job_description_embedding_cluster: int
    industry_ai_adoption_stage: str
    seniority_level: str

def clean_json_string(s):
    try:
        start = s.find('{')
        end = s.rfind('}') + 1
        if start != -1 and end != 0:
            return s[start:end]
    except:
        pass
    return s

@app.post("/infer-features")
def infer_features(data: JobInfo):
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OpenRouter API Key not configured")

    prompt_content = f"""
Job Title: {data.job_title}
Industry: {data.industry}
Experience (years): {data.experience_years}
Responsibilities: {data.job_responsibilities}
Seniority Level: {data.seniority_level}
Company Size: {data.company_size}
Primary Work Type: {data.work_type}
AI Exposure: {data.ai_exposure}
 
Infer the following machine-learning features:
- ai_intensity_score (float between 0 and 1)
- industry_ai_adoption_stage (Emerging / Growing / Mature)
- job_description_embedding_cluster (integer between 0 and 19)
 
Output only JSON with these fields plus an overall confidence score (0–1).
"""

    def call_llm():
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://ai-job-risk.vercel.app",
                "X-Title": "AI Job Risk Predictor",
            },
            data=json.dumps({
                "model": "allenai/molmo-2-8b:free",
                "messages": [
                    {"role": "system", "content": "You convert job descriptions into structured ML features. Output JSON only. No explanations."},
                    {"role": "user", "content": prompt_content}
                ]
            }),
            timeout=30
        )
        return response.json()

    try:
        llm_result = call_llm()
        content = llm_result['choices'][0]['message']['content']
        
        try:
            inferred = json.loads(clean_json_string(content))
        except:
            llm_result = call_llm()
            content = llm_result['choices'][0]['message']['content']
            inferred = json.loads(clean_json_string(content))

        # Validation
        inferred['ai_intensity_score'] = max(0.0, min(1.0, float(inferred.get('ai_intensity_score', 0.5))))
        inferred['job_description_embedding_cluster'] = max(0, min(19, int(inferred.get('job_description_embedding_cluster', 0))))
        
        valid_stages = ['Emerging', 'Growing', 'Mature']
        stage = str(inferred.get('industry_ai_adoption_stage', 'Growing')).capitalize()
        if stage not in valid_stages:
            stage = 'Growing'
        inferred['industry_ai_adoption_stage'] = stage
        
        confidence = float(inferred.get('confidence', 0.5))
        inferred['is_low_confidence'] = confidence < 0.6
        
        return inferred
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict")
def predict(data: PredictionInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        input_data = pd.DataFrame([{
            'ai_intensity_score': data.ai_intensity_score,
            'job_description_embedding_cluster': str(data.job_description_embedding_cluster),
            'industry_ai_adoption_stage': data.industry_ai_adoption_stage,
            'seniority_level': data.seniority_level
        }])
        
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
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
