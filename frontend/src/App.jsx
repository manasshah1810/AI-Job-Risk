import { useState } from 'react'
import axios from 'axios'
import { motion, AnimatePresence } from 'framer-motion'
import { AlertTriangle, CheckCircle, Info, Loader2, Sparkles, Edit3, ShieldCheck, ChevronDown, ChevronUp } from 'lucide-react'
import './App.css'

function App() {
  // Step 1: User Input Data
  const [jobInfo, setJobInfo] = useState({
    job_title: '',
    industry: '',
    experience_years: '',
    job_responsibilities: '',
    seniority_level: 'Mid',
    company_size: 'Mid-size',
    work_type: 'Engineering',
    ai_exposure: 'Medium'
  })

  // Step 2: Inferred Features (Internal State)
  const [inferredFeatures, setInferredFeatures] = useState(null)
  const [showAdvanced, setShowAdvanced] = useState(false)

  const [step, setStep] = useState(1) // 1: Input, 2: Review & Predict
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleInputChange = (e) => {
    const { name, value } = e.target
    setJobInfo(prev => ({ ...prev, [name]: value }))
  }

  const handleInfer = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    try {
      const response = await axios.post('http://localhost:5000/infer-features', jobInfo)
      setInferredFeatures(response.data)
      setStep(2)
    } catch (err) {
      setError('Failed to infer features. Ensure backend is running and API key is valid.')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const handlePredict = async () => {
    setLoading(true)
    setError(null)
    try {
      const predictionPayload = {
        ...inferredFeatures,
        seniority_level: jobInfo.seniority_level
      }
      const response = await axios.post('http://localhost:5000/predict', predictionPayload)
      setResult(response.data)
    } catch (err) {
      setError('Prediction failed.')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  // Mapping functions for UI to ML features
  const getAIExposureLabel = (score) => {
    if (score <= 0.3) return "None or very little";
    if (score <= 0.6) return "Some parts of my work";
    if (score <= 0.85) return "A large part of my work";
    return "Almost all of my work";
  }

  const getMaturityLabel = (stage) => {
    const mapping = { 'Emerging': 'Rare or experimental', 'Growing': 'Used by leading companies', 'Mature': 'Standard across most companies' };
    return mapping[stage] || 'Used by leading companies';
  }

  const getWorkTypeLabel = (cluster) => {
    const mapping = { 0: 'Research & experimentation', 1: 'Software / ML engineering', 2: 'Operations / monitoring', 3: 'Business / coordination', 4: 'Creative / design' };
    return mapping[cluster] || 'Operations / monitoring';
  }

  return (
    <div className="app-container">
      <header>
        <motion.div
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          className="logo-badge"
        >
          <Sparkles size={20} /> AI-Powered
        </motion.div>
        <motion.h1
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          AI Job Displacement Predictor
        </motion.h1>
        <p className="subtitle">Assess how automation trends may impact your role</p>
      </header>

      <div className="main-content">
        <motion.div
          className="card"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
        >
          <AnimatePresence mode="wait">
            {step === 1 ? (
              <motion.div
                key="step1"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0, x: -20 }}
              >
                <h3 className="step-title">Step 1: Job Information</h3>
                <form onSubmit={handleInfer}>
                  <div className="form-grid">
                    <div className="form-group">
                      <label>Job Title</label>
                      <input required name="job_title" value={jobInfo.job_title} onChange={handleInputChange} placeholder="e.g. Software Engineer" />
                    </div>
                    <div className="form-group">
                      <label>Industry</label>
                      <input name="industry" value={jobInfo.industry} onChange={handleInputChange} placeholder="e.g. Fintech" />
                    </div>
                  </div>

                  <div className="form-group">
                    <label>Job Responsibilities</label>
                    <textarea
                      name="job_responsibilities"
                      value={jobInfo.job_responsibilities}
                      onChange={handleInputChange}
                      placeholder="Describe your daily tasks..."
                      rows="3"
                    />
                  </div>

                  <div className="form-grid">
                    <div className="form-group">
                      <label>Seniority</label>
                      <select name="seniority_level" value={jobInfo.seniority_level} onChange={handleInputChange}>
                        {['Intern', 'Junior', 'Mid', 'Senior', 'Lead'].map(opt => <option key={opt} value={opt}>{opt}</option>)}
                      </select>
                    </div>
                    <div className="form-group">
                      <label>Company Size</label>
                      <select name="company_size" value={jobInfo.company_size} onChange={handleInputChange}>
                        {['Startup', 'Mid-size', 'Enterprise'].map(opt => <option key={opt} value={opt}>{opt}</option>)}
                      </select>
                    </div>
                  </div>

                  <div className="form-grid">
                    <div className="form-group">
                      <label>Work Type</label>
                      <select name="work_type" value={jobInfo.work_type} onChange={handleInputChange}>
                        {['Research', 'Engineering', 'Operations', 'Support', 'Creative'].map(opt => <option key={opt} value={opt}>{opt}</option>)}
                      </select>
                    </div>
                    <div className="form-group">
                      <label>AI Exposure</label>
                      <select name="ai_exposure" value={jobInfo.ai_exposure} onChange={handleInputChange}>
                        {['Low', 'Medium', 'High'].map(opt => <option key={opt} value={opt}>{opt}</option>)}
                      </select>
                    </div>
                  </div>

                  <button type="submit" className="submit-btn" disabled={loading}>
                    {loading ? <Loader2 className="loading-spinner" /> : <><Sparkles size={18} /> Infer Features</>}
                  </button>
                </form>
              </motion.div>
            ) : (
              <motion.div
                key="step2"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                  <h3 className="step-title" style={{ marginBottom: 0 }}>Step 2: Confirm Your Role Details</h3>
                  <button onClick={() => setStep(1)} className="text-btn"><Edit3 size={14} /> Edit Info</button>
                </div>
                <p className="step-subtitle">We interpreted your job based on your input. Please confirm or adjust the details below to improve accuracy.</p>

                {inferredFeatures?.is_low_confidence && (
                  <div className="warning-box">
                    <AlertTriangle size={16} /> ⚠️ We’re less confident because your job description was brief or ambiguous. Please confirm the details below to improve accuracy.
                  </div>
                )}

                <div className="form-group">
                  <label>How much of your daily work involves interacting with AI systems or automation?</label>
                  <div className="option-grid">
                    {[
                      { label: "None or very little", val: 0.15 },
                      { label: "Some parts of my work", val: 0.45 },
                      { label: "A large part of my work", val: 0.70 },
                      { label: "Almost all of my work", val: 0.90 }
                    ].map(opt => (
                      <button
                        key={opt.val}
                        className={`option-btn ${inferredFeatures.ai_intensity_score === opt.val ? 'active' : ''}`}
                        onClick={() => setInferredFeatures(prev => ({ ...prev, ai_intensity_score: opt.val }))}
                      >
                        {opt.label}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="form-group">
                  <label>How widely is AI used in your industry today?</label>
                  <div className="option-grid">
                    {[
                      { label: "Rare or experimental", val: "Emerging" },
                      { label: "Used by leading companies", val: "Growing" },
                      { label: "Standard across most companies", val: "Mature" }
                    ].map(opt => (
                      <button
                        key={opt.val}
                        className={`option-btn ${inferredFeatures.industry_ai_adoption_stage === opt.val ? 'active' : ''}`}
                        onClick={() => setInferredFeatures(prev => ({ ...prev, industry_ai_adoption_stage: opt.val }))}
                      >
                        {opt.label}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="form-group">
                  <label>Which best describes the main nature of your work?</label>
                  <div className="option-grid">
                    {[
                      { label: "Research & experimentation", val: 0 },
                      { label: "Software / ML engineering", val: 1 },
                      { label: "Operations / monitoring", val: 2 },
                      { label: "Business / coordination", val: 3 },
                      { label: "Creative / design", val: 4 }
                    ].map(opt => (
                      <button
                        key={opt.val}
                        className={`option-btn ${inferredFeatures.job_description_embedding_cluster == opt.val ? 'active' : ''}`}
                        onClick={() => setInferredFeatures(prev => ({ ...prev, job_description_embedding_cluster: opt.val }))}
                      >
                        {opt.label}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="advanced-section">
                  <button className="advanced-toggle" onClick={() => setShowAdvanced(!showAdvanced)}>
                    Advanced (optional) {showAdvanced ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                  </button>
                  {showAdvanced && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      className="advanced-content"
                    >
                      <div className="debug-item">
                        <span>AI exposure level:</span>
                        <span className="debug-val">{inferredFeatures.ai_intensity_score > 0.7 ? 'High' : inferredFeatures.ai_intensity_score > 0.3 ? 'Medium' : 'Low'}</span>
                      </div>
                      <div className="debug-item">
                        <span>Industry AI maturity:</span>
                        <span className="debug-val">{inferredFeatures.industry_ai_adoption_stage}</span>
                      </div>
                    </motion.div>
                  )}
                </div>

                <button onClick={handlePredict} className="submit-btn predict-btn" disabled={loading}>
                  {loading ? <Loader2 className="loading-spinner" /> : <><ShieldCheck size={18} /> Assess My Job Risk</>}
                </button>
                <p className="helper-text">Your answers will be used to assess how automation trends may impact similar roles.</p>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>

        <motion.div
          className="card"
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
        >
          <AnimatePresence mode="wait">
            {result ? (
              <motion.div
                key="result"
                className="result-container"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
              >
                <div className="risk-header">
                  <span className="risk-title">AI Displacement Risk</span>
                  <div className="tooltip-container">
                    <Info size={16} className="info-icon" />
                    <div className="tooltip-text">
                      This is not a prediction about you personally. It reflects patterns observed across similar jobs in historical data.
                    </div>
                  </div>
                </div>

                <div className="risk-gauge">
                  <svg viewBox="0 0 100 100" style={{ transform: 'rotate(-90deg)' }}>
                    <circle cx="50" cy="50" r="45" fill="none" stroke="rgba(255,255,255,0.1)" strokeWidth="8" />
                    <motion.circle
                      cx="50" cy="50" r="45" fill="none"
                      stroke={result.risk_score >= 0.5 ? '#ef4444' : result.risk_score >= 0.2 ? '#f59e0b' : '#22c55e'}
                      strokeWidth="8"
                      strokeDasharray="283"
                      initial={{ strokeDashoffset: 283 }}
                      animate={{ strokeDashoffset: 283 - (283 * result.risk_score) }}
                      transition={{ duration: 1.5, ease: "easeOut" }}
                    />
                  </svg>
                  <div className={`risk-value ${result.risk_score >= 0.5 ? 'high-risk' : result.risk_score >= 0.2 ? 'moderate-risk' : 'low-risk'}`}>
                    {Math.round(result.risk_score * 100)}%
                  </div>
                </div>

                <div className={`risk-label ${result.risk_score >= 0.5 ? 'high-risk' : result.risk_score >= 0.2 ? 'moderate-risk' : 'low-risk'}`}>
                  {result.risk_score >= 0.5 ? "High Displacement Risk" : result.risk_score >= 0.2 ? "Moderate Displacement Risk" : "Low Displacement Risk"}
                </div>

                <p className="risk-helper">
                  This represents the estimated likelihood that similar roles may be impacted by AI-driven automation. Lower values indicate lower risk.
                </p>

                <p className="result-desc">
                  {result.risk_score >= 0.5
                    ? "Many tasks in similar roles are susceptible to automation based on current AI trends."
                    : result.risk_score >= 0.2
                      ? "Some aspects of similar roles may be impacted by AI, but human decision-making remains important."
                      : "Roles like yours typically involve skills that are harder to automate with current AI systems."}
                </p>

                <button onClick={() => { setResult(null); setStep(1) }} className="reset-btn">New Analysis</button>
              </motion.div>
            ) : (
              <div className="result-container">
                <div className="empty-state">
                  <div className="pulse-icon"><Info size={32} /></div>
                  <p>Complete Step 1 and 2 to generate your AI Risk Report.</p>
                </div>
              </div>
            )}
          </AnimatePresence>

          {error && (
            <div className="error-box">
              <AlertTriangle size={16} /> {error}
            </div>
          )}
        </motion.div>
      </div>
    </div>
  )
}

export default App
