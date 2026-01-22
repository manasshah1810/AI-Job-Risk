import { useState } from 'react'
import axios from 'axios'
import { motion, AnimatePresence } from 'framer-motion'
import { AlertTriangle, CheckCircle, Info, Loader2, Sparkles, Edit3, ShieldCheck, ChevronDown, ChevronUp, Briefcase, BarChart3, Target, Cpu } from 'lucide-react'
import './App.css'

const API_BASE = import.meta.env.VITE_API_URL;

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
      const response = await axios.post(`${API_BASE}/infer-features`, jobInfo)
      setInferredFeatures(response.data)
      setStep(2)
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || 'An unexpected error occurred';
      setError(`Inference Error: ${msg}`);
      console.error('Full Error Object:', err);
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
        seniority_level: jobInfo.seniority_level,
        industry: jobInfo.industry,
        company_size: jobInfo.company_size
      }
      const response = await axios.post(`${API_BASE}/predict`, predictionPayload)
      setResult(response.data)
    } catch (err) {
      setError('Prediction failed.')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app-container">
      <header>
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="logo-badge"
        >
          <Cpu size={18} /> AI Intelligence Engine
        </motion.div>
        <motion.h1
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          AI Job Risk Predictor
        </motion.h1>
        <motion.p
          className="subtitle"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
        >
          Quantify the impact of automation on your career using advanced machine learning models.
        </motion.p>
      </header>

      <div className="main-content">
        <motion.div
          className="card"
          initial={{ opacity: 0, x: -30 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6 }}
        >
          <div className="step-indicator" style={{ display: 'flex', gap: '12px', marginBottom: '48px' }}>
            <div style={{ height: '6px', flex: 1, background: step >= 1 ? 'var(--c-5)' : 'var(--bg-input)', borderRadius: '100px', transition: 'var(--transition-smooth)' }} />
            <div style={{ height: '6px', flex: 1, background: step >= 2 ? 'var(--c-5)' : 'var(--bg-input)', borderRadius: '100px', transition: 'var(--transition-smooth)' }} />
            <div style={{ height: '6px', flex: 1, background: result ? 'var(--c-5)' : 'var(--bg-input)', borderRadius: '100px', transition: 'var(--transition-smooth)' }} />
          </div>

          <AnimatePresence mode="wait">
            {step === 1 ? (
              <motion.div
                key="step1"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, x: -20 }}
              >
                <h3 className="step-title"><Briefcase size={24} /> Job Information</h3>
                <p className="step-subtitle">Tell us about your current role to begin the analysis.</p>

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
                      placeholder="Describe your primary tasks and daily responsibilities..."
                      rows="4"
                    />
                  </div>

                  <div className="form-grid">
                    <div className="form-group">
                      <label>Seniority Level</label>
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

                  <button type="submit" className="submit-btn" disabled={loading}>
                    {loading ? <Loader2 className="loading-spinner" /> : <><Sparkles size={20} /> Analyze Role Details</>}
                  </button>
                </form>
              </motion.div>
            ) : (
              <motion.div
                key="step2"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                  <h3 className="step-title" style={{ marginBottom: 0 }}><Target size={24} /> Refine Analysis</h3>
                  <button onClick={() => setStep(1)} className="text-btn"><Edit3 size={16} /> Edit Info</button>
                </div>
                <p className="step-subtitle">We've inferred these details from your description. Please verify them for maximum accuracy.</p>

                {inferredFeatures?.is_low_confidence && (
                  <div className="warning-box">
                    <AlertTriangle size={20} />
                    <div>
                      <strong>Low Confidence Detection</strong>
                      <p style={{ margin: '4px 0 0 0', fontSize: '0.85rem', opacity: 0.9 }}>Your description was a bit brief. Please ensure the selections below accurately reflect your role.</p>
                    </div>
                  </div>
                )}

                <div className="form-group">
                  <label>AI Interaction Intensity</label>
                  <div className="option-grid">
                    {[
                      { label: "Minimal", val: 0.15 },
                      { label: "Moderate", val: 0.45 },
                      { label: "Significant", val: 0.70 },
                      { label: "Core Component", val: 0.90 }
                    ].map(opt => (
                      <button
                        key={opt.val}
                        type="button"
                        className={`option-btn ${inferredFeatures.ai_intensity_score === opt.val ? 'active' : ''}`}
                        onClick={() => setInferredFeatures(prev => ({ ...prev, ai_intensity_score: opt.val }))}
                      >
                        {opt.label}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="form-group">
                  <label>Industry AI Adoption Stage</label>
                  <div className="option-grid">
                    {[
                      { label: "Emerging", val: "Emerging" },
                      { label: "Growing", val: "Growing" },
                      { label: "Mature", val: "Mature" }
                    ].map(opt => (
                      <button
                        key={opt.val}
                        type="button"
                        className={`option-btn ${inferredFeatures.industry_ai_adoption_stage === opt.val ? 'active' : ''}`}
                        onClick={() => setInferredFeatures(prev => ({ ...prev, industry_ai_adoption_stage: opt.val }))}
                      >
                        {opt.label}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="form-group">
                  <label>Primary Nature of Work</label>
                  <div className="option-grid">
                    {[
                      { label: "Research", val: 0 },
                      { label: "Engineering", val: 1 },
                      { label: "Operations", val: 2 },
                      { label: "Business", val: 3 },
                      { label: "Creative", val: 4 }
                    ].map(opt => (
                      <button
                        key={opt.val}
                        type="button"
                        className={`option-btn ${inferredFeatures.job_description_embedding_cluster == opt.val ? 'active' : ''}`}
                        onClick={() => setInferredFeatures(prev => ({ ...prev, job_description_embedding_cluster: opt.val }))}
                      >
                        {opt.label}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="advanced-section">
                  <button type="button" className="advanced-toggle" onClick={() => setShowAdvanced(!showAdvanced)}>
                    {showAdvanced ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
                    Model Parameters
                  </button>
                  <AnimatePresence>
                    {showAdvanced && (
                      <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        className="advanced-content"
                      >
                        <div className="debug-item">
                          <span>Exposure Score:</span>
                          <span className="debug-val">{(inferredFeatures.ai_intensity_score * 100).toFixed(0)}%</span>
                        </div>
                        <div className="debug-item">
                          <span>Market Maturity:</span>
                          <span className="debug-val">{inferredFeatures.industry_ai_adoption_stage}</span>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>

                <button onClick={handlePredict} className="submit-btn predict-btn" disabled={loading}>
                  {loading ? <Loader2 className="loading-spinner" /> : <><ShieldCheck size={20} /> Generate Risk Report</>}
                </button>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>

        <motion.div
          className="card"
          initial={{ opacity: 0, x: 30 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          <AnimatePresence mode="wait">
            {result ? (
              <motion.div
                key="result"
                className="result-container"
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
              >
                <div className="risk-header">
                  <BarChart3 size={20} className="info-icon" />
                  <span className="risk-title">Displacement Analysis</span>
                  <div className="tooltip-container">
                    <Info size={16} className="info-icon" />
                    <div className="tooltip-text">
                      This score is derived from historical automation trends and current AI capability benchmarks for similar roles.
                    </div>
                  </div>
                </div>

                <div className="risk-gauge">
                  <svg viewBox="0 0 100 100" style={{ transform: 'rotate(-90deg)' }}>
                    <circle cx="50" cy="50" r="45" fill="none" stroke="rgba(216, 243, 220, 0.05)" strokeWidth="6" />
                    <motion.circle
                      cx="50" cy="50" r="45" fill="none"
                      stroke={result.risk_score >= 0.5 ? 'var(--risk-high)' : result.risk_score >= 0.2 ? 'var(--risk-mid)' : 'var(--risk-low)'}
                      strokeWidth="8"
                      strokeLinecap="round"
                      strokeDasharray="283"
                      initial={{ strokeDashoffset: 283 }}
                      animate={{ strokeDashoffset: 283 - (283 * result.risk_score) }}
                      transition={{ duration: 2, ease: "easeOut" }}
                    />
                  </svg>
                  <div className={`risk-value ${result.risk_score >= 0.5 ? 'high-risk' : result.risk_score >= 0.2 ? 'moderate-risk' : 'low-risk'}`}>
                    {Math.round(result.risk_score * 100)}%
                  </div>
                </div>

                <div className={`risk-label ${result.risk_score >= 0.5 ? 'high-risk' : result.risk_score >= 0.2 ? 'moderate-risk' : 'low-risk'}`}>
                  {result.risk_score >= 0.5 ? "High Risk" : result.risk_score >= 0.2 ? "Moderate Risk" : "Low Risk"}
                </div>

                <p className="risk-helper">
                  Likelihood of significant task automation within the next 3-5 years.
                </p>

                <div className="result-desc">
                  {result.risk_score >= 0.5
                    ? "Your role involves a high volume of tasks that align with current AI capabilities. Focus on developing high-level strategic and interpersonal skills."
                    : result.risk_score >= 0.2
                      ? "While some tasks may be automated, your role likely requires human intuition and complex decision-making that AI currently lacks."
                      : "Your role is highly resilient to current automation trends, relying on skills that are difficult for AI to replicate effectively."}
                </div>

                <button onClick={() => { setResult(null); setStep(1) }} className="reset-btn">Start New Analysis</button>
              </motion.div>
            ) : (
              <div className="result-container">
                <div className="empty-state">
                  <div className="pulse-icon"><BarChart3 size={48} /></div>
                  <h3 style={{ color: 'var(--c-1)', marginBottom: '8px' }}>Awaiting Data</h3>
                  <p style={{ maxWidth: '250px' }}>Complete the role assessment on the left to generate your personalized risk report.</p>
                </div>
              </div>
            )}
          </AnimatePresence>

          {error && (
            <motion.div
              className="error-box"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <AlertTriangle size={20} /> {error}
            </motion.div>
          )}
        </motion.div>
      </div>
    </div>
  )
}

export default App
