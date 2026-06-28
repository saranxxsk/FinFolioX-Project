import axios from 'axios';

const API_BASE = 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_BASE,
  headers: { 'Content-Type': 'application/json' },
});

// POST /api/analyze/ — Run full LangGraph inference
export const analyzeStock = (ticker) =>
  api.post('/analyze/', { ticker }).then(res => res.data);

// GET /api/history/ — Fetch decision ledger
export const getHistory = () =>
  api.get('/history/').then(res => res.data);

// GET /api/trust-scores/ — Fetch trust multipliers
export const getTrustScores = () =>
  api.get('/trust-scores/').then(res => res.data);

// POST /api/evaluate/ — Trigger T+5 hindsight evaluation
export const runEvaluation = () =>
  api.post('/evaluate/').then(res => res.data);

// POST /api/simulate/ — Phase 21: Digital Twin Simulation
export const runSimulation = (params) =>
  api.post('/simulate/', params).then(res => res.data);

export default api;
