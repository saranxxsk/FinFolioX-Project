import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
    Brain, RefreshCw, Gauge, TrendingUp, TrendingDown, Minus,
    Activity, Shield, AlertTriangle, CheckCircle, XCircle, Clock
} from 'lucide-react';
import { getTrustScores, runEvaluation, getHistory } from '../services/api';

export default function MetaAgent() {
    const [scores, setScores] = useState(null);
    const [loading, setLoading] = useState(true);
    const [evaluating, setEvaluating] = useState(false);
    const [evalResult, setEvalResult] = useState(null);
    const [history, setHistory] = useState(null);
    const [error, setError] = useState('');

    const fetchScores = async () => {
        try {
            const data = await getTrustScores();
            setScores(data);
            setError('');
        } catch (err) {
            setError('Failed to load trust scores');
        } finally {
            setLoading(false);
        }
    };

    const fetchHistory = async () => {
        try {
            const data = await getHistory();
            setHistory(data);
        } catch {
            // History not available yet
        }
    };

    useEffect(() => {
        fetchScores();
        fetchHistory();
    }, []);

    const handleEvaluate = async () => {
        setEvaluating(true);
        setEvalResult(null);
        try {
            const data = await runEvaluation();
            setEvalResult(data);
            fetchScores();
            fetchHistory();
        } catch (err) {
            setError(err.response?.data?.error || 'Evaluation failed');
        } finally {
            setEvaluating(false);
        }
    };

    const agents = ['technical', 'sentiment', 'regime'];

    const getStatusBadge = (status) => {
        const s = (status || '').toUpperCase();
        if (s === 'BOOSTED') return 'badge-boosted';
        if (s === 'PENALIZED') return 'badge-penalized';
        return 'badge-normal';
    };

    const getBarColor = (val) => {
        if (val > 1.05) return 'var(--accent-green)';
        if (val < 0.95) return 'var(--accent-red)';
        return 'var(--accent-blue)';
    };

    const getIcon = (status) => {
        const s = (status || '').toUpperCase();
        if (s === 'BOOSTED') return <TrendingUp size={14} />;
        if (s === 'PENALIZED') return <TrendingDown size={14} />;
        return <Minus size={14} />;
    };

    // Calculate data sufficiency
    const evalCount = scores?.evaluation_count || 0;
    const MIN_EVALUATIONS = 10;
    const sufficiencyPct = Math.min((evalCount / MIN_EVALUATIONS) * 100, 100);
    const isSufficient = evalCount >= MIN_EVALUATIONS;

    // Derive stats from history
    const decisions = history?.decisions || [];
    const evaluatedDecisions = decisions.filter(d => d.evaluated === 'YES');
    const totalDecisions = decisions.length;
    const correctCount = evaluatedDecisions.filter(d =>
        (d.final_decision === 'BUY' && d.grade_overall === 'RIGHT') ||
        (d.final_decision === 'SELL' && d.grade_overall === 'RIGHT') ||
        (d.final_decision === 'HOLD')
    ).length;
    const accuracy = evaluatedDecisions.length > 0
        ? ((correctCount / evaluatedDecisions.length) * 100).toFixed(1)
        : '--';

    return (
        <div className="page-container">
            <h1 className="page-title">Meta-Agent Control Center</h1>
            <p className="page-subtitle">
                Self-correcting trust multipliers with temporal decay & regime-aware learning
            </p>

            {loading ? (
                <div className="loading-overlay">
                    <div className="spinner" />
                    <p className="loading-text">Loading trust scores...</p>
                </div>
            ) : error && !scores ? (
                <div className="card" style={{ borderColor: 'var(--accent-red)' }}>
                    <p style={{ color: 'var(--accent-red)' }}>{error}</p>
                </div>
            ) : scores && (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>

                    {/* Stats Row */}
                    <div className="stats-row" style={{ marginBottom: '2rem' }}>
                        <div className="stat-chip">
                            <span className="stat-chip-label">Total Decisions</span>
                            <span className="stat-chip-value" style={{ color: 'var(--accent-blue)' }}>
                                {totalDecisions}
                            </span>
                        </div>
                        <div className="stat-chip">
                            <span className="stat-chip-label">Evaluated</span>
                            <span className="stat-chip-value" style={{ color: 'var(--accent-purple)' }}>
                                {evaluatedDecisions.length}
                            </span>
                        </div>
                        <div className="stat-chip">
                            <span className="stat-chip-label">Accuracy</span>
                            <span className="stat-chip-value" style={{
                                color: accuracy !== '--' && parseFloat(accuracy) >= 60
                                    ? 'var(--accent-green)' : 'var(--accent-amber)',
                            }}>
                                {accuracy}{accuracy !== '--' ? '%' : ''}
                            </span>
                        </div>
                        <div className="stat-chip">
                            <span className="stat-chip-label">Data Sufficiency</span>
                            <span className="stat-chip-value" style={{
                                color: isSufficient ? 'var(--accent-green)' : 'var(--accent-red)',
                            }}>
                                {isSufficient ? '✓' : `${evalCount}/${MIN_EVALUATIONS}`}
                            </span>
                            <div className="progress-bar-track" style={{ width: '100%', height: '4px', marginTop: '4px' }}>
                                <div className="progress-bar-fill" style={{
                                    width: `${sufficiencyPct}%`,
                                    background: isSufficient ? 'var(--accent-green)' : 'var(--accent-amber)',
                                }} />
                            </div>
                        </div>
                    </div>

                    {/* Trust Score Cards */}
                    <div className="grid-3" style={{ marginBottom: '2rem' }}>
                        {agents.map((agent) => {
                            const val = scores[agent] || 1.0;
                            const status = scores[`${agent}_status`] || 'NORMAL';
                            const pct = ((val - 0.5) / 1.0) * 100;

                            return (
                                <div className="card" key={agent}>
                                    <div className="card-header">
                                        <Gauge size={16} />
                                        {agent.charAt(0).toUpperCase() + agent.slice(1)} Agent
                                    </div>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '1rem' }}>
                                        <span className="metric-value">{val.toFixed(3)}</span>
                                        <span className={`badge ${getStatusBadge(status)}`}>
                                            {getIcon(status)} {status}
                                        </span>
                                    </div>
                                    <div className="progress-bar-track" style={{ height: '12px' }}>
                                        <div
                                            className="progress-bar-fill"
                                            style={{
                                                width: `${Math.max(0, Math.min(100, pct))}%`,
                                                background: getBarColor(val),
                                            }}
                                        />
                                    </div>
                                    <div style={{
                                        display: 'flex', justifyContent: 'space-between',
                                        marginTop: '4px', fontSize: '0.65rem', color: 'var(--text-muted)'
                                    }}>
                                        <span>0.50 (Floor)</span>
                                        <span>1.00</span>
                                        <span>1.50 (Ceiling)</span>
                                    </div>

                                    {/* Trust Direction Indicator */}
                                    <div style={{
                                        marginTop: '10px', padding: '6px 10px',
                                        background: val > 1.0 ? 'var(--accent-green-glow)' : val < 1.0 ? 'var(--accent-red-glow)' : 'rgba(100,116,139,0.1)',
                                        borderRadius: '6px', fontSize: '0.7rem',
                                        color: val > 1.0 ? 'var(--accent-green)' : val < 1.0 ? 'var(--accent-red)' : 'var(--text-muted)',
                                        fontWeight: 600, textAlign: 'center',
                                    }}>
                                        {val > 1.05 ? '↑ Agent performing well — boosted influence'
                                            : val < 0.95 ? '↓ Agent underperforming — reduced influence'
                                                : '→ Agent within normal range'}
                                    </div>
                                </div>
                            );
                        })}
                    </div>

                    {/* Meta Info + Evaluate */}
                    <div className="grid-2" style={{ marginBottom: '2rem' }}>
                        <div className="card">
                            <div className="card-header"><Brain /> System Status</div>
                            <div className="metric" style={{ marginBottom: '0.5rem' }}>
                                <span className="metric-label">Last Updated</span>
                                <span style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
                                    {scores.last_updated || 'Never'}
                                </span>
                            </div>
                            <div className="metric" style={{ marginBottom: '0.5rem' }}>
                                <span className="metric-label">Total Evaluations</span>
                                <span style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
                                    {evalCount} decisions graded
                                </span>
                            </div>

                            {/* Data Sufficiency Warning */}
                            {!isSufficient && (
                                <div style={{
                                    marginTop: '0.75rem', padding: '8px 12px',
                                    background: 'var(--accent-amber-glow)',
                                    borderRadius: '8px', fontSize: '0.75rem',
                                    color: 'var(--accent-amber)', fontWeight: 500,
                                    display: 'flex', alignItems: 'center', gap: '6px',
                                }}>
                                    <AlertTriangle size={14} />
                                    Trust scores may be unreliable — need ≥{MIN_EVALUATIONS} evaluations for statistical significance
                                </div>
                            )}
                            {isSufficient && (
                                <div style={{
                                    marginTop: '0.75rem', padding: '8px 12px',
                                    background: 'var(--accent-green-glow)',
                                    borderRadius: '8px', fontSize: '0.75rem',
                                    color: 'var(--accent-green)', fontWeight: 500,
                                    display: 'flex', alignItems: 'center', gap: '6px',
                                }}>
                                    <CheckCircle size={14} />
                                    Sufficient data — trust scores are statistically meaningful
                                </div>
                            )}
                        </div>

                        {/* Evaluate Button */}
                        <div className="card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '1rem' }}>
                            <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', textAlign: 'center' }}>
                                Run the T+5 Hindsight Evaluator to grade past decisions and update trust scores
                            </p>
                            <button
                                className="btn btn-primary"
                                onClick={handleEvaluate}
                                disabled={evaluating}
                                style={{ minWidth: '240px' }}
                            >
                                <RefreshCw size={18} className={evaluating ? 'spin' : ''} />
                                {evaluating ? 'Evaluating...' : 'Run Hindsight Evaluation'}
                            </button>
                        </div>
                    </div>

                    {/* Evaluation Result */}
                    {evalResult && (
                        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
                            <div className="card" style={{ marginBottom: '2rem' }}>
                                <div className="card-header"><Brain /> Evaluation Result</div>
                                <p style={{ color: 'var(--accent-green)', marginBottom: '1rem', fontWeight: 600 }}>
                                    {evalResult.message}
                                </p>
                                <div className="grid-2">
                                    <div>
                                        <span className="metric-label">Trust Before</span>
                                        {Object.entries(evalResult.trust_before || {}).map(([k, v]) => (
                                            <p key={k} style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                                                {k}: {Number(v).toFixed(3)}
                                            </p>
                                        ))}
                                    </div>
                                    <div>
                                        <span className="metric-label">Trust After</span>
                                        {Object.entries(evalResult.trust_after || {}).map(([k, v]) => {
                                            const before = evalResult.trust_before?.[k] || 1.0;
                                            const delta = Number(v) - Number(before);
                                            return (
                                                <p key={k} style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                                                    {k}: {Number(v).toFixed(3)}
                                                    {delta !== 0 && (
                                                        <span style={{
                                                            marginLeft: '6px', fontSize: '0.7rem', fontWeight: 700,
                                                            color: delta > 0 ? 'var(--accent-green)' : 'var(--accent-red)',
                                                        }}>
                                                            ({delta > 0 ? '+' : ''}{delta.toFixed(3)})
                                                        </span>
                                                    )}
                                                </p>
                                            );
                                        })}
                                    </div>
                                </div>
                                <p style={{ marginTop: '1rem', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                                    Total evaluated: {evalResult.total_evaluated} decisions
                                </p>
                            </div>
                        </motion.div>
                    )}

                    {/* Recent Decision History */}
                    {decisions.length > 0 && (
                        <motion.div
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.15 }}
                        >
                            <div className="card">
                                <div className="card-header"><Activity /> Recent Decisions</div>
                                <div style={{ overflowX: 'auto' }}>
                                    <table className="data-table">
                                        <thead>
                                            <tr>
                                                <th>Ticker</th>
                                                <th>Decision</th>
                                                <th>Price</th>
                                                <th>LSTM</th>
                                                <th>Sentiment</th>
                                                <th>Regime</th>
                                                <th>Status</th>
                                                <th>Grade</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {decisions.slice(-15).reverse().map((d, idx) => (
                                                <tr key={idx}>
                                                    <td style={{ fontWeight: 600, color: 'var(--accent-blue)' }}>
                                                        {d.ticker}
                                                    </td>
                                                    <td>
                                                        <span className={`badge ${d.final_decision === 'BUY' ? 'badge-buy'
                                                                : d.final_decision === 'SELL' ? 'badge-sell'
                                                                    : 'badge-hold'
                                                            }`}>
                                                            {d.final_decision}
                                                        </span>
                                                    </td>
                                                    <td>${Number(d.price_at_decision || 0).toFixed(2)}</td>
                                                    <td>{Number(d.lstm_score || 0).toFixed(3)}</td>
                                                    <td style={{
                                                        color: (d.sent_score || 0) > 0 ? 'var(--accent-green)'
                                                            : (d.sent_score || 0) < 0 ? 'var(--accent-red)'
                                                                : 'var(--text-secondary)',
                                                    }}>
                                                        {Number(d.sent_score || 0).toFixed(3)}
                                                    </td>
                                                    <td>
                                                        <span className={`badge ${d.regime_label === 'Bull' ? 'badge-buy'
                                                                : d.regime_label === 'Bear' ? 'badge-sell'
                                                                    : 'badge-hold'
                                                            }`} style={{ fontSize: '0.6rem' }}>
                                                            {d.regime_label}
                                                        </span>
                                                    </td>
                                                    <td>
                                                        {d.evaluated === 'YES' ? (
                                                            <CheckCircle size={14} style={{ color: 'var(--accent-green)' }} />
                                                        ) : (
                                                            <Clock size={14} style={{ color: 'var(--text-muted)' }} />
                                                        )}
                                                    </td>
                                                    <td>
                                                        {d.evaluated === 'YES' && d.grade_overall ? (
                                                            <span className={`badge ${d.grade_overall === 'RIGHT' ? 'badge-buy' : 'badge-sell'}`}
                                                                style={{ fontSize: '0.6rem' }}>
                                                                {d.grade_overall === 'RIGHT' ? '✓' : '✗'} {d.grade_overall}
                                                            </span>
                                                        ) : (
                                                            <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>—</span>
                                                        )}
                                                    </td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </motion.div>
                    )}
                </motion.div>
            )}
        </div>
    );
}
