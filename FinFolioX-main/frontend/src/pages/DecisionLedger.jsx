import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { History, BookOpen, AlertTriangle } from 'lucide-react';
import { getHistory } from '../services/api';

export default function DecisionLedger() {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');

    useEffect(() => {
        const fetch = async () => {
            try {
                const result = await getHistory();
                // Show newest searches at the TOP
                if (result?.decisions) {
                    result.decisions = result.decisions.reverse();
                }
                setData(result);
            } catch (err) {
                setError('Failed to load decision history');
            } finally {
                setLoading(false);
            }
        };
        fetch();
    }, []);

    const getDecisionBadge = (action) => {
        if (!action) return 'badge-hold';
        const a = action.toUpperCase();
        if (a.includes('BUY')) return 'badge-buy';
        if (a.includes('SELL')) return 'badge-sell';
        return 'badge-hold';
    };

    const getGradeBadge = (grade) => {
        if (grade === 'RIGHT') return 'badge-buy';
        if (grade === 'WRONG') return 'badge-sell';
        return 'badge-normal';
    };

    return (
        <div className="page-container">
            <h1 className="page-title">Decision Lineage Ledger</h1>
            <p className="page-subtitle">
                Complete audit trail — every decision, every grade, every regret
            </p>

            {loading ? (
                <div className="loading-overlay">
                    <div className="spinner" />
                    <p className="loading-text">Loading decision history...</p>
                </div>
            ) : error ? (
                <div className="card" style={{ borderColor: 'var(--accent-red)' }}>
                    <p style={{ color: 'var(--accent-red)' }}>{error}</p>
                </div>
            ) : data && (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                    <div style={{ display: 'flex', gap: '1rem', marginBottom: '1.5rem' }}>
                        <span className="badge badge-normal">{data.count} Total Decisions</span>
                        <span className="badge badge-buy">
                            {data.decisions?.filter(d => d.evaluated === 'YES').length || 0} Evaluated
                        </span>
                        <span className="badge badge-hold">
                            {data.decisions?.filter(d => d.evaluated === 'NO').length || 0} Pending
                        </span>
                    </div>

                    {/* Main Table */}
                    <div className="card" style={{ overflow: 'auto' }}>
                        <table className="data-table">
                            <thead>
                                <tr>
                                    <th>Date</th>
                                    <th>Ticker</th>
                                    <th>Decision</th>
                                    <th>Price</th>
                                    <th>Confidence</th>
                                    <th>Status</th>
                                    <th>T+5 Price</th>
                                    <th>Change</th>
                                    <th>LSTM</th>
                                    <th>Sent</th>
                                    <th>Regime</th>
                                    <th>Optimal</th>
                                    <th>Regret</th>
                                </tr>
                            </thead>
                            <tbody>
                                {data.decisions?.map((d, i) => (
                                    <tr key={i}>
                                        <td>{d.timestamp?.split(' ')[0]}</td>
                                        <td style={{ fontWeight: 700, color: 'var(--text-primary)' }}>{d.ticker}</td>
                                        <td>
                                            <span className={`badge ${getDecisionBadge(d.final_decision)}`}>
                                                {d.final_decision}
                                            </span>
                                        </td>
                                        <td>${d.price_at_decision?.toFixed(2)}</td>
                                        <td>{d.fusion_confidence?.toFixed(4)}</td>
                                        <td>
                                            <span className={`badge ${d.evaluated === 'YES' ? 'badge-buy' : 'badge-hold'}`}>
                                                {d.evaluated}
                                            </span>
                                        </td>
                                        <td>{d.actual_price_t5 ? `$${d.actual_price_t5.toFixed(2)}` : '—'}</td>
                                        <td style={{
                                            color: d.price_change_pct > 0 ? 'var(--accent-green)' :
                                                d.price_change_pct < 0 ? 'var(--accent-red)' : 'var(--text-muted)'
                                        }}>
                                            {d.price_change_pct != null ? `${(d.price_change_pct * 100).toFixed(2)}%` : '—'}
                                        </td>
                                        <td>
                                            {d.lstm_grade ? (
                                                <span className={`badge ${getGradeBadge(d.lstm_grade)}`}>{d.lstm_grade}</span>
                                            ) : '—'}
                                        </td>
                                        <td>
                                            {d.sent_grade ? (
                                                <span className={`badge ${getGradeBadge(d.sent_grade)}`}>{d.sent_grade}</span>
                                            ) : '—'}
                                        </td>
                                        <td>
                                            {d.regime_grade ? (
                                                <span className={`badge ${getGradeBadge(d.regime_grade)}`}>{d.regime_grade}</span>
                                            ) : '—'}
                                        </td>
                                        <td>
                                            {d.optimal_decision ? (
                                                <span className={`badge ${getDecisionBadge(d.optimal_decision)}`}>
                                                    {d.optimal_decision}
                                                </span>
                                            ) : '—'}
                                        </td>
                                        <td style={{
                                            color: d.regret_score > 0.05 ? 'var(--accent-red)' :
                                                d.regret_score === 0 ? 'var(--accent-green)' : 'var(--text-muted)'
                                        }}>
                                            {d.regret_score != null ? `${(d.regret_score * 100).toFixed(2)}%` : '—'}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>

                    {/* Retrospective Section */}
                    {data.decisions?.filter(d => d.llm_retrospective).length > 0 && (
                        <div style={{ marginTop: '1.5rem' }}>
                            <h2 style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
                                <BookOpen size={18} /> AI Trader's Diary (Phase 15 Retrospectives)
                            </h2>
                            {data.decisions
                                .filter(d => d.llm_retrospective)
                                .map((d, i) => (
                                    <div className="card" key={i} style={{ marginBottom: '1rem' }}>
                                        <div style={{ display: 'flex', gap: '8px', marginBottom: '0.5rem' }}>
                                            <span className={`badge ${getDecisionBadge(d.final_decision)}`}>{d.final_decision}</span>
                                            <span className="badge badge-normal">{d.ticker}</span>
                                            <span className="badge badge-normal">{d.timestamp?.split(' ')[0]}</span>
                                        </div>
                                        <p className="summary-text">{d.llm_retrospective}</p>
                                    </div>
                                ))}
                        </div>
                    )}
                </motion.div>
            )}
        </div>
    );
}
