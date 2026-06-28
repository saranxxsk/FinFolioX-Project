import { useState } from 'react';
import { motion } from 'framer-motion';
import {
    Play, Pause, Zap, TrendingUp, TrendingDown, Activity,
    BarChart3, AlertTriangle, Settings, FlaskConical
} from 'lucide-react';
import axios from 'axios';

const API_BASE = 'http://localhost:8000/api';

export default function SimulationLab() {
    const [ticker, setTicker] = useState('AAPL');
    const [startDate, setStartDate] = useState('2024-01-01');
    const [endDate, setEndDate] = useState('2024-12-31');
    const [capital, setCapital] = useState(10000);
    const [interval, setInterval_] = useState(5);
    const [dataMode, setDataMode] = useState('historical');
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState('');

    // Scenarios
    const [flashCrashDay, setFlashCrashDay] = useState('');
    const [flashCrashPct, setFlashCrashPct] = useState(15);
    const [regimeShiftDay, setRegimeShiftDay] = useState('');

    const handleSimulate = async () => {
        setLoading(true);
        setError('');
        setResult(null);

        const scenarios = [];
        if (flashCrashDay) {
            scenarios.push({ day: parseInt(flashCrashDay), type: 'flash_crash', params: { drop_pct: flashCrashPct / 100 } });
        }
        if (regimeShiftDay) {
            scenarios.push({ day: parseInt(regimeShiftDay), type: 'regime_shift', params: { direction: 'bear' } });
        }

        try {
            const res = await axios.post(`${API_BASE}/simulate/`, {
                ticker: ticker.toUpperCase(),
                start_date: startDate,
                end_date: endDate,
                starting_capital: capital,
                decision_interval: interval,
                data_mode: dataMode,
                scenarios,
            });
            setResult(res.data);
        } catch (err) {
            setError(err.response?.data?.error || err.message || 'Simulation failed');
        } finally {
            setLoading(false);
        }
    };

    const getReturnColor = (val) => val > 0 ? 'var(--accent-green)' : val < 0 ? 'var(--accent-red)' : 'var(--text-primary)';

    return (
        <div className="page-container">
            <h1 className="page-title">
                <FlaskConical size={28} style={{ verticalAlign: 'middle', marginRight: '8px' }} />
                Simulation Lab (Digital Twin)
            </h1>
            <p className="page-subtitle">
                Phase 21 — Trap the AI in The Matrix. Test strategies across years of market data in seconds.
            </p>

            {/* Control Panel */}
            <div className="grid-2" style={{ marginBottom: '1.5rem' }}>
                {/* Parameters */}
                <div className="card">
                    <div className="card-header"><Settings /> Simulation Parameters</div>

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                        <div>
                            <label className="metric-label">Ticker</label>
                            <input className="search-input" style={{ width: '100%' }} value={ticker}
                                onChange={(e) => setTicker(e.target.value.toUpperCase())} disabled={loading} />
                        </div>
                        <div>
                            <label className="metric-label">Data Mode</label>
                            <select className="search-input" style={{ width: '100%' }} value={dataMode}
                                onChange={(e) => setDataMode(e.target.value)} disabled={loading}>
                                <option value="historical">Historical Replay</option>
                                <option value="gbm">Synthetic (GBM)</option>
                            </select>
                        </div>
                        <div>
                            <label className="metric-label">Start Date</label>
                            <input className="search-input" type="date" style={{ width: '100%' }} value={startDate}
                                onChange={(e) => setStartDate(e.target.value)} disabled={loading} />
                        </div>
                        <div>
                            <label className="metric-label">End Date</label>
                            <input className="search-input" type="date" style={{ width: '100%' }} value={endDate}
                                onChange={(e) => setEndDate(e.target.value)} disabled={loading} />
                        </div>
                        <div>
                            <label className="metric-label">Starting Capital ($)</label>
                            <input className="search-input" type="number" style={{ width: '100%' }} value={capital}
                                onChange={(e) => setCapital(Number(e.target.value))} disabled={loading} />
                        </div>
                        <div>
                            <label className="metric-label">Decision Every (N days)</label>
                            <input className="search-input" type="number" style={{ width: '100%' }} value={interval}
                                onChange={(e) => setInterval_(Number(e.target.value))} disabled={loading} min={1} max={30} />
                        </div>
                    </div>
                </div>

                {/* Scenario Injector */}
                <div className="card">
                    <div className="card-header"><Zap /> Scenario Injector (Black Swan Events)</div>
                    <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
                        Inject synthetic crashes to torture-test your AI. Leave blank for clean simulation.
                    </p>

                    <div style={{ marginBottom: '12px' }}>
                        <label className="metric-label">Flash Crash on Day #</label>
                        <div style={{ display: 'flex', gap: '8px' }}>
                            <input className="search-input" type="number" placeholder="e.g., 30" style={{ flex: 1 }}
                                value={flashCrashDay} onChange={(e) => setFlashCrashDay(e.target.value)} disabled={loading} />
                            <input className="search-input" type="number" placeholder="%" style={{ width: '80px' }}
                                value={flashCrashPct} onChange={(e) => setFlashCrashPct(Number(e.target.value))} disabled={loading} />
                            <span style={{ alignSelf: 'center', fontSize: '0.75rem', color: 'var(--text-muted)' }}>% drop</span>
                        </div>
                    </div>

                    <div style={{ marginBottom: '16px' }}>
                        <label className="metric-label">Regime Shift (Bear) on Day #</label>
                        <input className="search-input" type="number" placeholder="e.g., 60" style={{ width: '100%' }}
                            value={regimeShiftDay} onChange={(e) => setRegimeShiftDay(e.target.value)} disabled={loading} />
                    </div>

                    <button className="btn btn-primary" onClick={handleSimulate}
                        disabled={loading} style={{ width: '100%' }}>
                        {loading ? <><div className="spinner" style={{ width: 18, height: 18, borderWidth: 2 }} /> Running Simulation...</>
                            : <><Play size={18} /> Run Digital Twin</>}
                    </button>
                </div>
            </div>

            {/* Error */}
            {error && (
                <div className="card" style={{ borderColor: 'var(--accent-red)', marginBottom: '1.5rem' }}>
                    <div className="card-header"><AlertTriangle /> Error</div>
                    <p style={{ color: 'var(--accent-red)' }}>{error}</p>
                </div>
            )}

            {/* Loading */}
            {loading && (
                <div className="loading-overlay">
                    <div className="spinner" />
                    <p className="loading-text">Running Digital Twin Simulation...</p>
                    <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>
                        Stepping through {startDate} → {endDate} day by day...
                    </p>
                </div>
            )}

            {/* Results */}
            {result && (
                <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>

                    {/* Performance Metrics */}
                    <div className="grid-4" style={{ marginBottom: '1.5rem' }}>
                        <div className="card" style={{ textAlign: 'center' }}>
                            <div className="metric-label">Total Return</div>
                            <div className="metric-value" style={{ color: getReturnColor(result.metrics?.total_return_pct), fontSize: '2rem' }}>
                                {result.metrics?.total_return_pct > 0 ? '+' : ''}{result.metrics?.total_return_pct}%
                            </div>
                        </div>
                        <div className="card" style={{ textAlign: 'center' }}>
                            <div className="metric-label">Sharpe Ratio</div>
                            <div className="metric-value" style={{ fontSize: '2rem' }}>
                                {result.metrics?.sharpe_ratio}
                            </div>
                        </div>
                        <div className="card" style={{ textAlign: 'center' }}>
                            <div className="metric-label">Max Drawdown</div>
                            <div className="metric-value" style={{ color: 'var(--accent-red)', fontSize: '2rem' }}>
                                -{result.metrics?.max_drawdown_pct}%
                            </div>
                        </div>
                        <div className="card" style={{ textAlign: 'center' }}>
                            <div className="metric-label">Win Rate</div>
                            <div className="metric-value" style={{ fontSize: '2rem' }}>
                                {result.metrics?.win_rate_pct}%
                            </div>
                        </div>
                    </div>

                    {/* Portfolio Summary */}
                    <div className="grid-2" style={{ marginBottom: '1.5rem' }}>
                        <div className="card">
                            <div className="card-header"><BarChart3 /> Portfolio Summary</div>
                            <div className="grid-3" style={{ textAlign: 'center' }}>
                                <div className="metric">
                                    <span className="metric-label">Starting Capital</span>
                                    <span className="metric-value small">${result.metrics?.starting_capital?.toLocaleString()}</span>
                                </div>
                                <div className="metric">
                                    <span className="metric-label">Final Value</span>
                                    <span className="metric-value small" style={{ color: getReturnColor(result.metrics?.total_return_pct) }}>
                                        ${result.metrics?.final_value?.toLocaleString()}
                                    </span>
                                </div>
                                <div className="metric">
                                    <span className="metric-label">Total Trades</span>
                                    <span className="metric-value small">{result.metrics?.total_trades}</span>
                                </div>
                            </div>
                        </div>

                        {/* Trust Score Evolution */}
                        <div className="card">
                            <div className="card-header"><Activity /> Trust Score Evolution</div>
                            {result.trust_evolution?.length > 0 ? (
                                <div>
                                    {['technical', 'sentiment', 'regime'].map(agent => {
                                        const first = result.trust_evolution[0]?.[agent] || 1.0;
                                        const last = result.trust_evolution[result.trust_evolution.length - 1]?.[agent] || 1.0;
                                        const diff = last - first;
                                        return (
                                            <div key={agent} className="progress-bar-container">
                                                <div className="progress-bar-label">
                                                    <span>{agent}</span>
                                                    <span style={{ color: diff > 0 ? 'var(--accent-green)' : diff < 0 ? 'var(--accent-red)' : 'var(--text-muted)' }}>
                                                        {first.toFixed(3)} → {last.toFixed(3)} ({diff > 0 ? '+' : ''}{diff.toFixed(3)})
                                                    </span>
                                                </div>
                                                <div className="progress-bar-track">
                                                    <div className="progress-bar-fill" style={{
                                                        width: `${Math.min((last / 1.5) * 100, 100)}%`,
                                                        background: last > 1.05 ? 'var(--accent-green)' : last < 0.95 ? 'var(--accent-red)' : 'var(--accent-blue)',
                                                    }} />
                                                </div>
                                            </div>
                                        );
                                    })}
                                </div>
                            ) : <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>No trust evolution data</p>}
                        </div>
                    </div>

                    {/* Equity Curve (ASCII-style with CSS bars) */}
                    <div className="card" style={{ marginBottom: '1.5rem' }}>
                        <div className="card-header"><TrendingUp /> Equity Curve</div>
                        <div style={{ display: 'flex', alignItems: 'flex-end', gap: '1px', height: '200px', padding: '0 4px' }}>
                            {result.equity_curve && (() => {
                                const values = result.equity_curve.map(d => d.portfolio_value);
                                const min = Math.min(...values);
                                const max = Math.max(...values);
                                const range = max - min || 1;
                                // Show max 200 bars
                                const step = Math.max(1, Math.floor(values.length / 200));
                                const sampled = values.filter((_, i) => i % step === 0);

                                return sampled.map((v, i) => {
                                    const pct = ((v - min) / range) * 100;
                                    const isProfit = v >= result.metrics?.starting_capital;
                                    return (
                                        <div key={i} style={{
                                            flex: 1,
                                            minWidth: '1px',
                                            height: `${Math.max(pct, 2)}%`,
                                            background: isProfit
                                                ? 'linear-gradient(to top, rgba(34,197,94,0.3), rgba(34,197,94,0.8))'
                                                : 'linear-gradient(to top, rgba(239,68,68,0.3), rgba(239,68,68,0.8))',
                                            borderRadius: '1px 1px 0 0',
                                            transition: 'height 0.3s ease',
                                        }} title={`$${v.toLocaleString()}`} />
                                    );
                                });
                            })()}
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '8px', fontSize: '0.65rem', color: 'var(--text-muted)' }}>
                            <span>{result.equity_curve?.[0]?.date}</span>
                            <span>{result.equity_curve?.[result.equity_curve.length - 1]?.date}</span>
                        </div>
                    </div>

                    {/* Decision Log Table */}
                    <div className="card" style={{ overflow: 'auto' }}>
                        <div className="card-header"><TrendingDown /> Decision Log ({result.decisions?.length} decisions)</div>
                        <table className="data-table">
                            <thead>
                                <tr>
                                    <th>Day</th>
                                    <th>Date</th>
                                    <th>Decision</th>
                                    <th>Price</th>
                                    <th>Confidence</th>
                                    <th>Allocation</th>
                                    <th>Regime</th>
                                    <th>GDI</th>
                                    <th>Portfolio</th>
                                </tr>
                            </thead>
                            <tbody>
                                {result.decisions?.map((d, i) => (
                                    <tr key={i}>
                                        <td>{d.day}</td>
                                        <td>{d.date}</td>
                                        <td>
                                            <span className={`badge ${d.decision === 'BUY' ? 'badge-buy' : d.decision === 'SELL' ? 'badge-sell' : 'badge-hold'}`}>
                                                {d.decision}
                                            </span>
                                        </td>
                                        <td>${d.price?.toFixed(2)}</td>
                                        <td>{d.confidence?.toFixed(4)}</td>
                                        <td>{d.alloc_pct?.toFixed(1)}%</td>
                                        <td>
                                            <span className={`badge ${d.regime === 'Bull' ? 'badge-buy' : d.regime === 'Bear' ? 'badge-sell' : 'badge-hold'}`}>
                                                {d.regime}
                                            </span>
                                        </td>
                                        <td>{d.gdi?.toFixed(1)}%</td>
                                        <td style={{ color: d.portfolio_value >= capital ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                                            ${d.portfolio_value?.toLocaleString()}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </motion.div>
            )}
        </div>
    );
}
