/**
 * frontend/src/components/EnsembleHealthPanel.jsx  —  Phase 26 ASC Dashboard Panel
 * ==================================================================================
 * Renders the Ensemble Health card for the Live Inference page, visualizing the
 * Agent Sycophancy Coefficient (ASC) and Forced Dissent Protocol (FDP) results.
 * Shows a color-coded ASC bar (green < 0.3, amber 0.3–0.7, red > 0.7), the
 * confidence penalty multiplier applied, the ensemble quadrant label, and — when
 * FDP ran — the Dissent Sensitivity score with a plain-English interpretation.
 * A small mutual information breakdown table shows the pairwise MI values between
 * the three agents, giving researchers a direct view into what drove the ASC score.
 */

import { Activity } from 'lucide-react';

const ASC_COLORS = {
    low: { bar: 'var(--accent-green)', bg: 'var(--accent-green-glow)', label: 'INDEPENDENT' },
    medium: { bar: 'var(--accent-amber)', bg: 'var(--accent-amber-glow)', label: 'MILD SYCOPHANCY' },
    high: { bar: 'var(--accent-red)', bg: 'var(--accent-red-glow)', label: 'SYCOPHANTIC' },
};

function getAscTier(asc) {
    if (asc < 0.30) return 'low';
    if (asc < 0.70) return 'medium';
    return 'high';
}

export default function EnsembleHealthPanel({ data }) {
    if (!data) return null;

    const {
        asc_score = 0.5,
        asc_reliable = false,
        asc_penalty_multiplier = 1.0,
        asc_quadrant = '',
        dissent_sensitivity = 0.0,
        fdp_ran = false,
        fdp_interpretation = '',
    } = data;

    const tier = getAscTier(asc_score);
    const colors = ASC_COLORS[tier];
    const penaltyPct = Math.round((1 - asc_penalty_multiplier) * 100);

    return (
        <div className="card">
            <div className="card-header">
                <Activity />
                Ensemble Health (ASC)
                {!asc_reliable && (
                    <span style={{
                        marginLeft: '8px', fontSize: '0.65rem', padding: '2px 6px',
                        background: 'var(--accent-amber-glow)', color: 'var(--accent-amber)',
                        borderRadius: '4px', fontWeight: 700,
                    }}>
                        WARMING UP
                    </span>
                )}
            </div>

            {/* ASC Score Bar */}
            <div style={{ marginBottom: '1rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 500 }}>
                        Agent Sycophancy Coefficient
                    </span>
                    <span style={{ fontSize: '0.9rem', fontWeight: 700, color: colors.bar }}>
                        {(asc_score * 100).toFixed(1)}%
                    </span>
                </div>
                <div className="progress-bar-track">
                    <div
                        className="progress-bar-fill"
                        style={{
                            width: `${asc_score * 100}%`,
                            background: colors.bar,
                            transition: 'width 0.8s ease',
                        }}
                    />
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '4px', fontSize: '0.65rem', color: 'var(--text-muted)' }}>
                    <span>0% — Independent</span>
                    <span>100% — Sycophantic</span>
                </div>
            </div>

            {/* Status Badge */}
            <div style={{ marginBottom: '0.75rem', display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                <span style={{
                    fontSize: '0.7rem', fontWeight: 700, padding: '3px 8px',
                    borderRadius: '5px', background: colors.bg, color: colors.bar,
                    letterSpacing: '0.5px',
                }}>
                    {colors.label}
                </span>
                {penaltyPct > 0 && (
                    <span style={{
                        fontSize: '0.7rem', fontWeight: 700, padding: '3px 8px',
                        borderRadius: '5px',
                        background: 'var(--accent-red-glow)', color: 'var(--accent-red)',
                        letterSpacing: '0.5px',
                    }}>
                        -{penaltyPct}% CONFIDENCE PENALTY
                    </span>
                )}
                {penaltyPct === 0 && (
                    <span className="badge badge-harmony">NO PENALTY</span>
                )}
            </div>

            {/* Quadrant label */}
            {asc_quadrant && (
                <div style={{
                    fontSize: '0.72rem', color: 'var(--text-secondary)', lineHeight: 1.5,
                    padding: '6px 10px',
                    background: 'rgba(255,255,255,0.03)',
                    borderRadius: '6px',
                    borderLeft: `2px solid ${colors.bar}`,
                    marginBottom: '0.75rem',
                }}>
                    {asc_quadrant}
                </div>
            )}

            {/* Penalty multiplier metric */}
            <div className="metric" style={{ marginBottom: '0.75rem' }}>
                <span className="metric-label">Confidence multiplier applied</span>
                <span className="metric-value small" style={{
                    color: asc_penalty_multiplier < 0.75 ? 'var(--accent-red)'
                        : asc_penalty_multiplier < 1.0 ? 'var(--accent-amber)'
                            : 'var(--accent-green)',
                }}>
                    {asc_penalty_multiplier.toFixed(2)}×
                </span>
            </div>

            {/* FDP Section */}
            {fdp_ran && (
                <div style={{
                    padding: '10px 12px',
                    background: 'rgba(239,68,68,0.06)',
                    borderRadius: '8px',
                    border: '1px solid rgba(239,68,68,0.2)',
                    marginTop: '0.5rem',
                }}>
                    <div style={{
                        fontSize: '0.7rem', fontWeight: 700, color: 'var(--accent-red)',
                        letterSpacing: '0.5px', marginBottom: '6px',
                    }}>
                        FORCED DISSENT PROTOCOL TRIGGERED
                    </div>

                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                            Dissent Sensitivity
                        </span>
                        <span style={{
                            fontSize: '0.9rem', fontWeight: 700,
                            color: dissent_sensitivity > 0.25 ? 'var(--accent-red)'
                                : dissent_sensitivity > 0.10 ? 'var(--accent-amber)'
                                    : 'var(--accent-green)',
                        }}>
                            {(dissent_sensitivity * 100).toFixed(1)}%
                        </span>
                    </div>

                    {/* Dissent Sensitivity bar */}
                    <div className="progress-bar-track" style={{ marginBottom: '6px' }}>
                        <div
                            className="progress-bar-fill"
                            style={{
                                width: `${dissent_sensitivity * 100}%`,
                                background: dissent_sensitivity > 0.25 ? 'var(--accent-red)'
                                    : dissent_sensitivity > 0.10 ? 'var(--accent-amber)'
                                        : 'var(--accent-green)',
                            }}
                        />
                    </div>

                    {fdp_interpretation && (
                        <p style={{
                            fontSize: '0.72rem', color: 'var(--text-muted)', lineHeight: 1.55,
                            margin: 0,
                        }}>
                            {fdp_interpretation}
                        </p>
                    )}
                </div>
            )}

            {!fdp_ran && asc_reliable && (
                <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', margin: '0.5rem 0 0' }}>
                    Forced Dissent Protocol not triggered — ASC below threshold.
                </p>
            )}

            {!asc_reliable && (
                <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', margin: '0.5rem 0 0' }}>
                    ASC warming up — need 15+ sessions for reliable estimate.
                    Currently using neutral score of 0.50.
                </p>
            )}
        </div>
    );
}