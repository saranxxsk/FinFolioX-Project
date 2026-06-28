import { useState, useEffect, useRef, useCallback, useMemo } from "react";

// ══════════════════════════════════════════════════════════════════════════════
//  PHASE 25 — CAUSAL DISCOVERY AGENT DASHBOARD
//  FinFolio-X | Judea Pearl's Do-Calculus Visualiser
//
//  Aesthetic: Warm amber obsidian — like equations chalked on a blackboard.
//  Font: Playfair Display (academic headers) + JetBrains Mono (data).
//  Layout: Asymmetric — DAG dominates left, evidence panel right.
//
//  Components:
//    1. CausalDAG            — Animated directed acyclic graph
//    2. DoCalculusTable      — ρ (correlation) vs β_do (causal effect)
//    3. CounterfactualPanel  — "What if X = y?" explorer
//    4. DriversList          — True causal drivers ranked
//    5. CausalDashboard      — Root
// ══════════════════════════════════════════════════════════════════════════════

// ────────────────────────────────────────────────────────────────────────────
//  MOCK DATA
// ────────────────────────────────────────────────────────────────────────────

const MOCK = {
    ticker: "AAPL",
    causal_score: 0.74,
    causal_modifier: 1.08,
    counterfactual_delta: 0.00178,
    counterfactual_narrative:
        "If VIX had been 1.5 standard deviations BELOW its historical mean (do(VIX=0.004) instead of observed 0.031), AAPL would have returned +0.41% instead of the factual +0.23%. Δ = +0.18%. Causal effect used: β_do(VIX→AAPL) = −0.03124.",
    confounders_removed: ["QQQ", "GLD"],
    true_causal_drivers: [
        { variable: "SPY", label: "S&P 500 (Market Proxy)", causal_effect: 0.0423, p_value: 0.012, significant: true, direction: "↑" },
        { variable: "VIX", label: "CBOE Volatility Index (Fear)", causal_effect: -0.0312, p_value: 0.034, significant: true, direction: "↓" },
        { variable: "TLT", label: "20Y Treasury Bond ETF", causal_effect: -0.0089, p_value: 0.087, significant: false, direction: "↓" },
    ],
    dag_edges: [
        { source: "VIX", target: "SPY", strength: 0.90, causal: false, effect: 0 },
        { source: "TLT", target: "SPY", strength: 0.55, causal: false, effect: 0 },
        { source: "TLT", target: "GLD", strength: 0.62, causal: false, effect: 0 },
        { source: "DXY", target: "GLD", strength: 0.48, causal: false, effect: 0 },
        { source: "DXY", target: "TLT", strength: 0.35, causal: false, effect: 0 },
        { source: "VIX", target: "QQQ", strength: 0.70, causal: false, effect: 0 },
        { source: "SPY", target: "TARGET", strength: 0.80, causal: true, effect: 0.0423 },
        { source: "VIX", target: "TARGET", strength: 0.60, causal: true, effect: -0.0312 },
        { source: "TLT", target: "TARGET", strength: 0.25, causal: true, effect: -0.0089 },
        { source: "QQQ", target: "TARGET", strength: 0.40, causal: false, effect: 0 },
    ],
    correlation_vs_causal: [
        { variable: "SPY", label: "S&P 500", correlation: 0.68, causal_effect: 0.0423, gap: 0.638, is_confounder: false, is_causal: true },
        { variable: "QQQ", label: "NASDAQ-100", correlation: 0.61, causal_effect: 0.0061, gap: 0.604, is_confounder: true, is_causal: false },
        { variable: "VIX", label: "Volatility (Fear)", correlation: -0.43, causal_effect: -0.0312, gap: 0.399, is_confounder: false, is_causal: true },
        { variable: "TLT", label: "Treasury Bonds", correlation: -0.22, causal_effect: -0.0089, gap: 0.211, is_confounder: false, is_causal: false },
        { variable: "GLD", label: "Gold ETF", correlation: 0.14, causal_effect: 0.0010, gap: 0.139, is_confounder: true, is_causal: false },
        { variable: "DXY", label: "USD Index", correlation: -0.18, causal_effect: 0.0028, gap: 0.177, is_confounder: true, is_causal: false },
    ],
    status: "ok",
};

// ────────────────────────────────────────────────────────────────────────────
//  THEME
// ────────────────────────────────────────────────────────────────────────────

const T = {
    bg: "#0a0803",
    surface: "#100e08",
    panel: "#16130c",
    border: "#2a2416",
    amber: "#f59e0b",
    gold: "#d97706",
    dimgold: "#92681e",
    orange: "#ea580c",
    red: "#dc2626",
    green: "#16a34a",
    teal: "#0d9488",
    text: "#fef3c7",
    muted: "#78716c",
    faint: "#3a3120",
    conf: "#7c3aed",   // confounder = purple warning
    causal: "#f59e0b",   // true causal = amber
};

// DAG node positions (normalised 0–1)
const DAG_POSITIONS = {
    VIX: { x: 0.12, y: 0.20 },
    DXY: { x: 0.12, y: 0.70 },
    TLT: { x: 0.35, y: 0.55 },
    GLD: { x: 0.57, y: 0.75 },
    SPY: { x: 0.58, y: 0.22 },
    QQQ: { x: 0.35, y: 0.22 },
    TARGET: { x: 0.82, y: 0.48 },
};

// ────────────────────────────────────────────────────────────────────────────
//  COMPONENT: CausalDAG
// ────────────────────────────────────────────────────────────────────────────

function CausalDAG({ edges, confounders, ticker, width = 520, height = 340 }) {
    const [hovered, setHovered] = useState(null);
    const confounderSet = new Set(confounders || []);
    const allNodes = Object.keys(DAG_POSITIONS);

    const toSvg = (pos) => ({
        x: pos.x * (width - 80) + 40,
        y: pos.y * (height - 60) + 30,
    });

    // Arrow marker IDs
    const markerId = (causal) => (causal ? "arrow-causal" : "arrow-normal");

    return (
        <svg width={width} height={height} style={{ overflow: "visible", display: "block" }}>
            <defs>
                <marker id="arrow-causal" markerWidth="7" markerHeight="7"
                    refX="6" refY="3.5" orient="auto">
                    <polygon points="0 0, 7 3.5, 0 7" fill={T.amber} />
                </marker>
                <marker id="arrow-normal" markerWidth="7" markerHeight="7"
                    refX="6" refY="3.5" orient="auto">
                    <polygon points="0 0, 7 3.5, 0 7" fill={T.muted} opacity="0.5" />
                </marker>
                <marker id="arrow-conf" markerWidth="7" markerHeight="7"
                    refX="6" refY="3.5" orient="auto">
                    <polygon points="0 0, 7 3.5, 0 7" fill={T.conf} />
                </marker>
                <filter id="glow-amber">
                    <feGaussianBlur stdDeviation="3" result="coloredBlur" />
                    <feMerge><feMergeNode in="coloredBlur" /><feMergeNode in="SourceGraphic" /></feMerge>
                </filter>
            </defs>

            {/* Edges */}
            {edges.map((edge, i) => {
                const src = DAG_POSITIONS[edge.source];
                const tgt = DAG_POSITIONS[edge.target];
                if (!src || !tgt) return null;

                const s = toSvg(src);
                const t = toSvg(tgt);

                // Shorten endpoints to not overlap node circles
                const dx = t.x - s.x, dy = t.y - s.y;
                const len = Math.sqrt(dx * dx + dy * dy);
                const r = 22;
                const sx = s.x + (dx / len) * r;
                const sy = s.y + (dy / len) * r;
                const ex = t.x - (dx / len) * (r + 6);
                const ey = t.y - (dy / len) * (r + 6);

                const isConfPath = confounderSet.has(edge.source) || confounderSet.has(edge.target);
                const isCausal = edge.causal;
                const isActive = hovered === edge.source || hovered === edge.target;

                const color = isCausal ? T.amber : isConfPath ? T.conf : T.faint;
                const width_ = isCausal ? 2.2 : 1.0;
                const opacity = isActive ? 1 : isCausal ? 0.85 : 0.35;
                const marker = isCausal ? "url(#arrow-causal)" : isConfPath ? "url(#arrow-conf)" : "url(#arrow-normal)";

                // Slight curve to avoid straight overlaps
                const mx = (sx + ex) / 2 - dy * 0.12;
                const my = (sy + ey) / 2 + dx * 0.12;

                return (
                    <path
                        key={i}
                        d={`M ${sx} ${sy} Q ${mx} ${my} ${ex} ${ey}`}
                        fill="none"
                        stroke={color}
                        strokeWidth={width_}
                        opacity={opacity}
                        markerEnd={marker}
                        style={isCausal ? { filter: `drop-shadow(0 0 4px ${T.amber}88)` } : {}}
                    >
                        <title>
                            {edge.source} → {edge.target}
                            {isCausal ? ` | β_do = ${edge.effect > 0 ? "+" : ""}${edge.effect.toFixed(4)}` : ""}
                        </title>
                    </path>
                );
            })}

            {/* Nodes */}
            {allNodes.map((node) => {
                const pos = toSvg(DAG_POSITIONS[node]);
                const isTarget = node === "TARGET";
                const isConfounder = confounderSet.has(node);
                const isHovered = hovered === node;
                const nodeColor = isTarget ? T.amber : isConfounder ? T.conf : T.dimgold;
                const nodeBg = isTarget ? "#1a120000" : T.surface;

                return (
                    <g key={node} onMouseEnter={() => setHovered(node)}
                        onMouseLeave={() => setHovered(null)}
                        style={{ cursor: "pointer" }}>
                        <circle cx={pos.x} cy={pos.y} r={22}
                            fill={T.panel}
                            stroke={nodeColor}
                            strokeWidth={isTarget ? 2.5 : isHovered ? 2 : 1.5}
                            style={isTarget ? { filter: `drop-shadow(0 0 8px ${T.amber}99)` } : {}}
                        />
                        {/* Node label */}
                        <text x={pos.x} y={pos.y + 1}
                            textAnchor="middle" dominantBaseline="middle"
                            fill={nodeColor}
                            fontSize={isTarget ? 9.5 : 9}
                            fontWeight={isTarget ? "800" : "600"}
                            fontFamily="'JetBrains Mono', 'Fira Code', monospace"
                            letterSpacing="-0.02em"
                        >
                            {isTarget ? ticker.slice(0, 4) : node}
                        </text>

                        {/* Confounder badge */}
                        {isConfounder && (
                            <g>
                                <circle cx={pos.x + 16} cy={pos.y - 16} r={7}
                                    fill={T.conf} opacity={0.9} />
                                <text x={pos.x + 16} y={pos.y - 15.5}
                                    textAnchor="middle" dominantBaseline="middle"
                                    fill="#fff" fontSize={8} fontWeight="800">
                                    C
                                </text>
                            </g>
                        )}
                    </g>
                );
            })}

            {/* Legend */}
            <g transform={`translate(8, ${height - 56})`}>
                <line x1="0" y1="5" x2="18" y2="5" stroke={T.amber} strokeWidth={2.2}
                    markerEnd="url(#arrow-causal)" />
                <text x={22} y={9} fill={T.amber} fontSize={9} fontFamily="sans-serif">True Causal</text>
                <line x1="0" y1="20" x2="18" y2="20" stroke={T.conf} strokeWidth={1.5}
                    strokeDasharray="3 2" markerEnd="url(#arrow-conf)" />
                <text x={22} y={24} fill={T.conf} fontSize={9} fontFamily="sans-serif">Confounder Path</text>
                <line x1="0" y1="35" x2="18" y2="35" stroke={T.faint} strokeWidth={1} />
                <text x={22} y={39} fill={T.muted} fontSize={9} fontFamily="sans-serif">Non-causal</text>
                <circle cx={4} cy={47} r={5} fill="none" stroke={T.conf} strokeWidth={1.5} />
                <text x={12} y={47} fill={T.conf} fontSize={9} fontFamily="sans-serif"
                    dominantBaseline="middle">
                    C = Confounder (correlated, NOT causal)
                </text>
            </g>
        </svg>
    );
}

// ────────────────────────────────────────────────────────────────────────────
//  COMPONENT: DoCalculusTable
// ────────────────────────────────────────────────────────────────────────────

function DoCalculusTable({ rows = [] }) {
    const maxCorr = Math.max(...rows.map((r) => Math.abs(r.correlation)), 0.01);
    const maxCausal = Math.max(...rows.map((r) => Math.abs(r.causal_effect)), 0.001);

    return (
        <div style={{ overflowX: "auto" }}>
            {/* Header explanation */}
            <div style={{
                padding: "10px 14px", marginBottom: 10,
                background: `${T.amber}10`, border: `1px solid ${T.amber}30`, borderRadius: 8,
                fontSize: 10.5, color: T.text, lineHeight: 1.7,
                fontFamily: "'Playfair Display', Georgia, serif",
                fontStyle: "italic",
            }}>
                "The gap between ρ (correlation) and β_do (causal effect) is the measure of
                confounding. Large gap = spurious correlation. Small gap = genuine causation."
                <span style={{ color: T.muted, fontStyle: "normal", fontSize: 9.5 }}> — Pearl, 2009</span>
            </div>

            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
                <thead>
                    <tr style={{ borderBottom: `1px solid ${T.border}` }}>
                        {["Variable", "ρ  Correlation", "β_do  Causal Effect", "Gap", "Type"].map((h) => (
                            <th key={h} style={{
                                padding: "7px 10px", textAlign: "left",
                                color: T.muted, fontWeight: 600, letterSpacing: "0.06em",
                                fontSize: 9.5, fontFamily: "sans-serif",
                            }}>
                                {h}
                            </th>
                        ))}
                    </tr>
                </thead>
                <tbody>
                    {rows.map((row, i) => {
                        const isConfounder = row.is_confounder;
                        const isCausal = row.is_causal;
                        const typeColor = isConfounder ? T.conf : isCausal ? T.amber : T.muted;
                        const typeLabel = isConfounder ? "CONFOUNDER" : isCausal ? "CAUSAL" : "WEAK";
                        const corrW = Math.abs(row.correlation) / maxCorr * 60;
                        const causeW = Math.abs(row.causal_effect) / maxCausal * 60;

                        return (
                            <tr key={i} style={{
                                borderBottom: `1px solid ${T.faint}`,
                                background: i % 2 === 0 ? "transparent" : `${T.surface}80`,
                            }}>
                                <td style={{ padding: "8px 10px" }}>
                                    <div style={{
                                        fontWeight: 700, color: T.text,
                                        fontFamily: "'JetBrains Mono', monospace", fontSize: 11
                                    }}>
                                        {row.variable}
                                    </div>
                                    <div style={{ fontSize: 9, color: T.muted, marginTop: 1 }}>{row.label}</div>
                                </td>

                                {/* Correlation bar */}
                                <td style={{ padding: "8px 10px" }}>
                                    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                                        <div style={{
                                            width: corrW, height: 8, borderRadius: 3,
                                            background: `${T.conf}88`,
                                            minWidth: 2,
                                        }} />
                                        <span style={{
                                            fontSize: 10.5, color: T.conf,
                                            fontFamily: "'JetBrains Mono', monospace"
                                        }}>
                                            {row.correlation >= 0 ? "+" : ""}{row.correlation.toFixed(3)}
                                        </span>
                                    </div>
                                </td>

                                {/* Causal effect bar */}
                                <td style={{ padding: "8px 10px" }}>
                                    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                                        <div style={{
                                            width: Math.max(causeW, 2), height: 8, borderRadius: 3,
                                            background: isCausal ? `${T.amber}aa` : `${T.muted}44`,
                                            minWidth: 2,
                                            boxShadow: isCausal ? `0 0 6px ${T.amber}66` : "none",
                                        }} />
                                        <span style={{
                                            fontSize: 10.5,
                                            color: isCausal ? T.amber : T.muted,
                                            fontFamily: "'JetBrains Mono', monospace",
                                            fontWeight: isCausal ? 700 : 400,
                                        }}>
                                            {row.causal_effect >= 0 ? "+" : ""}{row.causal_effect.toFixed(4)}
                                        </span>
                                    </div>
                                </td>

                                {/* Gap */}
                                <td style={{ padding: "8px 10px" }}>
                                    <span style={{
                                        fontSize: 10, color: row.gap > 0.3 ? T.conf : T.muted,
                                        fontFamily: "'JetBrains Mono', monospace",
                                    }}>
                                        {row.gap.toFixed(3)}
                                    </span>
                                </td>

                                {/* Type badge */}
                                <td style={{ padding: "8px 10px" }}>
                                    <span style={{
                                        fontSize: 9, fontWeight: 700, letterSpacing: "0.06em",
                                        color: typeColor,
                                        background: `${typeColor}18`,
                                        padding: "3px 7px", borderRadius: 4,
                                        border: `1px solid ${typeColor}40`,
                                        fontFamily: "sans-serif",
                                    }}>
                                        {typeLabel}
                                    </span>
                                </td>
                            </tr>
                        );
                    })}
                </tbody>
            </table>
        </div>
    );
}

// ────────────────────────────────────────────────────────────────────────────
//  COMPONENT: CounterfactualPanel
// ────────────────────────────────────────────────────────────────────────────

function CounterfactualPanel({ narrative, delta, ticker }) {
    const direction = delta >= 0 ? "HIGHER" : "LOWER";
    const dirColor = delta >= 0 ? T.green : T.red;
    const absDelta = Math.abs(delta * 100).toFixed(3);

    return (
        <div style={{
            padding: "18px 20px",
            background: T.panel,
            border: `1px solid ${T.border}`,
            borderLeft: `3px solid ${T.amber}`,
            borderRadius: 10,
        }}>
            <div style={{
                fontSize: 10, color: T.muted, letterSpacing: "0.12em",
                fontWeight: 700, marginBottom: 12, fontFamily: "sans-serif",
            }}>
                COUNTERFACTUAL  ·  do(X = x) QUERY
            </div>

            {/* Delta display */}
            <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 14 }}>
                <div style={{ textAlign: "center" }}>
                    <div style={{
                        fontSize: 32, fontWeight: 900, color: dirColor,
                        fontFamily: "'JetBrains Mono', monospace",
                        textShadow: `0 0 16px ${dirColor}66`,
                    }}>
                        {delta >= 0 ? "+" : ""}{(delta * 100).toFixed(3)}%
                    </div>
                    <div style={{ fontSize: 9, color: T.muted, marginTop: 2 }}>
                        COUNTERFACTUAL Δ
                    </div>
                </div>

                <div style={{
                    fontSize: 11, color: T.text, lineHeight: 1.8,
                    flex: 1, fontFamily: "'Playfair Display', Georgia, serif",
                }}>
                    Under causal intervention, <strong style={{ color: ticker ? T.amber : T.text }}>
                        {ticker}
                    </strong> would have traded{" "}
                    <strong style={{ color: dirColor }}>{absDelta}% {direction}</strong> if the primary
                    causal driver had been at its historical neutral value.
                </div>
            </div>

            {/* Full narrative */}
            <div style={{
                padding: "12px 14px",
                background: `${T.amber}08`,
                border: `1px solid ${T.amber}20`,
                borderRadius: 7,
                fontSize: 10.5, color: T.muted, lineHeight: 1.8,
                fontFamily: "'JetBrains Mono', monospace",
                wordBreak: "break-word",
            }}>
                {narrative}
            </div>

            <div style={{
                marginTop: 10, fontSize: 9.5, color: T.muted,
                fontStyle: "italic", fontFamily: "serif",
            }}>
                Mathematical basis: Y_cf = Y_factual − β_do × (X_actual − X_mean),
                where β_do is estimated via Pearl's backdoor adjustment criterion.
            </div>
        </div>
    );
}

// ────────────────────────────────────────────────────────────────────────────
//  COMPONENT: CausalScoreGauge
// ────────────────────────────────────────────────────────────────────────────

function CausalScoreGauge({ score, modifier }) {
    const pct = Math.round(score * 100);
    const color = score >= 0.65 ? T.amber : score >= 0.40 ? T.gold : T.conf;
    const label = score >= 0.65 ? "HIGH CAUSAL CLARITY" : score >= 0.40 ? "MODERATE" : "LOW — CONFOUNDERS";
    const desc = score >= 0.65 ? "Strong causal drivers. High confidence."
        : score >= 0.40 ? "Mixed evidence. Moderate confidence."
            : "Mostly confounders. Reduce exposure confidence.";

    // Build arc
    const r = 54, cx = 70, cy = 70;
    const sweep = 240;
    const offset = -120;
    const toRad = (d) => (d * Math.PI) / 180;
    const arc = (a1, a2) => {
        const x1 = cx + r * Math.cos(toRad(a1));
        const y1 = cy + r * Math.sin(toRad(a1));
        const x2 = cx + r * Math.cos(toRad(a2));
        const y2 = cy + r * Math.sin(toRad(a2));
        return `M ${x1} ${y1} A ${r} ${r} 0 ${Math.abs(a2 - a1) > 180 ? 1 : 0} 1 ${x2} ${y2}`;
    };

    const endAngle = offset + sweep * score;

    return (
        <div style={{
            padding: "18px", background: T.panel,
            border: `1px solid ${T.border}`, borderRadius: 10,
            display: "flex", gap: 20, alignItems: "center",
        }}>
            <svg width={140} height={100} viewBox="0 0 140 100">
                <path d={arc(offset, offset + sweep)} fill="none"
                    stroke={T.faint} strokeWidth={10} strokeLinecap="round" />
                {score > 0.01 && (
                    <path d={arc(offset, endAngle)} fill="none"
                        stroke={color} strokeWidth={10} strokeLinecap="round"
                        style={{ filter: `drop-shadow(0 0 6px ${color}88)` }} />
                )}
                <text x={cx} y={cy + 4} textAnchor="middle" fill={color}
                    fontSize={22} fontWeight={900} fontFamily="'JetBrains Mono', monospace">
                    {pct}
                </text>
                <text x={cx} y={cy + 18} textAnchor="middle" fill={T.muted}
                    fontSize={8} fontFamily="sans-serif" letterSpacing="0.06em">
                    CAUSAL SCORE
                </text>
            </svg>

            <div style={{ flex: 1 }}>
                <div style={{
                    fontSize: 12, fontWeight: 800, color, letterSpacing: "0.05em",
                    marginBottom: 4, fontFamily: "sans-serif"
                }}>
                    {label}
                </div>
                <div style={{ fontSize: 10.5, color: T.muted, lineHeight: 1.7, marginBottom: 10 }}>
                    {desc}
                </div>
                <div style={{ display: "flex", gap: 6 }}>
                    <div style={{
                        fontSize: 11, fontWeight: 700,
                        color: modifier >= 1.0 ? T.green : T.conf,
                        fontFamily: "'JetBrains Mono', monospace",
                        background: `${modifier >= 1.0 ? T.green : T.conf}15`,
                        padding: "3px 10px", borderRadius: 5,
                        border: `1px solid ${modifier >= 1.0 ? T.green : T.conf}40`,
                    }}>
                        Fusion ×{modifier.toFixed(2)}
                    </div>
                    <div style={{
                        fontSize: 11, color: T.muted,
                        background: T.faint,
                        padding: "3px 10px", borderRadius: 5,
                        fontFamily: "sans-serif",
                    }}>
                        Judea Pearl Do-Calculus
                    </div>
                </div>
            </div>
        </div>
    );
}

// ────────────────────────────────────────────────────────────────────────────
//  COMPONENT: DriversPanel
// ────────────────────────────────────────────────────────────────────────────

function DriversPanel({ drivers, confounders }) {
    const maxEff = Math.max(...drivers.map((d) => Math.abs(d.causal_effect)), 0.001);
    return (
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {drivers.map((d, i) => {
                const barW = (Math.abs(d.causal_effect) / maxEff) * 100;
                const col = d.significant ? T.amber : T.dimgold;
                const posNeg = d.causal_effect >= 0 ? T.green : T.red;
                return (
                    <div key={i} style={{
                        padding: "12px 14px",
                        background: d.significant ? `${T.amber}08` : `${T.surface}`,
                        border: `1px solid ${d.significant ? T.amber + "40" : T.border}`,
                        borderRadius: 8,
                    }}>
                        <div style={{
                            display: "flex", justifyContent: "space-between",
                            alignItems: "flex-start", marginBottom: 6
                        }}>
                            <div>
                                <span style={{
                                    fontWeight: 800, color: col,
                                    fontFamily: "'JetBrains Mono', monospace", fontSize: 12
                                }}>
                                    {d.variable}
                                </span>
                                <span style={{ fontSize: 10, color: T.muted, marginLeft: 8 }}>
                                    {d.label}
                                </span>
                            </div>
                            <div style={{ textAlign: "right" }}>
                                <span style={{
                                    fontSize: 14, fontWeight: 800, color: posNeg,
                                    fontFamily: "'JetBrains Mono', monospace",
                                }}>
                                    β_do = {d.causal_effect >= 0 ? "+" : ""}{d.causal_effect.toFixed(4)}
                                </span>
                            </div>
                        </div>
                        {/* Effect bar */}
                        <div style={{ background: T.faint, borderRadius: 3, height: 5, overflow: "hidden" }}>
                            <div style={{
                                height: "100%", width: `${barW}%`,
                                background: posNeg,
                                borderRadius: 3,
                                boxShadow: `0 0 6px ${posNeg}66`,
                            }} />
                        </div>
                        <div style={{
                            display: "flex", justifyContent: "space-between",
                            marginTop: 4, fontSize: 9, color: T.muted
                        }}>
                            <span>p = {d.p_value.toFixed(3)} {d.significant ? "✓ significant" : "~ marginal"}</span>
                            <span>{d.direction} {d.significant ? "TRUE CAUSAL DRIVER" : "WEAK"}</span>
                        </div>
                    </div>
                );
            })}

            {confounders.length > 0 && (
                <div style={{
                    padding: "10px 14px",
                    background: `${T.conf}10`,
                    border: `1px solid ${T.conf}40`,
                    borderRadius: 8,
                }}>
                    <div style={{
                        fontSize: 10, fontWeight: 700, color: T.conf,
                        letterSpacing: "0.08em", marginBottom: 4, fontFamily: "sans-serif"
                    }}>
                        CONFOUNDERS ELIMINATED
                    </div>
                    <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                        {confounders.map((c) => (
                            <span key={c} style={{
                                fontSize: 10, fontWeight: 700, color: T.conf,
                                fontFamily: "'JetBrains Mono', monospace",
                                background: `${T.conf}20`, padding: "3px 8px",
                                borderRadius: 4, border: `1px solid ${T.conf}50`,
                            }}>
                                {c}
                            </span>
                        ))}
                    </div>
                    <div style={{ fontSize: 9.5, color: T.muted, marginTop: 6, fontStyle: "italic" }}>
                        These variables appeared correlated but are mathematically NOT causal —
                        eliminated via backdoor adjustment (Pearl's do-calculus).
                    </div>
                </div>
            )}
        </div>
    );
}

// ────────────────────────────────────────────────────────────────────────────
//  ROOT: CausalDashboard
// ────────────────────────────────────────────────────────────────────────────

export default function CausalDashboard() {
    const [ticker, setTicker] = useState("AAPL");
    const [inputVal, setInputVal] = useState("AAPL");
    const [data, setData] = useState(MOCK);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [useMock, setUseMock] = useState(true);
    const [activeTab, setActiveTab] = useState("dag");

    const fetchData = useCallback(async (sym) => {
        if (useMock) { setData({ ...MOCK, ticker: sym }); return; }
        setLoading(true); setError(null);
        try {
            const res = await fetch(`http://127.0.0.1:8000/api/causal/${sym}/`);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            setData(await res.json());
        } catch (e) {
            setError(e.message);
            setData({ ...MOCK, ticker: sym });
        } finally { setLoading(false); }
    }, [useMock]);

    useEffect(() => { fetchData(ticker); }, [ticker, fetchData]);

    const tabs = [
        { id: "dag", label: "Causal DAG" },
        { id: "table", label: "ρ vs β_do" },
        { id: "drivers", label: "Causal Drivers" },
    ];

    return (
        <div style={{
            minHeight: "100vh", background: T.bg, color: T.text,
            fontFamily: "'Georgia', serif",
            padding: "24px", boxSizing: "border-box",
        }}>
            {/* ── Google Fonts injection ── */}
            <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,600;0,800;1,600&family=JetBrains+Mono:wght@400;700&display=swap');
      `}</style>

            <div style={{ maxWidth: 1100, margin: "0 auto" }}>

                {/* ── HEADER ── */}
                <div style={{ marginBottom: 24 }}>
                    <div style={{
                        display: "flex", alignItems: "flex-start",
                        justifyContent: "space-between", flexWrap: "wrap", gap: 16
                    }}>
                        <div>
                            <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 6 }}>
                                <div style={{
                                    width: 40, height: 40, borderRadius: 10,
                                    background: `linear-gradient(135deg, ${T.gold}, ${T.orange})`,
                                    display: "flex", alignItems: "center", justifyContent: "center",
                                    fontSize: 20,
                                }}>
                                    ∂
                                </div>
                                <div>
                                    <div style={{
                                        fontSize: 22, fontWeight: 800, letterSpacing: "-0.02em",
                                        fontFamily: "'Playfair Display', Georgia, serif",
                                        color: T.text,
                                    }}>
                                        Causal Discovery Agent
                                    </div>
                                    <div style={{
                                        fontSize: 10.5, color: T.muted, letterSpacing: "0.1em",
                                        fontFamily: "sans-serif"
                                    }}>
                                        PHASE 25 · FINFOLIO-X · JUDEA PEARL DO-CALCULUS
                                    </div>
                                </div>
                            </div>
                            <div style={{
                                fontSize: 11.5, color: T.muted, maxWidth: 520, lineHeight: 1.7,
                                fontStyle: "italic", fontFamily: "'Playfair Display', Georgia, serif",
                            }}>
                                Upgrades the system from{" "}
                                <span style={{ color: T.conf }}>P(Y|X) correlation</span> to{" "}
                                <span style={{ color: T.amber }}>P(Y|do(X)) causal inference</span>.
                                The difference between seeing and doing.
                            </div>
                        </div>

                        {/* Controls */}
                        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                            <div style={{ display: "flex", gap: 8 }}>
                                <input
                                    value={inputVal}
                                    onChange={(e) => setInputVal(e.target.value.toUpperCase())}
                                    onKeyDown={(e) => e.key === "Enter" && (() => { setTicker(inputVal); fetchData(inputVal); })()}
                                    placeholder="TICKER"
                                    style={{
                                        background: T.surface, border: `1px solid ${T.border}`,
                                        borderRadius: 8, padding: "8px 14px",
                                        color: T.text, fontSize: 14,
                                        fontFamily: "'JetBrains Mono', monospace",
                                        fontWeight: 700, width: 110, outline: "none",
                                        caretColor: T.amber,
                                    }}
                                />
                                <button
                                    onClick={() => { setTicker(inputVal); fetchData(inputVal); }}
                                    disabled={loading}
                                    style={{
                                        background: loading ? T.faint : `linear-gradient(135deg, ${T.gold}, ${T.orange})`,
                                        border: "none", borderRadius: 8, padding: "8px 18px",
                                        color: loading ? T.muted : "#0a0803", fontWeight: 800, fontSize: 12,
                                        cursor: loading ? "not-allowed" : "pointer",
                                        letterSpacing: "0.05em", fontFamily: "sans-serif",
                                    }}>
                                    {loading ? "…" : "ANALYZE"}
                                </button>
                            </div>
                            <label style={{
                                display: "flex", alignItems: "center", gap: 6,
                                fontSize: 10, color: T.muted, cursor: "pointer",
                                fontFamily: "sans-serif"
                            }}>
                                <input type="checkbox" checked={useMock}
                                    onChange={(e) => setUseMock(e.target.checked)}
                                    style={{ accentColor: T.amber }} />
                                Demo data (no API)
                            </label>
                        </div>
                    </div>

                    {error && (
                        <div style={{
                            marginTop: 10, padding: "8px 14px",
                            background: `${T.red}18`, border: `1px solid ${T.red}40`,
                            borderRadius: 8, fontSize: 12, color: T.red
                        }}>
                            API error: {error} — showing demo data
                        </div>
                    )}
                </div>

                {/* ── MAIN GRID ── */}
                <div style={{
                    display: "grid",
                    gridTemplateColumns: "minmax(300px, 1.1fr) minmax(300px, 0.9fr)",
                    gap: 16, alignItems: "start"
                }}>

                    {/* ── LEFT: Score + DAG / Table / Drivers tabs ── */}
                    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                        <CausalScoreGauge
                            score={data.causal_score}
                            modifier={data.causal_modifier}
                        />

                        {/* Tab bar */}
                        <div style={{
                            display: "flex", gap: 2, padding: "3px",
                            background: T.surface, borderRadius: 9, border: `1px solid ${T.border}`,
                            width: "fit-content"
                        }}>
                            {tabs.map((t) => (
                                <button key={t.id}
                                    onClick={() => setActiveTab(t.id)}
                                    style={{
                                        padding: "7px 16px", borderRadius: 7,
                                        border: "none", cursor: "pointer",
                                        background: activeTab === t.id ? `linear-gradient(135deg, ${T.gold}44, ${T.amber}22)` : "transparent",
                                        color: activeTab === t.id ? T.amber : T.muted,
                                        fontSize: 11, fontWeight: 700, letterSpacing: "0.04em",
                                        transition: "all 0.15s",
                                        fontFamily: "sans-serif",
                                        outline: activeTab === t.id ? `1px solid ${T.amber}40` : "none",
                                    }}>
                                    {t.label}
                                </button>
                            ))}
                        </div>

                        {/* Tab content */}
                        <div style={{
                            background: T.panel, borderRadius: 12,
                            border: `1px solid ${T.border}`, padding: "16px", minHeight: 340
                        }}>
                            {activeTab === "dag" && (
                                <>
                                    <div style={{
                                        fontSize: 10.5, color: T.muted, marginBottom: 12,
                                        lineHeight: 1.7, fontFamily: "sans-serif"
                                    }}>
                                        Discovered causal structure of the market. Amber arrows carry true causal
                                        force — they survive do-calculus backdoor adjustment. Purple paths
                                        connect confounders that appear correlated but are NOT causal.
                                    </div>
                                    <div style={{ overflowX: "auto" }}>
                                        <CausalDAG
                                            edges={data.dag_edges}
                                            confounders={data.confounders_removed}
                                            ticker={data.ticker}
                                            width={490}
                                            height={320}
                                        />
                                    </div>
                                </>
                            )}

                            {activeTab === "table" && (
                                <DoCalculusTable rows={data.correlation_vs_causal} />
                            )}

                            {activeTab === "drivers" && (
                                <DriversPanel
                                    drivers={data.true_causal_drivers}
                                    confounders={data.confounders_removed}
                                />
                            )}
                        </div>
                    </div>

                    {/* ── RIGHT: Counterfactual + Drivers summary + Pearl quote ── */}
                    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

                        <CounterfactualPanel
                            narrative={data.counterfactual_narrative}
                            delta={data.counterfactual_delta}
                            ticker={data.ticker}
                        />

                        {/* Drivers quick list */}
                        <div style={{
                            background: T.panel, borderRadius: 10,
                            border: `1px solid ${T.border}`, padding: "14px 16px"
                        }}>
                            <div style={{
                                fontSize: 10, color: T.muted, letterSpacing: "0.1em",
                                fontWeight: 700, marginBottom: 10, fontFamily: "sans-serif"
                            }}>
                                TRUE CAUSAL DRIVERS  ·  P(Y | do(X))
                            </div>
                            {data.true_causal_drivers.map((d, i) => (
                                <div key={i} style={{
                                    display: "flex", justifyContent: "space-between",
                                    alignItems: "center", padding: "7px 0",
                                    borderBottom: i < data.true_causal_drivers.length - 1
                                        ? `1px solid ${T.faint}` : "none"
                                }}>
                                    <div>
                                        <span style={{
                                            fontWeight: 700, fontSize: 11,
                                            fontFamily: "'JetBrains Mono', monospace", color: T.amber
                                        }}>
                                            {d.variable}
                                        </span>
                                        <span style={{ fontSize: 10, color: T.muted, marginLeft: 8 }}>
                                            {d.label}
                                        </span>
                                    </div>
                                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                                        <span style={{
                                            fontSize: 12, fontWeight: 800,
                                            color: d.causal_effect >= 0 ? T.green : T.red,
                                            fontFamily: "'JetBrains Mono', monospace"
                                        }}>
                                            {d.causal_effect >= 0 ? "+" : ""}{d.causal_effect.toFixed(4)}
                                        </span>
                                        <span style={{ fontSize: 9, color: d.significant ? T.amber : T.muted }}>
                                            {d.significant ? "✓" : "~"}
                                        </span>
                                    </div>
                                </div>
                            ))}
                        </div>

                        {/* Academic citation block */}
                        <div style={{
                            padding: "14px 16px",
                            background: `${T.amber}06`,
                            border: `1px solid ${T.amber}20`,
                            borderRadius: 10,
                        }}>
                            <div style={{
                                fontSize: 10, color: T.muted, letterSpacing: "0.1em",
                                fontWeight: 700, marginBottom: 8, fontFamily: "sans-serif"
                            }}>
                                ACADEMIC BASIS
                            </div>
                            {[
                                { ref: "Pearl (2009)", desc: "Causality: Models, Reasoning, and Inference — Backdoor criterion & do-calculus" },
                                { ref: "Spirtes et al.", desc: "PC Algorithm — Constraint-based causal structure discovery" },
                                { ref: "Shimizu (2006)", desc: "LiNGAM — Linear Non-Gaussian Acyclic Model for causal ordering" },
                                { ref: "DoWhy (2021)", desc: "Microsoft Research — do-calculus Python library" },
                            ].map(({ ref, desc }) => (
                                <div key={ref} style={{
                                    display: "flex", gap: 8, marginBottom: 6,
                                    fontSize: 10.5, lineHeight: 1.6
                                }}>
                                    <span style={{
                                        color: T.amber, fontWeight: 700, flexShrink: 0,
                                        fontFamily: "'JetBrains Mono', monospace"
                                    }}>
                                        {ref}
                                    </span>
                                    <span style={{ color: T.muted }}>{desc}</span>
                                </div>
                            ))}
                        </div>

                        {/* Status strip */}
                        <div style={{
                            display: "flex", justifyContent: "space-between",
                            padding: "8px 14px", background: T.surface,
                            borderRadius: 8, border: `1px solid ${T.border}`,
                            fontSize: 10, color: T.muted, fontFamily: "sans-serif"
                        }}>
                            <span>
                                Ticker: <span style={{
                                    color: T.amber,
                                    fontFamily: "'JetBrains Mono', monospace"
                                }}>
                                    {data.ticker}
                                </span>
                                {" · "}n = {data.n_observations ?? 90} obs
                            </span>
                            <span>
                                Status:{" "}
                                <span style={{ color: (data.status || "").startsWith("ok") ? T.green : T.conf }}>
                                    {data.status}
                                </span>
                            </span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}