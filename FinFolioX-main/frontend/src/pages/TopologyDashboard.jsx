import { useState, useEffect, useRef, useCallback } from "react";

// ══════════════════════════════════════════════════════════════════════════════
//  PHASE 24 — TOPOLOGICAL SHAPE AGENT DASHBOARD
//  FinFolio-X | Persistent Homology Visualiser
//
//  Components:
//    1. PersistenceBarcode     — H0 + H1 barcodes (born/die on ε-axis)
//    2. AttractorScatter       — 3-D Takens embedding projected to 2-D
//    3. TopologyMetricsPanel   — Betti-0, Betti-1, Entropy gauges
//    4. StructureLabel         — Dominant structure + market signal badge
//    5. TopologyDashboard      — Root component (API call + layout)
// ══════════════════════════════════════════════════════════════════════════════

// ────────────────────────────────────────────────────────────────────────────
//  MOCK DATA (used when API is not connected)
// ────────────────────────────────────────────────────────────────────────────

const MOCK_DATA = {
    ticker: "AAPL",
    betti0: 0.34,
    betti1: 0.72,
    persistence_entropy: 0.51,
    topology_chaos_score: 0.57,
    dominant_structure: "LOOP",
    market_shape_signal: "SIDEWAYS",
    topology_modifier: 0.91,
    status: "ok",
    h0_bars: [
        [0.00, -1.0],
        [0.00, 0.14],
        [0.02, 0.09],
        [0.03, 0.07],
        [0.05, 0.06],
        [0.06, 0.08],
        [0.08, 0.11],
        [0.11, 0.13],
    ],
    h1_bars: [
        [0.12, 0.47],
        [0.15, 0.41],
        [0.18, 0.36],
        [0.22, 0.32],
        [0.24, 0.29],
        [0.28, -1.0],
    ],
    point_cloud_3d: (() => {
        const pts = [];
        for (let i = 0; i < 120; i++) {
            const t = (i / 120) * Math.PI * 4;
            const r = 0.3 + 0.1 * Math.sin(t * 1.3);
            pts.push([
                r * Math.cos(t) + (Math.random() - 0.5) * 0.08,
                r * Math.sin(t) + (Math.random() - 0.5) * 0.08,
                0.3 * Math.sin(t * 0.7) + (Math.random() - 0.5) * 0.06,
            ]);
        }
        return pts;
    })(),
};

// ────────────────────────────────────────────────────────────────────────────
//  THEME
// ────────────────────────────────────────────────────────────────────────────

const T = {
    bg: "#06070d",
    surface: "#0d0f1a",
    panel: "#11141f",
    border: "#1e2235",
    accent1: "#00d4ff",  // cyan — H0
    accent2: "#a855f7",  // violet — H1
    accent3: "#22c55e",  // green — entropy / ok
    warn: "#f59e0b",
    danger: "#ef4444",
    text: "#e2e8f0",
    muted: "#64748b",
    grid: "#1a1d2e",
};

const STRUCTURE_CONFIG = {
    LOOP: { color: T.accent2, icon: "⟳", label: "LOOP", desc: "Mean-Reverting Attractor" },
    TREND: { color: T.accent3, icon: "→", label: "TREND", desc: "Directional Attractor" },
    CHAOTIC: { color: T.danger, icon: "⚡", label: "CHAOTIC", desc: "Disordered Shape" },
    SMOOTH: { color: T.warn, icon: "〰", label: "SMOOTH", desc: "Transitional Geometry" },
    UNKNOWN: { color: T.muted, icon: "?", label: "UNKNOWN", desc: "Insufficient Data" },
};

const SIGNAL_CONFIG = {
    SIDEWAYS: { color: T.accent2, label: "SIDEWAYS", strategy: "Fade-the-move / Range" },
    TRENDING: { color: T.accent3, label: "TRENDING", strategy: "Momentum / Breakout" },
    CHAOTIC: { color: T.danger, label: "CHAOTIC", strategy: "Reduce Exposure" },
    NEUTRAL: { color: T.warn, label: "NEUTRAL", strategy: "Wait & Watch" },
    UNKNOWN: { color: T.muted, label: "UNKNOWN", strategy: "Insufficient Data" },
};

// ────────────────────────────────────────────────────────────────────────────
//  COMPONENT: ArcGauge
// ────────────────────────────────────────────────────────────────────────────

function ArcGauge({ value, label, sublabel, color, size = 100 }) {
    const r = (size / 2) * 0.72;
    const cx = size / 2;
    const cy = size / 2;
    const startAngle = -210;
    const sweepAngle = 240;
    const angle = startAngle + sweepAngle * Math.min(Math.max(value, 0), 1);

    const toRad = (deg) => (deg * Math.PI) / 180;
    const arcPath = (a1, a2) => {
        const x1 = cx + r * Math.cos(toRad(a1));
        const y1 = cy + r * Math.sin(toRad(a1));
        const x2 = cx + r * Math.cos(toRad(a2));
        const y2 = cy + r * Math.sin(toRad(a2));
        const large = Math.abs(a2 - a1) > 180 ? 1 : 0;
        return `M ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2}`;
    };

    return (
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
            <svg width={size} height={size * 0.75} viewBox={`0 0 ${size} ${size * 0.75}`} overflow="visible">
                <path
                    d={arcPath(startAngle, startAngle + sweepAngle)}
                    fill="none"
                    stroke={T.border}
                    strokeWidth={size * 0.065}
                    strokeLinecap="round"
                />
                {value > 0.01 && (
                    <path
                        d={arcPath(startAngle, angle)}
                        fill="none"
                        stroke={color}
                        strokeWidth={size * 0.065}
                        strokeLinecap="round"
                        style={{ filter: `drop-shadow(0 0 ${size * 0.04}px ${color})` }}
                    />
                )}
                <text x={cx} y={cy * 0.95} textAnchor="middle" dominantBaseline="middle"
                    fill={color} fontSize={size * 0.20} fontWeight="700"
                    fontFamily="'DM Mono', 'Courier New', monospace">
                    {(value * 100).toFixed(0)}
                </text>
            </svg>
            <div style={{ fontSize: 11, color: T.text, fontWeight: 600, letterSpacing: "0.08em" }}>{label}</div>
            <div style={{ fontSize: 9, color: T.muted, letterSpacing: "0.05em" }}>{sublabel}</div>
        </div>
    );
}

// ────────────────────────────────────────────────────────────────────────────
//  COMPONENT: PersistenceBarcode
// ────────────────────────────────────────────────────────────────────────────

function PersistenceBarcode({ h0_bars = [], h1_bars = [], width = 520, height = 220 }) {
    const PAD = { l: 50, r: 20, t: 30, b: 36 };
    const W = width - PAD.l - PAD.r;
    const maxEps = 0.55;

    const allFinite = [...h0_bars, ...h1_bars]
        .filter(([, d]) => d >= 0)
        .map(([, d]) => d);
    const domainMax = allFinite.length > 0 ? Math.max(...allFinite, 0.01) : 0.55;
    const scale = (v) => v < 0 ? W : (v / Math.max(domainMax, maxEps)) * W;

    // Sort by lifetime desc
    const sortedH0 = [...h0_bars].sort((a, b) => {
        const la = a[1] < 0 ? 999 : a[1] - a[0];
        const lb = b[1] < 0 ? 999 : b[1] - b[0];
        return lb - la;
    });
    const sortedH1 = [...h1_bars].sort((a, b) => {
        const la = a[1] < 0 ? 999 : a[1] - a[0];
        const lb = b[1] < 0 ? 999 : b[1] - b[0];
        return lb - la;
    });

    const all = [...sortedH0.map((b) => [b, 0]), ...sortedH1.map((b) => [b, 1])];
    const rowH = Math.max(5, Math.floor((height - PAD.t - PAD.b) / Math.max(all.length, 1)));
    const clampedH = Math.min(rowH, 11);
    const totalDataH = all.length * (clampedH + 3);
    const svgH = PAD.t + totalDataH + PAD.b;

    // Tick marks
    const ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5].filter((t) => t <= domainMax + 0.05);

    return (
        <svg width={width} height={svgH} style={{ display: "block", overflow: "visible" }}>
            {/* Grid lines */}
            {ticks.map((t) => (
                <line
                    key={t}
                    x1={PAD.l + scale(t)} y1={PAD.t - 6}
                    x2={PAD.l + scale(t)} y2={PAD.t + totalDataH}
                    stroke={T.grid} strokeWidth={1} strokeDasharray="3 3"
                />
            ))}

            {/* Axis */}
            <line x1={PAD.l} y1={PAD.t + totalDataH} x2={PAD.l + W} y2={PAD.t + totalDataH}
                stroke={T.border} strokeWidth={1} />
            <line x1={PAD.l} y1={PAD.t - 6} x2={PAD.l} y2={PAD.t + totalDataH}
                stroke={T.border} strokeWidth={1} />

            {/* Tick labels */}
            {ticks.map((t) => (
                <text key={t} x={PAD.l + scale(t)} y={svgH - 8}
                    textAnchor="middle" fill={T.muted} fontSize={9} fontFamily="'DM Mono', monospace">
                    {t.toFixed(1)}
                </text>
            ))}
            <text x={PAD.l + W / 2} y={svgH - 1} textAnchor="middle"
                fill={T.muted} fontSize={8.5} fontFamily="sans-serif" letterSpacing="0.08em">
                ε (filtration scale)
            </text>

            {/* Bars */}
            {all.map(([[birth, death], dim], i) => {
                const y = PAD.t + i * (clampedH + 3);
                const x1 = PAD.l + scale(birth);
                const x2 = PAD.l + (death < 0 ? W : scale(death));
                const col = dim === 0 ? T.accent1 : T.accent2;
                const inf = death < 0;

                return (
                    <g key={i}>
                        <rect x={x1} y={y} width={Math.max(x2 - x1, 2)} height={clampedH}
                            fill={col} rx={1.5} opacity={inf ? 0.95 : 0.75}
                            style={{ filter: `drop-shadow(0 0 2px ${col}88)` }}
                        />
                        {inf && (
                            <text x={x2 + 4} y={y + clampedH * 0.78}
                                fill={col} fontSize={9} fontFamily="'DM Mono', monospace" opacity={0.8}>
                                ∞
                            </text>
                        )}
                    </g>
                );
            })}

            {/* Legend */}
            <rect x={PAD.l + 4} y={PAD.t - 24} width={8} height={8} fill={T.accent1} rx={1} />
            <text x={PAD.l + 16} y={PAD.t - 17} fill={T.accent1} fontSize={9} fontFamily="sans-serif">
                H₀ (components)
            </text>
            <rect x={PAD.l + 100} y={PAD.t - 24} width={8} height={8} fill={T.accent2} rx={1} />
            <text x={PAD.l + 112} y={PAD.t - 17} fill={T.accent2} fontSize={9} fontFamily="sans-serif">
                H₁ (loops)
            </text>
        </svg>
    );
}

// ────────────────────────────────────────────────────────────────────────────
//  COMPONENT: AttractorScatter  (Takens embedding projection)
// ────────────────────────────────────────────────────────────────────────────

function AttractorScatter({ points = [], structure = "UNKNOWN", width = 260, height = 220 }) {
    const color = STRUCTURE_CONFIG[structure]?.color ?? T.muted;
    const PAD = 24;

    if (!points || points.length === 0) {
        return (
            <div style={{
                width, height, display: "flex", alignItems: "center", justifyContent: "center",
                color: T.muted, fontSize: 12, fontFamily: "sans-serif"
            }}>
                No point cloud data
            </div>
        );
    }

    // Project 3-D → 2-D using x and y axes (simple isometric hints via z offset)
    const xs = points.map(([x]) => x);
    const ys = points.map(([, y]) => y);
    const zs = points.map(([, , z]) => z ?? 0);
    const xmin = Math.min(...xs), xmax = Math.max(...xs);
    const ymin = Math.min(...ys), ymax = Math.max(...ys);
    const zmin = Math.min(...zs), zmax = Math.max(...zs);

    const norm = (v, mn, mx) => (mx - mn < 1e-9 ? 0.5 : (v - mn) / (mx - mn));

    const px = (x, z) => PAD + norm(x, xmin, xmax) * (width - 2 * PAD) + norm(z, zmin, zmax) * 8;
    const py = (y, z) => PAD + (1 - norm(y, ymin, ymax)) * (height - 2 * PAD) - norm(z, zmin, zmax) * 6;

    const sorted = points.map((p, i) => ({ p, i, z: p[2] ?? 0 }))
        .sort((a, b) => a.z - b.z);

    return (
        <svg width={width} height={height} style={{ overflow: "visible" }}>
            {/* Connecting trajectory line */}
            <polyline
                points={points.map(([x, , z], i) => {
                    const [, y] = points[i];
                    return `${px(x, z ?? 0)},${py(y, z ?? 0)}`;
                }).join(" ")}
                fill="none"
                stroke={`${color}28`}
                strokeWidth={0.8}
            />
            {/* Points — ordered back to front by z */}
            {sorted.map(({ p: [x, y, z], i }) => {
                const sz = 1.8 + norm(z ?? 0, zmin, zmax) * 2.2;
                const op = 0.45 + norm(z ?? 0, zmin, zmax) * 0.55;
                return (
                    <circle
                        key={i}
                        cx={px(x, z ?? 0)}
                        cy={py(y, z ?? 0)}
                        r={sz}
                        fill={color}
                        opacity={op}
                        style={{ filter: `drop-shadow(0 0 ${sz * 0.8}px ${color}88)` }}
                    />
                );
            })}
            {/* Origin cross */}
            <line x1={PAD - 4} y1={height / 2} x2={PAD + 2} y2={height / 2} stroke={T.border} strokeWidth={1} />
            <line x1={PAD} y1={height / 2 - 4} x2={PAD} y2={height / 2 + 4} stroke={T.border} strokeWidth={1} />
        </svg>
    );
}

// ────────────────────────────────────────────────────────────────────────────
//  COMPONENT: ChaosScoreBar
// ────────────────────────────────────────────────────────────────────────────

function ChaosScoreBar({ score = 0.5 }) {
    const clr = score < 0.35 ? T.accent3 : score < 0.65 ? T.warn : T.danger;
    const pct = `${(score * 100).toFixed(1)}%`;
    const segments = Array.from({ length: 40 }, (_, i) => i / 40);

    return (
        <div style={{ padding: "16px 20px", background: T.panel, borderRadius: 10, border: `1px solid ${T.border}` }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 10 }}>
                <span style={{ fontSize: 11, color: T.muted, letterSpacing: "0.1em", fontFamily: "sans-serif" }}>
                    TOPOLOGY CHAOS SCORE
                </span>
                <span style={{
                    fontSize: 26, fontWeight: 800, color: clr,
                    fontFamily: "'DM Mono', 'Courier New', monospace",
                    textShadow: `0 0 12px ${clr}88`
                }}>
                    {pct}
                </span>
            </div>

            {/* Segmented bar */}
            <div style={{ display: "flex", gap: 2, height: 12, borderRadius: 6, overflow: "hidden" }}>
                {segments.map((t) => (
                    <div key={t}
                        style={{
                            flex: 1,
                            background: t < score ? clr : T.border,
                            opacity: t < score ? (0.4 + t * 0.8) : 1,
                            borderRadius: 2,
                            transition: "background 0.3s",
                        }}
                    />
                ))}
            </div>

            <div style={{ display: "flex", justifyContent: "space-between", marginTop: 5 }}>
                <span style={{ fontSize: 9, color: T.accent3, fontFamily: "'DM Mono', monospace" }}>0% ORDERED</span>
                <span style={{ fontSize: 9, color: T.danger, fontFamily: "'DM Mono', monospace" }}>100% CHAOTIC</span>
            </div>
        </div>
    );
}

// ────────────────────────────────────────────────────────────────────────────
//  COMPONENT: StructureBadge
// ────────────────────────────────────────────────────────────────────────────

function StructureBadge({ structure, signal, modifier }) {
    const sc = STRUCTURE_CONFIG[structure] ?? STRUCTURE_CONFIG.UNKNOWN;
    const sg = SIGNAL_CONFIG[signal] ?? SIGNAL_CONFIG.UNKNOWN;

    return (
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {/* Dominant structure */}
            <div style={{
                display: "flex", alignItems: "center", gap: 12,
                padding: "14px 18px",
                background: `${sc.color}15`,
                border: `1px solid ${sc.color}50`,
                borderRadius: 10,
            }}>
                <div style={{ fontSize: 28, lineHeight: 1 }}>{sc.icon}</div>
                <div>
                    <div style={{ fontSize: 9, color: T.muted, letterSpacing: "0.1em", marginBottom: 2 }}>
                        DOMINANT STRUCTURE
                    </div>
                    <div style={{
                        fontSize: 18, fontWeight: 800, color: sc.color,
                        fontFamily: "'DM Mono', monospace", letterSpacing: "0.05em"
                    }}>
                        {sc.label}
                    </div>
                    <div style={{ fontSize: 10, color: T.muted }}>{sc.desc}</div>
                </div>
            </div>

            {/* Market shape signal */}
            <div style={{
                display: "flex", alignItems: "center", justifyContent: "space-between",
                padding: "12px 18px",
                background: `${sg.color}10`,
                border: `1px solid ${sg.color}40`,
                borderRadius: 10,
            }}>
                <div>
                    <div style={{ fontSize: 9, color: T.muted, letterSpacing: "0.1em", marginBottom: 2 }}>
                        MARKET SHAPE SIGNAL
                    </div>
                    <div style={{
                        fontSize: 15, fontWeight: 700, color: sg.color,
                        fontFamily: "'DM Mono', monospace"
                    }}>
                        {sg.label}
                    </div>
                    <div style={{ fontSize: 9, color: T.muted }}>{sg.strategy}</div>
                </div>
                <div style={{ textAlign: "right" }}>
                    <div style={{ fontSize: 9, color: T.muted, letterSpacing: "0.1em", marginBottom: 2 }}>
                        FUSION MODIFIER
                    </div>
                    <div style={{
                        fontSize: 20, fontWeight: 800,
                        color: modifier >= 1.0 ? T.accent3 : T.warn,
                        fontFamily: "'DM Mono', monospace",
                    }}>
                        {modifier?.toFixed(2)}×
                    </div>
                </div>
            </div>
        </div>
    );
}

// ────────────────────────────────────────────────────────────────────────────
//  COMPONENT: TopologyDashboard  (root)
// ────────────────────────────────────────────────────────────────────────────

export default function TopologyDashboard() {
    const [ticker, setTicker] = useState("AAPL");
    const [inputVal, setInputVal] = useState("AAPL");
    const [data, setData] = useState(MOCK_DATA);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [useMock, setUseMock] = useState(true);

    const fetchData = useCallback(async (sym) => {
        if (useMock) {
            setData({ ...MOCK_DATA, ticker: sym.toUpperCase() });
            return;
        }
        setLoading(true);
        setError(null);
        try {
            const res = await fetch(`http://127.0.0.1:8000/api/topology/${sym}/`);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const json = await res.json();
            setData(json);
        } catch (e) {
            setError(e.message);
            setData({ ...MOCK_DATA, ticker: sym.toUpperCase() });
        } finally {
            setLoading(false);
        }
    }, [useMock]);

    useEffect(() => { fetchData(ticker); }, [ticker, fetchData]);

    const handleSubmit = () => {
        const sym = inputVal.trim().toUpperCase();
        if (sym) { setTicker(sym); fetchData(sym); }
    };

    const { betti0, betti1, persistence_entropy, topology_chaos_score,
        dominant_structure, market_shape_signal, topology_modifier,
        h0_bars, h1_bars, point_cloud_3d } = data;

    return (
        <div style={{
            minHeight: "100vh",
            background: T.bg,
            color: T.text,
            fontFamily: "'Inter', 'Segoe UI', sans-serif",
            padding: "24px",
            boxSizing: "border-box",
        }}>

            {/* ── HEADER ── */}
            <div style={{ maxWidth: 1080, margin: "0 auto 24px" }}>
                <div style={{
                    display: "flex", alignItems: "flex-start", justifyContent: "space-between",
                    flexWrap: "wrap", gap: 16
                }}>
                    <div>
                        <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 4 }}>
                            <div style={{
                                width: 36, height: 36, borderRadius: 8,
                                background: `linear-gradient(135deg, ${T.accent2}, ${T.accent1})`,
                                display: "flex", alignItems: "center", justifyContent: "center",
                                fontSize: 18, fontWeight: 900,
                            }}>
                                ∮
                            </div>
                            <div>
                                <div style={{ fontSize: 20, fontWeight: 800, letterSpacing: "-0.02em" }}>
                                    Topological Shape Agent
                                </div>
                                <div style={{ fontSize: 11, color: T.muted, letterSpacing: "0.06em" }}>
                                    PHASE 24 · FINFOLIO-X · PERSISTENT HOMOLOGY
                                </div>
                            </div>
                        </div>
                        <div style={{ fontSize: 11, color: T.muted, maxWidth: 480, lineHeight: 1.6 }}>
                            Detects market regime geometry via{" "}
                            <span style={{ color: T.accent1 }}>Takens Delay Embedding</span> +{" "}
                            <span style={{ color: T.accent2 }}>Vietoris-Rips Persistent Homology</span>.
                            Orthogonal to HMM statistical regimes.
                        </div>
                    </div>

                    {/* Ticker input */}
                    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                        <div style={{ display: "flex", gap: 8 }}>
                            <input
                                value={inputVal}
                                onChange={(e) => setInputVal(e.target.value.toUpperCase())}
                                onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
                                placeholder="TICKER"
                                style={{
                                    background: T.surface, border: `1px solid ${T.border}`,
                                    borderRadius: 8, padding: "8px 14px",
                                    color: T.text, fontSize: 14, fontFamily: "'DM Mono', monospace",
                                    fontWeight: 700, width: 110, outline: "none",
                                }}
                            />
                            <button
                                onClick={handleSubmit}
                                disabled={loading}
                                style={{
                                    background: loading ? T.border : `linear-gradient(135deg, ${T.accent2}, ${T.accent1})`,
                                    border: "none", borderRadius: 8, padding: "8px 18px",
                                    color: loading ? T.muted : "#fff", fontWeight: 700, fontSize: 13,
                                    cursor: loading ? "not-allowed" : "pointer", letterSpacing: "0.04em",
                                }}
                            >
                                {loading ? "…" : "ANALYZE"}
                            </button>
                        </div>
                        <label style={{
                            display: "flex", alignItems: "center", gap: 6,
                            fontSize: 10, color: T.muted, cursor: "pointer"
                        }}>
                            <input type="checkbox" checked={useMock}
                                onChange={(e) => setUseMock(e.target.checked)}
                                style={{ accentColor: T.accent2 }} />
                            Use demo data (no API)
                        </label>
                    </div>
                </div>

                {error && (
                    <div style={{
                        marginTop: 10, padding: "8px 14px", background: `${T.danger}18`,
                        border: `1px solid ${T.danger}40`, borderRadius: 8, fontSize: 12, color: T.danger
                    }}>
                        API error: {error} — showing demo data
                    </div>
                )}
            </div>

            {/* ── MAIN GRID ── */}
            <div style={{
                maxWidth: 1080, margin: "0 auto", display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: 16
            }}>

                {/* ── LEFT COL: Metrics + Structure ── */}
                <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

                    {/* Chaos score bar */}
                    <ChaosScoreBar score={topology_chaos_score} />

                    {/* 3 gauges */}
                    <div style={{
                        display: "flex", justifyContent: "space-around", padding: "18px 12px",
                        background: T.panel, borderRadius: 10, border: `1px solid ${T.border}`,
                    }}>
                        <ArcGauge
                            value={betti0} label="H₀ BETTI-0" sublabel="Fragmentation"
                            color={T.accent1} size={90}
                        />
                        <ArcGauge
                            value={betti1} label="H₁ BETTI-1" sublabel="Oscillation/Loops"
                            color={T.accent2} size={90}
                        />
                        <ArcGauge
                            value={persistence_entropy} label="ENTROPY" sublabel="Global Chaos"
                            color={T.warn} size={90}
                        />
                    </div>

                    {/* Structure + signal badge */}
                    <StructureBadge
                        structure={dominant_structure}
                        signal={market_shape_signal}
                        modifier={topology_modifier}
                    />

                    {/* Mathematical note */}
                    <div style={{
                        padding: "12px 16px", background: T.panel,
                        border: `1px solid ${T.border}`, borderRadius: 10,
                        fontSize: 10, color: T.muted, lineHeight: 1.7,
                    }}>
                        <div style={{ color: T.text, fontWeight: 600, marginBottom: 4, fontSize: 11 }}>
                            Research Basis (Phase 24)
                        </div>
                        <div>
                            <span style={{ color: T.accent1 }}>Betti-0</span> counts connected components of the{" "}
                            Vietoris-Rips complex. <span style={{ color: T.accent2 }}>Betti-1</span> counts independent
                            1-cycles (loops) — each loop maps to a cyclic / mean-reverting structure in the price
                            attractor. <span style={{ color: T.warn }}>Persistence Entropy</span> measures complexity
                            of the full topological barcode. Reconstructed via{" "}
                            <span style={{ color: T.accent1 }}>Takens (1981) delay embedding</span>.
                        </div>
                    </div>
                </div>

                {/* ── RIGHT COL: Barcode + Attractor ── */}
                <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

                    {/* Persistence Barcode */}
                    <div style={{
                        background: T.panel, borderRadius: 10,
                        border: `1px solid ${T.border}`, padding: "16px",
                    }}>
                        <div style={{
                            fontSize: 11, color: T.muted, letterSpacing: "0.1em",
                            marginBottom: 12, fontWeight: 600
                        }}>
                            PERSISTENCE BARCODE
                        </div>
                        <div style={{ fontSize: 10, color: T.muted, marginBottom: 8, lineHeight: 1.6 }}>
                            Each bar = topological feature born at ε (left) and dying at ε (right).
                            Long bars = significant structure. ∞ = feature persists to all scales.
                        </div>
                        <div style={{ overflowX: "auto" }}>
                            <PersistenceBarcode
                                h0_bars={h0_bars}
                                h1_bars={h1_bars}
                                width={480}
                                height={Math.max(180, (h0_bars.length + h1_bars.length) * 16 + 60)}
                            />
                        </div>
                    </div>

                    {/* Takens Attractor Scatter */}
                    <div style={{
                        background: T.panel, borderRadius: 10,
                        border: `1px solid ${T.border}`, padding: "16px",
                    }}>
                        <div style={{
                            fontSize: 11, color: T.muted, letterSpacing: "0.1em",
                            marginBottom: 4, fontWeight: 600
                        }}>
                            TAKENS DELAY EMBEDDING — ATTRACTOR MANIFOLD
                        </div>
                        <div style={{ fontSize: 10, color: T.muted, marginBottom: 12, lineHeight: 1.6 }}>
                            3-D reconstruction of the market's dynamical attractor from Close prices.
                            Loops → SIDEWAYS regime. Clean spiral → TRENDING regime.
                        </div>
                        <div style={{ display: "flex", justifyContent: "center" }}>
                            <AttractorScatter
                                points={point_cloud_3d}
                                structure={dominant_structure}
                                width={460}
                                height={220}
                            />
                        </div>
                    </div>

                    {/* Raw values strip */}
                    <div style={{
                        display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 8,
                    }}>
                        {[
                            { key: "Betti-0", val: betti0?.toFixed(3), col: T.accent1 },
                            { key: "Betti-1", val: betti1?.toFixed(3), col: T.accent2 },
                            { key: "Entropy", val: persistence_entropy?.toFixed(3), col: T.warn },
                            {
                                key: "Modifier", val: `${topology_modifier?.toFixed(2)}×`, col:
                                    topology_modifier >= 1.0 ? T.accent3 : T.warn
                            },
                        ].map(({ key, val, col }) => (
                            <div key={key} style={{
                                background: T.surface, borderRadius: 8, padding: "10px 12px",
                                border: `1px solid ${T.border}`, textAlign: "center",
                            }}>
                                <div style={{
                                    fontSize: 17, fontWeight: 800, color: col,
                                    fontFamily: "'DM Mono', monospace"
                                }}>{val}</div>
                                <div style={{
                                    fontSize: 9, color: T.muted, marginTop: 2,
                                    letterSpacing: "0.06em"
                                }}>{key}</div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* ── FOOTER ── */}
            <div style={{
                maxWidth: 1080, margin: "20px auto 0",
                padding: "12px 16px", background: T.surface,
                borderRadius: 8, border: `1px solid ${T.border}`,
                display: "flex", justifyContent: "space-between", flexWrap: "wrap", gap: 8,
                fontSize: 10, color: T.muted,
            }}>
                <span>
                    Ticker: <span style={{ color: T.accent1, fontFamily: "'DM Mono', monospace" }}>
                        {data.ticker}
                    </span>
                    {" · "}Library: <span style={{ color: T.text }}>ripser + persim</span>
                    {" · "}Status: <span style={{ color: data.status === "ok" ? T.accent3 : T.warn }}>
                        {data.status}
                    </span>
                </span>
                <span>
                    Takens τ=5 · d=3 · lookback=60 bars
                    {" · "}Vietoris-Rips maxdim=1
                </span>
            </div>
        </div>
    );
}