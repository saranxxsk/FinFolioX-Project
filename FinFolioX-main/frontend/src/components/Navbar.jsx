import { NavLink } from 'react-router-dom';
import { Activity, Brain, History, Zap, FlaskConical, Grid3x3, Share2 } from 'lucide-react';

export default function Navbar() {
    return (
        <nav className="navbar">
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <NavLink to="/" className="navbar-brand">
                    <Zap size={20} />
                    FinFolio-X
                </NavLink>
                <span className="navbar-version">v16.0</span>
            </div>
            <div className="navbar-links">
                <NavLink to="/" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`} end>
                    <Activity size={16} /> Live Inference
                </NavLink>
                <NavLink to="/meta" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
                    <Brain size={16} /> Meta-Agent
                </NavLink>
                <NavLink to="/history" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
                    <History size={16} /> Decision Ledger
                </NavLink>
                <NavLink to="/simulation" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
                    <FlaskConical size={16} /> Simulation Lab
                </NavLink>
                <NavLink to="/topology" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
                    <Grid3x3 size={16} /> Topology
                </NavLink>
                <NavLink to="/causal" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
                    <Share2 size={16} /> Causal
                </NavLink>
            </div>
        </nav>
    );
}
