import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import LiveInference from './pages/LiveInference';
import MetaAgent from './pages/MetaAgent';
import DecisionLedger from './pages/DecisionLedger';
import SimulationLab from './pages/SimulationLab';
import TopologyDashboard from './pages/TopologyDashboard';
import CausalDashboard from './pages/CausalDashboard';
import './index.css';

export default function App() {
  return (
    <BrowserRouter>
      <Navbar />
      <Routes>
        <Route path="/" element={<LiveInference />} />
        <Route path="/meta" element={<MetaAgent />} />
        <Route path="/history" element={<DecisionLedger />} />
        <Route path="/simulation" element={<SimulationLab />} />
        <Route path="/topology" element={<TopologyDashboard />} />
        <Route path="/causal" element={<CausalDashboard />} />
      </Routes>
    </BrowserRouter>
  );
}
