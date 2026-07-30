import { motion } from "framer-motion";
import { FaRunning, FaBrain, FaChartLine, FaMagic, FaServer, FaReact } from "react-icons/fa";
import GlassCard from "../components/common/GlassCard";

const TECH_STACK = [
  { icon: <FaRunning />, name: "YOLO11-Pose", desc: "Markerless keypoint extraction from RGB video." },
  { icon: <FaBrain />, name: "LSTM Classifier", desc: "Sequence model producing the Good/Poor lift-quality prediction." },
  { icon: <FaChartLine />, name: "Biomechanics (Rule A–D)", desc: "Sequential extension, hip dominance, synchronization, RFD approximation." },
  { icon: <FaMagic />, name: "Captum", desc: "Integrated Gradients explainability for the trained model (offline analysis)." },
  { icon: <FaServer />, name: "FastAPI", desc: "Production REST API serving the prediction pipeline (unmodified)." },
  { icon: <FaReact />, name: "React + TypeScript", desc: "This dashboard — Vite, Bootstrap 5, Recharts, Framer Motion." },
];

export default function AboutPage() {
  return (
    <div className="container about-page py-5">
      <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
        <span className="eyebrow">About This Project</span>
        <h2>Hip &amp; Knee Lift-Quality Analysis</h2>
      </motion.div>

      <GlassCard className="my-4">
        <h6 className="text-uppercase text-accent small fw-bold mb-2">Research Objective</h6>
        <p className="mb-0">
          To determine whether markerless, RGB-video-based pose estimation combined with rule-based
          biomechanical scoring and deep-learning classification can reliably distinguish good vs. poor lift
          technique for the hip and knee joints — without wearables or force-plate instrumentation — as a
          scalable, accessible coaching-support tool for weightlifting injury prevention.
        </p>
      </GlassCard>

      <h5 className="mt-5 mb-3">Technology Stack</h5>
      <div className="row g-4">
        {TECH_STACK.map((tech) => (
          <div className="col-md-4" key={tech.name}>
            <GlassCard className="h-100">
              <div className="tech-icon mb-2">{tech.icon}</div>
              <strong className="d-block mb-1">{tech.name}</strong>
              <p className="text-muted-2 small mb-0">{tech.desc}</p>
            </GlassCard>
          </div>
        ))}
      </div>

      <div className="row g-4 mt-2">
        <div className="col-md-6">
          <GlassCard className="h-100">
            <h6 className="text-uppercase text-accent small fw-bold mb-2">Limitations</h6>
            <ul className="mb-0">
              <li>Single side-view RGB camera — no depth or 3D pose triangulation.</li>
              <li>Rule D ("Explosive Movement") is an angular-acceleration proxy, not a direct force measurement.</li>
              <li>Trained on a modest labelled dataset; generalization to unseen lifting styles is limited.</li>
              <li>Annotated video output exists on the server but has no HTTP-servable route in the current API.</li>
              <li>Not a medical device and not an official IWF judging tool.</li>
            </ul>
          </GlassCard>
        </div>
        <div className="col-md-6">
          <GlassCard className="h-100">
            <h6 className="text-uppercase text-accent small fw-bold mb-2">Future Work</h6>
            <ul className="mb-0">
              <li>Expand the training dataset across more lifters, camera angles, and lift variations.</li>
              <li>Expose live, per-request Captum explainability through the API.</li>
              <li>Add a static-file/streaming route so annotated videos can be viewed in-browser.</li>
              <li>Multi-camera / 3D pose fusion for more robust joint-angle estimation.</li>
              <li>Automated test coverage and CI for the prediction pipeline.</li>
            </ul>
          </GlassCard>
        </div>
      </div>
    </div>
  );
}
