import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import { FaArrowRight, FaVideo, FaRunning, FaBrain, FaChartLine, FaLightbulb } from "react-icons/fa";
import GlassCard from "../components/common/GlassCard";

const ARCHITECTURE_STEPS = [
  { icon: <FaVideo />, label: "Video Upload", desc: "User uploads a side-view lift clip (.mp4/.avi/.mov)" },
  { icon: <FaRunning />, label: "YOLO11 Pose Estimation", desc: "Markerless keypoint extraction, frame by frame" },
  { icon: <FaChartLine />, label: "Rule A–D Biomechanics", desc: "Sequential extension, hip dominance, sync, RFD approx." },
  { icon: <FaBrain />, label: "LSTM Classification", desc: "Confidence-weighted Good / Poor prediction" },
  { icon: <FaLightbulb />, label: "Explainability", desc: "Rule contribution breakdown & coaching-style feedback" },
];

export default function LandingPage() {
  const navigate = useNavigate();

  return (
    <div className="landing-page">
      <section className="hero-section">
        <div className="container">
          <div className="row align-items-center gy-5">
            <motion.div
              className="col-lg-7"
              initial={{ opacity: 0, x: -24 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, ease: "easeOut" }}
            >
              <span className="eyebrow">Final-Year Research Project — Demonstration Interface</span>
              <h1 className="hero-title">
                Hip &amp; Knee <span className="text-accent">Lift-Quality</span> Analysis
              </h1>
              <p className="hero-subtitle">
                Markerless video biomechanics for weightlifting technique assessment.
              </p>
              <p className="hero-description">
                This system runs a fully validated, already-finalized backend pipeline — YOLO11 pose
                estimation, anthropometric normalization, Rule A–D biomechanical analysis, confidence-weighted
                scoring, and an LSTM classifier — over a single uploaded lift video, then presents the results
                as an interactive analytics dashboard with explainable-AI feedback.
              </p>

              <GlassCard className="research-objective-card mb-4" delay={0.15}>
                <h6 className="text-uppercase text-accent small fw-bold mb-2">Research Objective</h6>
                <p className="mb-0">
                  To determine whether markerless, RGB-video-based pose estimation combined with rule-based
                  biomechanical scoring and deep-learning classification can reliably distinguish good vs. poor
                  lift technique for the hip and knee joints — without wearables or force-plate instrumentation
                  — as a scalable, accessible coaching-support tool.
                </p>
              </GlassCard>

              <div className="d-flex gap-3 flex-wrap">
                <button className="btn btn-accent btn-lg d-inline-flex align-items-center gap-2" onClick={() => navigate("/upload")}>
                  Start Analysis <FaArrowRight />
                </button>
                <button className="btn btn-outline-light btn-lg" onClick={() => navigate("/about")}>
                  Learn More
                </button>
              </div>

              <p className="disclaimer-text mt-4">
                Research prototype for a final-year dissertation. Not a medical device, not an official IWF
                judging tool, and not a substitute for qualified coaching.
              </p>
            </motion.div>

            <motion.div
              className="col-lg-5"
              initial={{ opacity: 0, scale: 0.94 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.6, delay: 0.2, ease: "easeOut" }}
            >
              <GlassCard className="architecture-card">
                <h6 className="text-uppercase text-accent small fw-bold mb-3">System Architecture</h6>
                <div className="architecture-flow">
                  {ARCHITECTURE_STEPS.map((step, i) => (
                    <div className="architecture-step" key={step.label}>
                      <div className="architecture-step-icon">{step.icon}</div>
                      <div className="architecture-step-text">
                        <strong>{step.label}</strong>
                        <span>{step.desc}</span>
                      </div>
                      {i < ARCHITECTURE_STEPS.length - 1 && <div className="architecture-connector" />}
                    </div>
                  ))}
                </div>
              </GlassCard>
            </motion.div>
          </div>
        </div>
      </section>
    </div>
  );
}
