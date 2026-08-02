import { NavLink, useNavigate } from "react-router-dom";
import { FaDumbbell, FaCircle } from "react-icons/fa";
import { useHealthCheck } from "../../hooks/useHealthCheck";
import { useAnalysis } from "../../context/AnalysisContext";

const STATUS_META: Record<string, { label: string; className: string }> = {
  ok: { label: "Backend Online", className: "text-success" },
  starting: { label: "Backend Starting…", className: "text-warning" },
  unreachable: { label: "Backend Unavailable", className: "text-danger" },
  unknown: { label: "Checking Backend…", className: "text-secondary" },
};

export default function AppNavbar() {
  const backendStatus = useHealthCheck();
  const { reset, result } = useAnalysis();
  const navigate = useNavigate();
  const meta = STATUS_META[backendStatus] ?? STATUS_META.unknown;

  function handleBrandClick() {
    reset();
    navigate("/");
  }

  return (
    <nav className="navbar navbar-expand-lg navbar-dark app-navbar sticky-top">
      <div className="container">
        <button className="navbar-brand btn btn-link p-0 d-flex align-items-center gap-2 text-decoration-none" onClick={handleBrandClick}>
          <span className="brand-mark">
            <FaDumbbell />
          </span>
          <span className="brand-text">
            Hip &amp; Knee <span className="text-accent">Lift Analytics</span>
          </span>
        </button>

        <button
          className="navbar-toggler"
          type="button"
          data-bs-toggle="collapse"
          data-bs-target="#mainNav"
          aria-controls="mainNav"
          aria-expanded="false"
          aria-label="Toggle navigation"
        >
          <span className="navbar-toggler-icon" />
        </button>

        <div className="collapse navbar-collapse" id="mainNav">
          <ul className="navbar-nav ms-auto align-items-lg-center gap-lg-2">
            <li className="nav-item">
              <NavLink to="/" end className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}>
                Home
              </NavLink>
            </li>
            <li className="nav-item">
              <NavLink to="/upload" className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}>
                Upload
              </NavLink>
            </li>
            <li className="nav-item">
              <NavLink
                to="/results"
                className={({ isActive }) => `nav-link ${isActive ? "active" : ""} ${!result ? "disabled" : ""}`}
                onClick={(e) => {
                  if (!result) e.preventDefault();
                }}
              >
                Results
              </NavLink>
            </li>
            <li className="nav-item">
              <NavLink to="/about" className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}>
                About
              </NavLink>
            </li>
            <li className="nav-item ms-lg-3 mt-2 mt-lg-0">
              <span className={`status-pill ${meta.className}`}>
                <FaCircle size={8} className="me-1" />
                {meta.label}
              </span>
            </li>
          </ul>
        </div>
      </div>
    </nav>
  );
}
