import { FaGithub } from "react-icons/fa";

export default function AppFooter() {
  const year = new Date().getFullYear();
  return (
    <footer className="app-footer mt-auto">
      <div className="container">
        <div className="row gy-4">
          <div className="col-md-4">
            <h6 className="footer-heading">Project</h6>
            <p className="footer-text">
              Hip &amp; Knee Lift-Quality Analysis — a markerless video biomechanics system for
              weightlifting technique assessment, developed as a Final Year Project.
            </p>
          </div>
          <div className="col-md-4">
            <h6 className="footer-heading">Academic Details</h6>
            <ul className="footer-list">
              <li>
                <span className="footer-label">University:</span> [Your University Name]
              </li>
              <li>
                <span className="footer-label">Research Project:</span> Weight-Lifting Injuries Prediction
              </li>
              <li>
                <span className="footer-label">Supervisor:</span> [Supervisor Name]
              </li>
            </ul>
          </div>
          <div className="col-md-4">
            <h6 className="footer-heading">Info</h6>
            <ul className="footer-list">
              <li>
                <span className="footer-label">Authors:</span> [Author Name(s)]
              </li>
              <li>
                <span className="footer-label">Version:</span> 2.0.0
              </li>
              <li>
                <a
                  href="https://github.com/"
                  target="_blank"
                  rel="noreferrer"
                  className="footer-link d-inline-flex align-items-center gap-1"
                >
                  <FaGithub /> Repository
                </a>
              </li>
            </ul>
          </div>
        </div>
        <hr className="footer-divider" />
        <p className="footer-copyright">
          © {year} Hip &amp; Knee Lift Analytics. Research prototype — not a medical device, not an
          official IWF judging tool.
        </p>
      </div>
    </footer>
  );
}
