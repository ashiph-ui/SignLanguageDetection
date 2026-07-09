import React from 'react';
import './App.css';
import CameraComponent from './CameraComponent';

const GitHubIcon = () => (
  <svg viewBox="0 0 16 16" width="16" height="16" fill="currentColor" aria-hidden="true">
    <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z" />
  </svg>
);

const LinkedInIcon = () => (
  <svg viewBox="0 0 16 16" width="16" height="16" fill="currentColor" aria-hidden="true">
    <path d="M13.63 13.63h-2.37V9.92c0-.89-.02-2.03-1.24-2.03-1.24 0-1.43.97-1.43 1.96v3.78H6.22V6h2.28v1.04h.03c.32-.6 1.09-1.24 2.25-1.24 2.4 0 2.85 1.58 2.85 3.64v4.19ZM3.54 4.96a1.38 1.38 0 1 1 0-2.76 1.38 1.38 0 0 1 0 2.76Zm1.19 8.67H2.35V6h2.38v7.63ZM14.82 0H1.18C.53 0 0 .52 0 1.16v13.68C0 15.48.53 16 1.18 16h13.64c.65 0 1.18-.52 1.18-1.16V1.16C16 .52 15.47 0 14.82 0Z" />
  </svg>
);

function App() {
  return (
    <div className="app">

      <nav className="navbar">
        <a className="brand" href="#top">
          <span className="brand-mark">🤟</span> ASL&nbsp;Detect
        </a>
        <ul className="nav-links">
          <li>
            <a href="https://github.com/ashiph-ui" target="_blank" rel="noreferrer">
              <GitHubIcon /> GitHub
            </a>
          </li>
          <li>
            <a href="https://www.linkedin.com/in/ashiph-rai-75609b217" target="_blank" rel="noreferrer">
              <LinkedInIcon /> LinkedIn
            </a>
          </li>
          <li>
            <a href="https://github.com/ashiph-ui?tab=repositories" target="_blank" rel="noreferrer">
              Other Projects
            </a>
          </li>
          <li>
            <a className="nav-cta" href="mailto:ashiphrai.work@gmail.com">Contact me</a>
          </li>
        </ul>
      </nav>

      <header className="hero" id="top">
        <span className="status-pill">
          <span className="status-dot" /> Model ready
        </span>
        <h1>
          Sign Language <span className="accent-text">Detection</span>
        </h1>
        <p className="hero-sub">
          Real-time American Sign Language recognition powered by deep learning —
          show a sign to your camera and watch it translate live.
        </p>
      </header>

      <main className="container">

        <section className="camera-section card spotlight">
          <h2>Live Detection</h2>
          <p className="section-sub">Hold a fingerspelling sign in front of your camera.</p>
          <CameraComponent />
        </section>

        <div className="two-col">
          <section className="card">
            <h2>How it works</h2>
            <ol className="steps">
              <li>
                <span className="step-num">1</span>
                <div>
                  <strong>Show a sign</strong>
                  <p>Your webcam captures a frame every few seconds.</p>
                </div>
              </li>
              <li>
                <span className="step-num">2</span>
                <div>
                  <strong>Landmarks extracted</strong>
                  <p>MediaPipe finds 21 hand landmarks in 3D space.</p>
                </div>
              </li>
              <li>
                <span className="step-num">3</span>
                <div>
                  <strong>Model predicts</strong>
                  <p>A PyTorch neural network classifies the letter A–Z.</p>
                </div>
              </li>
            </ol>
          </section>

          <section className="card">
            <h2>About this project</h2>
            <p className="about-text">
              A personal deep-learning project by Ashiph Rai. The frontend streams
              webcam frames to a FastAPI backend, where a MediaPipe + PyTorch
              pipeline recognises ASL fingerspelling in real time.
            </p>
            <div className="badges">
              <span className="badge">React</span>
              <span className="badge">FastAPI</span>
              <span className="badge">PyTorch</span>
              <span className="badge">MediaPipe</span>
              <span className="badge">OpenCV</span>
            </div>
          </section>
        </div>

        <section className="card reference">
          <h2>ASL Fingerspelling Reference</h2>
          <p className="section-sub">
            New to fingerspelling? Keep a chart handy while you practise.
          </p>
          {/* Drop an image at public/asl-chart.jpg and swap this link for:
              <img src="/asl-chart.jpg" alt="ASL fingerspelling chart" className="reference-img" /> */}
          <a
            className="reference-link"
            href="https://en.wikipedia.org/wiki/American_manual_alphabet"
            target="_blank"
            rel="noreferrer"
          >
            View the American Manual Alphabet →
          </a>
        </section>

      </main>

      <footer className="footer">
        <p>© {new Date().getFullYear()} Ashiph Rai · Built with React &amp; PyTorch</p>
      </footer>

    </div>
  );
}

export default App;
