import React from 'react';
import './App.css';
import CameraComponent from './CameraComponent';

function App() {
  return (
    <div className="container">

      <header className="title card">
        <h1>Ashiph's Signlanguage Project</h1>
      </header>

      <nav className="links card">
        <ul>
          <li><a href="https://github.com/ashiph-ui"target="_blank">GitHub</a></li>
          <li><a href="www.linkedin.com/in/ashiph-rai-75609b217"target="_blank">LinkedIn</a></li>
          <li><a href="#">Other Projects</a></li>
          <li><a href="#">Contact me</a></li>
        </ul>
      </nav>

      <section className="about card">
        <p>About us</p>
      </section>

      <section className="camera card">
        <CameraComponent />
      </section>

      <section className="reference-image card">
        <p>ASL Reference Image</p>
        {/* Optional: Add an image here */}
        {/* <img src="/asl-chart.jpg" alt="ASL Chart" style={{ width: "100%", borderRadius: "8px" }} /> */}
      </section>

    </div>
  );
}

export default App;
