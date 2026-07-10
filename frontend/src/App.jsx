import { useState } from 'react'

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

const AGE_GROUPS = [
  "0 - 10 years",
  "11 - 20 years",
  "21 - 30 years",
  "31 - 40 years",
  "41 - 50 years",
  "50+ years"
];

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [targetAge, setTargetAge] = useState(2); // Default to 21-30
  const [resultImage, setResultImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [elapsedMs, setElapsedMs] = useState(null);
  const [resultAge, setResultAge] = useState(null);
  const [dragActive, setDragActive] = useState(false); // presentational-only: dropzone drag-hover styling

  const loadFile = (file) => {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setError("Please upload an image file (JPEG, PNG or WEBP).");
      return;
    }
    setImageFile(file);
    const reader = new FileReader();
    reader.onloadend = () => {
      setSelectedImage(reader.result);
    };
    reader.onerror = () => {
      setError("Could not read the selected file.");
    };
    reader.readAsDataURL(file);
    setResultImage(null); // Reset result on new upload
    setError("");
  };

  const handleImageUpload = (e) => {
    loadFile(e.target.files[0]);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (loading) return; // a swap mid-request would mismatch input and result
    loadFile(e.dataTransfer.files[0]);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDragEnter = (e) => {
    e.preventDefault();
    setDragActive(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setDragActive(false);
  };

  const handleProcess = async () => {
    if (!imageFile) return;

    const requestAge = targetAge; // pin to the value at request time

    setLoading(true);
    setError("");
    setElapsedMs(null);

    const formData = new FormData();
    formData.append("file", imageFile);
    formData.append("target_age_group", requestAge.toString());

    const t0 = performance.now();
    try {
      const response = await fetch(`${API_URL}/api/progress_age`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        // Error bodies may be non-JSON (proxy errors) or carry a non-string
        // detail (FastAPI validation errors return an array)
        let message = "Failed to process image";
        try {
          const errData = await response.json();
          if (typeof errData.detail === "string") message = errData.detail;
          else if (typeof errData.message === "string") message = errData.message;
        } catch { /* keep generic message */ }
        throw new Error(message);
      }

      const data = await response.json();
      setResultImage(`data:image/jpeg;base64,${data.image_base64}`);
      setResultAge(requestAge);
      setElapsedMs(Math.round(performance.now() - t0));
    } catch (err) {
      setError(err.message || "Could not connect to the processing server.");
    } finally {
      setLoading(false);
    }
  };

  const status = error ? "ERROR" : loading ? "WORKING" : "READY";
  const statusClass = error ? "status-error" : loading ? "status-working" : "status-ready";

  return (
    <>
      <svg className="visually-hidden" aria-hidden="true" focusable="false">
        <symbol id="icon-corner-brackets" viewBox="0 0 48 48">
          <path d="M4 14V4h10" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          <path d="M44 14V4H34" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          <path d="M4 34v10h10" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          <path d="M44 34v10H34" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        </symbol>
        <symbol id="icon-crosshair" viewBox="0 0 48 48">
          <circle cx="24" cy="24" r="16" fill="none" stroke="currentColor" strokeWidth="1.5" />
          <path d="M24 2v10M24 36v10M2 24h10M36 24h10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
        </symbol>
      </svg>

      <a href="#console" className="skip-link">Skip to content</a>

      <main className="console" id="console">
        <header className="masthead">
          <div className="masthead-id">
            <h1 className="wordmark">Aegis Ident</h1>
            <p className="classification">Age progression for missing-persons casework</p>
          </div>
          <dl className="meta-row">
            <div className="meta-item">
              <dt>Model</dt>
              <dd>SAM-GAN</dd>
            </div>
            <div className="meta-item">
              <dt>Status</dt>
              <dd className={statusClass}>{status}</dd>
            </div>
          </dl>
        </header>

        <section className="grid">
          {/* Bay 01: Intake */}
          <article className="bay bay--intake">
            <div className="bay-head">
              <span className="panel-index" aria-hidden="true">01</span>
              <h2 className="panel-title">Subject &middot; Intake</h2>
            </div>

            <div className="field-group">
              <div
                className={`dropzone${dragActive ? " dropzone--active" : ""}`}
                onDrop={(e) => { setDragActive(false); handleDrop(e); }}
                onDragOver={handleDragOver}
                onDragEnter={handleDragEnter}
                onDragLeave={handleDragLeave}
              >
                <input
                  type="file"
                  id="subject-file"
                  className="dropzone-input"
                  accept="image/*"
                  onChange={handleImageUpload}
                  disabled={loading}
                />
                <label htmlFor="subject-file" className="dropzone-surface">
                  {selectedImage ? (
                    <img
                      src={selectedImage}
                      alt="Uploaded subject photograph"
                      className="preview"
                      width="600"
                      height="400"
                    />
                  ) : (
                    <>
                      <svg className="icon dropzone-icon" aria-hidden="true">
                        <use href="#icon-corner-brackets" />
                      </svg>
                      <p className="dropzone-text">Drop subject image &middot; or click to browse</p>
                      <p className="dropzone-caption">JPG &middot; PNG &middot; WEBP</p>
                    </>
                  )}
                </label>
              </div>
            </div>

            <div className="field-group">
              <span className="value">
                Target &middot; {AGE_GROUPS[targetAge].replace(" years", "").replace(" - ", "–")} yr
              </span>
              <fieldset className="age-segments" disabled={loading}>
                <legend className="visually-hidden">Target age group</legend>
                {AGE_GROUPS.map((group, i) => {
                  const short = group.replace(" years", "").replace(" - ", "–");
                  return (
                    <label
                      key={group}
                      className={`segment${targetAge === i ? " segment--selected" : ""}`}
                    >
                      <input
                        type="radio"
                        className="segment-input"
                        name="target-age-group"
                        value={i}
                        checked={targetAge === i}
                        onChange={() => setTargetAge(i)}
                      />
                      <span className="segment-text">{short}</span>
                    </label>
                  );
                })}
              </fieldset>
            </div>

            <button
              type="button"
              className="btn"
              onClick={handleProcess}
              disabled={!selectedImage || loading}
            >
              {loading ? (
                <>
                  Processing&hellip;
                  <span className="btn-tick" aria-hidden="true">
                    <span></span><span></span><span></span>
                  </span>
                </>
              ) : (
                "Run progression"
              )}
            </button>

            {error && (
              <div className="error" role="alert" aria-live="assertive">
                <span className="label">Error</span>
                <p>{error}</p>
              </div>
            )}
          </article>

          {/* Bay 02: Synthesis */}
          <article className="bay bay--synth">
            <div className="bay-head">
              <span className="panel-index" aria-hidden="true">02</span>
              <h2 className="panel-title">Progression &middot; Synthesis</h2>
            </div>

            <div className="synth-region" aria-live="polite" aria-busy={loading}>
              {loading ? (
                <div className="state state--loading">
                  <span className="label">Synthesizing &middot; GAN inference</span>
                  <div className="telemetry-bar" aria-hidden="true">
                    <div className="telemetry-bar-fill"></div>
                  </div>
                </div>
              ) : resultImage ? (
                <div className="state state--result">
                  <img
                    src={resultImage}
                    alt={`Age-progressed result, ${AGE_GROUPS[resultAge]}`}
                    className="result"
                    width="600"
                    height="400"
                  />
                  {elapsedMs !== null && (
                    <dl className="readout">
                      <div className="readout-row">
                        <dt>Synthesized</dt>
                        <dd>
                          <span className="readout-figure">{(elapsedMs / 1000).toFixed(2)}S</span>
                          {" · "}
                          {AGE_GROUPS[resultAge]}
                        </dd>
                      </div>
                    </dl>
                  )}
                  <a href={resultImage} download="age_progression.jpg" className="download">
                    Download <span aria-hidden="true">&darr;</span>
                  </a>
                </div>
              ) : (
                <div className="state state--empty">
                  <svg className="icon icon-lg state-icon" aria-hidden="true">
                    <use href="#icon-crosshair" />
                  </svg>
                  <span className="label">Awaiting synthesis</span>
                  <p className="helper">Select a subject and target age, then run.</p>
                </div>
              )}
            </div>
          </article>
        </section>

        <footer className="baseline">
          <p>Images processed in-session &middot; not stored</p>
          <p>Academic showcase build &middot; not for operational deployment</p>
        </footer>
      </main>
    </>
  )
}

export default App
