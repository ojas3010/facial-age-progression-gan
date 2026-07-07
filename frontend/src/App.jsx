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

  return (
    <>
      <h1>Aegis Ident</h1>
      <div className="subtitle">AI-Powered Facial Age Progression for Missing Persons</div>
      
      <div className="dashboard">
        {/* Left Panel: Input */}
        <div className="panel">
          <h2>Subject Input</h2>
          
          <div className="upload-area" onDrop={handleDrop} onDragOver={handleDragOver}>
            <input
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
              disabled={loading}
            />
            {selectedImage ? (
              <img src={selectedImage} alt="Subject" className="preview-image" />
            ) : (
              <div style={{color: '#9ca3af'}}>
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{marginBottom: '10px'}}>
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                  <polyline points="17 8 12 3 7 8"></polyline>
                  <line x1="12" y1="3" x2="12" y2="15"></line>
                </svg>
                <p>Click or drag image to upload</p>
              </div>
            )}
          </div>

          <div className="controls">
            <div className="slider-container">
              <label>Target Age Group: {AGE_GROUPS[targetAge]}</label>
              <input
                type="range"
                min="0"
                max="5"
                step="1"
                value={targetAge}
                onChange={(e) => setTargetAge(parseInt(e.target.value))}
                disabled={loading}
              />
              <div className="age-labels">
                <span>Infant</span>
                <span>Adult</span>
                <span>Senior</span>
              </div>
            </div>
            
            <button 
              onClick={handleProcess} 
              disabled={!selectedImage || loading}
            >
              {loading ? "Processing..." : "Generate Progression"}
            </button>
            {error && <p style={{color: '#ef4444', marginTop: '10px', fontSize: '0.9rem'}}>{error}</p>}
          </div>
        </div>

        {/* Right Panel: Output */}
        <div className="panel">
          <h2>Prediction Result</h2>
          
          {loading ? (
            <div className="result-placeholder">
              <div className="loader"></div>
              <p>Analyzing facial structures...</p>
              <p style={{fontSize: '0.8rem', color: '#6b7280'}}>Applying GAN transformations</p>
            </div>
          ) : resultImage ? (
            <>
              <img src={resultImage} alt="Result" className="preview-image" />
              <div className="result-actions">
                {elapsedMs !== null && (
                  <span className="latency-badge">
                    Synthesized in {(elapsedMs / 1000).toFixed(2)}s — {AGE_GROUPS[resultAge]}
                  </span>
                )}
                <a href={resultImage} download="age_progression.jpg" className="download-link">
                  Download Result
                </a>
              </div>
            </>
          ) : (
            <div className="result-placeholder">
              <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round" style={{opacity: 0.3}}>
                <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
                <circle cx="12" cy="7" r="4"></circle>
              </svg>
              <p>Result will appear here</p>
            </div>
          )}
        </div>
      </div>
    </>
  )
}

export default App
