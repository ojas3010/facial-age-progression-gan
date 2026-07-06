import { useState } from 'react'

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

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImageFile(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setSelectedImage(reader.result);
      };
      reader.readAsDataURL(file);
      setResultImage(null); // Reset result on new upload
      setError("");
    }
  };

  const handleProcess = async () => {
    if (!imageFile) return;

    setLoading(true);
    setError("");
    
    const formData = new FormData();
    formData.append("file", imageFile);
    formData.append("target_age_group", targetAge.toString());

    try {
      const response = await fetch("http://localhost:8000/api/progress_age", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.message || "Failed to process image");
      }

      setResultImage(`data:image/jpeg;base64,${data.image_base64}`);
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
          
          <div className="upload-area">
            <input 
              type="file" 
              accept="image/*" 
              onChange={handleImageUpload} 
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
            <img src={resultImage} alt="Result" className="preview-image" />
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
