
import { useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

const initialClinicalData = {
  serum_creatinine: "",
  gfr: "",
  bun: "",
  serum_calcium: "",
  oxalate_levels: "",
  urine_ph: "",
  blood_pressure: "",
  ana: "",
  c3_c4: "",
  hematuria: "",
  smoking: "",
  alcohol: "",
  painkiller_usage: "",
  family_history: "",
  physical_activity: "",
  diet: "",
  water_intake: "",
  weight_changes: "",
  stress_level: "",
  months: "",
};

function App() {
  const [page, setPage] = useState(1);

  const [patient, setPatient] = useState({
    patientId: "",
    patientName: "",
    age: "",
    gender: "",
  });

  const [clinicalData, setClinicalData] = useState(initialClinicalData);

  const [ultrasound, setUltrasound] = useState(null);
  const [preview, setPreview] = useState("");

  const [clinicalResult, setClinicalResult] = useState(null);
  const [finalResult, setFinalResult] = useState(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handlePatientChange = (e) => {
    setPatient({
      ...patient,
      [e.target.name]: e.target.value,
    });
  };

  const handleClinicalChange = (e) => {
    setClinicalData({
      ...clinicalData,
      [e.target.name]: e.target.value,
    });
  };

  const handleImageChange = (e) => {
    const file = e.target.files?.[0];

    if (!file) return;

    setUltrasound(file);
    setPreview(URL.createObjectURL(file));
    setError("");
  };

  const validatePageOne = () => {
    if (
      !patient.patientId ||
      !patient.patientName ||
      !patient.age ||
      !patient.gender
    ) {
      setError("Please complete all patient information.");
      return false;
    }

    const missingClinical = Object.values(clinicalData).some(
      (value) => value === ""
    );

    if (missingClinical) {
      setError("Please enter all clinical parameters before continuing.");
      return false;
    }

    setError("");
    return true;
  };

  const goToUltrasound = () => {
    if (validatePageOne()) {
      setPage(2);
      window.scrollTo({ top: 0, behavior: "smooth" });
    }
  };

  const convertClinicalData = () => {
    const converted = {};

    Object.entries(clinicalData).forEach(([key, value]) => {
      converted[key] = Number(value);
    });

    return converted;
  };

  const generateResults = async () => {
    if (!ultrasound) {
      setError("Please upload a kidney ultrasound image.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const clinicalPayload = {
        ...convertClinicalData(),
      };

      /*
       * STEP 1
       * Send clinical data to FastAPI.
       */
      const clinicalResponse = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(clinicalPayload),
      });

      if (!clinicalResponse.ok) {
        throw new Error("Clinical prediction failed.");
      }

      const clinical = await clinicalResponse.json();

      setClinicalResult(clinical);

      /*
       * STEP 2
       * Send clinical data + ultrasound to the fusion endpoint.
       *
       * If U-Net is unavailable on the backend, the backend will return
       * an error. We keep the clinical result and clearly show the
       * ultrasound status instead of inventing an image prediction.
       */
      let fusion = null;

      try {
        const formData = new FormData();

        formData.append(
          "patient",
          JSON.stringify(clinicalPayload)
        );

        formData.append("file", ultrasound);

        const fusionResponse = await fetch(
          `${API_URL}/fusion-predict`,
          {
            method: "POST",
            body: formData,
          }
        );

        if (fusionResponse.ok) {
          fusion = await fusionResponse.json();
        }
      } catch (imageError) {
        console.log("Ultrasound analysis unavailable:", imageError);
      }

      /*
       * STEP 3
       * Prepare dashboard result.
       */
      const clinicalPrediction =
        clinical.prediction ||
        clinical.result ||
        clinical.ckd_prediction ||
        "Unknown";

      const clinicalConfidence = Number(
        clinical.confidence ?? 0
      );

      const clinicalStage =
        clinical.stage ??
        clinical.ckd_stage ??
        null;

      const clinicalRisk =
        clinical.risk ||
        clinical.severity ||
        "Not available";

      const imageProbability =
        fusion?.image_probability ?? null;

      const finalProbability =
        fusion?.final_probability ??
        clinicalConfidence;

      const finalPrediction =
        fusion?.final_prediction ??
        clinicalPrediction;

      const finalStage =
        fusion?.stage ??
        clinicalStage;

      const finalSeverity =
        fusion?.severity ??
        clinicalRisk;

      setFinalResult({
        prediction: finalPrediction,
        stage: finalStage,
        severity: finalSeverity,
        confidence: Number(finalProbability),
        clinicalProbability: clinicalConfidence,
        imageProbability,
        fusionAvailable: Boolean(fusion),
        timestamp: new Date().toLocaleString(),
      });

      setPage(3);
      window.scrollTo({ top: 0, behavior: "smooth" });
    } catch (err) {
      setError(
        err.message ||
          "Unable to connect to the NephroPredictor backend."
      );
    } finally {
      setLoading(false);
    }
  };

  const startNewAssessment = () => {
    setPage(1);

    setPatient({
      patientId: "",
      patientName: "",
      age: "",
      gender: "",
    });

    setClinicalData(initialClinicalData);

    setUltrasound(null);
    setPreview("");

    setClinicalResult(null);
    setFinalResult(null);

    setError("");

    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const getStatusClass = (prediction) => {
    if (!prediction) return "neutral";

    const value = prediction.toLowerCase();

    if (
      value.includes("detected") ||
      value.includes("positive") ||
      value.includes("ckd")
    ) {
      return "danger";
    }

    if (
      value.includes("healthy") ||
      value.includes("negative")
    ) {
      return "success";
    }

    return "neutral";
  };

  const getSeverityClass = (severity) => {
    if (!severity) return "neutral";

    const value = String(severity).toLowerCase();

    if (value.includes("high") || value.includes("severe")) {
      return "danger";
    }

    if (
      value.includes("moderate") ||
      value.includes("medium")
    ) {
      return "warning";
    }

    if (value.includes("low") || value.includes("mild")) {
      return "success";
    }

    return "neutral";
  };

  const probability =
    finalResult?.confidence != null
      ? Math.max(
          0,
          Math.min(100, Number(finalResult.confidence))
        )
      : 0;

  return (
    <div className="app">

      {/* =====================================================
          HEADER
          ===================================================== */}

      <header className="header">
        <div className="header-content">

          <div className="logo-icon">
            NP
          </div>

          <div>
            <h1>NephroPredictor</h1>
            <p>
              AI-Assisted Chronic Kidney Disease Assessment
            </p>
          </div>

        </div>
      </header>


      <main className="container">

        {/* ===================================================
            WORKFLOW
            =================================================== */}

        <div className="workflow">

          <div
            className={`workflow-step ${
              page === 1 ? "active" : "completed"
            }`}
          >
            <span>1</span>
            <p>Clinical Data</p>
          </div>

          <div
            className={`workflow-line ${
              page > 1 ? "completed" : ""
            }`}
          />

          <div
            className={`workflow-step ${
              page === 2
                ? "active"
                : page > 2
                ? "completed"
                : ""
            }`}
          >
            <span>2</span>
            <p>Ultrasound</p>
          </div>

          <div
            className={`workflow-line ${
              page > 2 ? "completed" : ""
            }`}
          />

          <div
            className={`workflow-step ${
              page === 3 ? "active" : ""
            }`}
          >
            <span>3</span>
            <p>Results Dashboard</p>
          </div>

        </div>


        {/* ===================================================
            ERROR
            =================================================== */}

        {error && (
          <div className="error-card">
            <strong>Attention</strong>
            <p>{error}</p>
          </div>
        )}


        {/* ===================================================
            PAGE 1
            =================================================== */}

        {page === 1 && (
          <>
            <section className="intro-card">
              <div className="intro-icon">
                +
              </div>

              <div>
                <h2>Patient & Clinical Assessment</h2>

                <p>
                  Enter the patient's demographic and clinical
                  information to begin the CKD assessment.
                  All required parameters should be completed
                  before proceeding to ultrasound analysis.
                </p>
              </div>
            </section>


            <section className="form-card">

              <div className="section-heading">
                <div>
                  <h2>Patient Information</h2>
                  <p>
                    Basic patient identification details
                  </p>
                </div>

                <span>STEP 1 OF 3</span>
              </div>


              <div className="form-grid">

                <div className="field">
                  <label>Patient ID</label>

                  <input
                    name="patientId"
                    value={patient.patientId}
                    onChange={handlePatientChange}
                    placeholder="Enter patient ID"
                  />
                </div>


                <div className="field">
                  <label>Patient Name</label>

                  <input
                    name="patientName"
                    value={patient.patientName}
                    onChange={handlePatientChange}
                    placeholder="Enter patient name"
                  />
                </div>


                <div className="field">
                  <label>Age</label>

                  <input
                    type="number"
                    name="age"
                    min="0"
                    max="120"
                    value={patient.age}
                    onChange={handlePatientChange}
                    placeholder="Age"
                  />
                </div>


                <div className="field">
                  <label>Gender</label>

                  <select
                    name="gender"
                    value={patient.gender}
                    onChange={handlePatientChange}
                  >
                    <option value="">
                      Select gender
                    </option>

                    <option value="Male">
                      Male
                    </option>

                    <option value="Female">
                      Female
                    </option>

                    <option value="Other">
                      Other
                    </option>
                  </select>
                </div>

              </div>


              <div className="section-heading clinical-heading">

                <div>
                  <h2>Clinical Parameters</h2>

                  <p>
                    Enter the patient's clinical assessment
                    values.
                  </p>
                </div>

                <span>20 PARAMETERS</span>

              </div>


              <div className="form-grid">

                <ClinicalField
                  label="Serum Creatinine"
                  name="serum_creatinine"
                  value={clinicalData.serum_creatinine}
                  onChange={handleClinicalChange}
                  placeholder="e.g. 1.2"
                />

                <ClinicalField
                  label="GFR"
                  name="gfr"
                  value={clinicalData.gfr}
                  onChange={handleClinicalChange}
                  placeholder="e.g. 60"
                />

                <ClinicalField
                  label="BUN"
                  name="bun"
                  value={clinicalData.bun}
                  onChange={handleClinicalChange}
                  placeholder="e.g. 25"
                />

                <ClinicalField
                  label="Serum Calcium"
                  name="serum_calcium"
                  value={clinicalData.serum_calcium}
                  onChange={handleClinicalChange}
                  placeholder="Enter value"
                />

                <ClinicalField
                  label="Oxalate Levels"
                  name="oxalate_levels"
                  value={clinicalData.oxalate_levels}
                  onChange={handleClinicalChange}
                  placeholder="Enter value"
                />

                <ClinicalField
                  label="Urine pH"
                  name="urine_ph"
                  value={clinicalData.urine_ph}
                  onChange={handleClinicalChange}
                  placeholder="0 - 14"
                />

                <ClinicalField
                  label="Blood Pressure"
                  name="blood_pressure"
                  value={clinicalData.blood_pressure}
                  onChange={handleClinicalChange}
                  placeholder="e.g. 120"
                />

                <ClinicalField
                  label="ANA"
                  name="ana"
                  value={clinicalData.ana}
                  onChange={handleClinicalChange}
                  placeholder="Enter value"
                />

                <ClinicalField
                  label="C3 / C4"
                  name="c3_c4"
                  value={clinicalData.c3_c4}
                  onChange={handleClinicalChange}
                  placeholder="Enter value"
                />

                <ClinicalField
                  label="Hematuria"
                  name="hematuria"
                  value={clinicalData.hematuria}
                  onChange={handleClinicalChange}
                  placeholder="0 / 1"
                />

                <ClinicalField
                  label="Smoking"
                  name="smoking"
                  value={clinicalData.smoking}
                  onChange={handleClinicalChange}
                  placeholder="0 / 1"
                />

                <ClinicalField
                  label="Alcohol"
                  name="alcohol"
                  value={clinicalData.alcohol}
                  onChange={handleClinicalChange}
                  placeholder="0 / 1"
                />

                <ClinicalField
                  label="Painkiller Usage"
                  name="painkiller_usage"
                  value={clinicalData.painkiller_usage}
                  onChange={handleClinicalChange}
                  placeholder="0 / 1"
                />

                <ClinicalField
                  label="Family History"
                  name="family_history"
                  value={clinicalData.family_history}
                  onChange={handleClinicalChange}
                  placeholder="0 / 1"
                />

                <ClinicalField
                  label="Physical Activity"
                  name="physical_activity"
                  value={clinicalData.physical_activity}
                  onChange={handleClinicalChange}
                  placeholder="Enter value"
                />

                <ClinicalField
                  label="Diet"
                  name="diet"
                  value={clinicalData.diet}
                  onChange={handleClinicalChange}
                  placeholder="Enter value"
                />

                <ClinicalField
                  label="Water Intake"
                  name="water_intake"
                  value={clinicalData.water_intake}
                  onChange={handleClinicalChange}
                  placeholder="Enter value"
                />

                <ClinicalField
                  label="Weight Changes"
                  name="weight_changes"
                  value={clinicalData.weight_changes}
                  onChange={handleClinicalChange}
                  placeholder="Enter value"
                />

                <ClinicalField
                  label="Stress Level"
                  name="stress_level"
                  value={clinicalData.stress_level}
                  onChange={handleClinicalChange}
                  placeholder="Enter value"
                />

                <ClinicalField
                  label="Duration (Months)"
                  name="months"
                  value={clinicalData.months}
                  onChange={handleClinicalChange}
                  placeholder="Months"
                />

              </div>


              <div className="button-row">

                <button
                  className="predict-button"
                  onClick={goToUltrasound}
                >
                  Continue to Ultrasound
                  <span> →</span>
                </button>

              </div>


              <div className="research-note">

                <strong>
                  Clinical Decision-Support System
                </strong>

                <p>
                  NephroPredictor is an AI-assisted research
                  and decision-support application. Results
                  should be reviewed by a qualified healthcare
                  professional.
                </p>

              </div>

            </section>
          </>
        )}


        {/* ===================================================
            PAGE 2
            =================================================== */}

        {page === 2 && (
          <>
            <section className="intro-card">
              <div className="intro-icon">
                US
              </div>

              <div>
                <h2>Kidney Ultrasound Analysis</h2>

                <p>
                  Upload the patient's kidney ultrasound image
                  for image-based analysis and integration with
                  the clinical assessment.
                </p>
              </div>
            </section>


            <section className="image-card">

              <div className="section-heading">

                <div>
                  <h2>Ultrasound Image</h2>

                  <p>
                    Upload a supported kidney ultrasound image.
                  </p>
                </div>

                <span>STEP 2 OF 3</span>

              </div>


              <div className="patient-summary">

                <div>
                  <span>Patient ID</span>
                  <strong>{patient.patientId}</strong>
                </div>

                <div>
                  <span>Patient</span>
                  <strong>{patient.patientName}</strong>
                </div>

                <div>
                  <span>Age</span>
                  <strong>{patient.age}</strong>
                </div>

                <div>
                  <span>Gender</span>
                  <strong>{patient.gender}</strong>
                </div>

              </div>


              <div className="upload-area">

                <div className="upload-icon">
                  ↑
                </div>

                <h3>
                  Upload Kidney Ultrasound
                </h3>

                <p>
                  JPG, JPEG or PNG image
                </p>

                <input
                  type="file"
                  accept="image/png,image/jpeg,image/jpg"
                  onChange={handleImageChange}
                />

              </div>


              {preview && (
                <div className="image-preview">

                  <h3>
                    Selected Ultrasound
                  </h3>

                  <img
                    src={preview}
                    alt="Selected kidney ultrasound"
                  />

                  <p className="file-name">
                    {ultrasound?.name}
                  </p>

                </div>
              )}


              <div className="analysis-info">

                <h3>
                  Analysis Pipeline
                </h3>

                <div className="analysis-steps">

                  <div>
                    <span>1</span>
                    <p>
                      U-Net Kidney Segmentation
                    </p>
                  </div>

                  <div>
                    <span>2</span>
                    <p>
                      Image Feature Extraction
                    </p>
                  </div>

                  <div>
                    <span>3</span>
                    <p>
                      Image Classification
                    </p>
                  </div>

                  <div>
                    <span>4</span>
                    <p>
                      Clinical + Image Fusion
                    </p>
                  </div>

                </div>

                <p className="analysis-note">
                  The ultrasound model processes the image
                  independently and the final system can
                  combine image and clinical predictions.
                </p>

              </div>


              <div className="button-row">

                <button
                  className="clear-button"
                  onClick={() => {
                    setPage(1);
                    setError("");
                  }}
                >
                  ← Back
                </button>

                <button
                  className="predict-button"
                  disabled={!ultrasound || loading}
                  onClick={generateResults}
                >
                  {loading
                    ? "Analyzing Patient..."
                    : "Generate Results →"}
                </button>

              </div>

            </section>
          </>
        )}


        {/* ===================================================
            PAGE 3
            =================================================== */}

        {page === 3 && finalResult && (
          <>
            <section className="dashboard-header">

              <div>

                <span className="dashboard-label">
                  RESULTS DASHBOARD
                </span>

                <h2>
                  Patient Results Dashboard
                </h2>

                <p>
                  AI-assisted CKD assessment summary
                </p>

              </div>


              <button
                className="new-assessment-button"
                onClick={startNewAssessment}
              >
                + New Assessment
              </button>

            </section>


            {/* PATIENT INFORMATION */}

            <section className="dashboard-card">

              <div className="dashboard-card-header">

                <h3>
                  Patient Information
                </h3>

                <span className="model-tag">
                  PATIENT RECORD
                </span>

              </div>


              <div className="patient-summary">

                <div>
                  <span>Patient ID</span>
                  <strong>{patient.patientId}</strong>
                </div>

                <div>
                  <span>Name</span>
                  <strong>{patient.patientName}</strong>
                </div>

                <div>
                  <span>Age</span>
                  <strong>{patient.age} years</strong>
                </div>

                <div>
                  <span>Gender</span>
                  <strong>{patient.gender}</strong>
                </div>

              </div>

              <p className="assessment-time">
                Assessment generated: {finalResult.timestamp}
              </p>

            </section>


            {/* FOUR MAIN PARAMETERS */}

            <section className="dashboard-result-grid">

              <ResultCard
                title="CKD Status"
                value={finalResult.prediction}
                className={getStatusClass(
                  finalResult.prediction
                )}
              />

              <ResultCard
                title="CKD Stage"
                value={
                  finalResult.stage
                    ? `Stage ${finalResult.stage}`
                    : "Not Applicable"
                }
                className="blue"
              />

              <ResultCard
                title="Severity"
                value={
                  finalResult.severity || "Not Available"
                }
                className={getSeverityClass(
                  finalResult.severity
                )}
              />

              <ResultCard
                title="Confidence"
                value={`${probability.toFixed(1)}%`}
                className="purple"
              />

            </section>


            {/* CONFIDENCE VISUALIZATION */}

            <section className="dashboard-card">

              <div className="dashboard-card-header">

                <div>
                  <h3>
                    Assessment Confidence
                  </h3>

                  <p className="card-subtitle">
                    Combined prediction probability
                  </p>
                </div>

                <span className="model-tag">
                  AI ANALYSIS
                </span>

              </div>


              <div className="confidence-layout">

                <div
                  className="confidence-circle"
                  style={{
                    "--confidence": `${probability}%`,
                  }}
                >
                  <div>
                    <strong>
                      {probability.toFixed(1)}%
                    </strong>

                    <span>
                      Confidence
                    </span>
                  </div>
                </div>


                <div className="confidence-info">

                  <div className="metric-line">

                    <div>
                      <span>
                        Clinical Model
                      </span>

                      <strong>
                        {Number(
                          finalResult.clinicalProbability
                        ).toFixed(1)}
                        %
                      </strong>
                    </div>

                    <div className="progress">
                      <div
                        style={{
                          width: `${Math.min(
                            100,
                            Math.max(
                              0,
                              Number(
                                finalResult.clinicalProbability
                              )
                            )
                          )}%`,
                        }}
                      />
                    </div>

                  </div>


                  <div className="metric-line">

                    <div>
                      <span>
                        Ultrasound Model
                      </span>

                      <strong>
                        {finalResult.imageProbability !== null
                          ? `${Number(
                              finalResult.imageProbability
                            ).toFixed(1)}%`
                          : "Unavailable"}
                      </strong>
                    </div>

                    <div className="progress image-progress">
                      <div
                        style={{
                          width:
                            finalResult.imageProbability !==
                            null
                              ? `${Math.min(
                                  100,
                                  Math.max(
                                    0,
                                    Number(
                                      finalResult.imageProbability
                                    )
                                  )
                                )}%`
                              : "0%",
                        }}
                      />
                    </div>

                  </div>

                </div>

              </div>

            </section>


            {/* MODEL RESULTS */}

            <section className="dashboard-card">

              <div className="dashboard-card-header">

                <div>
                  <h3>
                    Model Assessment
                  </h3>

                  <p className="card-subtitle">
                    Individual model outputs
                  </p>
                </div>

              </div>


              <div className="model-result-grid">

                <div className="model-result clinical-model">

                  <div className="model-icon">
                    C
                  </div>

                  <div>
                    <span>
                      Clinical Analysis
                    </span>

                    <strong>
                      {clinicalResult?.prediction ||
                        finalResult.prediction}
                    </strong>

                    <small>
                      Random Forest clinical model
                    </small>
                  </div>

                </div>


                <div className="model-result ultrasound-model">

                  <div className="model-icon">
                    U
                  </div>

                  <div>
                    <span>
                      Ultrasound Analysis
                    </span>

                    <strong>
                      {finalResult.fusionAvailable
                        ? "Analyzed"
                        : "Pending"}
                    </strong>

                    <small>
                      U-Net + image classifier
                    </small>
                  </div>

                </div>

              </div>

            </section>


            {/* STAGE VISUALIZATION */}

            <section className="dashboard-card">

              <div className="dashboard-card-header">

                <div>
                  <h3>
                    CKD Stage Overview
                  </h3>

                  <p className="card-subtitle">
                    Five-stage clinical classification
                  </p>
                </div>

              </div>


              <div className="stage-chart">

                {[1, 2, 3, 4, 5].map((stage) => {

                  const currentStage =
                    Number(finalResult.stage);

                  const active =
                    currentStage === stage;

                  return (
                    <div
                      className={`stage-item ${
                        active ? "active" : ""
                      }`}
                      key={stage}
                    >

                      <div className="stage-bar">
                        <span />
                      </div>

                      <strong>
                        Stage {stage}
                      </strong>

                      <small>
                        {stage === 1 &&
                          "Mild / Normal"}
                        {stage === 2 &&
                          "Mild Reduction"}
                        {stage === 3 &&
                          "Moderate"}
                        {stage === 4 &&
                          "Severe"}
                        {stage === 5 &&
                          "Kidney Failure"}
                      </small>

                    </div>
                  );
                })}

              </div>

            </section>


            {/* FINAL ASSESSMENT */}

            <section className="final-assessment">

              <div>

                <span>
                  FINAL AI-ASSISTED ASSESSMENT
                </span>

                <h2>
                  {finalResult.prediction}
                </h2>

                <p>
                  The result combines available clinical
                  information and ultrasound analysis.
                  This output is intended to support,
                  not replace, professional clinical
                  evaluation.
                </p>

              </div>


              <div className="final-status">

                <span>
                  CURRENT STATUS
                </span>

                <strong>
                  {finalResult.prediction}
                </strong>

              </div>

            </section>


            {/* DISCLAIMER */}

            <div className="dashboard-disclaimer">

              <strong>
                Clinical Decision-Support Notice
              </strong>

              <p>
                NephroPredictor is an AI-assisted research
                and decision-support system. Results should
                be interpreted by a qualified healthcare
                professional together with the patient's
                clinical history, laboratory findings and
                imaging.
              </p>

            </div>


            <div className="dashboard-actions">

              <button
                className="clear-button"
                onClick={() => setPage(2)}
              >
                ← Review Ultrasound
              </button>

              <button
                className="predict-button"
                onClick={startNewAssessment}
              >
                Start New Assessment
              </button>

            </div>

          </>
        )}

      </main>


      <footer className="footer">
        NephroPredictor • AI-Assisted CKD Decision Support
        System
      </footer>

    </div>
  );
}


/* =========================================================
   CLINICAL FIELD COMPONENT
   ========================================================= */

function ClinicalField({
  label,
  name,
  value,
  onChange,
  placeholder,
}) {
  return (
    <div className="field">
      <label>{label}</label>

      <input
        type="number"
        step="any"
        name={name}
        value={value}
        onChange={onChange}
        placeholder={placeholder}
      />
    </div>
  );
}


/* =========================================================
   RESULT CARD
   ========================================================= */

function ResultCard({
  title,
  value,
  className,
}) {
  return (
    <div className={`result-card ${className}`}>

      <span>{title}</span>

      <strong>{value}</strong>

    </div>
  );
}

export default App;

