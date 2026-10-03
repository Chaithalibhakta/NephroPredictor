
import React, { useState } from "react";
import { jsPDF } from "jspdf";
import "./App.css";

const API_URL = "http://127.0.0.1:8001";

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

const demoCases = {
  "No CKD": {
    patient: {
      patientId: "DEMO-NO-CKD",
      patientName: "Demo Patient - No CKD",
      age: "40",
      gender: "Female",
    },
    clinical: {
      serum_creatinine: 0.8,
      gfr: 95,
      bun: 18,
      serum_calcium: 9.4,
      oxalate_levels: 1.2,
      urine_ph: 6.2,
      blood_pressure: 118,
      ana: 0,
      c3_c4: 1,
      hematuria: 0,
      smoking: 0,
      alcohol: 0,
      painkiller_usage: 0,
      family_history: 0,
      physical_activity: 2,
      diet: 0,
      water_intake: 3,
      weight_changes: 0,
      stress_level: 1,
      months: 6,
    },
  },

  "Stage 1": {
  patient: {
    patientId: "DEMO-STAGE-1",
    patientName: "Demo Patient - Stage 1",
    age: "42",
    gender: "Female",
  },
  clinical: {
    serum_creatinine: 0.3,
    gfr: 126.502590,
    bun: 44.286194,
    serum_calcium: 11.480861,
    oxalate_levels: 1.364023,
    urine_ph: 6.006007,
    blood_pressure: 121.636461,
    ana: 0,
    c3_c4: 1,
    hematuria: 1,
    smoking: 1,
    alcohol: 1,
    painkiller_usage: 0,
    family_history: 1,
    physical_activity: 2,
    diet: 0,
    water_intake: 3.189235,
    weight_changes: 1,
    stress_level: 1,
    months: 8,
  },
},
    

  "Stage 2": {
  patient: {
    patientId: "DEMO-STAGE-2",
    patientName: "Demo Patient - Stage 2",
    age: "48",
    gender: "Male",
  },
  clinical: {
    serum_creatinine: 1.7,
    gfr: 68,
    bun: 30,
    serum_calcium: 8.6,
    oxalate_levels: 2.8,
    urine_ph: 5.6,
    blood_pressure: 145,
    ana: 1,
    c3_c4: 1,
    hematuria: 1,
    smoking: 1,
    alcohol: 1,
    painkiller_usage: 1,
    family_history: 1,
    physical_activity: 0,
    diet: 2,
    water_intake: 1,
    weight_changes: 1,
    stress_level: 3,
    months: 18,
  },
},

  "Stage 3": {
    patient: {
      patientId: "DEMO-STAGE-3",
      patientName: "Demo Patient - Stage 3",
      age: "55",
      gender: "Male",
    },
    clinical: {
      serum_creatinine: 1.8,
      gfr: 42,
      bun: 32,
      serum_calcium: 8.7,
      oxalate_levels: 3,
      urine_ph: 5.7,
      blood_pressure: 145,
      ana: 1,
      c3_c4: 1,
      hematuria: 1,
      smoking: 1,
      alcohol: 1,
      painkiller_usage: 1,
      family_history: 1,
      physical_activity: 0,
      diet: 2,
      water_intake: 1,
      weight_changes: 1,
      stress_level: 3,
      months: 24,
    },
  },

  "Stage 4": {
    patient: {
      patientId: "DEMO-STAGE-4",
      patientName: "Demo Patient - Stage 4",
      age: "62",
      gender: "Male",
    },
    clinical: {
      serum_creatinine: 2.8,
      gfr: 25,
      bun: 48,
      serum_calcium: 8.1,
      oxalate_levels: 4,
      urine_ph: 5.4,
      blood_pressure: 160,
      ana: 1,
      c3_c4: 0,
      hematuria: 1,
      smoking: 1,
      alcohol: 1,
      painkiller_usage: 1,
      family_history: 1,
      physical_activity: 0,
      diet: 2,
      water_intake: 1,
      weight_changes: 2,
      stress_level: 3,
      months: 36,
    },
  },

  "Stage 5": {
    patient: {
      patientId: "DEMO-STAGE-5",
      patientName: "Demo Patient - Stage 5",
      age: "68",
      gender: "Male",
    },
    clinical: {
      serum_creatinine: 5.2,
      gfr: 10,
      bun: 82,
      serum_calcium: 7.4,
      oxalate_levels: 5,
      urine_ph: 5.1,
      blood_pressure: 175,
      ana: 1,
      c3_c4: 0,
      hematuria: 1,
      smoking: 1,
      alcohol: 1,
      painkiller_usage: 1,
      family_history: 1,
      physical_activity: 0,
      diet: 2,
      water_intake: 0,
      weight_changes: 2,
      stress_level: 3,
      months: 48,
    },
  },
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

  const [selectedDemo, setSelectedDemo] = useState("");

  const [ultrasound, setUltrasound] = useState(null);
  const [preview, setPreview] = useState("");

  const [clinicalResult, setClinicalResult] = useState(null);
  const [finalResult, setFinalResult] = useState(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleDemoChange = (e) => {
    const value = e.target.value;
    setSelectedDemo(value);

    if (!value || !demoCases[value]) {
      return;
    }

    const demo = demoCases[value];

    setPatient(demo.patient);
    setClinicalData(demo.clinical);
    setError("");
  };

  const handlePatientChange = (e) => {
    const { name, value } = e.target;

    setPatient((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleClinicalChange = (e) => {
    const { name, value } = e.target;

    setClinicalData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleImageChange = (e) => {
    const file = e.target.files?.[0];

    if (!file) return;

    const validTypes = ["image/jpeg", "image/png", "image/jpg"];

    if (!validTypes.includes(file.type)) {
      setError("Please upload a JPG, JPEG, or PNG ultrasound image.");
      return;
    }

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

    for (const [key, value] of Object.entries(clinicalData)) {
      if (value === "" || value === null || value === undefined) {
        setError(`Please enter a value for ${key.replaceAll("_", " ")}.`);
        return false;
      }
    }

    return true;
  };

  const validatePageTwo = () => {
    if (!ultrasound) {
      setError("Please upload a kidney ultrasound image.");
      return false;
    }

    return true;
  };

  const getStageSeverity = (stage) => {
    if (!stage) {
      return "Not available";
    }

    const stageNumber = Number(
      String(stage).replace(/\D/g, "")
    );

    switch (stageNumber) {
      case 1:
        return "Mild";
      case 2:
        return "Mild";
      case 3:
        return "Moderate";
      case 4:
        return "Severe";
      case 5:
        return "Kidney Failure";
      default:
        return "Not available";
    }
  };

  const generateResults = async () => {
    if (!validatePageTwo()) return;

    setLoading(true);
    setError("");

    try {
      const clinicalPayload = Object.fromEntries(
        Object.entries(clinicalData).map(([key, value]) => [
          key,
          Number(value),
        ])
      );

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

      const clinicalDataResponse = await clinicalResponse.json();

      setClinicalResult(clinicalDataResponse);

     const formData = new FormData();

formData.append(
  "patient",
  JSON.stringify(clinicalPayload)
);

formData.append(
  "file",
  ultrasound
);

      const fusionResponse = await fetch(
        `${API_URL}/fusion-predict`,
        {
          method: "POST",
          body: formData,
        }
      );

      if (!fusionResponse.ok) {
        throw new Error("Fusion prediction failed.");
      }

      const fusionData = await fusionResponse.json();

      const finalPrediction =
        fusionData.prediction ||
        fusionData.final_prediction ||
        fusionData.clinical_prediction ||
        "Not Available";

      const finalStage =
  fusionData.stage ||
  fusionData.stage_name ||
  (
    fusionData.stage_number
      ? `Stage ${fusionData.stage_number}`
      : null
  );

      const finalSeverity =
        finalPrediction === "CKD Detected"
          ? getStageSeverity(finalStage)
          : "Normal";

      const probability = Number(
        fusionData.final_probability ??
          fusionData.confidence ??
          0
      );

      setFinalResult({
        ...fusionData,
        prediction: finalPrediction,
        stage: finalStage,
        severity: finalSeverity,
        confidence: probability,
      });

      setPage(3);
    } catch (err) {
      console.error(err);

      setError(
        err.message ||
          "Something went wrong while generating the assessment."
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

    setSelectedDemo("");
    setUltrasound(null);
    setPreview("");

    setClinicalResult(null);
    setFinalResult(null);
    setError("");
  };

  const getStatusClass = (prediction) => {
    if (!prediction) return "neutral";

    const value = String(prediction).toLowerCase();

    if (
      value.includes("no ckd") ||
      value.includes("healthy") ||
      value.includes("negative")
    ) {
      return "success";
    }

    if (
      value.includes("detected") ||
      value.includes("positive")
    ) {
      return "danger";
    }

    return "neutral";
  };

  const getSeverityClass = (severity) => {
    if (!severity) return "neutral";

    const value = String(severity).toLowerCase();

    if (
      value.includes("kidney failure") ||
      value.includes("severe")
    ) {
      return "danger";
    }

    if (value.includes("moderate")) {
      return "warning";
    }

    if (
      value.includes("mild") ||
      value.includes("normal")
    ) {
      return "success";
    }

    return "neutral";
  };

  const formatStage = (stage) => {
    if (!stage) return "Not Applicable";

    const value = String(stage);

    if (
      value.toLowerCase().startsWith("stage")
    ) {
      return value;
    }

    return `Stage ${value}`;
  };

  const generatePDFReport = () => {
    if (!finalResult) return;

    const doc = new jsPDF();

    const pageWidth = doc.internal.pageSize.getWidth();
    const pageHeight = doc.internal.pageSize.getHeight();

    const margin = 18;

    const prediction =
      finalResult.prediction || "Not Available";

    const stage =
      prediction === "CKD Detected"
        ? formatStage(finalResult.stage)
        : "Not Applicable";

    const severity =
      finalResult.severity ||
      (prediction === "CKD Detected"
        ? getStageSeverity(finalResult.stage)
        : "Normal");

    const confidence = Number(
      finalResult.final_probability ??
        finalResult.confidence ??
        0
    );

    const clinicalProbability = Number(
      finalResult.clinical_probability ?? 0
    );

    const imageProbability = Number(
      finalResult.image_probability ?? 0
    );

    const assessmentDate = new Date().toLocaleString(
      "en-IN",
      {
        dateStyle: "medium",
        timeStyle: "short",
      }
    );

    const addFooter = () => {
      doc.setDrawColor(220, 225, 230);
      doc.line(
        margin,
        pageHeight - 16,
        pageWidth - margin,
        pageHeight - 16
      );

      doc.setFontSize(8);
      doc.setTextColor(120, 130, 140);

      doc.text(
        "NephroPredictor • AI-Assisted CKD Decision-Support Prototype",
        margin,
        pageHeight - 9
      );

      doc.text(
        `Page ${doc.internal.getNumberOfPages()}`,
        pageWidth - margin,
        pageHeight - 9,
        { align: "right" }
      );
    };

    const sectionTitle = (title, y) => {
      doc.setFontSize(13);
      doc.setFont("helvetica", "bold");
      doc.setTextColor(20, 65, 100);

      doc.text(title, margin, y);

      doc.setDrawColor(210, 220, 228);
      doc.line(
        margin,
        y + 3,
        pageWidth - margin,
        y + 3
      );

      return y + 13;
    };

    const addRow = (label, value, y) => {
      doc.setFontSize(10);
      doc.setFont("helvetica", "bold");
      doc.setTextColor(70, 80, 90);

      doc.text(label, margin, y);

      doc.setFont("helvetica", "normal");
      doc.setTextColor(30, 35, 40);

      doc.text(
        String(value ?? "Not available"),
        margin + 58,
        y
      );

      return y + 8;
    };

    // Header
    doc.setFillColor(15, 75, 110);
    doc.rect(0, 0, pageWidth, 34, "F");

    doc.setTextColor(255, 255, 255);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(21);

    doc.text(
      "NEPHROPREDICTOR",
      margin,
      14
    );

    doc.setFont("helvetica", "normal");
    doc.setFontSize(9);

    doc.text(
      "CKD Early Detection & Clinical Decision Support",
      margin,
      23
    );

    doc.setFontSize(8);
    doc.text(
      "AI-ASSISTED ASSESSMENT REPORT",
      pageWidth - margin,
      18,
      { align: "right" }
    );

    let y = 48;

    // Patient information
    y = sectionTitle(
      "1. Patient Information",
      y
    );

    y = addRow(
      "Patient ID",
      patient.patientId,
      y
    );

    y = addRow(
      "Patient Name",
      patient.patientName,
      y
    );

    y = addRow(
      "Age",
      `${patient.age} years`,
      y
    );

    y = addRow(
      "Gender",
      patient.gender,
      y
    );

    y = addRow(
      "Assessment Date",
      assessmentDate,
      y
    );

    y += 5;

    // Final assessment
    y = sectionTitle(
      "2. Final AI-Assisted Assessment",
      y
    );

    doc.setFillColor(
      prediction === "CKD Detected"
        ? 255
        : 239,
      prediction === "CKD Detected"
        ? 244
        : 250,
      prediction === "CKD Detected"
        ? 244
        : 245
    );

    doc.roundedRect(
      margin,
      y,
      pageWidth - margin * 2,
      42,
      4,
      4,
      "F"
    );

    doc.setFont("helvetica", "bold");
    doc.setFontSize(17);

    doc.setTextColor(
      prediction === "CKD Detected"
        ? 180
        : 25,
      prediction === "CKD Detected"
        ? 55
        : 105,
      prediction === "CKD Detected"
        ? 55
        : 75
    );

    doc.text(
      prediction,
      margin + 8,
      y + 13
    );

    doc.setFontSize(10);
    doc.setFont("helvetica", "normal");
    doc.setTextColor(70, 80, 90);

    doc.text(
      `Stage: ${stage}`,
      margin + 8,
      y + 24
    );

    doc.text(
      `Severity: ${severity}`,
      margin + 8,
      y + 32
    );

    doc.text(
      `Model Confidence: ${confidence.toFixed(1)}%`,
      pageWidth - margin - 8,
      y + 24,
      { align: "right" }
    );

    y += 54;

    // Model assessment
    y = sectionTitle(
      "3. Model Assessment",
      y
    );

    y = addRow(
      "Clinical Model",
      `${clinicalProbability.toFixed(1)}% CKD probability`,
      y
    );

    y = addRow(
      "Ultrasound Model",
      `${imageProbability.toFixed(1)}% pathological probability`,
      y
    );

    y = addRow(
      "Fusion Method",
      "Clinical + Ultrasound late fusion",
      y
    );

    y = addRow(
      "Clinical Weight",
      "60%",
      y
    );

    y = addRow(
      "Ultrasound Weight",
      "40%",
      y
    );

    y += 5;

    // Clinical parameters
    y = sectionTitle(
      "4. Clinical Parameters",
      y
    );

    const clinicalLabels = {
      serum_creatinine: "Serum Creatinine",
      gfr: "GFR",
      bun: "BUN",
      serum_calcium: "Serum Calcium",
      oxalate_levels: "Oxalate Levels",
      urine_ph: "Urine pH",
      blood_pressure: "Blood Pressure",
      ana: "ANA",
      c3_c4: "C3/C4",
      hematuria: "Hematuria",
      smoking: "Smoking",
      alcohol: "Alcohol",
      painkiller_usage: "Painkiller Usage",
      family_history: "Family History",
      physical_activity: "Physical Activity",
      diet: "Diet",
      water_intake: "Water Intake",
      weight_changes: "Weight Changes",
      stress_level: "Stress Level",
      months: "Duration",
    };

    const entries = Object.entries(
      clinicalData
    );

    let columnY = y;

    entries.forEach(([key, value], index) => {
      if (index === 10) {
        columnY = y;

        doc.setFontSize(9);
        doc.setFont("helvetica", "normal");
        doc.setTextColor(35, 40, 45);
      }

      const x =
        index < 10
          ? margin
          : pageWidth / 2 + 4;

      const rowIndex =
        index < 10
          ? index
          : index - 10;

      const currentY =
        y + rowIndex * 7;

      doc.setFont("helvetica", "bold");
      doc.setFontSize(8);
      doc.setTextColor(80, 90, 100);

      doc.text(
        clinicalLabels[key] || key,
        x,
        currentY
      );

      doc.setFont("helvetica", "normal");
      doc.setTextColor(30, 35, 40);

      doc.text(
        String(value),
        x + 48,
        currentY
      );
    });

    y += 78;

    // Stage overview
    if (y > pageHeight - 80) {
      doc.addPage();
      addFooter();
      y = 25;
    }

    y = sectionTitle(
      "5. CKD Stage Overview",
      y
    );

    const stages = [
      ["Stage 1", "Mild"],
      ["Stage 2", "Mild"],
      ["Stage 3", "Moderate"],
      ["Stage 4", "Severe"],
      ["Stage 5", "Kidney Failure"],
    ];

    stages.forEach(([stageName, description], index) => {
      const stageNumber = index + 1;

      const currentStageNumber = Number(
        String(finalResult.stage || "").replace(
          /\D/g,
          ""
        )
      );

      const active =
        prediction === "CKD Detected" &&
        currentStageNumber === stageNumber;

      if (active) {
        doc.setFillColor(226, 242, 249);

        doc.roundedRect(
          margin,
          y - 5,
          pageWidth - margin * 2,
          10,
          2,
          2,
          "F"
        );
      }

      doc.setFont(
        "helvetica",
        active ? "bold" : "normal"
      );

      doc.setFontSize(9);

      doc.setTextColor(
        active ? 15 : 70,
        active ? 90 : 80,
        active ? 120 : 90
      );

      doc.text(
        stageName,
        margin + 4,
        y + 2
      );

      doc.text(
        description,
        margin + 55,
        y + 2
      );

      if (active) {
        doc.text(
          "CURRENT",
          pageWidth - margin - 4,
          y + 2,
          { align: "right" }
        );
      }

      y += 11;
    });

    y += 5;

    // AI summary
    if (y > pageHeight - 80) {
      doc.addPage();
      addFooter();
      y = 25;
    }

    y = sectionTitle(
      "6. AI Analysis Summary",
      y
    );

    doc.setFont("helvetica", "normal");
    doc.setFontSize(10);
    doc.setTextColor(50, 60, 70);

    let summary;

    if (prediction === "CKD Detected") {
      summary =
        `The integrated assessment identified a CKD prediction of "${prediction}" ` +
        `with ${stage} classification and ${severity.toLowerCase()} severity. ` +
        `The final confidence value reported by the fusion model is ` +
        `${confidence.toFixed(1)}%. The assessment combines clinical-data ` +
        `prediction with kidney ultrasound image analysis.`;
    } else {
      summary =
        `The integrated assessment returned "${prediction}". ` +
        `The final confidence value reported by the fusion model is ` +
        `${confidence.toFixed(1)}%. The assessment combines clinical-data ` +
        `prediction with kidney ultrasound image analysis.`;
    }

    const summaryLines = doc.splitTextToSize(
      summary,
      pageWidth - margin * 2
    );

    doc.text(
      summaryLines,
      margin,
      y,
      {
        maxWidth: pageWidth - margin * 2,
        lineHeightFactor: 1.5,
      }
    );

    y +=
      summaryLines.length * 6 + 12;

    // Methodology
    y = sectionTitle(
      "7. System Methodology",
      y
    );

    const methodology =
      "Clinical data is processed using a Random Forest-based CKD model. " +
      "Kidney ultrasound images are processed through U-Net-based kidney " +
      "segmentation followed by image feature extraction and image classification. " +
      "The clinical and ultrasound predictions are combined using late fusion.";

    const methodologyLines =
      doc.splitTextToSize(
        methodology,
        pageWidth - margin * 2
      );

    doc.setFont("helvetica", "normal");
    doc.setFontSize(9);
    doc.setTextColor(65, 75, 85);

    doc.text(
      methodologyLines,
      margin,
      y,
      {
        lineHeightFactor: 1.5,
      }
    );

    y += methodologyLines.length * 5 + 10;

    // Disclaimer
    if (y > pageHeight - 65) {
      doc.addPage();
      addFooter();
      y = 25;
    }

    doc.setFillColor(248, 250, 252);

    doc.roundedRect(
      margin,
      y,
      pageWidth - margin * 2,
      35,
      4,
      4,
      "F"
    );

    doc.setFont("helvetica", "bold");
    doc.setFontSize(9);
    doc.setTextColor(50, 65, 80);

    doc.text(
      "Important Notice",
      margin + 7,
      y + 10
    );

    doc.setFont("helvetica", "normal");
    doc.setFontSize(8);
    doc.setTextColor(80, 90, 100);

    const disclaimer =
      "This report is generated by an AI-assisted research prototype " +
      "for clinical decision support. It is not a substitute for " +
      "professional medical diagnosis, interpretation, or treatment.";

    const disclaimerLines =
      doc.splitTextToSize(
        disclaimer,
        pageWidth - margin * 2 - 14
      );

    doc.text(
      disclaimerLines,
      margin + 7,
      y + 18,
      {
        lineHeightFactor: 1.4,
      }
    );

    // Footer on all pages
    const totalPages =
      doc.internal.getNumberOfPages();

    for (let i = 1; i <= totalPages; i++) {
      doc.setPage(i);
      addFooter();
    }

    const safePatientId =
      String(patient.patientId || "patient")
        .replace(/[^a-z0-9_-]/gi, "_");

    doc.save(
      `NephroPredictor_Report_${safePatientId}.pdf`
    );
  };

  const probability = Number(
    finalResult?.final_probability ??
      finalResult?.confidence ??
      0
  );

  return (
    <div className="app-shell">

      {/* HEADER */}
      <header className="app-header">
        <div className="header-content">

          <div className="brand">
            <div className="brand-logo">
              NP
            </div>

            <div>
              <h1>NephroPredictor</h1>

              <p>
                CKD Early Detection & Clinical
                Decision Support
              </p>
            </div>
          </div>

          <div className="system-status">
            <span className="status-dot" />
            SYSTEM READY
          </div>

        </div>
      </header>

      <main className="main-container">

        {/* WORKFLOW */}
        <section className="workflow-bar">

          <div
            className={`workflow-step ${
              page >= 1 ? "active" : ""
            }`}
          >
            <span>01</span>
            <div>
              <strong>Patient</strong>
              <small>Clinical profile</small>
            </div>
          </div>

          <div className="workflow-line" />

          <div
            className={`workflow-step ${
              page >= 2 ? "active" : ""
            }`}
          >
            <span>02</span>
            <div>
              <strong>Ultrasound</strong>
              <small>Image analysis</small>
            </div>
          </div>

          <div className="workflow-line" />

          <div
            className={`workflow-step ${
              page >= 3 ? "active" : ""
            }`}
          >
            <span>03</span>
            <div>
              <strong>Assessment</strong>
              <small>AI fusion results</small>
            </div>
          </div>

        </section>

        {error && (
          <div className="error-banner">
            <strong>Attention:</strong> {error}
          </div>
        )}

        {/* PAGE 1 */}
        {page === 1 && (
          <>
            <section className="hero-section">

              <div>
                <div className="eyebrow">
                  AI-ASSISTED CLINICAL SCREENING
                </div>

                <h2>
                  Integrated CKD
                  <br />
                  Early Detection
                </h2>

                <p>
                  Combine clinical parameters and
                  kidney ultrasound analysis into a
                  unified AI-assisted assessment.
                </p>
              </div>

              <div className="hero-icon">
                🩺
              </div>

            </section>

            <section className="demo-section">

              <div className="section-heading">
                <div>
                  <span className="section-kicker">
                    TEST ENVIRONMENT
                  </span>

                  <h3>Demo Assessment Cases</h3>

                  <p>
                    Select a predefined case to
                    demonstrate the complete workflow.
                  </p>
                </div>

                <select
                  value={selectedDemo}
                  onChange={handleDemoChange}
                  className="demo-select"
                >
                  <option value="">
                    Select Demo Case
                  </option>

                  <option value="No CKD">
                    No CKD Demo Case
                  </option>

                  <option value="Stage 1">
                    Stage 1 Demo Case
                  </option>

                  <option value="Stage 2">
                    Stage 2 Demo Case
                  </option>

                  <option value="Stage 3">
                    Stage 3 Demo Case
                  </option>

                  <option value="Stage 4">
                    Stage 4 Demo Case
                  </option>

                  <option value="Stage 5">
                    Stage 5 Demo Case
                  </option>
                </select>
              </div>

            </section>

            <section className="form-card">

              <div className="card-heading">
                <div>
                  <span className="section-kicker">
                    PATIENT RECORD
                  </span>

                  <h3>Patient Information</h3>
                </div>

                <span className="record-badge">
                  PRIVATE RECORD
                </span>
              </div>

              <div className="form-grid">

                <div className="field">
                  <label>Patient ID</label>

                  <input
                    name="patientId"
                    value={patient.patientId}
                    onChange={handlePatientChange}
                    placeholder="e.g. NP-2026-001"
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
                    value={patient.age}
                    onChange={handlePatientChange}
                    placeholder="Years"
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

                    <option value="Female">
                      Female
                    </option>

                    <option value="Male">
                      Male
                    </option>

                    <option value="Other">
                      Other
                    </option>
                  </select>
                </div>

              </div>

            </section>

            <section className="form-card">

              <div className="card-heading">
                <div>
                  <span className="section-kicker">
                    CLINICAL INPUTS
                  </span>

                  <h3>
                    Clinical Assessment
                  </h3>
                </div>
              </div>

              <div className="clinical-grid">

                {Object.entries(clinicalData).map(
                  ([key, value]) => (
                    <ClinicalField
                      key={key}
                      name={key}
                      value={value}
                      onChange={handleClinicalChange}
                    />
                  )
                )}

              </div>

            </section>

            <div className="bottom-action">

              <div className="decision-note">
                <span>●</span>

                <div>
                  <strong>
                    AI-assisted decision support
                  </strong>

                  <small>
                    Clinical inputs will be combined
                    with ultrasound analysis.
                  </small>
                </div>
              </div>

              <button
                className="primary-button"
                onClick={() => {
                  if (validatePageOne()) {
                    setPage(2);
                  }
                }}
              >
                Continue to Ultrasound
                <span>→</span>
              </button>

            </div>
          </>
        )}

        {/* PAGE 2 */}
        {page === 2 && (
          <>
            <section className="page-title">

              <div>
                <span className="section-kicker">
                  STEP 02 OF 03
                </span>

                <h2>
                  Kidney Ultrasound Analysis
                </h2>

                <p>
                  Upload the kidney ultrasound image
                  for AI-based image analysis and
                  clinical fusion.
                </p>
              </div>

              <div className="patient-mini-card">
                <strong>
                  {patient.patientName}
                </strong>

                <span>
                  {patient.patientId}
                </span>
              </div>

            </section>

            <section className="upload-layout">

              <div className="upload-card">

                {!preview ? (
                  <label className="upload-zone">

                    <input
                      type="file"
                      accept="image/png,image/jpeg,image/jpg"
                      onChange={handleImageChange}
                    />

                    <div className="upload-icon">
                      ↑
                    </div>

                    <h3>
                      Upload Ultrasound Image
                    </h3>

                    <p>
                      Drag and drop or click to browse
                    </p>

                    <span>
                      JPG, JPEG or PNG
                    </span>

                  </label>
                ) : (
                  <div className="image-preview-container">

                    <img
                      src={preview}
                      alt="Uploaded kidney ultrasound"
                      className="ultrasound-preview"
                    />

                    <div className="image-info">

                      <div>
                        <strong>
                          {ultrasound?.name}
                        </strong>

                        <span>
                          {(
                            ultrasound?.size /
                            1024 /
                            1024
                          ).toFixed(2)}{" "}
                          MB
                        </span>
                      </div>

                      <label className="change-image">
                        Change Image

                        <input
                          type="file"
                          accept="image/png,image/jpeg,image/jpg"
                          onChange={handleImageChange}
                        />
                      </label>

                    </div>

                  </div>
                )}

              </div>

              <div className="analysis-pipeline">

                <div className="pipeline-heading">
                  <span className="section-kicker">
                    AI PIPELINE
                  </span>

                  <h3>
                    Image Processing Workflow
                  </h3>
                </div>

                <PipelineStep
                  number="01"
                  title="Ultrasound Upload"
                  text="Input kidney ultrasound image"
                />

                <PipelineArrow />

                <PipelineStep
                  number="02"
                  title="U-Net Segmentation"
                  text="Identify kidney region of interest"
                />

                <PipelineArrow />

                <PipelineStep
                  number="03"
                  title="Feature Extraction"
                  text="Extract image-based kidney features"
                />

                <PipelineArrow />

                <PipelineStep
                  number="04"
                  title="Image Classification"
                  text="Estimate pathological probability"
                />

                <PipelineArrow />

                <PipelineStep
                  number="05"
                  title="Clinical + Image Fusion"
                  text="Combine both assessment streams"
                />

              </div>

            </section>

            <section className="info-strip">

              <div>
                <strong>
                  Clinical model
                </strong>

                <span>
                  Random Forest
                </span>
              </div>

              <div>
                <strong>
                  Image segmentation
                </strong>

                <span>
                  U-Net
                </span>
              </div>

              <div>
                <strong>
                  Image classification
                </strong>

                <span>
                  Random Forest / XGBoost
                </span>
              </div>

              <div>
                <strong>
                  Final assessment
                </strong>

                <span>
                  Late Fusion
                </span>
              </div>

            </section>

            <div className="bottom-action">

              <button
                className="secondary-button"
                onClick={() => setPage(1)}
              >
                ← Back
              </button>

              <button
                className="primary-button"
                onClick={generateResults}
                disabled={loading}
              >
                {loading
                  ? "Analyzing..."
                  : "Generate Assessment"}
                {!loading && <span>→</span>}
              </button>

            </div>
          </>
        )}

        {/* PAGE 3 */}
        {page === 3 && finalResult && (
          <>
            <section className="page-title results-title">

              <div>
                <span className="section-kicker">
                  FINAL ASSESSMENT
                </span>

                <h2>
                  Clinical Decision Dashboard
                </h2>

                <p>
                  Integrated clinical and ultrasound
                  assessment for {patient.patientName}.
                </p>
              </div>

              <div className="results-actions">

                <button
                  className="pdf-button"
                  onClick={generatePDFReport}
                >
                  <span>▣</span>
                  Generate Clinical Report
                </button>

                <button
                  className="secondary-button"
                  onClick={startNewAssessment}
                >
                  New Assessment
                </button>

              </div>

            </section>

            <section className="patient-summary-card">

              <div>
                <span>Patient ID</span>
                <strong>
                  {patient.patientId}
                </strong>
              </div>

              <div>
                <span>Patient Name</span>
                <strong>
                  {patient.patientName}
                </strong>
              </div>

              <div>
                <span>Age</span>
                <strong>
                  {patient.age} years
                </strong>
              </div>

              <div>
                <span>Gender</span>
                <strong>
                  {patient.gender}
                </strong>
              </div>

            </section>

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
                value={formatStage(
                  finalResult.stage
                )}
                className="blue"
              />

              <ResultCard
                title="Severity"
                value={
                  finalResult.severity ||
                  "Not Available"
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

            <section className="result-analysis-grid">

              <div className="result-card-large">

                <div className="card-heading">

                  <div>
                    <span className="section-kicker">
                      INTEGRATED RESULT
                    </span>

                    <h3>
                      AI Fusion Assessment
                    </h3>
                  </div>

                  <span className="fusion-badge">
                    FUSION COMPLETE
                  </span>

                </div>

                <div className="fusion-visual">

                  <div
  className="confidence-ring"
  style={{
    "--confidence": Math.min(probability, 100),
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

                  <div className="fusion-details">

                    <div className="model-row">

                      <div>
                        <span>
                          Clinical Model
                        </span>

                        <strong>
                          {Number(
                            finalResult.clinical_probability ??
                              0
                          ).toFixed(1)}
                          %
                        </strong>
                      </div>

                      <div className="model-bar">
                        <span
                          style={{
                            width: `${Math.min(
                              Number(
                                finalResult.clinical_probability ??
                                  0
                              ),
                              100
                            )}%`,
                          }}
                        />
                      </div>

                    </div>

                    <div className="model-row">

                      <div>
                        <span>
                          Ultrasound Model
                        </span>

                        <strong>
                          {Number(
                            finalResult.image_probability ??
                              0
                          ).toFixed(1)}
                          %
                        </strong>
                      </div>

                      <div className="model-bar">
                        <span
                          style={{
                            width: `${Math.min(
                              Number(
                                finalResult.image_probability ??
                                  0
                              ),
                              100
                            )}%`,
                          }}
                        />
                      </div>

                    </div>

                    <div className="fusion-equation">
                      <span>60% Clinical</span>
                      <strong>+</strong>
                      <span>40% Ultrasound</span>
                      <strong>=</strong>
                      <span>Final Fusion</span>
                    </div>

                  </div>

                </div>

              </div>

              <div className="result-card-large stage-card">

                <div className="card-heading">

                  <div>
                    <span className="section-kicker">
                      DISEASE PROGRESSION
                    </span>

                    <h3>
                      CKD Stage Overview
                    </h3>
                  </div>

                </div>

                <div className="stage-list">

                  {[1, 2, 3, 4, 5].map(
                    (stage) => {

                      const currentStage =
                        Number(
                          String(
                            finalResult.stage || ""
                          ).replace(/\D/g, "")
                        );

                      const active =
                        currentStage === stage;

                      return (
                        <div
                          className={`stage-item ${
                            active
                              ? "active"
                              : ""
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
                    }
                  )}

                </div>

              </div>

            </section>

            <section className="final-assessment-card">

              <div className="assessment-icon">
                ✓
              </div>

              <div>

                <span className="section-kicker">
                  AI ANALYSIS SUMMARY
                </span>

                <h3>
                  {finalResult.prediction}
                </h3>

                <p>
                  {finalResult.prediction ===
                  "CKD Detected"
                    ? `The integrated assessment indicates ${formatStage(
                        finalResult.stage
                      )} classification with ${finalResult.severity?.toLowerCase()}. The final model confidence is ${probability.toFixed(
                        1
                      )}%.`
                    : `The integrated assessment returned ${finalResult.prediction} with a final model confidence of ${probability.toFixed(
                        1
                      )}%.`}
                </p>

                <small>
                  This is an AI-assisted decision
                  support result and should be
                  interpreted alongside appropriate
                  clinical evaluation.
                </small>

              </div>

            </section>

            <div className="report-bottom-actions">

              <button
                className="pdf-button large"
                onClick={generatePDFReport}
              >
                <span>▣</span>
                Generate Clinical Report PDF
              </button>

              <button
                className="secondary-button"
                onClick={startNewAssessment}
              >
                Start New Assessment
              </button>

            </div>

          </>
        )}

      </main>

      <footer className="app-footer">

        <span>
          NephroPredictor
        </span>

        <span>
          AI-Assisted CKD Decision-Support
          Prototype
        </span>

      </footer>

    </div>
  );
}

/* ----------------------------- */
/* Clinical Field */
/* ----------------------------- */

function ClinicalField({
  name,
  value,
  onChange,
}) {
  const labels = {
    serum_creatinine: "Serum Creatinine",
    gfr: "GFR",
    bun: "BUN",
    serum_calcium: "Serum Calcium",
    oxalate_levels: "Oxalate Levels",
    urine_ph: "Urine pH",
    blood_pressure: "Blood Pressure",
    ana: "ANA",
    c3_c4: "C3 / C4",
    hematuria: "Hematuria",
    smoking: "Smoking",
    alcohol: "Alcohol",
    painkiller_usage: "Painkiller Usage",
    family_history: "Family History",
    physical_activity: "Physical Activity",
    diet: "Diet",
    water_intake: "Water Intake",
    weight_changes: "Weight Changes",
    stress_level: "Stress Level",
    months: "Duration (Months)",
  };

  return (
    <div className="field">

      <label>
        {labels[name] || name}
      </label>

      <input
        type="number"
        step="any"
        name={name}
        value={value}
        onChange={onChange}
        placeholder="Enter value"
      />

    </div>
  );
}

/* ----------------------------- */
/* Result Card */
/* ----------------------------- */

function ResultCard({
  title,
  value,
  className,
}) {
  return (
    <div
      className={`result-card ${className}`}
    >
      <span>{title}</span>

      <strong>{value}</strong>
    </div>
  );
}

/* ----------------------------- */
/* Pipeline Step */
/* ----------------------------- */

function PipelineStep({
  number,
  title,
  text,
}) {
  return (
    <div className="pipeline-step">

      <div className="pipeline-number">
        {number}
      </div>

      <div>
        <strong>{title}</strong>

        <small>{text}</small>
      </div>

    </div>
  );
}

/* ----------------------------- */
/* Pipeline Arrow */
/* ----------------------------- */

function PipelineArrow() {
  return (
    <div className="pipeline-arrow">
      ↓
    </div>
  );
}

export default App;

