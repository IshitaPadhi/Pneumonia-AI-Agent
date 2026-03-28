import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from agent.agent import agent_decision
from datetime import date

# Optional DB
# ✅ DB IMPORT (ONLY HERE — TOP OF FILE)
try:
    from database.mysql_connect import insert_patient
    DB_ENABLED = True
except Exception as e:
    DB_ENABLED = False
    st.error(f"DB Import Error: {e}")   # 👈 SHOW IN UI


# Extra features
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# ─────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="XPneumoNet | AI Radiology Assistant",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Sora:wght@300;400;600;700&display=swap');

/* ── Base reset ── */
html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}

/* ── App background ── */
.stApp {
    background: #09111f;
    color: #c8d6e5;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem; max-width: 1200px; }

/* ─── HEADER BANNER ─── */
.xpn-header {
    display: flex;
    align-items: center;
    gap: 1.2rem;
    padding: 1.6rem 2rem;
    background: linear-gradient(135deg, #0d1f35 0%, #0a2240 60%, #081830 100%);
    border: 1px solid #1a3a5c;
    border-radius: 14px;
    margin-bottom: 2rem;
    box-shadow: 0 4px 40px rgba(0,150,255,0.07);
}
.xpn-logo {
    font-size: 2.6rem;
    line-height: 1;
}
.xpn-title-block h1 {
    margin: 0;
    font-size: 1.65rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    color: #e8f4ff;
}
.xpn-title-block p {
    margin: 0.2rem 0 0;
    font-size: 0.78rem;
    font-family: 'DM Mono', monospace;
    color: #4a8fc0;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
.xpn-badge {
    margin-left: auto;
    background: #0a3d62;
    border: 1px solid #1a6ea8;
    color: #5bc0f8;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    letter-spacing: 0.08em;
}

/* ─── SECTION CARDS ─── */
.xpn-card {
    background: #0d1e31;
    border: 1px solid #1a3a5c;
    border-radius: 12px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.2rem;
}
.xpn-card-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #3a8fc8;
    margin-bottom: 1rem;
    border-bottom: 1px solid #1a3a5c;
    padding-bottom: 0.6rem;
}

/* ─── INPUTS ─── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: #081526 !important;
    border: 1px solid #1a3a5c !important;
    border-radius: 8px !important;
    color: #c8d6e5 !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.88rem !important;
    padding: 0.55rem 0.9rem !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: #2a7fc8 !important;
    box-shadow: 0 0 0 3px rgba(42,127,200,0.15) !important;
}
label[data-testid="stWidgetLabel"] p {
    font-size: 0.78rem !important;
    font-family: 'DM Mono', monospace !important;
    color: #4a8fc0 !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ─── FILE UPLOADER ─── */
[data-testid="stFileUploader"] {
    background: #081526;
    border: 2px dashed #1a4a7c;
    border-radius: 12px;
    padding: 1.2rem;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: #2a7fc8;
}
[data-testid="stFileUploader"] label {
    color: #5bc0f8 !important;
}

/* ─── PREDICT BUTTON ─── */
.stButton > button {
    background: linear-gradient(135deg, #0a4a8a, #0d6fc4) !important;
    color: #e8f4ff !important;
    border: 1px solid #1a7fe8 !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.12em !important;
    padding: 0.65rem 2rem !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0d5eaa, #1a8af0) !important;
    box-shadow: 0 0 18px rgba(26,138,240,0.3) !important;
    transform: translateY(-1px) !important;
}

/* ─── RESULT PILL ─── */
.diag-pill {
    display: inline-block;
    padding: 0.35rem 1rem;
    border-radius: 20px;
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    margin-top: 0.3rem;
}
.pill-bacteria  { background: #2d0e0e; color: #f87171; border: 1px solid #7f1d1d; }
.pill-virus     { background: #2d200a; color: #fbbf24; border: 1px solid #92400e; }
.pill-normal    { background: #0a2d1a; color: #4ade80; border: 1px solid #166534; }

/* ─── CONFIDENCE METER ─── */
.conf-bar-wrap {
    margin-top: 0.8rem;
}
.conf-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #4a8fc0;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.conf-track {
    background: #0a1e30;
    border-radius: 4px;
    height: 8px;
    overflow: hidden;
    border: 1px solid #1a3a5c;
}
.conf-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.8s ease;
}
.conf-high   { background: linear-gradient(90deg, #166534, #4ade80); }
.conf-medium { background: linear-gradient(90deg, #92400e, #fbbf24); }
.conf-low    { background: linear-gradient(90deg, #7f1d1d, #f87171); }
.conf-value {
    font-family: 'DM Mono', monospace;
    font-size: 1.2rem;
    font-weight: 500;
    margin-top: 0.4rem;
}

/* ─── SEVERITY BADGE ─── */
.severity-block {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    padding: 0.9rem 1.2rem;
    border-radius: 10px;
    margin-bottom: 0.8rem;
}
.sev-critical { background: #1a0808; border: 1px solid #7f1d1d; }
.sev-high     { background: #1a1008; border: 1px solid #7c3a00; }
.sev-moderate { background: #1a1808; border: 1px solid #8a7a00; }
.sev-low      { background: #081a10; border: 1px solid #166534; }
.sev-icon { font-size: 1.4rem; }
.sev-text h4  { margin: 0; font-size: 0.9rem; font-weight: 600; color: #e8f4ff; }
.sev-text p   { margin: 0.2rem 0 0; font-size: 0.78rem; color: #7a9ab8; font-family: 'DM Mono', monospace; }

/* ─── ADVICE BOX ─── */
.advice-box {
    background: #081830;
    border-left: 3px solid #2a7fc8;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem;
    font-size: 0.86rem;
    line-height: 1.65;
    color: #a8c4de;
}

/* ─── PROBABILITY BARS ─── */
.prob-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 0.65rem;
}
.prob-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #4a8fc0;
    width: 72px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    flex-shrink: 0;
}
.prob-track {
    flex: 1;
    background: #0a1e30;
    border-radius: 4px;
    height: 10px;
    overflow: hidden;
    border: 1px solid #1a3a5c;
}
.prob-fill-bacteria { background: linear-gradient(90deg, #7f1d1d, #f87171); height: 100%; border-radius: 4px; }
.prob-fill-normal   { background: linear-gradient(90deg, #166534, #4ade80); height: 100%; border-radius: 4px; }
.prob-fill-virus    { background: linear-gradient(90deg, #92400e, #fbbf24); height: 100%; border-radius: 4px; }
.prob-pct {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: #7a9ab8;
    width: 40px;
    text-align: right;
    flex-shrink: 0;
}

/* ─── DIVIDER ─── */
.xpn-divider {
    border: none;
    border-top: 1px solid #1a3a5c;
    margin: 1.5rem 0;
}

/* ─── METRIC CHIPS ─── */
.metric-row {
    display: flex;
    gap: 0.8rem;
    flex-wrap: wrap;
    margin-bottom: 1rem;
}
.metric-chip {
    background: #0a1e30;
    border: 1px solid #1a3a5c;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    flex: 1;
    min-width: 100px;
}
.metric-chip .mc-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    color: #3a8fc8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.metric-chip .mc-value {
    font-size: 1.05rem;
    font-weight: 600;
    color: #e8f4ff;
    margin-top: 0.2rem;
}

/* ─── ALERT OVERRIDES ─── */
.stSuccess, .stWarning, .stError {
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
}

/* ─── DOWNLOAD BUTTON ─── */
.stDownloadButton > button {
    background: #081526 !important;
    color: #5bc0f8 !important;
    border: 1px solid #1a4a7c !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.1em !important;
    width: 100% !important;
    margin-top: 0.5rem !important;
}
.stDownloadButton > button:hover {
    background: #0a2040 !important;
    border-color: #2a7fc8 !important;
}

/* ─── IMAGE display ─── */
[data-testid="stImage"] img {
    border-radius: 10px;
    border: 1px solid #1a3a5c;
}

/* ─── Streamlit pyplot ─── */
[data-testid="stPlotlyChart"], .stPyplot {
    background: transparent !important;
}

/* ─── FOOTER ─── */
.xpn-footer {
    text-align: center;
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #2a4a6a;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid #1a3a5c;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def severity_css(severity: str) -> str:
    s = severity.lower()
    if "critical" in s: return "sev-critical", "🔴"
    if "high"     in s: return "sev-high",     "🟠"
    if "moderate" in s: return "sev-moderate",  "🟡"
    return "sev-low", "🟢"

def confidence_css(conf: float) -> str:
    if conf > 0.85: return "conf-high"
    if conf > 0.65: return "conf-medium"
    return "conf-low"

def prediction_pill_css(pred: str) -> str:
    p = pred.lower()
    if p == "bacteria": return "pill-bacteria"
    if p == "virus":    return "pill-virus"
    return "pill-normal"


# ─────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────
model = tf.keras.models.load_model("model/best_pneumonia_model.keras")
classes = ["BACTERIA", "NORMAL", "VIRUS"]

last_conv_layer_name = None
for layer in reversed(model.layers):
    if "conv" in layer.name:
        last_conv_layer_name = layer.name
        break

grad_model = tf.keras.models.Model(
    inputs=model.input,
    outputs=[model.get_layer(last_conv_layer_name).output, model.output]
)


# ─────────────────────────────────────────────
#  GRAD-CAM
# ─────────────────────────────────────────────
def gradcam(img_array, class_index):
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8
    return heatmap


# ─────────────────────────────────────────────
#  PDF GENERATOR  (unchanged logic)
# ─────────────────────────────────────────────
def generate_pdf(patient_id, name, prediction, confidence, severity, report):
    os.makedirs("temp", exist_ok=True)
    pdf_path = "temp/report.pdf"
    c = canvas.Canvas(pdf_path, pagesize=letter)
    c.drawString(100, 750, "Pneumonia AI Diagnosis Report")
    c.drawString(100, 720, f"Patient ID: {patient_id}")
    c.drawString(100, 700, f"Name: {name}")
    c.drawString(100, 650, f"Prediction: {prediction}")
    c.drawString(100, 630, f"Confidence: {confidence:.2f}")
    c.drawString(100, 610, f"Severity: {severity}")
    c.drawString(100, 580, f"Advice: {report}")
    c.save()
    return pdf_path


# ═══════════════════════════════════════════════════════════
#  UI — HEADER
# ═══════════════════════════════════════════════════════════
st.markdown("""
<div class="xpn-header">
    <div class="xpn-logo">🫁</div>
    <div class="xpn-title-block">
        <h1>XPneumoNet</h1>
        <p>Explainable AI Radiology Assistant · DenseNet-121 + Grad-CAM</p>
    </div>
    <div class="xpn-badge">v1.0 · Three-Class Detection</div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  UI — PATIENT INTAKE  +  UPLOAD  (side by side)
# ═══════════════════════════════════════════════════════════
col_form, col_upload = st.columns([1, 1], gap="large")

with col_form:
    st.markdown('<div class="xpn-card">', unsafe_allow_html=True)
    st.markdown('<div class="xpn-card-title">Patient Information</div>', unsafe_allow_html=True)

    patient_id = st.text_input("Patient ID", placeholder="e.g. PT-20240001")
    name       = st.text_input("Full Name",  placeholder="e.g. Rajan Kumar")
    age        = st.number_input("Age (years)", min_value=0, max_value=120, value=30)

    st.markdown("</div>", unsafe_allow_html=True)

with col_upload:
    st.markdown('<div class="xpn-card">', unsafe_allow_html=True)
    st.markdown('<div class="xpn-card-title">Chest X-Ray Upload</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop JPEG / PNG X-ray here",
        type=["jpg", "png", "jpeg"],
        label_visibility="visible"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("⟡  RUN ANALYSIS", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  PREDICTION BLOCK
# ═══════════════════════════════════════════════════════════
if predict_btn:
    if uploaded_file is None:
        st.warning("⚠ Please upload a chest X-ray image before running analysis.")
    else:
        os.makedirs("temp", exist_ok=True)
        file_path = os.path.join("temp", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        img         = cv2.imread(file_path)
        img_resized = cv2.resize(img, (224, 224))
        img_norm    = img_resized / 255.0
        img_array   = np.reshape(img_norm, (1, 224, 224, 3))

        pred         = model.predict(img_array)[0]
        class_index  = np.argmax(pred)
        confidence   = pred[class_index]
        prediction   = classes[class_index]

        # Grad-CAM
        heatmap = gradcam(img_array, class_index)
        heatmap = cv2.resize(heatmap, (224, 224))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * 0.4 + img_resized
        heatmap_path = os.path.join("temp", "heatmap.jpg")
        cv2.imwrite(heatmap_path, superimposed_img)

        severity, report = agent_decision(prediction, confidence)

        # DB
        if DB_ENABLED:
            try:
                insert_patient((patient_id, name, age, date.today(),
                                prediction, float(confidence), severity, report, file_path))
            except Exception as e:
                st.warning(f"Database error: {e}")

        # ── Patient summary chips ──────────────────────────
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-chip">
                <div class="mc-label">Patient ID</div>
                <div class="mc-value">{patient_id or "—"}</div>
            </div>
            <div class="metric-chip">
                <div class="mc-label">Name</div>
                <div class="mc-value">{name or "—"}</div>
            </div>
            <div class="metric-chip">
                <div class="mc-label">Age</div>
                <div class="mc-value">{age} yrs</div>
            </div>
            <div class="metric-chip">
                <div class="mc-label">Scan Date</div>
                <div class="mc-value">{date.today().strftime("%d %b %Y")}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Two-column: images + diagnosis ────────────────
        col_imgs, col_diag = st.columns([1.1, 1], gap="large")

        with col_imgs:
            # Original X-ray
            st.markdown('<div class="xpn-card">', unsafe_allow_html=True)
            st.markdown('<div class="xpn-card-title">Input Radiograph</div>', unsafe_allow_html=True)
            st.image(file_path, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Grad-CAM
            st.markdown('<div class="xpn-card">', unsafe_allow_html=True)
            st.markdown('<div class="xpn-card-title">Grad-CAM Saliency Map · Model Focus Region</div>', unsafe_allow_html=True)
            st.image(heatmap_path, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_diag:
            # ── Primary diagnosis ──
            pill_cls  = prediction_pill_css(prediction)
            conf_cls  = confidence_css(confidence)
            conf_pct  = int(confidence * 100)
            conf_bar  = confidence * 100

            st.markdown('<div class="xpn-card">', unsafe_allow_html=True)
            st.markdown('<div class="xpn-card-title">Primary Diagnosis</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <span class="diag-pill {pill_cls}">{prediction}</span>
            <div class="conf-bar-wrap">
                <div class="conf-label">Model Confidence</div>
                <div class="conf-track">
                    <div class="conf-fill {conf_cls}" style="width:{conf_bar:.1f}%"></div>
                </div>
                <div class="conf-value" style="color:{'#4ade80' if conf_pct>85 else '#fbbf24' if conf_pct>65 else '#f87171'}">{conf_pct}%</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # ── Class probabilities ──
            st.markdown('<div class="xpn-card">', unsafe_allow_html=True)
            st.markdown('<div class="xpn-card-title">Class Probability Distribution</div>', unsafe_allow_html=True)
            cls_styles = {"BACTERIA": "bacteria", "NORMAL": "normal", "VIRUS": "virus"}
            for i, cls in enumerate(classes):
                pct = pred[i] * 100
                st.markdown(f"""
                <div class="prob-row">
                    <span class="prob-label">{cls}</span>
                    <div class="prob-track">
                        <div class="prob-fill-{cls_styles[cls]}" style="width:{pct:.1f}%"></div>
                    </div>
                    <span class="prob-pct">{pct:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # ── Severity ──
            sev_cls, sev_icon = severity_css(severity)
            st.markdown('<div class="xpn-card">', unsafe_allow_html=True)
            st.markdown('<div class="xpn-card-title">Clinical Severity Assessment</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="severity-block {sev_cls}">
                <span class="sev-icon">{sev_icon}</span>
                <div class="sev-text">
                    <h4>{severity}</h4>
                    <p>Agent-assessed severity level</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # ── AI Advice ──
            st.markdown('<div class="xpn-card">', unsafe_allow_html=True)
            st.markdown('<div class="xpn-card-title">Clinical Recommendation</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="advice-box">{report}</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Confusion matrix + PDF download ───────────────
        col_cm, col_pdf = st.columns([1.2, 0.8], gap="large")

        with col_cm:
            st.markdown('<div class="xpn-card">', unsafe_allow_html=True)
            st.markdown('<div class="xpn-card-title">Confusion Matrix · Current Prediction (Demo)</div>', unsafe_allow_html=True)
            cm = confusion_matrix([class_index], [class_index])
            fig, ax = plt.subplots(figsize=(4, 3))
            fig.patch.set_alpha(0)
            ax.set_facecolor("#081526")
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                ax=ax,
                linewidths=0.5, linecolor="#1a3a5c",
                annot_kws={"size": 12, "color": "#e8f4ff"}
            )
            ax.set_xlabel("Predicted", color="#4a8fc0", fontsize=9, labelpad=8)
            ax.set_ylabel("Actual",    color="#4a8fc0", fontsize=9, labelpad=8)
            ax.tick_params(colors="#7a9ab8", labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor("#1a3a5c")
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_pdf:
            st.markdown('<div class="xpn-card">', unsafe_allow_html=True)
            st.markdown('<div class="xpn-card-title">Export Report</div>', unsafe_allow_html=True)
            st.markdown("""
            <p style="font-size:0.8rem; color:#7a9ab8; line-height:1.6; margin-bottom:1rem;">
                Generate a structured PDF summary of the AI diagnosis, severity assessment, and clinical recommendation for this patient encounter.
            </p>
            """, unsafe_allow_html=True)
            pdf_file = generate_pdf(patient_id, name, prediction, confidence, severity, report)
            with open(pdf_file, "rb") as f:
                st.download_button(
                    "⬇  Download Medical Report (PDF)",
                    f,
                    file_name=f"XPneumoNet_{patient_id or 'report'}.pdf"
                )
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Disclaimer ────────────────────────────────────
        st.markdown("""
        <div style="background:#0a1020; border:1px solid #1a2a3c; border-radius:8px;
                    padding:0.9rem 1.2rem; margin-top:0.5rem;">
            <span style="font-family:'DM Mono',monospace; font-size:0.65rem;
                         color:#3a6a9a; letter-spacing:0.06em; text-transform:uppercase;">
                ⚠ Clinical Disclaimer
            </span>
            <p style="font-size:0.76rem; color:#4a6a8a; margin:0.4rem 0 0; line-height:1.55;">
                This output is generated by an AI model for research and educational purposes only.
                It is <strong>not a substitute for clinical diagnosis</strong> by a licensed physician.
                All findings must be reviewed and validated by a qualified radiologist before
                any clinical decision is made.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ─── FOOTER ───────────────────────────────────────────────
st.markdown("""
<div class="xpn-footer">
    XPneumoNet · Explainable Pneumonia Detection · DenseNet-121 + Grad-CAM ·
    Built for Research &amp; Academic Use
</div>
""", unsafe_allow_html=True)