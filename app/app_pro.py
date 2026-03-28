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
# Optional DB
try:
    from database.mysql_connect import insert_patient
    DB_ENABLED = True
except Exception as e:
    DB_ENABLED = False
    print("DB Import Error:", e)   # DEBUG

# Extra features
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# ─────────────────────────────────────────────
# PAGE CONFIG — must be first Streamlit call
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="XPneumoNet · AI Pneumonia Diagnostics",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# GLOBAL CSS — medical dark theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Sora:wght@300;400;600;700&display=swap');

:root {
  --bg:         #070d14;
  --surface:    #0e1825;
  --surface2:   #152030;
  --border:     #1e3048;
  --accent:     #00c6ff;
  --accent2:    #0072ff;
  --green:      #00e5a0;
  --amber:      #ffb830;
  --red:        #ff4b6e;
  --text:       #e4edf6;
  --muted:      #5c7a99;
  --font-ui:    'Sora', sans-serif;
  --font-mono:  'DM Mono', monospace;
}

html, body, [class*="css"] {
  font-family: var(--font-ui);
  background-color: var(--bg);
  color: var(--text);
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 3rem 2rem; max-width: 1200px; }

.xpn-header {
  display: flex;
  align-items: center;
  gap: 1.2rem;
  padding: 1.6rem 0 1rem 0;
  border-bottom: 1px solid var(--border);
  margin-bottom: 2rem;
}
.xpn-logo { font-size: 2.6rem; line-height: 1; }
.xpn-title-block h1 {
  font-size: 1.65rem;
  font-weight: 700;
  letter-spacing: -0.5px;
  background: linear-gradient(90deg, var(--accent), var(--accent2));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin: 0 0 2px 0;
}
.xpn-title-block p {
  font-size: 0.78rem;
  color: var(--muted);
  font-family: var(--font-mono);
  margin: 0;
  letter-spacing: 0.04em;
}
.xpn-badge {
  margin-left: auto;
  font-family: var(--font-mono);
  font-size: 0.7rem;
  color: var(--accent);
  border: 1px solid var(--accent);
  border-radius: 4px;
  padding: 4px 10px;
  letter-spacing: 0.06em;
}

.section-label {
  font-family: var(--font-mono);
  font-size: 0.68rem;
  letter-spacing: 0.14em;
  color: var(--muted);
  text-transform: uppercase;
  margin-bottom: 0.6rem;
}

.xpn-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.4rem 1.6rem;
  margin-bottom: 1.2rem;
}
.xpn-card-title {
  font-size: 0.72rem;
  font-family: var(--font-mono);
  letter-spacing: 0.12em;
  color: var(--accent);
  text-transform: uppercase;
  margin-bottom: 1rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
.xpn-card-title::before {
  content: '';
  display: inline-block;
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--accent);
}

/* ── Input overrides ── */
div[data-testid="stTextInput"] label,
div[data-testid="stNumberInput"] label {
  font-family: var(--font-mono) !important;
  font-size: 0.72rem !important;
  letter-spacing: 0.1em !important;
  color: var(--muted) !important;
  text-transform: uppercase !important;
}
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--text) !important;
  font-family: var(--font-mono) !important;
  font-size: 0.9rem !important;
  padding: 0.55rem 0.9rem !important;
}
div[data-testid="stTextInput"] input:focus,
div[data-testid="stNumberInput"] input:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(0,198,255,0.12) !important;
}

/* ── File uploader ── */
div[data-testid="stFileUploader"] > div {
  background: var(--surface2) !important;
  border: 1.5px dashed var(--border) !important;
  border-radius: 10px !important;
  transition: border-color 0.2s;
}
div[data-testid="stFileUploader"] > div:hover {
  border-color: var(--accent) !important;
}
div[data-testid="stFileUploader"] label {
  font-family: var(--font-mono) !important;
  font-size: 0.72rem !important;
  letter-spacing: 0.1em !important;
  color: var(--muted) !important;
  text-transform: uppercase !important;
}

/* ── Buttons ── */
div[data-testid="stButton"] > button {
  background: linear-gradient(135deg, var(--accent2), var(--accent)) !important;
  color: #fff !important;
  font-family: var(--font-ui) !important;
  font-weight: 600 !important;
  font-size: 0.88rem !important;
  letter-spacing: 0.04em !important;
  border: none !important;
  border-radius: 8px !important;
  padding: 0.65rem 2rem !important;
  width: 100% !important;
  transition: opacity 0.2s, transform 0.15s !important;
  box-shadow: 0 4px 20px rgba(0,114,255,0.35) !important;
}
div[data-testid="stButton"] > button:hover {
  opacity: 0.9 !important;
  transform: translateY(-1px) !important;
}
div[data-testid="stDownloadButton"] > button {
  background: var(--surface2) !important;
  color: var(--accent) !important;
  border: 1px solid var(--accent) !important;
  font-family: var(--font-mono) !important;
  font-size: 0.8rem !important;
  letter-spacing: 0.04em !important;
  border-radius: 8px !important;
  padding: 0.55rem 1.5rem !important;
  width: 100% !important;
}
div[data-testid="stDownloadButton"] > button:hover {
  background: rgba(0,198,255,0.1) !important;
}

/* ── Images ── */
div[data-testid="stImage"] img {
  border-radius: 10px;
  border: 1px solid var(--border);
  width: 100% !important;
}

/* ── Diagnosis pills ── */
.diag-pill {
  display: inline-block;
  font-family: var(--font-mono);
  font-size: 0.8rem;
  padding: 4px 14px;
  border-radius: 20px;
  font-weight: 500;
  letter-spacing: 0.06em;
}
.pill-bacteria { background: rgba(255,75,110,0.15); color: var(--red); border: 1px solid var(--red); }
.pill-virus    { background: rgba(255,184,48,0.15);  color: var(--amber); border: 1px solid var(--amber); }
.pill-normal   { background: rgba(0,229,160,0.15);  color: var(--green); border: 1px solid var(--green); }

/* ── Stat row ── */
.stat-row {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 0.8rem;
  margin-bottom: 1.2rem;
}
.stat-box {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1rem;
  text-align: center;
}
.stat-box .val {
  font-size: 1.6rem;
  font-weight: 700;
  font-family: var(--font-mono);
  line-height: 1.1;
}
.stat-box .lbl {
  font-size: 0.65rem;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.1em;
  margin-top: 4px;
  font-family: var(--font-mono);
}

/* ── Probability bars ── */
.prob-row {
  display: flex;
  align-items: center;
  gap: 0.8rem;
  margin-bottom: 0.7rem;
}
.prob-label {
  font-family: var(--font-mono);
  font-size: 0.75rem;
  width: 80px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  flex-shrink: 0;
}
.prob-track {
  flex: 1;
  background: var(--surface2);
  border-radius: 5px;
  height: 6px;
  overflow: hidden;
}
.prob-fill { height: 100%; border-radius: 5px; }
.prob-pct {
  font-family: var(--font-mono);
  font-size: 0.75rem;
  width: 42px;
  text-align: right;
  color: var(--text);
}

hr.xpn { border: none; border-top: 1px solid var(--border); margin: 1.5rem 0; }
.severity-high   { color: var(--red);   }
.severity-medium { color: var(--amber); }
.severity-low    { color: var(--green); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="xpn-header">
  <div class="xpn-logo">🫁</div>
  <div class="xpn-title-block">
    <h1>XPneumoNet</h1>
    <p>EXPLAINABLE AI · PNEUMONIA DETECTION · DenseNet-121 + Grad-CAM</p>
  </div>
  <div class="xpn-badge">v1.0 · CLINICAL DEMO</div>
</div>
""", unsafe_allow_html=True)
st.write("🛠 DB Enabled:", DB_ENABLED)


# ─────────────────────────────────────────────
# Load Model
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


# ─────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────
left, right = st.columns([1, 1.7], gap="large")

with left:
    st.markdown('<div class="xpn-card">', unsafe_allow_html=True)
    st.markdown('<div class="xpn-card-title">Patient Information</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        patient_id = st.text_input("Patient ID", placeholder="PT-00421")
    with col2:
        age = st.number_input("Age", min_value=0, max_value=120, value=0)
    name = st.text_input("Full Name", placeholder="e.g. Rajesh Kumar")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="xpn-card">', unsafe_allow_html=True)
    st.markdown('<div class="xpn-card-title">Chest X-Ray Upload</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drag & drop or browse — JPG / PNG / JPEG",
        type=["jpg", "png", "jpeg"],
        label_visibility="visible"
    )
    if uploaded_file:
        st.image(uploaded_file, use_container_width=True, caption="Uploaded X-ray preview")
    st.markdown('</div>', unsafe_allow_html=True)

    run = st.button("⚡  Run AI Diagnostic")

    st.markdown("""
    <div style="margin-top:1rem; font-family:var(--font-mono); font-size:0.66rem;
                color:var(--muted); line-height:1.7; border-top:1px solid var(--border); padding-top:0.8rem;">
      ⚠ FOR RESEARCH &amp; EDUCATIONAL USE ONLY<br>
      Not a substitute for clinical diagnosis.<br>
      Model: DenseNet-121 · Classes: Bacteria / Virus / Normal
    </div>
    """, unsafe_allow_html=True)


with right:
    if run:
        if uploaded_file is not None:
            os.makedirs("temp", exist_ok=True)
            file_path = os.path.join("temp", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            img = cv2.imread(file_path)
            img_resized = cv2.resize(img, (224, 224))
            img_norm = img_resized / 255.0
            img_array = np.reshape(img_norm, (1, 224, 224, 3))

            pred = model.predict(img_array)[0]
            class_index = np.argmax(pred)
            confidence = pred[class_index]
            prediction = classes[class_index]

            heatmap = gradcam(img_array, class_index)
            heatmap = cv2.resize(heatmap, (224, 224))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            superimposed_img = heatmap * 0.4 + img_resized
            heatmap_path = os.path.join("temp", "heatmap.jpg")
            cv2.imwrite(heatmap_path, superimposed_img)

            severity, report = agent_decision(prediction, confidence)

            if DB_ENABLED:
                try:
                    data = (patient_id, name, age, date.today(),
                            prediction, float(confidence), severity, report, file_path)
                    insert_patient(data)
                except Exception as e:
                    st.warning(f"Database error: {e}")

            pill_class = {"BACTERIA": "pill-bacteria", "VIRUS": "pill-virus", "NORMAL": "pill-normal"}.get(prediction, "pill-normal")
            severity_color = (
                "severity-high"   if "high"   in severity.lower() else
                "severity-medium" if "medium" in severity.lower() or "moderate" in severity.lower() else
                "severity-low"
            )
            conf_pct = int(confidence * 100)
            conf_color = "var(--green)" if conf_pct >= 85 else "var(--amber)" if conf_pct >= 65 else "var(--red)"

            # ── Summary stats ──
            st.markdown(f"""
            <div class="stat-row">
              <div class="stat-box">
                <div class="val"><span class="diag-pill {pill_class}">{prediction}</span></div>
                <div class="lbl">Diagnosis</div>
              </div>
              <div class="stat-box">
                <div class="val" style="color:{conf_color};">{conf_pct}%</div>
                <div class="lbl">Confidence</div>
              </div>
              <div class="stat-box">
                <div class="val {severity_color}" style="font-size:1rem; padding-top:0.3rem;">{severity}</div>
                <div class="lbl">Severity</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Images ──
            img_col1, img_col2 = st.columns(2)
            with img_col1:
                st.markdown('<div class="section-label">Original X-Ray</div>', unsafe_allow_html=True)
                st.image(file_path, use_container_width=True)
            with img_col2:
                st.markdown('<div class="section-label">Grad-CAM Activation</div>', unsafe_allow_html=True)
                st.image(heatmap_path, use_container_width=True)

            st.markdown('<hr class="xpn">', unsafe_allow_html=True)

            # ── Probability bars ──
            st.markdown('<div class="xpn-card">', unsafe_allow_html=True)
            st.markdown('<div class="xpn-card-title">Class Probability Distribution</div>', unsafe_allow_html=True)
            bar_colors = {"BACTERIA": "var(--red)", "NORMAL": "var(--green)", "VIRUS": "var(--amber)"}
            for i, cls in enumerate(classes):
                pct = int(pred[i] * 100)
                st.markdown(f"""
                <div class="prob-row">
                  <div class="prob-label">{cls}</div>
                  <div class="prob-track">
                    <div class="prob-fill" style="width:{pct}%; background:{bar_colors[cls]};"></div>
                  </div>
                  <div class="prob-pct">{pct}%</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # ── AI Advice ──
            st.markdown(f"""
            <div class="xpn-card">
              <div class="xpn-card-title">AI Clinical Recommendation</div>
              <p style="font-size:0.88rem; line-height:1.75; color:var(--text); margin:0;">{report}</p>
            </div>
            """, unsafe_allow_html=True)

            # ── Confusion Matrix ──
            st.markdown('<div class="xpn-card">', unsafe_allow_html=True)
            st.markdown('<div class="xpn-card-title">Prediction Confidence Matrix</div>', unsafe_allow_html=True)
            cm = confusion_matrix([class_index], [class_index])
            fig, ax = plt.subplots(figsize=(3.5, 2.5))
            fig.patch.set_facecolor("#0e1825")
            ax.set_facecolor("#0e1825")
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        linewidths=0.5, linecolor="#1e3048",
                        annot_kws={"fontsize": 11, "color": "white"})
            ax.set_xlabel("Predicted", color="#5c7a99", fontsize=8)
            ax.set_ylabel("Actual",    color="#5c7a99", fontsize=8)
            ax.tick_params(colors="#5c7a99", labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor("#1e3048")
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)

            # ── PDF download ──
            pdf_file = generate_pdf(patient_id, name, prediction, confidence, severity, report)
            with open(pdf_file, "rb") as f:
                st.download_button("📄  Download Full Diagnostic Report", f, file_name="xpneumonet_report.pdf")

        else:
            st.markdown("""
            <div class="xpn-card" style="text-align:center; padding:2.5rem 1rem; color:var(--muted);">
              <div style="font-size:2.5rem; margin-bottom:0.8rem;">🩻</div>
              <div style="font-family:var(--font-mono); font-size:0.8rem; letter-spacing:0.08em;">
                NO IMAGE UPLOADED<br>
                <span style="font-size:0.7rem;">Please upload a chest X-ray to begin analysis</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="xpn-card" style="text-align:center; padding:3rem 1.5rem;">
          <div style="font-size:3rem; margin-bottom:1rem;">🫁</div>
          <div style="font-family:var(--font-mono); font-size:0.85rem; color:var(--accent);
                      letter-spacing:0.1em; margin-bottom:0.6rem;">AWAITING SCAN</div>
          <div style="font-size:0.8rem; color:var(--muted); line-height:1.7; max-width:320px; margin:0 auto;">
            Enter patient details, upload a chest X-ray,
            and click <strong style="color:var(--text);">Run AI Diagnostic</strong> to receive
            an explainable AI prediction with Grad-CAM visualisation.
          </div>
          <hr class="xpn" style="max-width:280px; margin:1.5rem auto;">
          <div style="display:flex; justify-content:center; gap:1.2rem; flex-wrap:wrap;">
            <span style="font-family:var(--font-mono); font-size:0.65rem; color:var(--muted);
                         background:var(--surface2); border:1px solid var(--border);
                         border-radius:4px; padding:3px 10px;">DenseNet-121</span>
            <span style="font-family:var(--font-mono); font-size:0.65rem; color:var(--muted);
                         background:var(--surface2); border:1px solid var(--border);
                         border-radius:4px; padding:3px 10px;">Grad-CAM XAI</span>
            <span style="font-family:var(--font-mono); font-size:0.65rem; color:var(--muted);
                         background:var(--surface2); border:1px solid var(--border);
                         border-radius:4px; padding:3px 10px;">3-Class Detection</span>
          </div>
        </div>
        """, unsafe_allow_html=True)