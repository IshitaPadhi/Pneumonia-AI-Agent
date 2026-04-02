import sys
import os
sys.path.append(os.getcwd())
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from agent.agent import agent_decision
from datetime import date

# Optional DB
try:
    from database.mysql_connect import insert_patient
    DB_ENABLED = True
except Exception as e:
    DB_ENABLED = False
    print("DB Import Error:", e)

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="XPneumoNet · AI Pneumonia Diagnostics",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# GLOBAL CSS — light elegant medical theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg:         #f5f7fa;
  --white:      #ffffff;
  --surface:    #ffffff;
  --surface2:   #f0f4f8;
  --border:     #e2e8f0;
  --border2:    #cbd5e0;
  --accent:     #2563eb;
  --accent-lt:  #eff6ff;
  --accent2:    #1d4ed8;
  --teal:       #0891b2;
  --teal-lt:    #ecfeff;
  --green:      #059669;
  --green-lt:   #ecfdf5;
  --amber:      #d97706;
  --amber-lt:   #fffbeb;
  --red:        #dc2626;
  --red-lt:     #fef2f2;
  --text:       #0f172a;
  --text2:      #334155;
  --muted:      #64748b;
  --muted2:     #94a3b8;
  --font-ui:    'Plus Jakarta Sans', sans-serif;
  --font-mono:  'JetBrains Mono', monospace;
  --shadow-sm:  0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
  --shadow-md:  0 4px 12px rgba(0,0,0,0.08), 0 2px 4px rgba(0,0,0,0.04);
  --shadow-lg:  0 10px 30px rgba(0,0,0,0.1), 0 4px 8px rgba(0,0,0,0.04);
}

html, body, [class*="css"] {
  font-family: var(--font-ui) !important;
  background-color: var(--bg) !important;
  color: var(--text) !important;
}

/* Force Streamlit background */
.stApp {
  background-color: var(--bg) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container {
  padding: 0 2.5rem 4rem 2.5rem !important;
  max-width: 1280px !important;
}

/* ── Top Header ── */
.xpn-header {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 1.8rem 0 1.4rem 0;
  border-bottom: 2px solid var(--border);
  margin-bottom: 2rem;
}
.xpn-logo-wrap {
  width: 48px; height: 48px;
  background: linear-gradient(135deg, #2563eb, #0891b2);
  border-radius: 12px;
  display: flex; align-items: center; justify-content: center;
  font-size: 1.6rem;
  box-shadow: 0 4px 12px rgba(37,99,235,0.25);
  flex-shrink: 0;
}
.xpn-title-block h1 {
  font-size: 1.45rem;
  font-weight: 700;
  color: var(--text);
  margin: 0 0 2px 0;
  letter-spacing: -0.3px;
}
.xpn-title-block h1 span {
  color: var(--accent);
}
.xpn-title-block p {
  font-size: 0.73rem;
  color: var(--muted);
  font-family: var(--font-mono);
  margin: 0;
  letter-spacing: 0.04em;
}
.xpn-header-right {
  margin-left: auto;
  display: flex;
  align-items: center;
  gap: 0.7rem;
}
.xpn-badge {
  font-family: var(--font-mono);
  font-size: 0.68rem;
  color: var(--accent);
  background: var(--accent-lt);
  border: 1px solid #bfdbfe;
  border-radius: 6px;
  padding: 4px 12px;
  letter-spacing: 0.04em;
  font-weight: 500;
}
.xpn-status-dot {
  width: 8px; height: 8px;
  border-radius: 50%;
  background: var(--green);
  box-shadow: 0 0 0 3px var(--green-lt);
}

/* ── Card ── */
.xpn-card {
  background: var(--white);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 1.4rem 1.6rem;
  margin-bottom: 1.2rem;
  box-shadow: var(--shadow-sm);
}
.xpn-card-title {
  font-size: 0.7rem;
  font-family: var(--font-mono);
  letter-spacing: 0.1em;
  color: var(--muted);
  text-transform: uppercase;
  margin-bottom: 1rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-weight: 500;
}
.xpn-card-title .dot {
  width: 5px; height: 5px;
  border-radius: 50%;
  background: var(--accent);
  flex-shrink: 0;
}

/* ── Section label ── */
.section-label {
  font-family: var(--font-mono);
  font-size: 0.66rem;
  letter-spacing: 0.12em;
  color: var(--muted);
  text-transform: uppercase;
  margin-bottom: 0.5rem;
  font-weight: 500;
}

/* ── Input overrides ── */
div[data-testid="stTextInput"] label,
div[data-testid="stNumberInput"] label {
  font-family: var(--font-mono) !important;
  font-size: 0.69rem !important;
  letter-spacing: 0.08em !important;
  color: var(--muted) !important;
  text-transform: uppercase !important;
  font-weight: 500 !important;
}
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input {
  background: var(--surface2) !important;
  border: 1.5px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--text) !important;
  font-family: var(--font-ui) !important;
  font-size: 0.88rem !important;
  padding: 0.5rem 0.85rem !important;
  box-shadow: none !important;
  transition: border-color 0.15s, box-shadow 0.15s !important;
}
div[data-testid="stTextInput"] input:focus,
div[data-testid="stNumberInput"] input:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(37,99,235,0.1) !important;
  background: var(--white) !important;
}
div[data-testid="stTextInput"] input::placeholder,
div[data-testid="stNumberInput"] input::placeholder {
  color: var(--muted2) !important;
}

/* ── File uploader ── */
div[data-testid="stFileUploader"] > div {
  background: var(--surface2) !important;
  border: 2px dashed var(--border2) !important;
  border-radius: 10px !important;
  transition: border-color 0.2s, background 0.2s;
}
div[data-testid="stFileUploader"] > div:hover {
  border-color: var(--accent) !important;
  background: var(--accent-lt) !important;
}
div[data-testid="stFileUploader"] label {
  font-family: var(--font-mono) !important;
  font-size: 0.69rem !important;
  letter-spacing: 0.08em !important;
  color: var(--muted) !important;
  text-transform: uppercase !important;
  font-weight: 500 !important;
}

/* ── Primary button ── */
div[data-testid="stButton"] > button {
  background: var(--accent) !important;
  color: #fff !important;
  font-family: var(--font-ui) !important;
  font-weight: 600 !important;
  font-size: 0.88rem !important;
  letter-spacing: 0.01em !important;
  border: none !important;
  border-radius: 9px !important;
  padding: 0.65rem 2rem !important;
  width: 100% !important;
  box-shadow: 0 2px 8px rgba(37,99,235,0.3) !important;
  transition: background 0.15s, box-shadow 0.15s, transform 0.1s !important;
}
div[data-testid="stButton"] > button:hover {
  background: var(--accent2) !important;
  box-shadow: 0 4px 16px rgba(37,99,235,0.4) !important;
  transform: translateY(-1px) !important;
}

/* ── Download button ── */
div[data-testid="stDownloadButton"] > button {
  background: var(--white) !important;
  color: var(--accent) !important;
  border: 1.5px solid var(--accent) !important;
  font-family: var(--font-ui) !important;
  font-size: 0.85rem !important;
  font-weight: 600 !important;
  border-radius: 9px !important;
  padding: 0.6rem 1.5rem !important;
  width: 100% !important;
  box-shadow: var(--shadow-sm) !important;
  transition: background 0.15s !important;
}
div[data-testid="stDownloadButton"] > button:hover {
  background: var(--accent-lt) !important;
}

/* ── Images ── */
div[data-testid="stImage"] img {
  border-radius: 10px;
  border: 1px solid var(--border);
  width: 100% !important;
  box-shadow: var(--shadow-sm);
}

/* ── Alert overrides ── */
div[data-testid="stAlert"] {
  border-radius: 8px !important;
  font-family: var(--font-ui) !important;
  font-size: 0.82rem !important;
}

/* ── Diagnosis pills ── */
.diag-pill {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-family: var(--font-mono);
  font-size: 0.75rem;
  padding: 5px 14px;
  border-radius: 20px;
  font-weight: 600;
  letter-spacing: 0.05em;
}
.pill-bacteria {
  background: var(--red-lt);
  color: var(--red);
  border: 1.5px solid #fecaca;
}
.pill-virus {
  background: var(--amber-lt);
  color: var(--amber);
  border: 1.5px solid #fde68a;
}
.pill-normal {
  background: var(--green-lt);
  color: var(--green);
  border: 1.5px solid #a7f3d0;
}

/* ── Stat row ── */
.stat-row {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 0.85rem;
  margin-bottom: 1.3rem;
}
.stat-box {
  background: var(--white);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.1rem 1rem;
  text-align: center;
  box-shadow: var(--shadow-sm);
  position: relative;
  overflow: hidden;
}
.stat-box::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  border-radius: 12px 12px 0 0;
}
.stat-box.diag::before  { background: var(--accent); }
.stat-box.conf::before  { background: var(--teal); }
.stat-box.sev::before   { background: var(--amber); }
.stat-box .val {
  font-size: 1.5rem;
  font-weight: 700;
  font-family: var(--font-mono);
  line-height: 1.2;
  margin-bottom: 4px;
}
.stat-box .lbl {
  font-size: 0.64rem;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.1em;
  font-family: var(--font-mono);
  font-weight: 500;
}

/* ── Probability bars ── */
.prob-row {
  display: flex;
  align-items: center;
  gap: 0.9rem;
  margin-bottom: 0.75rem;
}
.prob-label {
  font-family: var(--font-mono);
  font-size: 0.72rem;
  width: 76px;
  color: var(--text2);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  flex-shrink: 0;
  font-weight: 500;
}
.prob-track {
  flex: 1;
  background: var(--surface2);
  border-radius: 999px;
  height: 7px;
  overflow: hidden;
}
.prob-fill {
  height: 100%;
  border-radius: 999px;
  transition: width 0.5s ease;
}
.prob-pct {
  font-family: var(--font-mono);
  font-size: 0.73rem;
  width: 40px;
  text-align: right;
  color: var(--text2);
  font-weight: 500;
}

/* ── Advice card ── */
.advice-card {
  background: var(--accent-lt);
  border: 1px solid #bfdbfe;
  border-left: 4px solid var(--accent);
  border-radius: 10px;
  padding: 1rem 1.2rem;
  font-size: 0.87rem;
  line-height: 1.75;
  color: var(--text2);
}

/* ── Divider ── */
hr.xpn {
  border: none;
  border-top: 1px solid var(--border);
  margin: 1.4rem 0;
}

/* ── Severity colors ── */
.severity-high   { color: var(--red);   }
.severity-medium { color: var(--amber); }
.severity-low    { color: var(--green); }

/* ── Disclaimer footer ── */
.xpn-disclaimer {
  margin-top: 1.2rem;
  font-family: var(--font-mono);
  font-size: 0.64rem;
  color: var(--muted2);
  line-height: 1.8;
  border-top: 1px solid var(--border);
  padding-top: 0.9rem;
}

/* ── Idle placeholder ── */
.idle-wrap {
  background: var(--white);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 3.5rem 2rem;
  text-align: center;
  box-shadow: var(--shadow-sm);
}
.idle-icon {
  font-size: 3rem;
  margin-bottom: 1rem;
  line-height: 1;
}
.idle-title {
  font-size: 0.9rem;
  font-weight: 600;
  color: var(--text2);
  margin-bottom: 0.5rem;
  letter-spacing: 0.01em;
}
.idle-sub {
  font-size: 0.78rem;
  color: var(--muted);
  line-height: 1.75;
  max-width: 300px;
  margin: 0 auto 1.4rem auto;
}
.tech-tag {
  display: inline-block;
  font-family: var(--font-mono);
  font-size: 0.63rem;
  color: var(--accent);
  background: var(--accent-lt);
  border: 1px solid #bfdbfe;
  border-radius: 5px;
  padding: 3px 10px;
  margin: 3px;
  font-weight: 500;
}

/* ── Pyplot white background ── */
.stPlotly, [data-testid="stPyplotUserWarning"] { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="xpn-header">
  <div class="xpn-logo-wrap">🫁</div>
  <div class="xpn-title-block">
    <h1>X<span>Pneumo</span>Net</h1>
    <p>EXPLAINABLE AI · PNEUMONIA DETECTION · DenseNet-121 + Grad-CAM</p>
  </div>
  <div class="xpn-header-right">
    <div class="xpn-status-dot"></div>
    <div class="xpn-badge">v1.0 · CLINICAL DEMO</div>
  </div>
</div>
""", unsafe_allow_html=True)


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
left, right = st.columns([1, 1.75], gap="large")

with left:
    # ── Patient Info ──
    st.markdown("""
    <div class="xpn-card">
      <div class="xpn-card-title"><span class="dot"></span>Patient Information</div>
    </div>
    """, unsafe_allow_html=True)

    # Re-open the card visually by nesting Streamlit widgets between styled divs
    st.markdown('<div style="background:white; border:1px solid #e2e8f0; border-radius:14px; padding:0 1.4rem 1.4rem 1.4rem; margin-top:-1.2rem; margin-bottom:1.2rem; box-shadow:0 1px 3px rgba(0,0,0,0.06);">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        patient_id = st.text_input("Patient ID", placeholder="PT-00421")
    with col2:
        age = st.number_input("Age (yrs)", min_value=0, max_value=120, value=0)
    name = st.text_input("Full Name", placeholder="e.g. Rajesh Kumar")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Upload ──
    st.markdown("""
    <div style="background:white; border:1px solid #e2e8f0; border-radius:14px; padding:1.2rem 1.4rem 0.4rem 1.4rem; margin-bottom:0.4rem; box-shadow:0 1px 3px rgba(0,0,0,0.06);">
      <div class="xpn-card-title"><span class="dot"></span>Chest X-Ray Upload</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="background:white; border:1px solid #e2e8f0; border-top:none; border-radius:0 0 14px 14px; padding:0 1.4rem 1.4rem 1.4rem; margin-top:-0.5rem; margin-bottom:1.2rem; box-shadow:0 1px 3px rgba(0,0,0,0.06);">', unsafe_allow_html=True)
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
    <div class="xpn-disclaimer">
      ⚠ FOR RESEARCH &amp; EDUCATIONAL USE ONLY<br>
      Not a substitute for professional clinical diagnosis.<br>
      Model: DenseNet-121 &nbsp;·&nbsp; 3 Classes: Bacteria / Virus / Normal
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# RESULTS PANEL
# ─────────────────────────────────────────────
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

            # ── Derived display values ──
            pill_class = {
                "BACTERIA": "pill-bacteria",
                "VIRUS":    "pill-virus",
                "NORMAL":   "pill-normal"
            }.get(prediction, "pill-normal")

            severity_color = (
                "severity-high"   if "high"    in severity.lower() else
                "severity-medium" if any(w in severity.lower() for w in ["medium", "moderate"]) else
                "severity-low"
            )

            conf_pct = int(confidence * 100)
            conf_color = (
                "#059669" if conf_pct >= 85 else
                "#d97706" if conf_pct >= 65 else
                "#dc2626"
            )

            bar_colors = {
                "BACTERIA": "#ef4444",
                "NORMAL":   "#10b981",
                "VIRUS":    "#f59e0b"
            }

            # ── Stat cards ──
            st.markdown(f"""
            <div class="stat-row">
              <div class="stat-box diag">
                <div class="val">
                  <span class="diag-pill {pill_class}">{prediction}</span>
                </div>
                <div class="lbl" style="margin-top:8px;">Diagnosis</div>
              </div>
              <div class="stat-box conf">
                <div class="val" style="color:{conf_color};">{conf_pct}%</div>
                <div class="lbl">Confidence</div>
              </div>
              <div class="stat-box sev">
                <div class="val {severity_color}" style="font-size:1rem; padding-top:0.25rem;">{severity}</div>
                <div class="lbl">Severity</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Side-by-side images ──
            img_col1, img_col2 = st.columns(2, gap="medium")
            with img_col1:
                st.markdown('<div class="section-label">Original X-Ray</div>', unsafe_allow_html=True)
                st.image(file_path, use_container_width=True)
            with img_col2:
                st.markdown('<div class="section-label">Grad-CAM Activation Map</div>', unsafe_allow_html=True)
                st.image(heatmap_path, use_container_width=True)

            st.markdown('<hr class="xpn">', unsafe_allow_html=True)

            # ── Probability distribution ──
            st.markdown("""
            <div style="background:white; border:1px solid #e2e8f0; border-radius:14px;
                        padding:1.2rem 1.5rem 1rem 1.5rem; margin-bottom:1.2rem;
                        box-shadow:0 1px 3px rgba(0,0,0,0.06);">
              <div class="xpn-card-title"><span class="dot"></span>Class Probability Distribution</div>
            """, unsafe_allow_html=True)

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

            # ── AI Recommendation ──
            st.markdown(f"""
            <div style="background:white; border:1px solid #e2e8f0; border-radius:14px;
                        padding:1.2rem 1.5rem; margin-bottom:1.2rem;
                        box-shadow:0 1px 3px rgba(0,0,0,0.06);">
              <div class="xpn-card-title"><span class="dot"></span>AI Clinical Recommendation</div>
              <div class="advice-card">{report}</div>
            </div>
            """, unsafe_allow_html=True)

            # ── Confusion Matrix ──
            st.markdown("""
            <div style="background:white; border:1px solid #e2e8f0; border-radius:14px;
                        padding:1.2rem 1.5rem 0.5rem 1.5rem; margin-bottom:1.2rem;
                        box-shadow:0 1px 3px rgba(0,0,0,0.06);">
              <div class="xpn-card-title"><span class="dot"></span>Prediction Confidence Matrix</div>
            </div>
            """, unsafe_allow_html=True)

            cm = confusion_matrix([class_index], [class_index])
            fig, ax = plt.subplots(figsize=(4, 3))
            fig.patch.set_facecolor("#ffffff")
            ax.set_facecolor("#ffffff")
            sns.heatmap(
                cm, annot=True, fmt="d",
                cmap=sns.light_palette("#2563eb", as_cmap=True),
                ax=ax, linewidths=1, linecolor="#e2e8f0",
                annot_kws={"fontsize": 12, "color": "#0f172a", "fontweight": "bold"}
            )
            ax.set_xlabel("Predicted", color="#64748b", fontsize=8, labelpad=8)
            ax.set_ylabel("Actual",    color="#64748b", fontsize=8, labelpad=8)
            ax.tick_params(colors="#64748b", labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor("#e2e8f0")
            fig.tight_layout(pad=1.5)
            st.pyplot(fig)

            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

            # ── PDF ──
            pdf_file = generate_pdf(patient_id, name, prediction, confidence, severity, report)
            with open(pdf_file, "rb") as f:
                st.download_button(
                    "📄  Download Full Diagnostic Report",
                    f,
                    file_name="xpneumonet_report.pdf"
                )

        else:
            st.markdown("""
            <div class="idle-wrap">
              <div class="idle-icon">🩻</div>
              <div class="idle-title">No Image Uploaded</div>
              <div class="idle-sub">Please upload a chest X-ray on the left panel before running the diagnostic.</div>
            </div>
            """, unsafe_allow_html=True)

    else:
        # ── Idle / welcome state ──
        st.markdown("""
        <div class="idle-wrap">
          <div class="idle-icon">🫁</div>
          <div class="idle-title">Ready for Analysis</div>
          <div class="idle-sub">
            Fill in the patient details, upload a chest X-ray,
            and click <strong>Run AI Diagnostic</strong> to receive an
            explainable AI prediction with Grad-CAM visualisation.
          </div>
          <div>
            <span class="tech-tag">DenseNet-121</span>
            <span class="tech-tag">Grad-CAM XAI</span>
            <span class="tech-tag">3-Class Detection</span>
            <span class="tech-tag">Monte Carlo Dropout</span>
          </div>
        </div>
        """, unsafe_allow_html=True)