import sys
import os
sys.path.append(os.getcwd())
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from agent.agent import agent_decision
from datetime import date

try:
    from database.mysql_connect import (
        insert_patient, get_recent_patients, get_prediction_stats,
        get_avg_confidence_by_class, get_total_count,
        get_severity_distribution, search_patient
    )
    DB_ENABLED = True
except Exception as e:
    DB_ENABLED = False

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas as rl_canvas

# ─────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="XPneumoNet · Clinical AI Platform",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  --bg:          #f1f5f9;
  --white:       #ffffff;
  --surface:     #f8fafc;
  --border:      #e2e8f0;
  --border2:     #cbd5e1;
  --blue:        #1d4ed8;
  --blue-mid:    #2563eb;
  --blue-lt:     #eff6ff;
  --blue-border: #bfdbfe;
  --teal:        #0e7490;
  --teal-lt:     #ecfeff;
  --green:       #047857;
  --green-lt:    #ecfdf5;
  --green-b:     #6ee7b7;
  --amber:       #b45309;
  --amber-lt:    #fffbeb;
  --amber-b:     #fcd34d;
  --red:         #b91c1c;
  --red-lt:      #fef2f2;
  --red-b:       #fca5a5;
  --text:        #0f172a;
  --text2:       #1e293b;
  --text3:       #334155;
  --muted:       #64748b;
  --muted2:      #94a3b8;
  --font:        'Inter', sans-serif;
  --mono:        'IBM Plex Mono', monospace;
  --r:           12px;
  --r-sm:        8px;
  --shadow:      0 1px 3px rgba(15,23,42,0.06), 0 1px 2px rgba(15,23,42,0.04);
  --shadow-md:   0 4px 16px rgba(15,23,42,0.08), 0 2px 4px rgba(15,23,42,0.04);
}

/* ── Base ── */
html, body, [class*="css"] { font-family: var(--font) !important; background: var(--bg) !important; color: var(--text) !important; }
.stApp { background: var(--bg) !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2.5rem 5rem !important; max-width: 1340px !important; }

/* ── Header ── */
.hdr {
  display: flex; align-items: center; gap: 1rem;
  padding: 1.6rem 0 1.2rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 0;
}
.hdr-icon {
  width: 44px; height: 44px; border-radius: 10px; flex-shrink: 0;
  background: linear-gradient(135deg, #1d4ed8, #0e7490);
  display: flex; align-items: center; justify-content: center;
  font-size: 1.4rem; box-shadow: 0 3px 12px rgba(29,78,216,.22);
}
.hdr-brand { line-height: 1.2; }
.hdr-brand .name { font-size: 1.25rem; font-weight: 700; color: var(--text); letter-spacing: -0.4px; }
.hdr-brand .name span { color: var(--blue-mid); }
.hdr-brand .sub  { font-size: 0.68rem; color: var(--muted); font-family: var(--mono); letter-spacing: .05em; margin-top: 1px; }
.hdr-right { margin-left: auto; display: flex; align-items: center; gap: .75rem; }
.badge {
  font-family: var(--mono); font-size: .65rem; font-weight: 500;
  padding: 3px 10px; border-radius: 5px; letter-spacing: .04em;
}
.badge-blue  { color: var(--blue-mid); background: var(--blue-lt); border: 1px solid var(--blue-border); }
.badge-green { color: var(--green);    background: var(--green-lt); border: 1px solid var(--green-b); display:flex; align-items:center; gap:5px; }
.live-dot { width:6px; height:6px; border-radius:50%; background:var(--green); animation: blink 2s infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.35} }

/* ── Tabs ── */
div[data-testid="stTabs"] [role="tablist"] {
  border-bottom: 1px solid var(--border) !important;
  background: transparent !important; gap: 0 !important;
  padding-top: 1.25rem;
}
div[data-testid="stTabs"] button[role="tab"] {
  font-family: var(--font) !important; font-size: .83rem !important;
  font-weight: 500 !important; color: var(--muted) !important;
  padding: .55rem 1.3rem !important; border: none !important;
  border-bottom: 2px solid transparent !important;
  background: transparent !important; margin-bottom: -1px !important;
  transition: color .15s !important;
}
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
  color: var(--blue-mid) !important; border-bottom-color: var(--blue-mid) !important; font-weight: 600 !important;
}
div[data-testid="stTabs"] [role="tabpanel"] { padding-top: 1.6rem !important; }

/* ── Cards ── */
.card {
  background: var(--white); border: 1px solid var(--border);
  border-radius: var(--r); padding: 1.25rem 1.4rem;
  margin-bottom: 1rem; box-shadow: var(--shadow);
}
.card-hd {
  font-family: var(--mono); font-size: .65rem; font-weight: 500;
  letter-spacing: .1em; color: var(--muted); text-transform: uppercase;
  margin-bottom: .9rem; display: flex; align-items: center; gap: 6px;
}
.card-hd .dot { width:5px; height:5px; border-radius:50%; background:var(--blue-mid); flex-shrink:0; }

/* ── Inputs ── */
div[data-testid="stTextInput"] label,
div[data-testid="stNumberInput"] label {
  font-family: var(--mono) !important; font-size: .65rem !important;
  letter-spacing: .08em !important; color: var(--muted) !important;
  text-transform: uppercase !important; font-weight: 500 !important;
}
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input {
  background: var(--surface) !important; border: 1.5px solid var(--border) !important;
  border-radius: var(--r-sm) !important; color: var(--text) !important;
  font-family: var(--font) !important; font-size: .875rem !important;
  padding: .5rem .85rem !important; box-shadow: none !important;
  transition: border-color .15s, box-shadow .15s !important;
}
div[data-testid="stTextInput"] input:focus,
div[data-testid="stNumberInput"] input:focus {
  border-color: var(--blue-mid) !important;
  box-shadow: 0 0 0 3px rgba(37,99,235,.1) !important;
  background: var(--white) !important;
}
div[data-testid="stTextInput"] input::placeholder,
div[data-testid="stNumberInput"] input::placeholder { color: var(--muted2) !important; }

/* ── File uploader ── */
div[data-testid="stFileUploader"] > div {
  background: var(--surface) !important; border: 2px dashed var(--border2) !important;
  border-radius: var(--r) !important; transition: all .2s;
}
div[data-testid="stFileUploader"] > div:hover {
  border-color: var(--blue-mid) !important; background: var(--blue-lt) !important;
}
div[data-testid="stFileUploader"] label {
  font-family: var(--mono) !important; font-size: .65rem !important;
  color: var(--muted) !important; text-transform: uppercase !important;
  letter-spacing: .08em !important; font-weight: 500 !important;
}

/* ── Primary button ── */
div[data-testid="stButton"] > button {
  background: var(--blue) !important; color: #fff !important;
  font-family: var(--font) !important; font-weight: 600 !important;
  font-size: .875rem !important; border: none !important;
  border-radius: var(--r-sm) !important; padding: .6rem 1.5rem !important;
  width: 100% !important; box-shadow: 0 2px 8px rgba(29,78,216,.25) !important;
  transition: background .15s, box-shadow .15s, transform .1s !important;
}
div[data-testid="stButton"] > button:hover {
  background: #1e40af !important; box-shadow: 0 4px 16px rgba(29,78,216,.35) !important;
  transform: translateY(-1px) !important;
}
div[data-testid="stDownloadButton"] > button {
  background: var(--white) !important; color: var(--blue) !important;
  border: 1.5px solid var(--blue) !important; font-family: var(--font) !important;
  font-weight: 600 !important; font-size: .85rem !important;
  border-radius: var(--r-sm) !important; padding: .55rem 1.5rem !important;
  width: 100% !important; box-shadow: var(--shadow) !important;
  transition: background .15s !important;
}
div[data-testid="stDownloadButton"] > button:hover { background: var(--blue-lt) !important; }

/* ── Search input (inside expander) ── */
div[data-testid="stTextInput"].search-override input {
  border-radius: 999px !important;
}

/* ── Images ── */
div[data-testid="stImage"] img {
  border-radius: var(--r); border: 1px solid var(--border);
  width: 100% !important; box-shadow: var(--shadow);
}

/* ── Diagnosis pills ── */
.pill {
  display: inline-flex; align-items: center; gap: 4px;
  font-family: var(--mono); font-size: .7rem;
  padding: 3px 12px; border-radius: 999px; font-weight: 600; letter-spacing: .04em;
}
.pill-bact { background: var(--red-lt);   color: var(--red);   border: 1px solid var(--red-b); }
.pill-vir  { background: var(--amber-lt); color: var(--amber); border: 1px solid var(--amber-b); }
.pill-norm { background: var(--green-lt); color: var(--green); border: 1px solid var(--green-b); }

/* ── Stat cards ── */
.kpi-grid { display:grid; grid-template-columns:1fr 1fr 1fr; gap:.85rem; margin-bottom:1.1rem; }
.kpi {
  background:var(--white); border:1px solid var(--border); border-radius:var(--r);
  padding:1.1rem 1rem 1rem; text-align:center; box-shadow:var(--shadow); position:relative; overflow:hidden;
}
.kpi::after {
  content:''; position:absolute; bottom:0; left:0; right:0; height:3px;
}
.kpi.k-blue::after  { background:var(--blue-mid); }
.kpi.k-teal::after  { background:var(--teal); }
.kpi.k-amber::after { background:#d97706; }
.kpi.k-red::after   { background:var(--red); }
.kpi.k-green::after { background:var(--green); }
.kpi .v { font-size:1.6rem; font-weight:700; font-family:var(--mono); line-height:1.15; margin-bottom:3px; }
.kpi .l { font-size:.6rem; color:var(--muted); text-transform:uppercase; letter-spacing:.1em; font-family:var(--mono); }

/* ── Probability bars ── */
.pb-row { display:flex; align-items:center; gap:.9rem; margin-bottom:.7rem; }
.pb-lbl { font-family:var(--mono); font-size:.68rem; width:72px; color:var(--text3); text-transform:uppercase; letter-spacing:.06em; flex-shrink:0; font-weight:500; }
.pb-track { flex:1; background:var(--surface); border-radius:999px; height:7px; overflow:hidden; border:1px solid var(--border); }
.pb-fill  { height:100%; border-radius:999px; transition:width .4s ease; }
.pb-pct   { font-family:var(--mono); font-size:.7rem; width:38px; text-align:right; color:var(--text3); font-weight:500; }

/* ── Advice block ── */
.advice {
  background:var(--blue-lt); border:1px solid var(--blue-border);
  border-left:3px solid var(--blue-mid); border-radius:var(--r-sm);
  padding:.9rem 1.1rem; font-size:.85rem; line-height:1.8; color:var(--text3);
}

/* ── Section label ── */
.slabel {
  font-family:var(--mono); font-size:.63rem; letter-spacing:.1em;
  color:var(--muted); text-transform:uppercase; margin-bottom:.45rem; font-weight:500;
}

/* ── History table ── */
.htable { border-radius:var(--r); overflow:hidden; border:1px solid var(--border); }
.hrow {
  display:grid;
  grid-template-columns: 100px 1fr 55px 95px 80px 85px;
  gap:.5rem; align-items:center;
  padding:.7rem 1rem; font-size:.82rem; color:var(--text2);
  border-bottom:1px solid var(--border);
  transition:background .1s;
}
.hrow:hover { background:var(--surface); }
.hrow:last-child { border-bottom:none; }
.hrow.hdr {
  background:var(--surface);
  font-family:var(--mono); font-size:.6rem; letter-spacing:.08em;
  color:var(--muted); text-transform:uppercase; font-weight:500;
}
.hrow.hdr:hover { background:var(--surface); }

/* ── Search bar ── */
.search-wrap { position:relative; margin-bottom:1rem; }
.search-wrap input {
  width:100%; padding:.55rem .9rem .55rem 2.4rem;
  border:1.5px solid var(--border); border-radius:999px;
  background:var(--white); font-family:var(--font); font-size:.85rem;
  color:var(--text); outline:none; transition:border-color .15s, box-shadow .15s;
}
.search-wrap input:focus { border-color:var(--blue-mid); box-shadow:0 0 0 3px rgba(37,99,235,.1); }
.search-icon { position:absolute; left:.75rem; top:50%; transform:translateY(-50%); color:var(--muted); font-size:.9rem; }

/* ── Horizontal rule ── */
hr.d { border:none; border-top:1px solid var(--border); margin:1.2rem 0; }

/* ── Idle ── */
.idle {
  background:var(--white); border:1px dashed var(--border2);
  border-radius:var(--r); padding:3.5rem 2rem; text-align:center; box-shadow:var(--shadow);
}
.idle-icon { font-size:2.8rem; margin-bottom:.8rem; }
.idle-t { font-size:.9rem; font-weight:600; color:var(--text2); margin-bottom:.4rem; }
.idle-s { font-size:.78rem; color:var(--muted); line-height:1.75; max-width:290px; margin:0 auto 1.1rem; }
.tag {
  display:inline-block; font-family:var(--mono); font-size:.6rem;
  color:var(--blue-mid); background:var(--blue-lt); border:1px solid var(--blue-border);
  border-radius:4px; padding:2px 9px; margin:2px; font-weight:500;
}

/* ── Disclaimer ── */
.disc { font-family:var(--mono); font-size:.61rem; color:var(--muted2); line-height:1.8; border-top:1px solid var(--border); padding-top:.75rem; margin-top:1rem; }

/* ── Alert overrides ── */
div[data-testid="stAlert"] { border-radius:var(--r-sm) !important; font-size:.82rem !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────
db_status = "LIVE" if DB_ENABLED else "OFFLINE"
st.markdown(f"""
<div class="hdr">
  <div class="hdr-icon">🫁</div>
  <div class="hdr-brand">
    <div class="name">X<span>Pneumo</span>Net</div>
    <div class="sub">EXPLAINABLE AI · PNEUMONIA DETECTION · DenseNet-121 + Grad-CAM</div>
  </div>
  <div class="hdr-right">
    <div class="badge badge-green"><div class="live-dot"></div>MODEL READY</div>
    <div class="badge badge-{'blue' if DB_ENABLED else 'blue'}">DB {db_status}</div>
    <div class="badge badge-blue">v1.0</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────
tab_dx, tab_hist, tab_analytics = st.tabs([
    "🔬  Diagnostic",
    "📋  Patient Records",
    "📊  Analytics",
])


# ─────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/best_pneumonia_model.keras")

model = load_model()
classes = ["BACTERIA", "NORMAL", "VIRUS"]

last_conv = None
for layer in reversed(model.layers):
    if "conv" in layer.name:
        last_conv = layer.name
        break

grad_model = tf.keras.models.Model(
    inputs=model.input,
    outputs=[model.get_layer(last_conv).output, model.output]
)

def gradcam(img_array, class_idx):
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        loss = preds[:, class_idx]
    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    hm = conv_out @ pooled[..., tf.newaxis]
    hm = tf.squeeze(hm)
    hm = np.maximum(hm, 0)
    hm /= np.max(hm) + 1e-8
    return hm

def generate_pdf(pid, pname, page, pred, conf, sev, rpt):
    os.makedirs("temp", exist_ok=True)
    path = "temp/report.pdf"
    c = rl_canvas.Canvas(path, pagesize=letter)
    W, _ = letter

    # Header band
    c.setFillColorRGB(0.11, 0.31, 0.85)
    c.rect(0, 720, W, 72, fill=1, stroke=0)
    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(48, 756, "XPneumoNet")
    c.setFont("Helvetica", 9)
    c.drawString(48, 740, "AI-Powered Pneumonia Detection Report")
    c.setFont("Helvetica", 9)
    c.drawRightString(W - 48, 748, f"Date: {date.today()}")

    # Patient details
    c.setFillColorRGB(0.06, 0.09, 0.25)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(48, 700, "PATIENT DETAILS")
    c.setStrokeColorRGB(0.88, 0.91, 0.94)
    c.line(48, 695, W - 48, 695)
    c.setFillColorRGB(0.2, 0.18, 0.3)
    c.setFont("Helvetica", 10)
    c.drawString(48, 678, f"Patient ID:   {pid}")
    c.drawString(48, 662, f"Name:         {pname}")
    c.drawString(48, 646, f"Age:          {page} years")

    # Result
    c.setFillColorRGB(0.06, 0.09, 0.25)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(48, 620, "DIAGNOSTIC RESULT")
    c.line(48, 615, W - 48, 615)
    c.setFont("Helvetica", 10)
    c.setFillColorRGB(0.2, 0.18, 0.3)
    c.drawString(48, 598, f"Prediction:   {pred}")
    c.drawString(48, 582, f"Confidence:   {conf:.1%}")
    c.drawString(48, 566, f"Severity:     {sev}")

    # Advice
    c.setFillColorRGB(0.06, 0.09, 0.25)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(48, 542, "AI CLINICAL RECOMMENDATION")
    c.line(48, 537, W - 48, 537)
    c.setFont("Helvetica", 9)
    c.setFillColorRGB(0.2, 0.18, 0.3)
    words, line, y = rpt.split(), "", 520
    for w in words:
        if len(line) + len(w) < 90:
            line += w + " "
        else:
            c.drawString(48, y, line); y -= 14; line = w + " "
    if line:
        c.drawString(48, y, line)

    # Footer
    c.setFillColorRGB(0.58, 0.64, 0.71)
    c.setFont("Helvetica-Oblique", 7.5)
    c.drawString(48, 40, "FOR RESEARCH & EDUCATIONAL USE ONLY — Not a substitute for professional clinical diagnosis.")
    c.save()
    return path


# ════════════════════════════════════════════════════
# TAB 1 — DIAGNOSTIC
# ════════════════════════════════════════════════════
with tab_dx:
    L, R = st.columns([1, 1.8], gap="large")

    with L:
        # Patient info card
        st.markdown("""
        <div class="card">
          <div class="card-hd"><span class="dot"></span>Patient Information</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("""<div style="background:white;border:1px solid #e2e8f0;border-top:none;
            border-radius:0 0 12px 12px;padding:0 1.3rem 1.3rem;margin-top:-1rem;margin-bottom:1rem;
            box-shadow:0 1px 3px rgba(15,23,42,.06);">""", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            patient_id = st.text_input("Patient ID", placeholder="PT-00421")
        with c2:
            age = st.number_input("Age (yrs)", min_value=0, max_value=120, value=0)
        name = st.text_input("Full Name", placeholder="e.g. Rajesh Kumar")
        st.markdown("</div>", unsafe_allow_html=True)

        # Upload card
        st.markdown("""
        <div class="card">
          <div class="card-hd"><span class="dot"></span>Chest X-Ray Upload</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("""<div style="background:white;border:1px solid #e2e8f0;border-top:none;
            border-radius:0 0 12px 12px;padding:0 1.3rem 1.3rem;margin-top:-1rem;margin-bottom:1rem;
            box-shadow:0 1px 3px rgba(15,23,42,.06);">""", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drag & drop or browse — JPG / PNG / JPEG",
            type=["jpg","png","jpeg"], label_visibility="visible"
        )
        if uploaded_file:
            st.image(uploaded_file, use_container_width=True, caption="Preview")
        st.markdown("</div>", unsafe_allow_html=True)

        run = st.button("⚡  Run AI Diagnostic")
        st.markdown("""
        <div class="disc">
          ⚠ FOR RESEARCH &amp; EDUCATIONAL USE ONLY<br>
          Not a substitute for professional clinical diagnosis.<br>
          Model: DenseNet-121 &nbsp;·&nbsp; Focal Loss &nbsp;·&nbsp; Classes: Bacteria / Virus / Normal
        </div>""", unsafe_allow_html=True)

    with R:
        if run:
            if uploaded_file is not None:
                with st.spinner("Analysing chest X-ray…"):
                    os.makedirs("temp", exist_ok=True)
                    fpath = os.path.join("temp", uploaded_file.name)
                    with open(fpath, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    img  = cv2.imread(fpath)
                    img_r = cv2.resize(img, (224, 224))
                    arr   = np.reshape(img_r / 255.0, (1, 224, 224, 3))

                    pred = model.predict(arr)[0]
                    ci   = np.argmax(pred)
                    conf = pred[ci]
                    label = classes[ci]

                    hm   = gradcam(arr, ci)
                    hm   = cv2.resize(hm, (224, 224))
                    hm   = cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_JET)
                    sup  = hm * 0.4 + img_r
                    hpath = os.path.join("temp", "heatmap.jpg")
                    cv2.imwrite(hpath, sup)

                    severity, report = agent_decision(label, conf)

                    if DB_ENABLED:
                        try:
                            insert_patient((patient_id, name, age, date.today(),
                                            label, float(conf), severity, report, fpath))
                        except Exception as e:
                            st.warning(f"DB write error: {e}")

                # ── Derived display values ──
                pill_map = {"BACTERIA":"pill-bact","VIRUS":"pill-vir","NORMAL":"pill-norm"}
                pc = pill_map.get(label, "pill-norm")
                sev_color = ("#b91c1c" if "high" in severity.lower()
                             else "#b45309" if any(w in severity.lower() for w in ["med","mod"])
                             else "#047857")
                conf_pct = int(conf * 100)
                conf_hex = "#047857" if conf_pct>=85 else "#b45309" if conf_pct>=65 else "#b91c1c"
                barcol = {"BACTERIA":"#ef4444","NORMAL":"#10b981","VIRUS":"#f59e0b"}

                # KPI strip
                st.markdown(f"""
                <div class="kpi-grid">
                  <div class="kpi k-blue">
                    <div class="v"><span class="pill {pc}">{label}</span></div>
                    <div class="l" style="margin-top:8px;">AI Diagnosis</div>
                  </div>
                  <div class="kpi k-teal">
                    <div class="v" style="color:{conf_hex};">{conf_pct}%</div>
                    <div class="l">Model Confidence</div>
                  </div>
                  <div class="kpi k-amber">
                    <div class="v" style="font-size:.95rem;padding-top:.3rem;color:{sev_color};">{severity}</div>
                    <div class="l">Severity Level</div>
                  </div>
                </div>""", unsafe_allow_html=True)

                # Side-by-side scans
                ic1, ic2 = st.columns(2, gap="medium")
                with ic1:
                    st.markdown('<div class="slabel">Original X-Ray</div>', unsafe_allow_html=True)
                    st.image(fpath, use_container_width=True)
                with ic2:
                    st.markdown('<div class="slabel">Grad-CAM Heatmap</div>', unsafe_allow_html=True)
                    st.image(hpath, use_container_width=True)

                st.markdown('<hr class="d">', unsafe_allow_html=True)

                bot1, bot2 = st.columns(2, gap="large")

                with bot1:
                    # Probability bars
                    st.markdown("""<div style="background:white;border:1px solid #e2e8f0;border-radius:12px;
                        padding:1.1rem 1.3rem 1rem;box-shadow:0 1px 3px rgba(15,23,42,.06);">
                      <div class="card-hd"><span class="dot"></span>Class Probabilities</div>""",
                        unsafe_allow_html=True)
                    for i, cls in enumerate(classes):
                        pct = int(pred[i] * 100)
                        st.markdown(f"""
                        <div class="pb-row">
                          <div class="pb-lbl">{cls}</div>
                          <div class="pb-track">
                            <div class="pb-fill" style="width:{pct}%;background:{barcol[cls]};"></div>
                          </div>
                          <div class="pb-pct">{pct}%</div>
                        </div>""", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                    # Confidence matrix
                    st.markdown("""<div style="background:white;border:1px solid #e2e8f0;border-radius:12px;
                        padding:1.1rem 1.3rem .6rem;margin-top:1rem;box-shadow:0 1px 3px rgba(15,23,42,.06);">
                      <div class="card-hd"><span class="dot"></span>Confidence Matrix</div>""",
                        unsafe_allow_html=True)
                    cm = confusion_matrix([ci], [ci])
                    fig, ax = plt.subplots(figsize=(3.2, 2.4))
                    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
                    sns.heatmap(cm, annot=True, fmt="d",
                                cmap=sns.light_palette("#1d4ed8", as_cmap=True),
                                ax=ax, linewidths=1, linecolor="#e2e8f0",
                                annot_kws={"fontsize":12,"color":"#0f172a","fontweight":"bold"})
                    ax.set_xlabel("Predicted", color="#64748b", fontsize=7.5, labelpad=6)
                    ax.set_ylabel("Actual", color="#64748b", fontsize=7.5, labelpad=6)
                    ax.tick_params(colors="#64748b", labelsize=7)
                    for sp in ax.spines.values(): sp.set_edgecolor("#e2e8f0")
                    fig.tight_layout(pad=1.2)
                    st.pyplot(fig)
                    st.markdown("</div>", unsafe_allow_html=True)

                with bot2:
                    # AI recommendation
                    st.markdown(f"""
                    <div style="background:white;border:1px solid #e2e8f0;border-radius:12px;
                        padding:1.1rem 1.3rem;box-shadow:0 1px 3px rgba(15,23,42,.06);height:100%;">
                      <div class="card-hd"><span class="dot"></span>AI Clinical Recommendation</div>
                      <div class="advice">{report}</div>
                    """, unsafe_allow_html=True)

                    # Confidence interpretation
                    if conf_pct >= 85:
                        badge_txt = "✓ High confidence — prediction is reliable"
                        badge_bg  = "var(--green-lt)"; badge_bdr = "var(--green-b)"; badge_col = "var(--green)"
                    elif conf_pct >= 65:
                        badge_txt = "⚠ Moderate confidence — consider clinical correlation"
                        badge_bg  = "var(--amber-lt)"; badge_bdr = "var(--amber-b)"; badge_col = "var(--amber)"
                    else:
                        badge_txt = "✕ Low confidence — retesting recommended"
                        badge_bg  = "var(--red-lt)"; badge_bdr = "var(--red-b)"; badge_col = "var(--red)"

                    st.markdown(f"""
                    <div style="margin-top:.9rem;background:{badge_bg};border:1px solid {badge_bdr};
                        border-radius:8px;padding:.7rem 1rem;font-size:.78rem;
                        color:{badge_col};font-weight:500;font-family:var(--mono);letter-spacing:.01em;">
                      {badge_txt}
                    </div></div>""", unsafe_allow_html=True)

                # PDF
                st.markdown("<div style='margin-top:.8rem'>", unsafe_allow_html=True)
                pdf = generate_pdf(patient_id, name, age, label, conf, severity, report)
                with open(pdf, "rb") as f:
                    st.download_button("📄  Download Diagnostic Report", f, file_name="xpneumonet_report.pdf")
                st.markdown("</div>", unsafe_allow_html=True)

            else:
                st.markdown("""
                <div class="idle">
                  <div class="idle-icon">🩻</div>
                  <div class="idle-t">No Image Provided</div>
                  <div class="idle-s">Upload a chest X-ray in the left panel before running the analysis.</div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="idle">
              <div class="idle-icon">🫁</div>
              <div class="idle-t">Ready for Diagnostic</div>
              <div class="idle-s">Enter patient details, upload a chest X-ray, and click <strong>Run AI Diagnostic</strong>.</div>
              <div>
                <span class="tag">DenseNet-121</span>
                <span class="tag">Grad-CAM XAI</span>
                <span class="tag">Focal Loss</span>
                <span class="tag">MC Dropout</span>
                <span class="tag">3-Class</span>
              </div>
            </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════
# TAB 2 — PATIENT RECORDS
# ════════════════════════════════════════════════════
with tab_hist:
    # Search bar + refresh
    sh1, sh2 = st.columns([3, 1], gap="medium")
    with sh1:
        search_q = st.text_input("", placeholder="🔍  Search by patient name or ID…", label_visibility="collapsed")
    with sh2:
        refresh = st.button("↻  Refresh")

    # Fetch rows
    if DB_ENABLED:
        try:
            if search_q:
                rows = search_patient(search_q)
            else:
                rows = get_recent_patients(limit=50)
        except Exception as e:
            st.error(f"Could not load records: {e}")
            rows = []
    else:
        # Demo data fallback
        rows = [
            ("PT-001", "Arjun Mehta",    38, "2025-06-01", "BACTERIA", 0.91, "High",   "Antibiotic therapy recommended."),
            ("PT-002", "Priya Sharma",   24, "2025-06-01", "NORMAL",   0.97, "Low",    "No abnormalities detected."),
            ("PT-003", "Rohit Verma",    55, "2025-05-31", "VIRUS",    0.76, "Medium", "Antiviral support suggested."),
            ("PT-004", "Sneha Iyer",     42, "2025-05-30", "BACTERIA", 0.88, "High",   "Immediate pulmonologist referral."),
            ("PT-005", "Karan Nair",     29, "2025-05-29", "NORMAL",   0.95, "Low",    "Lungs appear clear."),
            ("PT-006", "Meena Pillai",   61, "2025-05-28", "VIRUS",    0.72, "Medium", "Monitor oxygen saturation."),
            ("PT-007", "Dev Kapoor",     47, "2025-05-27", "BACTERIA", 0.84, "High",   "Broad-spectrum antibiotics advised."),
        ]

    # Quick summary strip
    if rows:
        total_r = len(rows)
        bact_r  = sum(1 for r in rows if r[4]=="BACTERIA")
        vir_r   = sum(1 for r in rows if r[4]=="VIRUS")
        norm_r  = sum(1 for r in rows if r[4]=="NORMAL")
        st.markdown(f"""
        <div style="display:flex;gap:.7rem;margin-bottom:1rem;flex-wrap:wrap;">
          <div class="kpi k-blue" style="flex:1;min-width:120px;padding:.8rem;">
            <div class="v" style="font-size:1.3rem;">{total_r}</div>
            <div class="l">Records Shown</div>
          </div>
          <div class="kpi k-red" style="flex:1;min-width:120px;padding:.8rem;">
            <div class="v" style="font-size:1.3rem;color:#b91c1c;">{bact_r}</div>
            <div class="l">Bacterial</div>
          </div>
          <div class="kpi k-amber" style="flex:1;min-width:120px;padding:.8rem;">
            <div class="v" style="font-size:1.3rem;color:#b45309;">{vir_r}</div>
            <div class="l">Viral</div>
          </div>
          <div class="kpi k-green" style="flex:1;min-width:120px;padding:.8rem;">
            <div class="v" style="font-size:1.3rem;color:#047857;">{norm_r}</div>
            <div class="l">Normal</div>
          </div>
        </div>""", unsafe_allow_html=True)

    # Table
    st.markdown('<div class="htable">', unsafe_allow_html=True)
    st.markdown("""
    <div class="hrow hdr">
      <span>Patient ID</span><span>Name</span><span>Age</span>
      <span>Date</span><span>Diagnosis</span><span>Confidence</span>
    </div>""", unsafe_allow_html=True)

    if rows:
        pill_map2 = {"BACTERIA":"pill-bact","VIRUS":"pill-vir","NORMAL":"pill-norm"}
        for r in rows:
            pid_  = r[0]; pname_ = r[1]; age_  = r[2]
            dt_   = str(r[3])[:10];  pred_ = r[4]
            conf_ = float(r[5]);     pc_   = pill_map2.get(pred_, "pill-norm")
            conf_str = f"{int(conf_*100)}%"
            st.markdown(f"""
            <div class="hrow">
              <span style="font-family:var(--mono);font-size:.72rem;color:var(--muted);">{pid_}</span>
              <span style="font-weight:500;">{pname_}</span>
              <span style="color:var(--muted);">{age_}</span>
              <span style="font-family:var(--mono);font-size:.72rem;color:var(--muted2);">{dt_}</span>
              <span><span class="pill {pc_}">{pred_}</span></span>
              <span style="font-family:var(--mono);font-size:.75rem;font-weight:500;">{conf_str}</span>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="padding:2.5rem;text-align:center;color:var(--muted);font-size:.82rem;">
          No records found. Run a diagnostic to populate the database.
        </div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    if not DB_ENABLED:
        st.markdown("""<div style="margin-top:.7rem;font-family:var(--mono);font-size:.62rem;
            color:var(--muted2);">⚠ Demo data — database not connected. Check your .env file.</div>""",
            unsafe_allow_html=True)


# ════════════════════════════════════════════════════
# TAB 3 — ANALYTICS
# ════════════════════════════════════════════════════
with tab_analytics:
    # Fetch from DB or use demo
    if DB_ENABLED:
        try:
            total_db   = get_total_count()
            stats_db   = get_prediction_stats()          # [(label, count), ...]
            conf_db    = get_avg_confidence_by_class()   # [(label, avg_conf), ...]
            sev_db     = get_severity_distribution()     # [(severity, count), ...]
            pred_map   = {k: v for k, v in stats_db}
            conf_map   = {k: float(v)*100 for k, v in conf_db}
        except:
            total_db   = 0; pred_map = {}; conf_map = {}; sev_db = []
    else:
        total_db = 127
        pred_map = {"BACTERIA": 54, "NORMAL": 48, "VIRUS": 25}
        conf_map = {"BACTERIA": 86.4, "NORMAL": 94.1, "VIRUS": 74.8}
        sev_db   = [("High", 54), ("Low", 48), ("Medium", 25)]

    b_ = pred_map.get("BACTERIA", 0)
    n_ = pred_map.get("NORMAL", 0)
    v_ = pred_map.get("VIRUS", 0)

    # KPI row
    st.markdown(f"""
    <div class="kpi-grid" style="grid-template-columns:repeat(5,1fr);">
      <div class="kpi k-blue">
        <div class="v" style="color:var(--blue-mid);">{total_db}</div>
        <div class="l">Total Scans</div>
      </div>
      <div class="kpi k-red">
        <div class="v" style="color:#b91c1c;">{b_}</div>
        <div class="l">Bacterial</div>
      </div>
      <div class="kpi k-amber">
        <div class="v" style="color:#b45309;">{v_}</div>
        <div class="l">Viral</div>
      </div>
      <div class="kpi k-green">
        <div class="v" style="color:#047857;">{n_}</div>
        <div class="l">Normal</div>
      </div>
      <div class="kpi k-teal">
        <div class="v" style="color:var(--teal);">{b_+v_}</div>
        <div class="l">Pneumonia</div>
      </div>
    </div>""", unsafe_allow_html=True)

    row1c1, row1c2 = st.columns(2, gap="large")

    with row1c1:
        st.markdown("""<div style="background:white;border:1px solid #e2e8f0;border-radius:12px;
            padding:1.1rem 1.3rem;box-shadow:0 1px 3px rgba(15,23,42,.06);">
          <div class="card-hd"><span class="dot"></span>Diagnosis Distribution</div>""",
            unsafe_allow_html=True)
        labels_ = list(pred_map.keys()) or ["BACTERIA","NORMAL","VIRUS"]
        sizes_  = list(pred_map.values()) or [54, 48, 25]
        colors_ = ["#ef4444","#10b981","#f59e0b"][:len(labels_)]
        fig1, ax1 = plt.subplots(figsize=(4, 3.6))
        fig1.patch.set_facecolor("white")
        wedges, texts, autotexts = ax1.pie(
            sizes_, labels=labels_, autopct="%1.0f%%",
            colors=colors_, startangle=140,
            wedgeprops=dict(edgecolor="white", linewidth=2.5),
            textprops=dict(fontsize=8.5, color="#334155")
        )
        for at in autotexts:
            at.set_fontsize(7.5); at.set_color("white"); at.set_fontweight("bold")
        ax1.set_title("Case Mix by Class", fontsize=8.5, color="#64748b", pad=10)
        fig1.tight_layout()
        st.pyplot(fig1)
        st.markdown("</div>", unsafe_allow_html=True)

    with row1c2:
        st.markdown("""<div style="background:white;border:1px solid #e2e8f0;border-radius:12px;
            padding:1.1rem 1.3rem;box-shadow:0 1px 3px rgba(15,23,42,.06);">
          <div class="card-hd"><span class="dot"></span>Average Confidence by Class</div>""",
            unsafe_allow_html=True)
        conf_classes = list(conf_map.keys()) or ["BACTERIA","NORMAL","VIRUS"]
        conf_vals    = list(conf_map.values()) or [86.4, 94.1, 74.8]
        bar_cols_    = [{"BACTERIA":"#ef4444","NORMAL":"#10b981","VIRUS":"#f59e0b"}.get(c,"#1d4ed8") for c in conf_classes]
        fig2, ax2 = plt.subplots(figsize=(4, 3.6))
        fig2.patch.set_facecolor("white"); ax2.set_facecolor("white")
        bars_ = ax2.bar(conf_classes, conf_vals, color=bar_cols_, width=0.5,
                        edgecolor="white", linewidth=2)
        for bar_ in bars_:
            ax2.text(bar_.get_x() + bar_.get_width()/2,
                     bar_.get_height() + .5,
                     f"{bar_.get_height():.1f}%",
                     ha="center", va="bottom", fontsize=8, color="#334155", fontweight="bold")
        ax2.set_ylabel("Avg Confidence (%)", color="#64748b", fontsize=7.5)
        ax2.set_ylim(0, 105)
        ax2.tick_params(colors="#64748b", labelsize=8)
        ax2.set_facecolor("white")
        ax2.axhline(y=85, color="#1d4ed8", linestyle="--", linewidth=1, alpha=0.5)
        ax2.text(len(conf_classes)-0.5, 86.5, "Target 85%",
                 color="#1d4ed8", fontsize=7, alpha=0.7)
        for sp in ax2.spines.values(): sp.set_edgecolor("#e2e8f0")
        ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
        ax2.set_title("Model Confidence per Class", fontsize=8.5, color="#64748b", pad=10)
        fig2.tight_layout()
        st.pyplot(fig2)
        st.markdown("</div>", unsafe_allow_html=True)

    # Severity breakdown bar
    st.markdown("""<div style="background:white;border:1px solid #e2e8f0;border-radius:12px;
        padding:1.1rem 1.3rem;box-shadow:0 1px 3px rgba(15,23,42,.06);margin-top:1rem;">
      <div class="card-hd"><span class="dot"></span>Severity Distribution</div>""",
        unsafe_allow_html=True)
    sev_map = {k: v for k, v in sev_db} if sev_db else {"High":54,"Low":48,"Medium":25}
    sev_labels = ["High","Medium","Low"]
    sev_vals   = [sev_map.get(s, 0) for s in sev_labels]
    sev_cols   = ["#ef4444","#f59e0b","#10b981"]
    fig3, ax3 = plt.subplots(figsize=(8, 2.2))
    fig3.patch.set_facecolor("white"); ax3.set_facecolor("white")
    ax3.barh(sev_labels, sev_vals, color=sev_cols, height=0.45, edgecolor="white")
    for i, v_ in enumerate(sev_vals):
        ax3.text(v_ + .3, i, str(v_), va="center", fontsize=8.5, color="#334155", fontweight="bold")
    ax3.set_xlabel("Patients", color="#64748b", fontsize=7.5)
    ax3.tick_params(colors="#64748b", labelsize=8)
    for sp in ax3.spines.values(): sp.set_edgecolor("#e2e8f0")
    ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)
    fig3.tight_layout(pad=1)
    st.pyplot(fig3)
    st.markdown("</div>", unsafe_allow_html=True)

    # Model insight card
    weak = min(conf_map, key=conf_map.get) if conf_map else "VIRUS"
    weak_conf = conf_map.get(weak, 74.8)
    st.markdown(f"""
    <div style="background:#eff6ff;border:1px solid #bfdbfe;border-left:3px solid #1d4ed8;
        border-radius:12px;padding:1rem 1.2rem;margin-top:1rem;">
      <div class="card-hd" style="margin-bottom:.5rem;"><span class="dot"></span>Model Insight</div>
      <div style="font-size:.84rem;color:#1e293b;line-height:1.8;">
        <strong>{weak}</strong> class shows the lowest average confidence at <strong>{weak_conf:.1f}%</strong>.
        Consider collecting additional <strong>{weak}</strong> samples and retraining to improve recall for this class.
        Use <code style="background:#dbeafe;padding:1px 5px;border-radius:3px;font-size:.78rem;">SELECT prediction, AVG(confidence) FROM patients GROUP BY prediction;</code>
        to monitor this over time.
      </div>
    </div>""", unsafe_allow_html=True)

    if not DB_ENABLED:
        st.markdown("""<div style="margin-top:.8rem;font-family:var(--mono);font-size:.62rem;
            color:var(--muted2);">⚠ Showing demo data. Connect your MySQL database via .env to see live analytics.</div>""",
            unsafe_allow_html=True)