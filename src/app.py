# NEUROSENSE — CLINICAL LIVE STREAMLIT UI 

import streamlit as st
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import subprocess
import os
from datetime import datetime

import matplotlib.pyplot as plt

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Image, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm


# CONFIGURE


LIVE_SWEEP_DIR = "live_sweeps"
REPORT_DIR = "reports"
MODEL_PATH = "model/stable_v2/neurosense_stable_v2.pth"

NANOVNA_SAVER_PATH = r"D:\Nanovna_Saver.exe"

TARGET_POINTS = 101
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(REPORT_DIR, exist_ok=True)

CLASS_MAP = {
    0: "Ischemia",
    1: "Hemorrhage",
    2: "Tumor",
    3: "Normal"
}

# PAGE

st.set_page_config(page_title="NeuroSENSE Clinical", layout="centered")

st.title("🧠 NeuroSENSE — Brain Diagnosis System")
st.caption("AI-Assisted Microwave Neurological Screening")

# CNN MODEL 

class NeuroSENSEStable(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1,32,7,padding=3),
            nn.GELU(),
            nn.Conv1d(32,64,5,padding=2),
            nn.GELU(),
            nn.Conv1d(64,128,3,padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(8)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*8,128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128,4)
        )

    def forward(self,x):
        return self.classifier(self.features(x))

# LOAD MODEL

@st.cache_resource
def load_model():
    model = NeuroSENSEStable().to(DEVICE)
    state = torch.load(MODEL_PATH,
                       map_location=DEVICE,
                       weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model

model = load_model()

# FUNCTIONS

def launch_nanovna():
    try:
        subprocess.Popen(NANOVNA_SAVER_PATH)
        return True
    except Exception as e:
        st.error(e)
        return False


def get_latest_s1p():
    files = list(Path(LIVE_SWEEP_DIR).glob("*.s1p"))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def read_s1p(filepath):

    freq, real, imag = [], [], []

    with open(filepath, "r") as f:
        for line in f:

            line=line.strip()

            if not line or line.startswith("!") or line.startswith("#"):
                continue

            parts=line.split()
            if len(parts)!=3:
                continue

            try:
                freq.append(float(parts[0]))
                real.append(float(parts[1]))
                imag.append(float(parts[2]))
            except:
                continue

    if len(freq)<20:
        return None

    freq=np.array(freq)
    complex_s=np.array(real)+1j*np.array(imag)

    logmag=20*np.log10(np.abs(complex_s)+1e-12)

    new_freq=np.linspace(freq.min(),freq.max(),TARGET_POINTS)
    return np.interp(new_freq,freq,logmag)


def normalize(x):
    return (x-x.mean())/(x.std()+1e-8)


def predict(signal):

    x=normalize(signal)

    tensor=torch.tensor(x).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits=model(tensor)
        probs=torch.softmax(logits,dim=1).cpu().numpy()[0]

    return np.argmax(probs),probs

# MEDICAL PDF REPORT GENERATION
def generate_pdf(label, confidence, probs, signal):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = f"{REPORT_DIR}/NeuroSENSE_Report_{timestamp}.pdf"

    sig_img = f"{REPORT_DIR}/sig_{timestamp}.png"
    prob_img = f"{REPORT_DIR}/prob_{timestamp}.png"

    # ===== LARGE HIGH-QUALITY PLOTS =====
    plt.figure(figsize=(6,3))
    plt.plot(signal)
    plt.title("Microwave Signature (S21 LogMag)", fontsize=12)
    plt.tight_layout()
    plt.savefig(sig_img, dpi=200)
    plt.close()

    plt.figure(figsize=(6,3))
    plt.bar(CLASS_MAP.values(), probs)
    plt.title("AI Prediction Probabilities", fontsize=12)
    plt.tight_layout()
    plt.savefig(prob_img, dpi=200)
    plt.close()

    # ===== PDF =====
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=12*mm,
        rightMargin=12*mm,
        topMargin=12*mm,
        bottomMargin=12*mm
    )

    styles = getSampleStyleSheet()

    # Bigger readable clinical font
    body = ParagraphStyle(
        "body",
        parent=styles["Normal"],
        fontSize=11,
        leading=14,
        spaceAfter=10
    )

    story = []

    # ===== TITLE =====
    story.append(Paragraph(
        "NeuroSENSE Clinical Diagnostic Report",
        ParagraphStyle(
            "title",
            parent=styles["Title"],
            fontSize=20,
            spaceAfter=14
        )
    ))

    # ===== INFO TABLE =====
    table = Table([
        ["Report ID", timestamp,
         "Date", datetime.now().strftime("%d-%m-%Y")],
        ["System", "NeuroSENSE AI Scanner",
         "Confidence", f"{confidence:.2f}%"]
    ], colWidths=[80,180,80,160])

    table.setStyle(TableStyle([
        ("GRID",(0,0),(-1,-1),1,colors.black),
        ("BACKGROUND",(0,0),(-1,0),colors.lightgrey),
        ("FONTSIZE",(0,0),(-1,-1),11),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("LEFTPADDING",(0,0),(-1,-1),6),
        ("RIGHTPADDING",(0,0),(-1,-1),6),
    ]))

    story.append(table)

    # ===== DIAGNOSIS TEXT =====
    diagnosis = f"""
    <b>Detected Condition:</b> {label}<br/><br/>
    NeuroSENSE analyzed microwave scattering signatures of cranial
    tissues using deep learning–based electromagnetic pattern
    recognition. The measured response characteristics show strongest
    correlation with <b>{label}</b>.
    """

    story.append(Spacer(1,12))
    story.append(Paragraph(diagnosis, body))

    # ===== GRAPH 1 =====
    story.append(Paragraph(
        "<b>Microwave Signature Analysis</b>", body))
    story.append(Image(sig_img, width=520, height=210))

    story.append(Spacer(1,12))

    # ===== GRAPH 2 =====
    story.append(Paragraph(
        "<b>AI Probability Distribution</b>", body))
    story.append(Image(prob_img, width=520, height=210))

    # ===== FOOTER =====
    footer = """
    AI-assisted screening output intended for research and preliminary
    diagnostic support. This system does not replace CT/MRI evaluation.
    Clinical confirmation by a certified medical professional is required.
    """

    story.append(Spacer(1,14))
    story.append(Paragraph(footer, body))

    doc.build(story)

    return pdf_path

# UI FLOW

st.markdown("### Step 1 — Acquire Microwave Sweep")

if st.button("📡 Open NanoVNA Saver"):
    if launch_nanovna():
        st.success("NanoVNA Saver launched. Opening Nanovna Saver.......")

st.markdown("---")

st.markdown("### Step 2 — Run AI Diagnosis")

if st.button("🔍 Run Diagnosis"):
    
    file = get_latest_s1p()

    if file is None:
        st.error("No sweep found.")
        st.stop()

    #SHOWS FETCHED FILE NAME
    st.info(f"Processing file: {file.name}")

    signal=read_s1p(file)

    if signal is None:
        st.error("Invalid S1P data.")
        st.stop()

    pred,probs=predict(signal)

    label=CLASS_MAP[pred]
    confidence=probs[pred]*100

    if label=="Normal":
        st.success(f"✅ Brain Status: {label}")
    else:
        st.error(f"⚠ Detected Condition: {label}")

    st.metric("Confidence",f"{confidence:.2f}%")

    st.line_chart(signal)
    st.bar_chart(probs)

    pdf_path=generate_pdf(label,confidence,probs,signal)

    st.success("Medical report generated.")

    st.download_button(
        "Download Medical Report",
        open(pdf_path,"rb"),
        file_name=os.path.basename(pdf_path)
    )

else:
    st.info("Open NanoVNA Saver → Acquire sweep → Run Diagnosis")