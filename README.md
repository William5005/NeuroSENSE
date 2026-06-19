# NeuroSENSE

*A Unified Deep Learning-Driven Microwave System for Detection of Neurological Disorders*

---

## Overview

NeuroSENSE is a non-invasive, low-cost microwave imaging system that leverages deep learning to detect and classify neurological disorders from NanoVNA S21 signal measurements. The system is designed to provide portable and efficient diagnosis for conditions such as brain hemorrhage, ischemic stroke, and tumors.

---

## Features

* Non-invasive neurological disorder detection
* NanoVNA-based microwave signal acquisition
* Deep learning-driven classification
* 1D CNN architecture
* Real-time inference support
* Streamlit-based user interface
* Automated evaluation and reporting

---

## Disorders Detected

* Normal Brain
* Hemorrhage
* Ischemia
* Brain Tumor

---

## System Workflow

```text
NanoVNA
    ↓
S21 Signal Acquisition
    ↓
Preprocessing
    ↓
Feature Extraction
    ↓
1D Convolutional Neural Network
    ↓
Classification
```

---

## Repository Structure

```text
NeuroSENSE/
│
├── datasets/
│   ├── nanovna_domain/
│   └── new/
│
├── images/
├── live_sweeps/
├── model/
├── output/
├── results/
├── src/
│
├── calibration.cal
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## Technologies Used

* Python
* PyTorch
* Streamlit
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* ReportLab

---

## Dataset Information

* Frequency Range: **1–2 GHz**
* Sweep Points: **101**
* Signal Parameter: **S21 Log Magnitude**
* Classes:

  * Normal
  * Hemorrhage
  * Ischemia
  * Tumor

---

## Model Architecture

The classification model employs a one-dimensional convolutional neural network (1D CNN) optimized for microwave signal analysis. The network learns discriminative features from S21 signatures and predicts the corresponding neurological condition.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/NeuroSENSE.git
cd NeuroSENSE
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

### Windows

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Application

```bash
streamlit run src/app.py
```

---

## Results

NeuroSENSE demonstrates strong classification performance on NanoVNA-domain datasets and provides a scalable framework for microwave-based neurological disorder detection.

---

## Future Improvements

* Multi-disorder classification enhancement
* Robustness testing
* Explainable AI integration
* Real-time embedded deployment
* Expanded clinical datasets

---

## Author

**William**

B.Tech Computer Science and Engineering

---

## License

This project is licensed under the MIT License.