# 🚀 Galactic 3D Company Relevance Analyzer

AI-Powered B2B Lead Classification Engine & Manufacturing Capability Matching Web Application.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)
![PyMuPDF](https://img.shields.io/badge/PyMuPDF-PDF_Extraction-orange.svg)
![Offline](https://img.shields.io/badge/Offline--First-Enabled-green.svg)

---

## 🏬 Overview

**Galactic 3D Relevance Analyzer** is an end-to-end B2B sales lead qualification platform designed for Galactic 3D's manufacturing capabilities. It cleans company datasets, extracts manufacturing skills from PDF brochures using PyMuPDF, evaluates company relevance across 8 target industry verticals using SentenceTransformers & TF-IDF vector similarity, and exports clean, qualified Excel reports.

---

### 🎯 Supported 8 Target Verticals
1. **Automotive**: Automobile, OEM, auto parts, engine components, transmission, suspension, pistons, shock absorbers, tyres/tubes, auto accessories.
2. **Aerospace**: Aircraft, aviation, turbine blades, jet engines, rocket components, satellite hardware, flight MRO.
3. **Medical**: MedTech, medical devices, surgical instruments, orthopedic tools, dental guides.
4. **Defense**: Military, naval, missile components, propulsion systems, UAV, defense R&D.
5. **Injection Moulding**: Mould tooling, mold inserts, conformal cooling, rapid tooling.
6. **3D Printing & Additive**: Metal AM, DMLS, SLM, LPBF, DfAM, generative design, topology optimization.
7. **Semiconductor**: Semiconductor equipment, wafer processing, gas delivery, vacuum components, heat exchangers.
8. **Oil & Gas**: Oilfield equipment, valves, pump components, impellers, manifolds, turbomachinery.

---

## ✨ Features

- 🧹 **Data Cleaning & Deduplication**: Auto-detects columns, normalizes text, drops duplicate companies, emails, websites, and exact duplicate rows.
- 📄 **PyMuPDF Brochure Extraction**: Reads brochure PDFs and extracts core manufacturing capabilities page-by-page.
- 🤖 **Hybrid ML Relevance Engine**: Combines **40% Category Overlap + 40% Keyword Overlap + 20% Vector Cosine Similarity**.
- 🛡️ **100% Offline-First Execution**: Automatic offline fallback to Scikit-Learn TF-IDF vectorizer when disconnected from internet.
- 🌆 **Location / City / State Filtering**: Filter metrics, charts, tables, and Excel downloads by specific locations (e.g. *Bengaluru, Mumbai, Pune, Delhi*).
- 📥 **Clean Excel Exporters**: Download qualified GOOD leads containing strictly original uploaded columns without audit score clutter.

---

## 🚀 Quick Start

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/galactic_verifier.git
cd galactic_verifier

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Application
```bash
python -m streamlit run app.py
```
Or double-click **`run_app.bat`** on Windows!

---

## 🛠️ Technology Stack
- **Framework**: Streamlit
- **Data Manipulation**: Pandas, NumPy
- **PDF Parser**: PyMuPDF (`fitz`)
- **Machine Learning / NLP**: SentenceTransformers (`all-MiniLM-L6-v2`), Scikit-Learn (TF-IDF & Cosine Similarity), PyTorch
- **Excel Generation**: OpenPyXL
- **Visualization**: Plotly Express & Graph Objects
