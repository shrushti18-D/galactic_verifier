# 🚀 Galactic 3D Company Relevance Analyzer & Business Intelligence Engine

A complete B2B business intelligence and lead qualification engine. Simply provide company names, categories, and service keywords, and the system automatically cleans data, evaluates manufacturing capability relevance, and classifies target businesses into **GOOD**, **MODERATE**, and **BAD** leads with downloadable Excel reports.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)
![PyMuPDF](https://img.shields.io/badge/PyMuPDF-PDF_Extraction-orange.svg)
![Offline](https://img.shields.io/badge/Offline--First-Enabled-green.svg)

---

## 🏬 Executive Summary & Business Intelligence Overview

This application acts as a comprehensive **Company Relevance & Capability Analyzer**. It gives sales and business development teams complete visibility into any company by analyzing their service keywords and industry categories against target manufacturing capabilities:

- 🔍 **Input**: Upload company spreadsheets or provide company keywords (`Company Name`, `Industry Category`, `Services/Keywords`, `City/Location`).
- ⚡ **Automated Intelligence**: Automatically cleans duplicates, normalizes company profiles, and extracts manufacturing capabilities from PDF brochures using PyMuPDF.
- 🎯 **Classification Output**: Evaluates relevance scores (0 to 100) and classifies leads into **GOOD (≥75)**, **MODERATE**, and **BAD** leads.
- 📥 **Export**: Download qualified lead lists filtered by location with clean original Excel columns.

---

### 🎯 Supported 8 Target Industry Verticals & Keywords
1. **Automotive**: Automobile, OEM, auto parts, autoparts, engine components, transmission, suspension, pistons, shock absorbers, tyres/tubes, auto accessories.
2. **Aerospace**: Aircraft, aviation, turbine blades, jet engines, rocket components, satellite hardware, flight MRO.
3. **Medical**: MedTech, medical devices, surgical instruments, orthopedic tools, dental guides.
4. **Defense**: Military, naval, missile components, propulsion systems, UAV, defense R&D.
5. **Injection Moulding**: Mould tooling, mold inserts, conformal cooling, rapid tooling.
6. **3D Printing & Additive**: Metal AM, DMLS, SLM, LPBF, DfAM, generative design, topology optimization.
7. **Semiconductor**: Semiconductor equipment, wafer processing, gas delivery, vacuum components, heat exchangers.
8. **Oil & Gas**: Oilfield equipment, valves, pump components, impellers, manifolds, turbomachinery.

---

## ✨ Key Features

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
git clone https://github.com/shrushti18-D/galactic_verifier.git
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
