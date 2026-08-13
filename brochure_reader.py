"""
brochure_reader.py
PyMuPDF PDF Brochure Reader & Galactic 3D Capability Extractor.

Reads PDF brochures if uploaded, extracts page text using PyMuPDF, dynamically appends brochure keywords,
filters out noise/URL artifacts (like '3D Com Page', 'www', 'page'), and builds clean capability profile.
"""

import os
import re
from typing import List, Dict, Tuple, Set, Optional
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer

# Strict User-Provided 8-Vertical Master Taxonomy Dictionary
USER_CUSTOM_TAXONOMY = {
    "automotive": [
        "automotive", "automobile", "automotive manufacturing", "automotive OEM",
        "automotive parts", "automotive components", "engine parts", "engine components",
        "transmission parts", "brake components", "manifolds", "shafts", "gears", "valves",
        "suspension parts", "precision components", "spare parts"
    ],

    "aerospace": [
        "aerospace", "aircraft", "aviation", "aeronautical", "spacecraft", "avionics",
        "aerospace manufacturing", "aerospace components", "aircraft components",
        "aircraft parts", "aerospace engineering", "aerospace OEM", "turbine blades",
        "turbine components", "jet engine", "aircraft engine", "rocket components",
        "rocket engine", "propulsion", "propulsion components", "combustion chamber",
        "satellite components", "space hardware", "flight hardware", "aerospace MRO"
    ],

    "medical": [
        "medical", "medtech", "medical devices", "surgical instruments", "surgical tools",
        "orthopedic instruments", "medical instruments", "dental instruments", "dental tools",
        "endoscopic instruments", "surgical guides", "medical device manufacturing"
    ],

    "defense": [
        "defense", "defence", "military", "naval", "defense manufacturing",
        "defense components", "military components", "military equipment",
        "defense OEM", "defense engineering", "naval engineering", "naval components",
        "missile components", "propulsion systems", "UAV", "UAS", "defense R&D"
    ],

    "injection moulding": [
        "injection moulding", "injection molding", "mould tooling", "mold tooling",
        "mould inserts", "mold inserts", "tooling inserts", "conformal cooling",
        "conformal cooling channels", "rapid tooling", "additive tooling"
    ],

    "3d printing": [
        "3D printing", "additive manufacturing", "metal additive manufacturing",
        "metal 3D printing", "industrial 3D printing", "metal AM", "DMLS", "SLM", "LPBF",
        "laser powder bed fusion", "direct metal laser sintering", "selective laser melting",
        "DfAM", "design for additive manufacturing", "generative design",
        "topology optimization", "lightweighting", "part consolidation"
    ],

    "semiconductor": [
        "semiconductor", "semiconductor manufacturing", "semiconductor equipment",
        "semiconductor components", "wafer processing equipment", "gas delivery components",
        "vacuum components", "heat exchangers", "thermal management", "cooling components"
    ],

    "oil and gas": [
        "oil and gas", "oil & gas", "oilfield equipment", "valves", "valve components",
        "pump components", "impellers", "manifolds", "heat exchangers", "burners",
        "nozzles", "turbomachinery"
    ]
}

# Flatten strict user keywords into master list
EXACT_GALACTIC_3D_KEYWORDS = sorted(list(set(
    [item for sublist in USER_CUSTOM_TAXONOMY.values() for item in sublist]
)))

# Noise filter set for brochure text artifacts (URLs, page headers, footers)
JUNK_NOISE_TERMS = {
    "3d com", "3d com page", "com page", "www galactic", "www galactic 3d",
    "galactic 3d com", "page", "com", "www", "http", "https", "galactic",
    "galactic 3d", "brochure", "email", "phone", "contact", "address", "website"
}


def is_valid_capability_term(term: str) -> bool:
    """Filters out junk header/footer PDF artifacts."""
    term_lower = term.lower().strip()
    if term_lower in JUNK_NOISE_TERMS:
        return False
    if any(junk in term_lower for junk in ["3d com", "com page", "www.", "http"]):
        return False
    if len(term_lower) <= 2 or term_lower.isnumeric():
        return False
    return True


def read_pdf_bytes(pdf_bytes: bytes, filename: str = "brochure.pdf") -> Dict[str, any]:
    """
    Parses a single PDF byte stream using PyMuPDF (fitz) and extracts text per page.
    """
    text_pages = []
    total_pages = 0
    full_text = ""

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(doc)
        for page_num in range(total_pages):
            page = doc.load_page(page_num)
            page_text = page.get_text("text")
            if page_text:
                text_pages.append(page_text.strip())
                full_text += f"\n--- Page {page_num + 1} ---\n" + page_text.strip()
        doc.close()
    except Exception as e:
        full_text = f"Error reading {filename}: {str(e)}"

    return {
        "filename": filename,
        "total_pages": total_pages,
        "pages": text_pages,
        "full_text": full_text.strip()
    }


def read_all_brochures(brochure_dir: str = "brochure") -> Dict[str, any]:
    """
    Reads all PDF files present in the specified directory using PyMuPDF.
    """
    combined_text = ""
    file_details = []
    total_pages = 0

    if os.path.exists(brochure_dir) and os.path.isdir(brochure_dir):
        pdf_files = [f for f in os.listdir(brochure_dir) if f.lower().endswith(".pdf")]
        for pdf_file in pdf_files:
            file_path = os.path.join(brochure_dir, pdf_file)
            try:
                with open(file_path, "rb") as f:
                    pdf_bytes = f.read()
                res = read_pdf_bytes(pdf_bytes, filename=pdf_file)
                file_details.append(res)
                total_pages += res["total_pages"]
                combined_text += f"\n\n=== FILE: {pdf_file} ===\n" + res["full_text"]
            except Exception as e:
                file_details.append({
                    "filename": pdf_file,
                    "total_pages": 0,
                    "pages": [],
                    "full_text": f"Failed to load: {str(e)}"
                })

    return {
        "total_files": len(file_details),
        "total_pages": total_pages,
        "combined_text": combined_text.strip(),
        "files": file_details
    }


def extract_capabilities(combined_text: str) -> Dict[str, any]:
    """
    Extracts capabilities strictly from user's master keyword list, and dynamically
    appends valid brochure terms if a PDF brochure is uploaded, filtering out page/URL junk.
    """
    text_lower = combined_text.lower() if combined_text else ""

    matched_terms: Set[str] = set()
    term_counts: Dict[str, int] = {}

    for term in EXACT_GALACTIC_3D_KEYWORDS:
        pattern = r'\b' + re.escape(term.lower()) + r'\b'
        matches = re.findall(pattern, text_lower)
        if matches:
            matched_terms.add(term)
            term_counts[term] = len(matches)

    # Dynamic TF-IDF extraction ONLY IF a brochure PDF is uploaded by the user
    pdf_discovered_keywords = []
    if len(text_lower.strip()) > 50:
        try:
            clean_doc = re.sub(r'[^a-zA-Z0-9\s]', ' ', text_lower)
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 3),
                max_features=40,
                stop_words='english',
                min_df=1
            )
            tfidf_matrix = vectorizer.fit_transform([clean_doc])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]

            top_indices = scores.argsort()[::-1]
            for idx in top_indices:
                kw = feature_names[idx].title()
                if is_valid_capability_term(kw):
                    pdf_discovered_keywords.append(kw)
                    if kw not in term_counts:
                        term_counts[kw] = int(scores[idx] * 100)
        except Exception:
            pass

    # Strictly combine exact user keywords and valid PDF terms
    raw_all = list(set(EXACT_GALACTIC_3D_KEYWORDS + list(matched_terms) + pdf_discovered_keywords))
    all_keywords = sorted([kw for kw in raw_all if is_valid_capability_term(kw)])

    sorted_terms = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)
    top_keywords = [item[0] for item in sorted_terms if is_valid_capability_term(item[0])][:20]
    if not top_keywords:
        top_keywords = all_keywords[:20]

    # Construct capability summary string strictly from user's 8 verticals
    summary_parts = [
        "Galactic 3D specializes in Additive Manufacturing, Metal & Industrial 3D Printing, DMLS, SLM, LPBF, Conformal Cooling, Injection Moulding Tooling, DfAM, Generative Design, and Topology Optimization.",
        "Target verticals include Automotive, Aerospace, Defense, Medical Devices, Semiconductor Equipment, Oil & Gas, Injection Moulding Tooling, and Additive Manufacturing.",
        "Core components & hardware: Turbine Blades, Jet Engines, Rocket Components, Satellite Components, Surgical Instruments, Military Equipment, Conformal Cooling Channels, Wafer Processing Equipment, Oilfield Impellers, Manifolds, Heat Exchangers, Valves, Engine Components, Gears, and Precision Components."
    ]
    capability_summary = " ".join(summary_parts)

    return {
        "keywords": all_keywords,
        "top_keywords": top_keywords,
        "capability_summary": capability_summary,
        "term_frequencies": term_counts
    }
