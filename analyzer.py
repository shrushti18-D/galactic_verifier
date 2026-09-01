"""
analyzer.py
Ultra-Fast Offline-First Machine Learning Analysis & Relevance Scoring Engine for Galactic 3D.

Uses SentenceTransformers (all-MiniLM-L6-v2) with automatic offline fallback to Scikit-Learn TF-IDF,
vectorized matrix operations, phrase-aware keyword matching, and complete combined Galactic 3D taxonomy.
"""

import os
import re
import difflib
from typing import List, Dict, Tuple, Optional, Set, Any
import numpy as np
import pandas as pd
import streamlit as st
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Optimize PyTorch CPU threading for maximum core utilization
try:
    torch.set_num_threads(min(8, os.cpu_count() or 4))
except Exception:
    pass

# Complete Combined Authoritative Galactic 3D Industry & Component Taxonomy Dictionary
INDUSTRY_SYNONYMS = {
    "automotive": [
        "automotive", "automobile", "automotive manufacturing", "automotive OEM",
        "automotive parts", "automotive components", "auto parts", "autoparts",
        "autopart", "auto component", "auto components", "car parts", "spare parts",
        "engine parts", "engine components", "transmission parts", "brake components",
        "manifolds", "shafts", "gears", "valves", "suspension parts", "precision components",
        "piston", "piston rings", "shock absorber", "tyres", "tires", "tubes",
        "auto accessories", "automobile accessories", "vehicle accessories",
        "automotive 3d printing"
    ],

    "aerospace": [
        "aerospace", "aircraft", "aviation", "aeronautical", "spacecraft", "avionics",
        "aerospace manufacturing", "aerospace components", "aircraft components",
        "aircraft parts", "aerospace engineering", "aerospace OEM", "turbine blades",
        "turbine components", "jet engine", "aircraft engine", "rocket components",
        "rocket engine", "propulsion", "propulsion components", "combustion chamber",
        "satellite components", "space hardware", "flight hardware", "aerospace MRO",
        "aerospace 3d printing"
    ],

    "medical": [
        "medical", "medtech", "medical devices", "surgical instruments", "surgical tools",
        "orthopedic instruments", "medical instruments", "dental instruments", "dental tools",
        "endoscopic instruments", "surgical guides", "medical device manufacturing",
        "medical 3d printing"
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
        "conformal cooling channels", "rapid tooling", "additive tooling",
        "plastic moulding", "cleanroom moulding", "molds", "dies", "engineering plastics"
    ],

    "3d printing": [
        "3D printing", "3d printing", "3d metal printing", "3D metal printing", "metal 3D printing",
        "metal 3d printing", "metal printing", "3d printing services", "industrial 3d printing",
        "custom 3d printing", "additive manufacturing", "metal additive manufacturing",
        "polymer 3d printing", "metal AM", "DMLS", "dmls printing", "SLM", "slm 3d printing",
        "LPBF", "laser powder bed fusion", "direct metal laser sintering", "selective laser melting",
        "SLS", "sls 3d printing", "SLA", "sla 3d printing", "FDM", "fdm 3d printing",
        "rapid prototyping", "DfAM", "design for additive manufacturing", "generative design",
        "topology optimization", "lightweighting", "part consolidation", "3d scanning",
        "3d scanning services", "cad design", "reverse engineering"
    ],

    "cnc machining": [
        "cnc", "cnc machining", "cnc milling", "cnc turning", "precision machining",
        "precision components", "precision", "wire edm", "lathe", "tool room", "tooling", "tooling solutions",
        "jigs & fixtures", "jig and fixture manufacturing", "die casting"
    ],

    "sheet metal": [
        "sheet metal", "sheet metal fabrication", "laser cutting", "bending", "metal stamping",
        "stamping", "welding", "fabrication", "metal enclosures"
    ],

    "materials": [
        "titanium", "titanium 3d printing", "stainless steel", "stainless steel 3d printing",
        "aluminum", "aluminum 3d printing", "inconel", "inconel 3d printing", "nylon",
        "carbon fiber", "engineering plastics"
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

# Authoritative Manufacturing Procurement & Buyer Intent Corpus for Contextual Scoring
PROCUREMENT_BUYER_CORPUS = """
seeking manufacturing suppliers approved vendor registration issuing RFQ for precision components
supplier portal onboarding vendor empanelment subcontracting CNC machining and 3D printing parts
purchasing industrial hardware procurement of aerospace components vendor list registration
become a supplier vendor management sourcing team supplier registration
"""

EXCLUDED_SELF_VENDOR_CORPUS = """
we are an IT vendor vendor of software services digital marketing agency legal services vendor
catering vendor cloud hosting provider software developers UI designers
"""


@st.cache_resource(show_spinner=False)
def load_sentence_transformer_model(model_name: str = "all-MiniLM-L6-v2"):
    """
    Loads and caches the SentenceTransformer model with offline fallback.
    If offline or HF Hub is unreachable, uses local cached files or returns None for TF-IDF fallback.
    """
    try:
        from sentence_transformers import SentenceTransformer
        try:
            return SentenceTransformer(model_name)
        except Exception:
            return SentenceTransformer(model_name, local_files_only=True)
    except Exception:
        return None


def compute_vector_similarities_offline(
    company_descriptions: List[str],
    profile_text: str,
    model=None
) -> np.ndarray:
    """
    Computes vector similarity scores (0-100).
    Uses SentenceTransformers if available, or fast Scikit-Learn TF-IDF vectorizer offline.
    """
    num_rows = len(company_descriptions)
    if num_rows == 0:
        return np.zeros(0)

    # Path A: SentenceTransformers Neural Model
    if model is not None:
        try:
            with torch.no_grad():
                profile_emb = model.encode(profile_text, convert_to_numpy=True, normalize_embeddings=True).reshape(1, -1)

                unique_texts = list(set(company_descriptions))
                unique_embs = model.encode(
                    unique_texts,
                    batch_size=128,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                emb_dict = {txt: emb for txt, emb in zip(unique_texts, unique_embs)}
                company_embs = np.array([emb_dict[txt] for txt in company_descriptions])

                cos_sims = cosine_similarity(company_embs, profile_emb).flatten()
                return np.clip(cos_sims * 100.0, 0.0, 100.0)
        except Exception:
            pass

    # Path B: 100% Offline Scikit-Learn TF-IDF Vectorizer Fallback (Zero Internet Needed)
    try:
        corpus = [profile_text] + company_descriptions
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=500, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(corpus)

        profile_vector = tfidf_matrix[0:1]
        company_vectors = tfidf_matrix[1:]

        sims = cosine_similarity(company_vectors, profile_vector).flatten()
        return np.clip(sims * 100.0, 0.0, 100.0)
    except Exception:
        return np.full(num_rows, 50.0)


def score_procurement_buyer_intent(text: str, url: str = "") -> Dict[str, Any]:
    """
    Contextual Procurement Buyer Intent Scoring using Scikit-Learn TF-IDF + Cosine Similarity.
    Differentiates between active buyers seeking vendors vs. self-referential IT/catering vendors.

    Returns a 3-state classification:
    - 'With Vendors (Active Buyer 🟢)'
    - 'Without Vendors (No Relevant Vendor Need 🔴)'
    - 'Uncertain / Insufficient Evidence 🟡'
    """
    text_lower = text.lower()
    url_lower = url.lower()

    # Fast affirmative in-house check for 'Without Vendors'
    inhouse_terms = ["100% in-house", "captive unit", "single-site manufacturing", "zero third-party outsourcing", "in-house tool room only"]
    if any(term in text_lower for term in inhouse_terms):
        return {
            "status": "Without Vendors (No Relevant Vendor Need 🔴)",
            "has_intent": False,
            "confidence": 0.90,
            "intent_score": 10.0,
            "signals": ["Affirmative In-House Manufacturing Only"],
            "reason": "Explicit evidence of 100% in-house manufacturing / zero external vendor requirement."
        }

    # Extract procurement signals
    procurement_terms = [
        "vendor registration", "vendor list", "approved vendor", "approved supplier",
        "supplier portal", "vendor portal", "issue rfq", "rfq", "rfp", "procurement",
        "subcontracting", "subcontractor", "vendor empanelment", "supplier onboarding",
        "seeking suppliers", "seeking vendors", "become a supplier", "sourcing",
        "purchasing", "supplier management"
    ]

    matched_signals = []
    for pterm in procurement_terms:
        if re.search(r'\b' + re.escape(pterm) + r'\b', text_lower):
            matched_signals.append(pterm.title())

    for path_term in ["vendor", "supplier", "procurement", "rfq", "subcontract", "sourcing"]:
        if path_term in url_lower:
            matched_signals.append(f"URL ({path_term.title()})")

    matched_signals = list(dict.fromkeys(matched_signals))

    # Scikit-Learn TF-IDF Contextual Similarity Check
    try:
        corpus = [PROCUREMENT_BUYER_CORPUS, EXCLUDED_SELF_VENDOR_CORPUS, text_lower]
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=300, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(corpus)

        buyer_vec = tfidf_matrix[0:1]
        self_vec = tfidf_matrix[1:2]
        doc_vec = tfidf_matrix[2:3]

        buyer_sim = float(cosine_similarity(doc_vec, buyer_vec)[0][0] * 100.0)
        self_sim = float(cosine_similarity(doc_vec, self_vec)[0][0] * 100.0)
    except Exception:
        buyer_sim = 20.0 if matched_signals else 0.0
        self_sim = 0.0

    # Decision Matrix
    is_self_referential = self_sim > (buyer_sim * 1.3) and "we are a vendor" in text_lower

    if matched_signals and buyer_sim >= 15.0 and not is_self_referential:
        return {
            "status": "With Vendors (Active Buyer 🟢)",
            "has_intent": True,
            "confidence": min(0.95, round(0.70 + (buyer_sim / 200.0), 2)),
            "intent_score": round(min(100.0, 60.0 + buyer_sim), 1),
            "signals": matched_signals,
            "reason": f"Strong contextual evidence of supplier/procurement activity (Signals: {', '.join(matched_signals[:3])})."
        }
    elif is_self_referential:
        return {
            "status": "Uncertain / Insufficient Evidence 🟡",
            "has_intent": False,
            "confidence": 0.40,
            "intent_score": 30.0,
            "signals": matched_signals,
            "reason": "Text mentions vendor terminology self-referentially (e.g. IT/service provider), not as an active buyer."
        }
    else:
        # Default 3-state fallback when evidence is missing or ambiguous
        return {
            "status": "Uncertain / Insufficient Evidence 🟡",
            "has_intent": False,
            "confidence": 0.50,
            "intent_score": 40.0,
            "signals": [],
            "reason": "No sufficient procurement portal or supplier onboarding evidence found on scanned pages."
        }


def evaluate_single_keyword_overlap(
    kw_str: str,
    target_keywords: List[str]
) -> Tuple[float, List[str]]:
    """
    Evaluates phrase-aware keyword overlap.
    """
    if not kw_str or not target_keywords:
        return 0.0, []

    kw_clean = str(kw_str).lower()
    matched = []

    # 1. Check exact taxonomy matches
    for main_domain, synonyms in INDUSTRY_SYNONYMS.items():
        for syn in synonyms:
            syn_lower = syn.lower()
            if re.search(r'\b' + re.escape(syn_lower) + r'\b', kw_clean):
                matched.append(syn.title())

    # 2. Check target brochure keywords
    for target in target_keywords:
        target_lower = str(target).lower()
        if len(target_lower) > 2 and re.search(r'\b' + re.escape(target_lower) + r'\b', kw_clean):
            matched.append(target.title())

    unique_matched = list(dict.fromkeys(matched))

    if not unique_matched:
        return 0.0, []

    score = min(100.0, len(unique_matched) * 35.0)
    return score, unique_matched


def fast_vectorized_category_scoring(
    categories: List[str],
    galactic_keywords: List[str]
) -> Tuple[np.ndarray, List[List[str]]]:
    """
    Fast vectorized category scoring.
    """
    num_rows = len(categories)
    cat_scores = np.zeros(num_rows)
    matched_categories: List[List[str]] = [[] for _ in range(num_rows)]

    target_cats = set(INDUSTRY_SYNONYMS.keys())
    for kw in galactic_keywords:
        kw_lower = kw.lower()
        if kw_lower in INDUSTRY_SYNONYMS:
            target_cats.add(kw_lower)

    for i, cat_text in enumerate(categories):
        if not cat_text:
            continue
        cat_lower = str(cat_text).lower()
        matched = []

        for target_cat in target_cats:
            synonyms = INDUSTRY_SYNONYMS.get(target_cat, [target_cat])
            for syn in synonyms:
                if re.search(r'\b' + re.escape(syn.lower()) + r'\b', cat_lower):
                    matched.append(target_cat.title())
                    break

        unique_matched = list(dict.fromkeys(matched))
        matched_categories[i] = unique_matched

        if unique_matched:
            cat_scores[i] = min(100.0, len(unique_matched) * 45.0)

    return cat_scores, matched_categories


def fast_phrase_keyword_scoring(
    keywords_list: List[str],
    galactic_keywords: List[str]
) -> Tuple[np.ndarray, List[List[str]]]:
    """
    Fast phrase keyword scoring.
    """
    num_rows = len(keywords_list)
    kw_scores = np.zeros(num_rows)
    matched_kws: List[List[str]] = [[] for _ in range(num_rows)]

    for i, kw_str in enumerate(keywords_list):
        score, matched = evaluate_single_keyword_overlap(kw_str, galactic_keywords)
        kw_scores[i] = score
        matched_kws[i] = matched

    return kw_scores, matched_kws


def generate_reason(
    matched_keywords: List[str],
    matched_categories: List[str],
    similarity_score: float,
    final_score: float
) -> str:
    """
    Generates human-readable classification reason.
    """
    reasons = []

    if matched_keywords:
        top_kws = matched_keywords[:4]
        more_count = len(matched_keywords) - 4
        kw_str = ", ".join(top_kws)
        if more_count > 0:
            kw_str += f" (+{more_count} more)"
        reasons.append(f"Matched keywords: {kw_str}")

    if matched_categories:
        reasons.append(f"Category matched ({', '.join(matched_categories)})")

    reasons.append(f"Semantic similarity: {similarity_score:.1f}%")

    if not matched_keywords and not matched_categories and final_score < 45:
        return "No target vertical overlap found; low semantic relevance to Galactic 3D."

    return " | ".join(reasons)


def analyze_companies_batch(
    df: pd.DataFrame,
    column_mapping: Dict[str, Optional[str]],
    galactic_profile: Dict[str, any],
    batch_size: int = 128,
    progress_callback=None
) -> pd.DataFrame:
    """
    Ultra-Fast Machine Learning Batch Analysis Engine with 3-State Vendor Classification.
    """
    if df.empty:
        df["Match Score"] = []
        df["Result"] = []
        df["Vendor Status"] = []
        df["Reason"] = []
        return df

    num_rows = len(df)

    if progress_callback:
        progress_callback(5, "Initializing Machine Learning Scoring Engine...")

    model = load_sentence_transformer_model("all-MiniLM-L6-v2")

    profile_text = galactic_profile.get("capability_summary", "")
    galactic_keywords = galactic_profile.get("keywords", [])

    co_col = column_mapping.get("co_name")
    cat_col = column_mapping.get("category")
    kw_col = column_mapping.get("keywords")
    web_col = column_mapping.get("website")

    co_list = df[co_col].astype(str).tolist() if co_col and co_col in df.columns else [""] * len(df)
    cat_list = df[cat_col].astype(str).tolist() if cat_col and cat_col in df.columns else [""] * len(df)
    kw_list = df[kw_col].astype(str).tolist() if kw_col and kw_col in df.columns else [""] * len(df)
    web_list = df[web_col].astype(str).tolist() if web_col and web_col in df.columns else [""] * len(df)

    company_descriptions: List[str] = []
    vendor_statuses: List[str] = []
    vendor_reasons: List[str] = []

    for i in range(num_rows):
        co_name = co_list[i]
        category = cat_list[i]
        keywords = kw_list[i]
        website = web_list[i]

        desc_parts = []
        if co_name:
            desc_parts.append(f"Company: {co_name}.")
        if category:
            desc_parts.append(f"Industry: {category}.")
        if keywords:
            desc_parts.append(f"Services: {keywords}.")

        text_snippet = " ".join(desc_parts)
        if not text_snippet.strip():
            text_snippet = "Company profile unknown."
        company_descriptions.append(text_snippet)

        # Scikit-Learn Contextual Procurement Intent Scoring
        proc_eval = score_procurement_buyer_intent(text_snippet, website)
        vendor_statuses.append(proc_eval["status"])
        vendor_reasons.append(proc_eval["reason"])

    if progress_callback:
        progress_callback(30, f"Phase 1/2: Vectorizing {num_rows:,} company profiles...")

    similarity_scores = compute_vector_similarities_offline(company_descriptions, profile_text, model)

    if progress_callback:
        progress_callback(75, "Phase 2/2: Computing Matrix Similarities & Hybrid Scores...")

    cat_scores, matched_cats = fast_vectorized_category_scoring(cat_list, galactic_keywords)
    kw_scores, matched_kws_all = fast_phrase_keyword_scoring(kw_list, galactic_keywords)

    if progress_callback:
        progress_callback(90, "Finalizing Classifications (GOOD / MODERATE / BAD)...")

    # Final Weighted Score Calculation
    final_scores = (0.40 * cat_scores) + (0.40 * kw_scores) + (0.20 * similarity_scores)

    # Adjust final scores based on Vendor Status
    for i in range(num_rows):
        status = vendor_statuses[i]
        if status == "With Vendors (Active Buyer 🟢)":
            final_scores[i] = min(100.0, final_scores[i] + 15.0)
        elif status == "Without Vendors (No Relevant Vendor Need 🔴)":
            final_scores[i] = min(35.0, final_scores[i])

    final_scores = np.round(np.clip(final_scores, 0.0, 100.0), 1)

    results = np.where(final_scores >= 75.0, "GOOD", np.where(final_scores >= 45.0, "MODERATE", "BAD")).tolist()

    reasons = [
        f"Vendor Status: {vendor_statuses[i]} | {generate_reason(matched_kws_all[i], matched_cats[i], similarity_scores[i], final_scores[i])}"
        for i in range(num_rows)
    ]

    if progress_callback:
        progress_callback(100, f"Analysis Complete for {num_rows:,} companies!")

    df_result = df.copy()
    df_result["Match Score"] = final_scores.tolist()
    df_result["Result"] = results
    df_result["Vendor Status"] = vendor_statuses
    df_result["Reason"] = reasons

    return df_result
