"""
analyzer.py
Ultra-Fast Offline-First Machine Learning Analysis & Relevance Scoring Engine for Galactic 3D.

Uses SentenceTransformers (all-MiniLM-L6-v2) with automatic offline fallback to Scikit-Learn TF-IDF,
vectorized matrix operations, phrase-aware keyword matching, and complete combined Galactic 3D taxonomy.
"""

import os
import re
import difflib
from typing import List, Dict, Tuple, Optional, Set
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
        "3D printing", "3d printing services", "industrial 3d printing", "custom 3d printing",
        "additive manufacturing", "metal additive manufacturing", "polymer 3d printing",
        "metal 3D printing", "metal AM", "DMLS", "dmls printing", "SLM", "slm 3d printing",
        "LPBF", "laser powder bed fusion", "direct metal laser sintering", "selective laser melting",
        "SLS", "sls 3d printing", "SLA", "sla 3d printing", "FDM", "fdm 3d printing",
        "rapid prototyping", "DfAM", "design for additive manufacturing", "generative design",
        "topology optimization", "lightweighting", "part consolidation", "3d scanning",
        "3d scanning services", "cad design", "reverse engineering"
    ],

    "cnc machining": [
        "cnc", "cnc machining", "cnc milling", "cnc turning", "precision machining",
        "precision components", "wire edm", "lathe", "tool room", "tooling", "tooling solutions",
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


def evaluate_single_keyword_overlap(
    kw_str: str,
    galactic_keywords: List[str],
    galactic_patterns: List[Tuple[str, re.Pattern]]
) -> Tuple[float, List[str]]:
    """
    Evaluates keyword overlap for a single keyword string against Galactic target terms.
    """
    if not kw_str or kw_str.strip() == "":
        return 0.0, []

    kw_clean = kw_str.lower()
    matched_words = []

    # 1. Exact & Regex Pattern Match
    for orig_kw, pattern in galactic_patterns:
        if pattern.search(kw_clean) or orig_kw.lower() in kw_clean:
            matched_words.append(orig_kw)

    # 2. Industry & Material Synonym Match
    for main_domain, synonyms in INDUSTRY_SYNONYMS.items():
        for syn in synonyms:
            if syn in kw_clean:
                matched_words.append(main_domain.title())
                break

    # Deduplicate preserving order
    matched_words = list(dict.fromkeys(matched_words))

    count = len(matched_words)
    if count >= 3:
        score = 100.0
    elif count == 2:
        score = 80.0
    elif count == 1:
        score = 60.0
    else:
        score = 0.0

    return score, matched_words


def fast_phrase_keyword_scoring(
    company_keywords: List[str],
    galactic_keywords: List[str]
) -> Tuple[np.ndarray, List[List[str]]]:
    """
    Ultra-fast phrase-aware keyword matching across all rows.
    """
    num_rows = len(company_keywords)
    kw_scores = np.zeros(num_rows)
    matched_kws_all = [[] for _ in range(num_rows)]

    if not galactic_keywords or num_rows == 0:
        return kw_scores, matched_kws_all

    galactic_patterns = []
    for g_kw in galactic_keywords:
        pattern = re.compile(r'\b' + re.escape(g_kw.lower()) + r'\b', re.IGNORECASE)
        galactic_patterns.append((g_kw, pattern))

    kw_cache: Dict[str, Tuple[float, List[str]]] = {}

    for i in range(num_rows):
        raw_kw = company_keywords[i]
        if raw_kw not in kw_cache:
            kw_cache[raw_kw] = evaluate_single_keyword_overlap(raw_kw, galactic_keywords, galactic_patterns)

        score, matched = kw_cache[raw_kw]
        kw_scores[i] = score
        matched_kws_all[i] = matched

    return kw_scores, matched_kws_all


def evaluate_single_category_match(
    cat_str: str,
    galactic_keywords: List[str]
) -> Tuple[float, Optional[str]]:
    """
    Evaluates industry category string against Galactic target vertical taxonomy.
    """
    if not cat_str or cat_str.strip() == "":
        return 0.0, None

    cat_clean = cat_str.lower().strip()
    galactic_lower = [k.lower() for k in galactic_keywords]

    # 1. Industry Synonym Match
    for main_domain, synonyms in INDUSTRY_SYNONYMS.items():
        for syn in synonyms:
            if syn in cat_clean:
                return 100.0, main_domain.title()

    # 2. Direct Substring Match
    for idx, g_kw in enumerate(galactic_lower):
        if g_kw in cat_clean or cat_clean in g_kw:
            return 100.0, galactic_keywords[idx]

    # 3. Fuzzy Check Fallback
    best_ratio = 0.0
    best_term = None
    for idx, g_kw in enumerate(galactic_lower):
        ratio = difflib.SequenceMatcher(None, cat_clean, g_kw).ratio() * 100.0
        if ratio > best_ratio:
            best_ratio = ratio
            best_term = galactic_keywords[idx]

    if best_ratio >= 60:
        return best_ratio, best_term
    elif best_ratio >= 40:
        return best_ratio * 0.7, best_term
    else:
        return max(best_ratio * 0.3, 0.0), None


def fast_vectorized_category_scoring(
    company_categories: List[str],
    galactic_keywords: List[str]
) -> Tuple[np.ndarray, List[Optional[str]]]:
    """
    Vectorized category matching against Galactic capability taxonomy.
    """
    num_rows = len(company_categories)
    cat_scores = np.zeros(num_rows)
    matched_cats = [None] * num_rows

    cat_cache: Dict[str, Tuple[float, Optional[str]]] = {}

    for i in range(num_rows):
        cat_raw = company_categories[i]
        if cat_raw not in cat_cache:
            cat_cache[cat_raw] = evaluate_single_category_match(cat_raw, galactic_keywords)

        score, term = cat_cache[cat_raw]
        cat_scores[i] = score
        matched_cats[i] = term

    return cat_scores, matched_cats


def generate_reason(
    matched_keywords: List[str],
    category_match: Optional[str],
    similarity_score: float,
    final_score: float
) -> str:
    """
    Constructs a clear, human-readable reason string for classification auditing.
    """
    reasons = []

    if matched_keywords:
        kw_str = ", ".join(matched_keywords[:4])
        if len(matched_keywords) > 4:
            kw_str += f" (+{len(matched_keywords)-4} more)"
        reasons.append(f"Matched keywords: {kw_str}")

    if category_match:
        reasons.append(f"Category matched ({category_match})")

    reasons.append(f"Semantic similarity: {similarity_score:.1f}%")

    if not matched_keywords and not category_match and final_score < 45:
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
    Ultra-Fast Machine Learning Batch Analysis Engine with Complete Combined Galactic Taxonomy.

    :param df: Cleaned company pandas DataFrame.
    :param column_mapping: Dictionary mapping canonical fields to DF columns.
    :param galactic_profile: Capability profile dict from brochure_reader.py.
    :param batch_size: Size of batch for vector encoding.
    :param progress_callback: Optional Streamlit progress callback function (current, total, stage).
    :return: DataFrame enriched with 'Match Score', 'Result', and 'Reason' columns.
    """
    if df.empty:
        df["Match Score"] = []
        df["Result"] = []
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

    co_list = df[co_col].astype(str).tolist() if co_col and co_col in df.columns else [""] * len(df)
    cat_list = df[cat_col].astype(str).tolist() if cat_col and cat_col in df.columns else [""] * len(df)
    kw_list = df[kw_col].astype(str).tolist() if kw_col and kw_col in df.columns else [""] * len(df)

    company_descriptions: List[str] = []
    for i in range(num_rows):
        co_name = co_list[i]
        category = cat_list[i]
        keywords = kw_list[i]

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

    if progress_callback:
        progress_callback(30, f"Phase 1/2: Vectorizing {num_rows:,} company profiles...")

    similarity_scores = compute_vector_similarities_offline(company_descriptions, profile_text, model)

    if progress_callback:
        progress_callback(75, "Phase 2/2: Computing Matrix Similarities & Hybrid Scores...")

    cat_scores, matched_cats = fast_vectorized_category_scoring(cat_list, galactic_keywords)
    kw_scores, matched_kws_all = fast_phrase_keyword_scoring(kw_list, galactic_keywords)

    if progress_callback:
        progress_callback(90, "Finalizing Classifications (GOOD / MODERATE / BAD)...")

    # Final Weighted Score Calculation (40% Category, 40% Keywords, 20% Similarity)
    final_scores = (0.40 * cat_scores) + (0.40 * kw_scores) + (0.20 * similarity_scores)
    final_scores = np.round(np.clip(final_scores, 0.0, 100.0), 1)

    # Classifications: GOOD >= 75, MODERATE >= 45, BAD < 45
    results = np.where(final_scores >= 75.0, "GOOD", np.where(final_scores >= 45.0, "MODERATE", "BAD")).tolist()

    reasons = [
        generate_reason(matched_kws_all[i], matched_cats[i], similarity_scores[i], final_scores[i])
        for i in range(num_rows)
    ]

    if progress_callback:
        progress_callback(100, f"Analysis Complete for {num_rows:,} companies!")

    df_result = df.copy()
    df_result["Match Score"] = final_scores.tolist()
    df_result["Result"] = results
    df_result["Reason"] = reasons

    return df_result
