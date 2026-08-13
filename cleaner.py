"""
cleaner.py
Data Cleaning and Column Detection Module for Galactic 3D Relevance Analyzer.

Handles column mapping auto-detection, whitespace stripping, case normalization,
duplicate removal across multiple key fields, and dataset health reporting.
"""

import re
import difflib
from typing import Dict, List, Tuple, Optional
import pandas as pd


# Canonical column definitions required by the system
REQUIRED_COLUMNS = {
    "co_name": ["company name", "company", "co_name", "co name", "firm name", "organization", "name", "firm", "business name"],
    "category": ["category", "industry", "sector", "domain", "business type", "categories", "segment", "industry category"],
    "keywords": ["keywords", "keyword", "services", "products", "tags", "capabilities", "specialties", "offerings", "description"],
    "city": ["city", "location", "town", "place", "district", "address", "region", "state", "city name", "location name"],
    "website": ["website", "site", "url", "web", "web address", "domain", "homepage", "link"],
    "email": ["email", "e-mail", "mail", "contact email", "email address", "info email"]
}


def fuzzy_similarity_ratio(str1: str, str2: str) -> float:
    """Computes similarity ratio between 0 and 100 using difflib standard library."""
    if not str1 or not str2:
        return 0.0
    return difflib.SequenceMatcher(None, str1.lower().strip(), str2.lower().strip()).ratio() * 100.0


def detect_column_mapping(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Automatically detects and maps raw DataFrame columns to canonical standard fields.

    :param df: Input pandas DataFrame
    :return: Dictionary mapping canonical key ('co_name', 'category', 'city', etc.) to matching raw column name (or None).
    """
    mapping: Dict[str, Optional[str]] = {}
    raw_columns = list(df.columns)
    normalized_raw = [str(col).strip().lower().replace("_", " ").replace("-", " ") for col in raw_columns]

    for canonical_key, synonyms in REQUIRED_COLUMNS.items():
        found_col = None
        # 1. Exact match check
        for synonym in synonyms:
            if synonym in normalized_raw:
                idx = normalized_raw.index(synonym)
                found_col = raw_columns[idx]
                break

        # 2. Substring/partial match check if exact match fails
        if not found_col:
            for synonym in synonyms:
                for idx, norm_col in enumerate(normalized_raw):
                    if synonym in norm_col or norm_col in synonym:
                        found_col = raw_columns[idx]
                        break
                if found_col:
                    break

        # 3. Fuzzy match fallback using difflib
        if not found_col:
            best_score = 0.0
            best_col = None
            for raw_col, norm_col in zip(raw_columns, normalized_raw):
                for synonym in synonyms:
                    score = fuzzy_similarity_ratio(synonym, norm_col)
                    if score > best_score:
                        best_score = score
                        best_col = raw_col
            if best_score >= 60.0:
                found_col = best_col

        mapping[canonical_key] = found_col

    return mapping


def clean_text_field(val: any) -> str:
    """
    Trims extra spaces, normalizes text, removes unprintable characters.

    :param val: Input value (string, float, NaN, etc.)
    :return: Cleaned normalized string.
    """
    if pd.isna(val) or val is None:
        return ""
    text = str(val)
    # Remove unprintable/control characters
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    # Collapse multiple whitespaces/newlines into single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def normalize_url(url: str) -> str:
    """
    Normalizes website URLs for deduplication (removes protocol, www, trailing slashes).
    """
    if not url or pd.isna(url):
        return ""
    u = str(url).lower().strip()
    u = re.sub(r'^https?://', '', u)
    u = re.sub(r'^www\.', '', u)
    u = u.split('/')[0]
    return u.strip()


def clean_dataset(
    df: pd.DataFrame,
    column_mapping: Dict[str, Optional[str]]
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Cleans company dataset, applies text normalization, and removes duplicate entries.

    :param df: Input raw pandas DataFrame.
    :param column_mapping: Dictionary mapping canonical keys to actual raw column names.
    :return: Tuple of (Cleaned DataFrame, Audit Statistics Dictionary).
    """
    if df.empty:
        return df, {
            "original_count": 0,
            "cleaned_count": 0,
            "total_removed": 0,
            "exact_duplicates": 0,
            "duplicate_companies": 0,
            "duplicate_websites": 0,
            "duplicate_emails": 0,
            "missing_values": {}
        }

    original_count = len(df)
    df_clean = df.copy()

    # Clean text across all string columns
    for col in df_clean.columns:
        if df_clean[col].dtype == object or df_clean[col].dtype == 'string':
            df_clean[col] = df_clean[col].apply(clean_text_field)

    # Calculate missing values count per column
    missing_overview = {str(col): int(df_clean[col].isna().sum() + (df_clean[col] == "").sum()) for col in df_clean.columns}

    # Step 1: Remove exact duplicate rows across all columns
    initial_rows = len(df_clean)
    df_clean.drop_duplicates(inplace=True)
    exact_duplicates_removed = initial_rows - len(df_clean)

    # Step 2: Remove duplicate company names (case-insensitive)
    co_col = column_mapping.get("co_name")
    dup_co_removed = 0
    if co_col and co_col in df_clean.columns:
        rows_before = len(df_clean)
        # Filter out empty names from duplicate check
        non_empty_mask = df_clean[co_col].str.strip() != ""
        empty_df = df_clean[~non_empty_mask]
        valid_df = df_clean[non_empty_mask]

        valid_df = valid_df.drop_duplicates(subset=[co_col], keep="first")
        df_clean = pd.concat([valid_df, empty_df], ignore_index=True)
        dup_co_removed = rows_before - len(df_clean)

    # Step 3: Remove duplicate websites if website column exists
    web_col = column_mapping.get("website")
    dup_web_removed = 0
    if web_col and web_col in df_clean.columns:
        rows_before = len(df_clean)
        df_clean["_norm_web"] = df_clean[web_col].apply(normalize_url)
        valid_web_mask = df_clean["_norm_web"] != ""

        valid_web_df = df_clean[valid_web_mask].drop_duplicates(subset=["_norm_web"], keep="first")
        empty_web_df = df_clean[~valid_web_mask]

        df_clean = pd.concat([valid_web_df, empty_web_df], ignore_index=True)
        df_clean.drop(columns=["_norm_web"], inplace=True)
        dup_web_removed = rows_before - len(df_clean)

    # Step 4: Remove duplicate emails if email column exists
    email_col = column_mapping.get("email")
    dup_email_removed = 0
    if email_col and email_col in df_clean.columns:
        rows_before = len(df_clean)
        df_clean["_norm_email"] = df_clean[email_col].astype(str).str.lower().str.strip()
        valid_email_mask = (df_clean["_norm_email"] != "") & (df_clean["_norm_email"] != "nan")

        valid_email_df = df_clean[valid_email_mask].drop_duplicates(subset=["_norm_email"], keep="first")
        empty_email_df = df_clean[~valid_email_mask]

        df_clean = pd.concat([valid_email_df, empty_email_df], ignore_index=True)
        df_clean.drop(columns=["_norm_email"], inplace=True)
        dup_email_removed = rows_before - len(df_clean)

    cleaned_count = len(df_clean)
    total_removed = original_count - cleaned_count

    stats = {
        "original_count": original_count,
        "cleaned_count": cleaned_count,
        "total_removed": total_removed,
        "exact_duplicates": exact_duplicates_removed,
        "duplicate_companies": dup_co_removed,
        "duplicate_websites": dup_web_removed,
        "duplicate_emails": dup_email_removed,
        "missing_values": missing_overview
    }

    return df_clean, stats
