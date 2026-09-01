from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import pandas as pd
import re
from urllib.parse import urlparse

from brochure_reader import read_pdf_bytes, extract_capabilities
from analyzer import analyze_companies_batch, score_procurement_buyer_intent, INDUSTRY_SYNONYMS
from vendor_crawler import deep_verify_vendor_intent

app = FastAPI(title="Galactic Verifier API")

# Allow the Chrome extension to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CompanyAnalysisRequest(BaseModel):
    url: Optional[str] = ""
    company_name: Optional[str] = ""
    title: Optional[str] = ""
    description: Optional[str] = ""
    content: Optional[str] = ""


@app.get("/")
def home():
    return {
        "status": "online",
        "message": "Galactic Verifier API is running",
        "endpoints": ["/analyze", "/analyze-company"]
    }


@app.post("/analyze")
async def analyze_brochure(file: UploadFile = File(...)):
    """
    Brochure PDF Analysis Endpoint (Mode 2).
    Extracts text from uploaded PDF brochures and identifies Galactic 3D capabilities.
    """
    pdf_bytes = await file.read()

    pdf_data = read_pdf_bytes(
        pdf_bytes,
        filename=file.filename
    )

    if not pdf_data["full_text"]:
        return {
            "success": False,
            "message": "Could not extract text from the PDF."
        }

    result = extract_capabilities(
        pdf_data["full_text"]
    )

    return {
        "success": True,
        "filename": file.filename,
        "total_pages": pdf_data["total_pages"],
        "keywords": result["keywords"],
        "top_keywords": result["top_keywords"],
        "capability_summary": result["capability_summary"]
    }


@app.post("/analyze-company")
async def analyze_company(req: CompanyAnalysisRequest):
    """
    Live Company Webpage Analysis Endpoint for Chrome Extension (Mode 1).
    Performs Two-Stage Contextual Vendor Verification & ML Capability Match Scoring.
    """
    title = (req.title or "").strip()
    description = (req.description or "").strip()
    content = (req.content or "").strip()
    company_name_input = (req.company_name or "").strip()
    raw_url = (req.url or "").strip()

    # Domain fallback for company name
    domain = ""
    if raw_url:
        try:
            parsed = urlparse(raw_url)
            domain = parsed.netloc or parsed.path
            if domain.startswith("www."):
                domain = domain[4:]
        except Exception:
            domain = raw_url

    # Calculate total useful extracted text
    combined_text = f"{title} {description} {content}".strip()
    words = combined_text.split()

    # Determine company display name
    company_name = company_name_input
    if not company_name:
        if title and not any(ext in title.lower() for ext in ["http", "www", "index"]):
            company_name = title.split("|")[0].split("-")[0].strip()
        if not company_name:
            company_name = domain if domain else "Target Business"

    # Handle short single page app text by enriching with domain context
    if len(words) < 5:
        combined_text = f"{company_name} {domain} online web application software platform".strip()

    # Stage 1: Contextual Scikit-Learn TF-IDF Procurement Intent Scoring
    stage1_eval = score_procurement_buyer_intent(combined_text, raw_url)
    vendor_status = stage1_eval["status"]
    has_vendor_intent = stage1_eval["has_intent"]
    confidence = stage1_eval["confidence"]
    vendor_signals = stage1_eval["signals"]
    evidence_reason = stage1_eval["reason"]

    preserved_evidence: Dict[str, Any] = {
        "source_url": raw_url,
        "page_title": title[:100] or domain,
        "snippet": combined_text[:250],
        "matched_signals": vendor_signals,
        "relevance_score": stage1_eval["intent_score"],
        "confidence": confidence,
        "reason": evidence_reason
    }

    # Stage 2: Trigger Playwright Deep Verification ONLY when Stage 1 is Uncertain and a valid HTTP URL is present
    if vendor_status == "Uncertain / Insufficient Evidence 🟡" and raw_url.startswith(("http://", "https://")):
        try:
            stage2_evidence = deep_verify_vendor_intent(raw_url, timeout_ms=6000)
            if stage2_evidence.get("success") and stage2_evidence.get("has_vendor_intent"):
                vendor_status = "With Vendors (Active Buyer 🟢)"
                has_vendor_intent = True
                confidence = stage2_evidence.get("confidence", 0.85)
                vendor_signals = stage2_evidence.get("matched_signals", [])
                evidence_reason = stage2_evidence.get("reason", "Playwright deep scan verified procurement portal.")

                preserved_evidence = {
                    "source_url": stage2_evidence.get("source_url", raw_url),
                    "page_title": stage2_evidence.get("page_title", title[:100]),
                    "snippet": stage2_evidence.get("snippet", combined_text[:250]),
                    "matched_signals": vendor_signals,
                    "relevance_score": stage2_evidence.get("relevance_score", 85.0),
                    "confidence": confidence,
                    "reason": evidence_reason
                }
        except Exception as p_err:
            pass  # Graceful fallback to Stage 1 result

    # Construct inputs for analyzer.py ML engine
    category_text = f"{description} {title} {combined_text}".strip()
    keywords_text = f"{content} {description} {title} {combined_text}".strip()

    single_company_df = pd.DataFrame([{
        "co_name": company_name,
        "category": category_text,
        "keywords": keywords_text,
        "website": raw_url
    }])

    column_mapping = {
        "co_name": "co_name",
        "category": "category",
        "keywords": "keywords",
        "website": "website",
        "email": None
    }

    # Get Galactic 3D capability profile
    galactic_profile = extract_capabilities("")

    # Run exact Galactic Verifier relevance scoring engine from analyzer.py
    analyzed_df = analyze_companies_batch(
        df=single_company_df,
        column_mapping=column_mapping,
        galactic_profile=galactic_profile
    )

    if analyzed_df.empty:
        return {
            "success": False,
            "message": "Relevance analysis pipeline produced no result."
        }

    raw_match_score = float(analyzed_df["Match Score"].iloc[0])
    raw_reason_str = str(analyzed_df["Reason"].iloc[0])

    # Apply 3-State Vendor Intent Weighting Matrix
    if vendor_status == "With Vendors (Active Buyer 🟢)":
        match_score = min(round(raw_match_score + 15.0, 1), 100.0)
    elif vendor_status == "Without Vendors (No Relevant Vendor Need 🔴)":
        match_score = min(raw_match_score, 35.0)
    else:  # Uncertain / Insufficient Evidence 🟡
        match_score = raw_match_score

    # Recalculate classification result
    if match_score >= 75.0:
        result_class = "GOOD"
    elif match_score >= 45.0:
        result_class = "MODERATE"
    else:
        result_class = "BAD"

    reason_str = f"Vendor Status: {vendor_status} | {raw_reason_str}"

    # Extract matched capabilities/keywords list for visual chips in Chrome extension
    matched_capabilities: List[str] = []
    category_matched = "General Business"

    if "Matched keywords:" in raw_reason_str:
        kw_part = raw_reason_str.split("Matched keywords:")[1].split("|")[0].strip()
        if "(+" in kw_part:
            kw_part = kw_part.split("(+")[0].strip()
        matched_capabilities = [k.strip() for k in kw_part.split(",") if k.strip()]

    if "Category matched (" in raw_reason_str:
        cat_part = raw_reason_str.split("Category matched (")[1].split(")")[0].strip()
        category_matched = cat_part
    elif matched_capabilities:
        category_matched = matched_capabilities[0]

    # Fallback capability tags if semantic similarity scored well
    if not matched_capabilities and match_score >= 45:
        combined_lower = combined_text.lower()
        for main_domain, synonyms in INDUSTRY_SYNONYMS.items():
            for syn in synonyms:
                if syn in combined_lower:
                    matched_capabilities.append(main_domain.title())
                    break
        matched_capabilities = list(dict.fromkeys(matched_capabilities))

    return {
        "success": True,
        "company_name": company_name,
        "domain": domain,
        "url": raw_url,
        "result": result_class,
        "match_score": match_score,
        "vendor_status": vendor_status,
        "has_vendor_intent": has_vendor_intent,
        "vendor_confidence": confidence,
        "vendor_signals": vendor_signals,
        "vendor_evidence": preserved_evidence,
        "category": category_matched,
        "matched_capabilities": matched_capabilities,
        "reason": reason_str
    }