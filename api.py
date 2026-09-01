from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
import re
from urllib.parse import urlparse

from brochure_reader import read_pdf_bytes, extract_capabilities
from analyzer import analyze_companies_batch, INDUSTRY_SYNONYMS

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


# Authoritative Procurement & Vendor Intent Signals
VENDOR_INTENT_TERMS = [
    "vendor registration", "vendor list", "approved vendor", "approved suppliers",
    "supplier portal", "vendor portal", "issue rfq", "rfq", "rfp", "procurement",
    "subcontracting", "subcontractor", "vendor empanelment", "supplier onboarding",
    "seeking suppliers", "seeking vendors", "looking for vendors", "outsourcing",
    "tier 1 supplier", "tier 2 supplier", "tier-1 supplier", "vendor network",
    "supply chain partner", "vendor", "vendors", "supplier", "suppliers"
]

DIRECT_INHOUSE_TERMS = [
    "100% in-house", "captive unit", "single-site manufacturing", "zero third-party",
    "in-house tool room", "in-house manufacturing"
]


def detect_vendor_buyer_intent(text: str, url: str = "") -> tuple[bool, str, List[str]]:
    """
    Analyzes webpage content for active Vendor Buyer & Procurement Intent Signals.
    Returns (has_vendor_intent, vendor_status_text, matched_vendor_signals).
    """
    text_clean = text.lower()
    url_clean = url.lower()
    matched_signals = []

    # Check URL path signals (e.g. /vendors, /procurement, /rfq)
    for path_term in ["vendor", "supplier", "procurement", "rfq", "subcontract"]:
        if path_term in url_clean:
            matched_signals.append(f"URL Path ({path_term.title()})")

    # Check Text Procurement Terms
    for term in VENDOR_INTENT_TERMS:
        if re.search(r'\b' + re.escape(term) + r'\b', text_clean):
            matched_signals.append(term.title())

    # Deduplicate signals
    matched_signals = list(dict.fromkeys(matched_signals))

    has_intent = len(matched_signals) > 0

    if has_intent:
        vendor_status = "With Vendors (Active Buyer 🟢)"
    else:
        vendor_status = "Without Vendors (No Vendor Need 🔴)"

    return has_intent, vendor_status, matched_signals


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
    Performs Vendor Buyer Intent Segregation & ML Capability Match Scoring.
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

    # Run Vendor Buyer Intent Detection
    has_vendor_intent, vendor_status, vendor_signals = detect_vendor_buyer_intent(combined_text, raw_url)

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
    raw_result_class = str(analyzed_df["Result"].iloc[0])
    raw_reason_str = str(analyzed_df["Reason"].iloc[0])

    # Apply Vendor Intent Weighting Matrix
    if has_vendor_intent:
        # Company actively hires vendors: Boost score by +15 points (capped at 100)
        match_score = min(round(raw_match_score + 15.0, 1), 100.0)
    else:
        # Company has NO vendor buying intent: Hard-cap score at max 40 (BAD MATCH)
        match_score = min(raw_match_score, 40.0)

    # Recalculate classification result
    if match_score >= 75.0:
        result_class = "GOOD"
    elif match_score >= 45.0:
        result_class = "MODERATE"
    else:
        result_class = "BAD"

    # Construct enhanced explanation reason with Vendor Segregation info
    vendor_reason_part = f"Vendor Status: {vendor_status}"
    if vendor_signals:
        vendor_reason_part += f" (Signals: {', '.join(vendor_signals[:3])})"

    reason_str = f"{vendor_reason_part} | {raw_reason_str}"

    # Extract matched capabilities/keywords list for visual chips in Chrome extension
    matched_capabilities: List[str] = []
    category_matched = "General Business / Web App"

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
        "vendor_signals": vendor_signals,
        "category": category_matched,
        "matched_capabilities": matched_capabilities,
        "reason": reason_str
    }