from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
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
    Live Company Webpage Analysis Endpoint (Mode 1).
    Reuses analyzer.py relevance classification & match score logic.
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

    if len(words) < 5 and len(combined_text) < 25:
        return {
            "success": False,
            "message": "Not enough company information found on this page."
        }

    # Determine company display name
    company_name = company_name_input
    if not company_name:
        if title and not any(ext in title.lower() for ext in ["http", "www", "index"]):
            company_name = title.split("|")[0].split("-")[0].strip()
        if not company_name:
            company_name = domain if domain else "Target Business"

    # Construct inputs for analyzer.py ML engine
    category_text = f"{description} {title}".strip()
    keywords_text = f"{content} {description} {title}".strip()

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

    match_score = float(analyzed_df["Match Score"].iloc[0])
    result_class = str(analyzed_df["Result"].iloc[0])
    reason_str = str(analyzed_df["Reason"].iloc[0])

    # Extract matched capabilities/keywords list for visual chips in Chrome extension
    matched_capabilities: List[str] = []
    category_matched = "General Business"

    if "Matched keywords:" in reason_str:
        kw_part = reason_str.split("Matched keywords:")[1].split("|")[0].strip()
        if "(+" in kw_part:
            kw_part = kw_part.split("(+")[0].strip()
        matched_capabilities = [k.strip() for k in kw_part.split(",") if k.strip()]

    if "Category matched (" in reason_str:
        cat_part = reason_str.split("Category matched (")[1].split(")")[0].strip()
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
        "category": category_matched,
        "matched_capabilities": matched_capabilities,
        "reason": reason_str
    }