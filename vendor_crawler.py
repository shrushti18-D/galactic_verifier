"""
vendor_crawler.py
Stage 2 Deep Browser Verification Module using Playwright for Galactic Verifier.

Performs targeted deep website verification of procurement, supplier, and vendor pages
when Stage 1 fast analysis yields uncertain or ambiguous evidence.
"""

import asyncio
import re
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse, urljoin

# Target Procurement & Supplier Subpage Patterns
PROCUREMENT_PATH_PATTERNS = [
    r"procurement", r"supplier", r"vendor", r"sourcing",
    r"purchasing", r"subcontract", r"become-a-supplier",
    r"supplier-portal", r"vendor-registration", r"supplier-management"
]

PROCUREMENT_KEYWORDS = [
    "vendor registration", "vendor list", "approved vendor", "approved supplier",
    "supplier portal", "vendor portal", "issue rfq", "rfq", "rfp", "procurement",
    "subcontracting", "subcontractor", "vendor empanelment", "supplier onboarding",
    "seeking suppliers", "seeking vendors", "become a supplier", "sourcing",
    "purchasing", "supplier management", "tier 1 supplier", "tier 2 supplier"
]

EXCLUDED_SELF_VENDOR_PATTERNS = [
    r"we are a vendor", r"we are an IT vendor", r"vendor of software",
    r"vendor of choice", r"our vendor services"
]


async def deep_verify_vendor_intent_async(url: str, timeout_ms: int = 8000) -> Dict[str, Any]:
    """
    Asynchronously launches Playwright in headless Chromium mode to inspect target company
    procurement & supplier subpages.
    """
    if not url or not (url.startswith("http://") or url.startswith("https://")):
        return {
            "success": False,
            "has_vendor_intent": False,
            "reason": "Invalid or non-HTTP URL provided for Playwright crawling."
        }

    try:
        from playwright.async_api import async_playwright
    except ImportError:
        return {
            "success": False,
            "has_vendor_intent": False,
            "reason": "Playwright package not installed. Skipping Stage 2 deep browser verification."
        }

    evidence = {
        "success": False,
        "has_vendor_intent": False,
        "source_url": url,
        "page_title": "",
        "snippet": "",
        "matched_signals": [],
        "relevance_score": 0.0,
        "confidence": 0.0,
        "reason": ""
    }

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            page = await context.new_page()

            # Navigate to target homepage
            try:
                await page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
            except Exception as nav_err:
                await browser.close()
                evidence["reason"] = f"Playwright navigation timeout: {str(nav_err)[:100]}"
                return evidence

            # Scroll to trigger dynamic JS content loading
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2)")
            await asyncio.sleep(0.5)

            # Find potential procurement links in navigation or footer
            links = await page.locator("a[href]").all()
            target_link_url = None
            target_link_text = ""

            for link in links:
                try:
                    href = await link.get_attribute("href") or ""
                    text = (await link.inner_text() or "").strip()
                    full_href = urljoin(url, href)

                    # Check if href or link text matches procurement terms
                    href_lower = href.lower()
                    text_lower = text.lower()

                    for pat in PROCUREMENT_PATH_PATTERNS:
                        if re.search(pat, href_lower) or re.search(pat, text_lower):
                            target_link_url = full_href
                            target_link_text = text
                            break
                    if target_link_url:
                        break
                except Exception:
                    continue

            # If a procurement subpage link was found, navigate to it
            active_url = url
            if target_link_url and target_link_url != url:
                try:
                    await page.goto(target_link_url, timeout=timeout_ms, wait_until="domcontentloaded")
                    active_url = target_link_url
                except Exception:
                    pass

            # Scroll and extract page details
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(0.5)

            page_title = await page.title() or ""
            body_text = await page.inner_text("body") or ""
            body_clean = re.sub(r"\s+", " ", body_text).strip()

            # Analyze extracted body text for procurement signals
            matched_signals = []
            body_lower = body_clean.lower()

            for kw in PROCUREMENT_KEYWORDS:
                if re.search(r"\b" + re.escape(kw) + r"\b", body_lower):
                    matched_signals.append(kw.title())

            matched_signals = list(dict.fromkeys(matched_signals))

            # Filter out self-referential vendor noise
            is_self_vendor = any(re.search(pat, body_lower) for pat in EXCLUDED_SELF_VENDOR_PATTERNS)

            await browser.close()

            if matched_signals and not is_self_vendor:
                # Extract relevant text snippet surrounding the first matched signal
                first_sig = matched_signals[0].lower()
                idx = body_lower.find(first_sig)
                snippet_start = max(0, idx - 100)
                snippet_end = min(len(body_clean), idx + 150)
                snippet = body_clean[snippet_start:snippet_end]

                evidence.update({
                    "success": True,
                    "has_vendor_intent": True,
                    "source_url": active_url,
                    "page_title": page_title[:100],
                    "snippet": f"...{snippet}...",
                    "matched_signals": matched_signals,
                    "relevance_score": min(100.0, 75.0 + len(matched_signals) * 5.0),
                    "confidence": min(0.95, 0.70 + len(matched_signals) * 0.05),
                    "reason": f"Playwright deep scan verified active procurement portal on {active_url}"
                })
            else:
                evidence.update({
                    "success": True,
                    "has_vendor_intent": False,
                    "source_url": active_url,
                    "page_title": page_title[:100],
                    "snippet": body_clean[:200],
                    "matched_signals": [],
                    "relevance_score": 30.0,
                    "confidence": 0.50,
                    "reason": "Playwright deep scan found no explicit procurement/supplier signals."
                })

            return evidence

    except Exception as err:
        evidence["reason"] = f"Playwright execution error: {str(err)[:100]}"
        return evidence


def deep_verify_vendor_intent(url: str, timeout_ms: int = 8000) -> Dict[str, Any]:
    """
    Synchronous wrapper for deep_verify_vendor_intent_async.
    """
    try:
        return asyncio.run(deep_verify_vendor_intent_async(url, timeout_ms))
    except Exception as err:
        return {
            "success": False,
            "has_vendor_intent": False,
            "reason": f"Sync wrapper exception: {str(err)[:100]}"
        }
