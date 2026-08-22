document.addEventListener("DOMContentLoaded", () => {
    const API_BASE_URL = "http://127.0.0.1:8000";

    // Navigation Tabs
    const tabWebBtn = document.getElementById("tabWebBtn");
    const tabPdfBtn = document.getElementById("tabPdfBtn");
    const modeWeb = document.getElementById("modeWeb");
    const modePdf = document.getElementById("modePdf");

    // UI Elements
    const currentDomainEl = document.getElementById("currentDomain");
    const analyzeWebBtn = document.getElementById("analyzeWebBtn");
    const analyzePdfBtn = document.getElementById("analyzePdfBtn");
    const manualInputBox = document.getElementById("manualInputBox");
    const manualCoName = document.getElementById("manualCoName");
    const manualDesc = document.getElementById("manualDesc");
    const toggleManualBtn = document.getElementById("toggleManualBtn");

    const dropzone = document.getElementById("dropzone");
    const pdfFileInput = document.getElementById("pdfFileInput");
    const fileLabel = document.getElementById("fileLabel");

    const loadingState = document.getElementById("loadingState");
    const errorState = document.getElementById("errorState");
    const resultCard = document.getElementById("resultCard");
    const resetBtn = document.getElementById("resetBtn");

    // Result Fields
    const badgeContainer = document.getElementById("badgeContainer");
    const scoreVal = document.getElementById("scoreVal");
    const progressFill = document.getElementById("progressFill");
    const resCompanyName = document.getElementById("resCompanyName");
    const resCategory = document.getElementById("resCategory");
    const chipGroup = document.getElementById("chipGroup");
    const resReason = document.getElementById("resReason");

    let currentTabData = null;
    let selectedPdfFile = null;

    // --- TAB SWITCHING LOGIC ---
    tabWebBtn.addEventListener("click", () => {
        tabWebBtn.classList.add("active");
        tabPdfBtn.classList.remove("active");
        modeWeb.style.display = "block";
        modePdf.style.display = "none";
        hideResults();
    });

    tabPdfBtn.addEventListener("click", () => {
        tabPdfBtn.classList.add("active");
        tabWebBtn.classList.remove("active");
        modePdf.style.display = "block";
        modeWeb.style.display = "none";
        hideResults();
    });

    // --- DETECT ACTIVE BROWSER TAB ---
    async function initCurrentTab() {
        try {
            if (!chrome || !chrome.tabs) {
                currentDomainEl.innerText = "Manual Entry Mode";
                showManualBox();
                return;
            }

            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            if (!tab) {
                currentDomainEl.innerText = "No Active Webpage";
                return;
            }

            let rawUrl = tab.url || tab.pendingUrl || "";
            let tabTitle = tab.title || "";

            if (!rawUrl || rawUrl.startsWith("chrome://") || rawUrl.startsWith("edge://") || rawUrl.startsWith("about:") || rawUrl.startsWith("chrome-extension://")) {
                currentDomainEl.innerText = "System Page (Manual Entry Mode)";
                showManualBox();
                return;
            }

            // Extract display domain
            try {
                const parsed = new URL(rawUrl);
                currentDomainEl.innerText = parsed.hostname.replace(/^www\./, "");
            } catch (e) {
                currentDomainEl.innerText = rawUrl;
            }

            // Execute content script on active tab
            if (chrome.scripting && tab.id) {
                try {
                    const [results] = await chrome.scripting.executeScript({
                        target: { tabId: tab.id },
                        func: extractWebpageInfo
                    });

                    if (results && results.result) {
                        currentTabData = results.result;
                    }
                } catch (err) {
                    console.warn("Scripting extraction fallback:", err);
                }
            }

            if (!currentTabData) {
                currentTabData = {
                    url: rawUrl,
                    title: tabTitle,
                    description: tabTitle,
                    content: tabTitle,
                    companyName: ""
                };
            }
        } catch (err) {
            console.warn("Tab initialization warning:", err);
            currentDomainEl.innerText = "Active Webpage";
        }
    }

    // Extractor script injected into webpage context
    function extractWebpageInfo() {
        const url = window.location.href;
        const title = document.title || "";

        let description = "";
        const metaDesc = document.querySelector('meta[name="description"]') ||
                         document.querySelector('meta[property="og:description"]');
        if (metaDesc) {
            description = metaDesc.getAttribute("content") || "";
        }

        const headings = Array.from(document.querySelectorAll('h1, h2, h3'))
                            .map(h => (h.innerText || '').trim())
                            .filter(t => t.length > 0)
                            .slice(0, 10)
                            .join(" | ");

        let bodyText = "";
        if (document.body) {
            bodyText = (document.body.innerText || '')
                        .replace(/\s+/g, ' ')
                        .trim()
                        .substring(0, 3000);
        }

        let companyName = "";
        const schemaOrg = document.querySelector('script[type="application/ld+json"]');
        if (schemaOrg) {
            try {
                const parsed = JSON.parse(schemaOrg.innerText);
                if (parsed.name) companyName = parsed.name;
                else if (parsed["@graph"]) {
                    const org = parsed["@graph"].find(item => item["@type"] === "Organization");
                    if (org && org.name) companyName = org.name;
                }
            } catch (e) {}
        }

        return {
            url: url,
            title: title,
            description: description,
            headings: headings,
            content: headings + " " + bodyText,
            companyName: companyName
        };
    }

    function showManualBox() {
        manualInputBox.style.display = "block";
        toggleManualBtn.innerText = "🙈 Hide Manual Fields";
    }

    toggleManualBtn.addEventListener("click", () => {
        if (manualInputBox.style.display === "none" || !manualInputBox.style.display) {
            manualInputBox.style.display = "block";
            toggleManualBtn.innerText = "🙈 Hide Manual Fields";
        } else {
            manualInputBox.style.display = "none";
            toggleManualBtn.innerText = "✏️ Edit Company Details Manually";
        }
    });

    // --- MODE 1: ANALYZE WEBPAGE COMPANY ---
    analyzeWebBtn.addEventListener("click", async () => {
        hideResults();
        showLoading("⚡ AI Neural Network Analyzing Company...");

        const manualCo = (manualCoName.value || "").trim();
        const manualD = (manualDesc.value || "").trim();

        let payload = {
            url: currentTabData ? currentTabData.url : "",
            company_name: manualCo || (currentTabData ? currentTabData.companyName : ""),
            title: manualCo || (currentTabData ? currentTabData.title : ""),
            description: manualD || (currentTabData ? currentTabData.description : ""),
            content: manualD ? (manualD + " " + (currentTabData ? currentTabData.content : "")) : (currentTabData ? currentTabData.content : "")
        };

        try {
            const response = await fetch(`${API_BASE_URL}/analyze-company`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                throw new Error(`Server returned HTTP ${response.status}`);
            }

            const data = await response.json();
            hideLoading();

            if (!data.success) {
                showError(data.message || "Not enough company information found on this page.");
                showManualBox();
                return;
            }

            renderCompanyResult(data);
        } catch (err) {
            hideLoading();
            console.error("API call error:", err);
            showError("❌ Could not connect to Galactic Verifier API.\nMake sure 'python -m uvicorn api:app --reload --port 8000' is running.");
        }
    });

    // --- MODE 2: ANALYZE BROCHURE PDF ---
    dropzone.addEventListener("click", () => pdfFileInput.click());

    pdfFileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) {
            selectedPdfFile = e.target.files[0];
            fileLabel.innerText = `📄 ${selectedPdfFile.name}`;
        }
    });

    analyzePdfBtn.addEventListener("click", async () => {
        if (!selectedPdfFile) {
            showError("Please select a PDF brochure file first.");
            return;
        }

        hideResults();
        showLoading("📄 Extracting PyMuPDF Brochure Capabilities...");

        const formData = new FormData();
        formData.append("file", selectedPdfFile);

        try {
            const response = await fetch(`${API_BASE_URL}/analyze`, {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Server returned HTTP ${response.status}`);
            }

            const data = await response.json();
            hideLoading();

            if (!data.success) {
                showError(data.message || "Could not analyze brochure.");
                return;
            }

            renderPdfResult(data);
        } catch (err) {
            hideLoading();
            console.error("PDF analysis error:", err);
            showError("❌ Could not connect to Galactic Verifier API.\nMake sure 'python -m uvicorn api:app --reload --port 8000' is running.");
        }
    });

    // --- RENDER RESULTS ---
    function renderCompanyResult(data) {
        resultCard.style.display = "block";

        const score = data.match_score || 0.0;
        const resultClass = data.result || "BAD";

        scoreVal.innerText = `${score.toFixed(1)}`;
        progressFill.style.width = `${Math.min(Math.max(score, 5), 100)}%`;

        let badgeClass = "badge-bad";
        let badgeIcon = "🔴";
        let fillClass = "fill-bad";

        if (resultClass === "GOOD") {
            badgeClass = "badge-good";
            badgeIcon = "🟢";
            fillClass = "fill-good";
        } else if (resultClass === "MODERATE") {
            badgeClass = "badge-moderate";
            badgeIcon = "🟡";
            fillClass = "fill-moderate";
        }

        progressFill.className = `progress-fill ${fillClass}`;
        badgeContainer.innerHTML = `<span class="result-badge ${badgeClass}">${badgeIcon} ${resultClass} MATCH</span>`;

        resCompanyName.innerText = data.company_name || data.domain || "Target Company";
        resCategory.innerText = data.category || "General Manufacturing";

        chipGroup.innerHTML = "";
        const caps = data.matched_capabilities || [];
        if (caps.length > 0) {
            caps.forEach(cap => {
                const chip = document.createElement("span");
                chip.className = "chip";
                chip.innerText = `• ${cap}`;
                chipGroup.appendChild(chip);
            });
        } else {
            chipGroup.innerHTML = `<span style="font-size: 11px; color: #64748b;">No specific target vertical chips matched</span>`;
        }

        resReason.innerText = data.reason || "Relevance analysis evaluated against Galactic 3D capabilities.";
    }

    function renderPdfResult(data) {
        resultCard.style.display = "block";

        scoreVal.innerText = `${data.keywords.length} Caps`;
        progressFill.style.width = "100%";
        progressFill.className = "progress-fill fill-good";

        badgeContainer.innerHTML = `<span class="result-badge badge-good">📄 BROCHURE EXTRACTED</span>`;

        resCompanyName.innerText = data.filename || "Uploaded PDF";
        resCategory.innerText = `${data.total_pages} Pages Processed`;

        chipGroup.innerHTML = "";
        const topKws = data.top_keywords || [];
        topKws.forEach(kw => {
            const chip = document.createElement("span");
            chip.className = "chip";
            chip.innerText = `• ${kw}`;
            chipGroup.appendChild(chip);
        });

        resReason.innerText = data.capability_summary || "PyMuPDF brochure capability extraction complete.";
    }

    function showLoading(msg) {
        loadingState.innerHTML = `<span class="spin-icon">⚡</span> ${msg}`;
        loadingState.style.display = "block";
    }

    function hideLoading() {
        loadingState.style.display = "none";
    }

    function showError(msg) {
        errorState.innerText = msg;
        errorState.style.display = "block";
    }

    function hideResults() {
        errorState.style.display = "none";
        loadingState.style.display = "none";
        resultCard.style.display = "none";
    }

    resetBtn.addEventListener("click", () => {
        hideResults();
        manualCoName.value = "";
        manualDesc.value = "";
    });

    // Initialize Active Browser Tab
    initCurrentTab();
});