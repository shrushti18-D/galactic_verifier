"""
app.py
Galactic 3D Company Relevance Analyzer - Main Streamlit Web Application.

A complete production-ready application to clean company data, extract Galactic 3D's
manufacturing capabilities from brochures using PyMuPDF, analyze company relevance using
SentenceTransformers (all-MiniLM-L6-v2) & cosine similarity, and classify companies into
GOOD, MODERATE, and BAD categories with downloadable formatted Excel reports and City/State filtering.
"""

import os
import io
from typing import Optional, Dict, List
import pandas as pd
import streamlit as st

# Import project modules
from cleaner import detect_column_mapping, clean_dataset
from brochure_reader import read_all_brochures, read_pdf_bytes, extract_capabilities
from analyzer import analyze_companies_batch
from utils import (
    apply_custom_css,
    plot_classification_pie,
    plot_score_histogram,
    plot_top_keywords_bar,
    generate_excel_download,
    generate_sample_dataset
)


# Configure Page Title & Layout
st.set_page_config(
    page_title="Galactic 3D Relevance Analyzer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)


def initialize_session_state():
    """Initializes persistent Streamlit session state variables."""
    if "raw_df" not in st.session_state:
        st.session_state["raw_df"] = None
    if "cleaned_df" not in st.session_state:
        st.session_state["cleaned_df"] = None
    if "clean_stats" not in st.session_state:
        st.session_state["clean_stats"] = None
    if "analyzed_df" not in st.session_state:
        st.session_state["analyzed_df"] = None
    if "brochure_profile" not in st.session_state:
        st.session_state["brochure_profile"] = None
    if "column_mapping" not in st.session_state:
        st.session_state["column_mapping"] = None
    if "last_uploaded_filename" not in st.session_state:
        st.session_state["last_uploaded_filename"] = None


def render_header():
    """Renders main visual banner header."""
    st.markdown(
        """
        <div class="galactic-header">
            <h1>🚀 Galactic 3D Company Relevance Analyzer</h1>
            <p>AI-Powered B2B Lead Classification Engine & Manufacturing Capability Matching</p>
        </div>
        """,
        unsafe_allow_html=True
    )


@st.cache_data(show_spinner=False)
def parse_file_cached(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    """
    Robust multi-engine file parser cached in memory.
    Handles CSV, XLSX, XLS (binary 97-2003), HTML table exports, and TSV files cleanly.
    """
    file_lower = file_name.lower()
    bio = io.BytesIO(file_bytes)

    if file_lower.endswith(".csv"):
        try:
            return pd.read_csv(bio, engine='c', low_memory=False, encoding='utf-8')
        except Exception:
            bio.seek(0)
            return pd.read_csv(bio, engine='python', low_memory=False, encoding='latin1')

    if file_lower.endswith(".xlsx"):
        try:
            return pd.read_excel(bio, engine='openpyxl')
        except Exception:
            bio.seek(0)
            return pd.read_excel(bio)

    if file_lower.endswith(".xls"):
        try:
            return pd.read_excel(bio, engine='xlrd')
        except Exception:
            pass
        bio.seek(0)
        try:
            return pd.read_excel(bio, engine='openpyxl')
        except Exception:
            pass
        bio.seek(0)
        try:
            dfs = pd.read_html(bio)
            if dfs:
                return dfs[0]
        except Exception:
            pass
        bio.seek(0)
        try:
            return pd.read_csv(bio, low_memory=False, encoding='latin1')
        except Exception:
            pass

    bio.seek(0)
    try:
        return pd.read_excel(bio)
    except Exception:
        bio.seek(0)
        return pd.read_csv(bio, low_memory=False, encoding='latin1')


def load_uploaded_file(uploaded_file) -> Optional[pd.DataFrame]:
    """Helper to parse uploaded Excel or CSV file into pandas DataFrame with visual spinner."""
    try:
        file_bytes = uploaded_file.read()
        df = parse_file_cached(file_bytes, uploaded_file.name)
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None


def get_streamlined_dataframe(df: pd.DataFrame, mapping: Dict[str, Optional[str]]) -> pd.DataFrame:
    """
    Constructs a streamlined, clean DataFrame containing key relevance fields:
    Company Name, Category, Keywords, City/Location, Match Score, Result (GOOD/MODERATE/BAD), and Reason.
    """
    co_col = mapping.get("co_name")
    cat_col = mapping.get("category")
    kw_col = mapping.get("keywords")
    city_col = mapping.get("city")

    result_df = pd.DataFrame()

    if co_col and co_col in df.columns:
        result_df["Company Name"] = df[co_col]
    elif "co_name" in df.columns:
        result_df["Company Name"] = df["co_name"]
    else:
        result_df["Company Name"] = df.iloc[:, 0] if len(df.columns) > 0 else ""

    if cat_col and cat_col in df.columns:
        result_df["Category"] = df[cat_col]
    elif "category" in df.columns:
        result_df["Category"] = df["category"]
    else:
        result_df["Category"] = ""

    if kw_col and kw_col in df.columns:
        result_df["Keywords"] = df[kw_col]
    elif "keywords" in df.columns:
        result_df["Keywords"] = df["keywords"]
    else:
        result_df["Keywords"] = ""

    if city_col and city_col in df.columns:
        result_df["City / State"] = df[city_col]
    elif "city" in df.columns:
        result_df["City / State"] = df["city"]
    else:
        result_df["City / State"] = ""

    result_df["Match Score"] = df["Match Score"] if "Match Score" in df.columns else 0.0
    result_df["Result"] = df["Result"] if "Result" in df.columns else "BAD"
    result_df["Reason"] = df["Reason"] if "Reason" in df.columns else ""

    return result_df


def main():
    apply_custom_css()
    initialize_session_state()
    render_header()

    # Sidebar Options
    with st.sidebar:
        st.header("⚙️ Settings & Configuration")
        st.markdown("---")
        st.subheader("🤖 ML Model Options")
        model_name = st.selectbox("SentenceTransformer Model", ["all-MiniLM-L6-v2"], index=0)
        batch_size = st.slider("Vector Batch Size", min_value=64, max_value=512, value=128, step=64)

        st.markdown("---")
        st.subheader("📚 Brochure Source")
        use_default_brochures = st.checkbox("Scan 'brochure/' folder", value=True)
        uploaded_brochures = st.file_uploader("Upload Additional Brochure PDFs", type=["pdf"], accept_multiple_files=True)

        st.markdown("---")
        st.markdown(
            """
            **Classification Thresholds:**
            - 🟢 **GOOD**: Score ≥ 75
            - 🟡 **MODERATE**: 45 ≤ Score < 75
            - 🔴 **BAD**: Score < 45

            **Scoring Weights:**
            - 40% Category Overlap
            - 40% Keyword Match
            - 20% Cosine Similarity
            """
        )

    # Workflow Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "1. 📥 Data Upload & Column Detection",
        "2. 🧹 Data Cleaning",
        "3. 📄 Galactic 3D Brochure Extraction",
        "4. 📊 Relevance Analysis & Dashboard"
    ])

    # =========================================================================
    # TAB 1: DATA UPLOAD & COLUMN DETECTION
    # =========================================================================
    with tab1:
        st.subheader("Step 1: Upload Company Dataset")
        col_up, col_sample = st.columns([3, 1])

        with col_up:
            uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
        with col_sample:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("✨ Load Sample Dataset", use_container_width=True):
                sample_df = generate_sample_dataset()
                st.session_state["raw_df"] = sample_df
                st.session_state["cleaned_df"] = None
                st.session_state["analyzed_df"] = None
                st.session_state["last_uploaded_filename"] = "sample_dataset"
                st.success("Loaded realistic sample dataset (20 companies with Cities)!")

        if uploaded_file is not None:
            if st.session_state.get("last_uploaded_filename") != uploaded_file.name:
                with st.spinner("⚡ Reading & parsing company file... Please wait a moment!"):
                    df = load_uploaded_file(uploaded_file)
                    if df is not None:
                        st.session_state["raw_df"] = df
                        st.session_state["cleaned_df"] = None
                        st.session_state["analyzed_df"] = None
                        st.session_state["last_uploaded_filename"] = uploaded_file.name
                        st.success(f"Successfully loaded file with {len(df):,} rows and {len(df.columns)} columns!")

        if st.session_state["raw_df"] is not None:
            raw_df = st.session_state["raw_df"]
            st.markdown("### 📋 Raw Dataset Preview")
            st.dataframe(raw_df.head(10), use_container_width=True)

            st.markdown("### 🔍 Automatic Column Detection")
            detected_map = detect_column_mapping(raw_df)

            col1, col2, col3, col4, col5, col6 = st.columns(6)
            all_cols = ["None"] + list(raw_df.columns)

            with col1:
                co_val = detected_map.get("co_name")
                co_idx = all_cols.index(co_val) if co_val in all_cols else 0
                sel_co = st.selectbox("Company Name", all_cols, index=co_idx)

            with col2:
                cat_val = detected_map.get("category")
                cat_idx = all_cols.index(cat_val) if cat_val in all_cols else 0
                sel_cat = st.selectbox("Category", all_cols, index=cat_idx)

            with col3:
                kw_val = detected_map.get("keywords")
                kw_idx = all_cols.index(kw_val) if kw_val in all_cols else 0
                sel_kw = st.selectbox("Keywords", all_cols, index=kw_idx)

            with col4:
                city_val = detected_map.get("city")
                city_idx = all_cols.index(city_val) if city_val in all_cols else 0
                sel_city = st.selectbox("City / State / Location", all_cols, index=city_idx)

            with col5:
                web_val = detected_map.get("website")
                web_idx = all_cols.index(web_val) if web_val in all_cols else 0
                sel_web = st.selectbox("Website", all_cols, index=web_idx)

            with col6:
                email_val = detected_map.get("email")
                email_idx = all_cols.index(email_val) if email_val in all_cols else 0
                sel_email = st.selectbox("Email", all_cols, index=email_idx)

            st.session_state["column_mapping"] = {
                "co_name": None if sel_co == "None" else sel_co,
                "category": None if sel_cat == "None" else sel_cat,
                "keywords": None if sel_kw == "None" else sel_kw,
                "city": None if sel_city == "None" else sel_city,
                "website": None if sel_web == "None" else sel_web,
                "email": None if sel_email == "None" else sel_email
            }

            if not st.session_state["column_mapping"]["co_name"]:
                st.warning("⚠️ Please select a valid Company Name column to proceed.")

    # =========================================================================
    # TAB 2: DATA CLEANING
    # =========================================================================
    with tab2:
        st.subheader("Step 2: Clean Data & Deduplicate")

        if st.session_state["raw_df"] is None:
            st.info("👈 Please upload a dataset in Step 1 first.")
        else:
            raw_df = st.session_state["raw_df"]
            mapping = st.session_state["column_mapping"]

            col_btn, _ = st.columns([2, 3])
            with col_btn:
                if st.button("🧹 Clean & Deduplicate Dataset Now", type="primary", use_container_width=True):
                    cleaned_df, stats = clean_dataset(raw_df, mapping)
                    st.session_state["cleaned_df"] = cleaned_df
                    st.session_state["clean_stats"] = stats
                    st.session_state["analyzed_df"] = None
                    st.success(f"Data cleaning completed! {stats['cleaned_count']:,} unique rows preserved, {stats['total_removed']:,} duplicates removed.")
                    st.rerun()

            if st.session_state["cleaned_df"] is not None:
                stats = st.session_state["clean_stats"]
                cleaned_df = st.session_state["cleaned_df"]

                st.markdown("### 📊 Dataset Cleaning Summary")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("Original Rows", f"{stats['original_count']:,}")
                with c2:
                    st.metric("Cleaned Rows", f"{stats['cleaned_count']:,}")
                with c3:
                    st.metric("Total Removed", f"{stats['total_removed']:,}")
                with c4:
                    st.metric("Duplicate Companies", f"{stats['duplicate_companies']:,}")

                with st.expander("🔍 View Detailed Audit Breakdown", expanded=False):
                    st.json({
                        "Exact Duplicate Rows Removed": stats["exact_duplicates"],
                        "Duplicate Company Names Removed": stats["duplicate_companies"],
                        "Duplicate Websites Removed": stats["duplicate_websites"],
                        "Duplicate Emails Removed": stats["duplicate_emails"],
                        "Missing Values Overview": stats["missing_values"]
                    })

                st.markdown("### ✨ Cleaned Dataset Preview")
                st.dataframe(cleaned_df.head(15), use_container_width=True)

    # =========================================================================
    # TAB 3: GALACTIC 3D BROCHURE EXTRACTION
    # =========================================================================
    with tab3:
        st.subheader("Step 3: PyMuPDF Galactic Brochure Extraction")
        st.write("Extract text page-by-page from brochures to build Galactic 3D's manufacturing capability profile.")

        brochure_texts = []
        file_details = []

        if use_default_brochures:
            b_info = read_all_brochures("brochure")
            if b_info["combined_text"]:
                brochure_texts.append(b_info["combined_text"])
                file_details.extend(b_info["files"])

        if uploaded_brochures:
            for up_pdf in uploaded_brochures:
                pdf_bytes = up_pdf.read()
                res = read_pdf_bytes(pdf_bytes, filename=up_pdf.name)
                if res["full_text"]:
                    brochure_texts.append(res["full_text"])
                    file_details.append(res)

        all_combined_text = "\n\n".join(brochure_texts)

        profile = extract_capabilities(all_combined_text)
        st.session_state["brochure_profile"] = profile

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Brochure Files Loaded", f"{len(file_details):,}")
        with c2:
            st.metric("Capabilities Extracted", f"{len(profile['keywords']):,}")

        st.markdown("### 🛠️ Discovered Manufacturing Capabilities")
        kw_tags = " ".join([f"`{kw}`" for kw in profile["keywords"]])
        st.markdown(kw_tags if kw_tags else "_No capabilities discovered_")

        st.markdown("### 📝 Consolidated Galactic Capability Profile")
        st.info(profile["capability_summary"])

        with st.expander("📄 View Full Raw Text Extracted from Brochures", expanded=False):
            if all_combined_text:
                st.text_area("Raw Brochure Text", all_combined_text, height=300)
            else:
                st.write("No PDF text available.")

    # =========================================================================
    # TAB 4: RELEVANCE ANALYSIS & DASHBOARD WITH CITY/STATE FILTERING
    # =========================================================================
    with tab4:
        st.subheader("Step 4: Machine Learning Relevance Analysis")

        cleaned_available = st.session_state.get("cleaned_df") is not None
        if cleaned_available:
            target_df = st.session_state["cleaned_df"]
            stats = st.session_state.get("clean_stats", {})
            orig_cnt = stats.get("original_count", len(target_df))
            rem_cnt = stats.get("total_removed", 0)
            st.success(f"✅ **Using Cleaned Dataset**: {len(target_df):,} unique companies (deduplicated from {orig_cnt:,} raw rows, {rem_cnt:,} duplicates dropped).")
        else:
            target_df = st.session_state.get("raw_df")
            if target_df is not None:
                st.info("ℹ️ **Tip**: You are currently analyzing raw data. Go to **Tab 2** to run cleaning & deduplication first if desired!")

        if target_df is None:
            st.warning("⚠️ Please upload a dataset in Step 1 first.")
        else:
            col_run, _ = st.columns([2, 3])
            with col_run:
                btn_label = f"🚀 Run Relevance Analysis on {len(target_df):,} Companies"
                if st.button(btn_label, type="primary", use_container_width=True):
                    profile = st.session_state.get("brochure_profile")
                    if not profile:
                        profile = extract_capabilities("")

                    progress_bar = st.progress(0, text="Initializing Machine Learning Engine...")

                    def update_progress(pct, text_msg):
                        progress_bar.progress(pct, text=text_msg)

                    with st.spinner("⚡ AI Neural Network is processing your companies... Please wait a few seconds!"):
                        analyzed_df = analyze_companies_batch(
                            df=target_df,
                            column_mapping=st.session_state["column_mapping"],
                            galactic_profile=profile,
                            batch_size=batch_size,
                            progress_callback=update_progress
                        )

                    progress_bar.progress(100, text="Analysis Complete!")
                    st.session_state["analyzed_df"] = analyzed_df
                    st.success(f"Analysis completed for {len(analyzed_df):,} companies!")
                    st.rerun()

            # Render Results Dashboard
            if st.session_state.get("analyzed_df") is not None:
                adf = st.session_state["analyzed_df"]
                mapping = st.session_state["column_mapping"]
                city_col = mapping.get("city")

                st.markdown("---")
                st.markdown("## 📈 Results & Relevance Dashboard")

                # CITY / STATE INTERACTIVE FILTER BAR PLACED RIGHT AT THE TOP!
                f1, f2, f3 = st.columns([2.2, 2, 2.5])

                unique_cities = []
                if city_col and city_col in adf.columns:
                    unique_cities = sorted([c for c in adf[city_col].dropna().astype(str).unique() if c.strip() != ""])

                with f1:
                    if unique_cities:
                        selected_cities = st.multiselect("Select City / State / Location", options=unique_cities, default=unique_cities)
                    else:
                        selected_cities = []

                with f2:
                    filter_res = st.multiselect("Filter Classification", ["GOOD", "MODERATE", "BAD"], default=["GOOD", "MODERATE", "BAD"])

                with f3:
                    search_query = st.text_input("Search Company, Category, or Keyword", value="")

                # Apply Location & Classification Filter FIRST
                filtered_adf = adf.copy()
                if unique_cities and selected_cities:
                    filtered_adf = filtered_adf[filtered_adf[city_col].astype(str).isin(selected_cities)]

                if filter_res:
                    filtered_adf = filtered_adf[filtered_adf["Result"].isin(filter_res)]

                # Metric Cards Calculate GOOD, MODERATE, BAD, TOTAL FOR THE SELECTED LOCATION ONLY!
                counts_location = filtered_adf["Result"].value_counts().to_dict()
                good_cnt_loc = counts_location.get("GOOD", 0)
                mod_cnt_loc = counts_location.get("MODERATE", 0)
                bad_cnt_loc = counts_location.get("BAD", 0)
                total_cnt_loc = len(filtered_adf)

                good_pct_loc = (good_cnt_loc / total_cnt_loc * 100) if total_cnt_loc else 0
                mod_pct_loc = (mod_cnt_loc / total_cnt_loc * 100) if total_cnt_loc else 0
                bad_pct_loc = (bad_cnt_loc / total_cnt_loc * 100) if total_cnt_loc else 0

                # Render Metric Cards (Dynamic for Selected Location!)
                mc1, mc2, mc3, mc4 = st.columns(4)
                with mc1:
                    st.markdown(
                        f"""
                        <div class="metric-card card-good">
                            <div class="card-title">GOOD MATCHES</div>
                            <div class="card-value">{good_cnt_loc:,}</div>
                            <div class="card-subtitle">{good_pct_loc:.1f}% of selected location</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with mc2:
                    st.markdown(
                        f"""
                        <div class="metric-card card-moderate">
                            <div class="card-title">MODERATE MATCHES</div>
                            <div class="card-value">{mod_cnt_loc:,}</div>
                            <div class="card-subtitle">{mod_pct_loc:.1f}% of selected location</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with mc3:
                    st.markdown(
                        f"""
                        <div class="metric-card card-bad">
                            <div class="card-title">BAD MATCHES</div>
                            <div class="card-value">{bad_cnt_loc:,}</div>
                            <div class="card-subtitle">{bad_pct_loc:.1f}% of selected location</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with mc4:
                    st.markdown(
                        f"""
                        <div class="metric-card card-total">
                            <div class="card-title">SELECTED TOTAL</div>
                            <div class="card-value">{total_cnt_loc:,}</div>
                            <div class="card-subtitle">Location Profiles</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # Charts Row 1: Pie Donut & Score Histogram (Dynamic for Selected Location!)
                ch1, ch2 = st.columns(2)
                with ch1:
                    fig_pie = plot_classification_pie(filtered_adf)
                    st.plotly_chart(fig_pie, use_container_width=True)
                with ch2:
                    fig_hist = plot_score_histogram(filtered_adf)
                    st.plotly_chart(fig_hist, use_container_width=True)

                # Charts Row 2: Top Matched Keywords Bar Chart
                st.markdown("### 🏆 Capability Overlap Insights")
                fig_kw = plot_top_keywords_bar(filtered_adf)
                st.plotly_chart(fig_kw, use_container_width=True)

                # Excel Downloads Section (Location Filtered!)
                st.markdown("---")
                st.markdown("### 📥 Excel Report Downloads (Location-Filtered)")

                good_only_filtered_df = filtered_adf[filtered_adf["Result"] == "GOOD"].copy()
                original_cols = [c for c in target_df.columns if c not in ["Match Score", "Result", "Reason"]]
                good_clean_original_df = good_only_filtered_df[original_cols] if not good_only_filtered_df.empty else pd.DataFrame()

                excel_good_clean_bytes = generate_excel_download(good_clean_original_df)
                excel_good_audit_bytes = generate_excel_download(good_only_filtered_df)
                excel_all_full_bytes = generate_excel_download(filtered_adf)

                streamlined_df = get_streamlined_dataframe(filtered_adf, mapping)
                excel_streamlined_bytes = generate_excel_download(streamlined_df)

                exp1, exp2, exp3 = st.columns(3)
                with exp1:
                    st.download_button(
                        label=f"🟢 Download GOOD Leads ({len(good_clean_original_df):,} Companies - Clean Original Columns)",
                        data=excel_good_clean_bytes,
                        file_name=f"Galactic_3D_GOOD_Leads_Clean_Original.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary",
                        use_container_width=True,
                        help="Downloads ONLY the GOOD companies for your selected location containing strictly your original uploaded Excel columns, without appending Match Score, Result, or Reason."
                    )
                with exp2:
                    st.download_button(
                        label=f"📊 Download GOOD Leads (With Audit Scores)",
                        data=excel_good_audit_bytes,
                        file_name=f"Galactic_3D_GOOD_Leads_With_Scores.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="secondary",
                        use_container_width=True,
                        help="Downloads GOOD companies for selected location with all original columns PLUS Match Score, Result, and Reason columns."
                    )
                with exp3:
                    st.download_button(
                        label=f"📁 Download All Filtered Leads ({len(filtered_adf):,} Companies)",
                        data=excel_all_full_bytes,
                        file_name=f"Galactic_3D_All_Filtered_Leads.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="secondary",
                        use_container_width=True,
                        help="Downloads all filtered companies for selected location with all original columns + scores."
                    )

                # Search & Filter Table
                st.markdown("### 📋 Filtered Results Table")
                display_df = streamlined_df.copy()
                if search_query.strip():
                    q = search_query.lower().strip()
                    mask = (
                        display_df.astype(str).apply(lambda row: row.str.lower().str.contains(q).any(), axis=1)
                    )
                    display_df = display_df[mask]

                st.markdown(f"**Showing {len(display_df):,} of {len(adf):,} total analyzed companies**")
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=500
                )


if __name__ == "__main__":
    main()
