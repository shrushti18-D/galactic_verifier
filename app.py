"""
app.py
Production-Grade Offline-First Streamlit Dashboard Application for Galactic 3D.
Features complete combined Galactic taxonomy, batch company relevance analysis,
PyMuPDF brochure capability extraction, city/state filtering, and automated Excel reporting.
"""

import os
import io
import re
import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from cleaner import (
    load_and_clean_dataset,
    detect_canonical_columns,
    generate_excel_download,
    get_streamlined_dataframe
)
from brochure_reader import (
    extract_text_from_pdf,
    extract_capabilities,
    get_sample_galactic_profile
)
from analyzer import (
    analyze_companies_batch,
    INDUSTRY_SYNONYMS
)
from utils import apply_custom_css, render_footer

# Configure Streamlit Page
st.set_page_config(
    page_title="Galactic 3D Company Relevance Analyzer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply global custom CSS styling
apply_custom_css()


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
        return pd.read_csv(bio, low_memory=False, encoding='utf-8', on_bad_lines='skip')


def plot_classification_pie(df: pd.DataFrame):
    """Generates interactive Plotly Donut Pie Chart for GOOD / MODERATE / BAD distribution."""
    if "Result" not in df.columns or df.empty:
        fig = go.Figure()
        fig.update_layout(title="No classification data available")
        return fig

    counts = df["Result"].value_counts().reset_index()
    counts.columns = ["Result", "Count"]

    color_map = {
        "GOOD": "#10b981",      # Emerald Green
        "MODERATE": "#f59e0b",  # Amber Yellow
        "BAD": "#ef4444"        # Rose Red
    }

    fig = px.pie(
        counts,
        names="Result",
        values="Count",
        hole=0.45,
        title="Lead Classification Distribution",
        color="Result",
        color_discrete_map=color_map
    )
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label+value',
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percent:.1%}<extra></extra>"
    )
    fig.update_layout(
        margin=dict(t=40, b=20, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        height=320
    )
    return fig


def plot_score_histogram(df: pd.DataFrame):
    """Generates score distribution histogram."""
    if "Match Score" not in df.columns or df.empty:
        fig = go.Figure()
        fig.update_layout(title="No score data available")
        return fig

    fig = px.histogram(
        df,
        x="Match Score",
        nbins=20,
        title="Match Score Distribution (0 - 100)",
        color="Result",
        color_discrete_map={"GOOD": "#10b981", "MODERATE": "#f59e0b", "BAD": "#ef4444"},
        range_x=[0, 100]
    )
    fig.update_layout(
        xaxis_title="Match Score",
        yaxis_title="Number of Companies",
        margin=dict(t=40, b=20, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        height=320
    )
    return fig


def plot_top_keywords_bar(df: pd.DataFrame):
    """Plots top matched keywords bar chart."""
    if "Reason" not in df.columns or df.empty:
        fig = go.Figure()
        fig.update_layout(title="No keyword insights available")
        return fig

    all_kws = []
    for reason in df["Reason"].dropna():
        if "Matched keywords:" in str(reason):
            kw_part = str(reason).split("Matched keywords:")[1].split("|")[0].strip()
            if "(+" in kw_part:
                kw_part = kw_part.split("(+")[0].strip()
            kws = [k.strip() for k in kw_part.split(",") if k.strip()]
            all_kws.extend(kws)

    if not all_kws:
        fig = go.Figure()
        fig.update_layout(title="No matched keyword tags found")
        return fig

    kw_df = pd.Series(all_kws).value_counts().head(12).reset_index()
    kw_df.columns = ["Keyword", "Frequency"]

    fig = px.bar(
        kw_df,
        x="Frequency",
        y="Keyword",
        orientation='h',
        title="Top Matched Capability Keywords across Leads",
        color="Frequency",
        color_continuous_scale="Blues"
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        margin=dict(t=40, b=20, l=20, r=20),
        height=340
    )
    return fig


def main():
    initialize_session_state()
    render_header()

    # Sidebar Options & Controls
    with st.sidebar:
        st.header("⚙️ Verifier Controls")

        st.subheader("1. Galactic Capability Profile")
        profile_mode = st.radio(
            "Profile Source:",
            ["Default Galactic 3D Profile", "Upload Custom PDF Brochure"],
            index=0
        )

        if profile_mode == "Upload Custom PDF Brochure":
            uploaded_pdf = st.file_uploader("Upload PDF Brochure", type=["pdf"])
            if uploaded_pdf is not None:
                pdf_bytes = uploaded_pdf.read()
                pdf_text, total_pages = extract_text_from_pdf(pdf_bytes, uploaded_pdf.name)
                profile = extract_capabilities(pdf_text)
                st.session_state["brochure_profile"] = profile
                st.success(f"Extracted capabilities from '{uploaded_pdf.name}' ({total_pages} pages)!")
            else:
                st.session_state["brochure_profile"] = get_sample_galactic_profile()
        else:
            st.session_state["brochure_profile"] = get_sample_galactic_profile()

        st.markdown("---")
        st.subheader("2. Scoring Engine Weights")
        st.info("⚡ Hybrid Weights: **40% Category Overlap + 40% Keyword Overlap + 20% Cosine Similarity**.")

        st.markdown("---")
        st.subheader("3. ML Batch Performance")
        batch_size = st.slider("Batch Vector Processing Size", min_value=32, max_value=512, value=128, step=32)

    # Main Workflow Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📁 Step 1: Upload Dataset",
        "🧹 Step 2: Clean & Deduplicate",
        "📄 Step 3: Brochure Reader",
        "📊 Step 4: Relevance Analysis"
    ])

    # =========================================================================
    # TAB 1: UPLOAD DATASET
    # =========================================================================
    with tab1:
        st.subheader("Step 1: Upload Company Dataset")
        st.write("Upload your company leads spreadsheet (CSV, XLSX, XLS) to begin relevance verification.")

        uploaded_file = st.file_uploader(
            "Choose a dataset file (CSV or Excel)",
            type=["csv", "xlsx", "xls"],
            key="file_uploader_tab1"
        )

        if uploaded_file is not None:
            file_bytes = uploaded_file.read()
            raw_df = parse_file_cached(file_bytes, uploaded_file.name)

            if raw_df is not None and not raw_df.empty:
                st.session_state["raw_df"] = raw_df
                st.session_state["last_uploaded_filename"] = uploaded_file.name

                # Detect canonical column mapping automatically
                mapping = detect_canonical_columns(raw_df)
                st.session_state["column_mapping"] = mapping

                st.success(f"Successfully loaded **{uploaded_file.name}** with **{len(raw_df):,} rows** and **{len(raw_df.columns)} columns**!")

                st.subheader("Automatic Column Mapping Detected")
                col_map_cols = st.columns(5)
                with col_map_cols[0]:
                    st.text_input("Company Name Column", value=mapping.get("co_name") or "Not Found", disabled=True)
                with col_map_cols[1]:
                    st.text_input("Category / Industry Column", value=mapping.get("category") or "Not Found", disabled=True)
                with col_map_cols[2]:
                    st.text_input("Keywords / Services Column", value=mapping.get("keywords") or "Not Found", disabled=True)
                with col_map_cols[3]:
                    st.text_input("Website Column", value=mapping.get("website") or "Not Found", disabled=True)
                with col_map_cols[4]:
                    st.text_input("City / Location Column", value=mapping.get("city") or "Not Found", disabled=True)

                st.markdown("### Raw Dataset Preview")
                st.dataframe(raw_df.head(50), use_container_width=True)

        elif st.session_state.get("raw_df") is not None:
            raw_df = st.session_state["raw_df"]
            st.info(f"Dataset **{st.session_state.get('last_uploaded_filename', '')}** loaded ({len(raw_df):,} rows).")
            st.dataframe(raw_df.head(30), use_container_width=True)

    # =========================================================================
    # TAB 2: CLEAN & DEDUPLICATE
    # =========================================================================
    with tab2:
        st.subheader("Step 2: Clean & Deduplicate Dataset")

        if st.session_state.get("raw_df") is None:
            st.warning("⚠️ Please upload a dataset in Step 1 first.")
        else:
            raw_df = st.session_state["raw_df"]
            mapping = st.session_state["column_mapping"]

            st.write("Clean company names, remove duplicate websites/emails, and normalize text fields.")

            if st.button("🧼 Run Cleaning & Deduplication Pipeline", type="primary"):
                with st.spinner("Cleaning and deduplicating records..."):
                    cleaned_df, stats = load_and_clean_dataset(raw_df, mapping)
                    st.session_state["cleaned_df"] = cleaned_df
                    st.session_state["clean_stats"] = stats

                st.success("Cleaning pipeline complete!")

            if st.session_state.get("cleaned_df") is not None:
                cdf = st.session_state["cleaned_df"]
                stats = st.session_state["clean_stats"]

                st.markdown("### Deduplication Summary")
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Original Records", f"{stats['original_count']:,}")
                s2.metric("Unique Cleaned Records", f"{stats['cleaned_count']:,}")
                s3.metric("Duplicates Removed", f"{stats['total_removed']:,}")
                s4.metric("Deduplication Efficiency", f"{(stats['total_removed'] / max(1, stats['original_count']) * 100):.1f}%")

                st.markdown("### Cleaned Dataset Preview")
                st.dataframe(cdf.head(50), use_container_width=True)

    # =========================================================================
    # TAB 3: BROCHURE READER
    # =========================================================================
    with tab3:
        st.subheader("Step 3: PyMuPDF Brochure Capability Extractor")
        st.write("Extract capability profile and keywords from Galactic 3D or target competitor brochures.")

        profile = st.session_state.get("brochure_profile") or get_sample_galactic_profile()

        st.markdown("### Extracted Capability Profile Summary")
        st.info(profile.get("capability_summary", "Galactic 3D Manufacturing Profile"))

        st.markdown("### Extracted Target Keyword Taxonomy")
        kws = profile.get("keywords", [])
        if kws:
            kw_chips = " ".join([f"`{kw}`" for kw in kws])
            st.markdown(kw_chips)

        if "full_text" in profile and profile["full_text"]:
            with st.expander("📄 View Extracted PDF Text Snippets"):
                all_combined_text = profile["full_text"]
                st.text_area("Raw Brochure Text", all_combined_text, height=300)

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

                # CITY / STATE & VENDOR STATUS INTERACTIVE FILTER BAR
                f1, f2, f3, f4 = st.columns([2, 1.8, 2.2, 2])

                unique_cities = []
                if city_col and city_col in adf.columns:
                    unique_cities = sorted([c for c in adf[city_col].dropna().astype(str).unique() if c.strip() != ""])

                with f1:
                    if unique_cities:
                        selected_cities = st.multiselect("Select City / Location", options=unique_cities, default=unique_cities)
                    else:
                        selected_cities = []

                with f2:
                    filter_res = st.multiselect("Filter Classification", ["GOOD", "MODERATE", "BAD"], default=["GOOD", "MODERATE", "BAD"])

                with f3:
                    vendor_options = ["With Vendors (Active Buyer 🟢)", "Without Vendors (No Relevant Vendor Need 🔴)", "Uncertain / Insufficient Evidence 🟡"]
                    avail_vendors = [v for v in vendor_options if v in adf.get("Vendor Status", pd.Series([])).unique()] or vendor_options
                    selected_vendor_statuses = st.multiselect("Vendor Status Filter", options=vendor_options, default=avail_vendors)

                with f4:
                    search_query = st.text_input("Search Company or Keywords", value="")

                # Apply Filters
                filtered_adf = adf.copy()
                if unique_cities and selected_cities:
                    filtered_adf = filtered_adf[filtered_adf[city_col].astype(str).isin(selected_cities)]

                if filter_res:
                    filtered_adf = filtered_adf[filtered_adf["Result"].isin(filter_res)]

                if "Vendor Status" in filtered_adf.columns and selected_vendor_statuses:
                    filtered_adf = filtered_adf[filtered_adf["Vendor Status"].isin(selected_vendor_statuses)]

                # Metric Cards Calculate GOOD, MODERATE, BAD, TOTAL FOR THE SELECTED FILTERS
                counts_location = filtered_adf["Result"].value_counts().to_dict()
                good_cnt_loc = counts_location.get("GOOD", 0)
                mod_cnt_loc = counts_location.get("MODERATE", 0)
                bad_cnt_loc = counts_location.get("BAD", 0)
                total_cnt_loc = len(filtered_adf)

                good_pct_loc = (good_cnt_loc / total_cnt_loc * 100) if total_cnt_loc else 0
                mod_pct_loc = (mod_cnt_loc / total_cnt_loc * 100) if total_cnt_loc else 0
                bad_pct_loc = (bad_cnt_loc / total_cnt_loc * 100) if total_cnt_loc else 0

                # Render Metric Cards
                mc1, mc2, mc3, mc4 = st.columns(4)
                with mc1:
                    st.markdown(
                        f"""
                        <div class="metric-card card-good">
                            <div class="card-title">GOOD MATCHES</div>
                            <div class="card-value">{good_cnt_loc:,}</div>
                            <div class="card-subtitle">{good_pct_loc:.1f}% of selected filters</div>
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
                            <div class="card-subtitle">{mod_pct_loc:.1f}% of selected filters</div>
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
                            <div class="card-subtitle">{bad_pct_loc:.1f}% of selected filters</div>
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
                            <div class="card-subtitle">Filtered Profiles</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # Charts Row 1: Pie Donut & Score Histogram
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

                # Excel Downloads Section
                st.markdown("---")
                st.markdown("### 📥 Excel Report Downloads (Filtered)")

                good_only_filtered_df = filtered_adf[filtered_adf["Result"] == "GOOD"].copy()
                original_cols = [c for c in target_df.columns if c not in ["Match Score", "Result", "Vendor Status", "Reason"]]
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
                        help="Downloads ONLY GOOD companies containing strictly your original uploaded Excel columns."
                    )
                with exp2:
                    st.download_button(
                        label=f"📊 Download GOOD Leads (With Audit Scores & Vendor Status)",
                        data=excel_good_audit_bytes,
                        file_name=f"Galactic_3D_GOOD_Leads_With_Scores.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="secondary",
                        use_container_width=True,
                        help="Downloads GOOD companies with Match Score, Result, Vendor Status, and Reason columns."
                    )
                with exp3:
                    st.download_button(
                        label=f"📁 Download All Filtered Leads ({len(filtered_adf):,} Companies)",
                        data=excel_all_full_bytes,
                        file_name=f"Galactic_3D_All_Filtered_Leads.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="secondary",
                        use_container_width=True,
                        help="Downloads all filtered companies with all original columns + scores and Vendor Status."
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
