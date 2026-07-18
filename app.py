"""
CensusInsight PK - Pakistan Population Data Analysis System
=============================================================
Main Streamlit application. Run with:  streamlit run app.py

Implements all 9 modules from the project proposal:
 1. Data Loading, Cleaning & Preprocessing
 2. National & Provincial Population Overview Dashboard
 3. Urban vs Rural Analysis
 4. Population Density & Growth Rate Analysis
 5. Gender Ratio & Demographic Breakdown
 6. Geospatial Heatmap & District-Level Map Visualization
 7. Population Growth Prediction using Machine Learning
 8. Population Projector (Interactive Prediction Tool)
 9. Automated Report Generation
"""

import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ml_model
import report_generator

st.set_page_config(page_title="CensusInsight PK", page_icon="🇵🇰", layout="wide")

DATA_PATH = "data/pakistan_census_2023.csv"
MODEL_PATH = "models/best_population_model.joblib"

# Approximate district centroids for the map (province-level fallback jitter
# used where an exact district centroid isn't hardcoded, for demo purposes).
PROVINCE_CENTROIDS = {
    "Punjab": (31.1, 72.7), "Sindh": (25.9, 68.8),
    "Khyber Pakhtunkhwa": (34.5, 72.0), "Balochistan": (28.5, 65.5),
    "Islamabad Capital Territory": (33.7, 73.1),
}

# ---------------------------------------------------------------------------
# MODULE 1: Data Loading, Cleaning & Preprocessing
# ---------------------------------------------------------------------------
@st.cache_data
def load_and_clean_data(uploaded_file=None):
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(DATA_PATH)

    # --- Cleaning steps ---
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    before = len(df)
    df = df.dropna(subset=["District", "Province", "Population_2023"])
    missing_report = before - len(df)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())

    df["District"] = df["District"].astype(str).str.strip()
    df["Province"] = df["Province"].astype(str).str.strip()

    return df, missing_report


@st.cache_resource
def get_or_train_model(df):
    if os.path.exists(MODEL_PATH):
        bundle = joblib.load(MODEL_PATH)
        train_out = ml_model.train_and_compare(df)  # re-run for display metrics
        return bundle, train_out
    train_out = ml_model.train_and_compare(df)
    ml_model.save_best_model(train_out, MODEL_PATH)
    bundle = joblib.load(MODEL_PATH)
    return bundle, train_out


def human_number(n):
    n = float(n)
    if n >= 1e9:
        return f"{n/1e9:.2f} B"
    if n >= 1e6:
        return f"{n/1e6:.2f} M"
    if n >= 1e3:
        return f"{n/1e3:.1f} K"
    return f"{n:.0f}"


# ---------------------------------------------------------------------------
# SIDEBAR - Module 1 UI
# ---------------------------------------------------------------------------
st.sidebar.title("🇵🇰 CensusInsight PK")
st.sidebar.caption("Pakistan Population Data Analysis System")
st.sidebar.markdown("**Module 1: Data Loading**")
uploaded = st.sidebar.file_uploader("Load PBS Census Excel/CSV file", type=["xlsx", "csv"])
df, missing_report = load_and_clean_data(uploaded)

st.sidebar.success(f"Loaded {len(df)} districts")
st.sidebar.markdown(
    f"""
    - **Total districts:** {len(df)}
    - **Provinces:** {df['Province'].nunique()}
    - **National population:** {df['Population_2023'].sum():,}
    - **Missing/incomplete rows removed:** {missing_report}
    """
)

page = st.sidebar.radio(
    "Go to Module",
    [
        "1. Data Summary",
        "2. National & Provincial Overview",
        "3. Urban vs Rural Analysis",
        "4. Density & Growth Rate",
        "5. Gender Ratio & Demographics",
        "6. Geospatial Map",
        "7. ML Population Prediction",
        "8. Population Projector",
        "9. Automated Report",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("Dataset: 7th Population & Housing Census 2023, PBS (pbos.gov.pk)")

# ===========================================================================
# MODULE 1 PAGE
# ===========================================================================
if page.startswith("1."):
    st.title("Module 1 — Data Loading, Cleaning & Preprocessing")
    st.markdown(
        "Census Excel files from all provinces are imported, cleaned, and "
        "standardized into one unified dataset used across every module."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Districts loaded", len(df))
    c2.metric("Provinces", df["Province"].nunique())
    c3.metric("National population", human_number(df["Population_2023"].sum()))
    c4.metric("Incomplete rows removed", missing_report)

    st.subheader("Cleaned & Standardized Dataset")
    st.dataframe(df, use_container_width=True, height=420)

    st.subheader("Column Data Types (verified numeric)")
    st.dataframe(df.dtypes.astype(str).rename("dtype"), use_container_width=True)

# ===========================================================================
# MODULE 2 PAGE
# ===========================================================================
elif page.startswith("2."):
    st.title("Module 2 — National & Provincial Population Overview")

    total_pop = df["Population_2023"].sum()
    total_hh = df["Households"].sum()
    urban_pct = df["Urban_Population"].sum() / total_pop * 100
    rural_pct = 100 - urban_pct

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total National Population", human_number(total_pop))
    c2.metric("Total Households", human_number(total_hh))
    c3.metric("Urban Population %", f"{urban_pct:.1f}%")
    c4.metric("Rural Population %", f"{rural_pct:.1f}%")

    prov = df.groupby("Province", as_index=False)["Population_2023"].sum().sort_values(
        "Population_2023"
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(prov, x="Population_2023", y="Province", orientation="h",
                     title="Population by Province", color="Province",
                     text_auto=".2s")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.pie(prov, names="Province", values="Population_2023",
                       title="Share of National Population by Province", hole=0.35)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Male vs Female Population by Province")
    gender = df.groupby("Province", as_index=False)[["Male_Population", "Female_Population"]].sum()
    gender_melt = gender.melt(id_vars="Province", var_name="Gender", value_name="Population")
    fig3 = px.bar(gender_melt, x="Province", y="Population", color="Gender", barmode="group")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Drill Down: District-Level Data for a Province")
    selected_prov = st.selectbox("Click / select a province", sorted(df["Province"].unique()))
    st.dataframe(
        df[df["Province"] == selected_prov][
            ["District", "Population_2023", "Population_Density", "Urban_Pct", "Growth_Rate_pct"]
        ].sort_values("Population_2023", ascending=False),
        use_container_width=True,
    )

# ===========================================================================
# MODULE 3 PAGE
# ===========================================================================
elif page.startswith("3."):
    st.title("Module 3 — Urban vs Rural Analysis")

    prov_ur = df.groupby("Province", as_index=False)[["Urban_Population", "Rural_Population"]].sum()
    prov_ur_melt = prov_ur.melt(id_vars="Province", var_name="Type", value_name="Population")
    fig = px.bar(prov_ur_melt, x="Province", y="Population", color="Type", barmode="stack",
                 title="Urban vs Rural Population by Province")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("District Breakdown by Province")
    prov_filter = st.selectbox("Filter by province", sorted(df["Province"].unique()), key="m3prov")
    sub = df[df["Province"] == prov_filter].sort_values("Urban_Pct", ascending=False)
    fig2 = px.bar(sub, x="District", y="Urban_Pct", title=f"Urbanization % by District — {prov_filter}",
                  color="Urban_Pct", color_continuous_scale="Teal")
    fig2.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig2, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top 10 Most Urbanized Districts**")
        st.dataframe(df.nlargest(10, "Urban_Pct")[["District", "Province", "Urban_Pct"]],
                     use_container_width=True)
    with col2:
        st.markdown("**Top 10 Most Rural Districts**")
        st.dataframe(df.nsmallest(10, "Urban_Pct")[["District", "Province", "Urban_Pct"]],
                     use_container_width=True)

    st.subheader("Growth Rate: Urban-Leaning vs Rural-Leaning Districts")
    fig3 = px.scatter(df, x="Urban_Pct", y="Growth_Rate_pct", color="Province",
                       hover_name="District", size="Population_2023",
                       title="Urbanization % vs Population Growth Rate")
    st.plotly_chart(fig3, use_container_width=True)
    st.caption(
        "Note: 2017 vs 2023 district boundary/urban definitions changed for some "
        "newly merged districts (e.g. former FATA); comparisons there are indicative."
    )

# ===========================================================================
# MODULE 4 PAGE
# ===========================================================================
elif page.startswith("4."):
    st.title("Module 4 — Population Density & Growth Rate Analysis")

    sorted_density = df.sort_values("Population_Density", ascending=False)
    fig = px.bar(sorted_density, x="District", y="Population_Density",
                 color="Population_Density", color_continuous_scale="Sunset",
                 title="Population Density by District (people/km²)")
    fig.update_layout(xaxis_tickangle=-90, height=500)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top 10 Most Densely Populated Districts**")
        st.dataframe(df.nlargest(10, "Population_Density")[["District", "Province", "Population_Density"]],
                     use_container_width=True)
    with col2:
        st.markdown("**Top 10 Least Densely Populated Districts**")
        st.dataframe(df.nsmallest(10, "Population_Density")[["District", "Province", "Population_Density"]],
                     use_container_width=True)

    st.subheader("Growth Rate Comparison (2017 → 2023)")
    prov_growth = df.groupby("Province", as_index=False)["Growth_Rate_pct"].mean()
    fig2 = px.bar(prov_growth, x="Province", y="Growth_Rate_pct", color="Province",
                  title="Average Annual Growth Rate by Province")
    st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**🔺 High Growth Zones (> 3% per year)**")
        st.dataframe(df[df["Growth_Rate_pct"] > 3][["District", "Province", "Growth_Rate_pct"]]
                     .sort_values("Growth_Rate_pct", ascending=False), use_container_width=True)
    with col4:
        st.markdown("**🔻 Low Growth Zones (< 1.5% per year)**")
        st.dataframe(df[df["Growth_Rate_pct"] < 1.5][["District", "Province", "Growth_Rate_pct"]]
                     .sort_values("Growth_Rate_pct"), use_container_width=True)

    st.subheader("Average Household Size by Province")
    hh = df.groupby("Province", as_index=False)["Avg_Household_Size"].mean()
    fig3 = px.bar(hh, x="Province", y="Avg_Household_Size", color="Province")
    st.plotly_chart(fig3, use_container_width=True)

# ===========================================================================
# MODULE 5 PAGE
# ===========================================================================
elif page.startswith("5."):
    st.title("Module 5 — Gender Ratio & Demographic Breakdown")

    prov_gender = df.groupby("Province", as_index=False).apply(
        lambda g: pd.Series({
            "Gender_Ratio_F_per_100M": g["Female_Population"].sum() / g["Male_Population"].sum() * 100
        }), include_groups=False
    ).reset_index()
    natl_ratio = df["Female_Population"].sum() / df["Male_Population"].sum() * 100

    fig = px.bar(prov_gender, x="Province", y="Gender_Ratio_F_per_100M", color="Province",
                 title=f"Females per 100 Males by Province (National avg = {natl_ratio:.1f})")
    fig.add_hline(y=100, line_dash="dash", line_color="grey", annotation_text="Parity (100)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Districts with Notably Unbalanced Gender Ratio")
    df["ratio_deviation"] = (df["Gender_Ratio_F_per_100M"] - 100).abs()
    st.dataframe(
        df.nlargest(15, "ratio_deviation")[["District", "Province", "Gender_Ratio_F_per_100M"]],
        use_container_width=True,
    )

    st.subheader("Gender Ratio Grid — All Districts by Province")
    fig2 = px.treemap(df, path=["Province", "District"], values="Population_2023",
                       color="Gender_Ratio_F_per_100M", color_continuous_scale="RdYlGn",
                       color_continuous_midpoint=100,
                       title="District Gender Ratio (color) sized by Population")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Age Group Distribution by Province")
    age = df.groupby("Province", as_index=False)[
        ["Child_Pct_0_14", "Working_Pct_15_64", "Elderly_Pct_65plus"]
    ].mean()
    age_melt = age.melt(id_vars="Province", var_name="Age Group", value_name="Percent")
    fig3 = px.bar(age_melt, x="Province", y="Percent", color="Age Group", barmode="stack")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Male vs Female Literacy Rate by Province")
    lit = df.groupby("Province", as_index=False)[["Male_Literacy_Rate", "Female_Literacy_Rate"]].mean()
    lit_melt = lit.melt(id_vars="Province", var_name="Group", value_name="Literacy Rate")
    fig4 = px.bar(lit_melt, x="Province", y="Literacy Rate", color="Group", barmode="group")
    st.plotly_chart(fig4, use_container_width=True)
    st.caption("Hover over any bar above to see exact figures for that province/district.")

# ===========================================================================
# MODULE 6 PAGE
# ===========================================================================
elif page.startswith("6."):
    st.title("Module 6 — Geospatial Heatmap & District-Level Map")

    indicator = st.selectbox(
        "Choose indicator to visualize on the map",
        ["Population_2023", "Population_Density", "Growth_Rate_pct", "Urban_Pct",
         "Gender_Ratio_F_per_100M", "Avg_Household_Size"],
    )

    m = folium.Map(location=[30.3753, 69.3451], zoom_start=5.3, tiles="CartoDB positron")

    vmin, vmax = df[indicator].min(), df[indicator].max()

    def color_scale(v):
        ratio = (v - vmin) / (vmax - vmin + 1e-9)
        r = int(255 * ratio)
        b = int(255 * (1 - ratio))
        return f"#{r:02x}30{b:02x}"

    rng = np.random.default_rng(7)
    for row in df.itertuples():
        base_lat, base_lon = PROVINCE_CENTROIDS[row.Province]
        lat = base_lat + rng.uniform(-1.6, 1.6)
        lon = base_lon + rng.uniform(-1.6, 1.6)
        val = getattr(row, indicator)
        radius = 4 + 12 * ((row.Population_2023 - df["Population_2023"].min()) /
                            (df["Population_2023"].max() - df["Population_2023"].min() + 1e-9))
        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=color_scale(val),
            fill=True,
            fill_color=color_scale(val),
            fill_opacity=0.75,
            popup=folium.Popup(
                f"<b>{row.District}</b><br>Province: {row.Province}<br>"
                f"Total Population: {row.Population_2023:,}<br>"
                f"{indicator}: {val:,.1f}", max_width=250,
            ),
            tooltip=row.District,
        ).add_to(m)

    st.caption(
        "District markers are positioned with a demo jitter around provincial "
        "centroids (no official district boundary shapefile bundled in this "
        "environment). For production, replace with the official HDX district "
        "boundary GeoJSON (data.humdata.org) for exact choropleth polygons."
    )
    st_folium(m, width=1100, height=560)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Top 10 districts by {indicator}**")
        st.dataframe(df.nlargest(10, indicator)[["District", "Province", indicator]],
                     use_container_width=True)
    with col2:
        st.markdown(f"**Bottom 10 districts by {indicator}**")
        st.dataframe(df.nsmallest(10, indicator)[["District", "Province", indicator]],
                     use_container_width=True)

# ===========================================================================
# MODULE 7 PAGE
# ===========================================================================
elif page.startswith("7."):
    st.title("Module 7 — Population Growth Prediction (Machine Learning)")

    with st.spinner("Training and comparing regression models..."):
        bundle, train_out = get_or_train_model(df)

    st.subheader("Model Comparison")
    comp_rows = []
    for name, r in train_out["results"].items():
        comp_rows.append({"Model": name, "MAE": round(r["MAE"], 0), "R2 Score": round(r["R2"], 4)})
    comp_df = pd.DataFrame(comp_rows)
    st.dataframe(comp_df, use_container_width=True)
    st.success(f"Best performing model: **{train_out['best_name']}** "
               f"(R² = {train_out['results'][train_out['best_name']]['R2']:.4f})")

    st.subheader("Predicted vs Actual Population (Test Set)")
    best_preds = train_out["predictions"][train_out["best_name"]]
    scatter_df = pd.DataFrame({
        "Actual": train_out["y_test"].values,
        "Predicted": best_preds,
    })
    fig = px.scatter(scatter_df, x="Actual", y="Predicted",
                      title=f"{train_out['best_name']}: Predicted vs Actual Population")
    max_val = max(scatter_df["Actual"].max(), scatter_df["Predicted"].max())
    fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode="lines",
                              name="Perfect Prediction", line=dict(dash="dash", color="grey")))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature Importance (Random Forest)")
    fi = train_out["feature_importance"].reset_index()
    fi.columns = ["Feature", "Importance"]
    fig2 = px.bar(fi, x="Importance", y="Feature", orientation="h", color="Importance",
                  color_continuous_scale="Viridis")
    st.plotly_chart(fig2, use_container_width=True)

    st.info(f"Model saved to `{MODEL_PATH}` and ready to be reused by Module 8 "
            f"without retraining.")

# ===========================================================================
# MODULE 8 PAGE
# ===========================================================================
elif page.startswith("8."):
    st.title("Module 8 — Population Projector (Interactive Prediction Tool)")

    if not os.path.exists(MODEL_PATH):
        st.warning("No trained model found yet — visiting Module 7 first will train and save one. "
                   "Training now automatically...")
    bundle, train_out = get_or_train_model(df)

    district_name = st.selectbox("Select a district", sorted(df["District"].unique()))
    row = df[df["District"] == district_name].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Province", row["Province"])
    c2.metric("2023 Population", f"{row['Population_2023']:,}")
    c3.metric("2017 Population", f"{row['Population_2017']:,}")
    c4.metric("Current Growth Rate", f"{row['Growth_Rate_pct']:.2f}%/yr")

    growth_rate = st.slider(
        "Adjust growth rate (e.g., simulate a new highway / industrial zone)",
        min_value=0.0, max_value=8.0, value=float(row["Growth_Rate_pct"]), step=0.1,
    )

    if st.button("🔮 Project Population", type="primary"):
        proj_2030 = ml_model.project_population(bundle, row.to_dict(), 2030, growth_rate)
        proj_2035 = ml_model.project_population(bundle, row.to_dict(), 2035, growth_rate)

        chart_df = pd.DataFrame({
            "Year": ["2017", "2023", "2030 (proj.)", "2035 (proj.)"],
            "Population": [row["Population_2017"], row["Population_2023"], proj_2030, proj_2035],
        })
        fig = px.bar(chart_df, x="Year", y="Population", color="Year",
                     title=f"Population Trajectory — {district_name}", text_auto=".2s")
        st.plotly_chart(fig, use_container_width=True)

        st.success(
            f"📌 At the current growth rate of **{growth_rate:.1f}%**, **{district_name}** is "
            f"projected to reach **{proj_2030:,}** people by 2030 and **{proj_2035:,}** by 2035, "
            f"which will require significant expansion of housing, schools, healthcare, and "
            f"water supply infrastructure."
        )

# ===========================================================================
# MODULE 9 PAGE
# ===========================================================================
elif page.startswith("9."):
    st.title("Module 9 — Automated Report Generation")
    st.markdown(
        "Generate a professional PDF summary report combining the national "
        "overview, density rankings, urban/rural breakdown, gender findings, "
        "ML model accuracy, and sample projections — ready for project "
        "submission or the viva."
    )

    student_name = st.text_input("Student Name (for report header)", value="Student Name")
    project_name = st.text_input("Project Name", value="CensusInsight PK")

    if st.button("📄 Generate PDF Report", type="primary"):
        with st.spinner("Building charts and compiling report..."):
            os.makedirs("output", exist_ok=True)

            # Chart 1: province overview (matplotlib - no browser dependency)
            prov = df.groupby("Province", as_index=False)["Population_2023"].sum().sort_values("Population_2023")
            fig1, ax1 = plt.subplots(figsize=(9, 4.8))
            ax1.barh(prov["Province"], prov["Population_2023"], color="#117A65")
            ax1.set_xlabel("Population")
            ax1.set_title("Population by Province")
            fig1.tight_layout()
            fig1.savefig("output/chart_province.png", dpi=150)
            plt.close(fig1)

            # Chart 2: urban vs rural
            prov_ur = df.groupby("Province", as_index=False)[["Urban_Population", "Rural_Population"]].sum()
            fig2, ax2 = plt.subplots(figsize=(9, 4.8))
            ax2.bar(prov_ur["Province"], prov_ur["Urban_Population"], label="Urban", color="#2E86C1")
            ax2.bar(prov_ur["Province"], prov_ur["Rural_Population"],
                    bottom=prov_ur["Urban_Population"], label="Rural", color="#F5B041")
            ax2.set_ylabel("Population")
            ax2.set_title("Urban vs Rural by Province")
            ax2.legend()
            plt.setp(ax2.get_xticklabels(), rotation=20, ha="right")
            fig2.tight_layout()
            fig2.savefig("output/chart_urban_rural.png", dpi=150)
            plt.close(fig2)

            # ML results + scatter
            bundle, train_out = get_or_train_model(df)
            best_preds = train_out["predictions"][train_out["best_name"]]
            actual = train_out["y_test"].values
            fig3, ax3 = plt.subplots(figsize=(7, 6))
            ax3.scatter(actual, best_preds, alpha=0.7, color="#922B21")
            max_val = max(actual.max(), best_preds.max())
            ax3.plot([0, max_val], [0, max_val], "--", color="grey", label="Perfect Prediction")
            ax3.set_xlabel("Actual Population")
            ax3.set_ylabel("Predicted Population")
            ax3.set_title(f"{train_out['best_name']}: Predicted vs Actual")
            ax3.legend()
            fig3.tight_layout()
            fig3.savefig("output/chart_ml.png", dpi=150)
            plt.close(fig3)

            chart_paths = {
                "province_overview": "output/chart_province.png",
                "urban_rural": "output/chart_urban_rural.png",
                "ml_scatter": "output/chart_ml.png",
            }

            out_path = report_generator.generate_report(
                df, train_out, chart_paths,
                output_path="output/CensusInsight_PK_Report.pdf",
                student_name=student_name, project_name=project_name,
            )

        st.success("Report generated successfully!")
        with open(out_path, "rb") as f:
            st.download_button("⬇️ Download PDF Report", f, file_name="CensusInsight_PK_Report.pdf",
                                mime="application/pdf")
