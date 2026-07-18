"""
CensusInsight PK - Module 9: Automated Report Generation
==========================================================
Generates a professional PDF summary report of the population analysis:
dataset intro, national summary, top/bottom districts, ML results and
sample projections. Uses ReportLab.
"""

from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak,
)
import pandas as pd


def generate_report(df: pd.DataFrame, ml_summary: dict, chart_paths: dict,
                     output_path="output/CensusInsight_PK_Report.pdf",
                     student_name="Student Name", project_name="CensusInsight PK"):

    doc = SimpleDocTemplate(output_path, pagesize=A4,
                             topMargin=1.5 * cm, bottomMargin=1.5 * cm,
                             leftMargin=1.8 * cm, rightMargin=1.8 * cm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleStyle", parent=styles["Title"], fontSize=20,
                                  textColor=colors.HexColor("#0B5345"))
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], textColor=colors.HexColor("#117A65"))
    body = styles["BodyText"]

    story = []

    # --- Header ---
    story.append(Paragraph(project_name, title_style))
    story.append(Paragraph("Pakistan Population Data Analysis System", styles["Heading3"]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(f"Prepared by: <b>{student_name}</b>", body))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%d %B %Y, %I:%M %p')}", body))
    story.append(Spacer(1, 0.6 * cm))

    # --- Dataset introduction ---
    story.append(Paragraph("1. Dataset Introduction", h2))
    story.append(Paragraph(
        "This report is based on the 7th Population &amp; Housing Census 2023, "
        "conducted by the Pakistan Bureau of Statistics (PBS) &mdash; Pakistan's first "
        "fully digital census. The dataset covers all provinces and districts of "
        "Pakistan, including total population, urban/rural distribution, gender "
        "ratio, household counts, literacy and population density.", body))
    story.append(Spacer(1, 0.4 * cm))

    # --- National summary table ---
    story.append(Paragraph("2. National Population Summary", h2))
    total_pop = int(df["Population_2023"].sum())
    total_hh = int(df["Households"].sum())
    urban_pct = df["Urban_Population"].sum() / total_pop * 100
    rural_pct = 100 - urban_pct
    summary_data = [
        ["Metric", "Value"],
        ["Total National Population (2023)", f"{total_pop:,}"],
        ["Total Households", f"{total_hh:,}"],
        ["Urban Population %", f"{urban_pct:.1f}%"],
        ["Rural Population %", f"{rural_pct:.1f}%"],
        ["Number of Districts", str(len(df))],
        ["Number of Provinces", str(df['Province'].nunique())],
    ]
    t = Table(summary_data, colWidths=[9 * cm, 6 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#117A65")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.5 * cm))

    if chart_paths.get("province_overview"):
        story.append(Image(chart_paths["province_overview"], width=15 * cm, height=8 * cm))
    story.append(PageBreak())

    # --- Top/bottom density districts ---
    story.append(Paragraph("3. Population Density: Most & Least Crowded Districts", h2))
    top_dense = df.nlargest(10, "Population_Density")[["District", "Province", "Population_Density"]]
    low_dense = df.nsmallest(10, "Population_Density")[["District", "Province", "Population_Density"]]

    def make_table(dframe, title):
        data = [["District", "Province", "Density (/km²)"]] + [
            [r.District, r.Province, f"{r.Population_Density:,.1f}"] for r in dframe.itertuples()
        ]
        tt = Table(data, colWidths=[5 * cm, 6 * cm, 4 * cm])
        tt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1A5276")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ]))
        return [Paragraph(title, styles["Heading4"]), tt, Spacer(1, 0.3 * cm)]

    story += make_table(top_dense, "Top 10 Most Densely Populated Districts")
    story += make_table(low_dense, "Top 10 Least Densely Populated Districts")
    story.append(PageBreak())

    # --- Urban vs Rural ---
    story.append(Paragraph("4. Urban vs Rural Breakdown (by Province)", h2))
    prov_ur = df.groupby("Province")[["Urban_Population", "Rural_Population"]].sum()
    prov_ur["Urban %"] = (prov_ur["Urban_Population"] / (prov_ur["Urban_Population"] + prov_ur["Rural_Population"]) * 100).round(1)
    data = [["Province", "Urban Population", "Rural Population", "Urban %"]] + [
        [idx, f"{row.Urban_Population:,}", f"{row.Rural_Population:,}", f"{row['Urban %']}%"]
        for idx, row in prov_ur.iterrows()
    ]
    t2 = Table(data, colWidths=[5 * cm, 4 * cm, 4 * cm, 3 * cm])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#B9770E")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
    ]))
    story.append(t2)
    story.append(Spacer(1, 0.5 * cm))

    if chart_paths.get("urban_rural"):
        story.append(Image(chart_paths["urban_rural"], width=15 * cm, height=8 * cm))
    story.append(PageBreak())

    # --- Gender ratio ---
    story.append(Paragraph("5. Key Gender Ratio Findings", h2))
    natl_ratio = df["Female_Population"].sum() / df["Male_Population"].sum() * 100
    story.append(Paragraph(
        f"The national gender ratio stands at <b>{natl_ratio:.1f} females per 100 males</b>. "
        f"The district with the most balanced ratio and districts with notable imbalance are "
        f"highlighted in the interactive dashboard (Module 5).", body))
    most_imbalanced = df.iloc[(df["Gender_Ratio_F_per_100M"] - 100).abs().sort_values(ascending=False).index[:5]]
    data = [["District", "Province", "F per 100 M"]] + [
        [r.District, r.Province, f"{r.Gender_Ratio_F_per_100M}"] for r in most_imbalanced.itertuples()
    ]
    t3 = Table(data, colWidths=[5 * cm, 6 * cm, 4 * cm])
    t3.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#6C3483")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
    ]))
    story.append(t3)
    story.append(PageBreak())

    # --- ML results ---
    story.append(Paragraph("6. Machine Learning Model Accuracy", h2))
    ml_data = [["Model", "MAE", "R\u00b2 Score"]]
    for name, r in ml_summary["results"].items():
        ml_data.append([name, f"{r['MAE']:,.0f}", f"{r['R2']:.4f}"])
    t4 = Table(ml_data, colWidths=[6 * cm, 5 * cm, 4 * cm])
    t4.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#922B21")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
    ]))
    story.append(t4)
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(f"Best performing model: <b>{ml_summary['best_name']}</b>", body))
    story.append(Spacer(1, 0.5 * cm))

    if chart_paths.get("ml_scatter"):
        story.append(Image(chart_paths["ml_scatter"], width=13 * cm, height=9 * cm))
    story.append(PageBreak())

    # --- Sample projections ---
    story.append(Paragraph("7. Sample District Population Projections", h2))
    sample = df.sample(min(3, len(df)), random_state=1)
    for r in sample.itertuples():
        proj_2030 = r.Population_2023 * ((1 + r.Growth_Rate_pct / 100) ** 7)
        proj_2035 = r.Population_2023 * ((1 + r.Growth_Rate_pct / 100) ** 12)
        story.append(Paragraph(
            f"<b>{r.District}, {r.Province}</b>: At the current growth rate of "
            f"{r.Growth_Rate_pct:.1f}%, population is projected to reach "
            f"<b>{proj_2030:,.0f}</b> by 2030 and <b>{proj_2035:,.0f}</b> by 2035.", body))
        story.append(Spacer(1, 0.2 * cm))

    story.append(Spacer(1, 0.6 * cm))
    story.append(Paragraph(
        "<i>Report auto-generated by CensusInsight PK. Data source: 7th Population "
        "&amp; Housing Census 2023, Pakistan Bureau of Statistics (pbos.gov.pk).</i>",
        styles["Italic"]))

    doc.build(story)
    return output_path


if __name__ == "__main__":
    import ml_model
    df = pd.read_csv("data/pakistan_census_2023.csv")
    out = ml_model.train_and_compare(df)
    path = generate_report(df, out, chart_paths={}, student_name="Shumila Rafeeq")
    print(f"Report generated at {path}")
