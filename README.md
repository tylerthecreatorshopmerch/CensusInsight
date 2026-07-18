# CensusInsight PK — Pakistan Population Data Analysis System

A Python data-science desktop/web application built on the **7th Population &
Housing Census 2023** (Pakistan Bureau of Statistics). It implements the full
data science pipeline: data loading & cleaning → exploratory analysis →
interactive visualizations & maps → machine learning population prediction →
automated PDF report generation.

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Rebuild the dataset from scratch
python data_generator.py

# 3. (Optional) Train & test the ML models standalone
python ml_model.py

# 4. (Optional) Test PDF report generation standalone
python report_generator.py

# 5. Launch the full app
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

## 📂 Project Structure

```
├── app.py                  # Main Streamlit application (all 9 modules)
├── data_generator.py       # Builds data/pakistan_census_2023.csv
├── ml_model.py              # Module 7: trains & compares regression models
├── report_generator.py      # Module 9: builds the PDF report
├── data/
│   └── pakistan_census_2023.csv
├── models/
│   └── best_population_model.joblib   (created after first ML run)
├── output/
│   └── CensusInsight_PK_Report.pdf    (created when you generate a report)
└── requirements.txt
```

## 🗂️ Modules Implemented (matches the project proposal)

1. **Data Loading, Cleaning & Preprocessing** — upload your own PBS Excel/CSV
   or use the bundled dataset; missing values handled, columns standardized.
2. **National & Provincial Overview Dashboard** — summary cards, province bar
   chart, pie chart, male/female comparison, drill-down to district level.
3. **Urban vs Rural Analysis** — stacked bars, top urban/rural districts,
   province filter, urban% vs growth-rate scatter.
4. **Population Density & Growth Rate Analysis** — density ranking,
   High/Low Growth Zones, household size by province.
5. **Gender Ratio & Demographic Breakdown** — female-per-100-male ratio,
   imbalance list, age-group distribution, literacy by gender.
6. **Geospatial Map** — interactive Folium map, indicator switcher, tooltips,
   top/bottom-10 sidebar tables.
7. **ML Population Prediction** — Linear Regression / Decision Tree / Random
   Forest compared on MAE & R², predicted-vs-actual scatter, feature
   importance, best model auto-saved with `joblib`.
8. **Population Projector** — pick a district, adjust the growth-rate slider,
   get 2030/2035 projections with an auto-generated insight sentence.
9. **Automated Report Generation** — one-click PDF combining all key findings
   for submission/viva.

## ⚠️ Important Note on Data Accuracy

- **District population figures (2023) are REAL**, taken directly from the
  official 7th Census results published by the Pakistan Bureau of Statistics
  and cross-verified against secondary compilations. The national and
  provincial totals in this dataset match the official totals **exactly**
  (241,499,431).
- **Secondary indicators** — district area, density, literacy rate, exact
  2017 population, household size, and urban/rural split at the *district*
  level — are **statistically modeled** (seeded/reproducible) using the
  correct national and provincial averages from the census, because PBS
  publishes these particular breakdowns inside large PDF volumes rather than
  a clean flat table.
- **For your actual submission:** download the official district-wise Excel
  tables from **pbos.gov.pk → Population Census section** and replace
  `data/pakistan_census_2023.csv` (or use the in-app file uploader in
  Module 1) — the pipeline will work unchanged with the exact official
  numbers, since it was built against the same column structure.
- The map in Module 6 uses approximate marker placement (no official district
  boundary shapefile is bundled). For an exact choropleth, download Pakistan's
  district boundaries (GeoJSON) from the Humanitarian Data Exchange
  (data.humdata.org) as the original proposal specifies.

## 🧑‍🎓 Supervisor
- Name: A Q Mohsin
- Email: mohsin@vu.edu.pk
