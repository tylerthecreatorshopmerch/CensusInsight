"""
CensusInsight PK - Data Generator
==================================
Builds the unified district-level dataset used by the whole application.

Population figures (Population_2023, Province) are REAL figures from the
7th Population & Housing Census 2023, Pakistan Bureau of Statistics (PBS).
Source cross-checked against: pbs.gov.pk / census23.pbs.gov.pk and
secondary compilations (pakgeography.com, Wikipedia "2023 Pakistani census").

Because PBS releases the full breakdown (area, literacy, exact household
counts per district, urban/rural per district) inside large multi-hundred
page PDF volumes rather than a clean flat table, this script MODELS the
secondary indicators (area, density, literacy, urban%, household size,
2017 population) at district level using documented NATIONAL and
PROVINCIAL level anchors from the census, distributed across districts
with a reproducible random model (seeded) so that every run gives the
same numbers. Students should replace `data/pakistan_census_2023.csv`
with the exact PBS table for a final submission; the pipeline (Module 1)
already supports loading any correctly-columned Excel/CSV file.
"""

import pandas as pd
import numpy as np

np.random.seed(42)

# ---------------------------------------------------------------------------
# REAL district population figures, Census 2023 (PBS)
# ---------------------------------------------------------------------------
PUNJAB = {
    "Bahawalnagar": 3550342, "Bahawalpur": 4284964, "Rahim Yar Khan": 5564703,
    "Dera Ghazi Khan": 3393705, "Layyah": 2102386, "Muzaffargarh": 5015325,
    "Rajanpur": 2381049, "Chiniot": 1563024, "Faisalabad": 9075819,
    "Jhang": 3077720, "Toba Tek Singh": 2511963, "Gujranwala": 5959750,
    "Gujrat": 3219375, "Hafizabad": 1319909, "Mandi Bahauddin": 1829486,
    "Narowal": 1950954, "Sialkot": 4499394, "Kasur": 4084286, "Lahore": 13004135,
    "Nankana Sahib": 1634871, "Sheikhupura": 4049418, "Khanewal": 3364077,
    "Lodhran": 1928299, "Multan": 5362305, "Vehari": 3430421, "Attock": 2170423,
    "Chakwal": 1734854, "Jhelum": 1382308, "Rawalpindi": 6118911, "Okara": 3515490,
    "Pakpattan": 2136170, "Sahiwal": 2881811, "Bhakkar": 1957470, "Khushab": 1501089,
    "Mianwali": 1798268, "Sargodha": 4334448,
}

SINDH = {
    "Jacobabad": 1174097, "Kambar Shahdadkot": 1514869, "Kashmore": 1233957,
    "Larkana": 1784453, "Shikarpur": 1386330, "Karachi Central": 3822325,
    "Karachi East": 3950031, "Karachi South": 2329764, "Karachi West": 2679380,
    "Keamari": 2068451, "Korangi": 3128971, "Malir": 2403959, "Badin": 1947081,
    "Dadu": 1742320, "Hyderabad": 2432540, "Jamshoro": 1117308, "Matiari": 849383,
    "Sujawal": 839292, "Tando Allahyar": 922012, "Tando Muhammad Khan": 726119,
    "Thatta": 1083191, "Mirpur Khas": 1681386, "Tharparkar": 1778407,
    "Umerkot": 1159831, "Naushahro Feroze": 1777082, "Sanghar": 2308465,
    "Shaheed Benazirabad": 1845102, "Sukkur": 1639897, "Khairpur": 2597535,
    "Ghotki": 1772609,
}

KPK = {
    "Bannu": 1357890, "Lakki Marwat": 1040856, "North Waziristan": 693332,
    "Dera Ismail Khan": 1829811, "South Waziristan": 888675, "Tank": 470293,
    "Abbottabad": 1419072, "Battagram": 554133, "Haripur": 1174783,
    "Kolai Palas Kohistan": 280162, "Lower Kohistan": 340017, "Mansehra": 1797177,
    "Torghar": 200445, "Upper Kohistan": 422947, "Hangu": 528902, "Karak": 815878,
    "Kohat": 1234661, "Kurram": 785434, "Orakzai": 387561, "Bajaur": 1287960,
    "Buner": 1016869, "Lower Chitral": 320407, "Lower Dir": 1650183,
    "Malakand": 826250, "Shangla": 891252, "Swat": 2687384, "Upper Chitral": 195528,
    "Upper Dir": 1083566, "Mardan": 2744898, "Swabi": 1894600, "Charsadda": 1835504,
    "Khyber": 1146267, "Mohmand": 553933, "Nowshera": 1740705, "Peshawar": 4758762,
}

BALOCHISTAN = {
    "Gwadar": 305160, "Panjgur": 509781, "Kech": 1060931, "Awaran": 178958,
    "Kalat": 272506, "Khuzdar": 997214, "Lasbela": 680977, "Mastung": 313271,
    "Surab": 278092, "Barkhan": 210249, "Duki": 205044, "Loralai": 272432,
    "Musakhel": 182275, "Jaffarabad": 594558, "Jhal Magsi": 203368,
    "Kachhi": 442612, "Nasirabad": 563377, "Sohbatpur": 240106, "Chaman": 466218,
    "Killa Abdullah": 361971, "Pishin": 835482, "Quetta": 2595492, "Chagai": 269192,
    "Kharan": 260352, "Nushki": 207834, "Washuk": 302623, "Dera Bugti": 355274,
    "Harnai": 127571, "Kohlu": 260220, "Sibi": 224148, "Ziarat": 189535,
    "Killa Saifullah": 380200, "Sherani": 191687, "Zhob": 355692,
}

ICT = {"Islamabad": 2363863}

# Real, approximate PROVINCE areas (sq km) - well documented figures
PROVINCE_AREA_KM2 = {
    "Punjab": 205344, "Sindh": 140914, "Khyber Pakhtunkhwa": 101741,
    "Balochistan": 347190, "Islamabad Capital Territory": 906,
}

# National anchors, Census 2023 (PBS)
NATIONAL_URBAN_SHARE = 93884702 / 241499431      # 38.87%
NATIONAL_MALE_SHARE = 124324406 / 241499431      # 51.48%
NATIONAL_AVG_HH_SIZE = 6.30
NATIONAL_AVG_GROWTH = 2.55                        # % per year, 2017->2023
NATIONAL_LITERACY = 60.7

# Districts / cities known to be highly urbanized -> higher urban weight
HIGH_URBAN_DISTRICTS = {
    "Lahore", "Karachi Central", "Karachi East", "Karachi South", "Karachi West",
    "Keamari", "Korangi", "Malir", "Islamabad", "Rawalpindi", "Faisalabad",
    "Multan", "Gujranwala", "Peshawar", "Quetta", "Hyderabad", "Sialkot",
    "Sargodha", "Sukkur",
}

def build_dataset():
    rows = []
    provinces = {
        "Punjab": PUNJAB, "Sindh": SINDH, "Khyber Pakhtunkhwa": KPK,
        "Balochistan": BALOCHISTAN, "Islamabad Capital Territory": ICT,
    }

    for province, districts in provinces.items():
        province_total = sum(districts.values())
        # random area weights per district within the province, normalized
        weights = np.random.dirichlet(np.ones(len(districts)) * 1.5)
        area_total = PROVINCE_AREA_KM2[province]

        for (district, pop2023), w in zip(districts.items(), weights):
            # --- Area & density ---
            area_km2 = max(round(area_total * w, 1), 50)

            # --- 2017 population (back-calculated with per-district noise) ---
            growth_noise = np.random.normal(0, 0.6)
            district_growth = np.clip(NATIONAL_AVG_GROWTH + growth_noise, 0.2, 5.5)
            if district in ("Lahore", "Karachi Central", "Karachi East", "Islamabad",
                             "Karachi West", "Rawalpindi", "Faisalabad"):
                district_growth = np.clip(district_growth + 0.5, 0.2, 6.0)
            pop2017 = pop2023 / ((1 + district_growth / 100) ** 6)

            # --- Urban / rural split ---
            base_urban = NATIONAL_URBAN_SHARE
            if district in HIGH_URBAN_DISTRICTS:
                urban_pct = np.clip(np.random.normal(0.80, 0.08), 0.55, 0.99)
            else:
                urban_pct = np.clip(np.random.normal(base_urban * 0.75, 0.12), 0.05, 0.65)
            urban_pop = int(pop2023 * urban_pct)
            rural_pop = pop2023 - urban_pop

            # --- Gender split ---
            male_share = np.clip(np.random.normal(NATIONAL_MALE_SHARE, 0.01), 0.49, 0.54)
            male_pop = int(pop2023 * male_share)
            female_pop = pop2023 - male_pop

            # --- Households ---
            hh_size = np.clip(np.random.normal(NATIONAL_AVG_HH_SIZE, 0.5), 4.5, 8.5)
            households = int(pop2023 / hh_size)

            # --- Literacy ---
            literacy_boost = 12 if district in HIGH_URBAN_DISTRICTS else 0
            literacy = np.clip(np.random.normal(NATIONAL_LITERACY + literacy_boost, 8), 25, 92)
            male_literacy = np.clip(literacy + np.random.uniform(8, 15), 30, 97)
            female_literacy = np.clip(literacy - np.random.uniform(8, 15), 15, 90)

            # --- Age structure (approx national shares with noise) ---
            child_pct = np.clip(np.random.normal(37.2, 3), 28, 45)
            elderly_pct = np.clip(np.random.normal(4.2, 1), 2, 8)
            working_pct = 100 - child_pct - elderly_pct

            rows.append({
                "District": district,
                "Province": province,
                "Population_2023": int(pop2023),
                "Population_2017": int(pop2017),
                "Male_Population": male_pop,
                "Female_Population": female_pop,
                "Urban_Population": urban_pop,
                "Rural_Population": rural_pop,
                "Households": households,
                "Avg_Household_Size": round(hh_size, 2),
                "Area_km2": area_km2,
                "Literacy_Rate": round(literacy, 1),
                "Male_Literacy_Rate": round(male_literacy, 1),
                "Female_Literacy_Rate": round(female_literacy, 1),
                "Child_Pct_0_14": round(child_pct, 1),
                "Working_Pct_15_64": round(working_pct, 1),
                "Elderly_Pct_65plus": round(elderly_pct, 1),
            })

    df = pd.DataFrame(rows)
    df["Growth_Rate_pct"] = (
        ((df["Population_2023"] / df["Population_2017"]) ** (1 / 6) - 1) * 100
    ).round(2)
    df["Population_Density"] = (df["Population_2023"] / df["Area_km2"]).round(1)
    df["Urban_Pct"] = (df["Urban_Population"] / df["Population_2023"] * 100).round(1)
    df["Rural_Pct"] = (df["Rural_Population"] / df["Population_2023"] * 100).round(1)
    df["Gender_Ratio_F_per_100M"] = (
        df["Female_Population"] / df["Male_Population"] * 100
    ).round(1)

    df = df.sort_values(["Province", "District"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = build_dataset()
    df.to_csv("data/pakistan_census_2023.csv", index=False)
    print(f"Dataset built: {len(df)} districts, {df['Province'].nunique()} provinces")
    print(f"Total national population (sum of districts): {df['Population_2023'].sum():,}")
    print(df.groupby("Province")["Population_2023"].sum())
