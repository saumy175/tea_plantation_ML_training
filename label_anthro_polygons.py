import geopandas as gpd
import pandas as pd

OVERLAP_THRESHOLD = 0.30

anthro = gpd.read_file(
    "combined_candidate_tea_area_filtered_1000_fixed.geojson"
)

manual = gpd.read_file(
    "manual_tea_polygons_fixed.geojson"
)
# Ensure same CRS
manual = manual.to_crs(anthro.crs)

# Create IDs if needed
if "poly_id" not in anthro.columns:
    anthro["poly_id"] = range(len(anthro))

# Work in metric CRS
anthro_m = anthro.to_crs("EPSG:32646")
manual_m = manual.to_crs("EPSG:32646")

anthro["tea_label"] = -1

# Spatial index speeds things up
manual_sindex = manual_m.sindex

for idx, anthro_row in anthro_m.iterrows():

    geom = anthro_row.geometry

    candidates = list(
        manual_sindex.intersection(
            geom.bounds
        )
    )

    if not candidates:
        continue

    candidate_manual = manual_m.iloc[candidates]

    best_overlap = 0
    best_label = None

    anthro_area = geom.area

    for _, manual_row in candidate_manual.iterrows():

        try:
            inter = geom.intersection(
                manual_row.geometry
            )
        except Exception: 
            continue

        if inter.is_empty:
            continue

        overlap = (
            inter.area / anthro_area
        )

        if overlap > best_overlap:

            best_overlap = overlap

            best_label = int(
                bool(manual_row["tea"])
            )

    if best_overlap >= OVERLAP_THRESHOLD:

        anthro.at[
            idx,
            "tea_label"
        ] = best_label

print()

print(
    anthro["tea_label"]
    .value_counts()
)

anthro.to_file(
    "anthro_labeled.geojson",
    driver="GeoJSON"
)

print()

print("Saved anthro_labeled.geojson")