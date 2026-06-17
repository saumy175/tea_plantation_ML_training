import geopandas as gpd
import pandas as pd

gdf = gpd.read_file(
    "combined_candidate_tea_area_filtered_1000.geojson"
)

pred = pd.read_csv(
    "tea_predictions.csv"
)

gdf = gdf.merge(

    pred[[
        "poly_id",
        "tea_probability",
        "predicted_tea"
    ]],

    on="poly_id",

    how="left"
)

gdf.to_file(
    "final_tea_map.geojson",
    driver="GeoJSON"
)

print("Saved final_tea_map.geojson")