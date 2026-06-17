import geopandas as gpd

FILES = [
    "combined_candidate_tea_area_filtered_1000.geojson",
    "manual_tea_polygons.geojson",
]

for file in FILES:

    gdf = gpd.read_file(file)

    print(file)

    print(
        "Invalid:",
        (~gdf.geometry.is_valid).sum()
    )

    # Repair
    gdf["geometry"] = gdf.geometry.make_valid()

    # Remove empty geometries
    gdf = gdf[
        (~gdf.geometry.is_empty)
        & gdf.geometry.notnull()
    ]

    output = file.replace(
        ".geojson",
        "_fixed.geojson"
    )

    gdf.to_file(
        output,
        driver="GeoJSON"
    )

    print("Saved:", output)
    print()