import geopandas as gpd

INPUT_FILE = "combined_candidate_tea.geojson"
OUTPUT_FILE = "combined_candidate_tea_area_filtered_1000.geojson"
MIN_AREA_M2 = 1000

gdf = gpd.read_file(INPUT_FILE)

# Clean geometry first
gdf = gdf[gdf.geometry.notnull()].copy()
gdf = gdf[~gdf.geometry.is_empty].copy()

# Use a metric CRS for area calculation
gdf_m = gdf.to_crs("EPSG:32646")  # UTM zone 46N

gdf["area_m2"] = gdf_m.area

# Filter
filtered = gdf[gdf["area_m2"] >= MIN_AREA_M2].copy()

# Save
filtered.drop(columns=["area_m2"], inplace=True)
filtered.to_file(OUTPUT_FILE, driver="GeoJSON")

print(f"Saved: {OUTPUT_FILE}")
print(f"Original polygons: {len(gdf)}")
print(f"Kept polygons: {len(filtered)}")
print(f"Removed polygons: {len(gdf) - len(filtered)}")