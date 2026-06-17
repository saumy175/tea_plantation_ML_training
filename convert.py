import geopandas as gpd

gdf = gpd.read_file(
    "combined_candidate_tea_area_filtered_1000.geojson"
)

gdf["poly_id"] = range(len(gdf))

rename_map = {
    # your columns
    "candidate_tea": "cand_tea",
    "source_estate": "src_est",
    "source_latitude": "src_lat",
    "source_longitude": "src_lon",
    "source_buffer_id": "src_buf",

    # AnthroKrishi columns
    "class_confidence": "cls_conf",
    "capture_timestamp_sec": "cap_time",
}

existing = {k: v for k, v in rename_map.items() if k in gdf.columns}
gdf = gdf.rename(columns=existing)

print(gdf.columns.tolist())

gdf.to_file(
    "tea_fields_upload",
    driver="ESRI Shapefile"
)