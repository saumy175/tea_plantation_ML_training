#!/usr/bin/env python3
import os
import math
from pathlib import Path

import ee
import geopandas as gpd
import pandas as pd
from dotenv import load_dotenv

# =============================================================================
# CONFIG
# =============================================================================

INPUT_GEOJSON = "combined_candidate_tea_area_filtered_1000.geojson"
EXPORT_FOLDER = "tea_polygon_pixel_batches"   # Drive folder
DATE_START = "2023-01-01"
DATE_END = "2025-01-01"
CLOUDY_PIXEL_PERCENTAGE = 20
SCALE_M = 10
BATCH_SIZE = 200

load_dotenv()
GEE_PROJECT_ID = os.getenv("GEE_PROJECT_ID", "<project_id>")

if not GEE_PROJECT_ID or GEE_PROJECT_ID == "<project_id>":
    raise RuntimeError("Set GEE_PROJECT_ID in your environment.")

# =============================================================================
# HELPERS
# =============================================================================

def drop_unwanted_attributes(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    cols_to_drop = [c for c in ["candidate_tea", "tea"] if c in gdf.columns]
    if cols_to_drop:
        gdf = gdf.drop(columns=cols_to_drop)
    return gdf


def ensure_poly_id(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    if "poly_id" not in gdf.columns:
        gdf["poly_id"] = range(len(gdf))
    else:
        gdf["poly_id"] = pd.to_numeric(gdf["poly_id"], errors="coerce").astype("Int64")
        if gdf["poly_id"].isna().any():
            raise ValueError("poly_id contains non-numeric values.")
    return gdf


def mask_s2_sr(image: ee.Image) -> ee.Image:
    qa = image.select("QA60")
    cloud_bit = 1 << 10
    cirrus_bit = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit).eq(0).And(qa.bitwiseAnd(cirrus_bit).eq(0))
    return image.updateMask(mask).divide(10000)


def build_composite() -> ee.Image:
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(DATE_START, DATE_END)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUDY_PIXEL_PERCENTAGE))
        .map(mask_s2_sr)
    )

    img = s2.median()

    ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
    gndvi = img.normalizedDifference(["B8", "B3"]).rename("GNDVI")
    ndwi = img.normalizedDifference(["B3", "B8"]).rename("NDWI")
    evi = img.expression(
        "2.5 * ((nir - red) / (nir + 6.0 * red - 7.5 * blue + 1.0))",
        {
            "nir": img.select("B8"),
            "red": img.select("B4"),
            "blue": img.select("B2"),
        },
    ).rename("EVI")

    model_bands = [
        "B2", "B3", "B4", "B5", "B6", "B7",
        "B8", "B8A", "B11", "B12"
    ]

    return img.select(model_bands).addBands([ndvi, gndvi, ndwi, evi])


def make_ee_feature_collection(chunk: gpd.GeoDataFrame) -> ee.FeatureCollection:
    feats = []
    for _, row in chunk.iterrows():
        geom = ee.Geometry(row.geometry.__geo_interface__)
        feats.append(ee.Feature(geom, {"poly_id": int(row["poly_id"])}))
    return ee.FeatureCollection(feats)

# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ee.Initialize(project=GEE_PROJECT_ID)

    gdf = gpd.read_file(INPUT_GEOJSON)
    gdf = drop_unwanted_attributes(gdf)
    gdf = ensure_poly_id(gdf)

    if gdf.empty:
        raise RuntimeError("Input GeoJSON contains no polygons.")

    composite = build_composite()

    n = len(gdf)
    n_batches = math.ceil(n / BATCH_SIZE)
    print(f"Polygons: {n}")
    print(f"Batches: {n_batches} (batch size = {BATCH_SIZE})")
    print(f"Drive folder: {EXPORT_FOLDER}")
    print("Starting export tasks...")

    for i in range(n_batches):
        start = i * BATCH_SIZE
        end = min((i + 1) * BATCH_SIZE, n)
        chunk = gdf.iloc[start:end].copy()

        print(f"Batch {i+1}/{n_batches}  rows {start}:{end}")

        fc = make_ee_feature_collection(chunk)

        sampled = composite.sampleRegions(
            collection=fc,
            properties=["poly_id"],
            scale=SCALE_M,
            geometries=False,
            tileScale=4,
        )

        task_name = f"tea_pixels_batch_{i+1:03d}"
        task = ee.batch.Export.table.toDrive(
            collection=sampled,
            description=task_name,
            folder=EXPORT_FOLDER,
            fileNamePrefix=task_name,
            fileFormat="CSV",
        )
        task.start()
        print(f"  Started: {task_name}")

    print("\nDone starting tasks.")
    print("Download the CSVs from Google Drive, put them into a local folder,")
    print("then run the local aggregation script below.")

if __name__ == "__main__":
    main()