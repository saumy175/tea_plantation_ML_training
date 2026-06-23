#!/usr/bin/env python3
"""
Predict tea probability for polygons in a GeoJSON using a saved XGBoost model.

What it does:
1) Reads an input GeoJSON of polygons.
2) Removes existing candidate_tea/tea attributes if present.
3) Uses Google Earth Engine to compute the model's feature set per polygon.
4) Runs the saved model to obtain tea_probability.
5) Writes:
   - polygons_with_tea_probability.geojson   (all polygons + probability + tea label)
   - polygons_tea_1.geojson                 (tea_probability > 0.995)
   - polygons_tea_0.geojson                 (tea_probability <= 0.995)

Requirements:
    pip install geopandas pandas numpy joblib scikit-learn xgboost earthengine-api geemap python-dotenv

Before running:
    1) Put your GEE project ID in the environment variable GEE_PROJECT_ID
       or replace <project_id> below.
    2) Make sure Earth Engine is authenticated.
"""

import os
import math
from pathlib import Path

import ee
import joblib
import geopandas as gpd
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# =============================================================================
# CONFIG
# =============================================================================

INPUT_GEOJSON = "combined_candidate_tea_area_filtered_1000.geojson"
MODEL_PATH = "tea_xgb_pixel_model.pkl"

OUTPUT_ALL_GEOJSON = "polygons_with_tea_probability.geojson"
OUTPUT_TEA1_GEOJSON = "polygons_tea_1.geojson"
OUTPUT_TEA0_GEOJSON = "polygons_tea_0.geojson"

# Earth Engine / imagery settings
DATE_START = "2023-01-01"
DATE_END = "2025-01-01"
CLOUDY_PIXEL_PERCENTAGE = 20
SCALE_M = 10   # use 10 m; change to 20 if you want a coarser polygon mean
BATCH_SIZE = 200  # smaller = safer, larger = faster

# If you prefer hardcoding, replace <project_id> directly.
load_dotenv()
GEE_PROJECT_ID = os.getenv("GEE_PROJECT_ID", "<project_id>")

if not GEE_PROJECT_ID or GEE_PROJECT_ID == "<project_id>":
    raise RuntimeError(
        "Set GEE_PROJECT_ID in your environment (or replace <project_id> in the script)."
    )

# =============================================================================
# HELPERS
# =============================================================================

def drop_unwanted_attributes(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Remove candidate_tea and tea if they exist, preserve everything else."""
    cols_to_drop = [c for c in ["candidate_tea", "tea"] if c in gdf.columns]
    if cols_to_drop:
        gdf = gdf.drop(columns=cols_to_drop)
    return gdf


def ensure_poly_id(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Guarantee a stable unique key for merging predictions back."""
    if "poly_id" not in gdf.columns:
        gdf = gdf.copy()
        gdf["poly_id"] = range(len(gdf))
    else:
        # Keep as integer if possible
        gdf = gdf.copy()
        gdf["poly_id"] = pd.to_numeric(gdf["poly_id"], errors="coerce").astype("Int64")
        if gdf["poly_id"].isna().any():
            raise ValueError("poly_id contains non-numeric values.")
    return gdf


def mask_s2_sr(image: ee.Image) -> ee.Image:
    """Basic QA60 cloud/cirrus mask for Sentinel-2 SR Harmonized."""
    qa = image.select("QA60")
    cloud_bit = 1 << 10
    cirrus_bit = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit).eq(0).And(qa.bitwiseAnd(cirrus_bit).eq(0))
    return image.updateMask(mask).divide(10000)


def build_composite() -> ee.Image:
    """Build a cloud-masked Sentinel-2 median composite with all model bands."""
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(DATE_START, DATE_END)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUDY_PIXEL_PERCENTAGE))
        .map(mask_s2_sr)
    )

    img = s2.median()

    # Indices
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


def make_ee_feature_collection(chunk: gpd.GeoDataFrame, prop_cols: list[str]) -> ee.FeatureCollection:
    """Convert a GeoDataFrame chunk to an EE FeatureCollection."""
    feats = []
    for _, row in chunk.iterrows():
        props = {}
        for c in prop_cols:
            val = row[c]
            if pd.isna(val):
                props[c] = None
            elif isinstance(val, (np.integer, np.int64)):
                props[c] = int(val)
            elif isinstance(val, (np.floating, np.float64)):
                props[c] = float(val)
            else:
                props[c] = val

        geom = ee.Geometry(row.geometry.__geo_interface__)
        feats.append(ee.Feature(geom, props))

    return ee.FeatureCollection(feats)


def ee_reduce_batch(image: ee.Image, fc: ee.FeatureCollection, feature_cols: list[str]) -> pd.DataFrame:
    """
    Reduce image over polygons for one batch and return a DataFrame containing
    poly_id plus the model feature columns.
    """
    reduced = image.reduceRegions(
        collection=fc,
        reducer=ee.Reducer.mean(),
        scale=SCALE_M,
        tileScale=4,
    )

    info = reduced.getInfo()
    rows = []
    for f in info.get("features", []):
        props = f.get("properties", {})
        row = {}
        # Keep only the required join key and model features
        row["poly_id"] = props.get("poly_id")
        for col in feature_cols:
            row[col] = props.get(col, np.nan)
        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    # Load model first so we know the exact feature list / order
    model = joblib.load(MODEL_PATH)

    if not hasattr(model, "feature_names_in_"):
        raise RuntimeError(
            "The saved model does not expose feature_names_in_. "
            "Please retrain/save with a DataFrame that has named columns."
        )

    feature_cols = list(model.feature_names_in_)
    print("Model features:", feature_cols)

    # Load polygons
    gdf = gpd.read_file(INPUT_GEOJSON)
    gdf = drop_unwanted_attributes(gdf)
    gdf = ensure_poly_id(gdf)

    # Keep geometry and original attrs, but work on a copy
    if gdf.empty:
        raise RuntimeError("Input GeoJSON contains no polygons.")

    # Initialize EE
    ee.Initialize(project=GEE_PROJECT_ID)

    # Build composite once
    composite = build_composite()

    # Property columns to carry into Earth Engine; exclude geometry
    prop_cols = [c for c in gdf.columns if c != "geometry"]

    # Batch process polygons to avoid timeouts
    n = len(gdf)
    n_batches = math.ceil(n / BATCH_SIZE)
    print(f"Polygons: {n}")
    print(f"Batches: {n_batches} (batch size = {BATCH_SIZE})")

    feature_frames = []

    for i in range(n_batches):
        start = i * BATCH_SIZE
        end = min((i + 1) * BATCH_SIZE, n)
        chunk = gdf.iloc[start:end].copy()

        print(f"Processing batch {i+1}/{n_batches}  rows {start}:{end}")

        fc = make_ee_feature_collection(chunk, prop_cols=prop_cols)
        batch_df = ee_reduce_batch(composite, fc, feature_cols=feature_cols)

        if batch_df.empty:
            print("  WARNING: batch returned no rows.")
            continue

        feature_frames.append(batch_df)

    if not feature_frames:
        raise RuntimeError("No features were extracted from Earth Engine.")

    features_df = pd.concat(feature_frames, ignore_index=True)

    # Merge back to original geometry/attributes
    merged = gdf.merge(features_df, on="poly_id", how="left", suffixes=("", "_feat"))

    # Ensure model feature order and numeric dtype
    X = merged[feature_cols].apply(pd.to_numeric, errors="coerce")

    # Predict probabilities
    merged["tea_probability"] = model.predict_proba(X)[:, 1]

    # Binary tea label
    merged["tea"] = (merged["tea_probability"] > 0.995).astype(int)

    # Write outputs
    merged.to_file(OUTPUT_ALL_GEOJSON, driver="GeoJSON")

    tea1 = merged[merged["tea"] == 1].copy()
    tea0 = merged[merged["tea"] == 0].copy()

    tea1.to_file(OUTPUT_TEA1_GEOJSON, driver="GeoJSON")
    tea0.to_file(OUTPUT_TEA0_GEOJSON, driver="GeoJSON")

    print("\nDone.")
    print(f"Saved: {OUTPUT_ALL_GEOJSON}")
    print(f"Saved: {OUTPUT_TEA1_GEOJSON}  (tea=1: {len(tea1)})")
    print(f"Saved: {OUTPUT_TEA0_GEOJSON}  (tea=0: {len(tea0)})")


if __name__ == "__main__":
    main()
