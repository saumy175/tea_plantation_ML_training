#!/usr/bin/env python3
"""
Predict tea probability for polygons in a GeoJSON using a saved XGBoost model.

Process:
Polygon
  -> sample all Sentinel pixels inside polygon
  -> run pixel model on each pixel
  -> aggregate to polygon metrics:
       tea_probability = mean(pixel probabilities)
       tea_fraction    = fraction of pixels with p > 0.95
       n_pixels        = number of valid pixels sampled
  -> tea = 1 if tea_fraction >= TEA_FRACTION_THRESHOLD else 0

Outputs:
- polygons_with_tea_probability.geojson
- polygons_tea_1.geojson
- polygons_tea_0.geojson
- polygons_tea_metrics.csv
"""

import os
import math

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
OUTPUT_METRICS_CSV = "polygons_tea_metrics.csv"

DATE_START = "2023-01-01"
DATE_END = "2025-01-01"
CLOUDY_PIXEL_PERCENTAGE = 20

SCALE_M = 10
BATCH_SIZE = 200

PIXEL_PROB_THRESHOLD = 0.95
TEA_FRACTION_THRESHOLD = 0.20  # conservative starting point

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


def make_ee_feature_collection(chunk: gpd.GeoDataFrame, prop_cols: list[str]) -> ee.FeatureCollection:
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


def ee_sample_pixels_batch(image: ee.Image, fc: ee.FeatureCollection, feature_cols: list[str]) -> pd.DataFrame:
    """
    Sample all valid pixels inside polygons for one batch.
    Returns a DataFrame with poly_id + model feature columns.
    """
    sampled = image.sampleRegions(
        collection=fc,
        properties=["poly_id"],
        scale=SCALE_M,
        geometries=False,
        tileScale=4,
    )

    info = sampled.getInfo()
    rows = []

    for f in info.get("features", []):
        props = f.get("properties", {})
        row = {"poly_id": props.get("poly_id")}
        for col in feature_cols:
            row[col] = props.get(col, np.nan)
        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    model = joblib.load(MODEL_PATH)

    if not hasattr(model, "feature_names_in_"):
        raise RuntimeError(
            "The saved model does not expose feature_names_in_. "
            "Please retrain/save with a DataFrame that has named columns."
        )

    feature_cols = list(model.feature_names_in_)
    print("Model features:", feature_cols)

    gdf = gpd.read_file(INPUT_GEOJSON)
    gdf = drop_unwanted_attributes(gdf)
    gdf = ensure_poly_id(gdf)

    if gdf.empty:
        raise RuntimeError("Input GeoJSON contains no polygons.")

    ee.Initialize(project=GEE_PROJECT_ID)
    composite = build_composite()

    prop_cols = [c for c in gdf.columns if c != "geometry"]

    n = len(gdf)
    n_batches = math.ceil(n / BATCH_SIZE)
    print(f"Polygons: {n}")
    print(f"Batches: {n_batches} (batch size = {BATCH_SIZE})")

    per_batch_frames = []

    for i in range(n_batches):
        start = i * BATCH_SIZE
        end = min((i + 1) * BATCH_SIZE, n)
        chunk = gdf.iloc[start:end].copy()

        print(f"Processing batch {i+1}/{n_batches}  rows {start}:{end}")

        fc = make_ee_feature_collection(chunk, prop_cols=prop_cols)
        px_df = ee_sample_pixels_batch(composite, fc, feature_cols=feature_cols)

        if px_df.empty:
            print("  WARNING: batch returned no sampled pixels.")
            continue

        # Convert model inputs to numeric and drop rows with missing feature values
        X_px = px_df[feature_cols].apply(pd.to_numeric, errors="coerce")
        valid_mask = X_px.notna().all(axis=1)

        px_df = px_df.loc[valid_mask].copy()
        X_px = X_px.loc[valid_mask].copy()

        if px_df.empty:
            print("  WARNING: batch had pixels, but all were invalid after numeric conversion.")
            continue

        # Pixel-level inference
        px_df["pixel_probability"] = model.predict_proba(X_px)[:, 1]
        px_df["pixel_tea"] = (px_df["pixel_probability"] > PIXEL_PROB_THRESHOLD).astype(int)

        # Aggregate to polygon-level metrics
        agg = (
            px_df.groupby("poly_id", as_index=False)
            .agg(
                tea_probability=("pixel_probability", "mean"),
                tea_fraction=("pixel_tea", "mean"),
                n_pixels=("pixel_probability", "size"),
            )
        )

        per_batch_frames.append(agg)

    if not per_batch_frames:
        raise RuntimeError("No pixel samples were extracted from Earth Engine.")

    metrics_df = pd.concat(per_batch_frames, ignore_index=True)

    # If the same poly_id appears in multiple batches, combine again
    metrics_df = (
        metrics_df.groupby("poly_id", as_index=False)
        .agg(
            tea_probability=("tea_probability", "mean"),
            tea_fraction=("tea_fraction", "mean"),
            n_pixels=("n_pixels", "sum"),
        )
    )

    merged = gdf.merge(metrics_df, on="poly_id", how="left")

    # Fill polygons with no valid pixels
    merged["tea_probability"] = merged["tea_probability"].fillna(0.0)
    merged["tea_fraction"] = merged["tea_fraction"].fillna(0.0)
    merged["n_pixels"] = merged["n_pixels"].fillna(0).astype(int)

    # Final binary label
    merged["tea"] = (merged["tea_fraction"] >= TEA_FRACTION_THRESHOLD).astype(int)

    # Save compact metrics CSV for future use
    merged[["poly_id", "tea_probability", "tea_fraction", "n_pixels", "tea"]].to_csv(
        OUTPUT_METRICS_CSV,
        index=False,
    )

    # Save GeoJSON outputs
    merged.to_file(OUTPUT_ALL_GEOJSON, driver="GeoJSON")

    tea1 = merged[merged["tea"] == 1].copy()
    tea0 = merged[merged["tea"] == 0].copy()

    tea1.to_file(OUTPUT_TEA1_GEOJSON, driver="GeoJSON")
    tea0.to_file(OUTPUT_TEA0_GEOJSON, driver="GeoJSON")

    print("\nDone.")
    print(f"Saved: {OUTPUT_ALL_GEOJSON}")
    print(f"Saved: {OUTPUT_TEA1_GEOJSON}  (tea=1: {len(tea1)})")
    print(f"Saved: {OUTPUT_TEA0_GEOJSON}  (tea=0: {len(tea0)})")
    print(f"Saved: {OUTPUT_METRICS_CSV}")


if __name__ == "__main__":
    main()