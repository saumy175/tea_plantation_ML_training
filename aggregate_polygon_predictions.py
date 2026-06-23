#!/usr/bin/env python3
import os
from pathlib import Path

import joblib
import geopandas as gpd
import numpy as np
import pandas as pd

# =============================================================================
# CONFIG
# =============================================================================

MODEL_PATH = "tea_xgb_pixel_model.pkl"
INPUT_GEOJSON = "combined_candidate_tea_area_filtered_1000.geojson"

# Put all exported CSVs from Drive into this folder
PIXEL_CSV_DIR = "tea_polygon_pixel_batches"

OUTPUT_ALL_GEOJSON = "polygons_with_tea_probability.geojson"
OUTPUT_TEA1_GEOJSON = "polygons_tea_1.geojson"
OUTPUT_TEA0_GEOJSON = "polygons_tea_0.geojson"
OUTPUT_METRICS_CSV = "polygons_tea_metrics.csv"

PIXEL_PROB_THRESHOLD = 0.95
TEA_FRACTION_THRESHOLD = 0.20   # conservative starting point

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

    feature_cols = [str(x) for x in model.feature_names_in_]
    print("Model features:", feature_cols)

    pixel_dir = Path(PIXEL_CSV_DIR)
    csv_files = sorted(pixel_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {PIXEL_CSV_DIR}")

    print(f"Found {len(csv_files)} pixel CSV files")

    agg_frames = []

    for i, csv_path in enumerate(csv_files, start=1):
        print(f"[{i}/{len(csv_files)}] Reading {csv_path.name}")

        df = pd.read_csv(csv_path)

        # Drop junk columns if present
        drop_cols = [c for c in ["system:index", ".geo"] if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)

        if "poly_id" not in df.columns:
            raise ValueError(f"poly_id missing in {csv_path.name}")

        # Keep only required model columns
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{csv_path.name} is missing model features: {missing}")

        X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
        valid_mask = X.notna().all(axis=1)

        df = df.loc[valid_mask].copy()
        X = X.loc[valid_mask].copy()

        if df.empty:
            print("  WARNING: no valid pixels after numeric conversion")
            continue

        # Pixel-level prediction
        df["pixel_probability"] = model.predict_proba(X)[:, 1]
        df["pixel_tea"] = (df["pixel_probability"] > PIXEL_PROB_THRESHOLD).astype(int)

        # Aggregate to polygon level
        agg = (
            df.groupby("poly_id", as_index=False)
            .agg(
                tea_probability=("pixel_probability", "mean"),
                tea_fraction=("pixel_tea", "mean"),
                n_pixels=("pixel_probability", "size"),
            )
        )

        agg_frames.append(agg)

    if not agg_frames:
        raise RuntimeError("No usable pixels found in the CSV batches.")

    metrics_df = pd.concat(agg_frames, ignore_index=True)

    # If poly_id appears across multiple batch files, combine again
    metrics_df = (
        metrics_df.groupby("poly_id", as_index=False)
        .agg(
            tea_probability=("tea_probability", "mean"),
            tea_fraction=("tea_fraction", "mean"),
            n_pixels=("n_pixels", "sum"),
        )
    )

    metrics_df.to_csv(OUTPUT_METRICS_CSV, index=False)

    # Merge back to polygons
    gdf = gpd.read_file(INPUT_GEOJSON)
    gdf = drop_unwanted_attributes(gdf)
    gdf = ensure_poly_id(gdf)

    merged = gdf.merge(metrics_df, on="poly_id", how="left")

    # Fill polygons that got no valid pixels
    merged["tea_probability"] = merged["tea_probability"].fillna(0.0)
    merged["tea_fraction"] = merged["tea_fraction"].fillna(0.0)
    merged["n_pixels"] = merged["n_pixels"].fillna(0).astype(int)

    # Final label
    merged["tea"] = (merged["tea_fraction"] >= TEA_FRACTION_THRESHOLD).astype(int)

    # Save outputs
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