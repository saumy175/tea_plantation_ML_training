import ee
import geopandas as gpd
import pandas as pd

from config import (
    GOOGLE_AGRI_API_KEY,
    GEE_PROJECT_ID
)

# ==========================
# CONFIG
# ==========================


PROJECT_ID = GEE_PROJECT_ID

INPUT_GEOJSON = "manual_tea_polygons.geojson"

OUTPUT_CSV = "manual_tea_features.csv"

# ==========================
# INIT
# ==========================

ee.Initialize(project=PROJECT_ID)

gdf = gpd.read_file(INPUT_GEOJSON)

print(f"Polygons: {len(gdf)}")

# Add poly_id if missing
if "poly_id" not in gdf.columns:
    gdf["poly_id"] = range(len(gdf))

# ==========================
# Convert to EE FeatureCollection
# ==========================

features = []

for _, row in gdf.iterrows():

    geom = ee.Geometry(
        row.geometry.__geo_interface__
    )

    props = {
        "poly_id": int(row["poly_id"]),
        "tea": int(bool(row["tea"]))
    }

    features.append(
        ee.Feature(
            geom,
            props
        )
    )

fc = ee.FeatureCollection(features)

# ==========================
# Sentinel-2
# ==========================

s2 = (
    ee.ImageCollection(
        "COPERNICUS/S2_SR_HARMONIZED"
    )
    .filterDate(
        "2023-01-01",
        "2025-01-01"
    )
    .filter(
        ee.Filter.lt(
            "CLOUDY_PIXEL_PERCENTAGE",
            20
        )
    )
)

def mask_s2(image):

    qa = image.select("QA60")

    cloud = 1 << 10
    cirrus = 1 << 11

    mask = (
        qa.bitwiseAnd(cloud)
        .eq(0)
        .And(
            qa.bitwiseAnd(cirrus)
            .eq(0)
        )
    )

    return (
        image
        .updateMask(mask)
        .divide(10000)
    )

image = (
    s2
    .map(mask_s2)
    .median()
)

# ==========================
# Vegetation indices
# ==========================

ndvi = image.normalizedDifference(
    ["B8", "B4"]
).rename("NDVI")

ndwi = image.normalizedDifference(
    ["B3", "B8"]
).rename("NDWI")

evi = image.expression(
    "2.5*((nir-red)/(nir+6*red-7.5*blue+1))",
    {
        "nir": image.select("B8"),
        "red": image.select("B4"),
        "blue": image.select("B2"),
    },
).rename("EVI")

img = (
    image
    .select([
        "B2",
        "B3",
        "B4",
        "B8",
        "B11",
        "B12",
    ])
    .addBands([
        ndvi,
        ndwi,
        evi,
    ])
)

# ==========================
# Reduce regions
# ==========================

out = img.reduceRegions(
    collection=fc,
    reducer=ee.Reducer.mean(),
    scale=10,
    tileScale=4,
)

task = ee.batch.Export.table.toDrive(
    collection=out,
    description="manual_tea_features",
    folder="tea_project",
    fileFormat="CSV",
)

task.start()

print("Export task started.")
print("Go to:")
print("https://code.earthengine.google.com/tasks")


print("Saved:", OUTPUT_CSV)