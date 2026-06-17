import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import geopandas as gpd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from shapely.geometry import Point
from shapely.ops import transform
from pyproj import Transformer

try:
    from shapely.validation import make_valid
except Exception:
    make_valid = None


from config import (
    GOOGLE_AGRI_API_KEY,
    GEE_PROJECT_ID
)

# =========================
# CONFIG
# =========================


INPUT_CSV = "estate_centroids_lat_lon.csv"
OUTPUT_GEOJSON = "combined_candidate_tea_buffer500.geojson"
FAILURE_CSV = "api_failures_buffer500.csv"
BUFFER_GEOJSON = "estate_buffers_buffer500.geojson"

BASE_URL = "https://agriculturalunderstanding.googleapis.com/v1:lookupLandscape"

BUFFER_M = 500
GRID_SPACING_M = 250
MAX_WORKERS = 4
REQUEST_TIMEOUT = 60
RETRIES = 3
BACKOFF_FACTOR = 1.5


# =========================
# HELPERS
# =========================
def pick_column(columns, candidates):
    cols = {c.lower().strip(): c for c in columns}
    for cand in candidates:
        if cand in cols:
            return cols[cand]
    return None


def utm_epsg_for_lonlat(lon, lat):
    zone = int((lon + 180) // 6) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone


def repair_geometry(geom):
    if geom is None or geom.is_empty:
        return geom
    if geom.is_valid:
        return geom
    if make_valid is not None:
        try:
            fixed = make_valid(geom)
            if fixed is not None and not fixed.is_empty:
                return fixed
        except Exception:
            pass
    try:
        return geom.buffer(0)
    except Exception:
        return geom


def generate_buffer_and_points(lat, lon, buffer_m=500, spacing_m=250):
    """
    Returns:
      buffer_polygon_wgs84, sample_points_wgs84(list of (lat, lon))
    """
    src_epsg = 4326
    dst_epsg = utm_epsg_for_lonlat(lon, lat)

    fwd = Transformer.from_crs(src_epsg, dst_epsg, always_xy=True)
    inv = Transformer.from_crs(dst_epsg, src_epsg, always_xy=True)

    x, y = fwd.transform(lon, lat)
    center = Point(x, y)

    buffer_poly_proj = center.buffer(buffer_m)

    # Small grid of sample points within the buffer
    offsets = [-spacing_m, 0, spacing_m]
    sample_points = []
    seen = set()

    for dx in offsets:
        for dy in offsets:
            p = Point(x + dx, y + dy)
            if p.distance(center) <= buffer_m:
                lon2, lat2 = inv.transform(p.x, p.y)
                key = (round(lat2, 7), round(lon2, 7))
                if key not in seen:
                    seen.add(key)
                    sample_points.append((lat2, lon2))

    buffer_poly_wgs84 = transform(lambda x, y: inv.transform(x, y), buffer_poly_proj)
    return buffer_poly_wgs84, sample_points


def query_landscape(session, estate, lat, lon, buffer_id):
    url = f"{BASE_URL}?key={GOOGLE_AGRI_API_KEY}"
    payload = {
        "locationSpecifier": {
            "coordinates": {
                "latitude": float(lat),
                "longitude": float(lon),
            }
        }
    }

    try:
        r = session.post(url, json=payload, timeout=REQUEST_TIMEOUT)

        if r.status_code != 200:
            return [], {
                "Estate": estate,
                "Latitude": lat,
                "Longitude": lon,
                "BufferID": buffer_id,
                "Status": r.status_code,
                "Message": r.text[:1000],
            }

        data = r.json()
        geojson_text = data["landscape"]["geojson"]
        geojson = json.loads(geojson_text)

        features = geojson.get("features", [])
        for feat in features:
            props = feat.setdefault("properties", {})
            props["source_estate"] = estate
            props["source_latitude"] = float(lat)
            props["source_longitude"] = float(lon)
            props["source_buffer_id"] = buffer_id

        return features, None

    except Exception as e:
        return [], {
            "Estate": estate,
            "Latitude": lat,
            "Longitude": lon,
            "BufferID": buffer_id,
            "Status": "EXCEPTION",
            "Message": str(e),
        }


# =========================
# MAIN
# =========================
def main():
    df = pd.read_csv(INPUT_CSV)

    lat_col = pick_column(df.columns, ["latitude", "lat", "y"])
    lon_col = pick_column(df.columns, ["longitude", "lon", "lng", "long", "x"])
    name_col = pick_column(df.columns, ["estate", "name", "tea_estate", "site", "location"])

    if lat_col is None or lon_col is None:
        raise ValueError(f"Could not find latitude/longitude columns in {INPUT_CSV}")

    if name_col is None:
        df["_estate_name"] = [f"estate_{i+1}" for i in range(len(df))]
        name_col = "_estate_name"

    df = df[[name_col, lat_col, lon_col]].copy()
    df = df.dropna(subset=[lat_col, lon_col]).copy()
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df = df.dropna(subset=[lat_col, lon_col]).copy()
    df = df.reset_index(drop=True)

    if df.empty:
        raise RuntimeError("No valid coordinates found in the CSV.")

    buffers = []
    sample_records = []

    for i, row in df.iterrows():
        estate = str(row[name_col])
        lat = float(row[lat_col])
        lon = float(row[lon_col])

        buffer_poly, sample_points = generate_buffer_and_points(
            lat=lat,
            lon=lon,
            buffer_m=BUFFER_M,
            spacing_m=GRID_SPACING_M,
        )

        buffer_id = f"{i+1:04d}"
        buffers.append({
            "buffer_id": buffer_id,
            "Estate": estate,
            "geometry": buffer_poly,
        })

        for s_lat, s_lon in sample_points:
            sample_records.append({
                "Estate": estate,
                "buffer_id": buffer_id,
                "Latitude": s_lat,
                "Longitude": s_lon,
            })

    buffers_gdf = gpd.GeoDataFrame(buffers, crs="EPSG:4326")
    buffers_gdf.to_file(BUFFER_GEOJSON, driver="GeoJSON")

    session = requests.Session()
    retry = Retry(
        total=RETRIES,
        connect=RETRIES,
        read=RETRIES,
        status=RETRIES,
        backoff_factor=BACKOFF_FACTOR,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["POST"]),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    all_features = []
    failures = []

    print(f"Loaded {len(df)} estates")
    print(f"Generated {len(sample_records)} sample points")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {
            executor.submit(
                query_landscape,
                session,
                rec["Estate"],
                rec["Latitude"],
                rec["Longitude"],
                rec["buffer_id"],
            ): rec
            for rec in sample_records
        }

        done = 0
        for future in as_completed(future_map):
            done += 1
            rec = future_map[future]
            try:
                features, failure = future.result()
                if failure is not None:
                    failures.append(failure)
                    print(f"[{done}/{len(sample_records)}] {rec['Estate']} -> FAILED")
                else:
                    all_features.extend(features)
                    print(f"[{done}/{len(sample_records)}] {rec['Estate']} -> {len(features)} features")
            except Exception as e:
                failures.append({
                    "Estate": rec["Estate"],
                    "Latitude": rec["Latitude"],
                    "Longitude": rec["Longitude"],
                    "BufferID": rec["buffer_id"],
                    "Status": "EXCEPTION",
                    "Message": str(e),
                })
                print(f"[{done}/{len(sample_records)}] {rec['Estate']} -> EXCEPTION")

    if not all_features:
        raise RuntimeError("No features were returned from the API.")

    gdf = gpd.GeoDataFrame.from_features(all_features, crs="EPSG:4326")
    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()

    gdf["geometry"] = gdf["geometry"].apply(repair_geometry)
    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    try:
        gdf["geometry"] = gdf.geometry.normalize()
    except Exception:
        pass

    gdf["geom_wkb"] = gdf.geometry.to_wkb()
    gdf = gdf.drop_duplicates(subset=["geom_wkb"]).copy()
    gdf = gdf.drop(columns=["geom_wkb"]).reset_index(drop=True)

    gdf["poly_id"] = gdf.index
    gdf["candidate_tea"] = 0
    gdf["tea"] = -1

    matches = gpd.sjoin(
        gdf[["poly_id", "geometry"]],
        buffers_gdf[["buffer_id", "geometry"]],
        how="inner",
        predicate="intersects",
    )

    candidate_ids = set(matches["poly_id"].unique())
    gdf.loc[gdf["poly_id"].isin(candidate_ids), "candidate_tea"] = 1
    gdf.loc[gdf["poly_id"].isin(candidate_ids), "tea"] = 1

    gdf = gdf.drop(columns=["poly_id"])
    gdf.to_file(OUTPUT_GEOJSON, driver="GeoJSON")
    pd.DataFrame(failures).to_csv(FAILURE_CSV, index=False)

    print("\nDone.")
    print(f"Combined polygons: {len(gdf)}")
    print(f"candidate_tea = 1: {(gdf['candidate_tea'] == 1).sum()}")
    print(f"tea = 1: {(gdf['tea'] == 1).sum()}")
    print(f"tea = -1: {(gdf['tea'] == -1).sum()}")
    print(f"Saved polygons: {OUTPUT_GEOJSON}")
    print(f"Saved buffers: {BUFFER_GEOJSON}")
    print(f"Saved failures: {FAILURE_CSV}")


if __name__ == "__main__":
    main()