# Tea Plantation Identification using AnthroKrishi + Sentinel-2

## Overview

This project aims to identify tea plantations in Dibrugarh (Assam, India) and nearby regions using:

* Google's Agricultural Understanding (AnthroKrishi) API
* Sentinel-2 satellite imagery
* Human-labelled tea/non-tea polygons
* Machine learning

The ultimate goal is:

> Classify every AnthroKrishi farm boundary polygon as either tea plantation or non-tea plantation.

---

# Environment

Python: 3.13 (3.14 mostly works but some GIS libraries may be unstable)

Main libraries:

* geopandas
* pandas
* numpy
* shapely
* scikit-learn
* xgboost
* earthengine-api
* geemap
* pyogrio
* python-dotenv


---

# Project objective

Given AnthroKrishi farm boundaries, determine which polygons correspond to tea plantations.

Desired final output:

```text
poly_id

tea = 1
tea = 0

tea_probability
```

for every AnthroKrishi polygon.

---

# Study region

Primary region:

* Dibrugarh, Assam, India

Nearby regions may also be included.

Jalpaiguri was initially investigated but API coverage was inconsistent.

---

# Data pipeline

estate_centroids_lat_lon.csv

↓

AnthroKrishi API queries

↓

combined_candidate_tea.geojson

↓

deduplication

↓

area filtering

↓

combined_candidate_tea_area_filtered_1000.geojson

↓

Sentinel-2 feature extraction

↓

tea_features_all.csv

---

manual_tea_polygons.geojson

↓

Sentinel-2 feature extraction

↓

manual_tea_features.csv

---

# Chronological history of experiments

## Stage 1: Direct district-level AnthroKrishi usage

Goal:

Use AnthroKrishi directly on Jalpaiguri district inside Earth Engine.

Problems:

* Earth Engine Code Editor cannot use fetch()
* Earth Engine is not intended for arbitrary external API requests

Attempted:

* lookupLandscape
* lookupLandscapeCaptureDate

Result:

Capture dates worked.

District-level retrieval was unreliable.

Jalpaiguri returned:

```text
404 NOT_FOUND
```

Conclusion:

Direct district-level workflow was abandoned.

---

## Stage 2: Estate centroid workflow

Input:

estate_centroids_lat_lon.csv

Contains:

Tea estate centroids.

Idea:

Use each centroid to query AnthroKrishi.

Problem:

Estate centroids are not guaranteed to lie inside tea fields.

They may lie on:

* roads
* offices
* housing
* ponds
* empty spaces

Conclusion:

Centroid-based labels are unreliable.

---

## Stage 3: Buffer sampling

Goal:

Reduce centroid noise.

Method:

For each centroid:

* create buffer
* generate sample points
* query AnthroKrishi

Result:

combined_candidate_tea.geojson

~119,191 polygons.

Label strategy:

candidate_tea = 1

if polygon originated from tea estate buffer.

candidate_tea = 0

otherwise.

These are weak labels, NOT ground truth.

---

## Stage 4: Deduplication

Problem:

API queries returned many duplicate polygons.

Solution:

Geometry-based deduplication.

Methods:

* geometry normalization
* WKB representation
* drop_duplicates()

Outcome:

Duplicates removed.

---

## Stage 5: Area analysis

Problem:

Many polygons were extremely tiny.

Area statistics:

```text
min      ~15 m²
1%       ~43 m²
5%       ~69 m²
10%      ~93 m²
25%      ~164 m²
50%      ~350 m²
75%      ~812 m²
90%      ~1944 m²
95%      ~3663 m²
99%      ~14535 m²
max      ~470746 m²
```

Multiple thresholds were tested.

1000 m² was chosen.

Reason:

Visual inspection showed it removed noisy tiny polygons while preserving useful regions.

Output:

combined_candidate_tea_area_filtered_1000.geojson

Statistics:

24378 polygons.

candidate_tea:

```text
1 -> 15348
0 -> 9030
```

This is the main inference dataset.

---

## Stage 6: Manual ground truth creation

Centroid labels were abandoned.

Manual polygons were created.

File:

manual_tea_polygons.geojson

Properties:

141 polygons.

Approximately:

110 tea

31 non-tea

Feature extraction later showed:

104 tea

37 non-tea

These are the ONLY trusted labels.

These polygons are manually drawn.

They are NOT AnthroKrishi polygons.

They are spread across Dibrugarh and nearby regions.

---

## Stage 7: Sentinel-2 feature extraction

Method:

Earth Engine + geemap.

Features extracted:

```text
B2
B3
B4

B8

B11
B12

NDVI
NDWI
EVI
```

Outputs:

manual_tea_features.csv

tea_features_all.csv

manual_tea_features.csv:

141 rows

tea_features_all.csv:

24378 rows

---

## Stage 8: First machine learning attempt

Train:

manual_tea_features.csv

Predict:

tea_features_all.csv

Models tested:

* Logistic Regression
* Random Forest
* XGBoost

Results:

```text
logistic

F1=0.918

ROC_AUC=0.960

----------------

random_forest

F1=0.976

ROC_AUC=0.996

----------------

xgboost

F1=0.976

ROC_AUC=0.992
```

Random Forest selected.

---

## Stage 9: Prediction failure

Prediction output:

```text
tea = 1   -> 21104

tea = -1  -> 3102

tea = 0   -> 172
```

Probability distribution:

```text
median = 0.974

75% = 1.000
```

Model became overconfident.

Conclusion:

Training distribution and inference distribution are different.

---

## Root cause discovered

Domain mismatch.

Training polygons:

manual polygons

Inference polygons:

AnthroKrishi farm boundaries

Manual polygons:

* large
* homogeneous

AnthroKrishi polygons:

* small
* irregular
* fragmented

The model learned manual drawing characteristics instead of tea characteristics.

---

## Stage 10: Manual → AnthroKrishi overlap transfer

Idea:

Use manual labels to label AnthroKrishi polygons.

Procedure:

Compute overlap.

Output:

anthro_labeled.geojson

Result:

```text
tea = 4363

non-tea = 35

unlabelled = 19980
```

Problem:

Extreme imbalance.

4363:35

Conclusion:

Not suitable for training.

---

# Important conclusions

DO NOT repeat these failed approaches.

FAILED:

1. Centroid -> tea label

Reason:

Centroid does not imply tea.

---

FAILED:

2. Manual polygons -> directly train -> AnthroKrishi prediction

Reason:

Domain mismatch.

---

FAILED:

3. Overlap propagation

Reason:

Severe imbalance.

4363 tea

35 non-tea

---

# Current status

Available datasets:

Trusted:

manual_tea_polygons.geojson

manual_tea_features.csv

---

Inference dataset:

combined_candidate_tea_area_filtered_1000.geojson

tea_features_all.csv

---

Experimental:

anthro_labeled.geojson

Do NOT currently use for training.

---

# File descriptions

estate_centroids_lat_lon.csv

Purpose:

Tea estate centroid coordinates.

---

combined_candidate_tea.geojson

Purpose:

Raw AnthroKrishi output.

~119k polygons.

---

combined_candidate_tea_area_filtered_1000.geojson

Purpose:

Primary AnthroKrishi dataset.

24378 polygons.

Area >=1000 m².

---

manual_tea_polygons.geojson

Purpose:

Human-labelled polygons.

141 polygons.

tea=True/False

---

manual_tea_features.csv

Purpose:

Sentinel-2 features for manual polygons.

Training dataset.

---

tea_features_all.csv

Purpose:

Sentinel-2 features for all AnthroKrishi polygons.

Inference dataset.

---

anthro_labeled.geojson

Purpose:

Experimental overlap labels.

Do not currently use for training.

---

tea_predictions.csv

Purpose:

Output of failed first ML attempt.

Not trustworthy.

---

# Recommended future directions

Priority 1:

Active learning.

Procedure:

1. Train on manual polygons.
2. Predict AnthroKrishi polygons.
3. Select uncertain polygons.
4. Manually label them.
5. Retrain.
6. Repeat.

Priority 2:

Add additional features:

* area
* perimeter
* compactness
* texture features
* DEM

Priority 3:

Increase non-tea labels.

Especially:

* dense forest
* orchards
* shrubland
* agricultural fields

Avoid obvious negatives.

---

# Notes for future

This project has already explored many dead ends.

Do NOT restart from centroid-based labeling.

Do NOT directly train manual polygons and predict AnthroKrishi polygons without addressing domain mismatch.

The core challenge is creating reliable labels for AnthroKrishi farm boundaries.
