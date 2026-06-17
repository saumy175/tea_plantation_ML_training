import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# =====================
# Load datasets
# =====================

train_df = pd.read_csv("manual_tea_features.csv")

all_df = pd.read_csv("tea_features_all.csv")

feature_cols = [
    "B2",
    "B3",
    "B4",
    "B8",
    "B11",
    "B12",
    "NDVI",
    "NDWI",
    "EVI",
]

# =====================
# Training data
# =====================

X_train = train_df[feature_cols]

y_train = train_df["tea"].astype(int)

# =====================
# Build model
# =====================

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),

    ("rf", RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )),
])

model.fit(
    X_train,
    y_train
)

# =====================
# Predict all polygons
# =====================

X_all = all_df[feature_cols]

all_df["tea_probability"] = (
    model.predict_proba(X_all)[:, 1]
)

# Conservative thresholding
def classify(p):

    if p >= 0.85:
        return 1

    if p <= 0.15:
        return 0

    return -1


all_df["predicted_tea"] = (
    all_df["tea_probability"]
    .apply(classify)
)

# =====================
# Save
# =====================

all_df.to_csv(
    "tea_predictions.csv",
    index=False
)

print(
    all_df["predicted_tea"]
    .value_counts()
)

print("Saved tea_predictions.csv")