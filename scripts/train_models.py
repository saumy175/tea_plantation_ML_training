import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

# ==================
# Load data
# ==================

df = pd.read_csv("manual_tea_features.csv")

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

X = df[feature_cols]

y = df["tea"].astype(int)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

models = {

    "logistic": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    ),

    "random_forest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
    ),

    "xgboost": XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
}

results = {}

for name, model in models.items():

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model),
    ])

    scores = cross_validate(
        pipe,
        X,
        y,
        cv=cv,
        scoring={
            "f1": "f1",
            "roc_auc": "roc_auc",
        },
    )

    f1 = scores["test_f1"].mean()

    auc = scores["test_roc_auc"].mean()

    results[name] = auc

    print(
        f"{name:15s}"
        f" F1={f1:.3f}"
        f" ROC_AUC={auc:.3f}"
    )

best = max(results, key=results.get)

print()
print("Best model:", best)