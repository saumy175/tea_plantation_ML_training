import geopandas as gpd
import pandas as pd

RANDOM_STATE = 42

geo = gpd.read_file(
    "anthro_labeled.geojson"
)

features = pd.read_csv(
    "tea_features_all.csv"
)

# Merge labels onto features
df = features.merge(
    geo[["poly_id", "tea_label"]],
    on="poly_id",
    how="inner"
)

tea = df[df["tea_label"] == 1]

nontea = df[df["tea_label"] == 0]

print("Tea:", len(tea))
print("Non-tea:", len(nontea))

# Downsample tea
tea_sample = tea.sample(
    n=min(len(tea), 10 * len(nontea)),
    random_state=RANDOM_STATE,
)

train = pd.concat([
    tea_sample,
    nontea
])

train = train.sample(
    frac=1,
    random_state=RANDOM_STATE
)

train.to_csv(
    "anthro_training.csv",
    index=False
)

print()

print(train["tea_label"].value_counts())

print()

print("Saved anthro_training.csv")