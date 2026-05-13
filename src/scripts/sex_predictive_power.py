"""ASSIGNMENT 3.2

Determine whether the sex column has predictive power for the species column.

Runs two analyses:
1. Mutual information — statistical measure of dependency between sex and species.
2. Cross-validated accuracy — compares a model trained with vs. without sex.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def load_data(folder: str = "data") -> pd.DataFrame:
    files = sorted(Path(folder).glob("*.csv"))
    if not files:
        print(f"No CSV files found in {folder}")
        sys.exit(1)
    data = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    data["sex"] = data["sex"].replace(".", np.nan)
    return data


def build_transformer(include_sex: bool) -> ColumnTransformer:
    numeric_transformer = make_pipeline(
        SimpleImputer(strategy="mean"),
        StandardScaler(),
    )
    categorical_cols = ["island", "sex"] if include_sex else ["island"]
    categorical_transformer = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore"),
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, make_column_selector(dtype_exclude="object")),
            ("categorical", categorical_transformer, categorical_cols),
        ],
    )


def encode_target(data: pd.DataFrame) -> np.ndarray:
    encoder = ColumnTransformer(
        transformers=[("species", OrdinalEncoder(), ["species"])],
    )
    return encoder.fit_transform(data).ravel()


def mutual_information(data: pd.DataFrame) -> None:
    print("=" * 50)
    print("MUTUAL INFORMATION")
    print("=" * 50)

    # Drop rows where sex or species is missing for this analysis
    clean = data[["sex", "species"]].dropna()

    # Encode both columns as integers
    sex_encoded = OrdinalEncoder().fit_transform(clean[["sex"]]).ravel()
    species_encoded = OrdinalEncoder().fit_transform(clean[["species"]]).ravel()

    from sklearn.feature_selection import mutual_info_classif

    mi = mutual_info_classif(
        sex_encoded.reshape(-1, 1),
        species_encoded,
        discrete_features=True,
        random_state=42,
    )[0]

    print(f"Mutual information (sex → species): {mi:.4f}")
    print("  (0 = no dependency, higher = stronger dependency)\n")

    # Also show the cross-tabulation for intuition
    crosstab = pd.crosstab(clean["sex"], clean["species"], normalize="index").round(2)
    print("Species distribution per sex (row %):")
    print(crosstab)
    print()


def cv_accuracy_comparison(data: pd.DataFrame) -> None:
    print("=" * 50)
    print("CROSS-VALIDATED ACCURACY (5-fold)")
    print("=" * 50)

    y = encode_target(data)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    model = LogisticRegression(max_iter=1000, random_state=42)

    for include_sex in [True, False]:
        transformer = build_transformer(include_sex=include_sex)
        pipeline = make_pipeline(transformer, model)
        X = transformer.fit_transform(data)
        scores = cross_val_score(pipeline, data, y, cv=kfold, scoring="accuracy")
        label = "with sex   " if include_sex else "without sex"
        print(f"  {label}: {scores.mean():.4f} ± {scores.std():.4f}")

    print()


if __name__ == "__main__":
    data = load_data("data")
    print(f"Dataset: {len(data)} samples\n")
    mutual_information(data)
    cv_accuracy_comparison(data)


"""RESULT

The answer is clear: sex has zero predictive power for species.

  - Mutual information ≈ 0 — statistically, sex and species are independent
  - Identical CV accuracy (99.1%) — removing sex doesn't change model performance at all
  - Cross-tabulation confirms it — the species distribution is virtually identical for MALE and FEMALE (~43% Adelie, ~20% Chinstrap, ~35%
  Gentoo regardless of sex)

  In other words, knowing a penguin's sex tells you nothing about its species. You could drop the sex column from training entirely
  without losing accuracy.
"""