"""Cleaning and feature pipeline shared by training, evaluation and the API.

Keeping it in one place means the API transforms a customer exactly the way
the training data was transformed (no "training / serving skew").
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

TARGET = "Churn"
ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "num_addons"]
BINARY = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
          "PaperlessBilling"] + ADDONS
MULTI = ["InternetService", "Contract", "PaymentMethod"]
RAW_INPUTS = ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService", "MultipleLines",
              "InternetService"] + ADDONS + ["Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges",
                                              "TotalCharges"]

PAYMENT_FIX = {"electronic check": "Electronic check", "mailed check": "Mailed check",
               "bank transfer (automatic)": "Bank transfer (automatic)",
               "credit card (automatic)": "Credit card (automatic)"}


def clean(df: pd.DataFrame, dedupe: bool = True) -> pd.DataFrame:
    """Fix the raw export: types, blanks, duplicates, inconsistent labels.

    dedupe=False at prediction time: two identical customers must both get a score.
    """
    df = df.copy()
    if dedupe:
        if "customerID" in df:
            df = df.drop_duplicates(subset="customerID")
        df = df.drop_duplicates()

    # TotalCharges is text with blanks for brand-new customers -> number, 0 when never billed
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.loc[df["TotalCharges"].isna() & (df["tenure"] == 0), "TotalCharges"] = 0.0
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"])

    df["PaymentMethod"] = df["PaymentMethod"].str.strip().str.lower().map(PAYMENT_FIX).fillna(df["PaymentMethod"])
    if TARGET in df and not pd.api.types.is_numeric_dtype(df[TARGET]):
        df[TARGET] = (df[TARGET].astype(str).str.strip() == "Yes").astype(int)
    return df.reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Simple engineered features. Works on one row (API) or many."""
    df = df.copy()
    # "No internet service" is already captured by InternetService == "No"; keep add-ons as plain Yes/No
    for col in ADDONS:
        df[col] = df[col].replace("No internet service", "No")
    df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")
    df["SeniorCitizen"] = df["SeniorCitizen"].map({1: "Yes", 0: "No", "1": "Yes", "0": "No"}).fillna(df["SeniorCitizen"])
    df["num_addons"] = (df[ADDONS] == "Yes").sum(axis=1)
    return df


def build_preprocessor() -> ColumnTransformer:
    """Scale numbers, one-hot encode categories.

    drop="first" leaves one reference level per category, so each logistic
    regression coefficient reads as "compared with the reference level".
    Unseen categories at prediction time become all-zeros (= the reference).
    """
    numeric = Pipeline([
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),  # charges are skewed
        ("scale", StandardScaler()),
    ])
    categorical = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer([
        ("num", numeric, NUMERIC),
        ("cat", categorical, BINARY + MULTI),
    ], verbose_feature_names_out=False)


def prepare(df: pd.DataFrame, dedupe: bool = True):
    """clean + add_features, then split into X and y (y is None if there is no label)."""
    df = add_features(clean(df, dedupe=dedupe))
    y = df[TARGET].values if TARGET in df else None
    return df[NUMERIC + BINARY + MULTI], y
