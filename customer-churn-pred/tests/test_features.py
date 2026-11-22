import numpy as np

from features import NUMERIC, add_features, build_preprocessor, clean, prepare


def test_clean_fixes_raw_export(raw_df):
    df = clean(raw_df)
    assert df["TotalCharges"].dtype.kind == "f" and df["TotalCharges"].notna().all()
    assert (df.loc[df.tenure == 0, "TotalCharges"] == 0).all()   # blanks become 0 for new customers
    assert df["customerID"].is_unique                              # duplicates removed
    assert df["PaymentMethod"].nunique() == 4                      # casing / spaces fixed
    assert set(df["Churn"].unique()) <= {0, 1}


def test_add_features_counts_addons(raw_df):
    df = add_features(clean(raw_df))
    assert "No internet service" not in set(df["TechSupport"])
    assert df["num_addons"].between(0, 6).all()


def test_preprocessor_is_finite_and_handles_unknown(raw_df):
    X, y = prepare(raw_df)
    prep = build_preprocessor().fit(X)
    Z = prep.transform(X)
    assert np.isfinite(Z).all() and Z.shape[0] == len(y)
    odd = X.head(1).copy()
    odd["PaymentMethod"] = "Crypto"                               # unseen category must not crash
    assert np.isfinite(prep.transform(odd)).all()
    assert all(c in X for c in NUMERIC)
