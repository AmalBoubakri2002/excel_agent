import pandas as pd
import numpy as np
from sklearn.linear_model   import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics         import mean_squared_error, r2_score, mean_absolute_error


def run_regression_analysis(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
) -> dict:
    """
    Entraîne une régression linéaire et évalue ses performances.
    """
    if not target_col or not feature_cols:
        return {"error": "Target ou features non définis"}

    # Vérifier que les colonnes existent
    missing = [c for c in [target_col] + feature_cols if c not in df.columns]
    if missing:
        return {"error": f"Colonnes manquantes : {missing}"}

    # Préparer X et y
    X = df[feature_cols].select_dtypes(include=[np.number])
    y = df[target_col]

    # Supprimer les lignes avec nulls
    mask = X.notna().all(axis=1) & y.notna()
    X, y = X[mask], y[mask]

    if len(X) < 10:
        return {"error": f"Pas assez de données : {len(X)} lignes (min 10)"}

    # Split train/test 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Entraînement
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prédictions
    y_pred = model.predict(X_test)

    # Métriques
    r2  = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Coefficients
    coefficients = {
        col: round(float(coef), 4)
        for col, coef in zip(X.columns, model.coef_)
    }

    return {
        "r2_score"      : round(float(r2), 4),
        "rmse"          : round(float(np.sqrt(mse)), 4),
        "mae"           : round(float(mae), 4),
        "intercept"     : round(float(model.intercept_), 4),
        "coefficients"  : coefficients,
        "n_train"       : len(X_train),
        "n_test"        : len(X_test),
        "feature_cols"  : list(X.columns),
        "target_col"    : target_col,
        "interpretation": (
            f"Le modèle explique {r2*100:.1f}% de la variance de '{target_col}'"
        ),
    }