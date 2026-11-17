"""
Módulo de Treinamento e Avaliação Inicial para DoorDash Delivery ETA.

Implementa:
- Auditoria pré-treino da divisão temporal (garantindo isolamento do teste).
- Pipeline de pré-processamento ajustado exclusivamente no treino.
- Baseline (mediana) e modelos iniciais (Linear Regression, Random Forest,
  HistGradientBoosting) usando apenas features confirmadas.
- Experimentos de ablação separados para features de disponibilidade incerta.
- Métricas completas (MAE, RMSE, MedAE, P90 AE, R², viés e análise por faixa).
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.feature_engineering import CONFIRMED_FEATURES, FORBIDDEN_LEAKAGE_COLS


TARGET_COL = "delivery_duration_minutes"

# store_id (6.743 níveis) é excluído do One-Hot Encoding nesta etapa inicial
# para evitar explosão dimensional; historical_store_order_count e metadados
# de mercado/categoria/protocolo permanecem no conjunto conservador.
CONFIRMED_MODEL_FEATURES: List[str] = [
    c for c in CONFIRMED_FEATURES if c != "store_id"
]

CATEGORICAL_COLS: List[str] = [
    "market_id",
    "store_primary_category",
    "order_protocol",
    "created_time_block",
]

BOOLEAN_ABLATION_COLS: List[str] = ["busy_greater_than_onshift"]


def audit_temporal_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_start: pd.Timestamp,
    test_start: pd.Timestamp,
) -> Dict[str, object]:
    """Audita dimensões, ausência de sobreposição e isolamento temporal."""
    train_idx = set(train_df.index)
    val_idx = set(val_df.index)
    test_idx = set(test_df.index)

    assert len(train_idx & val_idx) == 0, "Sobreposição entre treino e validação!"
    assert len(train_idx & test_idx) == 0, "Sobreposição entre treino e teste!"
    assert len(val_idx & test_idx) == 0, "Sobreposição entre validação e teste!"

    assert train_df["actual_delivery_time"].max() < val_start, "Vazamento temporal: treino -> val!"
    assert val_df["actual_delivery_time"].max() < test_start, "Vazamento temporal: val -> test!"
    assert test_df["created_at"].min() >= test_start, "Teste contém registros anteriores ao corte!"

    return {
        "train_n": len(train_df),
        "val_n": len(val_df),
        "test_n": len(test_df),
        "train_delivery_max": train_df["actual_delivery_time"].max(),
        "val_delivery_max": val_df["actual_delivery_time"].max(),
        "test_created_min": test_df["created_at"].min(),
    }


def prepare_feature_frame(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Padroniza colunas categóricas e booleanas para entrada segura no sklearn."""
    for col in feature_cols:
        if col in FORBIDDEN_LEAKAGE_COLS or col == "actual_delivery_time":
            raise ValueError(f"Coluna proibida detectada nas features: {col}")

    out = df[feature_cols].copy()
    for col in CATEGORICAL_COLS:
        if col in out.columns:
            out[col] = out[col].astype("object").where(out[col].notna(), np.nan).astype(str)
            out.loc[out[col] == "nan", col] = np.nan
            out.loc[out[col] == "<NA>", col] = np.nan

    for col in BOOLEAN_ABLATION_COLS:
        if col in out.columns:
            # Converte boolean nullable (True/False/<NA>) em float (1.0/0.0/NaN)
            out[col] = out[col].astype("float64")

    return out


def build_preprocessor(feature_cols: List[str], scale_numeric: bool = False) -> ColumnTransformer:
    """Cria ColumnTransformer ajustado apenas no treino (sem vazamento global)."""
    cat_cols = [c for c in CATEGORICAL_COLS if c in feature_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps), num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calcula MAE, RMSE, MedAE, P90 do erro absoluto, R² e viés."""
    abs_err = np.abs(y_true - y_pred)
    diff = y_pred - y_true  # positivo = superestimação, negativo = subestimação
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "RMSE": round(float(root_mean_squared_error(y_true, y_pred)), 4),
        "MedAE": round(float(median_absolute_error(y_true, y_pred)), 4),
        "P90_AE": round(float(np.percentile(abs_err, 90)), 4),
        "R2": round(float(r2_score(y_true, y_pred)), 4),
        "Mean_Bias": round(float(np.mean(diff)), 4),
        "Median_Bias": round(float(np.median(diff)), 4),
    }


def evaluate_by_duration_band(y_true: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
    """Avalia erro absoluto por faixa de duração real, destacando > 120 min."""
    bins = [0, 30, 45, 60, 90, 120, np.inf]
    labels = ["<= 30 min", "30-45 min", "45-60 min", "60-90 min", "90-120 min", "> 120 min"]
    df_eval = pd.DataFrame({
        "y_true": y_true.values,
        "y_pred": y_pred,
        "abs_err": np.abs(y_true.values - y_pred),
        "bias": y_pred - y_true.values,
        "band": pd.cut(y_true.values, bins=bins, labels=labels, right=True),
    })
    return df_eval.groupby("band", observed=False).agg(
        n=("y_true", "count"),
        MAE=("abs_err", "mean"),
        MedAE=("abs_err", "median"),
        P90_AE=("abs_err", lambda s: np.percentile(s, 90)),
        Mean_Bias=("bias", "mean"),
    ).round(2)


def train_and_evaluate_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, Pipeline]]:
    """Treina baseline e modelos iniciais nas features confirmadas e avalia na validação."""
    cols = feature_cols if feature_cols is not None else CONFIRMED_MODEL_FEATURES
    X_train = prepare_feature_frame(train_df, cols)
    y_train = train_df[TARGET_COL].values
    X_val = prepare_feature_frame(val_df, cols)
    y_val = val_df[TARGET_COL].values

    models = {
        "Baseline (Median)": Pipeline([
            ("prep", build_preprocessor(cols, scale_numeric=False)),
            ("model", DummyRegressor(strategy="median")),
        ]),
        "Linear Regression": Pipeline([
            ("prep", build_preprocessor(cols, scale_numeric=True)),
            ("model", LinearRegression()),
        ]),
        "Random Forest": Pipeline([
            ("prep", build_preprocessor(cols, scale_numeric=False)),
            ("model", RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_leaf=10,
                n_jobs=-1,
                random_state=42,
            )),
        ]),
        "HistGradientBoosting": Pipeline([
            ("prep", build_preprocessor(cols, scale_numeric=False)),
            ("model", HistGradientBoostingRegressor(
                max_iter=200,
                learning_rate=0.08,
                max_leaf_nodes=63,
                min_samples_leaf=30,
                random_state=42,
            )),
        ]),
    }

    results = []
    preds: Dict[str, np.ndarray] = {}
    fitted: Dict[str, Pipeline] = {}

    for name, pipe in models.items():
        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0
        y_pred = pipe.predict(X_val)
        metrics = compute_regression_metrics(y_val, y_pred)
        metrics["Model"] = name
        metrics["Train_Time_s"] = round(elapsed, 2)
        metrics["N_Features"] = len(cols)
        results.append(metrics)
        preds[name] = y_pred
        fitted[name] = pipe

    results_df = pd.DataFrame(results).sort_values("MAE", ascending=True).reset_index(drop=True)
    cols_order = ["Model", "MAE", "RMSE", "MedAE", "P90_AE", "R2", "Mean_Bias", "Median_Bias", "Train_Time_s", "N_Features"]
    return results_df[cols_order], preds, fitted


def run_ablation_experiments(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Executa experimentos de ablação (EXPERIMENTAIS) adicionando separadamente
    grupos de variáveis de disponibilidade incerta ao HistGradientBoosting.
    """
    ablation_sets = {
        "Conservador (Confirmadas)": CONFIRMED_MODEL_FEATURES,
        "Exp 1 (+ Estimativas Driving/Total)": CONFIRMED_MODEL_FEATURES + [
            "estimated_store_to_consumer_driving_duration",
            "estimated_total_duration",
        ],
        "Exp 2 (+ Operacionais Base)": CONFIRMED_MODEL_FEATURES + [
            "total_onshift_dashers",
            "total_busy_dashers",
            "total_outstanding_orders",
            "dasher_count_difference",
            "busy_greater_than_onshift",
            "is_operational_missing",
        ],
        "Exp 3 (+ Razoes Operacionais)": CONFIRMED_MODEL_FEATURES + [
            "outstanding_orders_per_onshift_dasher",
            "busy_dashers_ratio",
        ],
        "Exp 4 (+ Todas Experimentais)": CONFIRMED_MODEL_FEATURES + [
            "estimated_store_to_consumer_driving_duration",
            "estimated_total_duration",
            "total_onshift_dashers",
            "total_busy_dashers",
            "total_outstanding_orders",
            "dasher_count_difference",
            "busy_greater_than_onshift",
            "is_operational_missing",
            "outstanding_orders_per_onshift_dasher",
            "busy_dashers_ratio",
        ],
    }

    y_train = train_df[TARGET_COL].values
    y_val = val_df[TARGET_COL].values
    rows = []

    for exp_name, cols in ablation_sets.items():
        X_train = prepare_feature_frame(train_df, cols)
        X_val = prepare_feature_frame(val_df, cols)
        pipe = Pipeline([
            ("prep", build_preprocessor(cols, scale_numeric=False)),
            ("model", HistGradientBoostingRegressor(
                max_iter=200,
                learning_rate=0.08,
                max_leaf_nodes=63,
                min_samples_leaf=30,
                random_state=42,
            )),
        ])
        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0
        y_pred = pipe.predict(X_val)
        metrics = compute_regression_metrics(y_val, y_pred)
        metrics["Experimento"] = exp_name
        metrics["Train_Time_s"] = round(elapsed, 2)
        metrics["N_Features"] = len(cols)
        rows.append(metrics)

    res = pd.DataFrame(rows).sort_values("MAE", ascending=True).reset_index(drop=True)
    cols_order = ["Experimento", "MAE", "RMSE", "MedAE", "P90_AE", "R2", "Mean_Bias", "Median_Bias", "Train_Time_s", "N_Features"]
    return res[cols_order]
