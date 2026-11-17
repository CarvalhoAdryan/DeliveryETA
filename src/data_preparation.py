"""
Módulo de preparação controlada dos dados para o projeto DoorDash Delivery ETA.

Responsável por carregar o dataset bruto de forma somente leitura, gerar as
cópias de trabalho (`df_raw`, `df_analysis`, `df_model`), padronizar tipos,
derivar a variável-alvo, criar indicadores de validade, tratar valores negativos
nas features especificadas (convertendo-os em NaN apenas em `df_model`) e
gerar atributos temporais e operacionais iniciais.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


NEGATIVE_TO_NAN_COLS = (
    "min_item_price",
    "total_onshift_dashers",
    "total_busy_dashers",
    "total_outstanding_orders",
)

CATEGORICAL_ID_COLS = ("market_id", "order_protocol")


@dataclass
class PreparationSummary:
    """Resumo estruturado das transformações aplicadas na preparação."""

    raw_shape: Tuple[int, int]
    analysis_shape: Tuple[int, int]
    model_shape: Tuple[int, int]
    target_counts: Dict[str, int]
    removed_rows: int
    removed_reasons: Dict[str, int]
    negatives_converted_to_nan: Dict[str, int]


def load_raw_data(filepath: str | Path) -> pd.DataFrame:
    """Carrega o arquivo CSV original sem modificá-lo."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {path}")
    return pd.read_csv(path)


def _convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """Converte colunas temporais e categóricas preservando identificadores."""
    out = df.copy()
    out["created_at"] = pd.to_datetime(out["created_at"], errors="coerce")
    out["actual_delivery_time"] = pd.to_datetime(out["actual_delivery_time"], errors="coerce")

    for col in CATEGORICAL_ID_COLS:
        if col in out.columns:
            # Converte para Int64 antes de category para evitar sufixo '.0' e preservar NaN
            out[col] = out[col].astype("Int64").astype("category")

    return out


def _add_target_and_validity_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula a duração da entrega (sem arredondamento) e indicadores de validade."""
    out = df.copy()
    duration_sec = (out["actual_delivery_time"] - out["created_at"]).dt.total_seconds()
    out["delivery_duration_seconds"] = duration_sec
    out["delivery_duration_minutes"] = duration_sec / 60.0

    out["is_target_missing"] = duration_sec.isna()
    out["is_target_non_positive"] = duration_sec.notna() & (duration_sec <= 0)
    out["is_target_valid"] = duration_sec.notna() & (duration_sec > 0)
    out["is_duration_gt_120m"] = out["delivery_duration_minutes"] > 120.0
    out["is_duration_gt_180m"] = out["delivery_duration_minutes"] > 180.0
    out["is_duration_gt_24h"] = out["delivery_duration_minutes"] > 1440.0
    return out


def _add_initial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria atributos temporais e indicadores operacionais iniciais."""
    out = df.copy()

    out["created_hour"] = out["created_at"].dt.hour
    out["created_dayofweek"] = out["created_at"].dt.dayofweek
    out["created_month"] = out["created_at"].dt.month
    out["created_is_weekend"] = (out["created_dayofweek"] >= 5).astype("int8")

    both_valid = out["total_busy_dashers"].notna() & out["total_onshift_dashers"].notna()
    out["busy_greater_than_onshift"] = pd.Series(
        np.where(
            both_valid,
            out["total_busy_dashers"] > out["total_onshift_dashers"],
            pd.NA,
        ),
        index=out.index,
        dtype="boolean",
    )
    out["dasher_count_difference"] = (
        out["total_onshift_dashers"] - out["total_busy_dashers"]
    )
    return out


def prepare_datasets(
    filepath: str | Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, PreparationSummary]:
    """
    Executa o pipeline controlado de preparação dos dados.

    Retorna:
        (df_raw, df_analysis, df_model, summary)
    """
    df_raw = load_raw_data(filepath)

    # 1. Cópia para análise: tipos corrigidos e variáveis derivadas, sem exclusões
    df_analysis = _convert_types(df_raw)
    df_analysis = _add_target_and_validity_flags(df_analysis)
    df_analysis = _add_initial_features(df_analysis)

    # Contabilização dos alvos antes do filtro
    target_counts = {
        "total_rows": int(len(df_analysis)),
        "missing": int(df_analysis["is_target_missing"].sum()),
        "non_positive_le_zero": int(df_analysis["is_target_non_positive"].sum()),
        "valid_gt_zero": int(df_analysis["is_target_valid"].sum()),
        "gt_120_min": int(df_analysis["is_duration_gt_120m"].sum()),
        "gt_180_min": int(df_analysis["is_duration_gt_180m"].sum()),
        "gt_24_hours": int(df_analysis["is_duration_gt_24h"].sum()),
    }

    # 2. Cópia para modelagem: exclui apenas alvos ausentes, zero ou negativos
    df_model = _convert_types(df_raw)
    df_model = _add_target_and_validity_flags(df_model)

    valid_mask = df_model["is_target_valid"]
    removed_reasons = {
        "target_missing": int(df_model["is_target_missing"].sum()),
        "target_non_positive": int(df_model["is_target_non_positive"].sum()),
    }
    df_model = df_model.loc[valid_mask].copy()

    # Converte valores negativos em NaN somente nas 4 colunas especificadas (sem imputação)
    negatives_converted: Dict[str, int] = {}
    for col in NEGATIVE_TO_NAN_COLS:
        neg_mask = df_model[col] < 0
        neg_count = int(neg_mask.sum())
        negatives_converted[col] = neg_count
        if neg_count > 0:
            df_model.loc[neg_mask, col] = np.nan

    # Atributos iniciais calculados após conversão dos negativos em NaN
    df_model = _add_initial_features(df_model)

    summary = PreparationSummary(
        raw_shape=df_raw.shape,
        analysis_shape=df_analysis.shape,
        model_shape=df_model.shape,
        target_counts=target_counts,
        removed_rows=int(len(df_raw) - len(df_model)),
        removed_reasons=removed_reasons,
        negatives_converted_to_nan=negatives_converted,
    )

    return df_raw, df_analysis, df_model, summary
