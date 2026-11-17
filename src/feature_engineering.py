"""
Módulo de Engenharia de Atributos (Feature Engineering) para DoorDash Delivery ETA.

Implementa a criação de atributos temporais, de pedido, operacionais,
estimativas experimentais e contagem histórica estritamente causal por loja,
com controle rigoroso contra vazamento de dados (data leakage) e separação
explícita por nível de disponibilidade no momento da criação do pedido.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


CONFIRMED_FEATURES: List[str] = [
    # Identificadores / Metadados do pedido
    "market_id",
    "store_id",
    "store_primary_category",
    "order_protocol",
    # Composição e valores do carrinho
    "total_items",
    "subtotal",
    "num_distinct_items",
    "min_item_price",
    "max_item_price",
    # Estimativa de colocação (99,9% constante ligada ao protocolo)
    "estimated_order_place_duration",
    # Temporais (baseadas no timestamp registrado)
    "created_hour",
    "created_dayofweek",
    "created_month",
    "created_is_weekend",
    "created_time_block",
    # Novas features do pedido
    "average_item_price",
    "item_price_range",
    "invalid_item_price_range",
    "distinct_item_ratio",
    "is_single_item",
    "is_large_order",
    # Contagem histórica estritamente anterior (sem uso do alvo)
    "historical_store_order_count",
]

UNCERTAIN_EXPERIMENTAL_FEATURES: List[str] = [
    # Operacionais (disponibilidade no checkout não confirmada; 95,47% ausente no market 6)
    "total_onshift_dashers",
    "total_busy_dashers",
    "total_outstanding_orders",
    "dasher_count_difference",
    "busy_greater_than_onshift",
    "outstanding_orders_per_onshift_dasher",
    "busy_dashers_ratio",
    "is_operational_missing",
    # Estimativas com disponibilidade incerta no checkout
    "estimated_store_to_consumer_driving_duration",
    "estimated_total_duration",
]

FORBIDDEN_LEAKAGE_COLS: List[str] = [
    "actual_delivery_time",
    "delivery_duration_seconds",
    "delivery_duration_minutes",
    "is_target_missing",
    "is_target_non_positive",
    "is_target_valid",
    "is_duration_gt_120m",
    "is_duration_gt_180m",
    "is_duration_gt_24h",
]

FEATURE_AVAILABILITY: Dict[str, Dict[str, str]] = {
    "created_hour": {
        "status": "Disponibilidade confirmada",
        "risk": "Baixo (fuso horário aparente UTC sem offset local por mercado)",
    },
    "created_dayofweek": {
        "status": "Disponibilidade confirmada",
        "risk": "Baixo (fuso horário aparente UTC)",
    },
    "created_is_weekend": {
        "status": "Disponibilidade confirmada",
        "risk": "Baixo (fuso horário aparente UTC)",
    },
    "created_time_block": {
        "status": "Disponibilidade confirmada",
        "risk": "Baixo (faixas do timestamp registrado, sem assumir horário local)",
    },
    "average_item_price": {
        "status": "Disponibilidade confirmada",
        "risk": "Baixo (calculada a partir do carrinho fechado; NaN se total_items <= 0)",
    },
    "item_price_range": {
        "status": "Disponibilidade confirmada",
        "risk": "Baixo (NaN preservado quando min_item_price é NaN; valores negativos mantidos)",
    },
    "invalid_item_price_range": {
        "status": "Disponibilidade confirmada",
        "risk": "Baixo (flag quando max_item_price < min_item_price)",
    },
    "distinct_item_ratio": {
        "status": "Disponibilidade confirmada",
        "risk": "Baixo (razão entre itens distintos e total de itens)",
    },
    "is_single_item": {
        "status": "Disponibilidade confirmada",
        "risk": "Baixo",
    },
    "is_large_order": {
        "status": "Disponibilidade confirmada",
        "risk": "Baixo (limiar total_items >= 5 definido a partir da EDA)",
    },
    "historical_store_order_count": {
        "status": "Disponibilidade confirmada",
        "risk": "Baixo (contagem acumulada estritamente anterior por store_id via created_at, sem usar o alvo)",
    },
    "dasher_count_difference": {
        "status": "Disponibilidade incerta",
        "risk": "Médio (depende de snapshot operacional no momento exato do pedido)",
    },
    "busy_greater_than_onshift": {
        "status": "Disponibilidade incerta",
        "risk": "Médio (depende de snapshot operacional; <NA> preservado)",
    },
    "outstanding_orders_per_onshift_dasher": {
        "status": "Disponibilidade incerta",
        "risk": "Médio (razão calculada apenas se total_onshift_dashers > 0; NaN caso contrário)",
    },
    "busy_dashers_ratio": {
        "status": "Disponibilidade incerta",
        "risk": "Médio (razão sem teto de 100%, calculada apenas se total_onshift_dashers > 0)",
    },
    "is_operational_missing": {
        "status": "Disponibilidade incerta",
        "risk": "Médio (sinaliza ausência operacional concentrada no market_id = 6)",
    },
    "estimated_total_duration": {
        "status": "Disponibilidade incerta",
        "risk": "Médio (soma de estimativas; driving_duration tem disponibilidade incerta no checkout)",
    },
    "actual_delivery_time": {
        "status": "Indisponível no momento do pedido",
        "risk": "Crítico (vazamento direto do alvo)",
    },
    "delivery_duration_seconds": {
        "status": "Indisponível no momento do pedido",
        "risk": "Crítico (variável-alvo)",
    },
    "delivery_duration_minutes": {
        "status": "Indisponível no momento do pedido",
        "risk": "Crítico (variável-alvo)",
    },
}


def _assign_time_block(hour: pd.Series) -> pd.Series:
    """Agrupa o horário registrado em faixas de 6 horas sem assumir fuso local."""
    bins = [-1, 5, 11, 17, 23]
    labels = ["00-05h_reg", "06-11h_reg", "12-17h_reg", "18-23h_reg"]
    return pd.cut(hour, bins=bins, labels=labels)


def build_features(df_model: pd.DataFrame) -> pd.DataFrame:
    """
    Constrói o conjunto de features a partir de `df_model` (saída de `prepare_datasets`).

    Preserva a ordem original do índice ao final e não realiza imputações nem
    utiliza a variável-alvo no cálculo de nenhuma feature.
    """
    df = df_model.copy()

    # 1. Features temporais (mantidas no horário registrado)
    if "created_hour" not in df.columns:
        df["created_hour"] = df["created_at"].dt.hour
    if "created_dayofweek" not in df.columns:
        df["created_dayofweek"] = df["created_at"].dt.dayofweek
    if "created_is_weekend" not in df.columns:
        df["created_is_weekend"] = (df["created_dayofweek"] >= 5).astype("int8")
    df["created_time_block"] = _assign_time_block(df["created_hour"])

    # 2. Features do pedido
    valid_items = df["total_items"].notna() & (df["total_items"] > 0)
    df["average_item_price"] = np.where(
        valid_items & df["subtotal"].notna(),
        df["subtotal"] / df["total_items"],
        np.nan,
    )
    df["item_price_range"] = df["max_item_price"] - df["min_item_price"]
    df["invalid_item_price_range"] = (
        df["item_price_range"].notna() & (df["item_price_range"] < 0)
    ).astype("int8")
    df["distinct_item_ratio"] = np.where(
        valid_items & df["num_distinct_items"].notna(),
        df["num_distinct_items"] / df["total_items"],
        np.nan,
    )
    df["is_single_item"] = (df["total_items"] == 1).astype("int8")
    df["is_large_order"] = (df["total_items"] >= 5).astype("int8")

    # 3. Features operacionais (sem substituir ausências por zero e sem teto de 100%)
    df["is_operational_missing"] = (
        df[["total_onshift_dashers", "total_busy_dashers", "total_outstanding_orders"]]
        .isna()
        .any(axis=1)
        .astype("int8")
    )

    valid_onshift = df["total_onshift_dashers"].notna() & (df["total_onshift_dashers"] > 0)
    df["outstanding_orders_per_onshift_dasher"] = np.where(
        valid_onshift & df["total_outstanding_orders"].notna(),
        df["total_outstanding_orders"] / df["total_onshift_dashers"],
        np.nan,
    )
    df["busy_dashers_ratio"] = np.where(
        valid_onshift & df["total_busy_dashers"].notna(),
        df["total_busy_dashers"] / df["total_onshift_dashers"],
        np.nan,
    )

    # 4. Estimativas de duração (feature experimental)
    df["estimated_total_duration"] = (
        df["estimated_order_place_duration"] + df["estimated_store_to_consumer_driving_duration"]
    )

    # 5. Contagem histórica estritamente anterior por loja (sem uso do alvo)
    # Agrupamos por (store_id, created_at) para que pedidos simultâneos recebam
    # exatamente a mesma contagem anterior e não contem uns aos outros.
    ts_counts = (
        df.groupby(["store_id", "created_at"], sort=True)
        .size()
        .rename("_ts_count")
        .reset_index()
        .sort_values(["store_id", "created_at"], kind="mergesort")
    )
    ts_counts["historical_store_order_count"] = (
        ts_counts.groupby("store_id", sort=False)["_ts_count"].cumsum() - ts_counts["_ts_count"]
    )
    df = df.merge(
        ts_counts[["store_id", "created_at", "historical_store_order_count"]],
        on=["store_id", "created_at"],
        how="left",
    )

    return df
